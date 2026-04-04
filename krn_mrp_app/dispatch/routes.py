from datetime import date, timedelta, datetime
from typing import Any, Dict, List, Optional
import json, io, csv

from fastapi import APIRouter, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from starlette.templating import Jinja2Templates
from sqlalchemy import text

from krn_mrp_app.deps import engine, require_roles

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# ---------------- DDL / safety ----------------
with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS dispatch_customers (
            id SERIAL PRIMARY KEY,
            customer_name TEXT NOT NULL UNIQUE,
            customer_gstin TEXT,
            contact TEXT,
            customer_address TEXT,
            is_active BOOLEAN NOT NULL DEFAULT TRUE
        )
    """))

    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS dispatch_sales_orders (
            id SERIAL PRIMARY KEY,
            order_no TEXT NOT NULL UNIQUE,
            order_date DATE NOT NULL,
            customer_name TEXT NOT NULL,
            customer_gstin TEXT,
            contact TEXT,
            customer_address TEXT,
            remarks TEXT,
            status TEXT NOT NULL DEFAULT 'OPEN',
            created_by TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """))

    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS dispatch_sales_order_items (
            id SERIAL PRIMARY KEY,
            sales_order_id INT NOT NULL,
            fg_grade TEXT NOT NULL,
            ordered_qty_kg DOUBLE PRECISION NOT NULL DEFAULT 0,
            dispatched_qty_kg DOUBLE PRECISION NOT NULL DEFAULT 0,
            remarks TEXT
        )
    """))

    # Link dispatch execution back to customer orders
    try:
        conn.execute(text("ALTER TABLE dispatch_orders ADD COLUMN IF NOT EXISTS sales_order_id INT"))
    except Exception:
        pass
    try:
        conn.execute(text("ALTER TABLE dispatch_orders ADD COLUMN IF NOT EXISTS sales_order_no TEXT"))
    except Exception:
        pass


def _dispatch_customers(active_only: bool = True):
    with engine.begin() as conn:
        sql = "SELECT * FROM dispatch_customers" + (" WHERE COALESCE(is_active,TRUE)=TRUE" if active_only else "") + " ORDER BY customer_name"
        return [dict(r) for r in conn.execute(text(sql)).mappings().all()]


def _dispatch_grade_options() -> List[str]:
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT COALESCE(fg_grade,'') AS fg_grade
            FROM fg_lots
            WHERE COALESCE(fg_grade,'') <> ''
            ORDER BY fg_grade
        """)).mappings().all()
    return [str(r['fg_grade']) for r in rows if str(r.get('fg_grade') or '').strip()]


# ---------- Helpers: mirror style of other modules ----------
def _tpl_auth(request: Request) -> dict:
    sess = (getattr(request, "session", {}) or {})
    return {"user": sess.get("username", "") or "", "role": sess.get("role", "guest") or "guest"}


def _is_admin(request: Request) -> bool:
    s = getattr(request, "state", None)
    if not s: return False
    if getattr(s, "is_admin", False): return True
    role = getattr(s, "role", None)
    if isinstance(role, str) and role.lower() == "admin": return True
    roles = getattr(s, "roles", None)
    return isinstance(roles, (list,set,tuple)) and "admin" in roles


def _fg_available_rows():
    with engine.begin() as conn:
        lots = conn.execute(text("""
            SELECT id, lot_no, date, family, fg_grade,
                   COALESCE(weight_kg,0)::float AS weight_kg,
                   COALESCE(cost_per_kg,0)::float AS cost_per_kg,
                   qa_status, remarks
            FROM fg_lots
            WHERE UPPER(COALESCE(qa_status,''))='APPROVED'
            ORDER BY date DESC, id DESC
        """)).mappings().all()
        used_by_fg = dict(conn.execute(text("""
            SELECT fg_lot_id, COALESCE(SUM(qty_kg),0) AS used
            FROM dispatch_items
            GROUP BY fg_lot_id
        """)).all() or [])
    out = []
    for r in lots:
        used = float(used_by_fg.get(r["id"], 0.0))
        avail = float(r["weight_kg"] or 0.0) - used
        if avail > 0.0001:
            out.append({
                "fg_lot_id": r["id"],
                "fg_lot_no": r["lot_no"],
                "date": r["date"],
                "family": r["family"],
                "fg_grade": r["fg_grade"],
                "available_kg": avail,
                "cost_per_kg": float(r["cost_per_kg"] or 0.0),
                "remarks": r.get("remarks",""),
            })
    return out


def _fetch_order(order_id: int):
    with engine.begin() as conn:
        head = conn.execute(text("SELECT * FROM dispatch_orders WHERE id=:id"), {"id": order_id}).mappings().first()
        items = conn.execute(text("SELECT * FROM dispatch_items WHERE dispatch_id=:id ORDER BY id"), {"id": order_id}).mappings().all()
    return (dict(head) if head else None, [dict(i) for i in items])


def _dispatch_rows_in_range(start: date, end: date):
    with engine.begin() as conn:
        orders = conn.execute(text("""
            SELECT id, order_no, date, customer_name, transporter, vehicle_no, lr_no, remarks,
                   COALESCE(sales_order_no,'') AS sales_order_no
            FROM dispatch_orders
            WHERE date BETWEEN :s AND :e
            ORDER BY date DESC, id DESC
        """), {"s": start, "e": end}).mappings().all()

        out = []
        for o in orders:
            totals = conn.execute(text("""
                SELECT COALESCE(SUM(qty_kg),0) AS qty,
                       COALESCE(SUM(value),0) AS val
                FROM dispatch_items WHERE dispatch_id=:d
            """), {"d":o["id"]}).mappings().first() or {"qty":0,"val":0}
            d = dict(o)
            d["total_qty_kg"] = float(totals["qty"] or 0.0)
            d["total_value"]  = float(totals["val"] or 0.0)
            out.append(d)
    return out


def _fg_latest_coa_for_lot(fg_lot_id: int):
    with engine.begin() as conn:
        fgqa = conn.execute(text("""
            SELECT id, decision, remarks
            FROM fg_qa WHERE fg_lot_id=:id
            ORDER BY id DESC LIMIT 1
        """), {"id": fg_lot_id}).mappings().first()
        if fgqa:
            params = conn.execute(text("""
                SELECT param_name, param_value
                FROM fg_qa_params WHERE fg_qa_id=:qid ORDER BY id
            """), {"qid": fgqa["id"]}).mappings().all()
            return {
                "header": {"decision": fgqa["decision"], "remarks": fgqa.get("remarks","")},
                "params": [{"name":p["param_name"], "value":p["param_value"], "unit":"","spec_min":"","spec_max":""} for p in params]
            }
        src = conn.execute(text("SELECT src_alloc_json FROM fg_lots WHERE id=:id"), {"id": fg_lot_id}).scalar()
        amap = {}
        try:
            amap = json.loads(src or "{}")
        except Exception:
            pass
        if not amap: return None
        top = max(amap.items(), key=lambda kv: float(kv[1] or 0.0))
        grd_no = top[0]
        grd = conn.execute(text("SELECT id FROM grinding_lots WHERE lot_no=:ln"), {"ln": grd_no}).mappings().first()
        if not grd: return None
        grqa = conn.execute(text("""
            SELECT id, decision, remarks, oxygen, compressibility
            FROM grinding_qa WHERE grinding_lot_id=:gid
            ORDER BY id DESC LIMIT 1
        """), {"gid": grd["id"]}).mappings().first()
        if not grqa: return None
        params = conn.execute(text("""
            SELECT param_name, param_value
            FROM grinding_qa_params WHERE grinding_qa_id=:qid ORDER BY id
        """), {"qid": grqa["id"]}).mappings().all()
        payload = {
            "header": {"decision": grqa["decision"], "remarks": grqa.get("remarks","")},
            "params": [{"name":p["param_name"], "value":p["param_value"], "unit":"","spec_min":"","spec_max":""} for p in params]
        }
        if grqa.get("oxygen") is not None:
            payload["params"].insert(0, {"name":"Oxygen","value":f"{float(grqa['oxygen']):.3f}","unit":"","spec_min":"","spec_max":""})
        if grqa.get("compressibility") is not None:
            payload["params"].insert(1, {"name":"Compressibility","value":f"{float(grqa['compressibility']):.2f}","unit":"","spec_min":"","spec_max":""})
        return payload


def _sales_orders(status_filter: str = "open") -> List[Dict[str, Any]]:
    with engine.begin() as conn:
        sql = "SELECT * FROM dispatch_sales_orders"
        params: Dict[str, Any] = {}
        if status_filter == "open":
            sql += " WHERE UPPER(COALESCE(status,'OPEN')) IN ('OPEN','PARTIAL')"
        sql += " ORDER BY order_date DESC, id DESC"
        heads = [dict(r) for r in conn.execute(text(sql), params).mappings().all()]
        for h in heads:
            items = [dict(r) for r in conn.execute(text("""
                SELECT * FROM dispatch_sales_order_items
                WHERE sales_order_id=:id
                ORDER BY id
            """), {"id": h['id']}).mappings().all()]
            total_ordered = sum(float(i.get('ordered_qty_kg') or 0.0) for i in items)
            total_dispatched = sum(float(i.get('dispatched_qty_kg') or 0.0) for i in items)
            h['items'] = items
            h['total_ordered_kg'] = total_ordered
            h['total_dispatched_kg'] = total_dispatched
            h['total_pending_kg'] = max(total_ordered - total_dispatched, 0.0)
    return heads


def _fetch_sales_order(order_id: int) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    with engine.begin() as conn:
        head = conn.execute(text("SELECT * FROM dispatch_sales_orders WHERE id=:id"), {"id": order_id}).mappings().first()
        items = conn.execute(text("SELECT * FROM dispatch_sales_order_items WHERE sales_order_id=:id ORDER BY id"), {"id": order_id}).mappings().all()
    if not head:
        return None, []
    head_d = dict(head)
    item_list = [dict(r) for r in items]
    for it in item_list:
        it['pending_qty_kg'] = max(float(it.get('ordered_qty_kg') or 0.0) - float(it.get('dispatched_qty_kg') or 0.0), 0.0)
    head_d['total_ordered_kg'] = sum(float(i.get('ordered_qty_kg') or 0.0) for i in item_list)
    head_d['total_dispatched_kg'] = sum(float(i.get('dispatched_qty_kg') or 0.0) for i in item_list)
    head_d['total_pending_kg'] = max(head_d['total_ordered_kg'] - head_d['total_dispatched_kg'], 0.0)
    return head_d, item_list


def _update_sales_order_status(conn, sales_order_id: int) -> None:
    row = conn.execute(text("""
        SELECT
            COALESCE(SUM(ordered_qty_kg),0)::float AS ordered,
            COALESCE(SUM(dispatched_qty_kg),0)::float AS dispatched
        FROM dispatch_sales_order_items
        WHERE sales_order_id=:id
    """), {"id": sales_order_id}).mappings().first() or {"ordered":0.0, "dispatched":0.0}
    ordered = float(row.get('ordered') or 0.0)
    dispatched = float(row.get('dispatched') or 0.0)
    status = 'OPEN'
    if ordered > 0 and dispatched >= ordered - 0.0001:
        status = 'CLOSED'
    elif dispatched > 0:
        status = 'PARTIAL'
    conn.execute(text("""
        UPDATE dispatch_sales_orders
        SET status=:st, updated_at=CURRENT_TIMESTAMP
        WHERE id=:id
    """), {"st": status, "id": sales_order_id})


def _apply_dispatch_to_sales_order(conn, sales_order_id: int, dispatch_items: List[Dict[str, Any]]) -> None:
    pending_rows = [dict(r) for r in conn.execute(text("""
        SELECT id, fg_grade,
               COALESCE(ordered_qty_kg,0)::float AS ordered_qty_kg,
               COALESCE(dispatched_qty_kg,0)::float AS dispatched_qty_kg
        FROM dispatch_sales_order_items
        WHERE sales_order_id=:id
        ORDER BY id
    """), {"id": sales_order_id}).mappings().all()]

    by_grade: Dict[str, List[Dict[str, Any]]] = {}
    for r in pending_rows:
        grade = str(r.get('fg_grade') or '').strip()
        by_grade.setdefault(grade, []).append(r)

    for d in dispatch_items:
        grade = str(d.get('fg_grade') or '').strip()
        remaining = float(d.get('qty_kg') or 0.0)
        for row in by_grade.get(grade, []):
            if remaining <= 0.0001:
                break
            pending = max(float(row.get('ordered_qty_kg') or 0.0) - float(row.get('dispatched_qty_kg') or 0.0), 0.0)
            if pending <= 0.0001:
                continue
            take = min(pending, remaining)
            conn.execute(text("""
                UPDATE dispatch_sales_order_items
                SET dispatched_qty_kg = COALESCE(dispatched_qty_kg,0) + :q
                WHERE id=:id
            """), {"q": take, "id": row['id']})
            remaining -= take

    _update_sales_order_status(conn, sales_order_id)


def _sales_order_pending_by_grade(items: List[Dict[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for it in items:
        grade = str(it.get('fg_grade') or '').strip()
        pending = max(float(it.get('ordered_qty_kg') or 0.0) - float(it.get('dispatched_qty_kg') or 0.0), 0.0)
        out[grade] = out.get(grade, 0.0) + pending
    return out


@router.get('/customers', response_class=HTMLResponse)
async def dispatch_customers_get(request: Request, dep: None = Depends(require_roles('admin'))):
    return templates.TemplateResponse('dispatch_customers.html', {'request': request, 'rows': _dispatch_customers(False), 'is_admin': _is_admin(request), **_tpl_auth(request)})


@router.post('/customers')
async def dispatch_customers_post(request: Request, dep: None = Depends(require_roles('admin'))):
    form = await request.form()
    name = (form.get('customer_name') or '').strip()
    if not name:
        return RedirectResponse('/dispatch/customers', status_code=303)
    payload = {
        'customer_name': name,
        'customer_gstin': (form.get('customer_gstin') or '').strip(),
        'contact': (form.get('contact') or '').strip(),
        'customer_address': (form.get('customer_address') or '').strip(),
    }
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO dispatch_customers(customer_name, customer_gstin, contact, customer_address, is_active)
            VALUES (:customer_name,:customer_gstin,:contact,:customer_address,TRUE)
            ON CONFLICT (customer_name) DO UPDATE SET
                customer_gstin=EXCLUDED.customer_gstin,
                contact=EXCLUDED.contact,
                customer_address=EXCLUDED.customer_address,
                is_active=TRUE
        """), payload)
    return RedirectResponse('/dispatch/customers', status_code=303)


# ---------------- CUSTOMER ORDERS ----------------
@router.get('/customer-orders', response_class=HTMLResponse)
async def dispatch_customer_orders_get(request: Request, dep: None = Depends(require_roles('admin','dispatch','store','view'))):
    customers = _dispatch_customers(True)
    grades = _dispatch_grade_options()
    rows = _sales_orders('all')
    return templates.TemplateResponse('dispatch_customer_orders.html', {
        'request': request,
        'rows': rows,
        'customers': customers,
        'grades': grades,
        'today': date.today().isoformat(),
        'is_admin': _is_admin(request),
        'err': request.query_params.get('err',''),
        **_tpl_auth(request),
    })


@router.post('/customer-orders')
async def dispatch_customer_orders_post(request: Request, dep: None = Depends(require_roles('admin','dispatch','store'))):
    form = await request.form()
    customer_name = (form.get('customer_name') or form.get('customer_master') or '').strip()
    if not customer_name:
        return RedirectResponse('/dispatch/customer-orders?err=Customer+Name+is+required', status_code=303)
    try:
        order_date = date.fromisoformat((form.get('order_date') or date.today().isoformat()).strip())
    except Exception:
        return RedirectResponse('/dispatch/customer-orders?err=Order+Date+is+invalid', status_code=303)

    items = []
    for idx in range(1, 7):
        grade = (form.get(f'fg_grade_{idx}') or '').strip()
        qty_raw = (form.get(f'qty_{idx}') or '').strip()
        line_remarks = (form.get(f'line_remarks_{idx}') or '').strip()
        if not grade and not qty_raw:
            continue
        try:
            qty = float(qty_raw or 0.0)
        except Exception:
            qty = 0.0
        if not grade or qty <= 0:
            continue
        items.append({'fg_grade': grade, 'ordered_qty_kg': qty, 'remarks': line_remarks})

    if not items:
        return RedirectResponse('/dispatch/customer-orders?err=Add+at+least+one+grade+line+with+Qty+%3E+0', status_code=303)

    with engine.begin() as conn:
        prefix = 'ORD-' + order_date.strftime('%Y%m%d') + '-'
        last = conn.execute(text("""
            SELECT order_no FROM dispatch_sales_orders
            WHERE order_no LIKE :pfx
            ORDER BY order_no DESC
            LIMIT 1
        """), {'pfx': f'{prefix}%'}).scalar()
        seq = int(str(last).split('-')[-1]) + 1 if last else 1
        order_no = f'{prefix}{seq:03d}'
        order_id = conn.execute(text("""
            INSERT INTO dispatch_sales_orders
                (order_no, order_date, customer_name, customer_gstin, contact, customer_address, remarks, status, created_by)
            VALUES
                (:ono, :d, :cn, :gst, :ct, :addr, :rmk, 'OPEN', :cb)
            RETURNING id
        """), {
            'ono': order_no,
            'd': order_date,
            'cn': customer_name,
            'gst': (form.get('customer_gstin') or '').strip(),
            'ct': (form.get('contact') or '').strip(),
            'addr': (form.get('customer_address') or '').strip(),
            'rmk': (form.get('remarks') or '').strip(),
            'cb': getattr(getattr(request,'state',None),'username','')
        }).mappings().first()['id']

        conn.execute(text("""
            INSERT INTO dispatch_sales_order_items(sales_order_id, fg_grade, ordered_qty_kg, dispatched_qty_kg, remarks)
            VALUES (:sid, :g, :q, 0, :r)
        """), [{'sid': order_id, 'g': it['fg_grade'], 'q': it['ordered_qty_kg'], 'r': it.get('remarks','')} for it in items])
        _update_sales_order_status(conn, order_id)

    return RedirectResponse('/dispatch/customer-orders', status_code=303)


@router.get('/customer-orders/view/{order_id}', response_class=HTMLResponse)
async def dispatch_customer_order_view(request: Request, order_id: int, dep: None = Depends(require_roles('admin','dispatch','store','view'))):
    head, items = _fetch_sales_order(order_id)
    if not head:
        raise HTTPException(status_code=404, detail='Customer order not found')
    return templates.TemplateResponse('dispatch_customer_order_view.html', {
        'request': request,
        'head': head,
        'items': items,
        'is_admin': _is_admin(request),
        **_tpl_auth(request),
    })


# ---------------- HOME ----------------
@router.get('/', response_class=HTMLResponse)
async def dispatch_home(request: Request, dep: None = Depends(require_roles('admin','dispatch','store','view'))):
    rows = _fg_available_rows()
    total_avail = sum(float(r['available_kg'] or 0.0) for r in rows)
    open_orders = _sales_orders('open')
    return templates.TemplateResponse('dispatch_home.html', {
        'request': request,
        'rows': rows,
        'total_avail': total_avail,
        'open_orders': open_orders,
        'is_admin': _is_admin(request),
        'today': date.today().isoformat(),
        'customers': _dispatch_customers(True),
        **_tpl_auth(request),
    })


# ---------------- CREATE ----------------
@router.get('/create', response_class=HTMLResponse)
async def dispatch_create_get(request: Request, dep: None = Depends(require_roles('admin','dispatch','store'))):
    rows = _fg_available_rows()
    err = request.query_params.get('err','')
    sales_order_id_raw = request.query_params.get('sales_order_id','').strip()
    selected_sales_order = None
    selected_sales_order_items: List[Dict[str, Any]] = []
    if sales_order_id_raw:
        try:
            selected_sales_order, selected_sales_order_items = _fetch_sales_order(int(sales_order_id_raw))
        except Exception:
            selected_sales_order = None
            selected_sales_order_items = []
    return templates.TemplateResponse('dispatch_create.html', {
        'request': request,
        'rows': rows,
        'customers': _dispatch_customers(True),
        'open_sales_orders': _sales_orders('open'),
        'selected_sales_order': selected_sales_order,
        'selected_sales_order_items': selected_sales_order_items,
        'err': err,
        'is_admin': _is_admin(request),
        'today': date.today().isoformat(),
        **_tpl_auth(request),
    })


@router.post('/create')
async def dispatch_create_post(request: Request, dep: None = Depends(require_roles('admin','dispatch','store'))):
    form = await request.form()
    customer_name = (form.get('customer_name') or form.get('customer_master') or '').strip()
    if not customer_name:
        return RedirectResponse('/dispatch/create?err=Customer+Name+is+required', status_code=303)
    order_date = form.get('date') or date.today().isoformat()
    sales_order_id_raw = (form.get('sales_order_id') or '').strip()
    sales_order_id = int(sales_order_id_raw) if sales_order_id_raw.isdigit() else None

    payload_rows = _fg_available_rows()
    by_id = {r['fg_lot_id']: r for r in payload_rows}

    items: list[dict] = []
    for k, v in form.items():
        if not k.startswith('q_'): continue
        try:
            fg_id = int(k.split('_',1)[1])
        except Exception:
            continue
        try:
            qty = float(v or 0.0)
        except Exception:
            qty = 0.0
        if qty <= 0: continue
        src = by_id.get(fg_id)
        if not src:
            return RedirectResponse('/dispatch/create?err=Invalid+FG+lot+selected', status_code=303)
        if qty > float(src['available_kg'] or 0.0) + 1e-6:
            return RedirectResponse(f"/dispatch/create?err=Qty+exceeds+available+for+{src['fg_lot_no']}", status_code=303)
        items.append({
            'fg_lot_id': fg_id,
            'fg_lot_no': src['fg_lot_no'],
            'fg_grade': src['fg_grade'],
            'qty_kg': qty,
            'cost_per_kg': float(src['cost_per_kg'] or 0.0),
            'value': float(src['cost_per_kg'] or 0.0) * qty,
        })

    if not items:
        q = f'?err=Add+at+least+one+line+with+Qty+%3E+0'
        if sales_order_id:
            q += f'&sales_order_id={sales_order_id}'
        return RedirectResponse('/dispatch/create' + q, status_code=303)

    selected_sales_order = None
    selected_sales_order_items: List[Dict[str, Any]] = []
    if sales_order_id:
        selected_sales_order, selected_sales_order_items = _fetch_sales_order(sales_order_id)
        if not selected_sales_order:
            return RedirectResponse('/dispatch/create?err=Selected+customer+order+not+found', status_code=303)
        pending_by_grade = _sales_order_pending_by_grade(selected_sales_order_items)
        dispatch_by_grade: Dict[str, float] = {}
        for it in items:
            dispatch_by_grade[it['fg_grade']] = dispatch_by_grade.get(it['fg_grade'], 0.0) + float(it['qty_kg'] or 0.0)
        for grade, qty in dispatch_by_grade.items():
            allowed = float(pending_by_grade.get(grade, 0.0))
            if qty > allowed + 1e-6:
                return RedirectResponse(f"/dispatch/create?err=Dispatch+qty+for+{grade}+exceeds+pending+customer+order+qty.&sales_order_id={sales_order_id}", status_code=303)

    with engine.begin() as conn:
        prefix = 'DSP-' + date.today().strftime('%Y%m%d') + '-'
        last = conn.execute(text("""
            SELECT order_no FROM dispatch_orders
            WHERE order_no LIKE :pfx
            ORDER BY order_no DESC
            LIMIT 1
        """), {'pfx': f'{prefix}%'}).scalar()
        seq = int(str(last).split('-')[-1]) + 1 if last else 1
        order_no = f'{prefix}{seq:03d}'

        head_id = conn.execute(text("""
            INSERT INTO dispatch_orders
                (order_no, date, customer_name, customer_gstin, customer_address,
                 transporter, vehicle_no, lr_no, contact, remarks, created_by, sales_order_id, sales_order_no)
            VALUES
                (:ono, :d, :cn, :gst, :addr, :tr, :veh, :lr, :ct, :rmk, :cb, :soid, :sono)
            RETURNING id
        """), {
            'ono': order_no,
            'd': order_date,
            'cn': customer_name,
            'gst': (form.get('customer_gstin') or '').strip(),
            'addr': (form.get('customer_address') or '').strip(),
            'tr': (form.get('transporter') or '').strip(),
            'veh': (form.get('vehicle_no') or '').strip(),
            'lr': (form.get('lr_no') or '').strip(),
            'ct': (form.get('contact') or '').strip(),
            'rmk': (form.get('remarks') or '').strip(),
            'cb': getattr(getattr(request,'state',None),'username',''),
            'soid': sales_order_id,
            'sono': (selected_sales_order.get('order_no') if selected_sales_order else ''),
        }).mappings().first()['id']

        conn.execute(text("""
            INSERT INTO dispatch_items(dispatch_id, fg_lot_id, fg_lot_no, fg_grade, qty_kg, cost_per_kg, value)
            VALUES (:did, :fid, :fno, :g, :q, :c, :v)
        """), [{'did': head_id, 'fid': it['fg_lot_id'], 'fno': it['fg_lot_no'], 'g': it['fg_grade'], 'q': it['qty_kg'], 'c': it['cost_per_kg'], 'v': it['value']} for it in items])

        if sales_order_id:
            _apply_dispatch_to_sales_order(conn, sales_order_id, items)

    return RedirectResponse(f'/dispatch/view/{head_id}', status_code=303)


# ---------------- LIST + CSV ----------------
@router.get('/orders', response_class=HTMLResponse)
async def dispatch_orders(request: Request, dep: None = Depends(require_roles('admin','dispatch','store','view')), from_date: Optional[str] = None, to_date: Optional[str] = None, csv_export: int = 0):
    today = date.today().isoformat()
    s = from_date or today
    e = to_date or today
    try:
        s_d = date.fromisoformat(s)
        e_d = date.fromisoformat(e)
    except Exception:
        s_d = e_d = date.today()
        s = e = today
    rows = _dispatch_rows_in_range(s_d, e_d)
    if csv_export:
        out = io.StringIO(); w = csv.writer(out)
        w.writerow(['Dispatch No','Date','Customer','Linked Customer Order','Transporter','Vehicle','LR No','Qty (kg)','Value (₹)'])
        for r in rows:
            w.writerow([r['order_no'], r['date'], r['customer_name'], r.get('sales_order_no') or '', r['transporter'] or '', r['vehicle_no'] or '', r['lr_no'] or '', f"{(r['total_qty_kg'] or 0):.1f}", f"{(r['total_value'] or 0):.2f}"])
        return Response(out.getvalue(), media_type='text/csv', headers={'Content-Disposition':'attachment; filename=dispatch_orders.csv'})
    return templates.TemplateResponse('dispatch_list.html', {
        'request': request,
        'rows': rows,
        'from_date': s, 'to_date': e,
        'today': today,
        'is_admin': _is_admin(request),
        **_tpl_auth(request),
    })


# ---------------- VIEW + PDF ----------------
@router.get('/view/{order_id}', response_class=HTMLResponse)
async def dispatch_view(request: Request, order_id: int, dep: None = Depends(require_roles('admin','dispatch','store','view'))):
    head, items = _fetch_order(order_id)
    if not head:
        raise HTTPException(status_code=404, detail='Dispatch order not found')
    total_qty = sum(float(i['qty_kg'] or 0.0) for i in items)
    total_val = sum(float(i['value'] or 0.0) for i in items)
    return templates.TemplateResponse('dispatch_view.html', {
        'request': request,
        'head': head,
        'items': items,
        'total_qty': total_qty,
        'total_val': total_val,
        'is_admin': _is_admin(request),
        **_tpl_auth(request),
    })


@router.get('/pdf/{order_id}', response_class=HTMLResponse)
async def dispatch_pdf(request: Request, order_id: int, dep: None = Depends(require_roles('admin','dispatch','store','view'))):
    head, items = _fetch_order(order_id)
    if not head:
        raise HTTPException(status_code=404, detail='Dispatch order not found')
    total_qty = sum(float(i['qty_kg'] or 0.0) for i in items)
    total_val = sum(float(i['value'] or 0.0) for i in items)
    return templates.TemplateResponse('dispatch_pdf.html', {
        'request': request,
        'head': head, 'items': items,
        'total_qty': total_qty, 'total_val': total_val,
        'is_admin': _is_admin(request),
        **_tpl_auth(request),
    })


@router.get('/coa/{fg_lot_id}', response_class=HTMLResponse)
async def dispatch_coa_pdf(request: Request, fg_lot_id: int, dep: None = Depends(require_roles('admin','dispatch','qa','view'))):
    with engine.begin() as conn:
        fg = conn.execute(text("""
            SELECT id, lot_no, date, family, fg_grade, weight_kg, cost_per_kg, qa_status, remarks
            FROM fg_lots WHERE id=:id
        """), {'id': fg_lot_id}).mappings().first()
    if not fg:
        raise HTTPException(status_code=404, detail='FG lot not found')
    coa = _fg_latest_coa_for_lot(fg_lot_id)
    return templates.TemplateResponse('dispatch_coa_pdf.html', {
        'request': request,
        'fg': fg,
        'coa': coa,
        'is_admin': _is_admin(request),
        **_tpl_auth(request),
    })


@router.get('/trace/{fg_lot_no}', response_class=HTMLResponse)
async def dispatch_trace(request: Request, fg_lot_no: str, dep: None = Depends(require_roles('admin','qa','dispatch','view'))):
    fg_lot_no = fg_lot_no.strip().upper()
    trace = {'FG': None, 'Grinding': None, 'Annealing': None, 'Atomization': None, 'Melting': None, 'GRN': None}
    with engine.begin() as conn:
        fg = conn.execute(text('SELECT * FROM fg_lots WHERE UPPER(lot_no)=:ln'), {'ln': fg_lot_no}).mappings().first()
        if not fg:
            raise HTTPException(status_code=404, detail=f'FG Lot {fg_lot_no} not found')
        trace['FG'] = dict(fg)
        try:
            src_alloc = json.loads(fg['src_alloc_json'] or '{}')
        except Exception:
            src_alloc = {}
        grinding_keys = list(src_alloc.keys())
        grind_rows = []
        for glot in grinding_keys:
            g = conn.execute(text('SELECT * FROM grinding_lots WHERE lot_no=:ln'), {'ln': glot}).mappings().first()
            if g: grind_rows.append(dict(g))
        trace['Grinding'] = grind_rows
        anneal_rows = []
        for g in grind_rows:
            try:
                amap = json.loads(g['src_alloc_json'] or '{}')
            except Exception:
                amap = {}
            for al in amap.keys():
                a = conn.execute(text('SELECT * FROM annealing_lots WHERE lot_no=:ln'), {'ln': al}).mappings().first()
                if a: anneal_rows.append(dict(a))
        trace['Annealing'] = anneal_rows
        atom_rows = []
        for a in anneal_rows:
            try:
                amap = json.loads(a['src_alloc_json'] or '{}')
            except Exception:
                amap = {}
            for at in amap.keys():
                atm = conn.execute(text('SELECT * FROM atomization_lots WHERE lot_no=:ln'), {'ln': at}).mappings().first()
                if atm: atom_rows.append(dict(atm))
        trace['Atomization'] = atom_rows
        melt_rows = []
        for m in atom_rows:
            try:
                amap = json.loads(m['src_alloc_json'] or '{}')
            except Exception:
                amap = {}
            for ml in amap.keys():
                mel = conn.execute(text('SELECT * FROM melting_lots WHERE lot_no=:ln'), {'ln': ml}).mappings().first()
                if mel: melt_rows.append(dict(mel))
        trace['Melting'] = melt_rows
        grn_rows = []
        for mel in melt_rows:
            grn_no = mel.get('grn_no')
            if grn_no:
                grn = conn.execute(text('SELECT * FROM grn_headers WHERE grn_no=:no'), {'no': grn_no}).mappings().first()
                if grn: grn_rows.append(dict(grn))
        trace['GRN'] = grn_rows
    return templates.TemplateResponse('dispatch_trace.html', {
        'request': request,
        'fg_lot_no': fg_lot_no,
        'trace': trace,
        **_tpl_auth(request),
    })
