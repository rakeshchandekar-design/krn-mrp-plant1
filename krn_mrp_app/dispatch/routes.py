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

# ---------- Helpers: mirror style of other modules ----------
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
            SELECT id, order_no, date, customer_name, transporter, vehicle_no, lr_no, remarks
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
    # use main.py helper if imported; else inline fallback
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
        # else fallback to grinding from src_alloc_json
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

# ---------------- HOME ----------------
@router.get("/", response_class=HTMLResponse)
async def dispatch_home(request: Request, dep: None = Depends(require_roles("admin","dispatch","store","view"))):
    rows = _fg_available_rows()
    total_avail = sum(float(r["available_kg"] or 0.0) for r in rows)
    return templates.TemplateResponse("dispatch_home.html", {
        "request": request,
        "rows": rows,
        "total_avail": total_avail,
        "is_admin": _is_admin(request),
        "today": date.today().isoformat(),
    })

# ---------------- CREATE ----------------
@router.get("/create", response_class=HTMLResponse)
async def dispatch_create_get(request: Request, dep: None = Depends(require_roles("admin","dispatch","store"))):
    rows = _fg_available_rows()
    err = request.query_params.get("err","")
    return templates.TemplateResponse("dispatch_create.html", {
        "request": request,
        "rows": rows,
        "err": err,
        "is_admin": _is_admin(request),
        "today": date.today().isoformat(),
    })

@router.post("/create")
async def dispatch_create_post(
    request: Request, dep: None = Depends(require_roles("admin","dispatch","store"))
):
    form = await request.form()
    # Header fields
    customer_name = (form.get("customer_name") or "").strip()
    if not customer_name:
        return RedirectResponse("/dispatch/create?err=Customer+Name+is+required", status_code=303)
    order_date = form.get("date") or date.today().isoformat()
    payload_rows = _fg_available_rows()
    by_id = {r["fg_lot_id"]: r for r in payload_rows}

    # Collect line allocations: items like q_<fg_lot_id>
    items: list[dict] = []
    for k, v in form.items():
        if not k.startswith("q_"): continue
        try:
            fg_id = int(k.split("_",1)[1])
        except Exception:
            continue
        try:
            qty = float(v or 0.0)
        except Exception:
            qty = 0.0
        if qty <= 0: continue
        src = by_id.get(fg_id)
        if not src:
            return RedirectResponse("/dispatch/create?err=Invalid+FG+lot+selected", status_code=303)
        if qty > float(src["available_kg"] or 0.0) + 1e-6:
            return RedirectResponse(f"/dispatch/create?err=Qty+exceeds+available+for+{src['fg_lot_no']}", status_code=303)
        items.append({
            "fg_lot_id": fg_id,
            "fg_lot_no": src["fg_lot_no"],
            "fg_grade":  src["fg_grade"],
            "qty_kg":    qty,
            "cost_per_kg": float(src["cost_per_kg"] or 0.0),
            "value": float(src["cost_per_kg"] or 0.0) * qty
        })

    if not items:
        return RedirectResponse("/dispatch/create?err=Add+at+least+one+line+with+Qty+%3E+0", status_code=303)

    # Persist
    with engine.begin() as conn:
        prefix = "DSP-" + date.today().strftime("%Y%m%d") + "-"
        last = conn.execute(text("""
            SELECT order_no FROM dispatch_orders
            WHERE order_no LIKE :pfx
            ORDER BY order_no DESC
            LIMIT 1
        """), {"pfx": f"{prefix}%"}).scalar()
        seq = int(last.split("-")[-1]) + 1 if last else 1
        order_no = f"{prefix}{seq:03d}"

        head_id = conn.execute(text("""
            INSERT INTO dispatch_orders
                (order_no, date, customer_name, customer_gstin, customer_address,
                 transporter, vehicle_no, lr_no, contact, remarks, created_by)
            VALUES
                (:ono, :d, :cn, :gst, :addr, :tr, :veh, :lr, :ct, :rmk, :cb)
            RETURNING id
        """), {
            "ono": order_no,
            "d":   order_date,
            "cn":  customer_name,
            "gst": (form.get("customer_gstin") or "").strip(),
            "addr": (form.get("customer_address") or "").strip(),
            "tr":   (form.get("transporter") or "").strip(),
            "veh":  (form.get("vehicle_no") or "").strip(),
            "lr":   (form.get("lr_no") or "").strip(),
            "ct":   (form.get("contact") or "").strip(),
            "rmk":  (form.get("remarks") or "").strip(),
            "cb":   getattr(getattr(request,"state",None),"username","")
        }).mappings().first()["id"]

        conn.execute(text("""
            INSERT INTO dispatch_items(dispatch_id, fg_lot_id, fg_lot_no, fg_grade, qty_kg, cost_per_kg, value)
            VALUES (:did, :fid, :fno, :g, :q, :c, :v)
        """), [{"did": head_id, **it} for it in items])

    return RedirectResponse(f"/dispatch/view/{head_id}", status_code=303)

# ---------------- LIST + CSV ----------------
@router.get("/orders", response_class=HTMLResponse)
async def dispatch_orders(
    request: Request,
    dep: None = Depends(require_roles("admin","dispatch","store","view")),
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    csv_export: int = 0
):
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
        w.writerow(["Order No","Date","Customer","Transporter","Vehicle","LR No","Qty (kg)","Value (₹)"])
        for r in rows:
            w.writerow([r["order_no"], r["date"], r["customer_name"], r["transporter"] or "", r["vehicle_no"] or "", r["lr_no"] or "",
                        f"{(r['total_qty_kg'] or 0):.1f}", f"{(r['total_value'] or 0):.2f}"])
        return Response(
            out.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition":"attachment; filename=dispatch_orders.csv"}
        )
    return templates.TemplateResponse("dispatch_list.html", {
        "request": request,
        "rows": rows,
        "from_date": s, "to_date": e,
        "today": today,
        "is_admin": _is_admin(request),
    })

# ---------------- VIEW + PDF ----------------
@router.get("/view/{order_id}", response_class=HTMLResponse)
async def dispatch_view(request: Request, order_id: int, dep: None = Depends(require_roles("admin","dispatch","store","view"))):
    head, items = _fetch_order(order_id)
    if not head:
        raise HTTPException(status_code=404, detail="Dispatch order not found")
    total_qty = sum(float(i["qty_kg"] or 0.0) for i in items)
    total_val = sum(float(i["value"] or 0.0) for i in items)
    return templates.TemplateResponse("dispatch_view.html", {
        "request": request,
        "head": head,
        "items": items,
        "total_qty": total_qty,
        "total_val": total_val,
        "is_admin": _is_admin(request),
    })

@router.get("/pdf/{order_id}", response_class=HTMLResponse)
async def dispatch_pdf(request: Request, order_id: int, dep: None = Depends(require_roles("admin","dispatch","store","view"))):
    head, items = _fetch_order(order_id)
    if not head:
        raise HTTPException(status_code=404, detail="Dispatch order not found")
    total_qty = sum(float(i["qty_kg"] or 0.0) for i in items)
    total_val = sum(float(i["value"] or 0.0) for i in items)
    return templates.TemplateResponse("dispatch_pdf.html", {
        "request": request,
        "head": head, "items": items,
        "total_qty": total_qty, "total_val": total_val,
        "is_admin": _is_admin(request),
    })

# ---------------- CoA PDF for an FG lot ----------------
@router.get("/coa/{fg_lot_id}", response_class=HTMLResponse)
async def dispatch_coa_pdf(request: Request, fg_lot_id: int, dep: None = Depends(require_roles("admin","dispatch","qa","view"))):
    with engine.begin() as conn:
        fg = conn.execute(text("""
            SELECT id, lot_no, date, family, fg_grade, weight_kg, cost_per_kg, qa_status, remarks
            FROM fg_lots WHERE id=:id
        """), {"id": fg_lot_id}).mappings().first()
    if not fg:
        raise HTTPException(status_code=404, detail="FG lot not found")
    coa = _fg_latest_coa_for_lot(fg_lot_id)
    return templates.TemplateResponse("dispatch_coa_pdf.html", {
        "request": request,
        "fg": fg,
        "coa": coa,
        "is_admin": _is_admin(request),
    })
