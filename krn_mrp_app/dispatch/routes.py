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
            po_no TEXT,
            po_date DATE,
            due_date DATE,
            priority TEXT DEFAULT 'NORMAL',
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
            reserved_qty_kg DOUBLE PRECISION NOT NULL DEFAULT 0,
            committed_date DATE,
            source_preference TEXT DEFAULT 'AUTO',
            line_status TEXT DEFAULT 'OPEN',
            remarks TEXT
        )
    """))

    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS dispatch_stock_reservations (
            id SERIAL PRIMARY KEY,
            sales_order_id INT NOT NULL,
            sales_order_item_id INT NOT NULL,
            source_stage TEXT NOT NULL,
            fg_lot_id INT,
            rap_lot_id INT,
            source_lot_no TEXT,
            fg_grade TEXT NOT NULL,
            reserved_qty_kg DOUBLE PRECISION NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'ACTIVE',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            released_at TIMESTAMP
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

    # Allow main Dispatch to also handle semi-finished RAP dispatch rows
    for _ddl in [
        "ALTER TABLE dispatch_items ADD COLUMN IF NOT EXISTS source_stage TEXT DEFAULT 'FG'",
        "ALTER TABLE dispatch_items ADD COLUMN IF NOT EXISTS rap_lot_id INT",
        "ALTER TABLE dispatch_items ADD COLUMN IF NOT EXISTS source_lot_no TEXT",
    ]:
        try:
            conn.execute(text(_ddl))
        except Exception:
            pass

    for _ddl in [
        "ALTER TABLE dispatch_sales_orders ADD COLUMN IF NOT EXISTS po_no TEXT",
        "ALTER TABLE dispatch_sales_orders ADD COLUMN IF NOT EXISTS po_date DATE",
        "ALTER TABLE dispatch_sales_orders ADD COLUMN IF NOT EXISTS due_date DATE",
        "ALTER TABLE dispatch_sales_orders ADD COLUMN IF NOT EXISTS priority TEXT DEFAULT 'NORMAL'",
        "ALTER TABLE dispatch_sales_order_items ADD COLUMN IF NOT EXISTS reserved_qty_kg DOUBLE PRECISION DEFAULT 0",
        "ALTER TABLE dispatch_sales_order_items ADD COLUMN IF NOT EXISTS committed_date DATE",
        "ALTER TABLE dispatch_sales_order_items ADD COLUMN IF NOT EXISTS source_preference TEXT DEFAULT 'AUTO'",
        "ALTER TABLE dispatch_sales_order_items ADD COLUMN IF NOT EXISTS line_status TEXT DEFAULT 'OPEN'",
    ]:
        try:
            conn.execute(text(_ddl))
        except Exception:
            pass



def _table_exists(conn, table_name: str) -> bool:
    """Check if a table exists (works for SQLite & Postgres)."""
    try:
        if str(conn.engine.url).startswith("sqlite"):
            return bool(
                conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name=:t"), {"t": table_name}).fetchone()
            )
        return bool(
            conn.execute(text("SELECT to_regclass('public.'||:t)"), {"t": table_name}).scalar()
        )
    except Exception:
        return False

def _dispatch_customers(active_only: bool = True):
    with engine.begin() as conn:
        sql = "SELECT * FROM dispatch_customers" + (" WHERE COALESCE(is_active,TRUE)=TRUE" if active_only else "") + " ORDER BY customer_name"
        return [dict(r) for r in conn.execute(text(sql)).mappings().all()]


def _dispatch_grade_options() -> List[str]:
    grades: set[str] = set()

    # Primary source of truth from MRP Grade Master (step 1 foundation)
    try:
        with engine.begin() as conn:
            mrp_rows = conn.execute(text("""
                SELECT grade_code
                FROM mrp_grade_master
                WHERE COALESCE(is_active,TRUE)=TRUE AND COALESCE(dispatchable,FALSE)=TRUE
            """)).mappings().all()
        grades.update(str(r.get('grade_code') or '').strip() for r in mrp_rows if str(r.get('grade_code') or '').strip())
    except Exception:
        pass

    # Fallback/static baseline so order booking never goes blank
    grades.update({
        'KRIP', 'KRFS', 'KRM', 'KRSP',
        'KIP', 'KFS', 'KSP',
    })

    # Full configured FG grade master from FG module
    try:
        from krn_mrp_app.fg.routes import FG_SURCHARGE, FG_FAMILY
        grades.update(str(k).strip() for k in list(FG_SURCHARGE.keys()) + list(FG_FAMILY.keys()) if str(k).strip())
    except Exception:
        pass

    with engine.begin() as conn:
        # Also include any live/historic grades already created in FG and in sales orders
        fg_rows = conn.execute(text("""
            SELECT DISTINCT COALESCE(fg_grade,'') AS fg_grade
            FROM fg_lots
            WHERE COALESCE(fg_grade,'') <> ''
        """)).mappings().all()
        so_rows = conn.execute(text("""
            SELECT DISTINCT COALESCE(fg_grade,'') AS fg_grade
            FROM dispatch_sales_order_items
            WHERE COALESCE(fg_grade,'') <> ''
        """)).mappings().all()
        # Include RAP semi-finished grades that can also move through Dispatch
        rap_rows = []
        try:
            rap_rows = conn.execute(text("""
                SELECT DISTINCT COALESCE(l.grade,'') AS fg_grade
                FROM rap_lot rl
                JOIN lot l ON l.id = rl.lot_id
                WHERE COALESCE(l.grade,'') <> ''
            """)).mappings().all()
        except Exception:
            rap_rows = []

    grades.update(str(r.get('fg_grade') or '').strip() for r in fg_rows if str(r.get('fg_grade') or '').strip())
    grades.update(str(r.get('fg_grade') or '').strip() for r in so_rows if str(r.get('fg_grade') or '').strip())
    grades.update(str(r.get('fg_grade') or '').strip() for r in rap_rows if str(r.get('fg_grade') or '').strip())

    def _sort_key(g: str):
        g2 = (g or '').upper()
        if g2.startswith('KIP M') or g2.startswith('KIPM') or g2 == 'KRM':
            bucket = 1
        elif g2 in ('KRIP','KRFS','KRM'):
            bucket = 2
        elif g2.startswith('KIP') or g2.startswith('KIPH'):
            bucket = 3
        elif g2.startswith('KSP'):
            bucket = 4
        elif g2.startswith('KFS'):
            bucket = 5
        elif g2.startswith('PREMIX'):
            bucket = 6
        else:
            bucket = 9
        return (bucket, g2)

    return sorted(grades, key=_sort_key)


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


def _active_reservation_maps(conn, exclude_sales_order_id: Optional[int] = None) -> tuple[Dict[int, float], Dict[int, float]]:
    fg_reserved: Dict[int, float] = {}
    rap_reserved: Dict[int, float] = {}
    if not _table_exists(conn, "dispatch_stock_reservations"):
        return fg_reserved, rap_reserved
    q = """
        SELECT fg_lot_id, rap_lot_id, COALESCE(SUM(reserved_qty_kg),0)::float AS reserved_qty
        FROM dispatch_stock_reservations
        WHERE UPPER(COALESCE(status,'ACTIVE'))='ACTIVE'
    """
    params: Dict[str, Any] = {}
    if exclude_sales_order_id:
        q += " AND sales_order_id <> :sid"
        params['sid'] = exclude_sales_order_id
    q += " GROUP BY fg_lot_id, rap_lot_id"
    for r in conn.execute(text(q), params).mappings().all():
        if r.get('fg_lot_id') is not None:
            fg_reserved[int(r['fg_lot_id'])] = float(r.get('reserved_qty') or 0.0)
        if r.get('rap_lot_id') is not None:
            rap_reserved[int(r['rap_lot_id'])] = float(r.get('reserved_qty') or 0.0)
    return fg_reserved, rap_reserved


def _fg_available_rows_conn(conn, exclude_sales_order_id: Optional[int] = None, apply_reservations: bool = True):
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
        WHERE COALESCE(source_stage,'FG')='FG'
        GROUP BY fg_lot_id
    """)).all() or [])
    fg_reserved, _rap_reserved = _active_reservation_maps(conn, exclude_sales_order_id) if apply_reservations else ({}, {})
    out = []
    for r in lots:
        used = float(used_by_fg.get(r["id"], 0.0))
        reserved = float(fg_reserved.get(int(r['id']), 0.0))
        avail = float(r["weight_kg"] or 0.0) - used - reserved
        if avail > 0.0001:
            out.append({
                "source_stage": "FG",
                "source_key": f"FG:{r['id']}",
                "fg_lot_id": r["id"],
                "rap_lot_id": None,
                "fg_lot_no": r["lot_no"],
                "source_lot_no": r["lot_no"],
                "date": r["date"],
                "family": r["family"],
                "fg_grade": r["fg_grade"],
                "available_kg": avail,
                "cost_per_kg": float(r["cost_per_kg"] or 0.0),
                "remarks": r.get("remarks","") or '',
                "sort_seq": int(r['id'] or 0),
            })
    return out


def _fg_available_rows(exclude_sales_order_id: Optional[int] = None, apply_reservations: bool = True):
    with engine.begin() as conn:
        return _fg_available_rows_conn(conn, exclude_sales_order_id=exclude_sales_order_id, apply_reservations=apply_reservations)


def _rap_available_rows_conn(conn, exclude_sales_order_id: Optional[int] = None, apply_reservations: bool = True):
    rows = conn.execute(text("""
        SELECT rl.id AS rap_lot_id, l.id AS base_lot_id, l.lot_no,
               COALESCE(l.grade,'') AS grade, COALESCE(l.unit_cost,0)::float AS cost_per_kg,
               COALESCE(rl.available_qty,0)::float AS available_qty, COALESCE(rl.status,'') AS rap_status,
               COALESCE(l.qa_status,'') AS qa_status
        FROM rap_lot rl
        JOIN lot l ON l.id = rl.lot_id
        ORDER BY rl.id ASC
    """)).mappings().all()
    used_by_rap = dict(conn.execute(text("""
        SELECT rap_lot_id, COALESCE(SUM(qty_kg),0) AS used
        FROM dispatch_items
        WHERE COALESCE(source_stage,'FG')='RAP'
        GROUP BY rap_lot_id
    """)).all() or [])
    _fg_reserved, rap_reserved = _active_reservation_maps(conn, exclude_sales_order_id) if apply_reservations else ({}, {})
    out = []
    for r in rows:
        if str(r.get('qa_status') or '').upper() != 'APPROVED':
            continue
        grade = str(r.get('grade') or '').strip().upper()
        if not grade:
            continue
        used = float(used_by_rap.get(r['rap_lot_id'], 0.0))
        reserved = float(rap_reserved.get(int(r['rap_lot_id']), 0.0))
        avail = float(r.get('available_qty') or 0.0) - used - reserved
        if avail <= 0.0001:
            continue
        family = grade if grade in ('KRIP','KRFS','KRM','KRSP') else grade
        out.append({
            "source_stage": "RAP",
            "source_key": f"RAP:{r['rap_lot_id']}",
            "fg_lot_id": None,
            "rap_lot_id": r['rap_lot_id'],
            "base_lot_id": r.get('base_lot_id'),
            "fg_lot_no": r['lot_no'],
            "source_lot_no": r['lot_no'],
            "date": None,
            "family": family,
            "fg_grade": grade,
            "available_kg": avail,
            "cost_per_kg": float(r.get('cost_per_kg') or 0.0),
            "remarks": 'RAP Semi-Finished Dispatch',
            "sort_seq": int(r['rap_lot_id'] or 0),
        })
    return out


def _rap_available_rows(exclude_sales_order_id: Optional[int] = None, apply_reservations: bool = True):
    with engine.begin() as conn:
        return _rap_available_rows_conn(conn, exclude_sales_order_id=exclude_sales_order_id, apply_reservations=apply_reservations)


def _dispatch_available_rows(exclude_sales_order_id: Optional[int] = None, apply_reservations: bool = True):
    with engine.begin() as conn:
        return _fg_available_rows_conn(conn, exclude_sales_order_id=exclude_sales_order_id, apply_reservations=apply_reservations) + _rap_available_rows_conn(conn, exclude_sales_order_id=exclude_sales_order_id, apply_reservations=apply_reservations)


def _fetch_order(order_id: int):
    with engine.begin() as conn:
        head = conn.execute(text("SELECT * FROM dispatch_orders WHERE id=:id"), {"id": order_id}).mappings().first()
        items = conn.execute(text("""
            SELECT di.*,
                   COALESCE(di.source_stage,'FG') AS source_stage,
                   COALESCE(di.source_lot_no, di.fg_lot_no, '') AS source_lot_no,
                   rl.lot_id AS rap_base_lot_id
            FROM dispatch_items di
            LEFT JOIN rap_lot rl ON rl.id = di.rap_lot_id
            WHERE di.dispatch_id=:id
            ORDER BY di.id
        """), {"id": order_id}).mappings().all()
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
        totals_map = {
            int(r['dispatch_id']): (float(r.get('qty') or 0.0), float(r.get('val') or 0.0))
            for r in conn.execute(text("""
                SELECT dispatch_id, COALESCE(SUM(qty_kg),0) AS qty,
                       COALESCE(SUM(value),0) AS val
                FROM dispatch_items
                WHERE dispatch_id IN (
                    SELECT id FROM dispatch_orders WHERE date BETWEEN :s AND :e
                )
                GROUP BY dispatch_id
            """), {"s": start, "e": end}).mappings().all()
        }
        out = []
        for o in orders:
            qty, val = totals_map.get(int(o['id']), (0.0, 0.0))
            d = dict(o)
            d["total_qty_kg"] = qty
            d["total_value"] = val
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


def _sync_sales_order_item_metrics(conn, sales_order_id: int) -> None:
    item_rows = [dict(r) for r in conn.execute(text("""
        SELECT id,
               COALESCE(ordered_qty_kg,0)::float AS ordered_qty_kg,
               COALESCE(dispatched_qty_kg,0)::float AS dispatched_qty_kg,
               COALESCE(reserved_qty_kg,0)::float AS reserved_qty_kg
        FROM dispatch_sales_order_items
        WHERE sales_order_id=:id
        ORDER BY id
    """), {"id": sales_order_id}).mappings().all()]
    for row in item_rows:
        ordered = float(row.get('ordered_qty_kg') or 0.0)
        dispatched = float(row.get('dispatched_qty_kg') or 0.0)
        reserved = min(float(row.get('reserved_qty_kg') or 0.0), max(ordered - dispatched, 0.0))
        pending = max(ordered - dispatched, 0.0)
        if pending <= 0.0001:
            line_status = 'CLOSED'
        elif reserved >= pending - 0.0001:
            line_status = 'RESERVED'
        elif dispatched > 0 or reserved > 0:
            line_status = 'PARTIAL'
        else:
            line_status = 'OPEN'
        conn.execute(text("""
            UPDATE dispatch_sales_order_items
            SET reserved_qty_kg=:rq, line_status=:ls
            WHERE id=:id
        """), {"rq": reserved, "ls": line_status, "id": row['id']})


def _rebuild_sales_order_reservations(conn, sales_order_id: int) -> None:
    if not _table_exists(conn, 'dispatch_stock_reservations'):
        return
    conn.execute(text("DELETE FROM dispatch_stock_reservations WHERE sales_order_id=:id"), {'id': sales_order_id})
    conn.execute(text("UPDATE dispatch_sales_order_items SET reserved_qty_kg=0 WHERE sales_order_id=:id"), {'id': sales_order_id})

    items = [dict(r) for r in conn.execute(text("""
        SELECT id, fg_grade,
               COALESCE(ordered_qty_kg,0)::float AS ordered_qty_kg,
               COALESCE(dispatched_qty_kg,0)::float AS dispatched_qty_kg,
               COALESCE(source_preference,'AUTO') AS source_preference
        FROM dispatch_sales_order_items
        WHERE sales_order_id=:id
        ORDER BY id
    """), {'id': sales_order_id}).mappings().all()]

    avail_rows = _dispatch_available_rows(exclude_sales_order_id=sales_order_id, apply_reservations=True)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in avail_rows:
        grouped.setdefault(str(r.get('fg_grade') or '').strip(), []).append(dict(r))
    for grade in list(grouped.keys()):
        grouped[grade] = sorted(grouped[grade], key=lambda r: (str(r.get('date') or ''), int(r.get('sort_seq') or 0), str(r.get('source_lot_no') or '')))

    for it in items:
        grade = str(it.get('fg_grade') or '').strip()
        if not grade:
            continue
        pending = max(float(it.get('ordered_qty_kg') or 0.0) - float(it.get('dispatched_qty_kg') or 0.0), 0.0)
        if pending <= 0.0001:
            continue
        pref = str(it.get('source_preference') or 'AUTO').upper()
        reserved = 0.0
        for src in grouped.get(grade, []):
            if pending <= 0.0001:
                break
            if pref in ('FG','RAP') and str(src.get('source_stage') or '').upper() != pref:
                continue
            avail = float(src.get('available_kg') or 0.0)
            if avail <= 0.0001:
                continue
            take = min(avail, pending)
            conn.execute(text("""
                INSERT INTO dispatch_stock_reservations
                    (sales_order_id, sales_order_item_id, source_stage, fg_lot_id, rap_lot_id, source_lot_no, fg_grade, reserved_qty_kg, status)
                VALUES
                    (:sid, :siid, :ss, :fgid, :rid, :sln, :grade, :qty, 'ACTIVE')
            """), {
                'sid': sales_order_id,
                'siid': it['id'],
                'ss': src.get('source_stage') or 'FG',
                'fgid': src.get('fg_lot_id'),
                'rid': src.get('rap_lot_id'),
                'sln': src.get('source_lot_no') or src.get('fg_lot_no'),
                'grade': grade,
                'qty': take,
            })
            reserved += take
            pending -= take
            src['available_kg'] = avail - take
        conn.execute(text("UPDATE dispatch_sales_order_items SET reserved_qty_kg=:rq WHERE id=:id"), {'rq': reserved, 'id': it['id']})

    _sync_sales_order_item_metrics(conn, sales_order_id)
    _update_sales_order_status(conn, sales_order_id)


def _sales_orders(status_filter: str = "open") -> List[Dict[str, Any]]:
    with engine.begin() as conn:
        sql = "SELECT * FROM dispatch_sales_orders"
        params: Dict[str, Any] = {}
        if status_filter == "open":
            sql += " WHERE UPPER(COALESCE(status,'OPEN')) IN ('OPEN','PARTIAL','RESERVED')"
        sql += " ORDER BY COALESCE(due_date, order_date) ASC, order_date DESC, id DESC"
        heads = [dict(r) for r in conn.execute(text(sql), params).mappings().all()]
        for h in heads:
            _sync_sales_order_item_metrics(conn, h['id'])
            items = [dict(r) for r in conn.execute(text("""
                SELECT * FROM dispatch_sales_order_items
                WHERE sales_order_id=:id
                ORDER BY id
            """), {"id": h['id']}).mappings().all()]
            total_ordered = sum(float(i.get('ordered_qty_kg') or 0.0) for i in items)
            total_dispatched = sum(float(i.get('dispatched_qty_kg') or 0.0) for i in items)
            total_reserved = sum(float(i.get('reserved_qty_kg') or 0.0) for i in items)
            h['items'] = items
            h['total_ordered_kg'] = total_ordered
            h['total_dispatched_kg'] = total_dispatched
            h['total_reserved_kg'] = total_reserved
            h['total_pending_kg'] = max(total_ordered - total_dispatched, 0.0)
            h['total_shortage_kg'] = max(h['total_pending_kg'] - total_reserved, 0.0)
    return heads


def _fetch_sales_order(order_id: int) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    with engine.begin() as conn:
        _sync_sales_order_item_metrics(conn, order_id)
        head = conn.execute(text("SELECT * FROM dispatch_sales_orders WHERE id=:id"), {"id": order_id}).mappings().first()
        items = conn.execute(text("SELECT * FROM dispatch_sales_order_items WHERE sales_order_id=:id ORDER BY id"), {"id": order_id}).mappings().all()
    if not head:
        return None, []
    head_d = dict(head)
    item_list = [dict(r) for r in items]
    for it in item_list:
        ordered = float(it.get('ordered_qty_kg') or 0.0)
        dispatched = float(it.get('dispatched_qty_kg') or 0.0)
        reserved = float(it.get('reserved_qty_kg') or 0.0)
        it['pending_qty_kg'] = max(ordered - dispatched, 0.0)
        it['shortage_qty_kg'] = max(it['pending_qty_kg'] - reserved, 0.0)
    head_d['total_ordered_kg'] = sum(float(i.get('ordered_qty_kg') or 0.0) for i in item_list)
    head_d['total_dispatched_kg'] = sum(float(i.get('dispatched_qty_kg') or 0.0) for i in item_list)
    head_d['total_reserved_kg'] = sum(float(i.get('reserved_qty_kg') or 0.0) for i in item_list)
    head_d['total_pending_kg'] = max(head_d['total_ordered_kg'] - head_d['total_dispatched_kg'], 0.0)
    head_d['total_shortage_kg'] = max(head_d['total_pending_kg'] - head_d['total_reserved_kg'], 0.0)
    return head_d, item_list


def _update_sales_order_status(conn, sales_order_id: int) -> None:
    row = conn.execute(text("""
        SELECT
            COALESCE(SUM(ordered_qty_kg),0)::float AS ordered,
            COALESCE(SUM(dispatched_qty_kg),0)::float AS dispatched,
            COALESCE(SUM(reserved_qty_kg),0)::float AS reserved
        FROM dispatch_sales_order_items
        WHERE sales_order_id=:id
    """), {"id": sales_order_id}).mappings().first() or {"ordered":0.0, "dispatched":0.0, "reserved":0.0}
    ordered = float(row.get('ordered') or 0.0)
    dispatched = float(row.get('dispatched') or 0.0)
    reserved = float(row.get('reserved') or 0.0)
    pending = max(ordered - dispatched, 0.0)
    status = 'OPEN'
    if ordered > 0 and dispatched >= ordered - 0.0001:
        status = 'CLOSED'
    elif reserved >= pending - 0.0001 and pending > 0:
        status = 'RESERVED'
    elif dispatched > 0 or reserved > 0:
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
    _rebuild_sales_order_reservations(conn, sales_order_id)


def _fifo_dispatch_allocate(rows: List[Dict[str, Any]], requested_by_grade: Dict[str, float]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(str(r.get('fg_grade') or '').strip(), []).append(r)
    for grade, qty_need in requested_by_grade.items():
        remaining = float(qty_need or 0.0)
        eligible = sorted(grouped.get(grade, []), key=lambda r: (str(r.get('date') or ''), int(r.get('sort_seq') or 0), str(r.get('source_lot_no') or '')))
        total_avail = sum(float(r.get('available_kg') or 0.0) for r in eligible)
        if total_avail + 1e-6 < remaining:
            raise ValueError(f'Insufficient FIFO dispatch balance for {grade}. Pending {remaining:.1f} kg, available {total_avail:.1f} kg.')
        for src in eligible:
            if remaining <= 1e-6:
                break
            avail = float(src.get('available_kg') or 0.0)
            if avail <= 1e-6:
                continue
            take = min(avail, remaining)
            out.append({
                'source_stage': src.get('source_stage') or 'FG',
                'fg_lot_id': src.get('fg_lot_id'),
                'rap_lot_id': src.get('rap_lot_id'),
                'fg_lot_no': src.get('fg_lot_no') or src.get('source_lot_no'),
                'source_lot_no': src.get('source_lot_no') or src.get('fg_lot_no'),
                'fg_grade': src['fg_grade'],
                'qty_kg': take,
                'cost_per_kg': float(src.get('cost_per_kg') or 0.0),
                'value': float(src.get('cost_per_kg') or 0.0) * take,
            })
            remaining -= take
    return out


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
async def dispatch_customer_orders_get(request: Request, dep: None = Depends(require_roles('admin','dispatch','store','view','sales'))):
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
async def dispatch_customer_orders_post(request: Request, dep: None = Depends(require_roles('admin','dispatch','store','sales'))):
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
                (order_no, order_date, customer_name, customer_gstin, contact, customer_address, po_no, po_date, due_date, priority, remarks, status, created_by)
            VALUES
                (:ono, :d, :cn, :gst, :ct, :addr, :pono, :podt, :due, :prio, :rmk, 'OPEN', :cb)
            RETURNING id
        """), {
            'ono': order_no,
            'd': order_date,
            'cn': customer_name,
            'gst': (form.get('customer_gstin') or '').strip(),
            'ct': (form.get('contact') or '').strip(),
            'addr': (form.get('customer_address') or '').strip(),
            'pono': (form.get('po_no') or '').strip(),
            'podt': ((form.get('po_date') or '').strip() or None),
            'due': ((form.get('due_date') or '').strip() or None),
            'prio': ((form.get('priority') or 'NORMAL').strip() or 'NORMAL').upper(),
            'rmk': (form.get('remarks') or '').strip(),
            'cb': getattr(getattr(request,'state',None),'username','')
        }).mappings().first()['id']

        item_payload = []
        for idx, it in enumerate(items, start=1):
            item_payload.append({
                'sid': order_id,
                'g': it['fg_grade'],
                'q': it['ordered_qty_kg'],
                'r': it.get('remarks',''),
                'cd': ((form.get(f'committed_date_{idx}') or '').strip() or None),
                'sp': ((form.get(f'source_pref_{idx}') or 'AUTO').strip() or 'AUTO').upper(),
            })
        conn.execute(text("""
            INSERT INTO dispatch_sales_order_items(sales_order_id, fg_grade, ordered_qty_kg, dispatched_qty_kg, reserved_qty_kg, committed_date, source_preference, line_status, remarks)
            VALUES (:sid, :g, :q, 0, 0, :cd, :sp, 'OPEN', :r)
        """), item_payload)
        _rebuild_sales_order_reservations(conn, order_id)

    return RedirectResponse('/dispatch/customer-orders', status_code=303)


@router.get('/customer-orders/view/{order_id}', response_class=HTMLResponse)
async def dispatch_customer_order_view(request: Request, order_id: int, dep: None = Depends(require_roles('admin','dispatch','store','view','sales'))):
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


@router.get('/atp', response_class=HTMLResponse)
async def dispatch_atp(request: Request, dep: None = Depends(require_roles('admin','dispatch','store','view','sales'))):
    gross_rows = _dispatch_available_rows(apply_reservations=False)
    avail_rows = _dispatch_available_rows(apply_reservations=True)
    gross_by_grade: Dict[str, float] = {}
    avail_by_grade: Dict[str, float] = {}
    for r in gross_rows:
        g = str(r.get('fg_grade') or '').strip()
        gross_by_grade[g] = gross_by_grade.get(g, 0.0) + float(r.get('available_kg') or 0.0)
    for r in avail_rows:
        g = str(r.get('fg_grade') or '').strip()
        avail_by_grade[g] = avail_by_grade.get(g, 0.0) + float(r.get('available_kg') or 0.0)
    orders = _sales_orders('open')
    line_rows: List[Dict[str, Any]] = []
    for h in orders:
        for it in h.get('items', []):
            grade = str(it.get('fg_grade') or '').strip()
            ordered = float(it.get('ordered_qty_kg') or 0.0)
            dispatched = float(it.get('dispatched_qty_kg') or 0.0)
            reserved = float(it.get('reserved_qty_kg') or 0.0)
            pending = max(ordered - dispatched, 0.0)
            shortage = max(pending - reserved, 0.0)
            atp_avail = float(avail_by_grade.get(grade, 0.0))
            if pending <= 0.0001:
                atp_status = 'CLOSED'
            elif reserved >= pending - 0.0001:
                atp_status = 'READY'
            elif reserved > 0 or atp_avail > 0:
                atp_status = 'PARTIAL'
            else:
                atp_status = 'SHORTAGE'
            line_rows.append({
                'order_no': h.get('order_no'),
                'customer_name': h.get('customer_name'),
                'due_date': h.get('due_date'),
                'priority': h.get('priority') or 'NORMAL',
                'fg_grade': grade,
                'ordered_qty_kg': ordered,
                'dispatched_qty_kg': dispatched,
                'reserved_qty_kg': reserved,
                'pending_qty_kg': pending,
                'shortage_qty_kg': shortage,
                'gross_stock_kg': gross_by_grade.get(grade, 0.0),
                'available_stock_kg': atp_avail,
                'atp_status': atp_status,
                'sales_order_id': h.get('id'),
            })
    return templates.TemplateResponse('dispatch_atp.html', {
        'request': request,
        'rows': line_rows,
        'is_admin': _is_admin(request),
        **_tpl_auth(request),
    })


# ---------------- HOME ----------------
@router.get('/', response_class=HTMLResponse)
async def dispatch_home(request: Request, dep: None = Depends(require_roles('admin','dispatch','store','view','sales'))):
    rows = _dispatch_available_rows()
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
    err = request.query_params.get('err','')
    sales_order_id_raw = request.query_params.get('sales_order_id','').strip()
    rows = _dispatch_available_rows(exclude_sales_order_id=(int(sales_order_id_raw) if sales_order_id_raw.isdigit() else None))
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

    payload_rows = _dispatch_available_rows(exclude_sales_order_id=sales_order_id)
    requested_by_grade: Dict[str, float] = {}
    for k, v in form.items():
        if not k.startswith('q_'):
            continue
        try:
            fg_id = int(k.split('_',1)[1])
        except Exception:
            continue
        try:
            qty = float(v or 0.0)
        except Exception:
            qty = 0.0
        if qty <= 0:
            continue
        src = next((r for r in payload_rows if int(r.get('fg_lot_id') or 0) == fg_id), None)
        if not src:
            return RedirectResponse('/dispatch/create?err=Invalid+FG+lot+selected', status_code=303)
        requested_by_grade[src['fg_grade']] = requested_by_grade.get(src['fg_grade'], 0.0) + qty

    selected_sales_order = None
    selected_sales_order_items: List[Dict[str, Any]] = []
    if sales_order_id:
        selected_sales_order, selected_sales_order_items = _fetch_sales_order(sales_order_id)
        if not selected_sales_order:
            return RedirectResponse('/dispatch/create?err=Selected+customer+order+not+found', status_code=303)
        pending_by_grade = _sales_order_pending_by_grade(selected_sales_order_items)
        requested_by_grade = {g:q for g,q in pending_by_grade.items() if q > 0.0001}

    if not requested_by_grade:
        q = f'?err=Add+at+least+one+line+with+Qty+%3E+0'
        if sales_order_id:
            q += f'&sales_order_id={sales_order_id}'
        return RedirectResponse('/dispatch/create' + q, status_code=303)

    try:
        items: list[dict] = _fifo_dispatch_allocate(payload_rows, requested_by_grade)
    except ValueError as e:
        q = f'?err={str(e).replace(" ", "+")}'
        if sales_order_id:
            q += f'&sales_order_id={sales_order_id}'
        return RedirectResponse('/dispatch/create' + q, status_code=303)

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
            INSERT INTO dispatch_items(dispatch_id, fg_lot_id, rap_lot_id, source_stage, source_lot_no, fg_lot_no, fg_grade, qty_kg, cost_per_kg, value)
            VALUES (:did, :fid, :rid, :ss, :sno, :fno, :g, :q, :c, :v)
        """), [{'did': head_id, 'fid': it.get('fg_lot_id'), 'rid': it.get('rap_lot_id'), 'ss': it.get('source_stage') or 'FG', 'sno': it.get('source_lot_no') or it.get('fg_lot_no'), 'fno': it['fg_lot_no'], 'g': it['fg_grade'], 'q': it['qty_kg'], 'c': it['cost_per_kg'], 'v': it['value']} for it in items])

        if sales_order_id:
            _apply_dispatch_to_sales_order(conn, sales_order_id, items)

    return RedirectResponse(f'/dispatch/view/{head_id}', status_code=303)


# ---------------- LIST + CSV ----------------
@router.get('/orders', response_class=HTMLResponse)
async def dispatch_orders(request: Request, dep: None = Depends(require_roles('admin','dispatch','store','view','sales')), from_date: Optional[str] = None, to_date: Optional[str] = None, csv_export: int = 0):
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
async def dispatch_view(request: Request, order_id: int, dep: None = Depends(require_roles('admin','dispatch','store','view','sales'))):
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
async def dispatch_pdf(request: Request, order_id: int, dep: None = Depends(require_roles('admin','dispatch','store','view','sales'))):
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
