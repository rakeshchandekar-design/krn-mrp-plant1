from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import text
from starlette.templating import Jinja2Templates

from krn_mrp_app.deps import engine, require_roles

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# ---------------- DDL / seed ----------------
with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS mrp_grade_master (
            id SERIAL PRIMARY KEY,
            grade_code TEXT NOT NULL UNIQUE,
            grade_name TEXT,
            grade_family TEXT,
            stage_type TEXT,
            dispatchable BOOLEAN NOT NULL DEFAULT FALSE,
            uom TEXT NOT NULL DEFAULT 'KG',
            qa_required BOOLEAN NOT NULL DEFAULT TRUE,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            remarks TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS mrp_recipe_master (
            id SERIAL PRIMARY KEY,
            dispatch_grade TEXT NOT NULL,
            source_stage TEXT NOT NULL,
            source_family TEXT,
            source_grade TEXT,
            route_steps TEXT,
            fifo_mode TEXT NOT NULL DEFAULT 'STRICT',
            dispatchable_source_type TEXT NOT NULL DEFAULT 'AUTO',
            yield_basis_notes TEXT,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS mrp_yield_master (
            id SERIAL PRIMARY KEY,
            item_family TEXT NOT NULL,
            stage_name TEXT NOT NULL,
            standard_yield_pct DOUBLE PRECISION,
            recovery_pct DOUBLE PRECISION,
            oversize_target_pct DOUBLE PRECISION,
            capacity_kg_day DOUBLE PRECISION,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            remarks TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS mrp_run_header (
            id SERIAL PRIMARY KEY,
            run_label TEXT,
            created_by TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS mrp_run_lines (
            id SERIAL PRIMARY KEY,
            run_id INT NOT NULL,
            sales_order_id INT,
            sales_order_item_id INT,
            order_no TEXT,
            customer_name TEXT,
            due_date DATE,
            priority TEXT,
            dispatch_grade TEXT,
            source_stage TEXT,
            source_family TEXT,
            source_grade TEXT,
            ordered_qty_kg DOUBLE PRECISION DEFAULT 0,
            dispatched_qty_kg DOUBLE PRECISION DEFAULT 0,
            reserved_qty_kg DOUBLE PRECISION DEFAULT 0,
            pending_qty_kg DOUBLE PRECISION DEFAULT 0,
            gross_stock_kg DOUBLE PRECISION DEFAULT 0,
            available_stock_kg DOUBLE PRECISION DEFAULT 0,
            source_stage_stock_kg DOUBLE PRECISION DEFAULT 0,
            shortage_qty_kg DOUBLE PRECISION DEFAULT 0,
            atp_status TEXT,
            suggested_action TEXT,
            suggested_dispatch_qty_kg DOUBLE PRECISION DEFAULT 0,
            suggested_fg_qty_kg DOUBLE PRECISION DEFAULT 0,
            suggested_grinding_qty_kg DOUBLE PRECISION DEFAULT 0,
            suggested_anneal_qty_kg DOUBLE PRECISION DEFAULT 0,
            suggested_rap_qty_kg DOUBLE PRECISION DEFAULT 0,
            suggested_atom_qty_kg DOUBLE PRECISION DEFAULT 0,
            suggested_melt_qty_kg DOUBLE PRECISION DEFAULT 0,
            planned_capacity_days DOUBLE PRECISION,
            route_steps TEXT,
            line_status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS mrp_planned_order_header (
            id SERIAL PRIMARY KEY,
            plan_label TEXT,
            source_run_id INT,
            created_by TEXT,
            status TEXT NOT NULL DEFAULT 'OPEN',
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS mrp_planned_order_lines (
            id SERIAL PRIMARY KEY,
            plan_id INT NOT NULL,
            dispatch_grade TEXT,
            source_stage TEXT,
            source_family TEXT,
            source_grade TEXT,
            shortage_qty_kg DOUBLE PRECISION DEFAULT 0,
            planned_fg_qty_kg DOUBLE PRECISION DEFAULT 0,
            planned_grinding_qty_kg DOUBLE PRECISION DEFAULT 0,
            planned_anneal_qty_kg DOUBLE PRECISION DEFAULT 0,
            planned_rap_qty_kg DOUBLE PRECISION DEFAULT 0,
            planned_atom_qty_kg DOUBLE PRECISION DEFAULT 0,
            planned_melt_qty_kg DOUBLE PRECISION DEFAULT 0,
            planned_capacity_days DOUBLE PRECISION,
            route_steps TEXT,
            next_action_stage TEXT,
            next_action_qty_kg DOUBLE PRECISION DEFAULT 0,
            linked_run_line_count INT DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'OPEN',
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """))

    # seed grade master using current dispatch/FG logic
    seed_grades = [
        {"code":"KRIP","name":"KRIP Semi-Finished Powder","family":"KRIP","stage":"RAP","dispatchable":True,"remarks":"Direct dispatchable semi-finished grade"},
        {"code":"KRFS","name":"KRFS Semi-Finished Powder","family":"KRFS","stage":"RAP","dispatchable":True,"remarks":"Direct dispatchable semi-finished grade"},
        {"code":"KRM","name":"KRM Semi-Finished Powder","family":"KRM","stage":"RAP","dispatchable":True,"remarks":"Direct dispatchable semi-finished grade"},
        {"code":"KRSP","name":"KRSP Semi-Finished Powder","family":"KRSP","stage":"RAP","dispatchable":True,"remarks":"Direct dispatchable semi-finished grade"},
        {"code":"KIP","name":"KIP Intermediate Grade","family":"KIP","stage":"GRINDING","dispatchable":False,"remarks":"Intermediate family / planning bucket"},
        {"code":"KFS","name":"KFS Intermediate Grade","family":"KFS","stage":"GRINDING","dispatchable":False,"remarks":"Intermediate family / planning bucket"},
        {"code":"KIPM","name":"KIP M Intermediate Grade","family":"KIPM","stage":"GRINDING","dispatchable":False,"remarks":"Intermediate family / planning bucket"},
        {"code":"KSP","name":"KSP Intermediate Grade","family":"KSP","stage":"GRINDING","dispatchable":False,"remarks":"Intermediate family / planning bucket"},
    ]
    try:
        from krn_mrp_app.fg.routes import FG_FAMILY
        for grade, fam in FG_FAMILY.items():
            seed_grades.append({
                "code": str(grade),
                "name": str(grade),
                "family": str(fam),
                "stage": "FG",
                "dispatchable": True,
                "remarks": "Dispatchable FG grade"
            })
    except Exception:
        pass
    for g in seed_grades:
        conn.execute(text("""
            INSERT INTO mrp_grade_master(grade_code, grade_name, grade_family, stage_type, dispatchable, remarks)
            VALUES (:code, :name, :family, :stage, :dispatchable, :remarks)
            ON CONFLICT (grade_code) DO NOTHING
        """), g)

    seed_recipes = [
        {"dispatch_grade":"KRIP","source_stage":"RAP","source_family":"KRIP","source_grade":"KRIP","route_steps":"RAP -> Dispatch","yield_basis_notes":"Semi-finished dispatch directly from RAP"},
        {"dispatch_grade":"KRFS","source_stage":"RAP","source_family":"KRFS","source_grade":"KRFS","route_steps":"RAP -> Dispatch","yield_basis_notes":"Semi-finished dispatch directly from RAP"},
        {"dispatch_grade":"KRM","source_stage":"RAP","source_family":"KRM","source_grade":"KRM","route_steps":"RAP -> Dispatch","yield_basis_notes":"Semi-finished dispatch directly from RAP"},
        {"dispatch_grade":"KRSP","source_stage":"RAP","source_family":"KRSP","source_grade":"KRSP","route_steps":"RAP -> Dispatch","yield_basis_notes":"Semi-finished dispatch directly from RAP"},
    ]
    try:
        from krn_mrp_app.fg.routes import FG_FAMILY
        for grade, fam in FG_FAMILY.items():
            seed_recipes.append({
                "dispatch_grade": str(grade),
                "source_stage": "FG",
                "source_family": str(fam),
                "source_grade": str(grade),
                "route_steps": f"{fam} route -> Packing/FG -> Dispatch",
                "yield_basis_notes": "Dispatchable FG route"
            })
    except Exception:
        pass
    for r in seed_recipes:
        exists = conn.execute(text("""
            SELECT 1 FROM mrp_recipe_master
            WHERE dispatch_grade=:dispatch_grade AND source_stage=:source_stage AND COALESCE(source_grade,'')=COALESCE(:source_grade,'')
            LIMIT 1
        """), r).first()
        if not exists:
            conn.execute(text("""
                INSERT INTO mrp_recipe_master(dispatch_grade, source_stage, source_family, source_grade, route_steps, yield_basis_notes)
                VALUES (:dispatch_grade, :source_stage, :source_family, :source_grade, :route_steps, :yield_basis_notes)
            """), r)

    seed_yields = [
        {"item_family":"KIP","stage_name":"MELTING","standard_yield_pct":100.0,"recovery_pct":100.0,"oversize_target_pct":None,"capacity_kg_day":None,"remarks":"Starter master value - adjust plant actuals"},
        {"item_family":"KIP","stage_name":"ATOMIZATION","standard_yield_pct":95.0,"recovery_pct":95.0,"oversize_target_pct":None,"capacity_kg_day":None,"remarks":"Starter master value - adjust plant actuals"},
        {"item_family":"KIP","stage_name":"ANNEALING","standard_yield_pct":98.5,"recovery_pct":98.5,"oversize_target_pct":None,"capacity_kg_day":6000.0,"remarks":"Starter master value - adjust plant actuals"},
        {"item_family":"KIP","stage_name":"GRINDING","standard_yield_pct":87.5,"recovery_pct":87.5,"oversize_target_pct":12.5,"capacity_kg_day":10000.0,"remarks":"Starter master value - adjust plant actuals"},
        {"item_family":"KFS","stage_name":"MELTING","standard_yield_pct":100.0,"recovery_pct":100.0,"oversize_target_pct":None,"capacity_kg_day":None,"remarks":"Starter master value - adjust plant actuals"},
        {"item_family":"KFS","stage_name":"ATOMIZATION","standard_yield_pct":95.0,"recovery_pct":95.0,"oversize_target_pct":None,"capacity_kg_day":None,"remarks":"Starter master value - adjust plant actuals"},
        {"item_family":"KFS","stage_name":"ANNEALING","standard_yield_pct":98.5,"recovery_pct":98.5,"oversize_target_pct":None,"capacity_kg_day":6000.0,"remarks":"Starter master value - adjust plant actuals"},
        {"item_family":"KFS","stage_name":"GRINDING","standard_yield_pct":92.0,"recovery_pct":92.0,"oversize_target_pct":8.0,"capacity_kg_day":10000.0,"remarks":"Starter master value - adjust plant actuals"},
        {"item_family":"KIPM","stage_name":"ANNEALING","standard_yield_pct":98.5,"recovery_pct":98.5,"oversize_target_pct":None,"capacity_kg_day":6000.0,"remarks":"Starter master value - adjust plant actuals"},
        {"item_family":"KIPM","stage_name":"GRINDING","standard_yield_pct":87.5,"recovery_pct":87.5,"oversize_target_pct":12.5,"capacity_kg_day":10000.0,"remarks":"Starter master value - adjust plant actuals"},
        {"item_family":"KSP","stage_name":"ANNEALING","standard_yield_pct":98.5,"recovery_pct":98.5,"oversize_target_pct":None,"capacity_kg_day":6000.0,"remarks":"Starter master value - adjust plant actuals"},
        {"item_family":"KSP","stage_name":"GRINDING","standard_yield_pct":87.5,"recovery_pct":87.5,"oversize_target_pct":12.5,"capacity_kg_day":10000.0,"remarks":"Starter master value - adjust plant actuals"},
    ]
    for y in seed_yields:
        exists = conn.execute(text("""
            SELECT 1 FROM mrp_yield_master
            WHERE item_family=:item_family AND stage_name=:stage_name
            LIMIT 1
        """), y).first()
        if not exists:
            conn.execute(text("""
                INSERT INTO mrp_yield_master(item_family, stage_name, standard_yield_pct, recovery_pct, oversize_target_pct, capacity_kg_day, remarks)
                VALUES (:item_family, :stage_name, :standard_yield_pct, :recovery_pct, :oversize_target_pct, :capacity_kg_day, :remarks)
            """), y)


def _tpl_auth(request: Request) -> dict:
    sess = (getattr(request, "session", {}) or {})
    return {"user": sess.get("username", "") or "", "role": sess.get("role", "guest") or "guest"}


def _is_admin(request: Request) -> bool:
    sess = (getattr(request, "session", {}) or {})
    role = str(sess.get("role") or getattr(getattr(request, 'state', None), 'role', '') or '').lower()
    return role == 'admin'


def _rows(sql: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    with engine.begin() as conn:
        return [dict(r) for r in conn.execute(text(sql), params or {}).mappings().all()]


def _scalar(sql: str, params: Dict[str, Any] | None = None):
    with engine.begin() as conn:
        return conn.execute(text(sql), params or {}).scalar()


def _yield_pct(conn, family: str, stage_name: str, default_pct: float = 100.0) -> float:
    row = conn.execute(text("""
        SELECT COALESCE(recovery_pct, standard_yield_pct, :d)::float AS pct
        FROM mrp_yield_master
        WHERE UPPER(COALESCE(item_family,''))=:fam
          AND UPPER(COALESCE(stage_name,''))=:stg
          AND COALESCE(is_active,TRUE)=TRUE
        ORDER BY id DESC
        LIMIT 1
    """), {'fam': str(family or '').strip().upper(), 'stg': str(stage_name or '').strip().upper(), 'd': float(default_pct)}).mappings().first()
    pct = float((row or {}).get('pct') or default_pct)
    return pct if pct > 0 else default_pct


def _capacity_kg_day(conn, family: str, stage_name: str, default_val: float = 0.0) -> float:
    row = conn.execute(text("""
        SELECT COALESCE(capacity_kg_day, :d)::float AS cap
        FROM mrp_yield_master
        WHERE UPPER(COALESCE(item_family,''))=:fam
          AND UPPER(COALESCE(stage_name,''))=:stg
          AND COALESCE(is_active,TRUE)=TRUE
        ORDER BY id DESC
        LIMIT 1
    """), {'fam': str(family or '').strip().upper(), 'stg': str(stage_name or '').strip().upper(), 'd': float(default_val)}).mappings().first()
    cap = float((row or {}).get('cap') or default_val)
    return cap if cap > 0 else default_val


def _recipe_map(conn) -> Dict[str, Dict[str, Any]]:
    rows = conn.execute(text("""
        SELECT dispatch_grade, source_stage, source_family, source_grade, route_steps
        FROM mrp_recipe_master
        WHERE COALESCE(is_active,TRUE)=TRUE
        ORDER BY dispatch_grade, id
    """)).mappings().all()
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        dg = str(r.get('dispatch_grade') or '').strip()
        if dg and dg not in out:
            out[dg] = dict(r)
    return out


def _active_reservation_maps(conn) -> tuple[Dict[int, float], Dict[int, float]]:
    fg_reserved: Dict[int, float] = {}
    rap_reserved: Dict[int, float] = {}
    try:
        rows = conn.execute(text("""
            SELECT fg_lot_id, rap_lot_id, COALESCE(SUM(reserved_qty_kg),0)::float AS reserved_qty
            FROM dispatch_stock_reservations
            WHERE UPPER(COALESCE(status,'ACTIVE'))='ACTIVE'
            GROUP BY fg_lot_id, rap_lot_id
        """)).mappings().all()
        for r in rows:
            if r.get('fg_lot_id') is not None:
                fg_reserved[int(r['fg_lot_id'])] = float(r.get('reserved_qty') or 0.0)
            if r.get('rap_lot_id') is not None:
                rap_reserved[int(r['rap_lot_id'])] = float(r.get('reserved_qty') or 0.0)
    except Exception:
        pass
    return fg_reserved, rap_reserved


def _dispatch_stock_maps(conn) -> tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    gross: Dict[str, float] = {}
    available: Dict[str, float] = {}
    source_stage_stock: Dict[str, float] = {}

    fg_reserved, rap_reserved = _active_reservation_maps(conn)

    try:
        used_by_fg = {int(r['fg_lot_id']): float(r.get('used') or 0.0) for r in conn.execute(text("""
            SELECT fg_lot_id, COALESCE(SUM(qty_kg),0)::float AS used
            FROM dispatch_items
            WHERE COALESCE(source_stage,'FG')='FG'
            GROUP BY fg_lot_id
        """)).mappings().all() if r.get('fg_lot_id') is not None}
    except Exception:
        used_by_fg = {}

    try:
        fg_rows = conn.execute(text("""
            SELECT id, family, fg_grade, COALESCE(weight_kg,0)::float AS weight_kg
            FROM fg_lots
            WHERE UPPER(COALESCE(qa_status,''))='APPROVED'
        """)).mappings().all()
    except Exception:
        fg_rows = []
    for r in fg_rows:
        grade = str(r.get('fg_grade') or '').strip()
        fam = str(r.get('family') or '').strip().upper()
        used = float(used_by_fg.get(int(r['id']), 0.0))
        reserved = float(fg_reserved.get(int(r['id']), 0.0))
        gross_qty = max(float(r.get('weight_kg') or 0.0) - used, 0.0)
        avail_qty = max(gross_qty - reserved, 0.0)
        if grade:
            gross[grade] = gross.get(grade, 0.0) + gross_qty
            available[grade] = available.get(grade, 0.0) + avail_qty
        if fam:
            source_stage_stock[f'FG_FAMILY:{fam}'] = source_stage_stock.get(f'FG_FAMILY:{fam}', 0.0) + avail_qty

    try:
        used_by_rap = {int(r['rap_lot_id']): float(r.get('used') or 0.0) for r in conn.execute(text("""
            SELECT rap_lot_id, COALESCE(SUM(qty_kg),0)::float AS used
            FROM dispatch_items
            WHERE COALESCE(source_stage,'FG')='RAP'
            GROUP BY rap_lot_id
        """)).mappings().all() if r.get('rap_lot_id') is not None}
    except Exception:
        used_by_rap = {}

    try:
        rap_rows = conn.execute(text("""
            SELECT rl.id AS rap_lot_id, COALESCE(l.grade,'') AS grade,
                   COALESCE(rl.available_qty,0)::float AS available_qty,
                   COALESCE(l.qa_status,'') AS qa_status
            FROM rap_lot rl
            JOIN lot l ON l.id = rl.lot_id
        """)).mappings().all()
    except Exception:
        rap_rows = []
    for r in rap_rows:
        if str(r.get('qa_status') or '').upper() != 'APPROVED':
            continue
        grade = str(r.get('grade') or '').strip().upper()
        if not grade:
            continue
        used = float(used_by_rap.get(int(r['rap_lot_id']), 0.0))
        reserved = float(rap_reserved.get(int(r['rap_lot_id']), 0.0))
        gross_qty = max(float(r.get('available_qty') or 0.0) - used, 0.0)
        avail_qty = max(gross_qty - reserved, 0.0)
        gross[grade] = gross.get(grade, 0.0) + gross_qty
        available[grade] = available.get(grade, 0.0) + avail_qty
        source_stage_stock[f'RAP_GRADE:{grade}'] = source_stage_stock.get(f'RAP_GRADE:{grade}', 0.0) + avail_qty

    # Grinding family stock (approved main available after FG usage/reservations)
    try:
        fg_allocs = conn.execute(text("SELECT src_alloc_json FROM fg_lots WHERE UPPER(COALESCE(qa_status,'')) IN ('APPROVED','PENDING','HOLD')")).scalars().all()
        main_used: Dict[str, float] = {}
        for raw in fg_allocs:
            try:
                amap = json.loads(raw or '{}')
            except Exception:
                amap = {}
            for k, v in (amap.items() if isinstance(amap, dict) else []):
                key = str(k or '')
                if key.startswith('OV80|') or key.startswith('OV40|') or not key:
                    continue
                try:
                    main_used[key] = main_used.get(key, 0.0) + float(v or 0.0)
                except Exception:
                    pass
        grind_rows = conn.execute(text("""
            SELECT lot_no, grade, COALESCE(weight_kg,0)::float AS weight_kg,
                   COALESCE(oversize_p80_kg,0)::float AS p80,
                   COALESCE(oversize_p40_kg,0)::float AS p40
            FROM grinding_lots
            WHERE UPPER(COALESCE(qa_status,''))='APPROVED'
        """)).mappings().all()
    except Exception:
        grind_rows = []
        main_used = {}
    def _fam_from_grade(g: str) -> str:
        g2 = str(g or '').strip().upper()
        if g2.startswith('KIPM') or g2.startswith('KIP M') or g2.startswith('KRM'):
            return 'KIPM'
        if g2.startswith('KSP') or g2.startswith('KRSP'):
            return 'KSP'
        if g2.startswith('KFS') or g2.startswith('KRFS'):
            return 'KFS'
        return 'KIP'
    for r in grind_rows:
        fam = _fam_from_grade(r.get('grade'))
        main_qty = max(float(r.get('weight_kg') or 0.0) - float(r.get('p80') or 0.0) - float(r.get('p40') or 0.0), 0.0)
        avail_qty = max(main_qty - float(main_used.get(str(r.get('lot_no') or ''), 0.0)), 0.0)
        source_stage_stock[f'GRINDING_FAMILY:{fam}'] = source_stage_stock.get(f'GRINDING_FAMILY:{fam}', 0.0) + avail_qty

    # Anneal family stock available after grinding allocations
    try:
        used_by_anneal: Dict[str, float] = {}
        for raw in conn.execute(text("SELECT src_alloc_json FROM grinding_lots")).scalars().all():
            try:
                amap = json.loads(raw or '{}')
            except Exception:
                amap = {}
            for k, v in (amap.items() if isinstance(amap, dict) else []):
                used_by_anneal[str(k)] = used_by_anneal.get(str(k), 0.0) + float(v or 0.0)
        anneal_rows = conn.execute(text("""
            SELECT lot_no, grade, COALESCE(weight_kg,0)::float AS weight_kg
            FROM anneal_lots
            WHERE UPPER(COALESCE(qa_status,''))='APPROVED'
        """)).mappings().all()
    except Exception:
        used_by_anneal = {}
        anneal_rows = []
    for r in anneal_rows:
        fam = _fam_from_grade(r.get('grade'))
        avail_qty = max(float(r.get('weight_kg') or 0.0) - float(used_by_anneal.get(str(r.get('lot_no') or ''), 0.0)), 0.0)
        source_stage_stock[f'ANNEALING_FAMILY:{fam}'] = source_stage_stock.get(f'ANNEALING_FAMILY:{fam}', 0.0) + avail_qty

    return gross, available, source_stage_stock


def _open_sales_order_lines(conn) -> List[Dict[str, Any]]:
    rows = conn.execute(text("""
        SELECT h.id AS sales_order_id, h.order_no, h.customer_name, h.due_date, h.priority, h.status AS header_status,
               i.id AS sales_order_item_id, i.fg_grade, i.ordered_qty_kg, i.dispatched_qty_kg, i.reserved_qty_kg,
               i.committed_date, i.source_preference, i.line_status, i.remarks
        FROM dispatch_sales_orders h
        JOIN dispatch_sales_order_items i ON i.sales_order_id = h.id
        WHERE UPPER(COALESCE(h.status,'OPEN')) NOT IN ('CLOSED','CANCELLED')
        ORDER BY COALESCE(h.due_date, h.order_date), h.order_no, i.id
    """)).mappings().all()
    out = []
    for r in rows:
        d = dict(r)
        ordered = float(d.get('ordered_qty_kg') or 0.0)
        dispatched = float(d.get('dispatched_qty_kg') or 0.0)
        reserved = float(d.get('reserved_qty_kg') or 0.0)
        pending = max(ordered - dispatched, 0.0)
        if pending <= 0.0001:
            continue
        d['pending_qty_kg'] = pending
        d['shortage_qty_kg'] = max(pending - reserved, 0.0)
        out.append(d)
    return out


def _build_mrp_lines(conn) -> tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    recipe_map = _recipe_map(conn)
    gross_map, available_map, source_stock_map = _dispatch_stock_maps(conn)
    open_lines = _open_sales_order_lines(conn)
    lines: List[Dict[str, Any]] = []
    suggestion_bucket: Dict[tuple, Dict[str, Any]] = {}
    ready_lines = shortage_lines = 0
    total_pending = total_shortage = 0.0

    for ln in open_lines:
        grade = str(ln.get('fg_grade') or '').strip()
        pending = float(ln.get('pending_qty_kg') or 0.0)
        reserved = float(ln.get('reserved_qty_kg') or 0.0)
        dispatched = float(ln.get('dispatched_qty_kg') or 0.0)
        gross_stock = float(gross_map.get(grade, 0.0))
        avail_stock = float(available_map.get(grade, 0.0))
        shortage = max(pending - avail_stock, 0.0)
        recipe = recipe_map.get(grade, {})
        source_stage = str(recipe.get('source_stage') or 'FG').upper()
        source_family = str(recipe.get('source_family') or '').upper()
        source_grade = str(recipe.get('source_grade') or '').upper()
        route_steps = str(recipe.get('route_steps') or '')

        if source_stage == 'RAP':
            source_key = f'RAP_GRADE:{source_grade or grade.upper()}'
            source_stage_stock = float(source_stock_map.get(source_key, 0.0))
        elif source_stage == 'FG':
            source_key = f'GRINDING_FAMILY:{source_family or "KIP"}'
            source_stage_stock = float(source_stock_map.get(source_key, 0.0))
        else:
            source_key = f'{source_stage}_FAMILY:{source_family}'
            source_stage_stock = float(source_stock_map.get(source_key, 0.0))

        atp_status = 'READY' if shortage <= 0.0001 else ('PARTIAL' if reserved > 0.0001 or avail_stock > 0.0001 else 'SHORTAGE')
        if shortage <= 0.0001:
            suggested_action = 'Dispatch from available stock'
            ready_lines += 1
        elif source_stage == 'RAP':
            suggested_action = 'Produce / release RAP stock'
            shortage_lines += 1
        elif source_stage == 'FG':
            if source_stage_stock > 0.0001:
                suggested_action = 'Convert available source stock to FG'
            else:
                suggested_action = 'Plan upstream production for FG route'
            shortage_lines += 1
        else:
            suggested_action = f'Plan production at {source_stage}'
            shortage_lines += 1

        grind_y = _yield_pct(conn, source_family or 'KIP', 'GRINDING', 100.0) / 100.0
        anneal_y = _yield_pct(conn, source_family or 'KIP', 'ANNEALING', 100.0) / 100.0
        atom_y = _yield_pct(conn, source_family or 'KIP', 'ATOMIZATION', 100.0) / 100.0
        melt_y = _yield_pct(conn, source_family or 'KIP', 'MELTING', 100.0) / 100.0
        cap = _capacity_kg_day(conn, source_family or 'KIP', 'GRINDING' if source_stage == 'FG' else source_stage, 0.0)

        sug_dispatch = shortage
        sug_fg = shortage if source_stage == 'FG' else 0.0
        sug_grind = shortage if source_stage == 'FG' else 0.0
        sug_anneal = (sug_grind / grind_y) if (source_stage == 'FG' and grind_y > 0) else 0.0
        sug_rap = shortage if source_stage == 'RAP' else ((sug_anneal / anneal_y) if (source_stage == 'FG' and anneal_y > 0) else 0.0)
        sug_atom = (sug_rap / atom_y) if (source_stage in ('FG','RAP') and atom_y > 0 and source_stage != 'RAP') else 0.0
        sug_melt = (sug_atom / melt_y) if (sug_atom > 0 and melt_y > 0) else 0.0
        plan_days = (sug_grind / cap) if cap > 0 and sug_grind > 0 else None

        line = {
            **ln,
            'gross_stock_kg': gross_stock,
            'available_stock_kg': avail_stock,
            'source_stage': source_stage,
            'source_family': source_family,
            'source_grade': source_grade,
            'source_stage_stock_kg': source_stage_stock,
            'shortage_qty_kg': shortage,
            'atp_status': atp_status,
            'suggested_action': suggested_action,
            'suggested_dispatch_qty_kg': round(sug_dispatch, 2),
            'suggested_fg_qty_kg': round(sug_fg, 2),
            'suggested_grinding_qty_kg': round(sug_grind, 2),
            'suggested_anneal_qty_kg': round(sug_anneal, 2),
            'suggested_rap_qty_kg': round(sug_rap, 2),
            'suggested_atom_qty_kg': round(sug_atom, 2),
            'suggested_melt_qty_kg': round(sug_melt, 2),
            'planned_capacity_days': (round(plan_days, 2) if plan_days is not None else None),
            'route_steps': route_steps,
        }
        lines.append(line)
        total_pending += pending
        total_shortage += shortage
        key = (grade, source_stage, source_family, source_grade)
        bucket = suggestion_bucket.setdefault(key, {
            'dispatch_grade': grade,
            'source_stage': source_stage,
            'source_family': source_family,
            'source_grade': source_grade,
            'route_steps': route_steps,
            'shortage_qty_kg': 0.0,
            'suggested_fg_qty_kg': 0.0,
            'suggested_grinding_qty_kg': 0.0,
            'suggested_anneal_qty_kg': 0.0,
            'suggested_rap_qty_kg': 0.0,
            'suggested_atom_qty_kg': 0.0,
            'suggested_melt_qty_kg': 0.0,
            'planned_capacity_days': 0.0,
        })
        bucket['shortage_qty_kg'] += shortage
        bucket['suggested_fg_qty_kg'] += sug_fg
        bucket['suggested_grinding_qty_kg'] += sug_grind
        bucket['suggested_anneal_qty_kg'] += sug_anneal
        bucket['suggested_rap_qty_kg'] += sug_rap
        bucket['suggested_atom_qty_kg'] += sug_atom
        bucket['suggested_melt_qty_kg'] += sug_melt
        if plan_days:
            bucket['planned_capacity_days'] += plan_days

    summary = {
        'open_lines': len(lines),
        'ready_lines': ready_lines,
        'shortage_lines': shortage_lines,
        'total_pending_kg': round(total_pending, 2),
        'total_shortage_kg': round(total_shortage, 2),
    }
    suggestions = list(suggestion_bucket.values())
    suggestions.sort(key=lambda x: (-float(x.get('shortage_qty_kg') or 0.0), x.get('dispatch_grade') or ''))
    return lines, summary, suggestions


def _store_mrp_run(conn, created_by: str, lines: List[Dict[str, Any]]) -> int:
    run_row = conn.execute(text("""
        INSERT INTO mrp_run_header(run_label, created_by, notes)
        VALUES (:lbl, :by, :notes)
        RETURNING id
    """), {
        'lbl': datetime.now().strftime('MRP-%Y%m%d-%H%M%S'),
        'by': created_by,
        'notes': 'Auto-generated shortage and production suggestion snapshot'
    }).mappings().first()
    run_id = int(run_row['id'])
    if lines:
        conn.execute(text("""
            INSERT INTO mrp_run_lines(
                run_id, sales_order_id, sales_order_item_id, order_no, customer_name, due_date, priority,
                dispatch_grade, source_stage, source_family, source_grade,
                ordered_qty_kg, dispatched_qty_kg, reserved_qty_kg, pending_qty_kg,
                gross_stock_kg, available_stock_kg, source_stage_stock_kg, shortage_qty_kg,
                atp_status, suggested_action,
                suggested_dispatch_qty_kg, suggested_fg_qty_kg, suggested_grinding_qty_kg,
                suggested_anneal_qty_kg, suggested_rap_qty_kg, suggested_atom_qty_kg, suggested_melt_qty_kg,
                planned_capacity_days, route_steps, line_status
            ) VALUES (
                :run_id, :sales_order_id, :sales_order_item_id, :order_no, :customer_name, :due_date, :priority,
                :dispatch_grade, :source_stage, :source_family, :source_grade,
                :ordered_qty_kg, :dispatched_qty_kg, :reserved_qty_kg, :pending_qty_kg,
                :gross_stock_kg, :available_stock_kg, :source_stage_stock_kg, :shortage_qty_kg,
                :atp_status, :suggested_action,
                :suggested_dispatch_qty_kg, :suggested_fg_qty_kg, :suggested_grinding_qty_kg,
                :suggested_anneal_qty_kg, :suggested_rap_qty_kg, :suggested_atom_qty_kg, :suggested_melt_qty_kg,
                :planned_capacity_days, :route_steps, :line_status
            )
        """), [{**ln, 'run_id': run_id} for ln in lines])
    return run_id


def _fetch_run(run_id: int) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    with engine.begin() as conn:
        head = conn.execute(text("SELECT * FROM mrp_run_header WHERE id=:id"), {'id': run_id}).mappings().first()
        lines = conn.execute(text("SELECT * FROM mrp_run_lines WHERE run_id=:id ORDER BY due_date NULLS LAST, order_no, sales_order_item_id"), {'id': run_id}).mappings().all()
    return (dict(head) if head else None, [dict(r) for r in lines])


@router.get("/", response_class=HTMLResponse)
async def mrp_home(request: Request, dep: None = Depends(require_roles('admin','sales','view'))):
    counts = {
        'grades': int(_scalar("SELECT COUNT(*) FROM mrp_grade_master WHERE COALESCE(is_active,TRUE)=TRUE") or 0),
        'recipes': int(_scalar("SELECT COUNT(*) FROM mrp_recipe_master WHERE COALESCE(is_active,TRUE)=TRUE") or 0),
        'yields': int(_scalar("SELECT COUNT(*) FROM mrp_yield_master WHERE COALESCE(is_active,TRUE)=TRUE") or 0),
        'runs': int(_scalar("SELECT COUNT(*) FROM mrp_run_header") or 0),
        'plans': int(_scalar("SELECT COUNT(*) FROM mrp_planned_order_header") or 0),
    }
    dispatchable = _rows("""
        SELECT grade_code, grade_family, stage_type
        FROM mrp_grade_master
        WHERE COALESCE(is_active,TRUE)=TRUE AND COALESCE(dispatchable,FALSE)=TRUE
        ORDER BY stage_type, grade_code
        LIMIT 16
    """)
    latest_run = _rows("SELECT id, run_label, created_by, created_at FROM mrp_run_header ORDER BY id DESC LIMIT 1")
    latest_run = latest_run[0] if latest_run else None
    latest_summary = None
    if latest_run:
        latest_summary = _rows("""
            SELECT COUNT(*)::int AS open_lines,
                   COALESCE(SUM(CASE WHEN COALESCE(shortage_qty_kg,0) > 0.0001 THEN 1 ELSE 0 END),0)::int AS shortage_lines,
                   COALESCE(SUM(pending_qty_kg),0)::float AS total_pending_kg,
                   COALESCE(SUM(shortage_qty_kg),0)::float AS total_shortage_kg
            FROM mrp_run_lines WHERE run_id=:id
        """, {'id': latest_run['id']})
        latest_summary = latest_summary[0] if latest_summary else None
    latest_plan = _rows("SELECT id, plan_label, source_run_id, created_by, status, created_at FROM mrp_planned_order_header ORDER BY id DESC LIMIT 1")
    latest_plan = latest_plan[0] if latest_plan else None
    return templates.TemplateResponse('mrp_home.html', {
        'request': request,
        'counts': counts,
        'dispatchable': dispatchable,
        'latest_run': latest_run,
        'latest_summary': latest_summary,
        'latest_plan': latest_plan,
        'is_admin': _is_admin(request),
        **_tpl_auth(request),
    })


@router.get('/grades', response_class=HTMLResponse)
async def mrp_grades(request: Request, dep: None = Depends(require_roles('admin','sales','view'))):
    rows = _rows("SELECT * FROM mrp_grade_master ORDER BY COALESCE(is_active,TRUE) DESC, stage_type, grade_code")
    return templates.TemplateResponse('mrp_grade_master.html', {
        'request': request,
        'rows': rows,
        'is_admin': _is_admin(request),
        **_tpl_auth(request),
    })


@router.post('/grades')
async def mrp_grades_post(request: Request,
                          grade_code: str = Form(...),
                          grade_name: str = Form(''),
                          grade_family: str = Form(''),
                          stage_type: str = Form(''),
                          dispatchable: str = Form(''),
                          qa_required: str = Form(''),
                          uom: str = Form('KG'),
                          remarks: str = Form(''),
                          dep: None = Depends(require_roles('admin'))):
    code = (grade_code or '').strip().upper()
    if not code:
        return RedirectResponse('/mrp/grades?err=Grade+code+is+required', status_code=303)
    with engine.begin() as conn:
        exists = conn.execute(text('SELECT 1 FROM mrp_grade_master WHERE grade_code=:c LIMIT 1'), {'c': code}).first()
        if exists:
            return RedirectResponse('/mrp/grades?err=Grade+code+already+exists', status_code=303)
        conn.execute(text("""
            INSERT INTO mrp_grade_master(grade_code, grade_name, grade_family, stage_type, dispatchable, qa_required, uom, remarks)
            VALUES (:grade_code, :grade_name, :grade_family, :stage_type, :dispatchable, :qa_required, :uom, :remarks)
        """), {
            'grade_code': code,
            'grade_name': (grade_name or code).strip(),
            'grade_family': (grade_family or '').strip().upper(),
            'stage_type': (stage_type or '').strip().upper(),
            'dispatchable': bool(dispatchable),
            'qa_required': (False if qa_required == 'off' else True),
            'uom': (uom or 'KG').strip().upper(),
            'remarks': (remarks or '').strip(),
        })
    return RedirectResponse('/mrp/grades', status_code=303)


@router.post('/grades/{row_id}/toggle')
async def mrp_grades_toggle(row_id: int, dep: None = Depends(require_roles('admin'))):
    with engine.begin() as conn:
        conn.execute(text("UPDATE mrp_grade_master SET is_active = NOT COALESCE(is_active,TRUE), updated_at=CURRENT_TIMESTAMP WHERE id=:id"), {'id': row_id})
    return RedirectResponse('/mrp/grades', status_code=303)


@router.get('/recipes', response_class=HTMLResponse)
async def mrp_recipes(request: Request, dep: None = Depends(require_roles('admin','sales','view'))):
    rows = _rows("SELECT * FROM mrp_recipe_master ORDER BY COALESCE(is_active,TRUE) DESC, dispatch_grade, source_stage")
    grades = _rows("SELECT grade_code, grade_family, stage_type FROM mrp_grade_master WHERE COALESCE(is_active,TRUE)=TRUE ORDER BY grade_code")
    return templates.TemplateResponse('mrp_recipe_master.html', {
        'request': request,
        'rows': rows,
        'grades': grades,
        'is_admin': _is_admin(request),
        **_tpl_auth(request),
    })


@router.post('/recipes')
async def mrp_recipes_post(request: Request,
                           dispatch_grade: str = Form(...),
                           source_stage: str = Form(...),
                           source_family: str = Form(''),
                           source_grade: str = Form(''),
                           route_steps: str = Form(''),
                           fifo_mode: str = Form('STRICT'),
                           dispatchable_source_type: str = Form('AUTO'),
                           yield_basis_notes: str = Form(''),
                           dep: None = Depends(require_roles('admin'))):
    dg = (dispatch_grade or '').strip()
    stg = (source_stage or '').strip().upper()
    if not dg or not stg:
        return RedirectResponse('/mrp/recipes?err=Dispatch+grade+and+source+stage+are+required', status_code=303)
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO mrp_recipe_master(dispatch_grade, source_stage, source_family, source_grade, route_steps, fifo_mode, dispatchable_source_type, yield_basis_notes)
            VALUES (:dispatch_grade, :source_stage, :source_family, :source_grade, :route_steps, :fifo_mode, :dispatchable_source_type, :yield_basis_notes)
        """), {
            'dispatch_grade': dg,
            'source_stage': stg,
            'source_family': (source_family or '').strip().upper(),
            'source_grade': (source_grade or '').strip().upper(),
            'route_steps': (route_steps or '').strip(),
            'fifo_mode': (fifo_mode or 'STRICT').strip().upper(),
            'dispatchable_source_type': (dispatchable_source_type or 'AUTO').strip().upper(),
            'yield_basis_notes': (yield_basis_notes or '').strip(),
        })
    return RedirectResponse('/mrp/recipes', status_code=303)


@router.post('/recipes/{row_id}/toggle')
async def mrp_recipes_toggle(row_id: int, dep: None = Depends(require_roles('admin'))):
    with engine.begin() as conn:
        conn.execute(text("UPDATE mrp_recipe_master SET is_active = NOT COALESCE(is_active,TRUE), updated_at=CURRENT_TIMESTAMP WHERE id=:id"), {'id': row_id})
    return RedirectResponse('/mrp/recipes', status_code=303)


@router.get('/yields', response_class=HTMLResponse)
async def mrp_yields(request: Request, dep: None = Depends(require_roles('admin','sales','view'))):
    rows = _rows("SELECT * FROM mrp_yield_master ORDER BY COALESCE(is_active,TRUE) DESC, item_family, stage_name")
    return templates.TemplateResponse('mrp_yield_master.html', {
        'request': request,
        'rows': rows,
        'is_admin': _is_admin(request),
        **_tpl_auth(request),
    })


@router.post('/yields')
async def mrp_yields_post(request: Request,
                          item_family: str = Form(...),
                          stage_name: str = Form(...),
                          standard_yield_pct: str = Form(''),
                          recovery_pct: str = Form(''),
                          oversize_target_pct: str = Form(''),
                          capacity_kg_day: str = Form(''),
                          remarks: str = Form(''),
                          dep: None = Depends(require_roles('admin'))):
    fam = (item_family or '').strip().upper()
    stage = (stage_name or '').strip().upper()
    if not fam or not stage:
        return RedirectResponse('/mrp/yields?err=Family+and+Stage+are+required', status_code=303)
    def _num(v):
        try:
            s = (v or '').strip()
            return float(s) if s != '' else None
        except Exception:
            return None
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO mrp_yield_master(item_family, stage_name, standard_yield_pct, recovery_pct, oversize_target_pct, capacity_kg_day, remarks)
            VALUES (:item_family, :stage_name, :standard_yield_pct, :recovery_pct, :oversize_target_pct, :capacity_kg_day, :remarks)
        """), {
            'item_family': fam,
            'stage_name': stage,
            'standard_yield_pct': _num(standard_yield_pct),
            'recovery_pct': _num(recovery_pct),
            'oversize_target_pct': _num(oversize_target_pct),
            'capacity_kg_day': _num(capacity_kg_day),
            'remarks': (remarks or '').strip(),
        })
    return RedirectResponse('/mrp/yields', status_code=303)


@router.post('/yields/{row_id}/toggle')
async def mrp_yields_toggle(row_id: int, dep: None = Depends(require_roles('admin'))):
    with engine.begin() as conn:
        conn.execute(text("UPDATE mrp_yield_master SET is_active = NOT COALESCE(is_active,TRUE) WHERE id=:id"), {'id': row_id})
    return RedirectResponse('/mrp/yields', status_code=303)


@router.get('/run', response_class=HTMLResponse)
async def mrp_run_view(request: Request, run_id: Optional[int] = None, dep: None = Depends(require_roles('admin','sales','view'))):
    if run_id:
        head, lines = _fetch_run(run_id)
    else:
        latest = _rows("SELECT id FROM mrp_run_header ORDER BY id DESC LIMIT 1")
        if latest:
            head, lines = _fetch_run(int(latest[0]['id']))
        else:
            head, lines = None, []
    suggestions: List[Dict[str, Any]] = []
    summary = {'open_lines': 0, 'ready_lines': 0, 'shortage_lines': 0, 'total_pending_kg': 0.0, 'total_shortage_kg': 0.0}
    if lines:
        summary = {
            'open_lines': len(lines),
            'ready_lines': sum(1 for x in lines if str(x.get('atp_status') or '') == 'READY'),
            'shortage_lines': sum(1 for x in lines if float(x.get('shortage_qty_kg') or 0.0) > 0.0001),
            'total_pending_kg': round(sum(float(x.get('pending_qty_kg') or 0.0) for x in lines), 2),
            'total_shortage_kg': round(sum(float(x.get('shortage_qty_kg') or 0.0) for x in lines), 2),
        }
        bucket: Dict[tuple, Dict[str, Any]] = {}
        for ln in lines:
            key = (ln.get('dispatch_grade'), ln.get('source_stage'), ln.get('source_family'), ln.get('source_grade'))
            b = bucket.setdefault(key, {
                'dispatch_grade': ln.get('dispatch_grade'),
                'source_stage': ln.get('source_stage'),
                'source_family': ln.get('source_family'),
                'source_grade': ln.get('source_grade'),
                'shortage_qty_kg': 0.0,
                'suggested_fg_qty_kg': 0.0,
                'suggested_grinding_qty_kg': 0.0,
                'suggested_anneal_qty_kg': 0.0,
                'suggested_rap_qty_kg': 0.0,
                'suggested_atom_qty_kg': 0.0,
                'suggested_melt_qty_kg': 0.0,
                'planned_capacity_days': 0.0,
                'route_steps': ln.get('route_steps') or '',
            })
            for k in ['shortage_qty_kg','suggested_fg_qty_kg','suggested_grinding_qty_kg','suggested_anneal_qty_kg','suggested_rap_qty_kg','suggested_atom_qty_kg','suggested_melt_qty_kg']:
                b[k] += float(ln.get(k) or 0.0)
            b['planned_capacity_days'] += float(ln.get('planned_capacity_days') or 0.0)
        suggestions = sorted(bucket.values(), key=lambda x: (-float(x.get('shortage_qty_kg') or 0.0), str(x.get('dispatch_grade') or '')))
    return templates.TemplateResponse('mrp_run.html', {
        'request': request,
        'run': head,
        'lines': lines,
        'summary': summary,
        'suggestions': suggestions,
        'is_admin': _is_admin(request),
        **_tpl_auth(request),
    })


@router.post('/run')
async def mrp_run_now(request: Request, dep: None = Depends(require_roles('admin','sales'))):
    with engine.begin() as conn:
        lines, _summary, _suggestions = _build_mrp_lines(conn)
        run_id = _store_mrp_run(conn, (getattr(request, 'session', {}) or {}).get('username', '') or 'system', lines)
    return RedirectResponse(f'/mrp/run?run_id={run_id}', status_code=303)



def _group_suggestions_from_lines(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bucket: Dict[tuple, Dict[str, Any]] = {}
    for ln in lines or []:
        key = (ln.get('dispatch_grade'), ln.get('source_stage'), ln.get('source_family'), ln.get('source_grade'))
        b = bucket.setdefault(key, {
            'dispatch_grade': ln.get('dispatch_grade'),
            'source_stage': ln.get('source_stage'),
            'source_family': ln.get('source_family'),
            'source_grade': ln.get('source_grade'),
            'shortage_qty_kg': 0.0,
            'suggested_fg_qty_kg': 0.0,
            'suggested_grinding_qty_kg': 0.0,
            'suggested_anneal_qty_kg': 0.0,
            'suggested_rap_qty_kg': 0.0,
            'suggested_atom_qty_kg': 0.0,
            'suggested_melt_qty_kg': 0.0,
            'planned_capacity_days': 0.0,
            'route_steps': ln.get('route_steps') or '',
            'linked_run_line_count': 0,
        })
        for k in ['shortage_qty_kg','suggested_fg_qty_kg','suggested_grinding_qty_kg','suggested_anneal_qty_kg','suggested_rap_qty_kg','suggested_atom_qty_kg','suggested_melt_qty_kg']:
            b[k] += float(ln.get(k) or 0.0)
        b['planned_capacity_days'] += float(ln.get('planned_capacity_days') or 0.0)
        b['linked_run_line_count'] += 1
    out = list(bucket.values())
    out.sort(key=lambda x: (-float(x.get('shortage_qty_kg') or 0.0), str(x.get('dispatch_grade') or '')))
    return out


def _next_action_stage(s: Dict[str, Any]) -> tuple[str, float]:
    stage_order = [
        ('FG', float(s.get('suggested_fg_qty_kg') or 0.0)),
        ('GRINDING', float(s.get('suggested_grinding_qty_kg') or 0.0)),
        ('ANNEALING', float(s.get('suggested_anneal_qty_kg') or 0.0)),
        ('RAP', float(s.get('suggested_rap_qty_kg') or 0.0)),
        ('ATOMIZATION', float(s.get('suggested_atom_qty_kg') or 0.0)),
        ('MELTING', float(s.get('suggested_melt_qty_kg') or 0.0)),
    ]
    for stg, qty in stage_order:
        if qty > 0.0001:
            return stg, round(qty, 2)
    return str(s.get('source_stage') or 'DISPATCH'), round(float(s.get('shortage_qty_kg') or 0.0), 2)


def _store_planned_orders(conn, source_run_id: int, created_by: str, suggestions: List[Dict[str, Any]]) -> int:
    head = conn.execute(text("""
        INSERT INTO mrp_planned_order_header(plan_label, source_run_id, created_by, status, notes)
        VALUES (:lbl, :run_id, :by, 'OPEN', :notes)
        RETURNING id
    """), {
        'lbl': datetime.now().strftime('PLAN-%Y%m%d-%H%M%S'),
        'run_id': source_run_id,
        'by': created_by,
        'notes': 'Created from MRP Run suggestions'
    }).mappings().first()
    plan_id = int(head['id'])
    rows = []
    for s in suggestions or []:
        if float(s.get('shortage_qty_kg') or 0.0) <= 0.0001:
            continue
        next_stage, next_qty = _next_action_stage(s)
        rows.append({
            'plan_id': plan_id,
            'dispatch_grade': s.get('dispatch_grade'),
            'source_stage': s.get('source_stage'),
            'source_family': s.get('source_family'),
            'source_grade': s.get('source_grade'),
            'shortage_qty_kg': float(s.get('shortage_qty_kg') or 0.0),
            'planned_fg_qty_kg': float(s.get('suggested_fg_qty_kg') or 0.0),
            'planned_grinding_qty_kg': float(s.get('suggested_grinding_qty_kg') or 0.0),
            'planned_anneal_qty_kg': float(s.get('suggested_anneal_qty_kg') or 0.0),
            'planned_rap_qty_kg': float(s.get('suggested_rap_qty_kg') or 0.0),
            'planned_atom_qty_kg': float(s.get('suggested_atom_qty_kg') or 0.0),
            'planned_melt_qty_kg': float(s.get('suggested_melt_qty_kg') or 0.0),
            'planned_capacity_days': (float(s.get('planned_capacity_days') or 0.0) or None),
            'route_steps': s.get('route_steps') or '',
            'next_action_stage': next_stage,
            'next_action_qty_kg': next_qty,
            'linked_run_line_count': int(s.get('linked_run_line_count') or 0),
            'status': 'OPEN',
            'notes': ''
        })
    if rows:
        conn.execute(text("""
            INSERT INTO mrp_planned_order_lines(
                plan_id, dispatch_grade, source_stage, source_family, source_grade,
                shortage_qty_kg, planned_fg_qty_kg, planned_grinding_qty_kg, planned_anneal_qty_kg,
                planned_rap_qty_kg, planned_atom_qty_kg, planned_melt_qty_kg, planned_capacity_days,
                route_steps, next_action_stage, next_action_qty_kg, linked_run_line_count, status, notes
            ) VALUES (
                :plan_id, :dispatch_grade, :source_stage, :source_family, :source_grade,
                :shortage_qty_kg, :planned_fg_qty_kg, :planned_grinding_qty_kg, :planned_anneal_qty_kg,
                :planned_rap_qty_kg, :planned_atom_qty_kg, :planned_melt_qty_kg, :planned_capacity_days,
                :route_steps, :next_action_stage, :next_action_qty_kg, :linked_run_line_count, :status, :notes
            )
        """), rows)
    return plan_id


def _fetch_plan(plan_id: int) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    with engine.begin() as conn:
        head = conn.execute(text("SELECT * FROM mrp_planned_order_header WHERE id=:id"), {'id': plan_id}).mappings().first()
        lines = conn.execute(text("SELECT * FROM mrp_planned_order_lines WHERE plan_id=:id ORDER BY next_action_stage, dispatch_grade, id"), {'id': plan_id}).mappings().all()
    return (dict(head) if head else None, [dict(r) for r in lines])


@router.post('/planned-orders/from-run/{run_id}')
async def mrp_create_plan_from_run(run_id: int, request: Request, dep: None = Depends(require_roles('admin','sales'))):
    head, lines = _fetch_run(run_id)
    if not head:
        return RedirectResponse('/mrp/run?err=MRP+run+not+found', status_code=303)
    suggestions = _group_suggestions_from_lines(lines)
    if not suggestions:
        return RedirectResponse(f'/mrp/run?run_id={run_id}&err=No+shortage+suggestions+available+to+plan', status_code=303)
    with engine.begin() as conn:
        plan_id = _store_planned_orders(conn, run_id, (getattr(request, 'session', {}) or {}).get('username', '') or 'system', suggestions)
    return RedirectResponse(f'/mrp/planned-orders/{plan_id}', status_code=303)


@router.get('/planned-orders', response_class=HTMLResponse)
async def mrp_planned_orders(request: Request, dep: None = Depends(require_roles('admin','sales','view'))):
    rows = _rows("""
        SELECT h.*,
               COALESCE(cnt.line_count,0)::int AS line_count,
               COALESCE(cnt.open_count,0)::int AS open_count,
               COALESCE(cnt.total_shortage_kg,0)::float AS total_shortage_kg,
               COALESCE(cnt.total_next_action_qty_kg,0)::float AS total_next_action_qty_kg
        FROM mrp_planned_order_header h
        LEFT JOIN (
            SELECT plan_id,
                   COUNT(*) AS line_count,
                   SUM(CASE WHEN COALESCE(status,'OPEN') IN ('OPEN','RELEASED') THEN 1 ELSE 0 END) AS open_count,
                   SUM(COALESCE(shortage_qty_kg,0)) AS total_shortage_kg,
                   SUM(COALESCE(next_action_qty_kg,0)) AS total_next_action_qty_kg
            FROM mrp_planned_order_lines
            GROUP BY plan_id
        ) cnt ON cnt.plan_id = h.id
        ORDER BY h.id DESC
    """)
    return templates.TemplateResponse('mrp_planned_orders.html', {
        'request': request,
        'rows': rows,
        'is_admin': _is_admin(request),
        **_tpl_auth(request),
    })


@router.get('/planned-orders/{plan_id}', response_class=HTMLResponse)
async def mrp_planned_order_view(request: Request, plan_id: int, dep: None = Depends(require_roles('admin','sales','view'))):
    head, lines = _fetch_plan(plan_id)
    if not head:
        return RedirectResponse('/mrp/planned-orders?err=Planned+order+not+found', status_code=303)
    summary = {
        'line_count': len(lines),
        'open_count': sum(1 for x in lines if str(x.get('status') or 'OPEN') in ('OPEN','RELEASED')),
        'total_shortage_kg': round(sum(float(x.get('shortage_qty_kg') or 0.0) for x in lines), 2),
        'total_next_action_qty_kg': round(sum(float(x.get('next_action_qty_kg') or 0.0) for x in lines), 2),
    }
    return templates.TemplateResponse('mrp_planned_order_view.html', {
        'request': request,
        'plan': head,
        'lines': lines,
        'summary': summary,
        'is_admin': _is_admin(request),
        **_tpl_auth(request),
    })


@router.post('/planned-orders/{plan_id}/status')
async def mrp_planned_order_status(plan_id: int, status: str = Form(...), dep: None = Depends(require_roles('admin'))):
    st = str(status or '').strip().upper()
    if st not in {'OPEN','RELEASED','CLOSED','CANCELLED'}:
        st = 'OPEN'
    with engine.begin() as conn:
        conn.execute(text("UPDATE mrp_planned_order_header SET status=:st WHERE id=:id"), {'st': st, 'id': plan_id})
        if st in {'CLOSED','CANCELLED'}:
            conn.execute(text("UPDATE mrp_planned_order_lines SET status=:st WHERE plan_id=:id AND COALESCE(status,'OPEN') <> 'CLOSED'"), {'st': st, 'id': plan_id})
        elif st == 'RELEASED':
            conn.execute(text("UPDATE mrp_planned_order_lines SET status='RELEASED' WHERE plan_id=:id AND COALESCE(status,'OPEN')='OPEN'"), {'id': plan_id})
    return RedirectResponse(f'/mrp/planned-orders/{plan_id}', status_code=303)
