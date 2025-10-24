# krn_mrp_app/annealing/routes.py

from datetime import date, timedelta
from fastapi import APIRouter, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from urllib.parse import quote_plus
from starlette.templating import Jinja2Templates
from sqlalchemy import text, bindparam
from sqlalchemy import text
import json, io, csv
from typing import Any, Dict, List, Optional
from fastapi import Depends, HTTPException

from krn_mrp_app.deps import engine, require_roles

router = APIRouter()
templates = Jinja2Templates(directory="templates")

TARGET_KG_PER_DAY = 6000.0
ANNEAL_ADD_COST = 10.0  # ₹/kg add over weighted RAP cost


# ---- Plant-2 balance for Annealing (ONLY lots transferred to Plant 2) ----
def fetch_plant2_balance():
    """
    Returns rows: {lot_no, grade, cost_per_kg, available_kg}

    available_kg = (sum of rap_alloc.qty where kind='PLANT2')
                   - (qty already consumed in anneal_lots.src_alloc_json)
    Cost/kg is taken from lot.unit_cost.
    """
    # 1) Plant-2 transfers per RAP lot (from rap_alloc)
    with engine.begin() as conn:
        plant2 = conn.execute(text("""
            SELECT
                rl.id                  AS rap_lot_id,
                CASE
                  WHEN l.lot_no IS NULL OR l.lot_no = '' THEN 'LOT-' || l.id
                  ELSE l.lot_no
                END                     AS lot_no,
                COALESCE(l.grade,'')    AS grade,
                COALESCE(l.unit_cost,0) AS cost_per_kg,
                SUM(a.qty)::float       AS plant2_qty
            FROM public.rap_alloc  AS a
            JOIN public.rap_lot    AS rl ON rl.id = a.rap_lot_id
            JOIN public.lot        AS l  ON l.id  = rl.lot_id
            WHERE a.kind = 'PLANT2'
            GROUP BY rl.id, l.id, l.lot_no, l.grade, l.unit_cost
            ORDER BY rl.id
        """)).mappings().all()

        # 2) Usage already recorded by Anneal (from src_alloc_json)
        used_by_lotno: dict[str, float] = {}
        for (alloc_json,) in conn.execute(text("SELECT src_alloc_json FROM anneal_lots")).all():
            if not alloc_json:
                continue
            try:
                d = json.loads(alloc_json)
                for lot_no, qty in d.items():
                    used_by_lotno[lot_no] = used_by_lotno.get(lot_no, 0.0) + float(qty or 0)
            except Exception:
                # ignore bad JSON but keep going
                pass

    # 3) Remaining availability
    out = []
    for p in plant2:
        lot_no = p["lot_no"]
        transferred = float(p["plant2_qty"] or 0.0)
        used = float(used_by_lotno.get(lot_no, 0.0))
        avail = transferred - used
        if avail > 0.0001:
            out.append({
                "lot_no":       lot_no,
                "grade":        p["grade"],
                "available_kg": avail,
                "cost_per_kg":  float(p["cost_per_kg"] or 0.0),
            })
    return out

def plant2_available_rows(conn):
    """
    Returns rows: lot_no, grade, cost_per_kg, available_kg
    where available_kg = sum(rap_alloc.kind='PLANT2') - qty already used in anneal_lots.
    """
    sql = text("""
        WITH plant2 AS (
            SELECT rl.id AS rap_lot_id,
                   l.lot_no,
                   COALESCE(l.grade,'')      AS grade,
                   COALESCE(l.unit_cost, 0)  AS cost_per_kg,
                   SUM(CASE WHEN a.kind='PLANT2' THEN a.qty ELSE 0 END) AS plant2_qty
            FROM rap_alloc a
            JOIN rap_lot rl ON rl.id = a.rap_lot_id
            JOIN lot     l  ON l.id  = rl.lot_id
            GROUP BY rl.id, l.lot_no, l.grade, l.unit_cost
        ),
        used AS (
            SELECT key AS lot_no,
                   SUM( (src_alloc_json::jsonb ->> key)::numeric ) AS used_qty
            FROM anneal_lots,
                 LATERAL jsonb_object_keys(src_alloc_json::jsonb) AS key
            GROUP BY key
        )
        SELECT p.lot_no,
               p.grade,
               p.cost_per_kg,
               COALESCE(p.plant2_qty,0) - COALESCE(u.used_qty,0) AS available_kg
        FROM plant2 p
        LEFT JOIN used u ON u.lot_no = p.lot_no
        WHERE COALESCE(p.plant2_qty,0) - COALESCE(u.used_qty,0) > 0
        ORDER BY p.lot_no;
    """)
    return conn.execute(sql).mappings().all()

# --- role helpers (shared with other modules) ---
def current_role(request):
    """Return current user's role, default 'guest'."""
    if hasattr(request.state, "role") and request.state.role:
        return request.state.role
    return "guest"


def is_read_only(request):
    """Viewer/guest cannot modify data."""
    role = current_role(request)
    return role not in ("admin", "anneal")

# --- helper: robust admin check ---
def _is_admin(request: Request) -> bool:
    s = getattr(request, "state", None)
    if not s:
        return False
    if getattr(s, "is_admin", False):
        return True
    role = getattr(s, "role", None)
    if isinstance(role, str) and role.lower() == "admin":
        return True
    roles = getattr(s, "roles", None)
    if isinstance(roles, (list, set, tuple)) and "admin" in roles:
        return True
    return False

# ------------------ ROUTES ------------------

@router.get("/", response_class=HTMLResponse)
async def anneal_home(request: Request, dep: None = Depends(require_roles("admin","anneal","view"))):
    today = date.today()
    first_of_month = today.replace(day=1)
    yday = today - timedelta(days=1)

    with engine.begin() as conn:
        # headline KPIs
        lots_today = conn.execute(
            text("SELECT COUNT(*) FROM anneal_lots WHERE date=:d"), {"d": today}
        ).scalar() or 0

        nh3_today = conn.execute(
            text("SELECT COALESCE(SUM(ammonia_kg),0) FROM anneal_lots WHERE date=:d"), {"d": today}
        ).scalar() or 0.0

        produced_today = conn.execute(
            text("SELECT COALESCE(SUM(weight_kg),0) FROM anneal_lots WHERE date=:d"), {"d": today}
        ).scalar() or 0.0

        # --- NEW: adjust daily target by today's downtime (minutes) ---
        down_mins_today = conn.execute(
            text("SELECT COALESCE(SUM(minutes),0) FROM anneal_downtime WHERE date=:d"), {"d": today}
        ).scalar() or 0
        minutes_avail = max(1440 - int(down_mins_today), 0)
        target_today = TARGET_KG_PER_DAY * (minutes_avail / 1440.0)

        # efficiency uses the adjusted target
        eff_today = (produced_today / target_today * 100.0) if target_today > 0 else 0.0

        avg_cost_today = conn.execute(
            text("SELECT COALESCE(AVG(cost_per_kg),0) FROM anneal_lots WHERE date=:d"), {"d": today}
        ).scalar() or 0.0

        weighted_cost_today = conn.execute(
            text("""
                SELECT COALESCE(SUM(cost_per_kg * weight_kg)/NULLIF(SUM(weight_kg),0),0)
                FROM anneal_lots WHERE date=:d
            """),
            {"d": today}
        ).scalar() or 0.0

        # last 5 days production (most recent first)
        last5 = conn.execute(
            text("""
                SELECT date, COALESCE(SUM(weight_kg),0) AS qty
                FROM anneal_lots
                WHERE date >= :d5
                GROUP BY date
                ORDER BY date DESC
            """),
            {"d5": today - timedelta(days=4)}
        ).mappings().all()

        # live stock by grade (simple: sum of lot weights)
        live_stock = conn.execute(
            text("""
                SELECT grade, COALESCE(SUM(weight_kg),0) AS qty
                FROM anneal_lots
                GROUP BY grade
                ORDER BY grade
            """)
        ).mappings().all()

        # ammonia gas – yesterday and month-to-date
        nh3_yday = conn.execute(
            text("SELECT COALESCE(SUM(ammonia_kg),0) FROM anneal_lots WHERE date=:d"),
            {"d": yday}
        ).scalar() or 0.0

        nh3_mtd = conn.execute(
            text("SELECT COALESCE(SUM(ammonia_kg),0) FROM anneal_lots WHERE date >= :d"),
            {"d": first_of_month}
        ).scalar() or 0.0

    is_admin = (
        getattr(request.state, "is_admin", False)
        or getattr(request.state, "role", "").lower() == "admin"
        or ("admin" in (getattr(request.state, "roles", []) or []))
    )

    return templates.TemplateResponse("annealing_home.html", {
        "request": request,
        "role": current_role(request),          # <<< add this
        "read_only": is_read_only(request),     # (optional but recommended)
        "target": target_today,                 # <— adjusted for downtime
        "lots_today": lots_today,
        "nh3_today": nh3_today,
        "avg_cost_today": avg_cost_today,
        "weighted_cost_today": weighted_cost_today,
        "produced_today": produced_today,
        "eff_today": eff_today,                 # <— uses adjusted target
        "last5": last5,
        "live_stock": live_stock,
        "nh3_yday": nh3_yday,
        "nh3_mtd": nh3_mtd,
        "is_admin": is_admin,
    })

@router.get("/create", response_class=HTMLResponse)
async def anneal_create_get(
    request: Request,
    dep: None = Depends(require_roles("anneal","admin"))
):
    rap_rows = fetch_plant2_balance()
    err = request.query_params.get("err", "")

    return templates.TemplateResponse(
        "annealing_create.html",
        {
            "request": request,
            "role": current_role(request),          # <<< add this
            "read_only": is_read_only(request),     # (optional but recommended)"rap_rows": rap_rows,
            "err": err,
            "is_admin": _is_admin(request),   # <-- KEY: pass reliable flag
        },
    )


@router.post("/create")
async def anneal_create_post(
    request: Request,
    dep: None = Depends(require_roles("anneal", "admin")),
):
    form = await request.form()

    # ---- collect allocations from the form ----
    allocations: dict[str, float] = {}
    total_alloc = 0.0
    for k, v in form.items():
        if k.startswith("alloc_"):
            rap_lot_no = k.replace("alloc_", "")
            try:
                qty = float(v or 0)
            except Exception:
                qty = 0.0
            if qty > 0:
                allocations[rap_lot_no] = qty
                total_alloc += qty

    # ----- basic guard: must allocate something -----
    if total_alloc <= 0:
        msg = "Allocate quantity > 0 kg."
        return RedirectResponse(url=f"/anneal/create?err={quote_plus(msg)}", status_code=303)

    # ---- Lot Weight must be entered and must equal total allocation (±0.01 kg) ----
    lw_raw = form.get("lot_weight")
    try:
        lot_weight = float(lw_raw)
    except Exception:
        return RedirectResponse(url="/anneal/create?err=Lot%20Weight%20must%20be%20a%20number.", status_code=303)

    if lot_weight <= 0:
        return RedirectResponse(url="/anneal/create?err=Lot%20Weight%20must%20be%20%3E%200.", status_code=303)

    if abs(lot_weight - total_alloc) > 0.01:
        msg = f"Lot Weight mismatch: allocated {total_alloc:.2f} kg, entered {lot_weight:.2f} kg."
        return RedirectResponse(url=f"/anneal/create?err={quote_plus(msg)}", status_code=303)

    # ---- Ammonia must be at least 0.025 × lot weight ----
    try:
        ammonia_kg = float(form.get("ammonia_kg") or 0)
    except Exception:
        ammonia_kg = 0.0

    nh3_min = 0.025 * lot_weight
    if ammonia_kg < nh3_min:
        msg = f"Ammonia must be at least {nh3_min:.3f} kg for lot weight {lot_weight:.2f} kg."
        return RedirectResponse(url=f"/anneal/create?err={quote_plus(msg)}", status_code=303)

    # ---- availability & cost: use the same Plant-2 balance you show on GET ----
    #     (so POST validates against exactly what the page displayed)
    avail_rows = fetch_plant2_balance()       # returns list of mappings: lot_no, grade, cost_per_kg, available_kg
    avail_map = {r["lot_no"]: r for r in avail_rows}

    # ensure every selected lot exists and has enough Plant-2 balance
    for lot_no, qty in allocations.items():
        r = avail_map.get(lot_no)
        if (r is None) or (qty > float(r.get("available_kg") or 0)):
            msg = f"Selected RAP lot {lot_no} not found or insufficient Plant-2 balance."
            return RedirectResponse(url=f"/anneal/create?err={quote_plus(msg)}", status_code=303)

    # ---- single family rule (KRIP or KRFS only) ----
    fam = {avail_map[lot]["grade"] for lot in allocations.keys()}
    if len(fam) > 1:
        msg = "Only one grade allowed per anneal lot (KRIP or KRFS)."
        return RedirectResponse(url=f"/anneal/create?err={quote_plus(msg)}", status_code=303)

    rap_grade = next(iter(fam))
    out_grade = "KIP" if rap_grade == "KRIP" else "KFS"

    # ---- weighted RAP cost (from unit_cost surfaced by helper as cost_per_kg) ----
    rap_cost_wsum = sum(
        allocations[lot_no] * float(avail_map[lot_no].get("cost_per_kg") or 0)
        for lot_no in allocations.keys()
    )
    rap_cost_per_kg = rap_cost_wsum / lot_weight if lot_weight else 0.0
    cost_per_kg = rap_cost_per_kg + ANNEAL_ADD_COST  # rule: RAP + ₹10

    # ---- create ANL lot number and insert ----
    with engine.begin() as conn:
        prefix = "ANL-" + date.today().strftime("%Y%m%d") + "-"
        last = conn.execute(
            text("""
                SELECT lot_no
                  FROM anneal_lots
                 WHERE lot_no LIKE :pfx
                 ORDER BY lot_no DESC
                 LIMIT 1
            """),
            {"pfx": f"{prefix}%"},
        ).scalar()
        seq = int(last.split("-")[-1]) + 1 if last else 1
        lot_no = f"{prefix}{seq:03d}"

        conn.execute(
            text("""
                INSERT INTO anneal_lots
                    (date, lot_no, src_alloc_json, grade, weight_kg,
                     rap_cost_per_kg, cost_per_kg, ammonia_kg, qa_status)
                VALUES
                    (:date, :lot_no, :src_alloc_json, :grade, :weight_kg,
                     :rap_cost_per_kg, :cost_per_kg, :ammonia_kg, 'PENDING')
            """),
            {
                "date": date.today(),
                "lot_no": lot_no,
                "src_alloc_json": json.dumps(allocations),
                "grade": out_grade,
                "weight_kg": lot_weight,
                "rap_cost_per_kg": rap_cost_per_kg,
                "cost_per_kg": cost_per_kg,
                "ammonia_kg": ammonia_kg,
            },
        )

        # NOTE: no UPDATE to rap tables — Plant-2 availability is derived (PLANT2 allocations minus anneal usage)

    return RedirectResponse(url="/anneal/lots", status_code=303)


@router.get("/lots", response_class=HTMLResponse)
async def anneal_lots(
    request: Request,
    dep: None = Depends(require_roles("admin", "anneal", "view")),
    from_date: str = None,
    to_date: str = None,
    csv: int = 0
):
    # make sure this variable name is exactly 'query'
    query = """
        SELECT id, date, lot_no, grade, weight_kg, ammonia_kg,
               rap_cost_per_kg, cost_per_kg, qa_status
        FROM anneal_lots
        WHERE 1=1
    """
    params = {}
    if from_date:
        query += " AND date >= :from_date"
        params["from_date"] = from_date
    if to_date:
        query += " AND date <= :to_date"
        params["to_date"] = to_date
    query += " ORDER BY date DESC, id DESC"

    with engine.begin() as conn:
        rows = conn.execute(text(query), params).mappings().all()

    # augment rows with anneal_cost_per_kg + total_value (no SQL changes)
    augmented = []
    for r in rows:
        eff_cost = r["cost_per_kg"]
        if eff_cost is None or eff_cost == 0:
            eff_cost = (r["rap_cost_per_kg"] or 0) + ANNEAL_ADD_COST
        tot_val = eff_cost * (r["weight_kg"] or 0)

        d = dict(r)
        d["anneal_cost_per_kg"] = eff_cost
        d["total_value"] = tot_val
        augmented.append(d)

    visible_rows = [r for r in augmented if (r["weight_kg"] or 0) > 0]
    total_weight = sum((r["weight_kg"] or 0) for r in visible_rows)
    weighted_cost = (
        sum((r["anneal_cost_per_kg"] or 0) * (r["weight_kg"] or 0) for r in visible_rows) / total_weight
        if total_weight > 0 else 0.0
    )

    is_admin = (
        getattr(request.state, "is_admin", False)
        or getattr(request.state, "role", "").lower() == "admin"
        or ("admin" in (getattr(request.state, "roles", []) or []))
    )

    if csv:
        buf = io.StringIO()
        writer = csv.writer(buf)
        # CSV kept identical to your current version
        writer.writerow(["Date","Lot","Grade","Weight (kg)","Ammonia (kg)","RAP Cost/kg","Anneal Cost/kg","QA"])
        for r in rows:
            writer.writerow([
                r["date"], r["lot_no"], r["grade"],
                "%.0f" % (r["weight_kg"] or 0),
                "%.2f" % (r["ammonia_kg"] or 0),
                "%.2f" % (r["rap_cost_per_kg"] or 0),
                "%.2f" % ((r["cost_per_kg"] if r["cost_per_kg"] not in (None, 0) else (r["rap_cost_per_kg"] or 0) + ANNEAL_ADD_COST) or 0),
                r["qa_status"] or ""
            ])
        return Response(
            buf.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition":"attachment; filename=anneal_lots.csv"}
        )

    return templates.TemplateResponse("annealing_lot_list.html", {
        "request": request,
        "role": current_role(request),          # <<< add this
        "read_only": is_read_only(request),     # (optional but recommended)
        "lots": visible_rows,              # each row now has anneal_cost_per_kg & total_value
        "total_weight": total_weight,
        "weighted_cost": weighted_cost,    # uses effective anneal cost
        "from_date": from_date,
        "to_date": to_date,
        "today": date.today().isoformat(),
        "is_admin": is_admin,
    })


from datetime import date, timedelta, datetime  # make sure datetime is imported

@router.get("/downtime", response_class=HTMLResponse)
async def anneal_downtime_get(request: Request, dep: None = Depends(require_roles("anneal","admin"))):
    with engine.begin() as conn:
        logs = conn.execute(text("SELECT * FROM anneal_downtime ORDER BY date DESC, id DESC")).mappings().all()

    today = date.today()
    three_days_ago = today - timedelta(days=3)

    return templates.TemplateResponse("annealing_downtime.html", {
        "request": request,
        "role": current_role(request),          # <<< add this
        "read_only": is_read_only(request),     # (optional but recommended)
        "logs": logs,
        # used by the form to restrict picker
        "today": today.isoformat(),
        "min_date": three_days_ago.isoformat(),
        # optional error banner
        "err": request.query_params.get("err", "")
    })


@router.post("/downtime")
async def anneal_downtime_post(request: Request, dep: None = Depends(require_roles("anneal","admin"))):
    form = await request.form()

    # --- validate date: no future, only last 3 days allowed ---
    d_str = (form.get("date") or "").strip()
    try:
        d_obj = datetime.strptime(d_str, "%Y-%m-%d").date()
    except Exception:
        return RedirectResponse(url="/anneal/downtime?err=Invalid+date", status_code=303)

    today = date.today()
    if d_obj > today or d_obj < (today - timedelta(days=3)):
        return RedirectResponse(url="/anneal/downtime?err=Date+must+be+within+last+3+days", status_code=303)

    # minutes + dropdown "type" (stored in existing 'area' column)
    mins = int(form.get("minutes") or 0)
    area_or_type = (form.get("area") or "").strip()   # values: Production/Maintenance/Power/Other
    reason = (form.get("reason") or "").strip()

    if mins <= 0 or not area_or_type:
        return RedirectResponse(url="/anneal/downtime?err=Minutes+and+Type+are+required", status_code=303)

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO anneal_downtime (date, minutes, area, reason)
            VALUES (:date, :minutes, :area, :reason)
        """), {
            "date": d_obj,
            "minutes": mins,
            "area": area_or_type,
            "reason": reason
        })

    return RedirectResponse(url="/anneal/downtime", status_code=303)

@router.get("/downtime.csv")
async def anneal_downtime_csv(dep: None = Depends(require_roles("anneal","admin","view"))):
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT date, minutes, area, reason FROM anneal_downtime ORDER BY date DESC")).all()
    out = io.StringIO(); w = csv.writer(out)
    w.writerow(["date","minutes","area","reason"])
    for r in rows: w.writerow(r)
    data = io.BytesIO(out.getvalue().encode("utf-8")); data.seek(0)
    return HTMLResponse(
        content=out.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=anneal_downtime.csv"}
    )

# --- annealing trace/pdf (with QA params) ---
# Assumes: router, engine, templates, require_roles are already defined/imported in this module.

# ========== Small safe helpers ==========

def _safe_query(conn, sql: str, params: dict) -> list[dict]:
    """
    Run a 'probe' query inside a SAVEPOINT so failures don't abort the outer txn.
    Returns [] on any error.
    """
    try:
        with conn.begin_nested():  # SAVEPOINT
            rows = conn.execute(text(sql), params).mappings().all()
            return [dict(r) for r in rows]
    except Exception:
        return []

def _get_columns(conn, table_name: str) -> set[str]:
    """Safely fetch lowercase column names for a table."""
    try:
        with conn.begin_nested():
            rows = conn.execute(text("""
                SELECT lower(column_name)
                FROM information_schema.columns
                WHERE table_schema='public' AND table_name=:t
            """), {"t": table_name}).scalars().all()
            return set(rows or [])
    except Exception:
        return set()

def _table_exists(conn, table_name: str) -> bool:
    """Safely check if a table exists."""
    try:
        with conn.begin_nested():
            exists = conn.execute(
                text("SELECT to_regclass('public.'||:t) IS NOT NULL"),
                {"t": table_name}
            ).scalar()
            return bool(exists)
    except Exception:
        return False

# ========== Data fetchers ==========

def _fetch_anneal_header(conn, lot_id: int) -> Dict[str, Any] | None:
    row = conn.execute(text("""
        SELECT id, date, lot_no, grade, weight_kg, ammonia_kg,
               rap_cost_per_kg, cost_per_kg, src_alloc_json, qa_status
        FROM anneal_lots WHERE id = :id
    """), {"id": lot_id}).mappings().first()
    return dict(row) if row else None


def _fetch_rap_rows_for_alloc(conn, alloc_map: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    alloc_map: {"KRIP-20250919-003": 10.0, ...}

    Returns one row per RAP lot_no that exists in base table lot
    (reachable from rap_lot → lot). Includes base lot id/grade/cost
    so we can link to Traceability → Lot page (base lot id).
    """
    if not alloc_map:
        return []

    lot_nos = list(alloc_map.keys())

    rows = conn.execute(text("""
        SELECT
            rl.id                   AS rap_lot_id,
            l.id                    AS base_lot_id,
            l.lot_no                AS rap_lot_no,
            COALESCE(l.grade,'')    AS rap_grade,
            COALESCE(l.unit_cost,0) AS rap_cost_per_kg
        FROM rap_lot rl
        JOIN lot     l  ON l.id = rl.lot_id
        WHERE l.lot_no = ANY(:lot_nos)
        ORDER BY l.lot_no
    """), {"lot_nos": lot_nos}).mappings().all()

    out: List[Dict[str, Any]] = []
    for r in rows:
        ln = r["rap_lot_no"]
        out.append({
            "rap_lot_no": ln,
            "rap_lot_id": r["rap_lot_id"],
            "base_lot_id": r["base_lot_id"],
            "rap_grade": r["rap_grade"],
            "rap_cost_per_kg": float(r["rap_cost_per_kg"] or 0.0),
            "allocated_kg": float(alloc_map.get(ln, 0.0)),
        })

    # include any alloc entries that didn't match a base lot (avoid silently dropping)
    known = {r["rap_lot_no"] for r in out}
    for ln, q in alloc_map.items():
        if ln not in known:
            out.append({
                "rap_lot_no": ln,
                "rap_lot_id": None,
                "base_lot_id": None,
                "rap_grade": "",
                "rap_cost_per_kg": 0.0,
                "allocated_kg": float(q or 0.0),
            })
    return out


# (kept; used by some earlier iterations; harmless even if unused)
def _pick_qty_col(conn, table_name: str, candidates: list[str]) -> str | None:
    cols = _get_columns(conn, table_name)
    for c in candidates:
        if c.lower() in cols:
            return c
    return None


def _fetch_heats_for_base_lot(conn, base_lot_id: int) -> List[Dict[str, Any]]:
    """
    Return [{heat_id, heat_no, used_qty}] for a base RAP lot (lot.id).

    We prefer quantities from lot_heat (alloc_kg/qty). If nothing exists for
    this lot in lot_heat, we fall back to lot_heats (mapping only; qty = 0).
    Supports both 'heats' (plural) and 'heat' (singular).
    """
    # 1) Prefer quantity rows from lot_heat → join to heats (plural)
    rows = conn.execute(text("""
        SELECT
          h.id                                    AS heat_id,
          h.heat_no                               AS heat_no,
          COALESCE(lh.alloc_kg, lh.qty, 0)::float AS used_qty
        FROM lot_heat lh
        JOIN heats h ON h.id = lh.heat_id
        WHERE lh.lot_id = :lid
        ORDER BY h.id
    """), {"lid": base_lot_id}).mappings().all()
    if rows:
        return [dict(r) for r in rows]

    # 1b) If your DB uses 'heat' (singular) instead of 'heats'
    rows = conn.execute(text("""
        SELECT
          h.id                                    AS heat_id,
          h.heat_no                               AS heat_no,
          COALESCE(lh.alloc_kg, lh.qty, 0)::float AS used_qty
        FROM lot_heat lh
        JOIN heat h ON h.id = lh.heat_id
        WHERE lh.lot_id = :lid
        ORDER BY h.id
    """), {"lid": base_lot_id}).mappings().all()
    if rows:
        return [dict(r) for r in rows]

    # 2) Fallback: mapping only (no qty in lot_heats → show 0.00)
    rows = conn.execute(text("""
        SELECT
          h.id       AS heat_id,
          h.heat_no  AS heat_no,
          0.0::float AS used_qty
        FROM lot_heats m
        JOIN heats h ON h.id = m.heat_id
        WHERE m.lot_id = :lid
        ORDER BY h.id
    """), {"lid": base_lot_id}).mappings().all()
    if rows:
        return [dict(r) for r in rows]

    # 2b) Fallback with 'heat' (singular)
    rows = conn.execute(text("""
        SELECT
          h.id       AS heat_id,
          h.heat_no  AS heat_no,
          0.0::float AS used_qty
        FROM lot_heats m
        JOIN heat h ON h.id = m.heat_id
        WHERE m.lot_id = :lid
        ORDER BY h.id
    """), {"lid": base_lot_id}).mappings().all()

    return [dict(r) for r in rows]


def _fetch_grns_for_heat(conn, heat_id: int) -> List[Dict[str, Any]]:
    """
    Return [{grn_no, qty_kg}] for the heat.
    Try heat_inputs first, then heat_rm.
    Joins to either grns (plural) or grn (singular), depending on what exists.
    All probes are safe; if a path fails it falls back to the next.
    """
    # ---- heat_inputs path ----
    if _table_exists(conn, "heat_inputs"):
        hi_cols = _get_columns(conn, "heat_inputs")
        hi_fk   = next((c for c in ("grn_id", "grn", "grns_id") if c in hi_cols), None)
        hi_qty  = "qty_kg" if "qty_kg" in hi_cols else ("qty" if "qty" in hi_cols else None)
        if hi_fk and hi_qty:
            for tgt in ("grns", "grn"):  # prefer plural
                if _table_exists(conn, tgt):
                    rows = _safe_query(conn, f"""
                        SELECT COALESCE(g.grn_no, 'GRN-'||g.id) AS grn_no,
                               COALESCE(hi.{hi_qty}, 0)::float   AS qty_kg
                        FROM heat_inputs hi
                        LEFT JOIN {tgt} g ON g.id = hi.{hi_fk}
                        WHERE hi.heat_id = :hid
                        ORDER BY g.id
                    """, {"hid": heat_id})
                    if rows:
                        return rows

    # ---- heat_rm fallback ----
    if _table_exists(conn, "heat_rm"):
        hr_cols = _get_columns(conn, "heat_rm")
        hr_fk   = next((c for c in ("grn_id", "grn") if c in hr_cols), None)
        hr_qty  = "qty" if "qty" in hr_cols else ("qty_kg" if "qty_kg" in hr_cols else None)
        if hr_fk and hr_qty:
            for tgt in ("grn", "grns"):  # most schemas use singular; try both
                if _table_exists(conn, tgt):
                    rows = _safe_query(conn, f"""
                        SELECT COALESCE(g.grn_no, 'GRN-'||g.id) AS grn_no,
                               COALESCE(hr.{hr_qty}, 0)::float   AS qty_kg
                        FROM heat_rm hr
                        LEFT JOIN {tgt} g ON g.id = hr.{hr_fk}
                        WHERE hr.heat_id = :hid
                        ORDER BY g.id
                    """, {"hid": heat_id})
                    if rows:
                        return rows

    # Nothing linked
    return []


def _fetch_latest_anneal_qa_full(conn, anneal_lot_id: int) -> Dict[str, Any] | None:
    """
    Returns latest QA header + all parameter rows for the given anneal lot, if present.
    Uses your actual columns: decision, oxygen, remarks, lot_id.
    We alias decision -> status so templates can read qa.header.status.
    """
    qa_row = conn.execute(text("""
        SELECT
            id,
            anneal_lot_id,
            decision       AS status,
            oxygen,
            lot_id,
            COALESCE(remarks, '') AS remarks
        FROM anneal_qa
        WHERE anneal_lot_id = :lid
        ORDER BY id DESC
        LIMIT 1
    """), {"lid": anneal_lot_id}).mappings().first()

    if not qa_row:
        return None

    # Optional: parameters table (keep try/except in case it doesn't exist)
    try:
        param_rows = conn.execute(text("""
            SELECT
                param_name  AS name,
                param_value AS value,
                unit,
                spec_min,
                spec_max
            FROM anneal_qa_params
            WHERE anneal_qa_id = :qid
            ORDER BY id
        """), {"qid": qa_row["id"]}).mappings().all()
        params = [dict(r) for r in param_rows]
    except Exception:
        params = []

    return {
        "header": dict(qa_row),
        "params": params,
    }


@router.get("/trace/{anneal_id}", response_class=HTMLResponse)
async def anneal_trace_view(
    request: Request,
    anneal_id: int,
    dep: None = Depends(require_roles("admin", "anneal", "view")),
):
    with engine.begin() as conn:
        header = _fetch_anneal_header(conn, anneal_id)
        if not header:
            raise HTTPException(status_code=404, detail="Anneal lot not found")

        try:
            alloc_map = json.loads(header.get("src_alloc_json") or "{}")
        except Exception:
            alloc_map = {}

        rap_rows = _fetch_rap_rows_for_alloc(conn, alloc_map)

        # enrich each RAP base lot with upstream heats and GRNs
        for r in rap_rows:
            r["heats"] = []
            if r.get("base_lot_id"):
                heats = _fetch_heats_for_base_lot(conn, r["base_lot_id"])
                for h in heats:
                    h["grns"] = _fetch_grns_for_heat(conn, h["heat_id"])
                r["heats"] = heats

        qa = _fetch_latest_anneal_qa_full(conn, anneal_id)

    return templates.TemplateResponse(
        "anneal_trace.html",
        {
            "request": request,
            "header": header,     # anneal lot header
            "rap_rows": rap_rows, # RAP → heats → GRNs
            "qa": qa,             # latest QA + params (optional)
        },
    )


@router.get("/pdf/{anneal_id}", response_class=HTMLResponse)
async def anneal_pdf_view(
    request: Request,
    anneal_id: int,
    dep: None = Depends(require_roles("admin", "anneal", "view")),
):
    with engine.begin() as conn:
        header = _fetch_anneal_header(conn, anneal_id)
        if not header:
            raise HTTPException(status_code=404, detail="Anneal lot not found")

        try:
            alloc_map = json.loads(header.get("src_alloc_json") or "{}")
        except Exception:
            alloc_map = {}

        rap_rows = _fetch_rap_rows_for_alloc(conn, alloc_map)
        for r in rap_rows:
            r["heats"] = []
            if r.get("base_lot_id"):
                heats = _fetch_heats_for_base_lot(conn, r["base_lot_id"])
                for h in heats:
                    h["grns"] = _fetch_grns_for_heat(conn, h["heat_id"])
                r["heats"] = heats

        qa = _fetch_latest_anneal_qa_full(conn, anneal_id)

    # Print-friendly page; user can print to PDF or you can pipe to your PDF generator
    return templates.TemplateResponse(
        "anneal_pdf.html",
        {
            "request": request,
            "header": header,
            "rap_rows": rap_rows,
            "qa": qa,
        },
    )

# ---------- Anneal QA: helpers (prefill from RAP allocations) ----------
from typing import Any, Dict, List, Tuple
from sqlalchemy import text
from fastapi import Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
import json

# assumes: engine, templates, require_roles, router already imported in this module

CHEM_KEYS = ["c","si","s","p","cu","ni","mn","fe"]
PHYS_KEYS = ["ad","flow"]
PSD_KEYS  = ["p212","p180","n180p150","n150p75","n75p45","n45"]

def _parse_alloc_map(header: Dict[str, Any]) -> Dict[str, float]:
    try:
        raw = header.get("src_alloc_json") or "{}"
        m = json.loads(raw)
        out = {}
        for k, v in (m.items() if isinstance(m, dict) else []):
            try:
                out[str(k)] = float(v or 0)
            except Exception:
                out[str(k)] = 0.0
        return out
    except Exception:
        return {}

def _fetch_lot_blocks_for_alloc(conn, alloc_map: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    For lot_nos in alloc_map, pull lot + chemistry + phys + psd.
    Assumes tables: lot, lot_chem, lot_phys, lot_psd
    """
    if not alloc_map:
        return []
    lot_nos = list(alloc_map.keys())
    rows = conn.execute(text("""
        SELECT
            l.id, l.lot_no,
            lc.c, lc.si, lc.s, lc.p, lc.cu, lc.ni, lc.mn, lc.fe,
            lp.ad, lp.flow,
            psd.p212, psd.p180, psd.n180p150, psd.n150p75, psd.n75p45, psd.n45
        FROM lot l
        LEFT JOIN lot_chem lc ON lc.lot_id = l.id
        LEFT JOIN lot_phys lp ON lp.lot_id = l.id
        LEFT JOIN lot_psd  psd ON psd.lot_id = l.id
        WHERE l.lot_no = ANY(:lot_nos)
        ORDER BY l.id
    """), {"lot_nos": lot_nos}).mappings().all()
    out = []
    for r in rows:
        d = dict(r)
        d["alloc_kg"] = float(alloc_map.get(d["lot_no"], 0.0))
        out.append(d)
    return out

def _dominant_lot(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Any] | None, float]:
    tot = sum(r["alloc_kg"] for r in rows)
    if tot <= 0:
        return None, 0.0
    best = max(rows, key=lambda r: r["alloc_kg"])
    share = best["alloc_kg"] / tot
    return best, share

def _weighted_value(rows: List[Dict[str, Any]], key: str) -> float | None:
    tot_w = sum(r["alloc_kg"] for r in rows)
    if tot_w <= 0:
        return None
    num = 0.0
    den = 0.0
    for r in rows:
        try:
            v = r.get(key)
            if v is None or v == "":
                continue
            v = float(v)
            w = float(r["alloc_kg"])
            num += v * w
            den += w
        except Exception:
            continue
    if den <= 0:
        return None
    return num / den

def _anneal_default_blocks(conn, header: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    Build default CHEM/PHYS/PSD for Anneal QA from upstream RAP allocations:
      - If a single RAP lot >=60% of total AND has full chemistry -> use it.
      - Else compute allocation-weighted averages for each field.
    Returns strings formatted for form inputs ("" or 4 decimals).
    """
    alloc_map = _parse_alloc_map(header)
    rows = _fetch_lot_blocks_for_alloc(conn, alloc_map)
    if not rows:
        return {
            "chem": {k:"" for k in [k.capitalize() for k in CHEM_KEYS]},
            "phys": {"ad":"", "flow":""},
            "psd":  {k:"" for k in ["p212","p180","n180p150","n150p75","n75p45","n45"]},
        }

    dom, share = _dominant_lot(rows)

    def fmt4(x: float | None) -> str:
        return "" if x is None else f"{x:.4f}"

    def from_lot(r: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        chem = {k.capitalize(): fmt4(float(r.get(k)) if r.get(k) not in ("", None) else None) for k in CHEM_KEYS}
        phys = {k: fmt4(float(r.get(k)) if r.get(k) not in ("", None) else None) for k in PHYS_KEYS}
        psd  = {k: fmt4(float(r.get(k)) if r.get(k) not in ("", None) else None) for k in PSD_KEYS}
        return {"chem": chem, "phys": phys, "psd": psd}

    # If dominant lot covers >=60% and has all 8 chem values -> use it
    if dom and share >= 0.60:
        has_all = all(dom.get(k) not in (None, "") for k in CHEM_KEYS)
        if has_all:
            return from_lot(dom)

    # Otherwise weighted averages
    chem = {k.capitalize(): fmt4(_weighted_value(rows, k)) for k in CHEM_KEYS}
    phys = {k: fmt4(_weighted_value(rows, k)) for k in PHYS_KEYS}
    psd  = {k: fmt4(_weighted_value(rows, k)) for k in PSD_KEYS}
    return {"chem": chem, "phys": phys, "psd": psd}

def _get_latest_anneal_qa(conn, anneal_id: int) -> Dict[str, Any] | None:
    row = conn.execute(text("""
        SELECT id, anneal_lot_id, decision, oxygen, remarks
        FROM anneal_qa
        WHERE anneal_lot_id = :lid
        ORDER BY id DESC
        LIMIT 1
    """), {"lid": anneal_id}).mappings().first()
    return dict(row) if row else None

def _get_params_for_qa(conn, qa_id: int) -> Dict[str, str]:
    try:
        rows = conn.execute(text("""
            SELECT param_name, param_value
            FROM anneal_qa_params
            WHERE anneal_qa_id = :qid
            ORDER BY id
        """), {"qid": qa_id}).mappings().all()
        out = {}
        for r in rows:
            out[str(r["param_name"])] = str(r["param_value"] or "")
        return out
    except Exception:
        return {}

def _upsert_qa_params(conn, qa_id: int, params: Dict[str, float]) -> None:
    """
    Stores all fields (chem/phys/psd) as name/value rows.
    If anneal_qa_params table doesn't exist, we no-op safely.
    """
    try:
        conn.execute(text("DELETE FROM anneal_qa_params WHERE anneal_qa_id = :qid"), {"qid": qa_id})
        conn.execute(text("""
            INSERT INTO anneal_qa_params(anneal_qa_id, param_name, param_value)
            VALUES (:qid, :name, :val)
        """), [
            {"qid": qa_id, "name": name, "val": float(val)}
            for name, val in params.items()
        ])
    except Exception:
        # silently ignore if the params table is absent
        pass

# ---------- Anneal QA: routes ----------
# top of routes.py (if not already present)
from fastapi import Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import text
from typing import Dict, Any

@router.get("/qa/{anneal_id}", response_class=HTMLResponse)
async def anneal_qa_form(
    request: Request,
    anneal_id: int,
    dep: None = Depends(require_roles("admin", "qa", "anneal")),
):
    with engine.begin() as conn:
        header_row = conn.execute(text("""
        SELECT id, date, lot_no, grade,
        COALESCE(weight_kg, 0) AS weight_kg,
        ammonia_kg, rap_cost_per_kg, cost_per_kg, src_alloc_json, qa_status
        FROM anneal_lots
        WHERE id = :id
        """), {"id": anneal_id}).mappings().first()

        if not header_row:
            raise HTTPException(status_code=404, detail="Anneal lot not found")

        header = dict(header_row)  # ensure it's a plain dict

        defaults = _anneal_default_blocks(conn, header)
        qa = _get_latest_anneal_qa(conn, anneal_id)
        saved_params: Dict[str, str] = _get_params_for_qa(conn, qa["id"]) if qa else {}

    def _pick(name: str, group: str) -> str:
        v = str(saved_params.get(name, "")).strip()
        return v if v != "" else str(defaults[group].get(name, ""))

    chem = {k.capitalize(): _pick(k.capitalize(), "chem") for k in CHEM_KEYS}
    phys = {k: _pick(k, "phys") for k in PHYS_KEYS}
    psd  = {k: _pick(k, "psd") for k in PSD_KEYS}

    return templates.TemplateResponse(
        "annealing_qa_form.html",
        {
            "request": request,
            "header": header,  # <-- present for template
            "chem": chem,
            "phys": phys,
            "psd": psd,
            "qa": qa or {"decision": "", "oxygen": "", "remarks": ""},
            "read_only": False,
        },
    )

@router.post("/qa/{anneal_id}")
async def anneal_qa_save(
    anneal_id: int,
    request: Request,
    # chemistry (required > 0)
    C: str = Form(""), Si: str = Form(""), S: str = Form(""), P: str = Form(""),
    Cu: str = Form(""), Ni: str = Form(""), Mn: str = Form(""), Fe: str = Form(""),
    # physical (required > 0)
    ad: str = Form(""), flow: str = Form(""),
    # psd (required > 0)
    p212: str = Form(""), p180: str = Form(""), n180p150: str = Form(""),
    n150p75: str = Form(""), n75p45: str = Form(""), n45: str = Form(""),
    # anneal-specific (required > 0)
    oxygen: str = Form(""),
    decision: str = Form("APPROVED"),
    remarks: str = Form(""),
    dep: None = Depends(require_roles("admin", "qa", "anneal")),
):
    # ---- helper for validation ----
    def _req_pos(name: str, s: str) -> float | None:
        s2 = (s or "").strip()
        try:
            v = float(s2)
        except Exception:
            return None
        if v <= 0:
            return None
        return v

    # collect and validate all numeric fields
    posted_chem = {"C": C, "Si": Si, "S": S, "P": P, "Cu": Cu, "Ni": Ni, "Mn": Mn, "Fe": Fe}
    posted_phys = {"ad": ad, "flow": flow}
    posted_psd  = {"p212": p212, "p180": p180, "n180p150": n180p150, "n150p75": n150p75, "n75p45": n75p45, "n45": n45}

    errors = []
    parsed = {}

    for k, v in posted_chem.items():
        val = _req_pos(k, v)
        if val is None:
            errors.append(f"{k} must be a number > 0.")
        else:
            parsed[k] = val

    for k, v in posted_phys.items():
        val = _req_pos(k.upper(), v)
        if val is None:
            errors.append(f"{k.upper()} must be a number > 0.")
        else:
            parsed[k] = val

    for k, v in posted_psd.items():
        label = { "p212":"+212", "p180":"+180", "n180p150":"-180+150",
                  "n150p75":"-150+75", "n75p45":"-75+45", "n45":"-45" }[k]
        val = _req_pos(label, v)
        if val is None:
            errors.append(f"{label} must be a number > 0.")
        else:
            parsed[k] = val

    oxy_val = _req_pos("Oxygen", oxygen)
    if oxy_val is None:
        errors.append("Oxygen must be a number > 0.")

    decision_norm = (decision or "").strip().upper()
    if decision_norm not in {"APPROVED", "HOLD", "REJECTED"}:
        errors.append("Decision must be one of: APPROVED, HOLD, REJECTED.")

    if errors:
        # Re-render the form with the user's values and an error line
        with engine.begin() as conn:
            header = conn.execute(text("""
            SELECT id, date, lot_no, grade,
            COALESCE(weight_kg, 0) AS weight_kg,
            ammonia_kg, rap_cost_per_kg, cost_per_kg, src_alloc_json, qa_status
            FROM anneal_lots
            WHERE id = :id
            """), {"id": anneal_id}).mappings().first()
            if not header:
                raise HTTPException(status_code=404, detail="Anneal lot not found")
            header = dict(header)

        # use what the user typed
        chem_ctx = {k: posted_chem[k] for k in ["C","Si","S","P","Cu","Ni","Mn","Fe"]}
        phys_ctx = {k: posted_phys[k] for k in ["ad","flow"]}
        psd_ctx  = {k: posted_psd[k]  for k in ["p212","p180","n180p150","n150p75","n75p45","n45"]}

        qa_ctx = {"decision": decision_norm, "oxygen": oxygen, "remarks": remarks}

        return templates.TemplateResponse(
            "annealing_qa_form.html",
            {
                "request": request,
                "header": header,
                "chem": chem_ctx,
                "phys": phys_ctx,
                "psd": psd_ctx,
                "qa": qa_ctx,
                "read_only": False,
                "error_text": " | ".join(errors),   # <— one thin line
            },
            status_code=400,
        )

    # ---- save when valid ----
    with engine.begin() as conn:
        exists = conn.execute(text("SELECT 1 FROM anneal_lots WHERE id = :id"), {"id": anneal_id}).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail="Anneal lot not found.")

        qa_row = conn.execute(text("""
            INSERT INTO anneal_qa (anneal_lot_id, decision, oxygen, remarks)
            VALUES (:lid, :decision, :oxygen, :remarks, NOW())
            RETURNING id
        """), {
            "lid": anneal_id,
            "decision": decision_norm,
            "oxygen": oxy_val,
            "remarks": remarks or "",
        }).mappings().first()
        qa_id = qa_row["id"]

        # snapshot params (as strings or numbers—your anneal_qa_params accepts text)
        params_payload = {
            "C": parsed["C"], "Si": parsed["Si"], "S": parsed["S"], "P": parsed["P"],
            "Cu": parsed["Cu"], "Ni": parsed["Ni"], "Mn": parsed["Mn"], "Fe": parsed["Fe"],
            "ad": parsed["ad"], "flow": parsed["flow"],
            "p212": parsed["p212"], "p180": parsed["p180"], "n180p150": parsed["n180p150"],
            "n150p75": parsed["n150p75"], "n75p45": parsed["n75p45"], "n45": parsed["n45"],
        }
        _upsert_qa_params(conn, qa_id, {k: f"{v:.4f}" for k,v in params_payload.items()})

        conn.execute(text("""
            UPDATE anneal_lots
            SET qa_status = :st
            WHERE id = :lid
        """), {"st": decision_norm, "lid": anneal_id})

    return RedirectResponse("/qa-dashboard", status_code=303)

# === Anneal Trace & PDF (drop-in, paste into annealing/routes.py) ===

@router.get("/trace/{anneal_id}", response_class=HTMLResponse)
async def anneal_trace_view(
    request: Request,
    anneal_id: int,
    dep: None = Depends(require_roles("admin", "qa", "anneal", "view")),
):
    with engine.begin() as conn:
        header = _fetch_anneal_header(conn, anneal_id)
        if not header:
            raise HTTPException(status_code=404, detail="Anneal lot not found")

        # parse allocations safely
        try:
            alloc_map = json.loads(header.get("src_alloc_json") or "{}")
        except Exception:
            alloc_map = {}

        rap_rows = _fetch_rap_rows_for_alloc(conn, alloc_map)

        # enrich RAP rows with upstream heats and GRNs
        for r in rap_rows:
            r["heats"] = []
            if r.get("base_lot_id"):
                heats = _fetch_heats_for_base_lot(conn, r["base_lot_id"])
        for h in heats:
            grns = _fetch_grns_for_heat(conn, h["heat_id"])
            h["grns"] = grns

            # --- NEW: fallback for Used Qty ---
            # If used_qty is NULL/0, compute from GRNs
            try:
                uq = float(h.get("used_qty") or 0.0)
            except Exception:
                uq = 0.0
            if uq <= 0.0:
                uq = sum(float(g.get("qty_kg") or 0.0) for g in grns)
            h["used_qty"] = uq
        r["heats"] = heats

        qa = _fetch_latest_anneal_qa_full(conn, anneal_id)

    return templates.TemplateResponse(
        "anneal_trace.html",
        {
            "request": request,
            "header": header,     # Anneal lot header
            "rap_rows": rap_rows, # RAP → Heats → GRNs
            "qa": qa,             # Latest QA + params (optional)
        },
    )


@router.get("/pdf/{anneal_id}", response_class=HTMLResponse)
async def anneal_pdf_view(
    request: Request,
    anneal_id: int,
    dep: None = Depends(require_roles("admin", "qa", "anneal", "view")),
):
    with engine.begin() as conn:
        header = _fetch_anneal_header(conn, anneal_id)
        if not header:
            raise HTTPException(status_code=404, detail="Anneal lot not found")

        try:
            alloc_map = json.loads(header.get("src_alloc_json") or "{}")
        except Exception:
            alloc_map = {}

        rap_rows = _fetch_rap_rows_for_alloc(conn, alloc_map)
        for r in rap_rows:
            r["heats"] = []
            if r.get("base_lot_id"):
                heats = _fetch_heats_for_base_lot(conn, r["base_lot_id"])
                for h in heats:
                    h["grns"] = _fetch_grns_for_heat(conn, h["heat_id"])
                r["heats"] = heats

        qa = _fetch_latest_anneal_qa_full(conn, anneal_id)

    return templates.TemplateResponse(
        "anneal_pdf.html",
        {
            "request": request,
            "header": header,
            "rap_rows": rap_rows,
            "qa": qa,
        },
    )
