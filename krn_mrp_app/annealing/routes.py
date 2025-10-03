# krn_mrp_app/annealing/routes.py

from datetime import date, timedelta
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from urllib.parse import quote_plus
from starlette.templating import Jinja2Templates
from sqlalchemy import text, bindparam
import json, io, csv

from krn_mrp_app.deps import engine, require_roles


router = APIRouter()
templates = Jinja2Templates(directory="templates")  # we will place annealing HTML in global /templates

TARGET_KG_PER_DAY = 6000.0
ANNEAL_ADD_COST = 10.0  # ₹/kg add over weighted RAP cost

# ---- helper: fetch approved RAP with balance (JOIN rap_lot -> lot) ----
def fetch_approved_rap_balance():
    sql = text("""
    SELECT
        rl.id AS rap_row_id,
        l.id AS lot_id,
        CASE WHEN l.lot_no IS NULL OR l.lot_no = '' THEN 'LOT-' || l.id ELSE l.lot_no END AS lot_no,
        COALESCE(l.grade,'') AS grade,
        0.0 AS cost_per_kg,   -- TEMP fallback, since these cols don’t exist
        rl.available_qty AS available_kg
    FROM rap_lot rl
    JOIN lot l ON l.id = rl.lot_id
    WHERE rl.available_qty > 0
    ORDER BY rl.id ASC
 """)
    with engine.begin() as conn:
        return conn.execute(sql).mappings().all()

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

    return templates.TemplateResponse("annealing_home.html", {
        "request": request,
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
    })


@router.get("/create", response_class=HTMLResponse)
async def anneal_create_get(request: Request, dep: None = Depends(require_roles("anneal","admin"))):
    rap_rows = fetch_approved_rap_balance()
    err = request.query_params.get("err", "")
    return templates.TemplateResponse(
        "annealing_create.html",
        {"request": request, "rap_rows": rap_rows, "err": err}
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
   
    # ---- read RAP rows for the selected RAP lot_nos ----
    lot_nos = list(allocations.keys())
    q = text("""
        SELECT l.lot_no,
               COALESCE(l.grade,'')     AS grade,
               COALESCE(l.unit_cost, 0) AS cost_per_kg
        FROM rap_lot rl
        JOIN lot l ON l.id = rl.lot_id
        WHERE l.lot_no IN :lot_nos
          AND rl.available_qty > 0
    """).bindparams(bindparam("lot_nos", expanding=True))

    with engine.begin() as conn:
        rows = conn.execute(q, {"lot_nos": lot_nos}).mappings().all()
        if not rows:
            msg = "Selected RAP lots not found or no availability."
            return RedirectResponse(url=f"/anneal/create?err={quote_plus(msg)}", status_code=303)

        # ---- single family rule (KRIP or KRFS only) ----
        fam = {r["grade"] for r in rows}
        if len(fam) > 1:
            msg = "Only one grade allowed per anneal lot (KRIP or KRFS)."
            return RedirectResponse(url=f"/anneal/create?err={quote_plus(msg)}", status_code=303)

        rap_grade = next(iter(fam))
        out_grade = "KIP" if rap_grade == "KRIP" else "KFS"

        # ---- weighted RAP cost ----
        rap_cost_wsum = sum(
            allocations[r["lot_no"]] * float(r["cost_per_kg"] or 0.0) for r in rows
        )
        rap_cost_per_kg = rap_cost_wsum / lot_weight if lot_weight else 0.0
        cost_per_kg = rap_cost_per_kg + ANNEAL_ADD_COST  # rule: RAP + ₹10

        # ---- new anneal lot number ----
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

        # ---- insert new anneal lot ----
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

        # ---- deduct from RAP ----
        for rap_lot_no, qty in allocations.items():
            conn.execute(
                text("""
                    UPDATE rap_lot
                       SET available_qty = available_qty - :q
                     WHERE lot_id IN (SELECT id FROM lot WHERE lot_no = :lot)
                """),
                {"q": qty, "lot": rap_lot_no},
            )

    return RedirectResponse(url="/anneal/lots", status_code=303)


@router.get("/lots", response_class=HTMLResponse)
async def anneal_lots(request: Request, dep: None = Depends(require_roles("admin","anneal","view"))):
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT id, date, lot_no, grade, weight_kg, ammonia_kg, rap_cost_per_kg, cost_per_kg, qa_status
            FROM anneal_lots
            ORDER BY date DESC, id DESC      -- was: date DESC, lot_no DESC
        """)).mappings().all()

    total_weight = sum((r["weight_kg"] or 0.0) for r in rows)
    weighted_cost = (
        sum(((r["cost_per_kg"] or 0.0) * (r["weight_kg"] or 0.0)) for r in rows) / total_weight
        if total_weight > 0 else 0.0
    )

    return templates.TemplateResponse("annealing_lot_list.html", {
        "request": request,
        "lots": rows,
        "total_weight": total_weight,
        "weighted_cost": weighted_cost,
    })


@router.get("/qa/{lot_id}", response_class=HTMLResponse)
async def anneal_qa_get(lot_id: int, request: Request, dep: None = Depends(require_roles("qa","admin"))):
    with engine.begin() as conn:
        lot = conn.execute(text("SELECT * FROM anneal_lots WHERE id=:i"), {"i": lot_id}).mappings().first()
    if not lot:
        raise HTTPException(status_code=404)
    return templates.TemplateResponse("annealing_qa_form.html", {"request": request, "lot": lot})

@router.post("/qa/{lot_id}")
async def anneal_qa_post(lot_id: int, request: Request, dep: None = Depends(require_roles("qa","admin"))):
    form = await request.form()
    try:
        o_pct = float(form["o_pct"]); comp = float(form["compressibility"])
        if o_pct <= 0 or comp <= 0:
            raise ValueError("Oxygen% and Compressibility must be > 0")
    except Exception:
        return RedirectResponse(url=f"/anneal/qa/{lot_id}", status_code=303)

    fields = {
        "qa_status": form.get("qa_status","APPROVED"),
        "o_pct": o_pct, "compressibility": comp,
        "c_pct": form.get("c_pct"), "si_pct": form.get("si_pct"),
        "mn_pct": form.get("mn_pct"), "s_pct": form.get("s_pct"),
        "p_pct": form.get("p_pct"), "remarks": form.get("remarks")
    }
    # build update SQL
    sets = []
    params = {"id": lot_id}
    for k,v in fields.items():
        if v is not None and v != "":
            sets.append(f"{k} = :{k}")
            params[k] = float(v) if k.endswith("_pct") or k in ("compressibility",) else v
    if sets:
        sql = text(f"UPDATE anneal_lots SET {', '.join(sets)} WHERE id=:id")
        with engine.begin() as conn:
            conn.execute(sql, params)

    return RedirectResponse(url="/anneal/lots", status_code=303)

from datetime import date, timedelta, datetime  # make sure datetime is imported

@router.get("/downtime", response_class=HTMLResponse)
async def anneal_downtime_get(request: Request, dep: None = Depends(require_roles("anneal","admin"))):
    with engine.begin() as conn:
        logs = conn.execute(text("SELECT * FROM anneal_downtime ORDER BY date DESC, id DESC")).mappings().all()

    today = date.today()
    three_days_ago = today - timedelta(days=3)

    return templates.TemplateResponse("annealing_downtime.html", {
        "request": request,
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
