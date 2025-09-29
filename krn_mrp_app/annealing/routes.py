# krn_mrp_app/annealing/routes.py
from fastapi import APIRouter, Request, Depends, HTTPException, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.templating import Jinja2Templates
from sqlalchemy import text
from datetime import date
import json, io, csv

from krn_mrp_app.main import engine, role_allowed  # uses your existing helpers

router = APIRouter()
templates = Jinja2Templates(directory="templates")  # we will place annealing HTML in global /templates

TARGET_KG_PER_DAY = 6000.0
ANNEAL_ADD_COST = 10.0  # ₹/kg add over weighted RAP cost

# ---- role dependency (FastAPI) ----
def require_roles(*roles):
    def _dep(request: Request):
        if not role_allowed(request, set(roles)):
            raise HTTPException(status_code=403, detail="Forbidden")
    return _dep

# ---- helper: fetch approved RAP with balance (JOIN rap_lot -> lot) ----
def fetch_approved_rap_balance():
    sql = text("""
        SELECT
          rl.id            AS rap_row_id,
          l.id             AS lot_id,
          COALESCE(l.lot_no, 'LOT-' || l.id) AS lot_no,
          l.grade          AS grade,          -- KRIP / KRFS
          l.cost_per_kg    AS cost_per_kg,    -- change this name if your cost column differs
          rl.available_qty AS available_kg
        FROM rap_lot rl
        JOIN lot l ON l.id = rl.lot_id
        WHERE rl.available_qty > 0
          AND (l.status = 'APPROVED' OR l.qa_status = 'APPROVED')
        ORDER BY l.date ASC, rl.id ASC
    """)
    with engine.begin() as conn:
        return conn.execute(sql).mappings().all()

# ------------------ ROUTES ------------------

@router.get("/", response_class=HTMLResponse)
async def anneal_home(request: Request, dep: None = Depends(require_roles("admin","anneal","view"))):
    with engine.begin() as conn:
        lots_today = conn.execute(
            text("SELECT COUNT(*) FROM anneal_lots WHERE date=:d"),
            {"d": date.today()}
        ).scalar() or 0
        nh3_today = conn.execute(
            text("SELECT COALESCE(SUM(ammonia_kg),0) FROM anneal_lots WHERE date=:d"),
            {"d": date.today()}
        ).scalar() or 0.0
        avg_cost_today = conn.execute(
            text("SELECT COALESCE(AVG(cost_per_kg),0) FROM anneal_lots WHERE date=:d"),
            {"d": date.today()}
        ).scalar() or 0.0
        weighted_cost_today = conn.execute(
            text("""
                SELECT COALESCE(SUM(cost_per_kg * weight_kg) / NULLIF(SUM(weight_kg),0),0)
                FROM anneal_lots WHERE date=:d
            """),
            {"d": date.today()}
        ).scalar() or 0.0

    return templates.TemplateResponse("annealing_home.html", {
        "request": request,
        "target": TARGET_KG_PER_DAY,
        "lots_today": lots_today,
        "nh3_today": nh3_today,
        "avg_cost_today": avg_cost_today,
        "weighted_cost_today": weighted_cost_today,
        "user": request.session.get("user"),
    })


@router.get("/create", response_class=HTMLResponse)
async def anneal_create_get(request: Request, dep: None = Depends(require_roles("anneal","admin"))):
    rap_rows = fetch_approved_rap_balance()
    return templates.TemplateResponse("annealing_create.html", {"request": request, "rap_rows": rap_rows})

@router.post("/create")
async def anneal_create_post(
    request: Request,
    dep: None = Depends(require_roles("anneal","admin"))
):
    form = await request.form()
    # collect allocations
    allocations = {}
    total_alloc = 0.0
    for k, v in form.items():
        if k.startswith("alloc_") and (v or "").strip():
            rap_lot_no = k.replace("alloc_", "")
            qty = float(v)
            if qty > 0:
                allocations[rap_lot_no] = qty
                total_alloc += qty
    if total_alloc <= 0:
        return RedirectResponse(url="/anneal/create", status_code=303)

    ammonia_kg = float(form.get("ammonia_kg") or 0)

    # get RAP rows for selected lots
    lots_tuple = tuple(allocations.keys())
    placeholders = ",".join([f":p{i}" for i in range(len(lots_tuple))]) or "NULL"
    params = {f"p{i}": lots_tuple[i] for i in range(len(lots_tuple))}
    q = text(f"""
        SELECT l.lot_no, l.grade, l.cost_per_kg
        FROM rap_lot rl
        JOIN lot l ON l.id = rl.lot_id
        WHERE l.lot_no IN ({placeholders})
    """)
    with engine.begin() as conn:
        rows = conn.execute(q, params).mappings().all()
        if not rows:
            return RedirectResponse(url="/anneal/create", status_code=303)

        fam = {r["grade"] for r in rows}
        if len(fam) > 1:
            # must allocate from one family only
            return RedirectResponse(url="/anneal/create", status_code=303)

        rap_grade = list(fam)[0]
        out_grade = "KIP" if rap_grade == "KRIP" else "KFS"

        # weighted RAP cost
        rap_cost_wsum = 0.0
        for r in rows:
            rap_cost_wsum += allocations.get(r["lot_no"], 0.0) * float(r["cost_per_kg"] or 0)
        rap_cost_per_kg = rap_cost_wsum / total_alloc if total_alloc else 0.0
        cost_per_kg = rap_cost_per_kg + ANNEAL_ADD_COST

        # new lot_no
        prefix = "ANL-" + date.today().strftime("%Y%m%d") + "-"
        last = conn.execute(text("SELECT lot_no FROM anneal_lots WHERE lot_no LIKE :pfx ORDER BY lot_no DESC LIMIT 1"),
                            {"pfx": f"{prefix}%"}).scalar()
        seq = int(last.split("-")[-1]) + 1 if last else 1
        lot_no = f"{prefix}{seq:03d}"

        # insert and deduct from RAP
        conn.execute(text("""
            INSERT INTO anneal_lots
              (lot_no, date, src_alloc_json, grade, weight_kg, rap_cost_per_kg, cost_per_kg, ammonia_kg, qa_status)
            VALUES
              (:lot_no, :date, :src_alloc_json, :grade, :weight_kg, :rap_cost_per_kg, :cost_per_kg, :ammonia_kg, 'PENDING')
        """), {
            "lot_no": lot_no, "date": date.today(),
            "src_alloc_json": json.dumps(allocations),
            "grade": out_grade, "weight_kg": total_alloc,
            "rap_cost_per_kg": rap_cost_per_kg, "cost_per_kg": cost_per_kg,
            "ammonia_kg": ammonia_kg
        })
        for rap_lot_no, qty in allocations.items():
            conn.execute(text("""
                UPDATE rap_lot SET available_qty = available_qty - :q
                WHERE lot_id IN (SELECT id FROM lot WHERE lot_no = :lot) AND available_qty >= :q
            """), {"q": qty, "lot": rap_lot_no})

    return RedirectResponse(url="/anneal/lots", status_code=303)

@router.get("/lots", response_class=HTMLResponse)
async def anneal_lots(request: Request, dep: None = Depends(require_roles("admin","anneal","view"))):
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT id, date, lot_no, grade, weight_kg, ammonia_kg, rap_cost_per_kg, cost_per_kg, qa_status
            FROM anneal_lots ORDER BY date DESC, lot_no DESC
        """)).mappings().all()
    return templates.TemplateResponse("annealing_lot_list.html", {"request": request, "lots": rows})

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

@router.get("/downtime", response_class=HTMLResponse)
async def anneal_downtime_get(request: Request, dep: None = Depends(require_roles("anneal","admin"))):
    with engine.begin() as conn:
        logs = conn.execute(text("SELECT * FROM anneal_downtime ORDER BY date DESC, id DESC")).mappings().all()
    return templates.TemplateResponse("annealing_downtime.html", {"request": request, "logs": logs})

@router.post("/downtime")
async def anneal_downtime_post(
    request: Request,
    dep: None = Depends(require_roles("anneal","admin"))
):
    form = await request.form()
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO anneal_downtime (date, minutes, area, reason)
            VALUES (:date, :minutes, :area, :reason)
        """), {
            "date": form["date"], "minutes": int(form["minutes"]),
            "area": form["area"], "reason": form["reason"]
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
