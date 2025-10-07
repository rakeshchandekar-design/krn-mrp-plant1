# krn_mrp_app/annealing/routes.py

from datetime import date, timedelta
from fastapi import APIRouter, Request, Depends, HTTPException
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
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

router = APIRouter()
templates = Jinja2Templates(directory="templates")

TARGET_KG_PER_DAY = 6000.0
ANNEAL_ADD_COST = 10.0  # ₹/kg add over weighted RAP cost

from fastapi import Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
import inspect

def _access_denied_html(title: str, line1: str, line2: str) -> str:
    return f"""
    <html>
      <head>
        <title>{title}</title>
        <meta name="viewport" content="width=device-width, initial-scale=1"/>
        <style>
          body {{ font-family: system-ui, Arial, sans-serif; display:flex; align-items:center;
                  justify-content:center; min-height: 60vh; }}
          .card {{ border:1px solid #e5e7eb; border-radius:10px; padding:20px 24px; max-width: 520px; }}
          h2 {{ color:#b91c1c; margin:0 0 8px 0; }}
          p {{ margin: 6px 0; color:#374151; }}
          a.btn {{ display:inline-block; margin-top:10px; padding:8px 14px; background:#111827; color:#fff;
                   border-radius:8px; text-decoration:none; }}
        </style>
      </head>
      <body>
        <div class="card">
          <h2>{line1}</h2>
          <p>{line2}</p>
          <p><a class="btn" href="/">Go Home</a></p>
        </div>
      </body>
    </html>
    """

async def anneal_guard(request: Request):
    """
    Wrap require_roles("admin","anneal","view") so that if it denies access,
    we return a pretty HTML page (or can be extended to JSON if you like).
    This avoids app-level exception handlers in main.py.
    """
    dep_fn = require_roles("admin", "anneal", "view")  # returns a dependency callable
    try:
        maybe_coro = dep_fn(request)
        if inspect.isawaitable(maybe_coro):
            await maybe_coro
    except HTTPException as exc:
        # Only customize 401/403; otherwise re-raise
        if exc.status_code in (401, 403):
            title = "Access Denied" if exc.status_code == 403 else "Unauthorized"
            line1 = "Sorry, Access is denied to this login." if exc.status_code == 403 else "Sorry, you are not logged in."
            line2 = "Please contact to Admin." if exc.status_code == 403 else "Please login and try again."
            return HTMLResponse(_access_denied_html(title, line1, line2), status_code=exc.status_code)
        raise

# ---------- Helper: detect if the client expects JSON (AJAX/fetch) ----------
def _wants_json(request: Request) -> bool:
    xrw = request.headers.get("X-Requested-With", "").lower()
    if xrw in ("xmlhttprequest", "fetch"):
        return True
    accept = request.headers.get("accept", "").lower()
    return ("application/json" in accept) and ("text/html" not in accept)

# ---------- Router-scoped exception handler (register it below) ----------
async def anneal_http_exception_handler(request: Request, exc: HTTPException):
    code = exc.status_code

    # Only customize 401/403; let others fall through with a simple response
    if code in (401, 403):
        msg_title = "Access Denied" if code == 403 else "Unauthorized"
        msg_line1 = "Sorry, Access is denied to this login." if code == 403 else "Sorry, you are not logged in."
        msg_line2 = "Please contact the Admin." if code == 403 else "Please login and try again."

        if _wants_json(request):
            return JSONResponse({"error": msg_title, "message": f"{msg_line1} {msg_line2}"}, status_code=code)

        html = f"""
        <html>
          <head>
            <title>{msg_title}</title>
            <meta name="viewport" content="width=device-width, initial-scale=1"/>
            <style>
              body {{ font-family: system-ui, Arial, sans-serif; display:flex; align-items:center; justify-content:center; min-height: 60vh; }}
              .card {{ border:1px solid #e5e7eb; border-radius:10px; padding:20px 24px; max-width: 520px; }}
              h2 {{ color:#b91c1c; margin:0 0 8px 0; }}
              p {{ margin: 6px 0; color:#374151; }}
              a.btn {{ display:inline-block; margin-top:10px; padding:8px 14px; background:#111827; color:#fff; border-radius:8px; text-decoration:none; }}
            </style>
          </head>
          <body>
            <div class="card">
              <h2>{msg_line1}</h2>
              <p>{msg_line2}</p>
              <p><a class="btn" href="/">Go Home</a></p>
            </div>
          </body>
        </html>
        """
        return HTMLResponse(content=html, status_code=code)

    # Default for other HTTP errors on this router
    if _wants_json(request):
        return JSONResponse({"error": f"HTTP {code}", "message": exc.detail}, status_code=code)
    return HTMLResponse(f"<h3>Error {code}</h3><p>{exc.detail}</p>", status_code=code)

# REGISTER the handler for this router (this line replaces the decorator)
router.add_exception_handler(HTTPException, anneal_http_exception_handler)
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
            "rap_rows": rap_rows,
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
        "lots": visible_rows,              # each row now has anneal_cost_per_kg & total_value
        "total_weight": total_weight,
        "weighted_cost": weighted_cost,    # uses effective anneal cost
        "from_date": from_date,
        "to_date": to_date,
        "today": date.today().isoformat(),
        "is_admin": is_admin,
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

# --- annealing trace/pdf (with QA params) ---
# Assumes: router, engine, templates, require_roles are already defined/imported in this module.

def _get_columns(conn, table_name: str) -> set[str]:
    rows = conn.execute(text("""
        SELECT lower(column_name)
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name=:t
    """), {"t": table_name}).scalars().all()
    return set(rows or [])


def _table_exists(conn, table_name: str) -> bool:
    return bool(conn.execute(text("""
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema='public' AND table_name=:t
        LIMIT 1
    """), {"t": table_name}).fetchone())


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


# ---------- NEW: tiny helper to detect which qty column exists ----------
def _pick_qty_col(conn, table_name: str, candidates: list[str]) -> str | None:
    cols = conn.execute(text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name=:t
    """), {"t": table_name}).scalars().all()
    have = {c.lower() for c in cols}
    for c in candidates:
        if c.lower() in have:
            return c
    return None
# -----------------------------------------------------------------------

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
    _guard: Response | None = Depends(anneal_guard),  # <— new guard
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
    _guard: Response | None = Depends(anneal_guard),  # <— new guard
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
