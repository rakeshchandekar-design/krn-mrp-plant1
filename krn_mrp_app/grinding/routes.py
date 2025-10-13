# krn_mrp_app/grinding/routes.py
from datetime import date, timedelta
from fastapi import APIRouter, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlalchemy import text
from starlette.templating import Jinja2Templates
from typing import Any, Dict, List
import json, io, csv
from datetime import date, timedelta, datetime

from krn_mrp_app.deps import engine, require_roles

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# --------- CONFIG ---------
GRIND_ADD_COST = 6.0  # ₹/kg add over weighted Anneal cost
OVERSIZE_MIN_SHARE = 0.07  # +80 + +40 must be ≥ 7% of created lot
TARGET_KG_PER_DAY = 10000.0  # daily target capacity for KPI

# --------- Helpers ---------
def _is_admin(request: Request) -> bool:
    s = getattr(request, "state", None)
    if not s: return False
    if getattr(s, "is_admin", False): return True
    role = getattr(s, "role", None)
    if isinstance(role, str) and role.lower() == "admin": return True
    roles = getattr(s, "roles", None)
    return isinstance(roles, (list, set, tuple)) and "admin" in roles

def _table_exists(conn, table_name: str) -> bool:
    """Check if a table exists (works for SQLite & Postgres)."""
    try:
        if str(conn.engine.url).startswith("sqlite"):
            return bool(
                conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name=:t"), {"t": table_name}).fetchone()
            )
        else:
            return bool(
                conn.execute(text("SELECT to_regclass('public.'||:t)"), {"t": table_name}).scalar()
            )
    except Exception:
        return False

def fetch_anneal_balance():
    """Fetch APPROVED Anneal lots and compute remaining available balance."""
    with engine.begin() as conn:
        anneal = conn.execute(text("""
            SELECT id, lot_no, grade,
                   COALESCE(weight_kg,0)::float AS weight_kg,
                   COALESCE(cost_per_kg, COALESCE(rap_cost_per_kg,0)+10)::float AS anneal_cost_per_kg
            FROM anneal_lots
            WHERE UPPER(COALESCE(qa_status,''))='APPROVED'
        """)).mappings().all()

        used_by_lotno: dict[str, float] = {}
        if _table_exists(conn, "grinding_lots"):
            for (alloc_json,) in conn.execute(text("SELECT src_alloc_json FROM grinding_lots")).all():
                if not alloc_json:
                    continue
                try:
                    data = json.loads(alloc_json)
                    for lot_no, qty in data.items():
                        used_by_lotno[lot_no] = used_by_lotno.get(lot_no, 0.0) + float(qty or 0)
                except Exception:
                    pass

    out = []
    for a in anneal:
        ln = a["lot_no"]
        avail = float(a["weight_kg"] or 0) - float(used_by_lotno.get(ln, 0.0))
        if avail > 0.0001:
            out.append({
                "anneal_id": a["id"],
                "lot_no": ln,
                "grade": a["grade"],
                "available_kg": avail,
                "anneal_cost_per_kg": float(a["anneal_cost_per_kg"] or 0.0),
            })
    return out

# ---------------- HOME ----------------
@router.get("/", response_class=HTMLResponse)
async def grind_home(request: Request, dep: None = Depends(require_roles("admin", "grind", "view"))):
    today = date.today()
    with engine.begin() as conn:
        lots_today = conn.execute(text("SELECT COUNT(*) FROM grinding_lots WHERE date=:d"), {"d": today}).scalar() or 0
        produced_today = conn.execute(text("SELECT COALESCE(SUM(weight_kg),0) FROM grinding_lots WHERE date=:d"), {"d": today}).scalar() or 0.0
        oversize_today = conn.execute(text("""
            SELECT COALESCE(SUM(COALESCE(oversize_p80_kg,0)+COALESCE(oversize_p40_kg,0)),0)
            FROM grinding_lots WHERE date=:d
        """), {"d": today}).scalar() or 0.0
        avg_cost_today = conn.execute(text("SELECT COALESCE(AVG(cost_per_kg),0) FROM grinding_lots WHERE date=:d"), {"d": today}).scalar() or 0.0
        weighted_cost_today = conn.execute(text("""
            SELECT COALESCE(SUM(cost_per_kg*weight_kg)/NULLIF(SUM(weight_kg),0),0)
            FROM grinding_lots WHERE date=:d
        """), {"d": today}).scalar() or 0.0
        last5 = conn.execute(text("""
            SELECT date, COALESCE(SUM(weight_kg),0) AS qty
            FROM grinding_lots WHERE date >= :d5
            GROUP BY date ORDER BY date DESC
        """), {"d5": today - timedelta(days=4)}).mappings().all()
        live_stock = conn.execute(text("""
            SELECT grade, COALESCE(SUM(weight_kg),0) AS qty
            FROM grinding_lots GROUP BY grade ORDER BY grade
        """)).mappings().all()

    utilization = round((produced_today / TARGET_KG_PER_DAY) * 100, 1) if TARGET_KG_PER_DAY else 0.0

    return templates.TemplateResponse("grinding_home.html", {
        "request": request,
        "lots_today": lots_today,
        "produced_today": produced_today,
        "oversize_today": oversize_today,
        "avg_cost_today": avg_cost_today,
        "weighted_cost_today": weighted_cost_today,
        "utilization": utilization,
        "last5": last5,
        "live_stock": live_stock,
        "is_admin": _is_admin(request),
    })

# ---------------- CREATE ----------------
@router.get("/create", response_class=HTMLResponse)
async def grind_create_get(request: Request, dep: None = Depends(require_roles("grind", "admin"))):
    rows = fetch_anneal_balance()
    err = request.query_params.get("err", "")
    return templates.TemplateResponse("grinding_create.html", {
        "request": request,
        "anneal_rows": rows,
        "err": err,
        "is_admin": _is_admin(request),
    })

@router.post("/create")
async def grind_create_post(request: Request, dep: None = Depends(require_roles("grind", "admin"))):
    form = await request.form()

    allocations, total_alloc = {}, 0.0
    for k, v in form.items():
        if k.startswith("alloc_"):
            lot_no = k.replace("alloc_", "")
            try:
                qty = float(v or 0)
            except Exception:
                qty = 0.0
            if qty > 0:
                allocations[lot_no] = qty
                total_alloc += qty

    if total_alloc <= 0:
        return RedirectResponse("/grind/create?err=Allocate+quantity+%3E+0+kg.", status_code=303)

    try:
        lot_weight = float(form.get("lot_weight") or 0)
    except Exception:
        return RedirectResponse("/grind/create?err=Lot+Weight+must+be+a+number.", status_code=303)
    if lot_weight <= 0:
        return RedirectResponse("/grind/create?err=Lot+Weight+must+be+%3E+0.", status_code=303)

    if abs(lot_weight - total_alloc) > 0.01:
        msg = f"Lot Weight mismatch: allocated {total_alloc:.2f} kg, entered {lot_weight:.2f} kg."
        return RedirectResponse(f"/grind/create?err={msg.replace(' ','+')}", status_code=303)

    try:
        p80 = float(form.get("oversize_p80") or 0)
        p40 = float(form.get("oversize_p40") or 0)
    except Exception:
        return RedirectResponse("/grind/create?err=Oversize+must+be+numeric.", status_code=303)

    if (p80 + p40) < (OVERSIZE_MIN_SHARE * lot_weight - 1e-6):
        min_need = OVERSIZE_MIN_SHARE * lot_weight
        msg = f"Oversize total (+80 + +40) must be ≥ {min_need:.2f} kg ({OVERSIZE_MIN_SHARE*100:.1f}%)."
        return RedirectResponse(f"/grind/create?err={msg.replace(' ','+')}", status_code=303)

    avail_rows = fetch_anneal_balance()
    amap = {r["lot_no"]: r for r in avail_rows}
    grades = set()
    for lot_no, qty in allocations.items():
        r = amap.get(lot_no)
        if (r is None) or (qty > float(r.get("available_kg") or 0)):
            return RedirectResponse(
                f"/grind/create?err=Anneal+lot+{lot_no}+not+found+or+insufficient+balance.",
                status_code=303
            )
        grades.add((r.get("grade") or "").strip().upper())

    fam = {"KIP" if g.startswith("KIP") else ("KFS" if g.startswith("KFS") else g) for g in grades}
    if len(fam) > 1:
        return RedirectResponse("/grind/create?err=Only+one+family+(KIP+or+KFS)+per+grind+lot.", status_code=303)
    out_grade = list(fam)[0] or ""

    wsum = sum(allocations[ln] * float(amap[ln].get("anneal_cost_per_kg") or 0.0) for ln in allocations)
    input_cost = wsum / lot_weight if lot_weight else 0.0
    cost_per_kg = input_cost + GRIND_ADD_COST

    with engine.begin() as conn:
        prefix = "GRD-" + date.today().strftime("%Y%m%d") + "-"
        last = conn.execute(text("""
            SELECT lot_no FROM grinding_lots
            WHERE lot_no LIKE :pfx ORDER BY lot_no DESC LIMIT 1
        """), {"pfx": f"{prefix}%"}).scalar()
        seq = int(last.split("-")[-1]) + 1 if last else 1
        lot_no = f"{prefix}{seq:03d}"

        conn.execute(text("""
            INSERT INTO grinding_lots
                (date, lot_no, src_alloc_json, grade, weight_kg,
                 input_cost_per_kg, process_cost_per_kg, cost_per_kg,
                 oversize_p80_kg, oversize_p40_kg, qa_status)
            VALUES
                (:date, :lot_no, :src_alloc_json, :grade, :weight_kg,
                 :input_cost, :proc_cost, :cost, :p80, :p40, 'PENDING')
        """), {
            "date": date.today(),
            "lot_no": lot_no,
            "src_alloc_json": json.dumps(allocations),
            "grade": out_grade,
            "weight_kg": lot_weight,
            "input_cost": input_cost,
            "proc_cost": GRIND_ADD_COST,
            "cost": cost_per_kg,
            "p80": p80,
            "p40": p40,
        })

    return RedirectResponse("/grind/lots", status_code=303)

# ---------------- LOT LIST ----------------
@router.get("/lots", response_class=HTMLResponse)
async def grind_lots(request: Request, dep: None = Depends(require_roles("admin","grind","view")),
                     from_date: str = None, to_date: str = None, csv: int = 0):
    query = """
        SELECT id, date, lot_no, grade, weight_kg,
               input_cost_per_kg, process_cost_per_kg, cost_per_kg,
               oversize_p80_kg, oversize_p40_kg, qa_status
        FROM grinding_lots WHERE 1=1
    """
    params = {}
    if from_date:
        query += " AND date >= :from_date"; params["from_date"] = from_date
    if to_date:
        query += " AND date <= :to_date"; params["to_date"] = to_date
    query += " ORDER BY date DESC, id DESC"

    with engine.begin() as conn:
        rows = conn.execute(text(query), params).mappings().all()

    total_weight = sum((r["weight_kg"] or 0) for r in rows)
    weighted_cost = (
        sum((r["cost_per_kg"] or 0) * (r["weight_kg"] or 0) for r in rows) / total_weight if total_weight > 0 else 0.0
    )
    oversize_total = sum((r["oversize_p80_kg"] or 0) + (r["oversize_p40_kg"] or 0) for r in rows)

    is_admin = _is_admin(request)

    if csv:
        out = io.StringIO(); w = csv.writer(out)
        if is_admin:
            w.writerow(["Date","Lot","Grade","Weight (kg)","+80 (kg)","+40 (kg)",
                        "Proc Cost/kg (₹)","Final Cost/kg (₹)","QA"])
        else:
            w.writerow(["Date","Lot","Grade","Weight (kg)","+80 (kg)","+40 (kg)","QA"])

        for r in rows:
            base = [
                r["date"], r["lot_no"], r["grade"],
                f"{(r['weight_kg'] or 0):.0f}",
                f"{(r['oversize_p80_kg'] or 0):.2f}",
                f"{(r['oversize_p40_kg'] or 0):.2f}",
            ]
            if is_admin:
                base += [
                    f"{(r['process_cost_per_kg'] or 0):.2f}",
                    f"{(r['cost_per_kg'] or 0):.2f}",
                ]
            base += [r["qa_status"] or ""]
            w.writerow(base)

        return Response(
            out.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition":"attachment; filename=grinding_lots.csv"}
        )

    return templates.TemplateResponse("grinding_lot_list.html", {
        "request": request, "lots": rows,
        "total_weight": total_weight, "weighted_cost": weighted_cost,
        "oversize_total": oversize_total,
        "from_date": from_date, "to_date": to_date,
        "today": date.today().isoformat(),
        "is_admin": _is_admin(request),
    })

# ---------------- TRACE + PDF ----------------
def _fetch_grind_header(conn, lot_id: int) -> Dict[str,Any] | None:
    row = conn.execute(text("""
        SELECT id, date, lot_no, grade, weight_kg,
               input_cost_per_kg, process_cost_per_kg, cost_per_kg,
               oversize_p80_kg, oversize_p40_kg, src_alloc_json, qa_status
        FROM grinding_lots WHERE id=:id
    """), {"id": lot_id}).mappings().first()
    return dict(row) if row else None

def _fetch_anneal_rows_for_alloc(conn, alloc_map: Dict[str, float]) -> List[Dict[str,Any]]:
    if not alloc_map: return []
    lot_nos = list(alloc_map.keys())
    rows = conn.execute(text("""
        SELECT id, lot_no, grade,
               COALESCE(cost_per_kg, COALESCE(rap_cost_per_kg,0)+10)::float AS anneal_cost_per_kg
        FROM anneal_lots WHERE lot_no = ANY(:lot_nos)
    """), {"lot_nos": lot_nos}).mappings().all()
    out = []
    for r in rows:
        ln = r["lot_no"]
        out.append({
            "anneal_id": r["id"], "anneal_lot_no": ln, "anneal_grade": r["grade"],
            "anneal_cost_per_kg": float(r["anneal_cost_per_kg"] or 0.0),
            "allocated_kg": float(alloc_map.get(ln, 0.0)),
        })
    return out

@router.get("/trace/{grind_id}", response_class=HTMLResponse)
async def grind_trace_view(request: Request, grind_id: int, dep: None = Depends(require_roles("admin","grind","view"))):
    with engine.begin() as conn:
        header = _fetch_grind_header(conn, grind_id)
        if not header:
            raise HTTPException(status_code=404, detail="Grinding lot not found")
        try:
            alloc_map = json.loads(header.get("src_alloc_json") or "{}")
        except Exception:
            alloc_map = {}
        anneal_rows = _fetch_anneal_rows_for_alloc(conn, alloc_map)
        qa = _fetch_latest_grind_qa_full(conn, grind_id)
    return templates.TemplateResponse("grinding_trace.html", {
        "request": request, "header": header, "anneal_rows": anneal_rows, "qa": qa
    })

@router.get("/pdf/{grind_id}", response_class=HTMLResponse)
async def grind_pdf_view(request: Request, grind_id: int, dep: None = Depends(require_roles("admin","grind","view"))):
    with engine.begin() as conn:
        header = _fetch_grind_header(conn, grind_id)
        if not header:
            raise HTTPException(status_code=404, detail="Grinding lot not found")
        try:
            alloc_map = json.loads(header.get("src_alloc_json") or "{}")
        except Exception:
            alloc_map = {}
        anneal_rows = _fetch_anneal_rows_for_alloc(conn, alloc_map)
        qa = _fetch_latest_grind_qa_full(conn, grind_id)
    return templates.TemplateResponse("grinding_pdf.html", {
        "request": request, "header": header, "anneal_rows": anneal_rows, "qa": qa
    })

# ---------------- QA ----------------
def _get_latest_grind_qa(conn, grind_id: int) -> Dict[str, Any] | None:
    row = conn.execute(text("""
        SELECT id, grinding_lot_id, decision, oxygen, compressibility, remarks
        FROM grinding_qa
        WHERE grinding_lot_id = :gid
        ORDER BY id DESC
        LIMIT 1
    """), {"gid": grind_id}).mappings().first()
    return dict(row) if row else None


def _get_params_for_grind_qa(conn, qa_id: int) -> Dict[str, str]:
    try:
        rows = conn.execute(text("""
            SELECT param_name, param_value
            FROM grinding_qa_params
            WHERE grinding_qa_id = :qid
            ORDER BY id
        """), {"qid": qa_id}).mappings().all()
        return {str(r["param_name"]): str(r["param_value"] or "") for r in rows}
    except Exception:
        return {}


def _upsert_grind_qa_params(conn, qa_id: int, params: Dict[str, float]) -> None:
    try:
        conn.execute(text("DELETE FROM grinding_qa_params WHERE grinding_qa_id=:qid"), {"qid": qa_id})
        conn.execute(text("""
            INSERT INTO grinding_qa_params(grinding_qa_id, param_name, param_value)
            VALUES (:qid, :name, :val)
        """), [{"qid": qa_id, "name": k, "val": float(v)} for k, v in params.items()])
    except Exception:
        pass


def _fetch_latest_grind_qa_full(conn, grind_id: int) -> Dict[str, Any] | None:
    qa = _get_latest_grind_qa(conn, grind_id)
    if not qa:
        return None
    params = _get_params_for_grind_qa(conn, qa["id"])
    return {
        "header": qa,
        "params": [{"name": k, "value": v, "unit": "", "spec_min": "", "spec_max": ""} for k, v in params.items()]
    }


@router.get("/qa/{grind_id}", response_class=HTMLResponse)
async def grind_qa_form(request: Request, grind_id: int, dep: None = Depends(require_roles("admin", "qa", "grind"))):
    with engine.begin() as conn:
        header = _fetch_grind_header(conn, grind_id)
        if not header:
            raise HTTPException(status_code=404, detail="Grinding lot not found")
        qa = _get_latest_grind_qa(conn, grind_id)
        saved = _get_params_for_grind_qa(conn, qa["id"]) if qa else {}
    return templates.TemplateResponse("grinding_qa_form.html", {
        "request": request,
        "header": header,
        "qa": qa or {"decision": "", "oxygen": "", "compressibility": "", "remarks": ""},
        "params": saved,
        "read_only": False,
    })


@router.post("/qa/{grind_id}")
async def grind_qa_save(
    grind_id: int,
    request: Request,
    # chemistry
    C: str = Form(""), Si: str = Form(""), S: str = Form(""), P: str = Form(""),
    Cu: str = Form(""), Ni: str = Form(""), Mn: str = Form(""), Fe: str = Form(""),
    # physical
    ad: str = Form(""), flow: str = Form(""), compressibility: str = Form(""),
    # psd
    p212: str = Form(""), p180: str = Form(""), n180p150: str = Form(""),
    n150p75: str = Form(""), n75p45: str = Form(""), n45: str = Form(""),
    # extra
    oxygen: str = Form(""),
    decision: str = Form("APPROVED"),
    remarks: str = Form("")
):
    def _req_pos(name: str, s: str) -> float | None:
        s2 = (s or "").strip()
        try:
            v = float(s2)
        except Exception:
            return None
        return v if v > 0 else None

    chem = {"C": C, "Si": Si, "S": S, "P": P, "Cu": Cu, "Ni": Ni, "Mn": Mn, "Fe": Fe}
    phys = {"ad": ad, "flow": flow, "compressibility": compressibility}
    psd = {"p212": p212, "p180": p180, "n180p150": n180p150, "n150p75": n150p75, "n75p45": n75p45, "n45": n45}

    errors, parsed = [], {}

    for k, v in chem.items():
        val = _req_pos(k, v)
        if val is None:
            errors.append(f"{k} must be a number > 0.")
        else:
            parsed[k] = val

    for k, v in phys.items():
        val = _req_pos(k, v)
        if val is None:
            errors.append(f"{k.upper()} must be a number > 0.")
        else:
            parsed[k] = val

    for k, v in psd.items():
        label = {"p212": "+212", "p180": "+180", "n180p150": "-180+150",
                 "n150p75": "-150+75", "n75p45": "-75+45", "n45": "-45"}[k]
        val = _req_pos(label, v)
        if val is None:
            errors.append(f"{label} must be a number > 0.")
        else:
            parsed[k] = val

    o2 = _req_pos("OXYGEN", oxygen)
    if o2 is None:
        errors.append("Oxygen must be a number > 0.")

    dnorm = (decision or "").strip().upper()
    if dnorm not in {"APPROVED", "HOLD", "REJECTED"}:
        errors.append("Decision must be APPROVED, HOLD, or REJECTED.")

    with engine.begin() as conn:
        header = _fetch_grind_header(conn, grind_id)
        if not header:
            raise HTTPException(status_code=404, detail="Grinding lot not found")

        if errors:
            return templates.TemplateResponse("grinding_qa_form.html", {
                "request": request,
                "header": header,
                "qa": {"decision": dnorm, "oxygen": oxygen, "compressibility": compressibility, "remarks": remarks},
                "params": {**chem, **phys, **psd},
                "read_only": False,
                "error_text": " | ".join(errors)
            }, status_code=400)

        qa_id = conn.execute(text("""
            INSERT INTO grinding_qa (grinding_lot_id, decision, oxygen, compressibility, remarks)
            VALUES (:gid, :d, :o2, :comp, :r)
            RETURNING id
        """), {"gid": grind_id, "d": dnorm, "o2": o2, "comp": parsed.get("compressibility", 0.0),
                "r": remarks or ""}).mappings().first()["id"]

        payload = {
            **{k: parsed[k] for k in chem.keys() if k in parsed},
            **{k: parsed[k] for k in phys.keys() if k in parsed},
            **{k: parsed[k] for k in psd.keys() if k in parsed},
        }

        _upsert_grind_qa_params(conn, qa_id, {k: f"{v:.4f}" for k, v in payload.items()})
        conn.execute(text("UPDATE grinding_lots SET qa_status=:st WHERE id=:id"), {"st": dnorm, "id": grind_id})

    return RedirectResponse("/qa-dashboard", status_code=303)

# ---------------- DOWNTIME ----------------
@router.get("/downtime", response_class=HTMLResponse)
async def grind_downtime_get(request: Request, dep: None = Depends(require_roles("admin","grind"))):
    today = date.today().isoformat()
    min_date = (date.today() - timedelta(days=90)).isoformat()
    with engine.begin() as conn:
        logs = conn.execute(text("""
            SELECT date, minutes, area, reason
            FROM grinding_downtime
            ORDER BY date DESC, id DESC
            LIMIT 200
        """)).mappings().all()
    return templates.TemplateResponse("grinding_downtime.html", {
        "request": request,
        "today": today,
        "min_date": min_date,
        "logs": logs,
        "err": request.query_params.get("err", "")
    })

@router.post("/downtime", response_class=HTMLResponse)
async def grind_downtime_post(
    request: Request,
    dep: None = Depends(require_roles("admin","grind")),
    date: str = Form(...),
    minutes: int = Form(...),
    area: str = Form(...),
    reason: str = Form(...)
):
    try:
        d = datetime.strptime(date, "%Y-%m-%d").date()
    except Exception:
        return RedirectResponse("/grind/downtime?err=Invalid+date", status_code=303)
    if (minutes or 0) <= 0:
        return RedirectResponse("/grind/downtime?err=Minutes+must+be+%3E+0", status_code=303)
    if not area or not reason:
        return RedirectResponse("/grind/downtime?err=Area+and+Reason+are+required", status_code=303)

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO grinding_downtime(date, minutes, area, reason)
            VALUES (:d, :m, :a, :r)
        """), {"d": d, "m": minutes, "a": area, "r": reason})

    return RedirectResponse("/grind/downtime", status_code=303)

@router.get("/downtime.csv")
async def grind_downtime_csv(request: Request, dep: None = Depends(require_roles("admin","grind"))):
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT date, minutes, area, reason
            FROM grinding_downtime
            ORDER BY date DESC, id DESC
        """)).mappings().all()
    out = io.StringIO(); w = csv.writer(out)
    w.writerow(["Date","Minutes","Type","Reason"])
    for r in rows:
        w.writerow([r["date"], r["minutes"], r["area"], r["reason"]])
    return Response(
        out.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition":"attachment; filename=grinding_downtime.csv"}
    )
