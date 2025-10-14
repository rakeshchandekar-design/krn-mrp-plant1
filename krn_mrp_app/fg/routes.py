# krn_mrp_app/fg/routes.py

from __future__ import annotations
from datetime import date, timedelta
from typing import Any, Dict, List, Optional
import json, io, csv

from fastapi import APIRouter, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from starlette.templating import Jinja2Templates
from sqlalchemy import text

from krn_mrp_app.deps import engine, require_roles

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# -----------------------------------
# Config: Surcharges (₹/kg) by FG grade
# -----------------------------------
FG_SURCHARGE: Dict[str, float] = {
    # From KIP grades
    "KIP 80.29": 10.0,
    "KIP 100.29": 15.0,
    "KIP 100.25": 15.0,

    # Premixes (from KIP)
    "Premixes 01.01": 25.0,
    "Premixes 01.02": 25.0,
    "Premixes 01.03": 25.0,
    "Premixes 02.01": 25.0,
    "Premixes 02.02": 25.0,
    "Premixes 02.03": 25.0,

    # From KFS grades
    "KFS 15/45": 20.0,
    "KFS 15/60": 20.0,

    # Oversize from Grinding (+80, +40)
    "KIP 40.29": 5.0,
}

# Which FG grades belong to which family (validation lock)
FG_FAMILY: Dict[str, str] = {
    # KIP family
    "KIP 80.29": "KIP",
    "KIP 100.29": "KIP",
    "KIP 100.25": "KIP",
    "Premixes 01.01": "KIP",
    "Premixes 01.02": "KIP",
    "Premixes 01.03": "KIP",
    "Premixes 02.01": "KIP",
    "Premixes 02.02": "KIP",
    "Premixes 02.03": "KIP",

    # KFS family
    "KFS 15/45": "KFS",
    "KFS 15/60": "KFS",

    # Oversize flow considered KIP product per spec
    "KIP 40.29": "KIP",
}

# For QA auto-prefill: take average of available params (same as your anneal logic style)
AUTO_QA_FIELDS_CHEM = ["C","Si","S","P","Cu","Ni","Mn","Fe"]
AUTO_QA_FIELDS_PHYS = ["ad","flow","compressibility"]
AUTO_QA_FIELDS_PSD  = ["p212","p180","n180p150","n150p75","n75p45","n45"]

# ----------------- Helpers -----------------
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
    return isinstance(roles, (list, set, tuple)) and "admin" in roles

def _table_exists(conn, name: str) -> bool:
    try:
        if str(engine.url).startswith("sqlite"):
            q = "SELECT name FROM sqlite_master WHERE type='table' AND name=:t"
            return bool(conn.execute(text(q), {"t": name}).first())
        else:
            q = "SELECT to_regclass('public.'||:t) IS NOT NULL"
            return bool(conn.execute(text(q), {"t": name}).scalar())
    except Exception:
        return False

def _load_grinding_allocations_used(conn) -> Dict[str, float]:
    """ Sum allocations already used in FG from fg_lots.src_alloc_json """
    used: Dict[str, float] = {}
    if not _table_exists(conn, "fg_lots"):
        return used
    for (alloc_json,) in conn.execute(text("SELECT src_alloc_json FROM fg_lots")).all():
        if not alloc_json:
            continue
        try:
            d = json.loads(alloc_json)
            for lot_no, qty in d.items():
                used[lot_no] = used.get(lot_no, 0.0) + float(qty or 0.0)
        except Exception:
            pass
    return used

def fetch_grind_balance() -> List[Dict[str, Any]]:
    """
    Source inventory for FG = APPROVED grinding lots.
    available_kg = grinding_lots.weight_kg - qty already used in fg_lots.src_alloc_json
    base cost = grinding_lots.cost_per_kg (weighted in create)
    Family = 'KIP' if grade startswith('KIP') else 'KFS'
    """
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT id, lot_no, date, grade,
                   COALESCE(weight_kg,0)::float           AS weight_kg,
                   COALESCE(cost_per_kg,0)::float         AS grind_cost_per_kg,
                   COALESCE(oversize_p80_kg,0)::float     AS p80,
                   COALESCE(oversize_p40_kg,0)::float     AS p40,
                   COALESCE(qa_status,'')                 AS qa_status
            FROM grinding_lots
            WHERE UPPER(COALESCE(qa_status,''))='APPROVED'
            ORDER BY date DESC, id DESC
        """)).mappings().all()

        used = _load_grinding_allocations_used(conn)

    out: List[Dict[str, Any]] = []
    for r in rows:
        ln = r["lot_no"]
        avail = float(r["weight_kg"] or 0) - float(used.get(ln, 0.0))
        if avail > 0.0001:
            g = (r["grade"] or "").strip().upper()
            family = "KIP" if g.startswith("KIP") else ("KFS" if g.startswith("KFS") else "KIP")
            out.append({
                "grind_id": r["id"],
                "lot_no": ln,
                "date": r["date"],
                "family": family,
                "grade": r["grade"],
                "available_kg": avail,
                "grind_cost_per_kg": float(r["grind_cost_per_kg"] or 0.0),
                "oversize_p80_kg": float(r["p80"] or 0.0),
                "oversize_p40_kg": float(r["p40"] or 0.0),
            })
    return out

def _fg_family_for_grade(fg_grade: str) -> str:
    return FG_FAMILY.get(fg_grade, "KIP")

def _surcharge_for_grade(fg_grade: str) -> float:
    return float(FG_SURCHARGE.get(fg_grade, 0.0))

# --------------- HOME (KPIs) ---------------
@router.get("/", response_class=HTMLResponse)
async def fg_home(request: Request, dep: None = Depends(require_roles("admin","fg","view"))):
    today = date.today()
    with engine.begin() as conn:
        lots_today = conn.execute(text("SELECT COUNT(*) FROM fg_lots WHERE date=:d"), {"d": today}).scalar() or 0
        produced_today = conn.execute(text("SELECT COALESCE(SUM(weight_kg),0) FROM fg_lots WHERE date=:d"),
                                      {"d": today}).scalar() or 0.0
        avg_cost_today = conn.execute(text("SELECT COALESCE(AVG(cost_per_kg),0) FROM fg_lots WHERE date=:d"),
                                      {"d": today}).scalar() or 0.0
        weighted_cost_today = conn.execute(text("""
            SELECT COALESCE(SUM(cost_per_kg*weight_kg)/NULLIF(SUM(weight_kg),0),0)
            FROM fg_lots WHERE date=:d
        """), {"d": today}).scalar() or 0.0
        last5 = conn.execute(text("""
            SELECT date, COALESCE(SUM(weight_kg),0) AS qty
            FROM fg_lots WHERE date >= :d5
            GROUP BY date ORDER BY date DESC
        """), {"d5": today - timedelta(days=4)}).mappings().all()
        live_stock = conn.execute(text("""
            SELECT fg_grade, COALESCE(SUM(weight_kg),0) AS qty
            FROM fg_lots GROUP BY fg_grade ORDER BY fg_grade
        """)).mappings().all()

    return templates.TemplateResponse("fg_home.html", {
        "request": request,
        "lots_today": lots_today,
        "produced_today": produced_today,
        "avg_cost_today": avg_cost_today,
        "weighted_cost_today": weighted_cost_today,
        "last5": last5,
        "live_stock": live_stock,
        "is_admin": _is_admin(request),
    })

# --------------- CREATE ---------------
@router.get("/create", response_class=HTMLResponse)
async def fg_create_get(request: Request, dep: None = Depends(require_roles("fg","admin"))):
    rows = fetch_grind_balance()
    err = request.query_params.get("err", "")
    return templates.TemplateResponse("fg_create.html", {
        "request": request,
        "grind_rows": rows,
        "surcharges": FG_SURCHARGE,
        "err": err,
        "is_admin": _is_admin(request),
    })

@router.post("/create")
async def fg_create_post(
    request: Request,
    dep: None = Depends(require_roles("fg","admin")),
    fg_grade: str = Form(...),
    lot_weight: str = Form(...),  # numeric string
    remarks: str = Form(""),
):
    # 1) parse weight
    try:
        fg_weight = float(lot_weight or 0)
    except Exception:
        return RedirectResponse("/fg/create?err=Lot+Weight+must+be+numeric.", status_code=303)
    if fg_weight <= 0:
        return RedirectResponse("/fg/create?err=Lot+Weight+must+be+%3E+0.", status_code=303)

    # 2) collect allocations from grinding
    form = await request.form()
    allocations: Dict[str, float] = {}
    total_alloc = 0.0
    for k, v in form.items():
        if k.startswith("alloc_"):  # alloc_<grind_lot_no>
            lot_no = k.replace("alloc_", "")
            try:
                qty = float(v or 0)
            except Exception:
                qty = 0.0
            if qty > 0:
                allocations[lot_no] = qty
                total_alloc += qty

    if total_alloc <= 0.0:
        return RedirectResponse("/fg/create?err=Allocate+quantity+%3E+0+kg.", status_code=303)

    # must match
    if abs(fg_weight - total_alloc) > 0.01:
        msg = f"Lot Weight mismatch: allocated {total_alloc:.2f} kg, entered {fg_weight:.2f} kg."
        return RedirectResponse(f"/fg/create?err={msg.replace(' ','+')}", status_code=303)

    # 3) validate family lock
    family_needed = _fg_family_for_grade(fg_grade)
    avail_rows = fetch_grind_balance()
    amap = {r["lot_no"]: r for r in avail_rows}

    for lot_no, qty in allocations.items():
        r = amap.get(lot_no)
        if (r is None) or (qty > float(r.get("available_kg") or 0)):
            return RedirectResponse(f"/fg/create?err=Grinding+lot+{lot_no}+not+found+or+insufficient+balance.",
                                    status_code=303)
        if r.get("family") != family_needed and fg_grade != "KIP 40.29":
            # 'KIP 40.29' is from oversize, but we still allow from KIP family grinding lots.
            return RedirectResponse("/fg/create?err=FG+grade+family+does+not+match+source+Grinding+family.",
                                    status_code=303)

    # 4) compute weighted base cost from grinding cost_per_kg
    wsum = sum(allocations[ln] * float(amap[ln].get("grind_cost_per_kg") or 0.0) for ln in allocations)
    base_cost = wsum / fg_weight if fg_weight else 0.0
    surcharge = _surcharge_for_grade(fg_grade)
    cost_per_kg = base_cost + surcharge

    # 5) create FG lot_no (FG-YYYYMMDD-###)
    with engine.begin() as conn:
        prefix = "FG-" + date.today().strftime("%Y%m%d") + "-"
        last = conn.execute(text("""
            SELECT lot_no FROM fg_lots
            WHERE lot_no LIKE :pfx
            ORDER BY lot_no DESC
            LIMIT 1
        """), {"pfx": f"{prefix}%"}).scalar()
        seq = int(last.split("-")[-1]) + 1 if last else 1
        lot_no = f"{prefix}{seq:03d}"

        conn.execute(text("""
            INSERT INTO fg_lots
                (date, lot_no, family, fg_grade, weight_kg,
                 base_cost_per_kg, surcharge_per_kg, cost_per_kg,
                 src_alloc_json, qa_status, remarks)
            VALUES
                (:date, :lot_no, :family, :fg_grade, :weight_kg,
                 :base_cost, :surcharge, :cost, :src_alloc_json, 'PENDING', :remarks)
        """), {
            "date": date.today(),
            "lot_no": lot_no,
            "family": family_needed if fg_grade != "KIP 40.29" else "KIP",
            "fg_grade": fg_grade,
            "weight_kg": fg_weight,
            "base_cost_per_kg": base_cost,
            "surcharge_per_kg": surcharge,
            "cost_per_kg": cost_per_kg,
            "src_alloc_json": json.dumps(allocations),
            "remarks": remarks or "",
        })

    return RedirectResponse("/fg/lots", status_code=303)

# --------------- LOT LIST ---------------
@router.get("/lots", response_class=HTMLResponse)
async def fg_lots(
    request: Request,
    dep: None = Depends(require_roles("admin","fg","view")),
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    csv: int = 0
):
    query = """
        SELECT id, date, lot_no, family, fg_grade, weight_kg,
               base_cost_per_kg, surcharge_per_kg, cost_per_kg,
               qa_status, remarks
        FROM fg_lots WHERE 1=1
    """
    params: Dict[str, Any] = {}
    if from_date:
        query += " AND date >= :fd"; params["fd"] = from_date
    if to_date:
        query += " AND date <= :td"; params["td"] = to_date
    query += " ORDER BY date DESC, id DESC"

    with engine.begin() as conn:
        rows = conn.execute(text(query), params).mappings().all()

    total_weight = sum(float(r["weight_kg"] or 0.0) for r in rows)
    weighted_cost = (
        sum((float(r["cost_per_kg"] or 0.0) * float(r["weight_kg"] or 0.0)) for r in rows) / total_weight
        if total_weight > 0 else 0.0
    )

    if csv:
        out = io.StringIO(); w = csv.writer(out)
        w.writerow(["Date","Lot","Family","FG Grade","Weight (kg)","Base Cost/kg","Surcharge/kg","Final Cost/kg","QA","Remarks"])
        for r in rows:
            w.writerow([
                r["date"], r["lot_no"], r["family"], r["fg_grade"],
                f"{float(r['weight_kg'] or 0):.0f}",
                f"{float(r['base_cost_per_kg'] or 0):.2f}",
                f"{float(r['surcharge_per_kg'] or 0):.2f}",
                f"{float(r['cost_per_kg'] or 0):.2f}",
                r["qa_status"] or "", r["remarks"] or ""
            ])
        return Response(out.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition":"attachment; filename=fg_lots.csv"})

    return templates.TemplateResponse("fg_lot_list.html", {
        "request": request,
        "lots": rows,
        "total_weight": total_weight,
        "weighted_cost": weighted_cost,
        "today": date.today().isoformat(),
        "is_admin": _is_admin(request),
    })

# --------------- Trace + PDF ---------------
def _fetch_fg_header(conn, lot_id: int) -> Optional[Dict[str, Any]]:
    row = conn.execute(text("""
        SELECT id, date, lot_no, family, fg_grade, weight_kg,
               base_cost_per_kg, surcharge_per_kg, cost_per_kg,
               src_alloc_json, qa_status, remarks
        FROM fg_lots WHERE id=:id
    """), {"id": lot_id}).mappings().first()
    return dict(row) if row else None

def _fetch_grind_rows_for_alloc(conn, alloc_map: Dict[str, float]) -> List[Dict[str, Any]]:
    if not alloc_map:
        return []
    lot_nos = list(alloc_map.keys())
    rows = conn.execute(text("""
        SELECT id, lot_no, grade, cost_per_kg
        FROM grinding_lots
        WHERE lot_no = ANY(:lot_nos)
        ORDER BY id
    """), {"lot_nos": lot_nos}).mappings().all()
    out = []
    for r in rows:
        ln = r["lot_no"]
        out.append({
            "grind_id": r["id"], "grind_lot_no": ln, "grind_grade": r["grade"],
            "grind_cost_per_kg": float(r["cost_per_kg"] or 0.0),
            "allocated_kg": float(alloc_map.get(ln, 0.0)),
        })
    # include unmapped if any
    known = {r["grind_lot_no"] for r in out}
    for ln, q in alloc_map.items():
        if ln not in known:
            out.append({
                "grind_id": None, "grind_lot_no": ln, "grind_grade": "",
                "grind_cost_per_kg": 0.0, "allocated_kg": float(q or 0.0),
            })
    return out

@router.get("/trace/{fg_id}", response_class=HTMLResponse)
async def fg_trace_view(request: Request, fg_id: int, dep: None = Depends(require_roles("admin","fg","view"))):
    with engine.begin() as conn:
        header = _fetch_fg_header(conn, fg_id)
        if not header: raise HTTPException(status_code=404, detail="FG lot not found")
        try:
            alloc_map = json.loads(header.get("src_alloc_json") or "{}")
        except Exception:
            alloc_map = {}
        grind_rows = _fetch_grind_rows_for_alloc(conn, alloc_map)
        qa = _fetch_latest_fg_qa_full(conn, fg_id)
    return templates.TemplateResponse("fg_trace.html", {
        "request": request, "header": header, "grind_rows": grind_rows, "qa": qa
    })

@router.get("/pdf/{fg_id}", response_class=HTMLResponse)
async def fg_pdf_view(request: Request, fg_id: int, dep: None = Depends(require_roles("admin","fg","view"))):
    with engine.begin() as conn:
        header = _fetch_fg_header(conn, fg_id)
        if not header: raise HTTPException(status_code=404, detail="FG lot not found")
        try:
            alloc_map = json.loads(header.get("src_alloc_json") or "{}")
        except Exception:
            alloc_map = {}
        grind_rows = _fetch_grind_rows_for_alloc(conn, alloc_map)
        qa = _fetch_latest_fg_qa_full(conn, fg_id)
    return templates.TemplateResponse("fg_pdf.html", {
        "request": request, "header": header, "grind_rows": grind_rows, "qa": qa
    })

# --------------- QA (auto-prefill from grinding QA params avg) ---------------
def _get_latest_fg_qa(conn, fg_id: int) -> Optional[Dict[str, Any]]:
    row = conn.execute(text("""
        SELECT id, fg_lot_id, decision, remarks
        FROM fg_qa
        WHERE fg_lot_id = :fid
        ORDER BY id DESC
        LIMIT 1
    """), {"fid": fg_id}).mappings().first()
    return dict(row) if row else None

def _get_params_for_fg_qa(conn, qa_id: int) -> Dict[str, str]:
    try:
        rows = conn.execute(text("""
            SELECT param_name, param_value
            FROM fg_qa_params
            WHERE fg_qa_id = :qid
            ORDER BY id
        """), {"qid": qa_id}).mappings().all()
        return {str(r["param_name"]): str(r["param_value"] or "") for r in rows}
    except Exception:
        return {}

def _upsert_fg_qa_params(conn, qa_id: int, params: Dict[str, float]) -> None:
    try:
        conn.execute(text("DELETE FROM fg_qa_params WHERE fg_qa_id=:qid"), {"qid": qa_id})
        conn.execute(text("""
            INSERT INTO fg_qa_params(fg_qa_id, param_name, param_value)
            VALUES (:qid, :name, :val)
        """), [{"qid": qa_id, "name": k, "val": str(v)} for k, v in params.items()])
    except Exception:
        pass

def _avg_from_grinding_for_fg(conn, fg_header: Dict[str, Any]) -> Dict[str, float]:
    """
    Average relevant params from the contributing Grinding QA params.
    (Simple average—if you want 60% rule later, you can plug it here.)
    """
    try:
        alloc_map = json.loads(fg_header.get("src_alloc_json") or "{}")
    except Exception:
        alloc_map = {}

    if not alloc_map:
        return {}

    lot_nos = list(alloc_map.keys())
    # join grinding_lots -> latest grinding_qa -> grinding_qa_params
    params = {}
    counts = {}
    for ln in lot_nos:
        # latest QA for each grinding lot
        row = conn.execute(text("""
            SELECT q.id
            FROM grinding_lots gl
            JOIN grinding_qa q ON q.grinding_lot_id = gl.id
            WHERE gl.lot_no = :ln
            ORDER BY q.id DESC
            LIMIT 1
        """), {"ln": ln}).mappings().first()
        if not row:
            continue

        qid = row["id"]
        prms = conn.execute(text("""
            SELECT param_name, param_value
            FROM grinding_qa_params
            WHERE grinding_qa_id = :qid
        """), {"qid": qid}).mappings().all()

        for p in prms:
            name = str(p["param_name"]).strip()
            if name not in (AUTO_QA_FIELDS_CHEM + AUTO_QA_FIELDS_PHYS + AUTO_QA_FIELDS_PSD):
                continue
            try:
                v = float(p["param_value"])
            except Exception:
                continue
            params[name] = params.get(name, 0.0) + v
            counts[name] = counts.get(name, 0) + 1

    avg = {}
    for k, s in params.items():
        c = counts.get(k, 0)
        if c > 0:
            avg[k] = round(s / c, 4)

    return avg

def _fetch_latest_fg_qa_full(conn, fg_id: int) -> Optional[Dict[str, Any]]:
    qa = _get_latest_fg_qa(conn, fg_id)
    if not qa:
        return None
    params = _get_params_for_fg_qa(conn, qa["id"])
    return {"header": qa, "params": [{"name": k, "value": v, "unit": "", "spec_min": "", "spec_max": ""} for k, v in params.items()]}

@router.get("/qa/{fg_id}", response_class=HTMLResponse)
async def fg_qa_form(request: Request, fg_id: int, dep: None = Depends(require_roles("admin","qa","fg"))):
    with engine.begin() as conn:
        header = _fetch_fg_header(conn, fg_id)
        if not header:
            raise HTTPException(status_code=404, detail="FG lot not found")

        qa = _get_latest_fg_qa(conn, fg_id)
        saved = _get_params_for_fg_qa(conn, qa["id"]) if qa else {}

        # auto-prefill averages if first time
        if not qa and not saved:
            auto = _avg_from_grinding_for_fg(conn, header)
            saved = {**auto}

    return templates.TemplateResponse("fg_qa_form.html", {
        "request": request,
        "header": header,
        "qa": qa or {"decision": "APPROVED", "remarks": ""},
        "params": saved,
        "read_only": False,
    })

@router.post("/qa/{fg_id}")
async def fg_qa_save(
    fg_id: int,
    request: Request,
    decision: str = Form("APPROVED"),
    remarks: str = Form(""),
):
    dnorm = (decision or "").strip().upper()
    if dnorm not in {"APPROVED", "HOLD", "REJECTED"}:
        dnorm = "HOLD"

    # collect dynamic params from form (anything not in fixed fields is treated as a param)
    form = await request.form()
    params: Dict[str, float] = {}
    for k, v in form.items():
        if k in {"decision", "remarks"}:
            continue
        try:
            params[k] = float(v)
        except Exception:
            # ignore non-numeric
            pass

    with engine.begin() as conn:
        header = _fetch_fg_header(conn, fg_id)
        if not header:
            raise HTTPException(status_code=404, detail="FG lot not found")

        qa_id = conn.execute(text("""
            INSERT INTO fg_qa (fg_lot_id, decision, remarks)
            VALUES (:fid, :d, :r)
            RETURNING id
        """), {"fid": fg_id, "d": dnorm, "r": remarks or ""}).mappings().first()["id"]

        if params:
            _upsert_fg_qa_params(conn, qa_id, params)

        conn.execute(text("UPDATE fg_lots SET qa_status=:st WHERE id=:id"), {"st": dnorm, "id": fg_id})

    return RedirectResponse("/qa-dashboard", status_code=303)
