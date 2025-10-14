# krn_mrp_app/fg/routes.py
from datetime import date, timedelta
from fastapi import APIRouter, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from starlette.templating import Jinja2Templates
from sqlalchemy import text
from typing import Any, Dict, List
import json, io, csv

from krn_mrp_app.deps import engine, require_roles

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# ---- CONFIG ----
FG_CAPACITY_PER_DAY = 6000.0  # kg/day
# grade → surcharge ₹/kg
FG_SURCHARGE = {
    # KIP finished products
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
    # KFS finished products
    "KFS 15/45": 20.0,
    "KFS 15/60": 20.0,
    # From oversize stream
    "KIP 40.29": 5.0,
}

KIP_GRADES = {"KIP 80.29","KIP 100.29","KIP 100.25","Premixes 01.01","Premixes 01.02","Premixes 01.03","Premixes 02.01","Premixes 02.02","Premixes 02.03","KIP 40.29"}
KFS_GRADES = {"KFS 15/45","KFS 15/60"}

def _is_admin(request: Request) -> bool:
    s = getattr(request, "state", None)
    if not s: return False
    if getattr(s, "is_admin", False): return True
    role = getattr(s, "role", None)
    if isinstance(role, str) and role.lower() == "admin": return True
    roles = getattr(s, "roles", None)
    return isinstance(roles, (list,set,tuple)) and "admin" in roles

def _table_exists(conn, t: str) -> bool:
    try:
        return bool(conn.execute(text("SELECT to_regclass('public.'||:t) IS NOT NULL"), {"t": t}).scalar())
    except Exception:
        # SQLite path
        try:
            conn.execute(text(f"SELECT 1 FROM {t} LIMIT 1"))
            return True
        except Exception:
            return False

# Pull APPROVED Grinding lots balances (like Anneal→Grinding)
def fetch_grinding_balance():
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT id, lot_no, grade, weight_kg::float AS weight_kg,
                   COALESCE(cost_per_kg,0)::float AS grind_cost_per_kg,
                   COALESCE(oversize_p80_kg,0)::float AS p80,
                   COALESCE(oversize_p40_kg,0)::float AS p40,
                   COALESCE(qa_status,'') AS qa
            FROM grinding_lots
            WHERE UPPER(COALESCE(qa_status,''))='APPROVED'
            ORDER BY id DESC
        """)).mappings().all()

        used: Dict[str, float] = {}
        if _table_exists(conn,"fg_lots"):
            for (alloc_json,) in conn.execute(text("SELECT src_alloc_json FROM fg_lots")).all():
                if not alloc_json: continue
                try:
                    d = json.loads(alloc_json)
                    for lot_no, qty in d.items():
                        used[lot_no] = used.get(lot_no, 0.0) + float(qty or 0)
                except Exception:
                    pass

    out = []
    for r in rows:
        ln = r["lot_no"]
        avail = float(r["weight_kg"] or 0) - float(used.get(ln, 0.0))
        if avail > 0.0001:
            out.append({
                "grind_id": r["id"],
                "lot_no": ln,
                "grade": (r["grade"] or "").strip().upper(),   # KIP / KFS family here (from Grinding)
                "available_kg": avail,
                "grind_cost_per_kg": float(r["grind_cost_per_kg"] or 0.0),
                "p80": float(r["p80"] or 0.0),
                "p40": float(r["p40"] or 0.0),
            })
    return out

# Weighted params from Grinding QA (latest snapshot per grinding lot)
def _latest_grind_params_map(conn, grind_ids: List[int]) -> Dict[int, Dict[str,float]]:
    if not grind_ids:
        return {}
    # header
    hdr = conn.execute(text("""
        SELECT id, grinding_lot_id
        FROM grinding_qa
        WHERE grinding_lot_id = ANY(:ids)
        ORDER BY id
    """), {"ids": grind_ids}).mappings().all()
    latest_by_gid: Dict[int,int] = {}
    for row in hdr:
        latest_by_gid[row["grinding_lot_id"]] = row["id"]
    if not latest_by_gid:
        return {}

    params = {}
    for gid, qid in latest_by_gid.items():
        rows = conn.execute(text("""
            SELECT param_name, param_value
            FROM grinding_qa_params
            WHERE grinding_qa_id=:qid
        """), {"qid": qid}).mappings().all()
        p = {}
        for r in rows:
            try:
                p[str(r["param_name"]).strip()] = float(r["param_value"])
            except Exception:
                pass
        params[gid] = p
    return params

def _family_for_fg_grade(g: str) -> str:
    g = (g or "").strip()
    if g in KIP_GRADES and g != "KIP 40.29": return "KIP"
    if g in KFS_GRADES: return "KFS"
    if g == "KIP 40.29": return "OVERSIZE"
    # fallback guess:
    return "KIP" if g.startswith("KIP") or g.startswith("Premixes") else "KFS"

# ---------------- HOME ----------------
@router.get("/", response_class=HTMLResponse)
async def fg_home(request: Request, dep: None = Depends(require_roles("fg","admin","view"))):
    today = date.today()
    with engine.begin() as conn:
        lots_today = conn.execute(text("SELECT COUNT(*) FROM fg_lots WHERE date=:d"), {"d": today}).scalar() or 0
        produced_today = conn.execute(text("SELECT COALESCE(SUM(weight_kg),0) FROM fg_lots WHERE date=:d"), {"d": today}).scalar() or 0.0
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
        "last5": last5,
        "live_stock": live_stock,
        "is_admin": _is_admin(request),
        "cap": FG_CAPACITY_PER_DAY,
    })

# ---------------- CREATE ----------------
@router.get("/create", response_class=HTMLResponse)
async def fg_create_get(request: Request, dep: None = Depends(require_roles("fg","admin"))):
    src = fetch_grinding_balance()
    err = request.query_params.get("err","")
    return templates.TemplateResponse("fg_create.html", {
        "request": request,
        "src_rows": src,
        "grades": sorted(FG_SURCHARGE.keys()),
        "err": err,
        "is_admin": _is_admin(request),
        "cap": FG_CAPACITY_PER_DAY
    })

@router.post("/create")
async def fg_create_post(request: Request, dep: None = Depends(require_roles("fg","admin"))):
    form = await request.form()
    fg_grade = (form.get("fg_grade") or "").strip()
    if fg_grade not in FG_SURCHARGE:
        return RedirectResponse("/fg/create?err=Select+a+valid+FG+grade.", status_code=303)

    # allocations from APPROVED grinding lots
    alloc: Dict[str,float] = {}
    total_alloc = 0.0
    for k,v in form.items():
        if k.startswith("alloc_"):   # alloc_<GRD lot>
            lot_no = k.replace("alloc_", "")
            try:
                q = float(v or 0)
            except Exception:
                q = 0.0
            if q > 0:
                alloc[lot_no] = q
                total_alloc += q

    try:
        lot_weight = float(form.get("lot_weight") or 0)
    except Exception:
        lot_weight = 0.0

    if lot_weight <= 0:
        return RedirectResponse("/fg/create?err=Lot+Weight+must+be+%3E+0.", status_code=303)
    if abs(lot_weight - total_alloc) > 0.01:
        return RedirectResponse("/fg/create?err=Lot+Weight+must+equal+allocated+qty.", status_code=303)

    # capacity guard
    today = date.today()
    with engine.begin() as conn:
        prod_today = conn.execute(text("SELECT COALESCE(SUM(weight_kg),0) FROM fg_lots WHERE date=:d"), {"d": today}).scalar() or 0.0
    if prod_today + lot_weight > FG_CAPACITY_PER_DAY + 1e-6:
        return RedirectResponse(f"/fg/create?err=Capacity+{FG_CAPACITY_PER_DAY:.0f}+kg/day+exceeded.", status_code=303)

    # validate families and cost base
    src_rows = fetch_grinding_balance()
    amap = {r["lot_no"]: r for r in src_rows}
    fam = set()
    wsum = 0.0
    for ln, q in alloc.items():
        r = amap.get(ln)
        if r is None or q > float(r.get("available_kg") or 0):
            return RedirectResponse(f"/fg/create?err=Grinding+lot+{ln}+not+found+or+insufficient.", status_code=303)
        fam.add(("KIP" if r["grade"].startswith("KIP") else "KFS"))
        wsum += q * float(r.get("grind_cost_per_kg") or 0.0)

    # FG grade must match family (except oversize KIP 40.29 which is from KIP)
    family = _family_for_fg_grade(fg_grade)
    if family == "OVERSIZE":
        if not all(x == "KIP" for x in fam):
            return RedirectResponse("/fg/create?err=KIP+40.29+can+only+use+KIP+source.", status_code=303)
    else:
        if len(fam) != 1 or (family not in fam):
            return RedirectResponse("/fg/create?err=FG+grade+family+and+source+family+must+match.", status_code=303)

    base_cost = (wsum / lot_weight) if lot_weight else 0.0
    surcharge = FG_SURCHARGE.get(fg_grade, 0.0)
    cost_per_kg = base_cost + surcharge

    # derive QA = weighted average of latest Grinding QA params (by allocated kg)
    with engine.begin() as conn:
        # map GRD lot_no → id
        gid_map = dict(conn.execute(text("SELECT id, lot_no FROM grinding_lots WHERE lot_no = ANY(:nos)"),
                                    {"nos": list(alloc.keys())}).all())
        grind_ids = list(gid_map.keys())
        params_map = _latest_grind_params_map(conn, grind_ids)  # {grind_id:{param:val}}
        # weights by gid
        weights: Dict[int,float] = {}
        for ln, q in alloc.items():
            # reverse lookup id from lot_no
            gid = None
            for i, lotno in gid_map.items():
                if lotno == ln: gid = i; break
            if gid is None: continue
            weights[gid] = weights.get(gid, 0.0) + float(q)

        total_w = sum(weights.values()) or 1.0
        agg: Dict[str,float] = {}
        for gid, w in weights.items():
            p = params_map.get(gid, {})
            for k, v in p.items():
                agg[k] = agg.get(k, 0.0) + v * (w / total_w)

        # create FG lot #
        prefix = "FG-" + today.strftime("%Y%m%d") + "-"
        last = conn.execute(text("""
            SELECT lot_no FROM fg_lots
            WHERE lot_no LIKE :pfx
            ORDER BY lot_no DESC
            LIMIT 1
        """), {"pfx": f"{prefix}%"}).scalar()
        seq = int(last.split("-")[-1]) + 1 if last else 1
        fg_no = f"{prefix}{seq:03d}"

        conn.execute(text("""
            INSERT INTO fg_lots
                (date, lot_no, family, fg_grade, weight_kg,
                 base_cost_per_kg, surcharge_per_kg, cost_per_kg,
                 src_alloc_json, qa_status)
            VALUES
                (:d, :no, :fam, :fg, :wt,
                 :base, :sur, :final,
                 :alloc, 'PENDING')
        """), {
            "d": today, "no": fg_no, "fam": family, "fg": fg_grade, "wt": lot_weight,
            "base": base_cost, "sur": surcharge, "final": cost_per_kg,
            "alloc": json.dumps(alloc)
        })

        # auto-create QA snapshot (params only; decision pending)
        # store the aggregated params into fg_qa_params linked to a new fg_qa row
        qa_id = conn.execute(text("""
            INSERT INTO fg_qa(fg_lot_id, decision, remarks)
            SELECT id, 'PENDING', 'Auto-fetched from Grinding QA' FROM fg_lots WHERE lot_no=:no
            RETURNING id
        """), {"no": fg_no}).mappings().first()["id"]

        if agg:
            conn.execute(text("""
                INSERT INTO fg_qa_params(fg_qa_id, param_name, param_value)
                VALUES (:qid, :n, :v)
            """), [{"qid": qa_id, "n": k, "v": f"{val:.4f}"} for k,val in agg.items()])

    return RedirectResponse("/fg/lots", status_code=303)

# ---------------- LIST ----------------
@router.get("/lots", response_class=HTMLResponse)
async def fg_lots(request: Request,
                  dep: None = Depends(require_roles("fg","admin","view")),
                  from_date: str = None, to_date: str = None, csv: int = 0):
    q = """
        SELECT id, date, lot_no, family, fg_grade, weight_kg,
               base_cost_per_kg, surcharge_per_kg, cost_per_kg,
               qa_status
        FROM fg_lots WHERE 1=1
    """
    params = {}
    if from_date:
        q += " AND date >= :f"; params["f"] = from_date
    if to_date:
        q += " AND date <= :t"; params["t"] = to_date
    q += " ORDER BY date DESC, id DESC"
    with engine.begin() as conn:
        rows = conn.execute(text(q), params).mappings().all()

    total_w = sum((r["weight_kg"] or 0) for r in rows) or 0.0
    weighted_cost = (sum((r["cost_per_kg"] or 0) * (r["weight_kg"] or 0) for r in rows) / total_w) if total_w else 0.0

    if csv:
        out = io.StringIO(); w = csv.writer(out)
        w.writerow(["Date","Lot","Family","FG Grade","Weight (kg)","Base ₹/kg","Surcharge ₹/kg","Final ₹/kg","QA"])
        for r in rows:
            w.writerow([
                r["date"], r["lot_no"], r["family"], r["fg_grade"],
                f"{(r['weight_kg'] or 0):.0f}",
                f"{(r['base_cost_per_kg'] or 0):.2f}",
                f"{(r['surcharge_per_kg'] or 0):.2f}",
                f"{(r['cost_per_kg'] or 0):.2f}",
                r["qa_status"] or ""
            ])
        return Response(out.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition":"attachment; filename=fg_lots.csv"})

    return templates.TemplateResponse("fg_lot_list.html", {
        "request": request,
        "lots": rows,
        "total_weight": total_w,
        "weighted_cost": weighted_cost,
        "is_admin": _is_admin(request),
        "today": date.today().isoformat()
    })

# ---------------- TRACE/PDF ----------------
def _fetch_fg_header(conn, fg_id: int) -> Dict[str,Any] | None:
    row = conn.execute(text("""
        SELECT id, date, lot_no, family, fg_grade, weight_kg,
               base_cost_per_kg, surcharge_per_kg, cost_per_kg,
               src_alloc_json, qa_status
        FROM fg_lots WHERE id=:id
    """), {"id": fg_id}).mappings().first()
    return dict(row) if row else None

def _fetch_grind_for_alloc(conn, alloc_map: Dict[str,float]) -> List[Dict[str,Any]]:
    if not alloc_map: return []
    lot_nos = list(alloc_map.keys())
    rows = conn.execute(text("""
        SELECT id, lot_no, grade,
               COALESCE(cost_per_kg,0)::float AS grind_cost_per_kg
        FROM grinding_lots
        WHERE lot_no = ANY(:lot_nos)
        ORDER BY id
    """), {"lot_nos": lot_nos}).mappings().all()
    out = []
    for r in rows:
        ln = r["lot_no"]
        out.append({
            "grind_id": r["id"], "grind_lot_no": ln, "grind_grade": r["grade"],
            "grind_cost_per_kg": float(r["grind_cost_per_kg"] or 0.0),
            "allocated_kg": float(alloc_map.get(ln, 0.0)),
        })
    # include unmapped if any
    known = {r["grind_lot_no"] for r in out}
    for ln, q in alloc_map.items():
        if ln not in known:
            out.append({
                "grind_id": None, "grind_lot_no": ln, "grind_grade": "",
                "grind_cost_per_kg": 0.0, "allocated_kg": float(q or 0.0)
            })
    return out

@router.get("/trace/{fg_id}", response_class=HTMLResponse)
async def fg_trace(request: Request, fg_id: int, dep: None = Depends(require_roles("fg","admin","view"))):
    with engine.begin() as conn:
        header = _fetch_fg_header(conn, fg_id)
        if not header: raise HTTPException(status_code=404, detail="FG lot not found")
        try:
            alloc_map = json.loads(header.get("src_alloc_json") or "{}")
        except Exception:
            alloc_map = {}
        grind_rows = _fetch_grind_for_alloc(conn, alloc_map)
    return templates.TemplateResponse("fg_trace.html", {"request": request, "header": header, "grind_rows": grind_rows, "is_admin": _is_admin(request)})

@router.get("/pdf/{fg_id}", response_class=HTMLResponse)
async def fg_pdf(request: Request, fg_id: int, dep: None = Depends(require_roles("fg","admin","view"))):
    with engine.begin() as conn:
        header = _fetch_fg_header(conn, fg_id)
        if not header: raise HTTPException(status_code=404, detail="FG lot not found")
        try:
            alloc_map = json.loads(header.get("src_alloc_json") or "{}")
        except Exception:
            alloc_map = {}
        grind_rows = _fetch_grind_for_alloc(conn, alloc_map)
    return templates.TemplateResponse("fg_pdf.html", {"request": request, "header": header, "grind_rows": grind_rows, "is_admin": _is_admin(request)})

# ---------------- QA minimal (decision only; params snap created on lot create) ----------------
@router.get("/qa/{fg_id}", response_class=HTMLResponse)
async def fg_qa_form(request: Request, fg_id: int, dep: None = Depends(require_roles("qa","admin","fg"))):
    with engine.begin() as conn:
        h = _fetch_fg_header(conn, fg_id)
        if not h: raise HTTPException(status_code=404, detail="FG lot not found")
        qa = conn.execute(text("""
            SELECT id, fg_lot_id, decision, remarks
            FROM fg_qa WHERE fg_lot_id=:id ORDER BY id DESC LIMIT 1
        """), {"id": fg_id}).mappings().first()
        params = {}
        if qa:
            rows = conn.execute(text("SELECT param_name, param_value FROM fg_qa_params WHERE fg_qa_id=:qid"),
                                {"qid": qa["id"]}).mappings().all()
            params = {r["param_name"]: r["param_value"] for r in rows}
    return templates.TemplateResponse("fg_qa_form.html", {"request": request, "header": h, "qa": qa, "params": params})

@router.post("/qa/{fg_id}")
async def fg_qa_save(fg_id: int, decision: str = Form("APPROVED"), remarks: str = Form(""),
                     dep: None = Depends(require_roles("qa","admin","fg"))):
    d = (decision or "").strip().upper()
    if d not in {"APPROVED","HOLD","REJECTED"}:
        return RedirectResponse(f"/fg/qa/{fg_id}?err=Decision+invalid", status_code=303)
    with engine.begin() as conn:
        qid = conn.execute(text("""
            INSERT INTO fg_qa(fg_lot_id, decision, remarks)
            VALUES (:id, :d, :r) RETURNING id
        """), {"id": fg_id, "d": d, "r": remarks or ""}).mappings().first()["id"]
        conn.execute(text("UPDATE fg_lots SET qa_status=:s WHERE id=:id"), {"s": d, "id": fg_id})
    return RedirectResponse("/qa-dashboard", status_code=303)
