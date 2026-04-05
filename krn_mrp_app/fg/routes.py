# krn_mrp_app/fg/routes.py

from __future__ import annotations
from datetime import date, timedelta
from typing import Any, Dict, List, Optional
import json, io, csv
from urllib.parse import quote_plus

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
    # KIP family (existing surcharges increased by ₹4)
    "KIP 80.29": 14.0,
    "KIP 100.29": 19.0,
    "KIP 100.25": 19.0,
    "KIPH 100.29": 20.0,
    "KIP 40.29": 9.0,
    "KIP 20": 6.0,
    # Motherson job-work family: no surcharge, same cost carries forward
    "KIP M 100.29": 0.0,
    "KIP M 80.29": 0.0,
    "KIP M 40.29": 0.0,
    "KIP M 20": 0.0,
    "KSP 100.25": 19.0,
    "KSP 100.29": 19.0,
    "KSP 40.29": 9.0,
    "KSP 20": 6.0,
    "KSP BW 100.25": 19.0,
    "KSP TW 100.25": 19.0,

    # Premixes (existing surcharges increased by ₹4)
    "Premixes 01.01": 29.0,
    "Premixes 01.02": 29.0,
    "Premixes 01.03": 29.0,
    "Premixes 02.01": 29.0,
    "Premixes 02.02": 29.0,
    "Premixes 02.03": 29.0,

    # KFS family (existing surcharges increased by ₹4)
    "KFS 15/45": 24.0,
    "KFS 15/60": 24.0,
    "KFS 40.29": 9.0,
    "KFS 20": 6.0,
}

# Which FG grades belong to which family (validation lock)
FG_FAMILY: Dict[str, str] = {
    # KIP family
    "KIP 80.29": "KIP",
    "KIP 100.29": "KIP",
    "KIP 100.25": "KIP",
    "KIPH 100.29": "KIP",
    "KIP 40.29": "KIP",
    "KIP 20": "KIP",
    "KIP M 100.29": "KIPM",
    "KIP M 80.29": "KIPM",
    "KIP M 40.29": "KIPM",
    "KIP M 20": "KIPM",
    "KSP 100.25": "KSP",
    "KSP 100.29": "KSP",
    "KSP 40.29": "KSP",
    "KSP 20": "KSP",
    "KSP BW 100.25": "KSP",
    "KSP TW 100.25": "KSP",

    # Premixes
    "Premixes 01.01": "KIP",
    "Premixes 01.02": "KIP",
    "Premixes 01.03": "KIP",
    "Premixes 02.01": "KIP",
    "Premixes 02.02": "KIP",
    "Premixes 02.03": "KIP",

    # KFS family
    "KFS 15/45": "KFS",
    "KFS 15/60": "KFS",
    "KFS 40.29": "KFS",
    "KFS 20": "KFS",
}

# For QA auto-prefill: take average of available params (same as your anneal logic style)
AUTO_QA_FIELDS_CHEM = ["C","Si","S","P","Cu","Ni","Mn","Fe"]
AUTO_QA_FIELDS_PHYS = ["ad","flow","compressibility"]
AUTO_QA_FIELDS_PSD  = ["p212","p180","n180p150","n150p75","n75p45","n45"]


def _family_from_grade_text(g: str) -> str:
    g = (g or "").strip().upper()
    if not g:
        return "KIP"
    if g.startswith("KIPM") or g.startswith("KIP M") or g.startswith("KRM"):
        return "KIPM"
    if g.startswith("KSP"):
        return "KSP"
    if g.startswith("KFS") or "15/45" in g or "15/60" in g or g.endswith("FS"):
        return "KFS"
    if g.startswith("KIP"):
        return "KIP"
    return "KIP"

def _compose_trace_id(parts):
    vals=[]
    for p in parts:
        s=(str(p or "").strip())
        if s and s not in vals:
            vals.append(s)
    return vals[0] if len(vals)==1 else ("+".join(vals)[:240] if vals else "")

# ----------------- Helpers -----------------
def _tpl_auth(request: Request) -> dict:
    sess = (getattr(request, "session", {}) or {})
    return {"user": sess.get("username", "") or "", "role": sess.get("role", "guest") or "guest"}

def _is_admin(request: Request) -> bool:
    s = getattr(request, "state", None)
    if s:
        if getattr(s, "is_admin", False):
            return True
        role = getattr(s, "role", None)
        if isinstance(role, str) and role.lower() == "admin":
            return True
        roles = getattr(s, "roles", None)
        if isinstance(roles, (list, set, tuple)) and "admin" in {str(x).lower() for x in roles}:
            return True

    sess = (getattr(request, "session", {}) or {})
    sess_role = str(sess.get("role") or "").strip().lower()
    if sess_role == "admin":
        return True

    cookie_role = str(getattr(request, "cookies", {}).get("role", "") or "").strip().lower()
    return cookie_role == "admin"

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

def _load_fg_allocations_used(conn) -> tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Split FG allocations into main grinding, +80 oversize and +40 oversize usage.
    Ignore legacy / void / rejected FG rows so old bad data does not block valid grinding stock.
    """
    main_used: Dict[str, float] = {}
    ov80_used: Dict[str, float] = {}
    ov40_used: Dict[str, float] = {}
    if not _table_exists(conn, "fg_lots"):
        return main_used, ov80_used, ov40_used

    cols = set()
    try:
        if str(engine.url).startswith("sqlite"):
            cols = {str(r[1]).lower() for r in conn.execute(text("PRAGMA table_info(fg_lots)"))}
        else:
            cols = {str(r[0]).lower() for r in conn.execute(text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema='public' AND table_name='fg_lots'
            """)).all()}
    except Exception:
        cols = set()

    select_cols = ["src_alloc_json"]
    if "qa_status" in cols:
        select_cols.append("qa_status")
    if "status" in cols:
        select_cols.append("status")
    sql = f"SELECT {', '.join(select_cols)} FROM fg_lots"

    active_qa_statuses = {"APPROVED", "PENDING", "HOLD"}
    ignored_qa_statuses = {"REJECTED", "CANCELLED", "VOID", "DELETED"}
    ignored_statuses = {"VOID", "CANCELLED", "DELETED"}

    for row in conn.execute(text(sql)).mappings().all():
        qa_status = str((row.get("qa_status") if "qa_status" in row else "") or "").strip().upper()
        status = str((row.get("status") if "status" in row else "") or "").strip().upper()

        if qa_status in ignored_qa_statuses:
            continue
        if status in ignored_statuses:
            continue

        # Older FG rows created during patch phases sometimes have blank/legacy QA status
        # but still carry src_alloc_json. Those rows must not block valid grinding source lots.
        if "qa_status" in row and qa_status and qa_status not in active_qa_statuses:
            continue
        if "qa_status" in row and not qa_status:
            continue

        alloc_json = row.get("src_alloc_json")
        if not alloc_json:
            continue
        try:
            d = json.loads(alloc_json)
            for key, qty in d.items():
                q = float(qty or 0.0)
                if q <= 0:
                    continue
                if key.startswith("OV80|"):
                    lot_no = key.split("|", 1)[1]
                    ov80_used[lot_no] = ov80_used.get(lot_no, 0.0) + q
                elif key.startswith("OV40|"):
                    lot_no = key.split("|", 1)[1]
                    ov40_used[lot_no] = ov40_used.get(lot_no, 0.0) + q
                else:
                    main_used[key] = main_used.get(key, 0.0) + q
        except Exception:
            pass
    return main_used, ov80_used, ov40_used


def _fifo_allocate_rows(rows: List[Dict[str, Any]], required_qty: float, key_field: str = "lot_no", avail_field: str = "available_kg") -> Dict[str, float]:
    remaining = float(required_qty or 0.0)
    out: Dict[str, float] = {}
    ordered = sorted(rows, key=lambda r: (str(r.get("date") or ""), str(r.get(key_field) or "")))
    for r in ordered:
        if remaining <= 1e-6:
            break
        avail = float(r.get(avail_field) or 0.0)
        if avail <= 1e-6:
            continue
        take = min(avail, remaining)
        out[str(r.get(key_field))] = take
        remaining -= take
    return out


def fetch_grind_balance() -> List[Dict[str, Any]]:
    """Approved grinding stock available for normal FG grades."""
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

        main_used, _ov80_used, _ov40_used = _load_fg_allocations_used(conn)

    out: List[Dict[str, Any]] = []
    for r in rows:
        ln = r["lot_no"]
        main_qty = float(r["weight_kg"] or 0.0) - float(r.get("p80") or 0.0) - float(r.get("p40") or 0.0)
        if main_qty < 0:
            main_qty = 0.0
        avail = main_qty - float(main_used.get(ln, 0.0))
        if avail > 0.0001:
            family = _family_from_grade_text(r.get("grade") or "")
            out.append({
                "source_type": "MAIN",
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



def fetch_oversize_balance(kind: str, family_needed: str | None = None) -> List[Dict[str, Any]]:
    """
    Oversize balances available for conversion to 40.29 / 20 FG grades.
    Oversize stays as BYPRODUCT until converted into FG.
    """
    kind = (kind or "").upper()
    family_needed = (family_needed or "").strip().upper()
    col = "p80" if kind == "P80" else "p40"
    key_prefix = "OV80|" if kind == "P80" else "OV40|"
    label = "+80 Oversize" if kind == "P80" else "+40 Oversize"

    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT id, lot_no, date, grade,
                   COALESCE(cost_per_kg,0)::float         AS grind_cost_per_kg,
                   COALESCE(oversize_p80_kg,0)::float     AS p80,
                   COALESCE(oversize_p40_kg,0)::float     AS p40,
                   COALESCE(qa_status,'')                 AS qa_status
            FROM grinding_lots
            WHERE UPPER(COALESCE(qa_status,''))='APPROVED'
            ORDER BY date DESC, id DESC
        """)).mappings().all()
        _main_used, ov80_used, ov40_used = _load_fg_allocations_used(conn)

    out: List[Dict[str, Any]] = []
    for r in rows:
        ln = r["lot_no"]
        fam = _family_from_grade_text(r["grade"] or "")
        if family_needed and fam != family_needed:
            continue
        used_map = ov80_used if kind == "P80" else ov40_used
        raw_qty = float(r[col] or 0.0)
        avail = raw_qty - float(used_map.get(ln, 0.0))
        if avail > 0.0001:
            out.append({
                "source_type": key_prefix.rstrip("|"),
                "grind_id": r["id"],
                "lot_no": f"{key_prefix}{ln}",
                "display_lot_no": ln,
                "date": r["date"],
                "family": fam,
                "grade": label,
                "available_kg": avail,
                "grind_cost_per_kg": float(r["grind_cost_per_kg"] or 0.0),
                "oversize_p80_kg": float(r["p80"] or 0.0),
                "oversize_p40_kg": float(r["p40"] or 0.0),
            })
    return out


def _family_matches_grade(value: str, family_needed: str) -> bool:
    fam = (family_needed or "").strip().upper()
    if not fam:
        return True
    return _family_from_grade_text(value) == fam


def fetch_fg_source_balance(fg_grade: str) -> List[Dict[str, Any]]:
    grade = (fg_grade or "").strip().upper()
    family_needed = _fg_family_for_grade(fg_grade)

    if grade.endswith("40.29"):
        rows = fetch_oversize_balance("P80", family_needed=family_needed)
        return sorted(rows, key=lambda r: (str(r.get("date") or ""), str(r.get("lot_no") or "")), reverse=True)

    if grade.endswith("20"):
        rows = fetch_oversize_balance("P40", family_needed=family_needed)
        return sorted(rows, key=lambda r: (str(r.get("date") or ""), str(r.get("lot_no") or "")), reverse=True)

    rows = []
    seen = set()
    for r in fetch_grind_balance():
        fam = (r.get("family") or "").strip().upper()
        grade_text = str(r.get("grade") or "")
        if fam == family_needed or _family_matches_grade(grade_text, family_needed):
            rows.append(r)
            seen.add(str(r.get("lot_no") or ""))

    # Extra-safe fallback for KFS family so approved KFS grinding lots are not missed due to old/inconsistent grade text.
    if family_needed == "KFS":
        with engine.begin() as conn:
            db_rows = conn.execute(text("""
                SELECT id, lot_no, date, grade,
                       COALESCE(weight_kg,0)::float       AS weight_kg,
                       COALESCE(cost_per_kg,0)::float     AS grind_cost_per_kg,
                       COALESCE(oversize_p80_kg,0)::float AS p80,
                       COALESCE(oversize_p40_kg,0)::float AS p40
                FROM grinding_lots
                WHERE UPPER(COALESCE(qa_status,''))='APPROVED'
                  AND UPPER(COALESCE(grade,'')) LIKE 'KFS%'
                ORDER BY date DESC, id DESC
            """)).mappings().all()
            main_used, _ov80_used, _ov40_used = _load_fg_allocations_used(conn)

        for r in db_rows:
            ln = str(r.get("lot_no") or "")
            if not ln or ln in seen:
                continue
            main_qty = float(r.get("weight_kg") or 0.0) - float(r.get("p80") or 0.0) - float(r.get("p40") or 0.0)
            if main_qty < 0:
                main_qty = 0.0
            avail = main_qty - float(main_used.get(ln, 0.0))
            if avail > 0.0001:
                rows.append({
                    "source_type": "MAIN",
                    "grind_id": r["id"],
                    "lot_no": ln,
                    "date": r["date"],
                    "family": "KFS",
                    "grade": r["grade"],
                    "available_kg": avail,
                    "grind_cost_per_kg": float(r.get("grind_cost_per_kg") or 0.0),
                    "oversize_p80_kg": float(r.get("p80") or 0.0),
                    "oversize_p40_kg": float(r.get("p40") or 0.0),
                })

    return sorted(rows, key=lambda r: (str(r.get("date") or ""), str(r.get("lot_no") or "")), reverse=True)


def fetch_fg_source_debug(fg_grade: str) -> List[Dict[str, Any]]:
    """Admin-only diagnostic view to explain why a grinding lot is or is not eligible for FG."""
    grade = (fg_grade or "").strip().upper()
    family_needed = _fg_family_for_grade(fg_grade)
    oversize_mode = "P80" if grade.endswith("40.29") else ("P40" if grade.endswith("20") else "MAIN")

    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT id, lot_no, date, grade,
                   COALESCE(weight_kg,0)::float       AS weight_kg,
                   COALESCE(cost_per_kg,0)::float     AS grind_cost_per_kg,
                   COALESCE(oversize_p80_kg,0)::float AS p80,
                   COALESCE(oversize_p40_kg,0)::float AS p40,
                   UPPER(COALESCE(qa_status,''))      AS qa_status
            FROM grinding_lots
            ORDER BY date DESC, id DESC
        """)).mappings().all()
        main_used, ov80_used, ov40_used = _load_fg_allocations_used(conn)

    out: List[Dict[str, Any]] = []
    for r in rows:
        lot_no = str(r.get("lot_no") or "")
        grade_text = str(r.get("grade") or "")
        family = _family_from_grade_text(grade_text)
        qa_status = str(r.get("qa_status") or "").strip().upper()
        weight = float(r.get("weight_kg") or 0.0)
        p80 = float(r.get("p80") or 0.0)
        p40 = float(r.get("p40") or 0.0)
        main_qty = max(0.0, weight - p80 - p40)
        main_used_qty = float(main_used.get(lot_no, 0.0))
        ov80_used_qty = float(ov80_used.get(lot_no, 0.0))
        ov40_used_qty = float(ov40_used.get(lot_no, 0.0))
        main_avail = max(0.0, main_qty - main_used_qty)
        p80_avail = max(0.0, p80 - ov80_used_qty)
        p40_avail = max(0.0, p40 - ov40_used_qty)

        eligible = False
        reason = ""
        available_kg = 0.0
        source_type = oversize_mode

        if qa_status != "APPROVED":
            reason = f"QA status is {qa_status or 'BLANK'}"
        elif oversize_mode == "MAIN":
            source_type = "MAIN"
            available_kg = main_avail
            if family != family_needed and not _family_matches_grade(grade_text, family_needed):
                reason = f"Family mismatch: resolved {family or '-'} vs needed {family_needed or '-'}"
            elif available_kg <= 0.0001:
                reason = f"No main balance: net 0.0 (main {main_qty:.1f} - used {main_used_qty:.1f})"
            else:
                eligible = True
                reason = "Eligible"
        elif oversize_mode == "P80":
            source_type = "OV80"
            available_kg = p80_avail
            if family != family_needed and not _family_matches_grade(grade_text, family_needed):
                reason = f"Family mismatch: resolved {family or '-'} vs needed {family_needed or '-'}"
            elif available_kg <= 0.0001:
                reason = f"No +80 balance: net 0.0 ({p80:.1f} - used {ov80_used_qty:.1f})"
            else:
                eligible = True
                reason = "Eligible"
        else:
            source_type = "OV40"
            available_kg = p40_avail
            if family != family_needed and not _family_matches_grade(grade_text, family_needed):
                reason = f"Family mismatch: resolved {family or '-'} vs needed {family_needed or '-'}"
            elif available_kg <= 0.0001:
                reason = f"No +40 balance: net 0.0 ({p40:.1f} - used {ov40_used_qty:.1f})"
            else:
                eligible = True
                reason = "Eligible"

        out.append({
            "lot_no": lot_no,
            "date": r.get("date"),
            "grade": grade_text,
            "family": family,
            "qa_status": qa_status,
            "source_type": source_type,
            "main_qty": main_qty,
            "main_used_qty": main_used_qty,
            "p80_qty": p80,
            "p80_used_qty": ov80_used_qty,
            "p40_qty": p40,
            "p40_used_qty": ov40_used_qty,
            "available_kg": available_kg,
            "eligible": eligible,
            "reason": reason,
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
    yday = today - timedelta(days=1)
    with engine.begin() as conn:
        lots_yday = conn.execute(text("SELECT COUNT(*) FROM fg_lots WHERE date=:d"), {"d": yday}).scalar() or 0
        produced_yday = conn.execute(text("SELECT COALESCE(SUM(weight_kg),0) FROM fg_lots WHERE date=:d"),
                                      {"d": yday}).scalar() or 0.0
        avg_cost_yday = conn.execute(text("SELECT COALESCE(AVG(cost_per_kg),0) FROM fg_lots WHERE date=:d"),
                                      {"d": yday}).scalar() or 0.0
        weighted_cost_yday = conn.execute(text("""
            SELECT COALESCE(SUM(cost_per_kg*weight_kg)/NULLIF(SUM(weight_kg),0),0)
            FROM fg_lots WHERE date=:d
        """), {"d": yday}).scalar() or 0.0
        last5 = conn.execute(text("""
            SELECT date, COALESCE(SUM(weight_kg),0) AS qty
            FROM fg_lots WHERE date >= :d5
            GROUP BY date ORDER BY date DESC
        """), {"d5": today - timedelta(days=4)}).mappings().all()
        live_stock = conn.execute(text("""
            SELECT fg_grade, COALESCE(SUM(weight_kg),0) AS qty
            FROM fg_lots
            WHERE UPPER(COALESCE(fg_grade,'')) NOT LIKE 'OVERSIZE %'
            GROUP BY fg_grade
            ORDER BY fg_grade
        """),).mappings().all()

    # Oversize byproduct bucket:
    # 1) live remaining +80/+40 from approved grinding lots minus FG conversions
    # 2) legacy opening stock already stored in fg_lots as Oversize 80 / Oversize 40
    byproduct_stock = []
    try:
        p80_rows = fetch_oversize_balance("P80")
        p40_rows = fetch_oversize_balance("P40")
        bucket = {}
        for r in p80_rows:
            fam = (r.get("family") or "MISC")
            bucket.setdefault(fam, {"family": fam, "p80_qty": 0.0, "p40_qty": 0.0, "value": 0.0})
            bucket[fam]["p80_qty"] += float(r.get("available_kg") or 0.0)
            bucket[fam]["value"] += float(r.get("available_kg") or 0.0) * float(r.get("grind_cost_per_kg") or 0.0)
        for r in p40_rows:
            fam = (r.get("family") or "MISC")
            bucket.setdefault(fam, {"family": fam, "p80_qty": 0.0, "p40_qty": 0.0, "value": 0.0})
            bucket[fam]["p40_qty"] += float(r.get("available_kg") or 0.0)
            bucket[fam]["value"] += float(r.get("available_kg") or 0.0) * float(r.get("grind_cost_per_kg") or 0.0)

        with engine.begin() as conn:
            cols = {str(r[0]).lower() for r in conn.execute(text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema='public' AND table_name='fg_lots'
            """)).all()}
            qty_expr = "COALESCE(weight_kg, 0)"
            if "remaining_qty" in cols:
                qty_expr = "COALESCE(remaining_qty, weight_kg, 0)"
            status_expr = "COALESCE(status,'ON_HAND')='ON_HAND'" if "status" in cols else "1=1"
            family_expr = "UPPER(COALESCE(family,''))" if "family" in cols else "''"

            legacy_rows = conn.execute(text(f"""
                SELECT
                    {family_expr} AS family,
                    UPPER(COALESCE(fg_grade,'')) AS fg_grade,
                    COALESCE(SUM({qty_expr}),0)::float AS qty,
                    COALESCE(SUM(({qty_expr}) * COALESCE(cost_per_kg,0)),0)::float AS value
                FROM fg_lots
                WHERE COALESCE(qa_status,'APPROVED')='APPROVED'
                  AND {status_expr}
                  AND UPPER(COALESCE(fg_grade,'')) IN ('OVERSIZE 80', 'OVERSIZE 40')
                GROUP BY {family_expr}, UPPER(COALESCE(fg_grade,''))
            """)).mappings().all()

        for r in legacy_rows:
            fam = (r["family"] or "").strip().upper()
            fg_grade = str(r["fg_grade"] or "").strip().upper()
            if not fam:
                fam = "KIPM" if "M 40.29" in fg_grade or "M 80.29" in fg_grade else "MISC"
            bucket.setdefault(fam, {"family": fam, "p80_qty": 0.0, "p40_qty": 0.0, "value": 0.0})
            qty = float(r["qty"] or 0.0)
            if fg_grade == "OVERSIZE 80":
                bucket[fam]["p80_qty"] += qty
            elif fg_grade == "OVERSIZE 40":
                bucket[fam]["p40_qty"] += qty
            bucket[fam]["value"] += float(r["value"] or 0.0)

        byproduct_stock = [v for v in bucket.values() if (float(v["p80_qty"] or 0)+float(v["p40_qty"] or 0)) > 0.0001]
        byproduct_stock.sort(key=lambda x: x["family"])
    except Exception:
        byproduct_stock = []

    return templates.TemplateResponse("fg_home.html", {
        "request": request,
        "lots_yday": lots_yday,
        "produced_yday": produced_yday,
        "avg_cost_yday": avg_cost_yday,
        "weighted_cost_yday": weighted_cost_yday,
        "last5": last5,
        "live_stock": live_stock,
        "byproduct_stock": byproduct_stock,
        "byproduct_totals": {"p80_qty": sum(float(r.get("p80_qty") or 0.0) for r in byproduct_stock), "p40_qty": sum(float(r.get("p40_qty") or 0.0) for r in byproduct_stock), "value": sum(float(r.get("value") or 0.0) for r in byproduct_stock)},
        "is_admin": _is_admin(request),
        "user": (request.session.get("username") if getattr(request, "session", None) else request.cookies.get("username", "")),
        "role": (request.session.get("role") if getattr(request, "session", None) else request.cookies.get("role", "guest")),
    })

# --------------- CREATE ---------------
@router.get("/create", response_class=HTMLResponse)
async def fg_create_get(request: Request, dep: None = Depends(require_roles("fg","admin"))):
    err = request.query_params.get("err", "")
    selected_fg_grade = request.query_params.get("fg_grade", "")
    rows = fetch_fg_source_balance(selected_fg_grade) if selected_fg_grade else []
    debug_rows = fetch_fg_source_debug(selected_fg_grade) if selected_fg_grade and _is_admin(request) else []
    today = date.today()
    min_date = (today - timedelta(days=4)).isoformat()
    selected_lot_date = request.query_params.get("lot_date", today.isoformat())
    return templates.TemplateResponse("fg_create.html", {
        "request": request,
        "src_rows": rows,
        "debug_rows": debug_rows,
        "selected_fg_grade": selected_fg_grade,
        "selected_lot_date": selected_lot_date,
        "grades": sorted(FG_SURCHARGE.keys()),
        "surcharges": FG_SURCHARGE,
        "err": err,
        "cap": 10000.0,
        "today": today.isoformat(),
        "min_date": min_date,
        "is_admin": _is_admin(request),
        "user": (request.session.get("username") if getattr(request, "session", None) else request.cookies.get("username", "")),
        "role": (request.session.get("role") if getattr(request, "session", None) else request.cookies.get("role", "guest")),
    })

@router.post("/create")
async def fg_create_post(
    request: Request,
    dep: None = Depends(require_roles("fg","admin")),
    lot_date: str = Form(...),
    fg_grade: str = Form(...),
    lot_weight: str = Form(...),
    remarks: str = Form(""),
):
    try:
        fg_lot_date = date.fromisoformat((lot_date or "").strip())
    except Exception:
        return RedirectResponse(f"/fg/create?err={quote_plus('FG lot date is invalid.')}&fg_grade={quote_plus(fg_grade)}&lot_date={quote_plus(lot_date or '')}", status_code=303)

    today = date.today()
    min_allowed_date = today - timedelta(days=4)
    if fg_lot_date > today:
        return RedirectResponse(f"/fg/create?err=Future+date+is+not+allowed.&fg_grade={quote_plus(fg_grade)}&lot_date={fg_lot_date.isoformat()}", status_code=303)
    if fg_lot_date < min_allowed_date:
        return RedirectResponse(f"/fg/create?err=Only+current+date+and+last+4+days+are+allowed+for+FG+lot+creation.&fg_grade={quote_plus(fg_grade)}&lot_date={fg_lot_date.isoformat()}", status_code=303)

    try:
        fg_weight = float(lot_weight or 0)
    except Exception:
        return RedirectResponse(f"/fg/create?err=Lot+Weight+must+be+numeric.&fg_grade={quote_plus(fg_grade)}&lot_date={fg_lot_date.isoformat()}", status_code=303)
    if fg_weight <= 0:
        return RedirectResponse(f"/fg/create?err=Lot+Weight+must+be+%3E+0.&fg_grade={quote_plus(fg_grade)}&lot_date={fg_lot_date.isoformat()}", status_code=303)

    form = await request.form()
    family_needed = _fg_family_for_grade(fg_grade)
    avail_rows = fetch_fg_source_balance(fg_grade)
    amap = {r["lot_no"]: r for r in avail_rows}
    total_alloc = sum(float(r.get("available_kg") or 0.0) for r in avail_rows)
    if total_alloc <= 0.0:
        return RedirectResponse(f"/fg/create?err=No+eligible+FIFO+source+balance+available.&fg_grade={quote_plus(fg_grade)}&lot_date={fg_lot_date.isoformat()}", status_code=303)
    if total_alloc + 1e-6 < fg_weight:
        return RedirectResponse(f"/fg/create?err=Insufficient+FIFO+source+balance+for+requested+FG+lot+weight.&fg_grade={quote_plus(fg_grade)}&lot_date={fg_lot_date.isoformat()}", status_code=303)

    allocations = _fifo_allocate_rows(avail_rows, fg_weight, key_field="lot_no", avail_field="available_kg")

    wsum = sum(allocations[ln] * float(amap[ln].get("grind_cost_per_kg") or 0.0) for ln in allocations)
    base_cost = wsum / fg_weight if fg_weight else 0.0
    surcharge = _surcharge_for_grade(fg_grade)
    cost_per_kg = base_cost + surcharge

    parent_trace_ids = []
    for ln in allocations.keys():
        parent_tid = str(amap[ln].get("trace_id") or "").strip()
        parent_trace_ids.append(parent_tid if parent_tid else ln)

    with engine.begin() as conn:
        prefix = "FG-" + fg_lot_date.strftime("%Y%m%d") + "-"
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
                 src_alloc_json, qa_status, remarks, trace_id, job_card_no)
            VALUES
                (:date, :lot_no, :family, :fg_grade, :weight_kg,
                 :base_cost_per_kg, :surcharge_per_kg, :cost_per_kg,
                 :src_alloc_json, 'PENDING', :remarks, :trace_id, :job_card_no)
        """), {
            "date": fg_lot_date,
            "lot_no": lot_no,
            "family": family_needed,
            "fg_grade": fg_grade,
            "weight_kg": fg_weight,
            "base_cost_per_kg": base_cost,
            "surcharge_per_kg": surcharge,
            "cost_per_kg": cost_per_kg,
            "src_alloc_json": json.dumps(allocations),
            "remarks": remarks or "",
            "trace_id": _compose_trace_id(parent_trace_ids),
            "job_card_no": f"JC-FG-{lot_no}",
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
        rows = [dict(r) for r in conn.execute(text(query), params).mappings().all()]
        dispatch_used: Dict[int, float] = {}
        if _table_exists(conn, "dispatch_items"):
            used_rows = conn.execute(text("""
                SELECT fg_lot_id, COALESCE(SUM(qty_kg),0)::float AS used
                FROM dispatch_items
                GROUP BY fg_lot_id
            """)).mappings().all()
            dispatch_used = {int(r["fg_lot_id"]): float(r["used"] or 0.0) for r in used_rows if r.get("fg_lot_id") is not None}

    show_all_history = bool(from_date or to_date)
    calc_rows = []
    for r in rows:
        prepared_qty = float(r.get("weight_kg") or 0.0)
        balance_qty = max(prepared_qty - float(dispatch_used.get(int(r.get("id") or 0), 0.0)), 0.0)
        r["prepared_qty"] = prepared_qty
        r["balance_qty"] = balance_qty
        if show_all_history or balance_qty > 0.0001:
            calc_rows.append(r)

    rows = calc_rows
    total_weight = sum(float(r["prepared_qty"] or 0.0) for r in rows)
    total_balance = sum(float(r["balance_qty"] or 0.0) for r in rows)
    weighted_cost = (
        sum((float(r["cost_per_kg"] or 0.0) * float(r["prepared_qty"] or 0.0)) for r in rows) / total_weight
        if total_weight > 0 else 0.0
    )

    if csv:
        out = io.StringIO(); w = csv.writer(out)
        w.writerow(["Date","Lot","Family","FG Grade","Prepared Qty (kg)","Balance Qty (kg)","Base Cost/kg","Surcharge/kg","Final Cost/kg","QA","Remarks"])
        for r in rows:
            w.writerow([
                r["date"], r["lot_no"], r["family"], r["fg_grade"],
                f"{float(r['prepared_qty'] or 0):.2f}",
                f"{float(r['balance_qty'] or 0):.2f}",
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
        "total_balance": total_balance,
        "weighted_cost": weighted_cost,
        "today": date.today().isoformat(),
        "showing_all_history": show_all_history,
        "is_admin": _is_admin(request),
        "user": (request.session.get("username") if getattr(request, "session", None) else request.cookies.get("username", "")),
        "role": (request.session.get("role") if getattr(request, "session", None) else request.cookies.get("role", "guest")),
    })

# --------------- Trace + PDF ---------------
def _fetch_fg_header(conn, lot_id: int) -> Optional[Dict[str, Any]]:
    row = conn.execute(text("""
        SELECT id, date, lot_no, family, fg_grade, weight_kg,
               base_cost_per_kg, surcharge_per_kg, cost_per_kg,
               src_alloc_json, qa_status, remarks, trace_id, job_card_no
        FROM fg_lots WHERE id=:id
    """), {"id": lot_id}).mappings().first()
    return dict(row) if row else None

def _fetch_grind_rows_for_alloc(conn, alloc_map: Dict[str, float]) -> List[Dict[str, Any]]:
    if not alloc_map:
        return []

    lot_nos = [str(k) for k in alloc_map.keys() if not str(k).startswith("OV80|") and not str(k).startswith("OV40|")]
    out: List[Dict[str, Any]] = []
    known = set()

    if lot_nos:
        rows = conn.execute(text("""
            SELECT id, lot_no, grade, trace_id, cost_per_kg
            FROM grinding_lots
            WHERE lot_no = ANY(:lot_nos)
            ORDER BY id
        """), {"lot_nos": lot_nos}).mappings().all()

        for r in rows:
            ln = r["lot_no"]
            known.add(ln)
            out.append({
                "grind_id": r["id"],
                "grind_lot_no": ln,
                "grind_grade": r["grade"],
                "grind_trace_id": str(r.get("trace_id") or ""),
                "grind_cost_per_kg": float(r["cost_per_kg"] or 0.0),
                "allocated_kg": float(alloc_map.get(ln, 0.0)),
            })

    for raw_key, q in alloc_map.items():
        key = str(raw_key)
        if key in known:
            continue
        if key.startswith("OV80|") or key.startswith("OV40|"):
            parts = key.split("|", 1)
            src_lot = parts[1] if len(parts) > 1 else key
            out.append({
                "grind_id": None,
                "grind_lot_no": src_lot,
                "grind_grade": "Oversize +80" if key.startswith("OV80|") else "Oversize +40",
                "grind_trace_id": "",
                "grind_cost_per_kg": 0.0,
                "allocated_kg": float(q or 0.0),
            })
        else:
            out.append({
                "grind_id": None,
                "grind_lot_no": key,
                "grind_grade": "",
                "grind_trace_id": "",
                "grind_cost_per_kg": 0.0,
                "allocated_kg": float(q or 0.0),
            })

    return out

@router.get("/trace/{fg_id}", response_class=HTMLResponse)
async def fg_trace_view(request: Request, fg_id: int, dep: None = Depends(require_roles("admin","fg","view","qa"))):
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
async def fg_pdf_view(request: Request, fg_id: int, dep: None = Depends(require_roles("admin","fg","view","qa"))):
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
    params: Dict[str, Any] = {}
    header = {"decision": "PENDING", "remarks": ""}

    if qa:
        header = qa
        params = _get_params_for_fg_qa(conn, qa["id"]) or {}

    if not params:
        fg_header = _fetch_fg_header(conn, fg_id)
        if fg_header:
            params = {k: f"{v:.4f}" for k, v in _avg_from_grinding_for_fg(conn, fg_header).items()}

    param_rows = [
        {"name": k, "value": v, "unit": "", "spec_min": "", "spec_max": ""}
        for k, v in params.items()
    ]

    if not qa and not param_rows:
        return None

    return {"header": header, "params": param_rows}

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


@router.get("/jobcard/{fg_id}", response_class=HTMLResponse)
async def fg_jobcard(request: Request, fg_id: int, dep: None = Depends(require_roles("admin","fg","view","qa"))):
    with engine.begin() as conn:
        head = conn.execute(text("SELECT * FROM fg_lots WHERE id=:id"), {"id": fg_id}).mappings().first()
        if not head: raise HTTPException(404, "FG lot not found")
    return templates.TemplateResponse("jobcard_stage.html", {"request": request, "stage": "FINAL PRODUCT", "header": head, "trace_id": head.get("trace_id") or head.get("lot_no"), "job_card_no": head.get("job_card_no") or f"JC-FG-{head.get('lot_no')}"})
