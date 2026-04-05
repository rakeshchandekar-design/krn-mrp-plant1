from datetime import datetime
from typing import Any, Dict, List

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


@router.get("/", response_class=HTMLResponse)
async def mrp_home(request: Request, dep: None = Depends(require_roles('admin','sales','view'))):
    counts = {
        'grades': int(_scalar("SELECT COUNT(*) FROM mrp_grade_master WHERE COALESCE(is_active,TRUE)=TRUE") or 0),
        'recipes': int(_scalar("SELECT COUNT(*) FROM mrp_recipe_master WHERE COALESCE(is_active,TRUE)=TRUE") or 0),
        'yields': int(_scalar("SELECT COUNT(*) FROM mrp_yield_master WHERE COALESCE(is_active,TRUE)=TRUE") or 0),
    }
    dispatchable = _rows("""
        SELECT grade_code, grade_family, stage_type
        FROM mrp_grade_master
        WHERE COALESCE(is_active,TRUE)=TRUE AND COALESCE(dispatchable,FALSE)=TRUE
        ORDER BY stage_type, grade_code
        LIMIT 16
    """)
    return templates.TemplateResponse('mrp_home.html', {
        'request': request,
        'counts': counts,
        'dispatchable': dispatchable,
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
