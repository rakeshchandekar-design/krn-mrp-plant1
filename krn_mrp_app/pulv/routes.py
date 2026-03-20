from datetime import date, timedelta, datetime
import io, csv
from fastapi import APIRouter, Depends, Request
from fastapi import Form
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.templating import Jinja2Templates
from sqlalchemy import text
import json
from krn_mrp_app.deps import engine, require_roles

router = APIRouter()
templates = Jinja2Templates(directory="templates")
PULV_PROCESS_COST_PER_KG = 3.0
PULV_DUST_LOSS_PCT = 3.0

with engine.begin() as conn:
    conn.execute(text("CREATE TABLE IF NOT EXISTS pulv_downtime (id SERIAL PRIMARY KEY, date DATE NOT NULL, minutes INTEGER NOT NULL, area TEXT NOT NULL, reason TEXT NOT NULL)"))

def current_role(request):
    try:
        if hasattr(request, "session"):
            role = request.session.get("role")
            if role:
                request.state.role = role
                return role
    except Exception:
        pass
    role = getattr(request.state, "role", None) or request.cookies.get("role") or "guest"
    request.state.role = role
    return role

def _is_admin(request: Request) -> bool:
    s = getattr(request, "state", None)
    if not s:
        return False
    if getattr(s, "is_admin", False):
        return True
    role = getattr(s, "role", None)
    return isinstance(role, str) and role.lower() == "admin"

def fetch_dri_balance():
    with engine.begin() as conn:
        grns = conn.execute(text("""
            SELECT id, grn_no, date, supplier, COALESCE(qty,0)::float qty,
                   COALESCE(remaining_qty,0)::float remaining_qty,
                   COALESCE(price,0)::float price
            FROM grn
            WHERE rm_type='DRI' AND COALESCE(remaining_qty,0) > 0
            ORDER BY date, id
        """)).mappings().all()
    return grns

def _used_by_grn():
    used = {}
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT src_grn_json FROM pulv_lots")).all()
    for (raw,) in rows:
        if not raw:
            continue
        try:
            d = json.loads(raw)
            for k, v in d.items():
                used[int(k)] = used.get(int(k), 0.0) + float(v or 0.0)
        except Exception:
            pass
    return used

@router.get('/', response_class=HTMLResponse)
def home(request: Request, dep: None = Depends(require_roles('admin','pulv','view'))):
    today = date.today()
    with engine.begin() as conn:
        lots_today = conn.execute(text("SELECT COUNT(*) FROM pulv_lots WHERE date=:d"), {'d': today}).scalar() or 0
        produced_today = conn.execute(text("SELECT COALESCE(SUM(mag_output_qty_kg),0) FROM pulv_lots WHERE date=:d"), {'d': today}).scalar() or 0.0
        avg_cost_today = conn.execute(text("SELECT COALESCE(SUM(mag_output_qty_kg*cost_per_kg)/NULLIF(SUM(mag_output_qty_kg),0),0) FROM pulv_lots WHERE date=:d"), {'d': today}).scalar() or 0.0
        last5 = conn.execute(text("SELECT date, COALESCE(SUM(mag_output_qty_kg),0) qty FROM pulv_lots WHERE date >= :d GROUP BY date ORDER BY date DESC"), {'d': today - timedelta(days=4)}).mappings().all()
        live_stock = conn.execute(text("SELECT grade, COALESCE(SUM(mag_output_qty_kg),0) qty FROM pulv_lots WHERE COALESCE(qa_status,'')='APPROVED' GROUP BY grade ORDER BY grade")).mappings().all()
        dt_today = conn.execute(text("SELECT COALESCE(SUM(minutes),0) FROM pulv_downtime WHERE date=:d"), {'d': today}).scalar() or 0
        dt_top = conn.execute(text("SELECT date, area, reason, minutes FROM pulv_downtime ORDER BY date DESC, id DESC LIMIT 10")).mappings().all()
    return templates.TemplateResponse('pulv_home.html', {
        'request': request, 'role': current_role(request), 'is_admin': _is_admin(request),
        'lots_today': lots_today, 'produced_today': produced_today,
        'avg_cost_today': avg_cost_today, 'last5': last5, 'live_stock': live_stock,
        'dt_today': int(dt_today or 0), 'dt_top': dt_top
    })

@router.get('/create', response_class=HTMLResponse)
def create_get(request: Request, dep: None = Depends(require_roles('admin','pulv'))):
    used = _used_by_grn()
    rows = []
    for r in fetch_dri_balance():
        avail = float(r['remaining_qty'] or 0) - float(used.get(int(r['id']), 0.0))
        if avail > 0.0001:
            x = dict(r)
            x['available_kg'] = avail
            rows.append(x)
    return templates.TemplateResponse('pulv_create.html', {
        'request': request, 'role': current_role(request), 'is_admin': _is_admin(request),
        'rows': rows, 'err': request.query_params.get('err','')
    })

@router.post('/create')
async def create_post(request: Request, dep: None = Depends(require_roles('admin','pulv'))):
    form = await request.form()
    allocs = {}
    total = 0.0
    for k, v in form.items():
        if k.startswith('alloc_'):
            try:
                q = float(v or 0)
            except Exception:
                q = 0.0
            if q > 0:
                allocs[int(k.replace('alloc_',''))] = q
                total += q
    if total <= 0:
        return RedirectResponse('/pulv/create?err=Allocate+DRI+quantity+%3E+0', status_code=303)
    try:
        mag_output = float(form.get('mag_output_qty_kg') or 0)
    except Exception:
        mag_output = 0.0
    if mag_output <= 0:
        return RedirectResponse('/pulv/create?err=Enter+magnetic+output+qty', status_code=303)

    used = _used_by_grn()
    avail = {r['id']: r for r in fetch_dri_balance()}
    input_value = 0.0
    trace_parts = []
    for gid, q in allocs.items():
        r = avail.get(gid)
        if not r:
            return RedirectResponse('/pulv/create?err=Invalid+DRI+GRN', status_code=303)
        effective = float(r['remaining_qty'] or 0) - float(used.get(gid, 0.0))
        if q > effective + 1e-9:
            return RedirectResponse('/pulv/create?err=Insufficient+DRI+balance', status_code=303)
        input_value += q * float(r['price'] or 0.0)
        trace_parts.append(r['grn_no'])

    dust_loss_qty = total * (PULV_DUST_LOSS_PCT / 100.0)
    feed_qty = total - dust_loss_qty
    if mag_output > feed_qty + 1e-9:
        return RedirectResponse('/pulv/create?err=Magnetic+output+cannot+exceed+feed+after+dust+loss', status_code=303)

    non_mag_qty = feed_qty - mag_output
    non_mag_pct = (non_mag_qty / feed_qty * 100.0) if feed_qty > 0 else 0.0
    cost_per_kg = (input_value + total * PULV_PROCESS_COST_PER_KG) / mag_output if mag_output > 0 else 0.0

    with engine.begin() as conn:
        prefix = 'PULV-' + date.today().strftime('%Y%m%d') + '-'
        last = conn.execute(text("SELECT lot_no FROM pulv_lots WHERE lot_no LIKE :pfx ORDER BY lot_no DESC LIMIT 1"), {'pfx': f'{prefix}%'}).scalar()
        seq = int(last.split('-')[-1]) + 1 if last else 1
        lot_no = f'{prefix}{seq:03d}'
        conn.execute(text("""
            INSERT INTO pulv_lots
            (date, lot_no, src_grn_json, grade, input_qty_kg, input_cost_per_kg, process_cost_per_kg,
             dust_loss_pct, dust_loss_qty, feed_qty_kg, mag_output_qty_kg, non_mag_qty_kg, non_mag_pct,
             cost_per_kg, qa_status, trace_id, job_card_no)
            VALUES
            (:date,:lot_no,:src_grn_json,'KRSP',:input_qty_kg,:input_cost_per_kg,:process_cost_per_kg,
             :dust_loss_pct,:dust_loss_qty,:feed_qty_kg,:mag_output_qty_kg,:non_mag_qty_kg,:non_mag_pct,
             :cost_per_kg,'PENDING',:trace_id,:job_card_no)
        """), {
            'date': date.today(), 'lot_no': lot_no, 'src_grn_json': json.dumps(allocs),
            'input_qty_kg': total,
            'input_cost_per_kg': (input_value / total if total > 0 else 0.0),
            'process_cost_per_kg': PULV_PROCESS_COST_PER_KG,
            'dust_loss_pct': PULV_DUST_LOSS_PCT,
            'dust_loss_qty': dust_loss_qty,
            'feed_qty_kg': feed_qty,
            'mag_output_qty_kg': mag_output,
            'non_mag_qty_kg': non_mag_qty,
            'non_mag_pct': non_mag_pct,
            'cost_per_kg': cost_per_kg,
            'trace_id': '+'.join(trace_parts)[:240],
            'job_card_no': f'JC-PULV-{lot_no}',
        })
    return RedirectResponse('/pulv/lots', status_code=303)

@router.get('/lots', response_class=HTMLResponse)
def lots(request: Request, dep: None = Depends(require_roles('admin','pulv','view'))):
    with engine.begin() as conn:
        rows = conn.execute(text('SELECT * FROM pulv_lots ORDER BY date DESC, id DESC')).mappings().all()
    return templates.TemplateResponse('pulv_lot_list.html', {'request': request, 'role': current_role(request), 'rows': rows, 'is_admin': _is_admin(request)})

@router.get('/qa/{lot_id}', response_class=HTMLResponse)
def qa_get(lot_id: int, request: Request, dep: None = Depends(require_roles('admin','qa','pulv'))):
    with engine.begin() as conn:
        lot = conn.execute(text('SELECT * FROM pulv_lots WHERE id=:id'), {'id': lot_id}).mappings().first()
        qa = conn.execute(text('SELECT * FROM pulv_qa WHERE pulv_lot_id=:id ORDER BY id DESC LIMIT 1'), {'id': lot_id}).mappings().first()
        params = conn.execute(text('SELECT param_name, param_value FROM pulv_qa_params WHERE pulv_qa_id=:qid ORDER BY id'), {'qid': (qa or {}).get('id', 0)}).mappings().all() if qa else []
    pmap = {p['param_name']: p['param_value'] for p in params}
    return templates.TemplateResponse('pulv_qa_form.html', {'request': request, 'role': current_role(request), 'lot': lot, 'qa': qa, 'pmap': pmap})

@router.post('/qa/{lot_id}')
async def qa_post(lot_id: int, request: Request, dep: None = Depends(require_roles('admin','qa','pulv'))):
    form = await request.form()
    decision = str(form.get('decision') or 'PENDING').upper()
    ad = float(form.get('ad') or 0)
    non_mag_pct = float(form.get('non_mag_pct') or 0)
    remarks = str(form.get('remarks') or '')
    psd = {k: form.get(k) or '' for k in ['p212','p180','n180p150','n150p75','n75p45','n45']}
    with engine.begin() as conn:
        qid = conn.execute(text("INSERT INTO pulv_qa (pulv_lot_id, decision, remarks, ad, non_mag_pct) VALUES (:lot,:d,:r,:ad,:n) RETURNING id"), {'lot': lot_id, 'd': decision, 'r': remarks, 'ad': ad, 'n': non_mag_pct}).scalar()
        for k, v in psd.items():
            conn.execute(text("INSERT INTO pulv_qa_params (pulv_qa_id, param_name, param_value) VALUES (:qid,:n,:v)"), {'qid': qid, 'n': k, 'v': str(v)})
        conn.execute(text("UPDATE pulv_lots SET qa_status=:d, qa_remarks=:r WHERE id=:id"), {'d': decision, 'r': remarks, 'id': lot_id})
    return RedirectResponse('/pulv/lots', status_code=303)


@router.get('/downtime', response_class=HTMLResponse)
def pulv_downtime_get(request: Request, dep: None = Depends(require_roles('admin','pulv'))):
    today = date.today().isoformat()
    min_date = (date.today() - timedelta(days=90)).isoformat()
    with engine.begin() as conn:
        logs = conn.execute(text("SELECT date, minutes, area, reason FROM pulv_downtime ORDER BY date DESC, id DESC LIMIT 200")).mappings().all()
    return templates.TemplateResponse('pulv_downtime.html', {
        'request': request, 'today': today, 'min_date': min_date, 'logs': logs,
        'err': request.query_params.get('err',''), 'role': current_role(request), 'is_admin': _is_admin(request)
    })

@router.post('/downtime')
async def pulv_downtime_post(request: Request, dep: None = Depends(require_roles('admin','pulv'))):
    form = await request.form()
    try:
        d = datetime.strptime(str(form.get('date') or ''), '%Y-%m-%d').date()
    except Exception:
        return RedirectResponse('/pulv/downtime?err=Invalid+date', status_code=303)
    mins = int(form.get('minutes') or 0)
    area = str(form.get('area') or '').strip()
    reason = str(form.get('reason') or '').strip()
    if mins <= 0:
        return RedirectResponse('/pulv/downtime?err=Minutes+must+be+%3E+0', status_code=303)
    if not area or not reason:
        return RedirectResponse('/pulv/downtime?err=Type+and+Reason+are+required', status_code=303)
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO pulv_downtime(date, minutes, area, reason) VALUES (:d,:m,:a,:r)"), {'d': d, 'm': mins, 'a': area, 'r': reason})
    return RedirectResponse('/pulv/downtime', status_code=303)

@router.get('/downtime.csv')
def pulv_downtime_csv(dep: None = Depends(require_roles('admin','pulv','view'))):
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT date, minutes, area, reason FROM pulv_downtime ORDER BY date DESC, id DESC")).mappings().all()
    out = io.StringIO(); w = csv.writer(out)
    w.writerow(['Date','Minutes','Type','Reason'])
    for r in rows:
        w.writerow([r['date'], r['minutes'], r['area'], r['reason']])
    return HTMLResponse(content=out.getvalue(), media_type='text/csv', headers={'Content-Disposition':'attachment; filename=pulv_downtime.csv'})
