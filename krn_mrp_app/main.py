from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from datetime import datetime
from .db import Base, engine, SessionLocal
from . import models

app = FastAPI(title="KRN MRP Plant1 – Full DB")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def root():
    return {"msg": "KRN MRP Plant1 – Full DB Ready"}

@app.get("/setup", response_class=HTMLResponse)
def setup(request: Request, db: Session = Depends(get_db)):
    Base.metadata.create_all(bind=engine)
    defaults = {"MS Scrap":34.0,"Turnings":32.0,"CRC":40.0,"TMT end cuts":37.0,"FeSi":104.0}
    for rm, price in defaults.items():
        rec = db.query(models.RMPrice).filter_by(rm_type=rm).first()
        if not rec:
            db.add(models.RMPrice(rm_type=rm, current_price=price))
    db.commit()
    return templates.TemplateResponse("setup_done.html", {"request": request})

@app.get("/grn", response_class=HTMLResponse)
def grn_list(request: Request, db: Session = Depends(get_db)):
    rows = db.query(models.GRN).order_by(models.GRN.id.desc()).all()
    prices = {p.rm_type: p.current_price for p in db.query(models.RMPrice).all()}
    return templates.TemplateResponse("grn_list.html", {"request": request, "rows": rows, "prices": prices})

@app.get("/grn/new", response_class=HTMLResponse)
def grn_new(request: Request, db: Session = Depends(get_db)):
    rm_types = [p.rm_type for p in db.query(models.RMPrice).all()]
    return templates.TemplateResponse("grn_new.html", {"request": request, "rm_types": rm_types})

@app.post("/grn/new")
def grn_create(rm_type: str = Form(...), qty_kg: float = Form(...), price_per_kg: float = Form(...), supplier: str = Form(""), db: Session = Depends(get_db)):
    amount = qty_kg * price_per_kg
    db.add(models.GRN(date=datetime.utcnow(), supplier=supplier, rm_type=rm_type, qty_kg=qty_kg, price_per_kg=price_per_kg, amount=amount))
    price = db.query(models.RMPrice).filter_by(rm_type=rm_type).first()
    if price:
        price.current_price = price_per_kg
    db.commit()
    return RedirectResponse(url="/grn", status_code=303)

def next_heat_no(db: Session) -> str:
    count = db.query(models.Heat).count() + 1
    return f"H{datetime.utcnow().strftime('%y%m%d')}-{count:04d}"

@app.get("/heat", response_class=HTMLResponse)
def heat_list(request: Request, db: Session = Depends(get_db)):
    heats = db.query(models.Heat).order_by(models.Heat.id.desc()).all()
    return templates.TemplateResponse("heat_list.html", {"request": request, "heats": heats})

@app.get("/heat/new", response_class=HTMLResponse)
def heat_new(request: Request, db: Session = Depends(get_db)):
    rm_types = [p.rm_type for p in db.query(models.RMPrice).all()]
    return templates.TemplateResponse("heat_new.html", {"request": request, "rm_types": rm_types})

@app.post("/heat/new")
def heat_create(rm1: str = Form(...), qty1: float = Form(...),
                rm2: str = Form(""), qty2: float = Form(0.0),
                rm3: str = Form(""), qty3: float = Form(0.0),
                actual_out_kg: float = Form(0.0),
                db: Session = Depends(get_db)):
    inputs = [(rm1, qty1)]
    if rm2 and qty2>0: inputs.append((rm2, qty2))
    if rm3 and qty3>0: inputs.append((rm3, qty3))
    total_in = sum(q for _, q in inputs)
    theo = round(total_in * 0.97, 2)
    grade = "KRFS" if any(rm == "FeSi" for rm, _ in inputs) else "KRIP"
    heat = models.Heat(heat_no=next_heat_no(db), grade=grade, input_total_kg=total_in, theoretical_out_kg=theo, actual_out_kg=actual_out_kg or 0.0)
    db.add(heat); db.flush()
    for rm, q in inputs:
        db.add(models.HeatInput(heat_id=heat.id, rm_type=rm, qty_kg=q))
    db.commit()
    return RedirectResponse(url="/heat", status_code=303)

@app.get("/heat/{heat_id}", response_class=HTMLResponse)
def heat_detail(heat_id: int, request: Request, db: Session = Depends(get_db)):
    heat = db.get(models.Heat, heat_id)
    if not heat: raise HTTPException(404)
    inp = db.query(models.HeatInput).filter_by(heat_id=heat_id).all()
    hqa = db.query(models.HeatQA).filter_by(heat_id=heat_id).first()
    return templates.TemplateResponse("heat_detail.html", {"request": request, "heat": heat, "inputs": inp, "hqa": hqa})

@app.post("/heat/{heat_id}/qa")
def heat_qa_submit(heat_id: int, c: float = Form(0.0), si: float = Form(0.0), s: float = Form(0.0), p: float = Form(0.0),
                   cu: float = Form(0.0), ni: float = Form(0.0), mn: float = Form(0.0), fe: float = Form(0.0),
                   approve: str = Form("no"), db: Session = Depends(get_db)):
    heat = db.get(models.Heat, heat_id)
    if not heat: raise HTTPException(404)
    hqa = db.query(models.HeatQA).filter_by(heat_id=heat_id).first()
    if not hqa:
        hqa = models.HeatQA(heat_id=heat_id); db.add(hqa)
    hqa.c, hqa.si, hqa.s, hqa.p, hqa.cu, hqa.ni, hqa.mn, hqa.fe = c, si, s, p, cu, ni, mn, fe
    hqa.approved = (approve == "yes")
    heat.qa_status = "APPROVED" if hqa.approved else "REJECTED"
    db.commit()
    return RedirectResponse(url=f"/heat/{heat_id}", status_code=303)

def next_lot_no(db: Session, grade: str) -> str:
    today = datetime.utcnow().strftime("%y%m")
    count = db.query(models.Lot).filter(models.Lot.grade==grade).count() + 1
    return f"{grade}-{today}-{count:03d}"

@app.get("/lot", response_class=HTMLResponse)
def lot_list(request: Request, db: Session = Depends(get_db)):
    lots = db.query(models.Lot).order_by(models.Lot.id.desc()).all()
    return templates.TemplateResponse("lot_list.html", {"request": request, "lots": lots})

@app.get("/lot/new", response_class=HTMLResponse)
def lot_new(request: Request, db: Session = Depends(get_db)):
    heats_krip = db.query(models.Heat).filter(models.Heat.grade=="KRIP", models.Heat.qa_status=="APPROVED").all()
    heats_krfs = db.query(models.Heat).filter(models.Heat.grade=="KRFS", models.Heat.qa_status=="APPROVED").all()
    return templates.TemplateResponse("lot_new.html", {"request": request, "heats_krip": heats_krip, "heats_krfs": heats_krfs})

@app.post("/lot/new")
def lot_create(grade: str = Form(...), heat_ids: str = Form(...), db: Session = Depends(get_db)):
    ids = [int(x) for x in heat_ids.split(",") if x.strip().isdigit()]
    if not ids: raise HTTPException(400, "No heats selected")
    heats = db.query(models.Heat).filter(models.Heat.id.in_(ids), models.Heat.grade==grade, models.Heat.qa_status=="APPROVED").all()
    if not heats: raise HTTPException(400, "No valid heats")
    total = 0.0
    for h in heats:
        total += h.actual_out_kg if h.actual_out_kg>0 else h.theoretical_out_kg
    if total < 3000:
        raise HTTPException(400, "Total kg must be at least 3000")
    lot = models.Lot(lot_no=next_lot_no(db, grade), grade=grade, total_kg=round(total,2))
    db.add(lot); db.flush()
    for h in heats:
        db.add(models.LotHeat(lot_id=lot.id, heat_id=h.id))
    db.commit()
    return RedirectResponse(url=f"/lot/{lot.id}", status_code=303)

@app.get("/lot/{lot_id}", response_class=HTMLResponse)
def lot_detail(lot_id: int, request: Request, db: Session = Depends(get_db)):
    lot = db.get(models.Lot, lot_id)
    if not lot: raise HTTPException(404)
    link_ids = [lh.heat_id for lh in db.query(models.LotHeat).filter_by(lot_id=lot_id)]
    hqas = db.query(models.HeatQA).filter(models.HeatQA.heat_id.in_(link_ids), models.HeatQA.approved==True).all()
    avg = None
    if hqas:
        n = len(hqas)
        def avgf(attr): return round(sum(getattr(x, attr) for x in hqas)/n, 4)
        avg = {k: avgf(k) for k in ["c","si","s","p","cu","ni","mn","fe"]}
    lqa = db.query(models.LotQA).filter_by(lot_id=lot_id).first()
    return templates.TemplateResponse("lot_detail.html", {"request": request, "lot": lot, "avg": avg, "lqa": lqa})

@app.post("/lot/{lot_id}/qa")
def lot_qa_submit(lot_id: int,
                  ad: float = Form(0.0), flow: float = Form(0.0),
                  plus_212: float = Form(0.0), plus_180: float = Form(0.0),
                  m180_p150: float = Form(0.0), m150_p75: float = Form(0.0),
                  m75_p45: float = Form(0.0), m45: float = Form(0.0),
                  approve: str = Form("no"),
                  db: Session = Depends(get_db)):
    lot = db.get(models.Lot, lot_id)
    if not lot: raise HTTPException(404)
    lqa = db.query(models.LotQA).filter_by(lot_id=lot_id).first()
    if not lqa:
        lqa = models.LotQA(lot_id=lot_id); db.add(lqa)
    link_ids = [lh.heat_id for lh in db.query(models.LotHeat).filter_by(lot_id=lot_id)]
    hqas = db.query(models.HeatQA).filter(models.HeatQA.heat_id.in_(link_ids), models.HeatQA.approved==True).all()
    if hqas:
        n = len(hqas)
        def avgf(attr): return round(sum(getattr(x, attr) for x in hqas)/n, 4)
        lqa.c, lqa.si, lqa.p, lqa.cu, lqa.ni, lqa.mn, lqa.fe = avgf("c"), avgf("si"), avgf("p"), avgf("cu"), avgf("ni"), avgf("mn"), avgf("fe")
        lqa.s = avgf("s")
    lqa.ad, lqa.flow = ad, flow
    lqa.plus_212, lqa.plus_180, lqa.m180_p150, lqa.m150_p75, lqa.m75_p45, lqa.m45 = plus_212, plus_180, m180_p150, m150_p75, m75_p45, m45
    lqa.approved = (approve == "yes")
    lot.qa_status = "APPROVED" if lqa.approved else "REJECTED"
    db.commit()
    return RedirectResponse(url=f"/lot/{lot_id}", status_code=303)
