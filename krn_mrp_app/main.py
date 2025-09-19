import os, io, datetime as dt
from typing import List, Optional

# FastAPI
from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

# SQLAlchemy
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey, func
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

# ---------------------------
# Database config
# ---------------------------
def _normalize_db_url(url: str) -> str:
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if url.startswith("postgresql://") and "+psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url

DATABASE_URL = _normalize_db_url(os.getenv("DATABASE_URL", "sqlite:///./krn_mrp.db"))

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# ---------------------------
# Constants
# ---------------------------
RM_TYPES = ["MS Scrap", "Turnings", "CRC", "TMT end cuts", "FeSi"]

def rm_price_defaults():
    return {"MS Scrap": 34.0, "Turnings": 33.0, "CRC": 40.0, "TMT end cuts": 37.0, "FeSi": 104.0}

# ---------------------------
# Models
# ---------------------------
class GRN(Base):
    __tablename__ = "grn"
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    supplier = Column(String, nullable=False)
    rm_type = Column(String, nullable=False)
    qty = Column(Float, nullable=False)
    remaining_qty = Column(Float, nullable=False)
    price = Column(Float, nullable=False)

class Heat(Base):
    __tablename__ = "heat"
    id = Column(Integer, primary_key=True)
    heat_no = Column(String, unique=True, index=True)
    notes = Column(String)
    slag_qty = Column(Float, default=0)
    total_inputs = Column(Float, default=0)
    actual_output = Column(Float, default=0)
    theoretical = Column(Float, default=0)
    qa_status = Column(String, default="PENDING")
    qa_remarks = Column(String)

    rm_consumptions = relationship("HeatRM", back_populates="heat", cascade="all, delete-orphan")
    chemistry = relationship("HeatChem", uselist=False, back_populates="heat")

class HeatRM(Base):
    __tablename__ = "heat_rm"
    id = Column(Integer, primary_key=True)
    heat_id = Column(Integer, ForeignKey("heat.id"))
    rm_type = Column(String, nullable=False)
    grn_id = Column(Integer, ForeignKey("grn.id"))
    qty = Column(Float, nullable=False)

    heat = relationship("Heat", back_populates="rm_consumptions")
    grn = relationship("GRN")

class HeatChem(Base):
    __tablename__ = "heat_chem"
    id = Column(Integer, primary_key=True)
    heat_id = Column(Integer, ForeignKey("heat.id"))
    c = Column(String); si = Column(String); s = Column(String); p = Column(String)
    cu = Column(String); ni = Column(String); mn = Column(String); fe = Column(String)
    heat = relationship("Heat", back_populates="chemistry")

class Lot(Base):
    __tablename__ = "lot"
    id = Column(Integer, primary_key=True)
    lot_no = Column(String, unique=True, index=True)
    weight = Column(Float, default=3000.0)
    grade = Column(String)  # KRIP / KRFS
    qa_status = Column(String, default="PENDING")
    qa_remarks = Column(String)

    heats = relationship("LotHeat", back_populates="lot", cascade="all, delete-orphan")
    chemistry = relationship("LotChem", uselist=False, back_populates="lot")
    phys = relationship("LotPhys", uselist=False, back_populates="lot")
    psd = relationship("LotPSD", uselist=False, back_populates="lot")

class LotHeat(Base):
    __tablename__ = "lot_heat"
    id = Column(Integer, primary_key=True)
    lot_id = Column(Integer, ForeignKey("lot.id"))
    heat_id = Column(Integer, ForeignKey("heat.id"))
    lot = relationship("Lot", back_populates="heats")
    heat = relationship("Heat")

class LotChem(Base):
    __tablename__ = "lot_chem"
    id = Column(Integer, primary_key=True)
    lot_id = Column(Integer, ForeignKey("lot.id"))
    c = Column(String); si = Column(String); s = Column(String); p = Column(String)
    cu = Column(String); ni = Column(String); mn = Column(String); fe = Column(String)
    lot = relationship("Lot", back_populates="chemistry")

class LotPhys(Base):
    __tablename__ = "lot_phys"
    id = Column(Integer, primary_key=True)
    lot_id = Column(Integer, ForeignKey("lot.id"))
    ad = Column(String); flow = Column(String)
    lot = relationship("Lot", back_populates="phys")

class LotPSD(Base):
    __tablename__ = "lot_psd"
    id = Column(Integer, primary_key=True)
    lot_id = Column(Integer, ForeignKey("lot.id"))
    p212 = Column(String); p180 = Column(String); n180p150 = Column(String)
    n150p75 = Column(String); n75p45 = Column(String); n45 = Column(String)
    lot = relationship("Lot", back_populates="psd")

# ---------------------------
# App + Templates
# ---------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "..", "templates"))

# ---------------------------
# DB dependency
# ---------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------------------
# Home + Setup
# ---------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/setup")
def setup(db: SessionLocal = Depends(get_db)):
    Base.metadata.create_all(bind=engine)
    return HTMLResponse('Tables created and ready. Go to <a href="/grn">GRN</a>.')

# ---------------------------
# GRN
# ---------------------------
@app.get("/grn", response_class=HTMLResponse)
def grn_list(request: Request, db: SessionLocal = Depends(get_db)):
    grns = db.query(GRN).order_by(GRN.id.desc()).all()
    return templates.TemplateResponse("grn.html", {"request": request, "grns": grns, "prices": rm_price_defaults()})

@app.get("/grn/new", response_class=HTMLResponse)
def grn_new(request: Request):
    return templates.TemplateResponse("grn_new.html", {"request": request, "rm_types": RM_TYPES})

@app.post("/grn/new")
def grn_new_post(
    date: str = Form(...), supplier: str = Form(...), rm_type: str = Form(...),
    qty: float = Form(...), price: float = Form(...), db: SessionLocal = Depends(get_db)
):
    g = GRN(date=dt.date.fromisoformat(date), supplier=supplier, rm_type=rm_type,
            qty=qty, remaining_qty=qty, price=price)
    db.add(g); db.commit()
    return RedirectResponse("/grn", status_code=303)

# ---------------------------
# Stock helpers (FIFO)
# ---------------------------
def available_stock(db, rm_type: str):
    rows = db.query(GRN).filter(GRN.rm_type == rm_type, GRN.remaining_qty > 0).order_by(GRN.id.asc()).all()
    return sum(r.remaining_qty for r in rows)

def consume_fifo(db, rm_type: str, qty_needed: float, heat: Heat):
    rows = db.query(GRN).filter(GRN.rm_type == rm_type, GRN.remaining_qty > 0).order_by(GRN.id.asc()).all()
    remaining = qty_needed
    for r in rows:
        if remaining <= 0:
            break
        take = min(r.remaining_qty, remaining)
        if take > 0:
            r.remaining_qty -= take
            db.add(HeatRM(heat=heat, rm_type=rm_type, grn_id=r.id, qty=take))
            remaining -= take
    if remaining > 1e-6:
        raise ValueError(f"Insufficient {rm_type} stock by {remaining:.1f} kg")

# ---------------------------
# Melting
# ---------------------------
@app.get("/melting", response_class=HTMLResponse)
def melting_page(request: Request, db: SessionLocal = Depends(get_db)):
    pending = db.query(Heat).order_by(Heat.id.desc()).all()
    return templates.TemplateResponse("melting.html", {"request": request, "rm_types": RM_TYPES, "pending": pending})

@app.post("/melting/new")
def melting_new(
    request: Request,
    notes: Optional[str] = Form(None),
    slag_qty: float = Form(0.0),
    db: SessionLocal = Depends(get_db),
    rm_type_1: Optional[str] = Form(None), rm_qty_1: Optional[float] = Form(None),
    rm_type_2: Optional[str] = Form(None), rm_qty_2: Optional[float] = Form(None),
    rm_type_3: Optional[str] = Form(None), rm_qty_3: Optional[float] = Form(None),
    rm_type_4: Optional[str] = Form(None), rm_qty_4: Optional[float] = Form(None),
):
    lines = [(t, float(q)) for t, q in [(rm_type_1, rm_qty_1), (rm_type_2, rm_qty_2), (rm_type_3, rm_qty_3), (rm_type_4, rm_qty_4)]
             if t and q and q > 0]
    if not lines:
        return PlainTextResponse("At least one RM line is required.", status_code=400)

    today = dt.date.today().strftime("%Y%m%d")
    seq = (db.query(func.count(Heat.id)).filter(Heat.heat_no.like(f"{today}-%")).scalar() or 0) + 1
    heat_no = f"{today}-{seq:03d}"
    heat = Heat(heat_no=heat_no, notes=notes or "", slag_qty=slag_qty)
    db.add(heat); db.flush()

    total_inputs = 0.0
    for t, q in lines:
        if available_stock(db, t) < q - 1e-6:
            db.rollback()
            return PlainTextResponse(
                f"Insufficient stock for {t}. Available {available_stock(db, t):.1f} kg", status_code=400
            )
    for t, q in lines:
        consume_fifo(db, t, q, heat)
        total_inputs += q

    heat.total_inputs = total_inputs
    heat.actual_output = total_inputs - (slag_qty or 0.0)
    heat.theoretical = total_inputs * 0.97
    db.commit()
    return RedirectResponse("/melting", status_code=303)

# ---------------------------
# QA Redirect (so /qa works)
# ---------------------------
@app.get("/qa")
def qa_redirect():
    return RedirectResponse("/qa-dashboard", status_code=303)

# ---------------------------
# QA Heat (auto-create chemistry)
# ---------------------------
@app.get("/qa/heat/{heat_id}", response_class=HTMLResponse)
def qa_heat_form(heat_id: int, request: Request, db: SessionLocal = Depends(get_db)):
    heat = db.get(Heat, heat_id)
    if not heat:
        return PlainTextResponse("Heat not found", status_code=404)

    chem = heat.chemistry
    if not chem:
        chem = HeatChem(heat=heat)
        db.add(chem); db.commit(); db.refresh(chem)

    return templates.TemplateResponse("qa_heat.html", {"request": request, "heat": heat, "chem": chem})

@app.post("/qa/heat/{heat_id}")
def qa_heat_save(
    heat_id: int, C: str = Form(""), Si: str = Form(""), S: str = Form(""), P: str = Form(""),
    Cu: str = Form(""), Ni: str = Form(""), Mn: str = Form(""), Fe: str = Form(""),
    decision: str = Form("APPROVED"), remarks: str = Form(""),
    db: SessionLocal = Depends(get_db),
):
    heat = db.get(Heat, heat_id)
    chem = heat.chemistry or HeatChem(heat=heat)
    chem.c = C; chem.si = Si; chem.s = S; chem.p = P
    chem.cu = Cu; chem.ni = Ni; chem.mn = Mn; chem.fe = Fe
    heat.qa_status = decision; heat.qa_remarks = remarks
    db.add_all([chem, heat]); db.commit()
    return RedirectResponse("/melting", status_code=303)

# ---------------------------
# Atomization
# ---------------------------
@app.get("/atomization", response_class=HTMLResponse)
def atom_page(request: Request, db: SessionLocal = Depends(get_db)):
    heats = db.query(Heat).filter(Heat.qa_status == "APPROVED").order_by(Heat.id.desc()).all()
    lots = db.query(Lot).order_by(Lot.id.desc()).all()
    return templates.TemplateResponse("atomization.html", {"request": request, "heats": heats, "lots": lots})

@app.post("/atomization/new")
def atom_new(lot_weight: float = Form(3000.0), includes_fesi: str = Form("no"),
             heat_ids: List[int] = Form([]), db: SessionLocal = Depends(get_db)):
    if not heat_ids:
        return PlainTextResponse("Select at least one heat", status_code=400)

    today = dt.date.today().strftime("%Y%m%d")
    seq = (db.query(func.count(Lot.id)).filter(Lot.lot_no.like(f"KR%{today}%")).scalar() or 0) + 1
    grade = "KRFS" if includes_fesi.lower() == "yes" else "KRIP"
    lot_no = f"{grade}-{today}-{seq:03d}"
    lot = Lot(lot_no=lot_no, weight=lot_weight, grade=grade)
    db.add(lot); db.flush()

    for hid in heat_ids:
        db.add(LotHeat(lot_id=lot.id, heat_id=int(hid)))

    heats = db.query(Heat).filter(Heat.id.in_(heat_ids)).all()
    vals = {k: [] for k in ["c", "si", "s", "p", "cu", "ni", "mn", "fe"]}
    for h in heats:
        ch = h.chemistry
        if not ch:
            continue
        for key in vals.keys():
            v = getattr(ch, key)
            try:
                vals[key].append(float(v))
            except:
                pass
    avg = {k: (sum(v) / len(v) if v else None) for k, v in vals.items()}
    lc = LotChem(lot=lot, **{k: (str(v) if v is not None else "") for k, v in avg.items()})
    db.add(lc); db.commit()
    return RedirectResponse("/atomization", status_code=303)

# ---------------------------
# QA Lot (auto-create chem/phys/psd)
# ---------------------------
@app.get("/qa/lot/{lot_id}", response_class=HTMLResponse)
def qa_lot_form(lot_id: int, request: Request, db: SessionLocal = Depends(get_db)):
    lot = db.get(Lot, lot_id)
    if not lot:
        return PlainTextResponse("Lot not found", status_code=404)

    chem = lot.chemistry
    phys = lot.phys
    psd  = lot.psd
    created = False
    if not chem:
        chem = LotChem(lot=lot); db.add(chem); created = True
    if not phys:
        phys = LotPhys(lot=lot); db.add(phys); created = True
    if not psd:
        psd = LotPSD(lot=lot); db.add(psd); created = True
    if created:
        db.commit()
        db.refresh(lot)

    psd_map = {
        "+212": psd.p212 or "", "+180": psd.p180 or "", "-180+150": psd.n180p150 or "",
        "-150+75": psd.n150p75 or "", "-75+45": psd.n75p45 or "", "-45": psd.n45 or ""
    }
    return templates.TemplateResponse(
        "qa_lot.html",
        {"request": request, "lot": lot, "chem": chem, "phys": phys, "psd": psd_map}
    )

@app.post("/qa/lot/{lot_id}")
def qa_lot_save(
    lot_id: int,
    C: str = Form(""), Si: str = Form(""), S: str = Form(""), P: str = Form(""),
    Cu: str = Form(""), Ni: str = Form(""), Mn: str = Form(""), Fe: str = Form(""),
    ad: str = Form(""), flow: str = Form(""),
    p212: str = Form(""), p180: str = Form(""),
    n180p150: str = Form(""), n150p75: str = Form(""),
    n75p45: str = Form(""), n45: str = Form(""),
    decision: str = Form("APPROVED"), remarks: str = Form(""),
    db: SessionLocal = Depends(get_db),
):
    lot = db.get(Lot, lot_id)
    if not lot:
        return PlainTextResponse("Lot not found", status_code=404)

    chem = lot.chemistry or LotChem(lot=lot)
    chem.c = C; chem.si = Si; chem.s = S; chem.p = P
    chem.cu = Cu; chem.ni = Ni; chem.mn = Mn; chem.fe = Fe

    phys = lot.phys or LotPhys(lot=lot)
    phys.ad = ad; phys.flow = flow

    psd = lot.psd or LotPSD(lot=lot)
    psd.p212 = p212; psd.p180 = p180
    psd.n180p150 = n180p150; psd.n150p75 = n150p75
    psd.n75p45 = n75p45; psd.n45 = n45

    lot.qa_status = decision
    lot.qa_remarks = remarks

    db.add_all([chem, phys, psd, lot])
    db.commit()
    return RedirectResponse("/atomization", status_code=303)

# ---------------------------
# QA Dashboard
# ---------------------------
@app.get("/qa-dashboard", response_class=HTMLResponse)
def qa_dashboard(request: Request, db: SessionLocal = Depends(get_db)):
    heats = db.query(Heat).order_by(Heat.id.desc()).all()
    lots = db.query(Lot).order_by(Lot.id.desc()).all()
    return templates.TemplateResponse("qa_dashboard.html", {"request": request, "heats": heats, "lots": lots})

# ---------------------------
# Traceability
# ---------------------------
@app.get("/traceability/lot/{lot_id}", response_class=HTMLResponse)
def trace_lot(lot_id: int, request: Request, db: SessionLocal = Depends(get_db)):
    lot = db.get(Lot, lot_id)
    heats = [lh.heat for lh in lot.heats]
    rows = []
    for h in heats:
        for cons in h.rm_consumptions:
            rows.append(
                type(
                    "Row",
                    (),
                    {
                        "heat_no": h.heat_no,
                        "rm_type": cons.rm_type,
                        "grn_id": cons.grn_id,
                        "supplier": cons.grn.supplier if cons.grn else "",
                        "qty": cons.qty,
                    },
                )
            )
    return templates.TemplateResponse("trace_lot.html", {"request": request, "lot": lot, "heats": heats, "grn_rows": rows})

# ---------------------------
# PDF
# ---------------------------
def draw_header(c: canvas.Canvas, title: str):
    width, height = A4
    logo_path = os.path.join(os.path.dirname(__file__), "..", "static", "KRN_Logo.png")
    if os.path.exists(logo_path):
        c.drawImage(logo_path, 1.5 * cm, height - 3 * cm, width=4 * cm, preserveAspectRatio=True, mask="auto")
    c.setFont("Helvetica-Bold", 14); c.drawString(7 * cm, height - 2 * cm, "KRN Alloys Pvt Ltd")
    c.setFont("Helvetica-Bold", 12); c.drawString(7 * cm, height - 2.7 * cm, title)
    c.line(1.5 * cm, height - 3.3 * cm, width - 1.5 * cm, height - 3.3 * cm)

@app.get("/pdf/lot/{lot_id}")
def pdf_lot(lot_id: int, db: SessionLocal = Depends(get_db)):
    lot = db.get(Lot, lot_id)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    draw_header(c, f"Traceability Report – Lot {lot.lot_no}")

    y = height - 4 * cm
    c.setFont("Helvetica", 11)
    c.drawString(2 * cm, y, f"Grade: {lot.grade}"); y -= 14
    c.drawString(2 * cm, y, f"Weight: {lot.weight:.1f} kg"); y -= 14
    c.drawString(2 * cm, y, f"Lot QA: {lot.qa_status}"); y -= 18

    c.setFont("Helvetica-Bold", 11); c.drawString(2 * cm, y, "Heats"); y -= 14
    c.setFont("Helvetica", 10)
    for lh in lot.heats:
        h = lh.heat
        c.drawString(2.2 * cm, y, f"{h.heat_no}  | Out: {h.actual_output:.1f} kg | QA: {h.qa_status}")
        y -= 12
        if y < 3 * cm:
            c.showPage(); draw_header(c, f"Traceability Report – Lot {lot.lot_no}"); y = height - 4 * cm

    y -= 6
    c.setFont("Helvetica-Bold", 11); c.drawString(2 * cm, y, "GRN Consumption (FIFO)"); y -= 14
    c.setFont("Helvetica", 10)
    for lh in lot.heats:
        h = lh.heat
        for cons in h.rm_consumptions:
            g = cons.grn
            c.drawString(2.2 * cm, y, f"Heat {h.heat_no} | {cons.rm_type} | GRN #{cons.grn_id} | {g.supplier if g else ''} | {cons.qty:.1f} kg")
            y -= 12
            if y < 3 * cm:
                c.showPage(); draw_header(c, f"Traceability Report – Lot {lot.lot_no}"); y = height - 4 * cm

    c.showPage(); c.save()
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="trace_{lot.lot_no}.pdf"'}
    )
