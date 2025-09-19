import os, io, datetime as dt
from typing import List, Optional, Dict, Tuple

# FastAPI / Starlette
from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

# SQLAlchemy
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey, func, text
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session

# -------------------------------------------------
# Costing constants
# -------------------------------------------------
MELT_COST_PER_KG_KRIP = 6.0
MELT_COST_PER_KG_KRFS = 8.0
ATOMIZATION_COST_PER_KG = 5.0
SURCHARGE_PER_KG = 2.0

# -------------------------------------------------
# Very simple users (demo)
# -------------------------------------------------
USERS = {
    # username: (password, role)
    "admin": ("admin", "ADMIN"),
    "rap": ("rap", "RAP"),
    "qa": ("qa", "QA"),
}

# -------------------------------------------------
# Database config
# -------------------------------------------------
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

# -------------------------------------------------
# Schema migration helpers
# -------------------------------------------------
def _table_has_column(conn, table: str, col: str) -> bool:
    if str(engine.url).startswith("sqlite"):
        rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
        names = [r[1] for r in rows]
        return col in names
    else:
        q = text("""
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = :t
              AND column_name = :c
            LIMIT 1
        """)
        return conn.execute(q, {"t": table, "c": col}).first() is not None

def migrate_schema(engine):
    with engine.begin() as conn:
        # Heat costing columns
        for col in ["rm_cost", "process_cost", "total_cost", "unit_cost"]:
            if not _table_has_column(conn, "heat", col):
                if str(engine.url).startswith("sqlite"):
                    conn.execute(text(f"ALTER TABLE heat ADD COLUMN {col} REAL DEFAULT 0"))
                else:
                    conn.execute(text(f"ALTER TABLE heat ADD COLUMN IF NOT EXISTS {col} DOUBLE PRECISION DEFAULT 0"))

        # Heat allocation tracking (partial)
        if not _table_has_column(conn, "heat", "alloc_used"):
            if str(engine.url).startswith("sqlite"):
                conn.execute(text("ALTER TABLE heat ADD COLUMN alloc_used REAL DEFAULT 0"))
            else:
                conn.execute(text("ALTER TABLE heat ADD COLUMN IF NOT EXISTS alloc_used DOUBLE PRECISION DEFAULT 0"))

        # Lot costing columns
        for col in ["unit_cost", "total_cost"]:
            if not _table_has_column(conn, "lot", col):
                if str(engine.url).startswith("sqlite"):
                    conn.execute(text(f"ALTER TABLE lot ADD COLUMN {col} REAL DEFAULT 0"))
                else:
                    conn.execute(text(f"ALTER TABLE lot ADD COLUMN IF NOT EXISTS {col} DOUBLE PRECISION DEFAULT 0"))

        # Lot status (Ready Stock & issue)
        if not _table_has_column(conn, "lot", "status"):
            if str(engine.url).startswith("sqlite"):
                conn.execute(text("ALTER TABLE lot ADD COLUMN status TEXT DEFAULT 'PENDING'"))
            else:
                conn.execute(text("ALTER TABLE lot ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'PENDING'"))
        if not _table_has_column(conn, "lot", "issued_to"):
            if str(engine.url).startswith("sqlite"):
                conn.execute(text("ALTER TABLE lot ADD COLUMN issued_to TEXT DEFAULT ''"))
            else:
                conn.execute(text("ALTER TABLE lot ADD COLUMN IF NOT EXISTS issued_to TEXT DEFAULT ''"))

        # LotHeat qty column (partial allocations)
        if not _table_has_column(conn, "lot_heat", "qty"):
            if str(engine.url).startswith("sqlite"):
                conn.execute(text("ALTER TABLE lot_heat ADD COLUMN qty REAL DEFAULT 0"))
            else:
                conn.execute(text("ALTER TABLE lot_heat ADD COLUMN IF NOT EXISTS qty DOUBLE PRECISION DEFAULT 0"))

# -------------------------------------------------
# Constants
# -------------------------------------------------
RM_TYPES = ["MS Scrap", "Turnings", "CRC", "TMT end cuts", "FeSi"]

def rm_price_defaults():
    return {"MS Scrap": 34.0, "Turnings": 33.0, "CRC": 40.0, "TMT end cuts": 37.0, "FeSi": 104.0}

# -------------------------------------------------
# Models
# -------------------------------------------------
class GRN(Base):
    __tablename__ = "grn"
    id = Column(Integer, primary_key=True)
    grn_no = Column(String, unique=True, index=True)  # NEW: human GRN code
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

    # costing
    rm_cost = Column(Float, default=0.0)
    process_cost = Column(Float, default=0.0)
    total_cost = Column(Float, default=0.0)
    unit_cost = Column(Float, default=0.0)

    # partial allocations
    alloc_used = Column(Float, default=0.0)  # total kg moved to lots

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

    # costing
    unit_cost = Column(Float, default=0.0)
    total_cost = Column(Float, default=0.0)

    # lifecycle
    status = Column(String, default="PENDING")  # PENDING / READY (QA approved) / ISSUED
    issued_to = Column(String, default="")      # DISPATCH / ANNEAL / empty

    heats = relationship("LotHeat", back_populates="lot", cascade="all, delete-orphan")
    chemistry = relationship("LotChem", uselist=False, back_populates="lot")
    phys = relationship("LotPhys", uselist=False, back_populates="lot")
    psd = relationship("LotPSD", uselist=False, back_populates="lot")

class LotHeat(Base):
    __tablename__ = "lot_heat"
    id = Column(Integer, primary_key=True)
    lot_id = Column(Integer, ForeignKey("lot.id"))
    heat_id = Column(Integer, ForeignKey("heat.id"))
    qty = Column(Float, default=0.0)  # allocated kg from the heat into this lot
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

# -------------------------------------------------
# App + Templates
# -------------------------------------------------
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET", "dev-secret"))  # simple session
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "..", "templates"))

# -------------------------------------------------
# DB dependency
# -------------------------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------------------------------------
# Auth helpers
# -------------------------------------------------
def current_role(request: Request) -> str:
    return request.session.get("role", "")

def require_roles(request: Request, *roles: str) -> Optional[RedirectResponse]:
    role = current_role(request)
    if not role or role not in roles:
        return RedirectResponse("/login", status_code=303)
    return None

def is_admin(request: Request) -> bool:
    return current_role(request) == "ADMIN"

def is_rap(request: Request) -> bool:
    return current_role(request) == "RAP"

def is_qa(request: Request) -> bool:
    return current_role(request) == "QA"

# -------------------------------------------------
# Utilities
# -------------------------------------------------
def heat_grade(heat: Heat) -> str:
    for cons in heat.rm_consumptions:
        if cons.rm_type == "FeSi":
            return "KRFS"
    return "KRIP"

def heat_available(db: Session, heat: Heat) -> float:
    used = db.query(func.coalesce(func.sum(LotHeat.qty), 0.0)).filter(LotHeat.heat_id == heat.id).scalar() or 0.0
    heat.alloc_used = float(used)
    return max((heat.actual_output or 0.0) - used, 0.0)

def make_grn_code(db: Session, date: dt.date) -> str:
    yym = date.strftime("%y%m")
    seq = (db.query(func.count(GRN.id)).filter(GRN.grn_no.like(f"GRN-{yym}-%")).scalar() or 0) + 1
    return f"GRN-{yym}-{seq:04d}"

# -------------------------------------------------
# Home + Auth
# -------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request, db: Session = Depends(get_db)):
    Base.metadata.create_all(bind=engine)
    migrate_schema(engine)
    # dashboard metrics (simple)
    today = dt.date.today()
    grn_today = db.query(func.count(GRN.id)).filter(GRN.date == today).scalar() or 0
    heats_today = db.query(func.count(Heat.id)).filter(Heat.heat_no.like(f"{today.strftime('%Y%m%d')}-%")).scalar() or 0
    lots_today = db.query(func.count(Lot.id)).filter(Lot.lot_no.like(f"%{today.strftime('%Y%m%d')}%")).scalar() or 0
    ready = db.query(func.count(Lot.id)).filter(Lot.qa_status == "APPROVED").scalar() or 0
    return templates.TemplateResponse("index.html",
        {"request": request, "role": current_role(request),
         "kpi": {"grn_today": grn_today, "heats_today": heats_today, "lots_today": lots_today, "ready": ready}})

@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username in USERS and USERS[username][0] == password:
        request.session["role"] = USERS[username][1]
        request.session["user"] = username
        return RedirectResponse("/", status_code=303)
    return PlainTextResponse("Invalid credentials", status_code=401)

@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=303)

# -------------------------------------------------
# GRN
# -------------------------------------------------
@app.get("/grn", response_class=HTMLResponse)
def grn_list(request: Request, db: Session = Depends(get_db),
             from_date: Optional[str] = None, to_date: Optional[str] = None, show_all: Optional[int] = 0):
    q = db.query(GRN)
    if from_date:
        q = q.filter(GRN.date >= dt.date.fromisoformat(from_date))
    if to_date:
        q = q.filter(GRN.date <= dt.date.fromisoformat(to_date))
    if not show_all:
        q = q.filter(GRN.remaining_qty > 0)  # hide exhausted
    grns = q.order_by(GRN.id.desc()).all()
    return templates.TemplateResponse("grn.html",
        {"request": request, "role": current_role(request), "grns": grns, "prices": rm_price_defaults(),
         "from_date": from_date or "", "to_date": to_date or "", "show_all": int(show_all)})

@app.get("/grn/new", response_class=HTMLResponse)
def grn_new(request: Request):
    if require_roles(request, "ADMIN", "RAP"):
        return require_roles(request, "ADMIN", "RAP")
    return templates.TemplateResponse("grn_new.html", {"request": request, "rm_types": RM_TYPES, "role": current_role(request)})

@app.post("/grn/new")
def grn_new_post(
    request: Request,
    date: str = Form(...), supplier: str = Form(...), rm_type: str = Form(...),
    qty: float = Form(...), price: float = Form(...), db: Session = Depends(get_db)
):
    if require_roles(request, "ADMIN", "RAP"):
        return require_roles(request, "ADMIN", "RAP")
    d = dt.date.fromisoformat(date)
    g = GRN(grn_no=make_grn_code(db, d), date=d, supplier=supplier, rm_type=rm_type,
            qty=qty, remaining_qty=qty, price=price)
    db.add(g); db.commit()
    return RedirectResponse("/grn", status_code=303)

# -------------------------------------------------
# Stock helpers (FIFO)
# -------------------------------------------------
def available_stock(db: Session, rm_type: str):
    rows = db.query(GRN).filter(GRN.rm_type == rm_type, GRN.remaining_qty > 0).order_by(GRN.id.asc()).all()
    return sum(r.remaining_qty for r in rows)

def consume_fifo(db: Session, rm_type: str, qty_needed: float, heat: Heat) -> float:
    rows = db.query(GRN).filter(GRN.rm_type == rm_type, GRN.remaining_qty > 0).order_by(GRN.id.asc()).all()
    remaining = qty_needed
    added_cost = 0.0
    for r in rows:
        if remaining <= 0:
            break
        take = min(r.remaining_qty, remaining)
        if take > 0:
            r.remaining_qty -= take
            db.add(HeatRM(heat=heat, rm_type=rm_type, grn_id=r.id, qty=take))
            added_cost += take * (r.price or 0.0)
            remaining -= take
    if remaining > 1e-6:
        raise ValueError(f"Insufficient {rm_type} stock by {remaining:.1f} kg")
    return added_cost

# -------------------------------------------------
# Melting
# -------------------------------------------------
@app.get("/melting", response_class=HTMLResponse)
def melting_page(request: Request, db: Session = Depends(get_db),
                 from_date: Optional[str] = None, to_date: Optional[str] = None):
    q = db.query(Heat)
    if from_date:
        q = q.filter(Heat.heat_no >= f"{dt.date.fromisoformat(from_date).strftime('%Y%m%d')}-000")
    if to_date:
        q = q.filter(Heat.heat_no <= f"{dt.date.fromisoformat(to_date).strftime('%Y%m%d')}-999")
    heats = q.order_by(Heat.id.desc()).all()
    grades = {h.id: heat_grade(h) for h in heats}
    return templates.TemplateResponse(
        "melting.html",
        {"request": request, "rm_types": RM_TYPES, "pending": heats, "heat_grades": grades,
         "role": current_role(request), "from_date": from_date or "", "to_date": to_date or ""}
    )

@app.post("/melting/new")
def melting_new(
    request: Request,
    notes: Optional[str] = Form(None),
    slag_qty: float = Form(...),  # mandatory
    db: Session = Depends(get_db),
    rm_type_1: Optional[str] = Form(None), rm_qty_1: Optional[float] = Form(None),
    rm_type_2: Optional[str] = Form(None), rm_qty_2: Optional[float] = Form(None),
    rm_type_3: Optional[str] = Form(None), rm_qty_3: Optional[float] = Form(None),
    rm_type_4: Optional[str] = Form(None), rm_qty_4: Optional[float] = Form(None),
):
    if require_roles(request, "ADMIN", "RAP"):
        return require_roles(request, "ADMIN", "RAP")
    # Need at least 2 RM lines with positive qty
    lines_raw = [(rm_type_1, rm_qty_1), (rm_type_2, rm_qty_2), (rm_type_3, rm_qty_3), (rm_type_4, rm_qty_4)]
    lines = [(t, float(q)) for t, q in lines_raw if t and q and float(q) > 0]
    if len(lines) < 2:
        return PlainTextResponse("At least two raw material lines are required.", status_code=400)

    # Create heat number
    today = dt.date.today().strftime("%Y%m%d")
    seq = (db.query(func.count(Heat.id)).filter(Heat.heat_no.like(f"{today}-%")).scalar() or 0) + 1
    heat_no = f"{today}-{seq:03d}"
    heat = Heat(heat_no=heat_no, notes=notes or "", slag_qty=slag_qty)
    db.add(heat); db.flush()

    # Check stock first
    for t, q in lines:
        if available_stock(db, t) < q - 1e-6:
            db.rollback()
            return PlainTextResponse(
                f"Insufficient stock for {t}. Available {available_stock(db, t):.1f} kg",
                status_code=400
            )

    # Consume FIFO + accumulate RM cost
    total_inputs = 0.0
    total_rm_cost = 0.0
    used_fesi = False
    for t, q in lines:
        if t == "FeSi":
            used_fesi = True
        total_rm_cost += consume_fifo(db, t, q, heat)
        total_inputs += q

    heat.total_inputs = total_inputs
    heat.actual_output = total_inputs - (slag_qty or 0.0)
    heat.theoretical = total_inputs * 0.97  # 3% theoretical loss

    melt_cost_per_kg = MELT_COST_PER_KG_KRFS if used_fesi else MELT_COST_PER_KG_KRIP
    if (heat.actual_output or 0) > 0:
        heat.rm_cost = total_rm_cost
        heat.process_cost = melt_cost_per_kg * heat.actual_output
        heat.total_cost = heat.rm_cost + heat.process_cost
        heat.unit_cost = heat.total_cost / heat.actual_output
    else:
        heat.rm_cost = total_rm_cost
        heat.process_cost = 0.0
        heat.total_cost = total_rm_cost
        heat.unit_cost = 0.0

    db.commit()
    return RedirectResponse("/melting", status_code=303)

# -------------------------------------------------
# QA Redirect
# -------------------------------------------------
@app.get("/qa")
def qa_redirect():
    return RedirectResponse("/qa-dashboard", status_code=303)

# -------------------------------------------------
# QA Heat
# -------------------------------------------------
@app.get("/qa/heat/{heat_id}", response_class=HTMLResponse)
def qa_heat_form(heat_id: int, request: Request, db: Session = Depends(get_db)):
    if require_roles(request, "ADMIN", "QA"):
        return require_roles(request, "ADMIN", "QA")
    heat = db.get(Heat, heat_id)
    if not heat:
        return PlainTextResponse("Heat not found", status_code=404)

    chem = heat.chemistry
    if not chem:
        chem = HeatChem(heat=heat)
        db.add(chem); db.commit(); db.refresh(chem)

    return templates.TemplateResponse(
        "qa_heat.html",
        {
            "request": request,
            "heat": heat,
            "role": current_role(request),
            "chem": {
                "C": (chem.c or ""),
                "Si": (chem.si or ""),
                "S": (chem.s or ""),
                "P": (chem.p or ""),
                "Cu": (chem.cu or ""),
                "Ni": (chem.ni or ""),
                "Mn": (chem.mn or ""),
                "Fe": (chem.fe or ""),
            },
            "grade": heat_grade(heat),
        },
    )

@app.post("/qa/heat/{heat_id}")
def qa_heat_save(
    request: Request,
    heat_id: int, C: str = Form(""), Si: str = Form(""), S: str = Form(""), P: str = Form(""),
    Cu: str = Form(""), Ni: str = Form(""), Mn: str = Form(""), Fe: str = Form(""),
    decision: str = Form("APPROVED"), remarks: str = Form(""),
    db: Session = Depends(get_db),
):
    if require_roles(request, "ADMIN", "QA"):
        return require_roles(request, "ADMIN", "QA")
    # simple numeric validation for chem (allow blanks)
    def _num_ok(v: str) -> bool:
        if v == "": return True
        try: float(v); return True
        except: return False
    for v in [C, Si, S, P, Cu, Ni, Mn, Fe]:
        if not _num_ok(v):
            return PlainTextResponse("Chemistry must be numeric (decimals allowed).", status_code=400)

    heat = db.get(Heat, heat_id)
    if not heat:
        return PlainTextResponse("Heat not found", status_code=404)

    chem = heat.chemistry or HeatChem(heat=heat)
    chem.c = C; chem.si = Si; chem.s = S; chem.p = P
    chem.cu = Cu; chem.ni = Ni; chem.mn = Mn; chem.fe = Fe
    heat.qa_status = decision; heat.qa_remarks = remarks
    db.add_all([chem, heat]); db.commit()
    return RedirectResponse("/melting", status_code=303)

# -------------------------------------------------
# Atomization (strict FIFO partial allocation)
# -------------------------------------------------
def _fifo_validate_allocations(db: Session, allocs: Dict[int, float]) -> Tuple[bool, str]:
    """Ensure heats are consumed in FIFO order: earlier heats must be fully used before taking from later ones."""
    if not allocs: return False, "No allocations."
    heat_ids = list(allocs.keys())
    heats = db.query(Heat).filter(Heat.id.in_(heat_ids)).all()
    by_id = {h.id: h for h in heats}
    # sort by heat_no asc
    ordered = sorted(heats, key=lambda h: h.heat_no)
    # iterate and ensure each earlier heat after allocation becomes ~zero available
    eps = 1e-6
    for i, h in enumerate(ordered):
        avail_before = heat_available(db, h)
        take = allocs.get(h.id, 0.0)
        if i < len(ordered) - 1:
            # earlier heat must be fully consumed if a later heat is used
            if any(allocs.get(lh.id, 0.0) > 0 for lh in ordered[i+1:]):
                if avail_before - take > eps:
                    return False, f"FIFO: Use full quantity from older heat {h.heat_no} before taking newer heats."
        # also, cannot over-allocate
        if take > avail_before + eps:
            return False, f"Over-allocation from heat {h.heat_no}. Available {avail_before:.1f} kg."
    return True, ""

@app.get("/atomization", response_class=HTMLResponse)
def atom_page(request: Request, db: Session = Depends(get_db),
              from_date: Optional[str] = None, to_date: Optional[str] = None):
    qh = db.query(Heat).filter(Heat.qa_status == "APPROVED")
    if from_date:
        qh = qh.filter(Heat.heat_no >= f"{dt.date.fromisoformat(from_date).strftime('%Y%m%d')}-000")
    if to_date:
        qh = qh.filter(Heat.heat_no <= f"{dt.date.fromisoformat(to_date).strftime('%Y%m%d')}-999")
    heats = qh.order_by(Heat.heat_no.asc()).all()  # ascending helps FIFO UI
    lots_q = db.query(Lot)
    if from_date:
        lots_q = lots_q.filter(Lot.lot_no.like(f"%{dt.date.fromisoformat(from_date).strftime('%Y%m%d')}%"))
    if to_date:
        lots_q = lots_q.filter(Lot.lot_no.like(f"%{dt.date.fromisoformat(to_date).strftime('%Y%m%d')}%"))
    lots = lots_q.order_by(Lot.id.desc()).all()

    grades = {h.id: heat_grade(h) for h in heats}
    available_map = {h.id: heat_available(db, h) for h in heats}
    db.commit()

    return templates.TemplateResponse(
        "atomization.html",
        {"request": request, "heats": heats, "lots": lots, "heat_grades": grades, "available_map": available_map,
         "role": current_role(request), "from_date": from_date or "", "to_date": to_date or ""}
    )

@app.post("/atomization/new")
async def atom_new(
    request: Request,
    lot_weight: float = Form(3000.0),
    db: Session = Depends(get_db)
):
    if require_roles(request, "ADMIN", "RAP"):
        return require_roles(request, "ADMIN", "RAP")

    form = await request.form()
    allocs: Dict[int, float] = {}
    for key, val in form.items():
        if key.startswith("alloc_"):
            try:
                hid = int(key.split("_", 1)[1])
                qty = float(val or 0)
                if qty > 0:
                    allocs[hid] = qty
            except:
                pass
    if not allocs:
        return PlainTextResponse("Enter allocation for at least one heat.", status_code=400)

    ok, msg = _fifo_validate_allocations(db, allocs)
    if not ok:
        return PlainTextResponse(msg, status_code=400)

    heats = db.query(Heat).filter(Heat.id.in_(allocs.keys())).all()
    any_fesi = any(heat_grade(h) == "KRFS" for h in heats)
    grade = "KRFS" if any_fesi else "KRIP"

    # Create lot number
    today = dt.date.today().strftime("%Y%m%d")
    seq = (db.query(func.count(Lot.id)).filter(Lot.lot_no.like(f"KR%{today}%")).scalar() or 0) + 1
    lot_no = f"{grade}-{today}-{seq:03d}"
    lot = Lot(lot_no=lot_no, weight=lot_weight, grade=grade)
    db.add(lot); db.flush()

    total_alloc = 0.0
    for h in heats:
        qty = allocs.get(h.id, 0.0)
        if qty <= 0:
            continue
        db.add(LotHeat(lot_id=lot.id, heat_id=h.id, qty=qty))
        h.alloc_used = (h.alloc_used or 0.0) + qty
        total_alloc += qty

    weighted_cost = 0.0
    for h in heats:
        qty = allocs.get(h.id, 0.0)
        if qty > 0:
            weighted_cost += (h.unit_cost or 0.0) * qty
    avg_heat_unit_cost = (weighted_cost / total_alloc) if total_alloc > 0 else 0.0

    lot.unit_cost = avg_heat_unit_cost + ATOMIZATION_COST_PER_KG + SURCHARGE_PER_KG
    lot.total_cost = lot.unit_cost * (lot.weight or 0.0)

    # weighted chemistry by allocated qty
    sums = {k: 0.0 for k in ["c", "si", "s", "p", "cu", "ni", "mn", "fe"]}
    for h in heats:
        q = allocs.get(h.id, 0.0)
        if q <= 0 or not h.chemistry:
            continue
        for k in list(sums.keys()):
            try:
                v = float(getattr(h.chemistry, k) or "")
                sums[k] += v * q
            except:
                pass
    avg = {}
    for k in list(sums.keys()):
        avg[k] = (sums[k] / total_alloc) if total_alloc > 0 else None
    lc = LotChem(lot=lot, **{k: (str(v) if v is not None else "") for k, v in avg.items()})
    db.add(lc)

    db.commit()
    return RedirectResponse("/atomization", status_code=303)

# -------------------------------------------------
# QA Lot (numeric forms with +212 field names)
# -------------------------------------------------
@app.get("/qa/lot/{lot_id}", response_class=HTMLResponse)
def qa_lot_form(lot_id: int, request: Request, db: Session = Depends(get_db)):
    if require_roles(request, "ADMIN", "QA"):
        return require_roles(request, "ADMIN", "QA")
    lot = db.get(Lot, lot_id)
    if not lot:
        return PlainTextResponse("Lot not found", status_code=404)

    created = False
    if not lot.chemistry:
        lot.chemistry = LotChem(lot=lot); db.add(lot.chemistry); created = True
    if not lot.phys:
        lot.phys = LotPhys(lot=lot); db.add(lot.phys); created = True
    if not lot.psd:
        lot.psd = LotPSD(lot=lot); db.add(lot.psd); created = True
    if created:
        db.commit()
        db.refresh(lot)

    psd_map = {
        "+212": lot.psd.p212 or "", "+180": lot.psd.p180 or "",
        "-180+150": lot.psd.n180p150 or "", "-150+75": lot.psd.n150p75 or "",
        "-75+45": lot.psd.n75p45 or "", "-45": lot.psd.n45 or ""
    }
    chem_map = {
        "C": lot.chemistry.c or "", "Si": lot.chemistry.si or "",
        "S": lot.chemistry.s or "", "P": lot.chemistry.p or "",
        "Cu": lot.chemistry.cu or "", "Ni": lot.chemistry.ni or "",
        "Mn": lot.chemistry.mn or "", "Fe": lot.chemistry.fe or "",
    }
    phys_map = {"ad": lot.phys.ad or "", "flow": lot.phys.flow or ""}

    return templates.TemplateResponse(
        "qa_lot.html",
        {"request": request, "lot": lot, "chem": chem_map, "phys": phys_map, "psd": psd_map, "grade": lot.grade, "role": current_role(request)}
    )

@app.post("/qa/lot/{lot_id}")
async def qa_lot_save(
    lot_id: int,
    request: Request,
    decision: str = Form("APPROVED"),
    remarks: str = Form(""),
    db: Session = Depends(get_db),
):
    if require_roles(request, "ADMIN", "QA"):
        return require_roles(request, "ADMIN", "QA")
    lot = db.get(Lot, lot_id)
    if not lot:
        return PlainTextResponse("Lot not found", status_code=404)

    form = await request.form()

    def _num_or_empty(name: str) -> str:
        v = form.get(name, "")
        if v == "": return ""
        try: float(v); return v
        except: raise ValueError(name)

    try:
        # Chemistry
        chem = lot.chemistry or LotChem(lot=lot)
        for f in ["C","Si","S","P","Cu","Ni","Mn","Fe"]:
            setattr(chem, f.lower(), _num_or_empty(f))

        # Physical
        phys = lot.phys or LotPhys(lot=lot)
        phys.ad   = _num_or_empty("ad")
        phys.flow = _num_or_empty("flow")

        # PSD
        psd = lot.psd or LotPSD(lot=lot)
        psd.p212      = _num_or_empty("+212")
        psd.p180      = _num_or_empty("+180")
        psd.n180p150  = _num_or_empty("-180+150")
        psd.n150p75   = _num_or_empty("-150+75")
        psd.n75p45    = _num_or_empty("-75+45")
        psd.n45       = _num_or_empty("-45")
    except ValueError as e:
        return PlainTextResponse(f"Field {e} must be numeric.", status_code=400)

    lot.qa_status = decision
    lot.qa_remarks = remarks
    if decision == "APPROVED":
        lot.status = "READY"  # goes to Ready Stock

    db.add(lot); db.commit()
    return RedirectResponse("/qa-dashboard", status_code=303)

# -------------------------------------------------
# Ready Stock
# -------------------------------------------------
@app.get("/ready-stock", response_class=HTMLResponse)
def ready_stock(request: Request, db: Session = Depends(get_db),
                from_date: Optional[str] = None, to_date: Optional[str] = None):
    q = db.query(Lot).filter(Lot.status == "READY")
    if from_date:
        q = q.filter(Lot.lot_no.like(f"%{dt.date.fromisoformat(from_date).strftime('%Y%m%d')}%"))
    if to_date:
        q = q.filter(Lot.lot_no.like(f"%{dt.date.fromisoformat(to_date).strftime('%Y%m%d')}%"))
    lots = q.order_by(Lot.id.desc()).all()
    return templates.TemplateResponse("ready_stock.html",
        {"request": request, "lots": lots, "role": current_role(request),
         "from_date": from_date or "", "to_date": to_date or ""})

@app.post("/ready-stock/issue/{lot_id}")
def issue_ready_stock(lot_id: int, to: str = Form(...), request: Request = None, db: Session = Depends(get_db)):
    if require_roles(request, "ADMIN", "RAP"):
        return require_roles(request, "ADMIN", "RAP")
    lot = db.get(Lot, lot_id)
    if not lot:
        return PlainTextResponse("Lot not found", status_code=404)
    if lot.status != "READY":
        return PlainTextResponse("Lot not in Ready status.", status_code=400)
    if to not in ("DISPATCH","ANNEAL"):
        return PlainTextResponse("Invalid destination.", status_code=400)
    lot.status = "ISSUED"
    lot.issued_to = to
    db.add(lot); db.commit()
    return RedirectResponse("/ready-stock", status_code=303)

# -------------------------------------------------
# QA Dashboard
# -------------------------------------------------
@app.get("/qa-dashboard", response_class=HTMLResponse)
def qa_dashboard(request: Request, db: Session = Depends(get_db)):
    heats = db.query(Heat).order_by(Heat.id.desc()).all()
    lots = db.query(Lot).order_by(Lot.id.desc()).all()
    heat_grades = {h.id: heat_grade(h) for h in heats}
    return templates.TemplateResponse("qa_dashboard.html",
        {"request": request, "heats": heats, "lots": lots, "heat_grades": heat_grades, "role": current_role(request)})

# -------------------------------------------------
# Traceability (allocated qty only)
# -------------------------------------------------
@app.get("/traceability/lot/{lot_id}", response_class=HTMLResponse)
def trace_lot(lot_id: int, request: Request, db: Session = Depends(get_db)):
    lot = db.get(Lot, lot_id)
    heats = [lh.heat for lh in lot.heats]
    rows = []
    for lh in lot.heats:
        h = lh.heat
        rows.append(
            type("Row", (), {
                "heat_no": h.heat_no,
                "rm_type": "Melt Output",
                "grn_id": "-",
                "supplier": "",
                "qty": lh.qty,              # allocated qty only
            })
        )
    return templates.TemplateResponse("trace_lot.html",
        {"request": request, "lot": lot, "heats": heats, "grn_rows": rows, "role": current_role(request)})

# -------------------------------------------------
# PDF
# -------------------------------------------------
def draw_header(c: canvas.Canvas, title: str):
    width, height = A4
    logo_path = os.path.join(os.path.dirname(__file__), "..", "static", "KRN_Logo.png")
    if os.path.exists(logo_path):
        c.drawImage(logo_path, 1.5 * cm, height - 3 * cm, width=4 * cm, preserveAspectRatio=True, mask="auto")
    c.setFont("Helvetica-Bold", 14); c.drawString(7 * cm, height - 2 * cm, "KRN Alloys Pvt Ltd")
    c.setFont("Helvetica-Bold", 12); c.drawString(7 * cm, height - 2.7 * cm, title)
    c.line(1.5 * cm, height - 3.3 * cm, width - 1.5 * cm, height - 3.3 * cm)

@app.get("/pdf/lot/{lot_id}")
def pdf_lot(lot_id: int, db: Session = Depends(get_db)):
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

    c.setFont("Helvetica-Bold", 11); c.drawString(2 * cm, y, "Heats (Allocated)"); y -= 14
    c.setFont("Helvetica", 10)
    for lh in lot.heats:
        h = lh.heat
        c.drawString(2.2 * cm, y, f"{h.heat_no}  | Alloc: {lh.qty:.1f} kg | QA: {h.qa_status}")
        y -= 12
        if y < 3 * cm:
            c.showPage(); draw_header(c, f"Traceability Report – Lot {lot.lot_no}"); y = height - 4 * cm

    c.showPage(); c.save()
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/pdf",
                             headers={"Content-Disposition": f'inline; filename=\"trace_{lot.lot_no}.pdf\"'})
