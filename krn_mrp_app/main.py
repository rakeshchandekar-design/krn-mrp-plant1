import os, io, datetime as dt
from typing import Optional, Dict, List, Tuple
from urllib.parse import quote
from enum import Enum

# FastAPI
from fastapi import FastAPI, Request, Form, Depends, Response
from fastapi.responses import HTMLResponse, RedirectResponse, PlainTextResponse, StreamingResponse
from fastapi.responses import HTMLResponse
def _alert_redirect(msg: str, url: str = "/atomization") -> HTMLResponse:
    safe = (msg or "").replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
    html = f'''<script>
      alert("{safe}");
      window.location.href = "{url}";
    </script>'''
    return HTMLResponse(html)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

# SQLAlchemy
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey, func, text
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session
from sqlalchemy.orm import joinedload  # eager load

# -------------------------------------------------
# Costing constants (baseline)
# -------------------------------------------------
MELT_COST_PER_KG_KRIP = 6.0
MELT_COST_PER_KG_KRFS = 8.0
ATOMIZATION_COST_PER_KG = 5.0
SURCHARGE_PER_KG = 2.0

# Melting capacity & power targets
DAILY_CAPACITY_KG = 7000.0          # melting 24h capacity
POWER_TARGET_KWH_PER_TON = 560.0    # target kWh/ton

# Atomization capacity
DAILY_CAPACITY_ATOM_KG = 6000.0     # atomization 24h capacity

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
# Schema migration (only fields actually used)
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

        # Power & per-heat downtime tracking
        for coldef in [
            "power_kwh REAL DEFAULT 0",
            "kwh_per_ton REAL DEFAULT 0",
            "downtime_min INTEGER DEFAULT 0",
            "downtime_type TEXT",
            "downtime_note TEXT",
        ]:
            col = coldef.split()[0]
            if not _table_has_column(conn, "heat", col):
                if str(engine.url).startswith("sqlite"):
                    conn.execute(text(f"ALTER TABLE heat ADD COLUMN {coldef}"))
                else:
                    sql = coldef.replace("REAL", "DOUBLE PRECISION")
                    conn.execute(text(f"ALTER TABLE heat ADD COLUMN IF NOT EXISTS {col} {sql.split(' ',1)[1]}"))

        # Track how much of a heat is allocated into lots (mirror for dashboards)
        if not _table_has_column(conn, "heat", "alloc_used"):
            if str(engine.url).startswith("sqlite"):
                conn.execute(text("ALTER TABLE heat ADD COLUMN alloc_used REAL DEFAULT 0"))
            else:
                conn.execute(text("ALTER TABLE heat ADD COLUMN IF NOT EXISTS alloc_used DOUBLE PRECISION DEFAULT 0"))

        # Lot costing
        for col in ["unit_cost", "total_cost"]:
            if not _table_has_column(conn, "lot", col):
                if str(engine.url).startswith("sqlite"):
                    conn.execute(text(f"ALTER TABLE lot ADD COLUMN {col} REAL DEFAULT 0"))
                else:
                    conn.execute(text(f"ALTER TABLE lot ADD COLUMN IF NOT EXISTS {col} DOUBLE PRECISION DEFAULT 0"))

        # LotHeat qty for partial allocation
        if not _table_has_column(conn, "lot_heat", "qty"):
            if str(engine.url).startswith("sqlite"):
                conn.execute(text("ALTER TABLE lot_heat ADD COLUMN qty REAL DEFAULT 0"))
            else:
                conn.execute(text("ALTER TABLE lot_heat ADD COLUMN IF NOT EXISTS qty DOUBLE PRECISION DEFAULT 0"))

        # GRN readable number
        if not _table_has_column(conn, "grn", "grn_no"):
            if str(engine.url).startswith("sqlite"):
                conn.execute(text("ALTER TABLE grn ADD COLUMN grn_no TEXT"))
            else:
                conn.execute(text("ALTER TABLE grn ADD COLUMN IF NOT EXISTS grn_no TEXT UNIQUE"))

        # Day-level downtime table for MELTING (existing)
        if str(engine.url).startswith("sqlite"):
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS downtime (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    minutes INTEGER NOT NULL DEFAULT 0,
                    kind TEXT,
                    remarks TEXT
                )
            """))
        else:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS downtime(
                    id SERIAL PRIMARY KEY,
                    date DATE NOT NULL,
                    minutes INT NOT NULL DEFAULT 0,
                    kind TEXT,
                    remarks TEXT
                )
            """))

        # NEW: Day-level downtime table for ATOMIZATION
        if str(engine.url).startswith("sqlite"):
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS atom_downtime (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    minutes INTEGER NOT NULL DEFAULT 0,
                    kind TEXT,
                    remarks TEXT
                )
            """))
        else:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS atom_downtime(
                    id SERIAL PRIMARY KEY,
                    date DATE NOT NULL,
                    minutes INT NOT NULL DEFAULT 0,
                    kind TEXT,
                    remarks TEXT
                )
            """))

        # RAP tables
        if str(engine.url).startswith("sqlite"):
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS rap_lot (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lot_id INTEGER UNIQUE,
                    available_qty REAL NOT NULL DEFAULT 0,
                    status TEXT,
                    FOREIGN KEY(lot_id) REFERENCES lot(id)
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS rap_alloc (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rap_lot_id INTEGER NOT NULL,
                    date DATE NOT NULL,
                    kind TEXT NOT NULL,
                    qty REAL NOT NULL DEFAULT 0,
                    remarks TEXT,
                    dest TEXT,
                    FOREIGN KEY(rap_lot_id) REFERENCES rap_lot(id)
                )
            """))
        else:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS rap_lot (
                    id SERIAL PRIMARY KEY,
                    lot_id INT UNIQUE,
                    available_qty DOUBLE PRECISION NOT NULL DEFAULT 0,
                    status TEXT,
                    FOREIGN KEY(lot_id) REFERENCES lot(id)
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS rap_alloc (
                    id SERIAL PRIMARY KEY,
                    rap_lot_id INT NOT NULL,
                    date DATE NOT NULL,
                    kind TEXT NOT NULL,
                    qty DOUBLE PRECISION NOT NULL DEFAULT 0,
                    remarks TEXT,
                    dest TEXT,
                    FOREIGN KEY(rap_lot_id) REFERENCES rap_lot(id)
                )
            """))

                # RAP Dispatch + Transfer tables
        if str(engine.url).startswith("sqlite"):
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS rap_dispatch(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    customer TEXT NOT NULL,
                    grade TEXT NOT NULL,
                    total_qty REAL DEFAULT 0,
                    total_cost REAL DEFAULT 0
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS rap_dispatch_item(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dispatch_id INTEGER,
                    lot_id INTEGER,
                    qty REAL DEFAULT 0,
                    cost REAL DEFAULT 0
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS rap_transfer(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    lot_id INTEGER,
                    qty REAL DEFAULT 0,
                    remarks TEXT
                )
            """))
        else:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS rap_dispatch(
                    id SERIAL PRIMARY KEY,
                    date DATE NOT NULL,
                    customer TEXT NOT NULL,
                    grade TEXT NOT NULL,
                    total_qty DOUBLE PRECISION DEFAULT 0,
                    total_cost DOUBLE PRECISION DEFAULT 0
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS rap_dispatch_item(
                    id SERIAL PRIMARY KEY,
                    dispatch_id INT REFERENCES rap_dispatch(id),
                    lot_id INT REFERENCES lot(id),
                    qty DOUBLE PRECISION DEFAULT 0,
                    cost DOUBLE PRECISION DEFAULT 0
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS rap_transfer(
                    id SERIAL PRIMARY KEY,
                    date DATE NOT NULL,
                    lot_id INT REFERENCES lot(id),
                    qty DOUBLE PRECISION DEFAULT 0,
                    remarks TEXT
                )
            """))


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
    grn_no = Column(String, unique=True, index=True)
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

    # power & downtime
    power_kwh = Column(Float, default=0.0)
    kwh_per_ton = Column(Float, default=0.0)
    downtime_min = Column(Integer, default=0)
    downtime_type = Column(String)
    downtime_note = Column(String)

    # partial allocations (mirror of allocated qty in lots)
    alloc_used = Column(Float, default=0.0)

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

    heats = relationship("LotHeat", back_populates="lot", cascade="all, delete-orphan")
    chemistry = relationship("LotChem", uselist=False, back_populates="lot")
    phys = relationship("LotPhys", uselist=False, back_populates="lot")
    psd = relationship("LotPSD", uselist=False, back_populates="lot")

class LotHeat(Base):
    __tablename__ = "lot_heat"
    id = Column(Integer, primary_key=True)
    lot_id = Column(Integer, ForeignKey("lot.id"))
    heat_id = Column(Integer, ForeignKey("heat.id"))
    qty = Column(Float, default=0.0)
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

# Optional day-level downtime table (Melting)
class Downtime(Base):
    __tablename__ = "downtime"
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    minutes = Column(Integer, default=0)
    kind = Column(String)
    remarks = Column(String)

# NEW: Optional day-level downtime table (Atomization)
class AtomDowntime(Base):
    __tablename__ = "atom_downtime"
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    minutes = Column(Integer, default=0)
    kind = Column(String)
    remarks = Column(String)

# -------------------------------
# RAP / ULA models (new)
# -------------------------------

class RAPLot(Base):
    """
    A mirror bucket for an Atomization lot that enters RAP stage.
    We keep available_qty here; allocations reduce it.
    """
    __tablename__ = "rap_lot"
    id = Column(Integer, primary_key=True)
    lot_id = Column(Integer, ForeignKey("lot.id"), unique=True, index=True)
    available_qty = Column(Float, default=0.0)
    # status is informational; available_qty==0 implies closed
    status = Column(String, default="OPEN")  # OPEN / CLOSED

    lot = relationship("Lot")

class RAPKind(str, Enum):
    DISPATCH = "DISPATCH"
    PLANT2   = "PLANT2"

class RAPAlloc(Base):
    """
    Individual allocation movement from RAP to a destination (Dispatch or Plant 2).
    """
    __tablename__ = "rap_alloc"
    id = Column(Integer, primary_key=True)
    rap_lot_id = Column(Integer, ForeignKey("rap_lot.id"), index=True)
    date = Column(Date, nullable=False)
    kind = Column(String, nullable=False)   # DISPATCH or PLANT2
    qty  = Column(Float, nullable=False, default=0.0)
    remarks = Column(String)
    dest = Column(String)  # customer name or "Plant 2"

    rap_lot = relationship("RAPLot")

# RAP Dispatch + Transfer Models
class RAPDispatch(Base):
    __tablename__ = "rap_dispatch"
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    customer = Column(String, nullable=False)
    grade = Column(String, nullable=False)
    total_qty = Column(Float, default=0.0)
    total_cost = Column(Float, default=0.0)

    items = relationship("RAPDispatchItem", back_populates="dispatch", cascade="all, delete-orphan")

class RAPDispatchItem(Base):
    __tablename__ = "rap_dispatch_item"
    id = Column(Integer, primary_key=True)
    dispatch_id = Column(Integer, ForeignKey("rap_dispatch.id"))
    lot_id = Column(Integer, ForeignKey("lot.id"))
    qty = Column(Float, default=0.0)
    cost = Column(Float, default=0.0)

    dispatch = relationship("RAPDispatch", back_populates="items")
    lot = relationship("Lot")

class RAPTransfer(Base):
    __tablename__ = "rap_transfer"
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    lot_id = Column(Integer, ForeignKey("lot.id"))
    qty = Column(Float, default=0.0)
    remarks = Column(String, default="")

    lot = relationship("Lot")


# -------------------------------------------------
# App + Templates (robust paths)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
if not os.path.isdir(TEMPLATES_DIR):
    TEMPLATES_DIR = os.path.join(BASE_DIR, "..", "templates")

STATIC_DIR = os.path.join(BASE_DIR, "static")
if not os.path.isdir(STATIC_DIR):
    STATIC_DIR = os.path.join(BASE_DIR, "..", "static")

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# expose python builtins to Jinja
templates.env.globals.update(max=max, min=min, round=round, int=int, float=float)

# -------------------------------------------------
# Startup
# -------------------------------------------------
@app.on_event("startup")
def _startup_migrate():
    Base.metadata.create_all(bind=engine)
    migrate_schema(engine)

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
# Helpers
# -------------------------------------------------
def heat_grade(heat: Heat) -> str:
    for cons in heat.rm_consumptions:
        if cons.rm_type == "FeSi":
            return "KRFS"
    return "KRIP"

def heat_available_fast(heat: Heat, used_map: Dict[int, float]) -> float:
    used = float(used_map.get(heat.id, 0.0))
    heat.alloc_used = used
    return max((heat.actual_output or 0.0) - used, 0.0)

def heat_available(db: Session, heat: Heat) -> float:
    used = db.query(func.coalesce(func.sum(LotHeat.qty), 0.0)).filter(LotHeat.heat_id == heat.id).scalar() or 0.0
    heat.alloc_used = float(used)
    return max((heat.actual_output or 0.0) - used, 0.0)

def next_grn_no(db: Session, on_date: dt.date) -> str:
    ymd = on_date.strftime("%Y%m%d")
    count = (db.query(func.count(GRN.id)).filter(GRN.grn_no.like(f"GRN-{ymd}-%")).scalar() or 0) + 1
    return f"GRN-{ymd}-{count:03d}"

def heat_date_from_no(heat_no: str) -> Optional[dt.date]:
    try:
        return dt.datetime.strptime(heat_no.split("-")[0], "%Y%m%d").date()
    except Exception:
        return None

def lot_date_from_no(lot_no: str) -> Optional[dt.date]:
    """Parse date from lot number e.g. KRIP-YYYYMMDD-###"""
    try:
        parts = lot_no.split("-")
        for p in parts:
            if len(p) == 8 and p.isdigit():
                return dt.datetime.strptime(p, "%Y%m%d").date()
    except Exception:
        pass
    return None

def day_available_minutes(db: Session, day: dt.date) -> int:
    """Melting: 1440 minus total of per-heat downtime + day-level downtime."""
    dn = day.strftime("%Y%m%d")
    heat_mins = (
        db.query(func.coalesce(func.sum(Heat.downtime_min), 0))
        .filter(Heat.heat_no.like(f"{dn}-%"))
        .scalar()
        or 0
    )
    extra_mins = db.query(func.coalesce(func.sum(Downtime.minutes), 0)).filter(Downtime.date == day).scalar() or 0
    mins = int(heat_mins) + int(extra_mins)
    return max(1440 - mins, 0)

def day_target_kg(db: Session, day: dt.date) -> float:
    return DAILY_CAPACITY_KG * (day_available_minutes(db, day) / 1440.0)

def atom_day_available_minutes(db: Session, day: dt.date) -> int:
    """Atomization: currently only day-level downtime (no per-lot downtime fields)."""
    extra_mins = db.query(func.coalesce(func.sum(AtomDowntime.minutes), 0)).filter(AtomDowntime.date == day).scalar() or 0
    return max(1440 - int(extra_mins), 0)

def atom_day_target_kg(db: Session, day: dt.date) -> float:
    return DAILY_CAPACITY_ATOM_KG * (atom_day_available_minutes(db, day) / 1440.0)

# -------------------------------
# RAP helpers
# -------------------------------
def rap_total_alloc_qty_for_lot(db: Session, lot_id: int) -> float:
    rap = db.query(RAPLot).filter(RAPLot.lot_id == lot_id).first()
    if not rap:
        return 0.0
    q = db.query(func.coalesce(func.sum(RAPAlloc.qty), 0.0)).filter(RAPAlloc.rap_lot_id == rap.id).scalar() or 0.0
    return float(q)

def ensure_rap_lot(db: Session, lot: Lot) -> RAPLot:
    """
    Ensure a RAPLot exists for an APPROVED lot.
    available_qty is recalculated as (lot.weight - total_allocs).
    """
    rap = db.query(RAPLot).filter(RAPLot.lot_id == lot.id).first()
    total_alloc = rap_total_alloc_qty_for_lot(db, lot.id) if rap else 0.0
    current_avail = max((lot.weight or 0.0) - total_alloc, 0.0)

    if rap:
        rap.available_qty = current_avail
        rap.status = "CLOSED" if current_avail <= 1e-6 else "OPEN"
        db.add(rap)
        return rap

    rap = RAPLot(lot_id=lot.id, available_qty=current_avail, status=("CLOSED" if current_avail <= 1e-6 else "OPEN"))
    db.add(rap); db.flush()
    return rap


# -------------------------------------------------
# Health + Setup + Home
# -------------------------------------------------
@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/setup")
def setup(db: Session = Depends(get_db)):
    Base.metadata.create_all(bind=engine)
    migrate_schema(engine)
    return HTMLResponse('Tables created/migrated. Go to <a href="/">Home</a>.')

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# -------------------------------------------------
# GRN (unchanged)
# -------------------------------------------------
@app.get("/grn", response_class=HTMLResponse)
def grn_list(
    request: Request,
    report: Optional[int] = 0,
    start: Optional[str] = None,
    end: Optional[str] = None,
    db: Session = Depends(get_db),
):
    q = db.query(GRN)
    if report:
        if start:
            q = q.filter(GRN.date >= dt.date.fromisoformat(start))
        if end:
            q = q.filter(GRN.date <= dt.date.fromisoformat(end))
    else:
        q = q.filter(GRN.remaining_qty > 0)
    grns = q.order_by(GRN.id.desc()).all()

    live = db.query(GRN).filter(GRN.remaining_qty > 0).all()
    rm_summary = []
    for rm in RM_TYPES:
        subset = [r for r in live if r.rm_type == rm]
        avail = sum(r.remaining_qty for r in subset)
        cost = sum((r.remaining_qty or 0.0) * (r.price or 0.0) for r in subset)
        rm_summary.append({"rm_type": rm, "available": avail, "cost": cost})

    today = dt.date.today()
    return templates.TemplateResponse(
        "grn.html",
        {
            "request": request,
            "grns": grns,
            "prices": rm_price_defaults(),
            "report": report,
            "start": start or "",
            "end": end or "",
            "rm_summary": rm_summary,
            "today": today,
            "today_iso": today.isoformat(),
        },
    )

@app.get("/grn/new", response_class=HTMLResponse)
def grn_new(request: Request):
    today = dt.date.today()
    min_date = (today - dt.timedelta(days=4)).isoformat()
    max_date = today.isoformat()
    return templates.TemplateResponse(
        "grn_new.html",
        {"request": request, "rm_types": RM_TYPES, "min_date": min_date, "max_date": max_date},
    )

@app.post("/grn/new")
def grn_new_post(
    date: str = Form(...), supplier: str = Form(...), rm_type: str = Form(...),
    qty: float = Form(...), price: float = Form(...), db: Session = Depends(get_db)
):
    today = dt.date.today()
    d = dt.date.fromisoformat(date)
    if d > today or d < (today - dt.timedelta(days=4)):
        return PlainTextResponse("Date must be today or within the last 4 days.", status_code=400)

    g = GRN(
        grn_no=next_grn_no(db, d),
        date=d,
        supplier=supplier,
        rm_type=rm_type,
        qty=qty,
        remaining_qty=qty,
        price=price,
    )
    db.add(g); db.commit()
    return RedirectResponse("/grn", status_code=303)

# ---------- CSV export for report range ----------
@app.get("/grn/export")
def grn_export(
    start: str,
    end: str,
    db: Session = Depends(get_db),
):
    s = dt.date.fromisoformat(start)
    e = dt.date.fromisoformat(end)
    rows = (
        db.query(GRN)
        .filter(GRN.date >= s, GRN.date <= e)
        .order_by(GRN.id.asc())
        .all()
    )
    out = io.StringIO()
    out.write("GRN No,Date,Supplier,RM Type,Qty (kg),Price (Rs/kg),Total Price,Remaining (kg),Remaining Cost\n")
    for r in rows:
        total_price = (r.qty or 0.0) * (r.price or 0.0)
        remaining_cost = (r.remaining_qty or 0.0) * (r.price or 0.0)
        out.write(
            f"{r.grn_no or ''},{r.date},{r.supplier},{r.rm_type},"
            f"{(r.qty or 0.0):.1f},{(r.price or 0.0):.2f},{total_price:.2f},"
            f"{(r.remaining_qty or 0.0):.1f},{remaining_cost:.2f}\n"
        )
    data = out.getvalue().encode("utf-8")
    filename = f"grn_report_{start}_to_{end}.csv"
    return Response(
        content=data,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

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
# Melting (enhanced performance, unchanged UX/logic)
# -------------------------------------------------
@app.get("/melting", response_class=HTMLResponse)
def melting_page(
    request: Request,
    start: Optional[str] = None,   # YYYY-MM-DD
    end: Optional[str] = None,     # YYYY-MM-DD
    all: Optional[int] = 0,        # if 1 => show all heats with available > 0
    db: Session = Depends(get_db),
):
    heats: List[Heat] = (
        db.query(Heat)
        .options(joinedload(Heat.rm_consumptions).joinedload(HeatRM.grn))
        .order_by(Heat.id.desc())
        .all()
    )

    used_map: Dict[int, float] = dict(
        db.query(LotHeat.heat_id, func.coalesce(func.sum(LotHeat.qty), 0.0))
          .group_by(LotHeat.heat_id)
          .all()
    )

    rows = []
    for h in heats:
        avail = heat_available_fast(h, used_map)
        rows.append({
            "heat": h,
            "grade": heat_grade(h),
            "available": avail,
            "date": heat_date_from_no(h.heat_no) or dt.date.today(),
        })

    today = dt.date.today()

    # KPIs (today)
    todays = [r["heat"] for r in rows if r["date"] == today and (r["heat"].actual_output or 0) > 0]
    kwhpt_vals = []
    for h in todays:
        if (h.power_kwh or 0) > 0 and (h.actual_output or 0) > 0:
            kwhpt_vals.append((h.power_kwh / h.actual_output) * 1000.0)
    today_kwhpt = (sum(kwhpt_vals) / len(kwhpt_vals)) if kwhpt_vals else 0.0

    tot_out_today = sum((r["heat"].actual_output or 0.0) for r in rows if r["date"] == today)
    tot_in_today  = sum((r["heat"].total_inputs  or 0.0) for r in rows if r["date"] == today)
    yield_today = (100.0 * tot_out_today / tot_in_today) if tot_in_today > 0 else 0.0
    eff_today   = (100.0 * tot_out_today / DAILY_CAPACITY_KG) if DAILY_CAPACITY_KG > 0 else 0.0

    # Last 5 days: actual, target adjusted
    last5 = []
    for i in range(4, -1, -1):
        d = today - dt.timedelta(days=i)
        actual = sum((r["heat"].actual_output or 0.0) for r in rows if r["date"] == d)
        target = day_target_kg(db, d)
        last5.append({"date": d.isoformat(), "actual": actual, "target": target})

    # Live stock summary by grade
    krip_qty = krip_val = krfs_qty = krfs_val = 0.0
    for r in rows:
        if r["available"] <= 0:
            continue
        val = r["available"] * (r["heat"].unit_cost or 0.0)
        if r["grade"] == "KRFS":
            krfs_qty += r["available"]; krfs_val += val
        else:
            krip_qty += r["available"]; krip_val += val

    # Date range defaults
    s = start or today.isoformat()
    e = end or today.isoformat()
    try:
        s_date = dt.date.fromisoformat(s); e_date = dt.date.fromisoformat(e)
    except Exception:
        s_date, e_date = today, today

    visible_heats = [r["heat"] for r in rows if s_date <= r["date"] <= e_date]
    if all and int(all) == 1:
        visible_heats = [r["heat"] for r in rows if (r["available"] or 0.0) > 0.0]

    trace_map: Dict[int, str] = {}
    for h in visible_heats:
        parts: List[str] = []
        by_rm: Dict[str, List[Tuple[int, float]]] = {}
        for c in h.rm_consumptions:
            by_rm.setdefault(c.rm_type, []).append((c.grn_id, c.qty or 0.0))
        for rm, items in by_rm.items():
            items_txt = ", ".join([f"GRN {gid}:{qty:.0f}" for gid, qty in items])
            parts.append(f"{rm}: {items_txt}")
        trace_map[h.id] = "; ".join(parts) if parts else "-"

    return templates.TemplateResponse(
        "melting.html",
        {
            "request": request,
            "rm_types": RM_TYPES,
            "pending": visible_heats,
            "heat_grades": {r["heat"].id: r["grade"] for r in rows},
            "today_kwhpt": today_kwhpt,
            "yield_today": yield_today,
            "eff_today": eff_today,
            "last5": last5,
            "stock": {"krip_qty": krip_qty, "krip_val": krip_val, "krfs_qty": krfs_qty, "krfs_val": krfs_val},
            "today_iso": today.isoformat(),
            "start": s, "end": e,
            "power_target": POWER_TARGET_KWH_PER_TON,
            "trace_map": trace_map,
        },
    )

# -------------------------------------------------
# Create Heat (same logic; inline alert if GRN insufficient)
# -------------------------------------------------
@app.post("/melting/new")
def melting_new(
    request: Request,
    notes: Optional[str] = Form(None),

    slag_qty: float = Form(...),
    power_kwh: float = Form(...),

    downtime_min: int = Form(...),
    downtime_type: str = Form("production"),
    downtime_note: str = Form(""),

    rm_type_1: Optional[str] = Form(None), rm_qty_1: Optional[str] = Form(None),
    rm_type_2: Optional[str] = Form(None), rm_qty_2: Optional[str] = Form(None),
    rm_type_3: Optional[str] = Form(None), rm_qty_3: Optional[str] = Form(None),
    rm_type_4: Optional[str] = Form(None), rm_qty_4: Optional[str] = Form(None),

    db: Session = Depends(get_db),
):
    def _to_float(x: Optional[str]) -> Optional[float]:
        try:
            if x is None: return None
            s = str(x).strip()
            if s == "": return None
            return float(s)
        except:
            return None

    parsed = []
    for t, q in [(rm_type_1, _to_float(rm_qty_1)),
                 (rm_type_2, _to_float(rm_qty_2)),
                 (rm_type_3, _to_float(rm_qty_3)),
                 (rm_type_4, _to_float(rm_qty_4))]:
        if t and q and q > 0:
            parsed.append((t, q))

    if len(parsed) < 2:
        return PlainTextResponse("Enter at least two RM lines.", status_code=400)
    if power_kwh is None or power_kwh <= 0:
        return PlainTextResponse("Power Units Consumed (kWh) must be > 0.", status_code=400)
    if downtime_min is None or downtime_min < 0:
        return PlainTextResponse("Downtime minutes must be 0 or more.", status_code=400)
    if int(downtime_min) > 0:
        if not (downtime_type and str(downtime_type).strip()):
            return PlainTextResponse("Downtime type is required when downtime > 0.", status_code=400)
        if not (downtime_note and str(downtime_note).strip()):
            return PlainTextResponse("Downtime remarks are required when downtime > 0.", status_code=400)
    else:
        downtime_type = None
        downtime_note = ""

    # Create heat number
    today = dt.date.today().strftime("%Y%m%d")
    seq = (db.query(func.count(Heat.id)).filter(Heat.heat_no.like(f"{today}-%")).scalar() or 0) + 1
    heat_no = f"{today}-{seq:03d}"
    heat = Heat(
        heat_no=heat_no,
        notes=notes or "",
        slag_qty=slag_qty,
        power_kwh=float(power_kwh),
        downtime_min=int(downtime_min),
        downtime_type=downtime_type,
        downtime_note=downtime_note,
    )
    db.add(heat); db.flush()

    # Stock checks
    for t, q in parsed:
        if available_stock(db, t) < q - 1e-6:
            db.rollback()
            msg = f"Insufficient stock for {t}. Available {available_stock(db, t):.1f} kg"
            html = f"""<script>alert("{msg}");window.location="/melting";</script>"""
            return HTMLResponse(html)

    # Consume FIFO + accumulate RM cost
    total_inputs = 0.0
    total_rm_cost = 0.0
    used_fesi = False
    for t, q in parsed:
        if t == "FeSi":
            used_fesi = True
        total_rm_cost += consume_fifo(db, t, q, heat)
        total_inputs += q

    heat.total_inputs = total_inputs
    heat.actual_output = total_inputs - (slag_qty or 0.0)
    heat.theoretical = total_inputs * 0.97

    melt_cost_per_kg = MELT_COST_PER_KG_KRFS if used_fesi else MELT_COST_PER_KG_KRIP
    if (heat.actual_output or 0) > 0:
        heat.rm_cost = total_rm_cost
        heat.process_cost = melt_cost_per_kg * heat.actual_output
        heat.total_cost = heat.rm_cost + heat.process_cost
        heat.unit_cost = heat.total_cost / heat.actual_output
        heat.kwh_per_ton = (heat.power_kwh or 0.0) / max((heat.actual_output or 0.0) / 1000.0, 1e-9)
    else:
        heat.rm_cost = total_rm_cost
        heat.process_cost = 0.0
        heat.total_cost = total_rm_cost
        heat.unit_cost = 0.0
        heat.kwh_per_ton = 0.0

    db.commit()
    return RedirectResponse("/melting", status_code=303)

# ---------- CSV export for melting report ----------
@app.get("/melting/export")
def melting_export(
    start: Optional[str] = None,
    end: Optional[str] = None,
    db: Session = Depends(get_db),
):
    heats = db.query(Heat).order_by(Heat.id.asc()).all()
    s = dt.date.fromisoformat(start) if start else None
    e = dt.date.fromisoformat(end) if end else None

    out = io.StringIO()
    out.write("Heat No,Date,Grade,QA,Output kg,Available kg,Unit Cost,Total Cost,Power kWh,kWh per ton,Downtime min,Type,Remark\n")
    for h in heats:
        d = heat_date_from_no(h.heat_no) or dt.date.today()
        if s and d < s:
            continue
        if e and d > e:
            continue
        used = db.query(func.coalesce(func.sum(LotHeat.qty), 0.0)).filter(LotHeat.heat_id == h.id).scalar() or 0.0
        avail = max((h.actual_output or 0.0) - used, 0.0)
        out.write(
            f"{h.heat_no},{d.isoformat()},{('KRFS' if any(c.rm_type=='FeSi' for c in h.rm_consumptions) else 'KRIP')},{h.qa_status or ''},"
            f"{h.actual_output or 0:.1f},{avail:.1f},{h.unit_cost or 0:.2f},{h.total_cost or 0:.2f},"
            f"{h.power_kwh or 0:.1f},{h.kwh_per_ton or 0:.1f},{int(h.downtime_min or 0)},{h.downtime_type or ''},{(h.downtime_note or '').replace(',', ' ')}\n"
        )
    data = out.getvalue().encode("utf-8")
    filename = f"melting_report_{(start or '').replace('-', '')}_{(end or '').replace('-', '')}.csv"
    if not filename.strip("_").strip():
        filename = "melting_report.csv"
    return StreamingResponse(
        io.BytesIO(data),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

# ---------- CSV export for melting downtime ----------
@app.get("/melting/downtime/export")
def downtime_export(db: Session = Depends(get_db)):
    out = io.StringIO()
    out.write("Source,Date,Heat No,Minutes,Type/Kind,Remarks\n")

    heats = db.query(Heat).order_by(Heat.id.asc()).all()
    for h in heats:
        mins = int(h.downtime_min or 0)
        if mins <= 0:
            continue
        d = heat_date_from_no(h.heat_no) or dt.date.today()
        out.write(f"HEAT,{d.isoformat()},{h.heat_no},{mins},{h.downtime_type or ''},{(h.downtime_note or '').replace(',', ' ')}\n")

    days = db.query(Downtime).order_by(Downtime.date.asc(), Downtime.id.asc()).all()
    for r in days:
        out.write(f"DAY,{r.date.isoformat()},,{int(r.minutes or 0)},{r.kind or ''},{(r.remarks or '').replace(',', ' ')}\n")

    data = out.getvalue().encode("utf-8")
    return StreamingResponse(
        io.BytesIO(data),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="downtime_export.csv"'}
    )

# -------------------------------------------------
# QA redirect + Heat QA
# -------------------------------------------------
@app.get("/qa")
def qa_redirect():
    return RedirectResponse("/qa-dashboard", status_code=303)

@app.get("/qa/heat/{heat_id}", response_class=HTMLResponse)
def qa_heat_form(heat_id: int, request: Request, db: Session = Depends(get_db)):
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
    heat_id: int, C: str = Form(""), Si: str = Form(""), S: str = Form(""), P: str = Form(""),
    Cu: str = Form(""), Ni: str = Form(""), Mn: str = Form(""), Fe: str = Form(""),
    decision: str = Form("APPROVED"), remarks: str = Form(""),
    db: Session = Depends(get_db),
):
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
# Atomization (ENHANCED — additions only; existing allocation UI untouched)
# -------------------------------------------------
# Atomization (ENHANCED — additions only; existing allocation UI untouched)
# -------------------------------------------------
@app.get("/atomization", response_class=HTMLResponse)
def atom_page(
    request: Request,
    start: Optional[str] = None,   # YYYY-MM-DD (for toolbar on atomization page)
    end: Optional[str] = None,     # YYYY-MM-DD
    db: Session = Depends(get_db)
):
    # Only APPROVED heats
    heats_all = (
        db.query(Heat)
        .filter(Heat.qa_status == "APPROVED")
        .order_by(Heat.id.desc())
        .all()
    )

    # Availability + grade
    available_map = {h.id: heat_available(db, h) for h in heats_all}
    grades = {h.id: heat_grade(h) for h in heats_all}

    # Hide zero-available heats
    heats = [h for h in heats_all if (available_map.get(h.id) or 0.0) > 0.0001]

    lots = db.query(Lot).order_by(Lot.id.desc()).all()

    # ----- KPIs & last-5-days for Atomization -----
    today = dt.date.today()

    # production today = sum of lot weights created today
    lots_with_dates = [(lot, lot_date_from_no(lot.lot_no) or today) for lot in lots]
    prod_today = sum((lot.weight or 0.0) for lot, d in lots_with_dates if d == today)
    eff_today = (100.0 * prod_today / DAILY_CAPACITY_ATOM_KG) if DAILY_CAPACITY_ATOM_KG > 0 else 0.0

    last5 = []
    for i in range(4, -1, -1):
        d = today - dt.timedelta(days=i)
        actual = sum((lot.weight or 0.0) for lot, dd in lots_with_dates if dd == d)
        target = atom_day_target_kg(db, d)
        last5.append({"date": d.isoformat(), "actual": actual, "target": target})

    # Live stock of lots (WIP) = ALL atomization lots (any QA) - RAP allocations (only approved lots allocate)
    stock = {"KRIP_qty": 0.0, "KRIP_val": 0.0, "KRFS_qty": 0.0, "KRFS_val": 0.0}

    # Preload RAP allocations per lot id (for speed)
    # We only ever create RAP entries for APPROVED lots, but we subtract allocs from total WIP as requested.
    rap_alloc_by_lot: Dict[int, float] = {}
    rap_pairs = (
        db.query(RAPLot.lot_id, func.coalesce(func.sum(RAPAlloc.qty), 0.0))
          .join(RAPAlloc, RAPAlloc.rap_lot_id == RAPLot.id, isouter=True)
          .group_by(RAPLot.lot_id)
          .all()
    )
    for lid, s in rap_pairs:
        rap_alloc_by_lot[int(lid)] = float(s or 0.0)

    for lot in lots:
        gross = float(lot.weight or 0.0)
        rap_taken = rap_alloc_by_lot.get(lot.id, 0.0)
        qty = max(gross - rap_taken, 0.0)
        if qty <= 0:
            continue
        val = qty * float(lot.unit_cost or 0.0)
        if (lot.grade or "KRIP") == "KRFS":
            stock["KRFS_qty"] += qty; stock["KRFS_val"] += val
        else:
            stock["KRIP_qty"] += qty; stock["KRIP_val"] += val

    # NEW: lowercase keys for the template (fixes NameError)
    lots_stock = {
        "krip_qty": stock.get("KRIP_qty", 0.0),
        "krip_val": stock.get("KRIP_val", 0.0),
        "krfs_qty": stock.get("KRFS_qty", 0.0),
        "krfs_val": stock.get("KRFS_val", 0.0),
    }

    # Date range defaults for the toolbar above Lots table
    s = start or today.isoformat()
    e = end or today.isoformat()
    try:
        s_date = dt.date.fromisoformat(s)
        e_date = dt.date.fromisoformat(e)
    except Exception:
        s_date, e_date = today, today  # kept for future use

    # NEW: read error banner text (if redirected with ?err=...)
    err = request.query_params.get("err")

    return templates.TemplateResponse(
        "atomization.html",
        {
            "request": request,
            "heats": heats,
            "lots": lots,
            "heat_grades": grades,
            "available_map": available_map,
            "today_iso": today.isoformat(),
            "start": s,
            "end": e,
            "atom_eff_today": eff_today,
            "atom_last5": last5,
            "atom_capacity": DAILY_CAPACITY_ATOM_KG,
            "atom_stock": stock,
            "lots_stock": lots_stock,
            "error_msg": err,   # <-- DO NOT MISS THIS
        }
    )


def _redir_err(msg: str) -> RedirectResponse:
    return RedirectResponse(f"/atomization?err={quote(msg)}", status_code=303)

@app.post("/atomization/new")
async def atom_new(
    request: Request,
    lot_weight: float = Form(3000.0),
    db: Session = Depends(get_db)
):
    try:
        form = await request.form()

        # ---- Parse allocations from form ----
        allocs: Dict[int, float] = {}
        for key, val in form.items():
            if key.startswith("alloc_"):
                try:
                    hid = int(key.split("_", 1)[1])
                    qty = float(val or 0)
                    if qty > 0:
                        allocs[hid] = qty
                except Exception:
                    pass

        if not allocs:
            return _alert_redirect("Enter allocation for at least one heat.")

        # ---- Fetch heats that were allocated ----
        heats = db.query(Heat).filter(Heat.id.in_(allocs.keys())).all()
        if not heats:
            return _alert_redirect("Selected heats not found.")

        # ---- Same-family rule (no KRIP & KRFS mixing) ----
        grades = {("KRFS" if heat_grade(h) == "KRFS" else "KRIP") for h in heats}
        if len(grades) > 1:
            return _alert_redirect("Mixing KRIP and KRFS in the same lot is not allowed.")

        # ---- Per-heat available check ----
        for h in heats:
            avail = heat_available(db, h)
            take = allocs.get(h.id, 0.0)
            if take > avail + 1e-6:
                return _alert_redirect(f"Over-allocation from heat {h.heat_no}. Available {avail:.1f} kg.")

        # ---- Total must equal lot weight (tiny tolerance) ----
        total_alloc = sum(allocs.values())
        tol = 0.05  # ~50 g to avoid float rounding issues
        if abs(total_alloc - float(lot_weight or 0.0)) > tol:
            return _alert_redirect(
                f"Allocated total ({total_alloc:.1f} kg) must equal Lot Weight ({float(lot_weight or 0):.1f} kg)."
            )

        # ---- Determine lot grade (any KRFS -> KRFS) ----
        any_fesi = any(heat_grade(h) == "KRFS" for h in heats)
        grade = "KRFS" if any_fesi else "KRIP"

        # ---- Create lot number ----
        today = dt.date.today().strftime("%Y%m%d")
        seq = (db.query(func.count(Lot.id)).filter(Lot.lot_no.like(f"KR%{today}%")).scalar() or 0) + 1
        lot_no = f"{grade}-{today}-{seq:03d}"

        # ---- Create lot ----
        lot = Lot(lot_no=lot_no, weight=float(lot_weight or 0.0), grade=grade)
        db.add(lot)
        db.flush()

        # ---- Link allocations & update mirror usage ----
        for h in heats:
            q = allocs.get(h.id, 0.0)
            if q > 0:
                db.add(LotHeat(lot_id=lot.id, heat_id=h.id, qty=q))
                h.alloc_used = float(h.alloc_used or 0.0) + q

        # ---- Costing: weighted avg heat cost + atom + surcharge ----
        weighted_cost = sum((h.unit_cost or 0.0) * allocs.get(h.id, 0.0) for h in heats)
        avg_heat_unit_cost = (weighted_cost / total_alloc) if total_alloc > 1e-9 else 0.0
        lot.unit_cost = avg_heat_unit_cost + ATOMIZATION_COST_PER_KG + SURCHARGE_PER_KG
        lot.total_cost = lot.unit_cost * (lot.weight or 0.0)

        # ---- Chemistry average (unchanged) ----
        sums = {k: 0.0 for k in ["c", "si", "s", "p", "cu", "ni", "mn", "fe"]}
        for h in heats:
            q = allocs.get(h.id, 0.0)
            if q <= 0 or not h.chemistry:
                continue
            for k in list(sums.keys()):
                try:
                    v = float(getattr(h.chemistry, k) or "")
                    sums[k] += v * q
                except Exception:
                    pass
        avg = {k: (sums[k] / total_alloc) if total_alloc > 1e-9 else None for k in sums.keys()}
        lc = LotChem(lot=lot, **{k: (str(v) if v is not None else "") for k, v in avg.items()})
        db.add(lc)

        db.commit()
        return RedirectResponse("/atomization", status_code=303)

    except Exception as e:
        db.rollback()
        return _alert_redirect(f"Unexpected error while creating lot: {type(e).__name__}")




# ---------- CSV export for atomization lots (NEW) ----------
@app.get("/atomization/export")
def atom_export(
    start: Optional[str] = None,
    end: Optional[str] = None,
    db: Session = Depends(get_db),
):
    lots = db.query(Lot).order_by(Lot.id.asc()).all()
    s = dt.date.fromisoformat(start) if start else None
    e = dt.date.fromisoformat(end) if end else None

    out = io.StringIO()
    out.write("Lot No,Date,Grade,QA,Weight kg,Unit Cost,Total Cost\n")
    for lot in lots:
        d = lot_date_from_no(lot.lot_no) or dt.date.today()
        if s and d < s:
            continue
        if e and d > e:
            continue
        out.write(
            f"{lot.lot_no},{d.isoformat()},{lot.grade or ''},{lot.qa_status or ''},"
            f"{lot.weight or 0:.1f},{lot.unit_cost or 0:.2f},{lot.total_cost or 0:.2f}\n"
        )
    data = out.getvalue().encode("utf-8")
    filename = f"atomization_report_{(start or '').replace('-', '')}_{(end or '').replace('-', '')}.csv"
    if not filename.strip("_").strip():
        filename = "atomization_report.csv"
    return StreamingResponse(
        io.BytesIO(data),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

# ---------- CSV export for atomization downtime (NEW) ----------
@app.get("/atomization/downtime/export")
def atom_downtime_export(db: Session = Depends(get_db)):
    out = io.StringIO()
    out.write("Source,Date,Minutes,Kind,Remarks\n")
    days = db.query(AtomDowntime).order_by(AtomDowntime.date.asc(), AtomDowntime.id.asc()).all()
    for r in days:
        out.write(f"DAY,{r.date.isoformat()},{int(r.minutes or 0)},{r.kind or ''},{(r.remarks or '').replace(',', ' ')}\n")
    data = out.getvalue().encode("utf-8")
    return StreamingResponse(
        io.BytesIO(data),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="atom_downtime_export.csv"'}
    )

# -------------------------------------------------
# QA Dashboard
# -------------------------------------------------
@app.get("/qa-dashboard", response_class=HTMLResponse)
def qa_dashboard(request: Request, db: Session = Depends(get_db)):
    heats = db.query(Heat).order_by(Heat.id.desc()).all()
    lots = db.query(Lot).order_by(Lot.id.desc()).all()
    heat_grades = {h.id: heat_grade(h) for h in heats}
    return templates.TemplateResponse("qa_dashboard.html", {"request": request, "heats": heats, "lots": lots, "heat_grades": heat_grades})

# -------------------------------------------------
# Traceability - LOT (existing)
# -------------------------------------------------
@app.get("/traceability/lot/{lot_id}", response_class=HTMLResponse)
def trace_lot(lot_id: int, request: Request, db: Session = Depends(get_db)):
    lot = db.get(Lot, lot_id)
    if not lot:
        return PlainTextResponse("Lot not found", status_code=404)

    # Allocation qty per heat for this lot
    alloc_rows = db.query(LotHeat).filter(LotHeat.lot_id == lot.id).all()
    alloc_map = {r.heat_id: float(r.qty or 0.0) for r in alloc_rows}
    heats = [db.get(Heat, r.heat_id) for r in alloc_rows]

    # FIFO GRN rows (unchanged)
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
    return templates.TemplateResponse(
        "trace_lot.html",
        {
            "request": request,
            "lot": lot,
            "heats": heats,
            "alloc_map": alloc_map,   # NEW: qty used from each heat for THIS lot
            "grn_rows": rows
        }
    )


# -------------------------------------------------
# Traceability - HEAT (for trace_heat.html)
# -------------------------------------------------
@app.get("/traceability/heat/{heat_id}", response_class=HTMLResponse)
def trace_heat(heat_id: int, request: Request, db: Session = Depends(get_db)):
    heat = (
        db.query(Heat)
        .options(joinedload(Heat.rm_consumptions).joinedload(HeatRM.grn))
        .filter(Heat.id == heat_id)
        .first()
    )
    if not heat:
        return PlainTextResponse("Heat not found", status_code=404)

    by_rm: Dict[str, List[Tuple[int, float, str]]] = {}
    for c in heat.rm_consumptions:
        supp = c.grn.supplier if c.grn else ""
        by_rm.setdefault(c.rm_type, []).append((c.grn_id, float(c.qty or 0.0), supp))

    return templates.TemplateResponse(
        "trace_heat.html",
        {"request": request, "heat": heat, "by_rm": by_rm}
    )

# -------------------------------------------------
# Day-level downtime pages (Melting & Atomization)
# -------------------------------------------------
@app.get("/melting/downtime", response_class=HTMLResponse)
def downtime_page(request: Request, db: Session = Depends(get_db)):
    today = dt.date.today()
    last = db.query(Downtime).order_by(Downtime.date.desc(), Downtime.id.desc()).limit(50).all()
    return templates.TemplateResponse(
        "downtime.html",
        {"request": request, "today": today.isoformat(), "rows": last}
    )

@app.post("/melting/downtime")
def downtime_add(
    date: str = Form(...),
    minutes: int = Form(...),
    kind: str = Form("PRODUCTION"),
    remarks: str = Form(""),
    db: Session = Depends(get_db),
):
    d = dt.date.fromisoformat(date)
    minutes = max(int(minutes), 0)
    db.add(Downtime(date=d, minutes=minutes, kind=kind, remarks=remarks))
    db.commit()
    return RedirectResponse("/melting/downtime", status_code=303)

# Atomization downtime page (renders your atom_down.html)
@app.get("/atomization/downtime", response_class=HTMLResponse)
def atom_downtime_page(request: Request, db: Session = Depends(get_db)):
    today = dt.date.today()
    last = db.query(AtomDowntime).order_by(AtomDowntime.date.desc(), AtomDowntime.id.desc()).limit(50).all()
    return templates.TemplateResponse(
        "atom_down.html",
        {"request": request, "today": today.isoformat(), "rows": last}
    )

@app.post("/atomization/downtime")
def atom_downtime_add(
    date: str = Form(...),
    minutes: int = Form(...),
    kind: str = Form("PRODUCTION"),
    remarks: str = Form(""),
    db: Session = Depends(get_db),
):
    d = dt.date.fromisoformat(date)
    minutes = max(int(minutes), 0)
    db.add(AtomDowntime(date=d, minutes=minutes, kind=kind, remarks=remarks))
    db.commit()
    return RedirectResponse("/atomization/downtime", status_code=303)

# -------------------------------------------------
# RAP – Ready After Atomization (final)
# -------------------------------------------------
@app.get("/rap", response_class=HTMLResponse)
def rap_page(request: Request, db: Session = Depends(get_db)):
    """
    Show APPROVED atomization lots in RAP, ensure RAPLot rows exist,
    and compute grade-wise available KPIs.
    """
    today = dt.date.today()

    # Bring in all APPROVED lots
    lots = (
        db.query(Lot)
        .filter(Lot.qa_status == "APPROVED")
        .order_by(Lot.id.desc())
        .all()
    )

    # Ensure RAP rows exist & are up to date
    rap_rows: List[RAPLot] = []
    for lot in lots:
        rap_rows.append(ensure_rap_lot(db, lot))
    db.commit()  # persist any ensure updates

    # KPIs: Available stock & value in RAP (ready stock)
    kpi = {"KRIP_qty": 0.0, "KRIP_val": 0.0, "KRFS_qty": 0.0, "KRFS_val": 0.0}
    for rap in rap_rows:
        lot = rap.lot
        qty = float(rap.available_qty or 0.0)
        if qty <= 0:
            continue
        val = qty * float(lot.unit_cost or 0.0)
        if (lot.grade or "KRIP") == "KRFS":
            kpi["KRFS_qty"] += qty
            kpi["KRFS_val"] += val
        else:
            kpi["KRIP_qty"] += qty
            kpi["KRIP_val"] += val

    # --- Monthly RAP KPI Trends ---
    month_start = today.replace(day=1)

    trend = (
        db.query(RAPAlloc.kind, Lot.grade, func.sum(RAPAlloc.qty))
        .join(RAPLot, RAPAlloc.rap_lot_id == RAPLot.id)
        .join(Lot, RAPLot.lot_id == Lot.id)
        .filter(RAPAlloc.date >= month_start)
        .group_by(RAPAlloc.kind, Lot.grade)
        .all()
    )

    kpi_trends = {
        "DISPATCH": {"KRIP": 0, "KRFS": 0},
        "PLANT2": {"KRIP": 0, "KRFS": 0},
    }
    for kind, grade, qty in trend:
        kpi_trends[kind][grade or "KRIP"] = float(qty or 0)

    # compute date boundaries for the allocation form
    today_iso = today.isoformat()
    min_date_iso = (today - dt.timedelta(days=3)).isoformat()

    # ✅ Correct indentation here (only 4 spaces in, not 8)
    return templates.TemplateResponse(
        "rap.html",
        {
            "request": request,
            "rows": rap_rows,
            "kpi": kpi,
            "kpi_trends": kpi_trends,
            "today": today_iso,
            "min_date": min_date_iso,
        }
    )
# -------------------------------------------------
# RAP per-lot allocation (Dispatch / Plant 2)
# -------------------------------------------------
@app.post("/rap/allocate")
def rap_allocate(
    rap_lot_id: int = Form(...),
    date: str = Form(...),
    kind: str = Form(...),            # "DISPATCH" or "PLANT2"
    qty: float = Form(...),
    dest: str = Form(""),
    remarks: str = Form(""),
    db: Session = Depends(get_db),
):
    # Fetch RAP lot
    rap = db.get(RAPLot, rap_lot_id)
    if not rap:
        return _alert_redirect("RAP lot not found.", url="/rap")

    # Date validation: only today or last 3 days; no future
    try:
        d = dt.date.fromisoformat(date)
    except Exception:
        return _alert_redirect("Invalid date.", url="/rap")
    today = dt.date.today()
    if d > today or d < (today - dt.timedelta(days=3)):
        return _alert_redirect("Date must be today or within the last 3 days.", url="/rap")

    # Quantity must be > 0
    try:
        qty = float(qty or 0.0)
    except Exception:
        qty = 0.0
    if qty <= 0:
        return _alert_redirect("Quantity must be > 0.", url="/rap")

    # Underlying lot must be APPROVED
    lot = db.get(Lot, rap.lot_id)
    if not lot or (lot.qa_status or "") != "APPROVED":
        return _alert_redirect("Underlying lot is not APPROVED.", url="/rap")

    # Available = lot.weight - total RAP allocations
    # If you already have rap_total_alloc_qty_for_lot(), this will use it.
    try:
        total_alloc = rap_total_alloc_qty_for_lot(db, lot.id)
    except NameError:
        total_alloc = (
            db.query(func.coalesce(func.sum(RAPAlloc.qty), 0.0))
              .join(RAPLot, RAPAlloc.rap_lot_id == RAPLot.id)
              .filter(RAPLot.lot_id == lot.id)
              .scalar()
            or 0.0
        )

    avail = max(float(lot.weight or 0.0) - float(total_alloc or 0.0), 0.0)
    if qty > avail + 1e-6:
        return _alert_redirect(f"Over-allocation. Available {avail:.1f} kg.", url="/rap")

    # Kind & destination rules
    kind = (kind or "").upper()
    if kind not in ("DISPATCH", "PLANT2"):
        kind = "DISPATCH"
    if kind == "DISPATCH":
        if not (dest and dest.strip()):
            return _alert_redirect("Customer name is required for Dispatch.", url="/rap")
    else:
        dest = "Plant 2"

    # Insert allocation and flush to get id
    rec = RAPAlloc(
        rap_lot_id=rap.id,
        date=d,
        kind=kind,
        qty=qty,
        remarks=remarks,
        dest=dest,
    )
    db.add(rec)
    db.flush()  # rec.id is now available

    # Update RAPLot mirror
    rap.available_qty = max(avail - qty, 0.0)
    rap.status = "CLOSED" if rap.available_qty <= 1e-6 else "OPEN"
    db.add(rap)
    db.commit()

    # If dispatch, open the Dispatch Note PDF; else return to RAP
    if rec.kind == "DISPATCH":
        return RedirectResponse(f"/rap/dispatch/{rec.id}/pdf", status_code=303)
    return RedirectResponse("/rap", status_code=303)

# -------------------------------------------------
# RAP Transfer CSV export (Plant-2 only)
# -------------------------------------------------
@app.get("/rap/transfer/export")
def export_rap_transfers(db: Session = Depends(get_db)):
    import csv
    from io import StringIO
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Date", "Lot No", "Grade", "Qty", "Unit Cost", "Customer"])

    transfers = (
        db.query(RAPAlloc)
        .join(RAPLot)
        .filter(RAPAlloc.kind == "PLANT2")
        .order_by(RAPAlloc.date)
        .all()
    )
    for a in transfers:
        lot = a.raplot
        writer.writerow([
            a.date.isoformat() if a.date else "",
            lot.lot_no if lot else "",
            lot.grade if lot else "",
            f"{a.qty:.2f}",
            f"{(lot.unit_cost or 0.0):.2f}" if lot else "",
            a.dest or "",
        ])

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=rap_transfers.csv"},
    )
    
# ---------- CSV export for Dispatch movements ----------


@app.get("/rap/dispatch/export")
def export_rap_dispatch(db: Session = Depends(get_db)):
    import csv, io
    from fastapi.responses import StreamingResponse

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "Date", "Lot", "Grade", "Qty (kg)", "Unit Cost (₹/kg)",
        "Value (₹)", "Customer", "Remarks", "Alloc ID"
    ])

    rows = (
        db.query(RAPAlloc, RAPLot, Lot)
          .join(RAPLot, RAPAlloc.rap_lot_id == RAPLot.id)
          .join(Lot, RAPLot.lot_id == Lot.id)
          .filter(RAPAlloc.kind == "DISPATCH")
          .order_by(RAPAlloc.date.asc(), RAPAlloc.id.asc())
          .all()
    )

    for alloc, rap, lot in rows:
        qty = float(alloc.qty or 0.0)
        unit = float(lot.unit_cost or 0.0)
        val = qty * unit
        writer.writerow([
            (alloc.date or dt.date.today()).isoformat(),
            lot.lot_no or "",
            lot.grade or "",
            f"{qty:.1f}",
            f"{unit:.2f}",
            f"{val:.2f}",
            alloc.dest or "",
            (alloc.remarks or "").replace(",", " "),
            alloc.id,
        ])

    buf.seek(0)
    return StreamingResponse(
        io.StringIO(buf.getvalue()),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="dispatch_movements.csv"'},
    )

# ---------- CSV export for Plant-2 transfers ----------
@app.get("/rap/transfer/export")
def export_rap_transfers(db: Session = Depends(get_db)):
    import csv, io
    from fastapi.responses import StreamingResponse

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Date", "Lot", "Grade", "Qty (kg)", "Unit Cost (₹/kg)", "Value (₹)"])

    rows = (
        db.query(RAPAlloc)
          .filter(RAPAlloc.kind == "PLANT2")
          .order_by(RAPAlloc.date.asc(), RAPAlloc.id.asc())
          .all()
    )

    for a in rows:
        rap = a.rap_lot                     # ✅ fixed
        lot = rap.lot if rap else None
        qty = float(a.qty or 0.0)
        unit = float(lot.unit_cost or 0.0) if lot else 0.0
        value = qty * unit
        writer.writerow([
            (a.date.isoformat() if a.date else ""),
            (lot.lot_no if lot else ""),
            (lot.grade if lot else ""),
            f"{qty:.1f}",
            f"{unit:.2f}",
            f"{value:.2f}",
        ])

    buf.seek(0)
    return StreamingResponse(
        io.StringIO(buf.getvalue()),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="plant2_transfers.csv"'},
    )
        
# -------------------------------------------------
# Lot quick views used by RAP "Docs" column
# -------------------------------------------------
@app.get("/lot/{lot_id}/trace", response_class=HTMLResponse)
def lot_trace_view(lot_id: int, db: Session = Depends(get_db)):
    lot = db.get(Lot, lot_id)
    if not lot:
        return HTMLResponse("<p>Lot not found</p>", status_code=404)

    rows = []
    for lh in getattr(lot, "heats", []):
        h = db.get(Heat, lh.heat_id)
        if not h:
            continue
        # parent row = heat
        rows.append({
            "heat_no": h.heat_no,
            "alloc": float(lh.qty or 0.0),
            "out": float(h.actual_output or 0.0),
            "qa": h.qa_status or "",
            "rm_type": "",
            "grn_id": "",
            "supplier": "",
            "rm_qty": ""
        })
        # children rows = FIFO GRNs
        for cons in getattr(h, "rm_consumptions", []):
            g = cons.grn
            rows.append({
                "heat_no": "",
                "alloc": "",
                "out": "",
                "qa": "",
                "rm_type": cons.rm_type,
                "grn_id": cons.grn_id,
                "supplier": (g.supplier if g else ""),
                "rm_qty": f"{float(cons.qty or 0.0):.1f}"
            })

    html = [
        "<h3>Trace – Lot ", lot.lot_no or "", "</h3>",
        "<table border=1 cellpadding=6 cellspacing=0>",
        "<thead><tr>",
        "<th>Heat</th><th>Alloc to Lot (kg)</th><th>Heat Out (kg)</th><th>QA</th>",
        "<th>RM Type</th><th>GRN #</th><th>Supplier</th><th>RM Qty (kg)</th>",
        "</tr></thead><tbody>"
    ]
    for r in rows:
        html.append(
            f"<tr><td>{r['heat_no']}</td><td>{r['alloc']}</td><td>{r['out']}</td><td>{r['qa']}</td>"
            f"<td>{r['rm_type']}</td><td>{r['grn_id']}</td><td>{r['supplier']}</td><td>{r['rm_qty']}</td></tr>"
        )
    html.append("</tbody></table>")
    html.append('<p style="margin-top:12px"><a href="/rap">← Back to RAP</a></p>')
    return HTMLResponse("".join(html))
    
@app.get("/lot/{lot_id}/qa", response_class=HTMLResponse)
def lot_qa_view(lot_id: int, db: Session = Depends(get_db)):
    lot = db.get(Lot, lot_id)
    if not lot:
        return PlainTextResponse("Lot not found", status_code=404)

    # helpers to show value or dash
    def v(x): return ("—" if x in (None, "",) else x)

    rows = []
    for lh in getattr(lot, "heats", []):
        h = db.get(Heat, lh.heat_id)
        if not h:
            continue
        rows.append({
            "heat_no": h.heat_no,
            "qa": getattr(h, "qa_status", "") or "—",
            "notes": getattr(h, "qa_notes", "") if hasattr(h, "qa_notes") else "—",
        })

    # Pull QA blocks from Lot if present (adjust attribute names to your models)
    chem = getattr(lot, "chemistry", None)
    phys = getattr(lot, "phys", None)
    psd  = getattr(lot, "psd", None)

    html = [
        f"<h2>QA snapshot – Lot {lot.lot_no}</h2>",
        "<table border='1' cellpadding='6' cellspacing='0'>",
        "<tr><th>Heat</th><th>QA</th><th>Notes</th></tr>",
    ]
    for r in rows:
        html.append(f"<tr><td>{r['heat_no']}</td><td>{r['qa']}</td><td>{r['notes']}</td></tr>")
    if not rows:
        html.append("<tr><td colspan='3'>No heats recorded.</td></tr>")
    html.append("</table><br>")

    # Chemistry table
    html += ["<h3>Chemistry</h3>",
             "<table border='1' cellpadding='6' cellspacing='0'>",
             "<tr><th>C</th><th>Si</th><th>S</th><th>P</th><th>Cu</th><th>Ni</th><th>Mn</th><th>Fe</th></tr>"]
    if chem:
        html.append(f"<tr><td>{v(getattr(chem,'c',None))}</td>"
                    f"<td>{v(getattr(chem,'si',None))}</td>"
                    f"<td>{v(getattr(chem,'s',None))}</td>"
                    f"<td>{v(getattr(chem,'p',None))}</td>"
                    f"<td>{v(getattr(chem,'cu',None))}</td>"
                    f"<td>{v(getattr(chem,'ni',None))}</td>"
                    f"<td>{v(getattr(chem,'mn',None))}</td>"
                    f"<td>{v(getattr(chem,'fe',None))}</td></tr>")
    else:
        html.append("<tr><td colspan='8'>—</td></tr>")
    html.append("</table><br>")

    # Physical table
    html += ["<h3>Physical</h3>",
             "<table border='1' cellpadding='6' cellspacing='0'>",
             "<tr><th>AD (g/cc)</th><th>Flow (s/50g)</th></tr>"]
    if phys:
        html.append(f"<tr><td>{v(getattr(phys,'ad',None))}</td><td>{v(getattr(phys,'flow',None))}</td></tr>")
    else:
        html.append("<tr><td colspan='2'>—</td></tr>")
    html.append("</table><br>")

    # PSD table
    html += ["<h3>PSD</h3>",
             "<table border='1' cellpadding='6' cellspacing='0'>",
             "<tr><th>+212</th><th>+180</th><th>-180+150</th><th>-150+75</th><th>-75+45</th><th>-45</th></tr>"]
    if psd:
        html.append(
            f"<tr><td>{v(getattr(psd,'p212',None))}</td>"
            f"<td>{v(getattr(psd,'p180',None))}</td>"
            f"<td>{v(getattr(psd,'n180p150',None))}</td>"
            f"<td>{v(getattr(psd,'n150p75',None))}</td>"
            f"<td>{v(getattr(psd,'n75p45',None))}</td>"
            f"<td>{v(getattr(psd,'n45',None))}</td></tr>"
        )
    else:
        html.append("<tr><td colspan='6'>—</td></tr>")
    html.append("</table><p><a href='/rap'>← Back to RAP</a></p>")
    return HTMLResponse("".join(html))

@app.get("/lot/{lot_id}/pdf")
def lot_pdf_view(lot_id: int, db: Session = Depends(get_db)):
    """Minimal 1-page summary PDF for the lot (separate from Dispatch Note)."""
    lot = db.get(Lot, lot_id)
    if not lot:
        return PlainTextResponse("Lot not found", status_code=404)

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    # simple header (re-use your existing helper if you have one)
    try:
        draw_header(c, "Lot Summary")
    except Exception:
        c.setFont("Helvetica-Bold", 18)
        c.drawString(2 * cm, h - 2.5 * cm, "KRN Alloys Pvt. Ltd")
        c.setFont("Helvetica", 12)
        c.drawString(2 * cm, h - 3.2 * cm, "Lot Summary")

    y = h - 4 * cm
    c.setFont("Helvetica", 11)
    c.drawString(2 * cm, y, f"Lot: {lot.lot_no}    Grade: {lot.grade or '-'}"); y -= 14
    c.drawString(2 * cm, y, f"Lot Weight: {float(lot.weight or 0):.1f} kg"); y -= 14
    c.drawString(2 * cm, y, f"Unit Cost: ₹{float(lot.unit_cost or 0):.2f}"); y -= 20

    # Heats + GRN (FIFO) section
    c.setFont("Helvetica-Bold", 11)
    c.drawString(2 * cm, y, "Heats & GRN (FIFO)"); y -= 14
    c.setFont("Helvetica", 10)

    def new_page():
        nonlocal y
        c.showPage()
        # re-draw header
        try:
            draw_header(c, "Lot Summary")
        except Exception:
            c.setFont("Helvetica-Bold", 18)
            c.drawString(2 * cm, h - 2.5 * cm, "KRN Alloys Pvt. Ltd")
            c.setFont("Helvetica", 12)
            c.drawString(2 * cm, h - 3.2 * cm, "Lot Summary")
        y = h - 4 * cm
        c.setFont("Helvetica-Bold", 11)
        c.drawString(2 * cm, y, "Heats & GRN (FIFO)"); y -= 14
        c.setFont("Helvetica", 10)

    for lh in getattr(lot, "heats", []):
        hobj = db.get(Heat, lh.heat_id)
        if not hobj:
            continue
        # Heat row
        if y < 3 * cm:
            new_page()
        c.drawString(2.2 * cm, y, f"{hobj.heat_no} | Alloc: {float(lh.qty or 0):.1f} kg | QA: {hobj.qa_status or ''}")
        y -= 12

        # GRN (FIFO) child rows under this heat
        for cons in getattr(hobj, "rm_consumptions", []):
            if y < 3 * cm:
                new_page()
            g = cons.grn
            supplier = (g.supplier if g else "")
            c.drawString(
                2.8 * cm, y,
                f"– {cons.rm_type} | GRN #{cons.grn_id} | {supplier} | {float(cons.qty or 0):.1f} kg"
            )
            y -= 12

    c.showPage(); c.save()
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="lot_{lot.lot_no}.pdf"'}
    )

# -------------------------------------------------
# RAP Dispatch + Transfer
# -------------------------------------------------
@app.get("/rap/dispatch/new", response_class=HTMLResponse)
def rap_dispatch_new(request: Request, db: Session = Depends(get_db)):
    lots = db.query(Lot).filter(Lot.qa_status == "APPROVED").all()
    return templates.TemplateResponse("rap_dispatch_new.html", {"request": request, "lots": lots})

# ---------- Multi-lot dispatch: create items + reduce RAP availability ----------
from fastapi import Request

@app.post("/rap/dispatch/save")
async def rap_dispatch_save(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    customer = (form.get("customer") or "").strip()
    if not customer:
        return _alert_redirect("Customer is required.", url="/rap")

    # selected RAPLot ids
    lot_ids = form.getlist("lot_ids")
    if not lot_ids:
        return _alert_redirect("Select at least one lot.", url="/rap")

    # date
    try:
        d = dt.date.fromisoformat(form.get("date") or "")
    except Exception:
        d = dt.date.today()

    # total qty (optional check – UI already enforces)
    total_qty = float(form.get("total_qty") or 0.0)

    # Build items, enforce single grade & availability
    items = []
    sel_grade = None
    for rid in lot_ids:
        rap = db.get(RAPLot, int(rid))
        if not rap:
            return _alert_redirect("RAP lot not found.", url="/rap")
        lot = db.get(Lot, rap.lot_id)
        if not lot or (lot.qa_status or "") != "APPROVED":
            return _alert_redirect("Underlying lot is not APPROVED.", url="/rap")

        q = float(form.get(f"qty_{rid}") or 0.0)
        if q <= 0:
            return _alert_redirect("Enter quantity for every selected lot.", url="/rap")

        # recompute available
        total_alloc = rap_total_alloc_qty_for_lot(db, lot.id)
        avail = max(float(lot.weight or 0.0) - float(total_alloc or 0.0), 0.0)
        if q > avail + 1e-6:
            return _alert_redirect(f"Over-allocation on {lot.lot_no}. Available {avail:.1f} kg.", url="/rap")

        g = (lot.grade or "KRIP")
        sel_grade = sel_grade or g
        if g != sel_grade:
            return _alert_redirect("Select lots of a single grade per dispatch.", url="/rap")

        items.append((rap, lot, q))

    if total_qty > 0:
        s = sum(q for _, _, q in items)
        if abs(total_qty - s) > 0.05:
            return _alert_redirect("Total Dispatch Qty must equal the sum of selected lot quantities.", url="/rap")

    # Create RAPDispatch + items
    disp = RAPDispatch(date=d, customer=customer, grade=sel_grade, total_qty=0.0, total_cost=0.0)
    db.add(disp); db.flush()

    total_qty_acc = 0.0
    total_cost_acc = 0.0

    for rap, lot, q in items:
        db.add(RAPDispatchItem(dispatch_id=disp.id, lot_id=lot.id, qty=q, cost=float(lot.unit_cost or 0.0)))
        total_qty_acc += q
        total_cost_acc += q * float(lot.unit_cost or 0.0)

        # Mirror a DISPATCH allocation (so RAP available reduces)
        alloc = RAPAlloc(
            rap_lot_id=rap.id,
            date=d,
            kind="DISPATCH",
            qty=q,
            remarks=f"Dispatch #{disp.id}",
            dest=customer,
        )
        db.add(alloc)
        db.flush()

        # Update RAPLot mirror
        # (available = lot.weight - sum(alloc))
        total_alloc_after = rap_total_alloc_qty_for_lot(db, lot.id)
        rap.available_qty = max(float(lot.weight or 0.0) - total_alloc_after, 0.0)
        rap.status = "CLOSED" if rap.available_qty <= 1e-6 else "OPEN"
        db.add(rap)

    disp.total_qty = total_qty_acc
    disp.total_cost = total_cost_acc
    db.add(disp)
    db.commit()

    # multi-lot PDF
    return RedirectResponse(f"/rap/dispatch/pdf/{disp.id}", status_code=303)

@app.get("/rap/dispatch/pdf/{disp_id}")
def rap_dispatch_pdf(disp_id: int, db: Session = Depends(get_db)):
    from .rap_dispatch_pdf import draw_dispatch_note
    disp = db.get(RAPDispatch, disp_id)
    if not disp:
        return PlainTextResponse("Dispatch not found", status_code=404)
    items = db.query(RAPDispatchItem).filter(RAPDispatchItem.dispatch_id == disp_id).all()
    buf = io.BytesIO()
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(buf, pagesize=A4)
    draw_dispatch_note(c, disp, items, db)
        # clickable "Back to RAP" text on the PDF (top-right)
    c.setFont("Helvetica", 9)
    c.drawRightString(20.0*cm, 1.2*cm, "Back to RAP")
    c.linkURL("/rap", (18.5*cm, 0.9*cm, 20.0*cm, 1.3*cm), relative=1)
    c.showPage(); c.save()
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename=\"dispatch_{disp.id}.pdf\"'})

@app.get("/rap/transfer/new", response_class=HTMLResponse)
def rap_transfer_new(request: Request, db: Session = Depends(get_db)):
    lots = db.query(Lot).filter(Lot.qa_status == "APPROVED").all()
    return templates.TemplateResponse("rap_transfer_new.html", {"request": request, "lots": lots})

@app.post("/rap/transfer/save")
def rap_transfer_save(
    lot_id: int = Form(...), qty: float = Form(...), remarks: str = Form(""),
    db: Session = Depends(get_db)
):
    today = dt.date.today()
    t = RAPTransfer(date=today, lot_id=lot_id, qty=qty, remarks=remarks)
    db.add(t); db.commit()
    return RedirectResponse("/rap", status_code=303)

# -------------------------------------------------
# RAP Dispatch Note PDF
# -------------------------------------------------
@app.get("/rap/dispatch/{alloc_id}/pdf")
def rap_dispatch_pdf(alloc_id: int, db: Session = Depends(get_db)):
    # Load allocation + linked RAPLot + Lot
    alloc = db.get(RAPAlloc, alloc_id)
    if not alloc:
        return PlainTextResponse("Dispatch allocation not found.", status_code=404)

    rap = db.get(RAPLot, alloc.rap_lot_id)
    if not rap:
        return PlainTextResponse("RAP lot not found.", status_code=404)

    lot = db.get(Lot, rap.lot_id)
    if not lot:
        return PlainTextResponse("Lot not found.", status_code=404)

    # Collect trace items (heats & FIFO rows) for annexure
    heats = []
    fifo_rows = []
    for lh in lot.heats:
        h = db.get(Heat, lh.heat_id)
        if not h:
            continue
        heats.append((h, float(lh.qty or 0.0)))
        for cons in h.rm_consumptions:
            g = cons.grn
            fifo_rows.append({
                "heat_no": h.heat_no,
                "rm_type": cons.rm_type,
                "grn_id": cons.grn_id,
                "supplier": g.supplier if g else "",
                "qty": float(cons.qty or 0.0)
            })

    # Make PDF
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    # Header
    draw_header(c, "Dispatch Note")

    # Top fields
    y = height - 4 * cm
    c.setFont("Helvetica", 11)
    c.drawString(2 * cm, y, f"Date/Time: {(alloc.date or dt.date.today()).isoformat()}  {dt.datetime.now().strftime('%H:%M')}"); y -= 14
    c.drawString(2 * cm, y, f"Customer: {alloc.dest or ''}"); y -= 14
    c.drawString(2 * cm, y, f"Note Type: DISPATCH"); y -= 14

    # Lot summary
    y -= 6
    c.setFont("Helvetica-Bold", 11); c.drawString(2 * cm, y, "Lot Details"); y -= 14
    c.setFont("Helvetica", 10)
    c.drawString(2 * cm, y, f"Lot: {lot.lot_no}    Grade: {lot.grade or '-'}"); y -= 12
    c.drawString(2 * cm, y, f"Allocated Qty (kg): {float(alloc.qty or 0):.1f}"); y -= 12
    c.drawString(2 * cm, y, f"Cost / kg (₹): {float(lot.unit_cost or 0):.2f}    Cost Value (₹): {float(lot.unit_cost or 0) * float(alloc.qty or 0):.2f}"); y -= 16

    # Blank sell-rate placeholder
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(2 * cm, y, "SELL RATE (₹/kg): _________________________     Amount (₹): _________________________"); y -= 18

    # Annexure – QA & Trace
    c.setFont("Helvetica-Bold", 11); c.drawString(2 * cm, y, "Annexure: QA Certificates & GRN Trace"); y -= 14
    c.setFont("Helvetica", 10)

    # Heats table (lot allocations)
    c.drawString(2 * cm, y, "Heats used in this lot (lot allocation vs. heat out / QA):"); y -= 12
    for h, qalloc in heats:
        c.drawString(2.2 * cm, y,
            f"{h.heat_no}  | Alloc to lot: {qalloc:.1f} kg  | Heat Out: {float(h.actual_output or 0):.1f} kg  | QA: {h.qa_status or ''}"
        )
        y -= 12
        if y < 3 * cm:
            c.showPage(); draw_header(c, "Dispatch Note"); y = height - 4 * cm

    # FIFO rows
    y -= 6
    c.setFont("Helvetica-Bold", 11); c.drawString(2 * cm, y, "GRN Consumption (FIFO)"); y -= 14
    c.setFont("Helvetica", 10)
    for r in fifo_rows:
        c.drawString(2.2 * cm, y,
            f"Heat {r['heat_no']} | {r['rm_type']} | GRN #{r['grn_id']} | {r['supplier']} | {r['qty']:.1f} kg"
        )
        y -= 12
        if y < 3 * cm:
            c.showPage(); draw_header(c, "Dispatch Note"); y = height - 4 * cm

    # Footer note & signature
    y -= 8
    c.setFont("Helvetica-Bold", 10)
    c.drawString(2 * cm, y, "This document is a Dispatch Note for Invoice purpose only."); y -= 16

    c.setFont("Helvetica", 10)
    c.drawString(2 * cm, y, "Authorized Sign: ____________________________"); y -= 24

        # --- QA Annexure for each lot in this dispatch ---
    for item in dispatch.items:
        draw_lot_qa_annexure(c, item.lot)

    c.showPage(); c.save()
    buf.seek(0)
    filename = f"Dispatch_{lot.lot_no}_{alloc.id}.pdf"
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'}
    )


# -------------------------------------------------
# PDF (no cost in PDF)
# -------------------------------------------------
def draw_header(c: canvas.Canvas, title: str):
    width, height = A4
    logo_w = 4 * cm
    logo_h = 3 * cm
    logo_x = 1.5 * cm
    logo_y = height - 3 * cm
    logo_path = os.path.join(os.path.dirname(__file__), "..", "static", "KRN_Logo.png")
    if os.path.exists(logo_path):
        c.drawImage(logo_path, logo_x, logo_y, width=logo_w, height=logo_h, preserveAspectRatio=True, mask="auto")

    c.setFont("Helvetica-Bold", 36)
    c.drawString(7 * cm, height - 2.0 * cm, "KRN Alloys Pvt Ltd")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(7 * cm, height - 2.7 * cm, title)
    c.line(1.5 * cm, height - 3.3 * cm, width - 1.5 * cm, height - 3.3 * cm)

def draw_lot_qa_annexure(c: canvas.Canvas, lot: Lot, start_y: float = None):
    """
    Add QA annexure pages (Chemistry, Physical, PSD) for a given lot.
    """
    width, height = A4
    y = start_y or (height - 4 * cm)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(2 * cm, y, f"QA Certificate – Lot {lot.lot_no}"); y -= 20

    # Chemistry
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Chemistry:"); y -= 16
    c.setFont("Helvetica", 10)
    if lot.chemistry:
        for k, v in vars(lot.chemistry).items():
            if k in ("id", "lot_id"): continue
            c.drawString(2.5 * cm, y, f"{k.upper()}: {v or ''}"); y -= 12
            if y < 3 * cm:
                c.showPage(); draw_header(c, f"QA Certificate – {lot.lot_no}"); y = height - 4 * cm
    else:
        c.drawString(2.5 * cm, y, "No chemistry data"); y -= 16

    # Physical
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Physical:"); y -= 16
    c.setFont("Helvetica", 10)
    if lot.phys:
        for k, v in vars(lot.phys).items():
            if k in ("id", "lot_id"): continue
            c.drawString(2.5 * cm, y, f"{k.upper()}: {v or ''}"); y -= 12
            if y < 3 * cm:
                c.showPage(); draw_header(c, f"QA Certificate – {lot.lot_no}"); y = height - 4 * cm
    else:
        c.drawString(2.5 * cm, y, "No physical data"); y -= 16

    # PSD
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Particle Size Distribution:"); y -= 16
    c.setFont("Helvetica", 10)
    if lot.psd:
        for k, v in vars(lot.psd).items():
            if k in ("id", "lot_id"): continue
            c.drawString(2.5 * cm, y, f"{k.upper()}: {v or ''}"); y -= 12
            if y < 3 * cm:
                c.showPage(); draw_header(c, f"QA Certificate – {lot.lot_no}"); y = height - 4 * cm
    else:
        c.drawString(2.5 * cm, y, "No PSD data"); y -= 16


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

    c.setFont("Helvetica-Bold", 11)
    c.drawString(2 * cm, y, "Heats (Allocation)")
    y -= 14

    c.setFont("Helvetica", 10)
    for lh in lot.heats:
        h = lh.heat
        c.drawString(
            2.2 * cm, y,
            f"{h.heat_no}  | Alloc to lot: {float(lh.qty or 0):.1f} kg  | Heat Out: {float(h.actual_output or 0):.1f} kg  | QA: {h.qa_status}"
        )
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
    return StreamingResponse(buf, media_type="application/pdf",
                             headers={"Content-Disposition": f'inline; filename="trace_{lot.lot_no}.pdf"'})
