import os, io, datetime as dt
from typing import Optional, Dict

# FastAPI
from fastapi import FastAPI, Request, Form, Depends, Response
from fastapi.responses import HTMLResponse, RedirectResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

# SQLAlchemy
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey, func, text
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session

# -------------------------------------------------
# Costing constants (baseline)
# -------------------------------------------------
MELT_COST_PER_KG_KRIP = 6.0
MELT_COST_PER_KG_KRFS = 8.0
ATOMIZATION_COST_PER_KG = 5.0
SURCHARGE_PER_KG = 2.0

# Melting capacity & power targets
DAILY_CAPACITY_KG = 7000.0          # 24h capacity
POWER_TARGET_KWH_PER_TON = 560.0    # target kWh/ton

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

        # GRN human-readable number (GRN-YYYYMMDD-###)
        if not _table_has_column(conn, "grn", "grn_no"):
            if str(engine.url).startswith("sqlite"):
                conn.execute(text("ALTER TABLE grn ADD COLUMN grn_no TEXT"))
            else:
                conn.execute(text("ALTER TABLE grn ADD COLUMN IF NOT EXISTS grn_no TEXT UNIQUE"))

        # Optional day-level downtime table (can be used in addition to per-heat downtime)
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
    grn_no = Column(String, unique=True, index=True)  # readable number
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
    power_kwh = Column(Float, default=0.0)     # user-entered kWh for the heat
    kwh_per_ton = Column(Float, default=0.0)   # computed = power_kwh / (actual_output/1000)
    downtime_min = Column(Integer, default=0)  # per-heat downtime minutes (0 allowed)
    downtime_type = Column(String)             # production/maintenance/power/other
    downtime_note = Column(String)             # remarks

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

# Optional day-level downtime table
class Downtime(Base):
    __tablename__ = "downtime"
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    minutes = Column(Integer, default=0)
    kind = Column(String)
    remarks = Column(String)

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

# expose python builtins to Jinja (prevents 'max is undefined' etc.)
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

def day_available_minutes(db: Session, day: dt.date) -> int:
    """Available minutes = 1440 - (sum per-heat downtime for the day + optional day-level downtime)."""
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
# GRN  (unchanged)
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
# Melting (enhanced)
# -------------------------------------------------
@app.get("/melting", response_class=HTMLResponse)
def melting_page(
    request: Request,
    start: Optional[str] = None,   # YYYY-MM-DD
    end: Optional[str] = None,     # YYYY-MM-DD
    db: Session = Depends(get_db),
):
    heats = db.query(Heat).order_by(Heat.id.desc()).all()

    # build helper rows + keep alloc_used mirror
    rows = []
    for h in heats:
        avail = heat_available(db, h)
        rows.append({
            "heat": h,
            "grade": heat_grade(h),
            "available": avail,
            "date": heat_date_from_no(h.heat_no) or dt.date.today(),
        })
    db.commit()

    today = dt.date.today()

    # today KPI: average kWh/ton across today's heats with output
    todays = [r["heat"] for r in rows if r["date"] == today and (r["heat"].actual_output or 0) > 0]
    kwhpt_vals = []
    for h in todays:
        if (h.power_kwh or 0) > 0 and (h.actual_output or 0) > 0:
            kwhpt_vals.append((h.power_kwh / h.actual_output) * 1000.0)
    today_kwhpt = (sum(kwhpt_vals) / len(kwhpt_vals)) if kwhpt_vals else 0.0

    # Last 5 days (including today): actual, target adjusted by per-heat downtime (+ optional day downtime)
    last5 = []
    for i in range(4, -1, -1):
        d = today - dt.timedelta(days=i)
        actual = sum((r["heat"].actual_output or 0.0) for r in rows if r["date"] == d)
        target = day_target_kg(db, d)
        last5.append({"date": d.isoformat(), "actual": actual, "target": target})

    # Live stock summary from heats (available qty & value) grouped by grade
    krip_qty = krip_val = krfs_qty = krfs_val = 0.0
    for r in rows:
        if r["available"] <= 0:
            continue
        val = r["available"] * (r["heat"].unit_cost or 0.0)
        if r["grade"] == "KRFS":
            krfs_qty += r["available"]; krfs_val += val
        else:
            krip_qty += r["available"]; krip_val += val

    # Extra KPIs for today
    today_inputs = 0.0
    today_outputs = 0.0
    for r in rows:
        if r["date"] == today:
            h = r["heat"]
            today_inputs  += (h.total_inputs or 0.0)
            today_outputs += (h.actual_output or 0.0)
    today_yield_pct = (today_outputs / today_inputs * 100.0) if today_inputs > 0 else 0.0
    yield_target_pct = 97.0  # 3% loss
    today_target_kg = day_target_kg(db, today)
    today_eff_pct = (today_outputs / today_target_kg * 100.0) if today_target_kg > 0 else 0.0

    s = start or today.isoformat()
    e = end or today.isoformat()

    return templates.TemplateResponse(
        "melting.html",
        {
            "request": request,
            "rm_types": RM_TYPES,
            "pending": [r["heat"] for r in rows],
            "heat_grades": {r["heat"].id: r["grade"] for r in rows},
            "today_kwhpt": today_kwhpt,
            "last5": last5,
            "stock": {"krip_qty": krip_qty, "krip_val": krip_val, "krfs_qty": krfs_qty, "krfs_val": krfs_val},
            "today_iso": today.isoformat(),
            "start": s,
            "end": e,
            "power_target": POWER_TARGET_KWH_PER_TON,
            "kpis": {
                "yield_actual": today_yield_pct,
                "yield_target": yield_target_pct,
                "eff_actual": today_eff_pct,
                "eff_target": 100.0,
            },
        },
    )

# Create Heat: 2 RM lines mandatory; slag required (0 allowed); power required >0; per-heat downtime required (0 allowed)
# If downtime == 0 -> type & note cleared; If downtime > 0 -> type & note required.
@app.post("/melting/new")
def melting_new(
    request: Request,
    notes: Optional[str] = Form(None),

    slag_qty: float = Form(...),          # required; can be 0
    power_kwh: float = Form(...),         # required; must be > 0

    downtime_min: int = Form(...),        # required; can be 0
    downtime_type: str = Form("production"),
    downtime_note: str = Form(""),

    # Accept RM qtys as strings so empty optional fields don't 400
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
            return PlainTextResponse(
                f"Insufficient stock for {t}. Available {available_stock(db, t):.1f} kg",
                status_code=400
            )

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
    heat.theoretical = total_inputs * 0.97  # 3% theoretical loss

    # Melting cost per kg depends on FeSi usage (KRFS vs KRIP)
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
        avail = heat_available(db, h)
        out.write(
            f"{h.heat_no},{d.isoformat()},{heat_grade(h)},{h.qa_status or ''},"
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
# Atomization (unchanged)
# -------------------------------------------------
@app.get("/atomization", response_class=HTMLResponse)
def atom_page(request: Request, db: Session = Depends(get_db)):
    heats = db.query(Heat).filter(Heat.qa_status == "APPROVED").order_by(Heat.id.desc()).all()
    lots = db.query(Lot).order_by(Lot.id.desc()).all()

    grades = {h.id: heat_grade(h) for h in heats}
    available_map = {h.id: heat_available(db, h) for h in heats}
    db.commit()  # persist alloc_used mirror

    return templates.TemplateResponse(
        "atomization.html",
        {"request": request, "heats": heats, "lots": lots, "heat_grades": grades, "available_map": available_map}
    )

@app.post("/atomization/new")
async def atom_new(
    request: Request,
    lot_weight: float = Form(3000.0),
    db: Session = Depends(get_db)
):
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

    heats = db.query(Heat).filter(Heat.id.in_(allocs.keys())).all()
    if not heats:
        return PlainTextResponse("Selected heats not found", status_code=404)

    for h in heats:
        avail = heat_available(db, h)
        take = allocs.get(h.id, 0.0)
        if take > avail + 1e-6:
            return PlainTextResponse(f"Over-allocation from heat {h.heat_no}. Available {avail:.1f} kg.", status_code=400)

    any_fesi = any(heat_grade(h) == "KRFS" for h in heats)
    grade = "KRFS" if any_fesi else "KRIP"

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
# QA Dashboard
# -------------------------------------------------
@app.get("/qa-dashboard", response_class=HTMLResponse)
def qa_dashboard(request: Request, db: Session = Depends(get_db)):
    heats = db.query(Heat).order_by(Heat.id.desc()).all()
    lots = db.query(Lot).order_by(Lot.id.desc()).all()
    heat_grades = {h.id: heat_grade(h) for h in heats}
    return templates.TemplateResponse("qa_dashboard.html", {"request": request, "heats": heats, "lots": lots, "heat_grades": heat_grades})

# -------------------------------------------------
# Traceability
# -------------------------------------------------
@app.get("/traceability/lot/{lot_id}", response_class=HTMLResponse)
def trace_lot(lot_id: int, request: Request, db: Session = Depends(get_db)):
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

# -------------------------------------------------
# Optional: simple endpoints for day-level downtime entries
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
    return StreamingResponse(buf, media_type="application/pdf",
                             headers={"Content-Disposition": f'inline; filename="trace_{lot.lot_no}.pdf"'})
