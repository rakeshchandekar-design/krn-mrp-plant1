# ==== PART A START ====

import os
import io
import secrets
import datetime as dt
from typing import Optional, List, Dict
from pathlib import Path
from urllib.parse import quote
from reportlab.pdfgen import canvas  # used later by PDF helpers

# FastAPI
from fastapi import FastAPI, Request, Depends, Form, status
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

# SQLAlchemy
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Boolean,
    ForeignKey, func, text
)
from sqlalchemy.orm import sessionmaker, relationship, Session, declarative_base, joinedload
from sqlalchemy import inspect

# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================

def _normalize_db_url(url: str) -> str:
    # Support postgres:// and add psycopg driver if missing
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if url.startswith("postgresql://") and "+psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url

DATABASE_URL = _normalize_db_url(os.getenv("DATABASE_URL", "sqlite:///./krn.db"))

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Stage capacities and costing constants
MELT_TARGET   = 12_000  # kg/day
ATOM_TARGET   = 8_000   # kg/day
ANNEAL_TARGET = 6_000   # kg/day
SCREEN_TARGET = 10_000  # kg/day

ANNEAL_COST = 11.0  # Rs/kg
SCREEN_COST = 5.0   # Rs/kg

# ============================================================
# APP INITIALIZATION
# ============================================================

app = FastAPI()

# Sessions (used for login/roles)
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET", secrets.token_hex(16)))

# --- Paths: repo has /static and /templates at the REPO ROOT ---
BASE_DIR = Path(__file__).resolve().parent          # .../krn_mrp_app
PROJECT_ROOT = BASE_DIR.parent                      # repo root
TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR    = PROJECT_ROOT / "static"

# Static & templates
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.globals.update(max=max, min=min, round=round, int=int, float=float)

# ============================================================
# AUTH / ROLES (username = department; password = same)
# ============================================================

USER_DB = {
    "admin":     {"password": "admin",     "role": "admin"},
    "store":     {"password": "store",     "role": "store"},
    "melting":   {"password": "melting",   "role": "melting"},
    "atom":      {"password": "atom",      "role": "atom"},
    "rap":       {"password": "rap",       "role": "rap"},
    "anneal":    {"password": "anneal",    "role": "anneal"},
    "screening": {"password": "screening", "role": "screening"},
    "qa":        {"password": "qa",        "role": "qa"},
    "krn":       {"password": "krn",       "role": "view"},
}

def current_username(request: Request) -> str:
    return (getattr(request, "session", {}) or {}).get("user", "") or ""

def current_role(request: Request) -> str:
    return (getattr(request, "session", {}) or {}).get("role", "guest") or "guest"

def role_allowed(request: Request, allowed: set[str]) -> bool:
    role = current_role(request)
    if role == "view":
        # view-only users can read anywhere
        return request.method in ("GET", "HEAD", "OPTIONS")
    return role in allowed

def _role_of(request: Request) -> str:
    try:
        return (request.session or {}).get("role", "guest")
    except Exception:
        return "guest"

def _is_read_only(request: Request) -> bool:
    return _role_of(request) == "view"

# Expose role helpers to all templates
templates.env.globals.update(role_of=_role_of, is_read_only=_is_read_only)

@app.middleware("http")
async def attach_role_flags(request: Request, call_next):
    request.state.role = _role_of(request)
    request.state.read_only = _is_read_only(request)
    return await call_next(request)

@app.middleware("http")
async def block_writes_for_view(request: Request, call_next):
    if _is_read_only(request) and request.method in ("POST", "PUT", "PATCH", "DELETE"):
        return PlainTextResponse("Read-only account: action blocked", status_code=403)
    return await call_next(request)

# ============================================================
# HELPERS
# ============================================================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def grade_map_for_anneal(grade: str) -> str:
    """Map RAP KRIP/KRFS → KIP/KFS; otherwise passthrough."""
    g = (grade or "").upper()
    if g.startswith("KRIP"): return g.replace("KRIP", "KIP", 1)
    if g.startswith("KRFS"): return g.replace("KRFS", "KFS", 1)
    return grade

def require_roles(*allowed_roles):
    """FastAPI dependency to restrict access by role."""
    def _guard(request: Request):
        role = request.session.get("role")
        if role not in allowed_roles:
            from fastapi import HTTPException
            raise HTTPException(status_code=403, detail="Not authorized")
    return Depends(_guard)

def _alert_redirect(msg: str, url: str = "/"):
    """Inline JS alert + redirect (used widely across parts)."""
    safe = (msg or "").replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
    html = f'''<script>alert("{safe}");window.location.href="{url}";</script>'''
    return HTMLResponse(html)

# ============================================================
# DATABASE MODELS (keep names & tablenames stable)
# ============================================================

class GRN(Base):
    _tablename_ = "grn"
    id = Column(Integer, primary_key=True)
    grn_no = Column(String, unique=True, index=True)
    date = Column(DateTime, default=dt.datetime.utcnow)
    supplier = Column(String)
    item = Column(String)                  # used by UI & PDFs
    qty = Column(Float, default=0)
    unit_cost = Column(Float, default=0)


class Heat(Base):
    _tablename_ = "heat"
    id = Column(Integer, primary_key=True)
    heat_no = Column(String, unique=True, index=True)
    grade = Column(String)
    qty = Column(Float, default=0)
    power_kwh = Column(Float, default=0)
    total_inputs = Column(Float, default=0)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    qa_status = Column(String, default="PENDING")
    qa_remarks = Column(String, default="")

    rm_consumptions = relationship("RMConsumption", back_populates="heat", cascade="all,delete-orphan")
    lots = relationship("Lot", back_populates="heat")


class Lot(Base):
    _tablename_ = "lot"
    id = Column(Integer, primary_key=True)
    lot_no = Column(String, unique=True, index=True)
    heat_id = Column(Integer, ForeignKey("heat.id"))
    grade = Column(String)
    qty = Column(Float, default=0)
    status = Column(String, default="Pending")
    qa_status = Column(String, default="PENDING")
    qa_remarks = Column(String, default="")
    unit_cost = Column(Float, default=0)
    total_cost = Column(Float, default=0)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    date = Column(DateTime, default=dt.datetime.utcnow)

    heat = relationship("Heat", back_populates="lots")
    heats = relationship("LotHeat", back_populates="lot", cascade="all,delete-orphan")

    @property
    def weight(self): return float(self.qty or 0.0)
    @weight.setter
    def weight(self, v): self.qty = float(v or 0.0)


class LotHeat(Base):
    _tablename_ = "lot_heat"
    id = Column(Integer, primary_key=True)
    lot_id = Column(Integer, ForeignKey("lot.id"))
    heat_id = Column(Integer, ForeignKey("heat.id"))
    qty = Column(Float, default=0)

    lot = relationship("Lot", back_populates="heats")
    heat = relationship("Heat")


class RMConsumption(Base):
    _tablename_ = "rm_consumption"
    id = Column(Integer, primary_key=True)
    heat_id = Column(Integer, ForeignKey("heat.id"))
    rm_type = Column(String)          # e.g. SS Scrap / Ferro / Etc
    grn_id = Column(Integer, ForeignKey("grn.id"))
    qty = Column(Float, default=0)

    heat = relationship("Heat", back_populates="rm_consumptions")
    grn  = relationship("GRN")


class RAPLot(Base):
    _tablename_ = "raplot"
    id = Column(Integer, primary_key=True)
    lot_id = Column(Integer, ForeignKey("lot.id"))
    grade = Column(String)
    qty = Column(Float, default=0)
    available_qty = Column(Float, default=0)
    status = Column(String, default="Approved")
    qa_status = Column(String, default="PENDING")
    qa_remarks = Column(String, default="")
    unit_cost = Column(Float, default=0)
    total_cost = Column(Float, default=0)
    date = Column(DateTime, default=dt.date.today)

    lot = relationship("Lot", backref="rap_entry")


class RAPAlloc(Base):
    _tablename_ = "rap_alloc"
    id = Column(Integer, primary_key=True)
    rap_lot_id = Column(Integer, ForeignKey("raplot.id"))
    date = Column(DateTime, default=dt.date.today)
    kind = Column(String)             # DISPATCH / ANNEAL / ADJUST
    qty = Column(Float, default=0.0)
    remarks = Column(String, default="")
    dest = Column(String, default="") # customer or next-process

    rap_lot = relationship("RAPLot")


class RAPTransfer(Base):
    _tablename_ = "rap_transfer"
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, default=dt.date.today)
    lot_id = Column(Integer, ForeignKey("lot.id"))
    qty = Column(Float, default=0)
    remarks = Column(String, default="")
    lot = relationship("Lot")


class RAPDispatch(Base):
    _tablename_ = "rap_dispatch"
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, default=dt.date.today)
    customer = Column(String, default="")
    grade = Column(String, default="")
    total_qty = Column(Float, default=0.0)
    total_cost = Column(Float, default=0.0)


class RAPDispatchItem(Base):
    _tablename_ = "rap_dispatch_item"
    id = Column(Integer, primary_key=True)
    dispatch_id = Column(Integer, ForeignKey("rap_dispatch.id"))
    lot_id = Column(Integer, ForeignKey("lot.id"))
    qty = Column(Float, default=0.0)
    cost = Column(Float, default=0.0)

    dispatch = relationship("RAPDispatch", backref="items")
    lot = relationship("Lot")


class AnnealLot(Base):
    _tablename_ = "anneal_lot"
    id = Column(Integer, primary_key=True)
    lot_no = Column(String, unique=True, index=True)
    rap_lot_id = Column(Integer, ForeignKey("raplot.id"))
    grade = Column(String)
    qty = Column(Float, default=0)
    available_qty = Column(Float, default=0)
    ammonia_kg = Column(Float, default=0)
    status = Column(String, default="Pending")
    qa_status = Column(String, default="PENDING")
    qa_remarks = Column(String, default="")
    unit_cost = Column(Float, default=0)
    total_cost = Column(Float, default=0)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    date = Column(DateTime, default=dt.datetime.utcnow)

    rap_lot = relationship("RAPLot", backref="anneal_lots")
    items = relationship("AnnealLotItem", backref="anneal_lot", cascade="all,delete-orphan")

    @property
    def weight(self): return float(self.qty or 0.0)
    @weight.setter
    def weight(self, v): self.qty = float(v or 0.0)


class AnnealLotItem(Base):
    _tablename_ = "anneal_lot_item"
    id = Column(Integer, primary_key=True)
    anneal_lot_id = Column(Integer, ForeignKey("anneal_lot.id"))
    rap_lot_id = Column(Integer, ForeignKey("raplot.id"))
    qty = Column(Float, default=0.0)
    rap_lot = relationship("RAPLot")


class AnnealQA(Base):
    _tablename_ = "anneal_qa"
    id = Column(Integer, primary_key=True)
    lot_id = Column(Integer, ForeignKey("lot.id"))
    oxygen = Column(String)
    decision = Column(String, default="PENDING")
    remarks = Column(String, default="")
    lot = relationship("Lot", backref="anneal_qa_row")


class AnnealDowntime(Base):
    _tablename_ = "anneal_downtime"
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, default=dt.date.today)
    minutes = Column(Integer, default=0)
    kind = Column(String)
    remarks = Column(String, default="")


class ScreenLot(Base):
    _tablename_ = "screen_lot"
    id = Column(Integer, primary_key=True)
    lot_no = Column(String, unique=True, index=True)
    anneal_lot_id = Column(Integer, ForeignKey("anneal_lot.id"))
    grade = Column(String)
    qty = Column(Float, default=0)
    available_qty = Column(Float, default=0)
    oversize_40 = Column(Float, default=0)
    oversize_80 = Column(Float, default=0)
    status = Column(String, default="Pending")
    qa_status = Column(String, default="PENDING")
    qa_remarks = Column(String, default="")
    unit_cost = Column(Float, default=0)
    total_cost = Column(Float, default=0)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    date = Column(DateTime, default=dt.datetime.utcnow)

    anneal_lot = relationship("AnnealLot", backref="screen_lots")
    items = relationship("GSLotItem", backref="gs_lot", cascade="all,delete-orphan")

    @property
    def weight(self): return float(self.qty or 0.0)
    @weight.setter
    def weight(self, v): self.qty = float(v or 0.0)


# alias so old routes keep working
GSLot = ScreenLot


class GSLotItem(Base):
    _tablename_ = "gs_lot_item"
    id = Column(Integer, primary_key=True)
    gs_lot_id = Column(Integer, ForeignKey("screen_lot.id"))
    anneal_lot_id = Column(Integer, ForeignKey("anneal_lot.id"))
    qty = Column(Float, default=0.0)
    anneal_lot = relationship("AnnealLot")


class ScreenQA(Base):
    _tablename_ = "screen_qa"
    id = Column(Integer, primary_key=True)
    lot_id = Column(Integer, ForeignKey("lot.id"))
    c = Column(String); si = Column(String); s = Column(String); p = Column(String)
    cu = Column(String); ni = Column(String); mn = Column(String); fe = Column(String)
    o = Column(String)
    compressibility = Column(String)
    decision = Column(String, default="PENDING")
    remarks = Column(String, default="")
    lot = relationship("Lot", backref="screen_qa_row")


class ScreenDowntime(Base):
    _tablename_ = "screen_downtime"
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, default=dt.date.today)
    minutes = Column(Integer, default=0)
    kind = Column(String)
    remarks = Column(String, default="")


# alias for backward compatibility
GSDowntime = ScreenDowntime


class LotChem(Base):
    _tablename_ = "lot_chem"
    id = Column(Integer, primary_key=True)
    lot_id = Column(Integer, ForeignKey("lot.id"))
    c = Column(String); si = Column(String); s = Column(String); p = Column(String)
    cu = Column(String); ni = Column(String); mn = Column(String); fe = Column(String); o = Column(String)
    lot = relationship("Lot", backref="chemistry")


class LotPhys(Base):
    _tablename_ = "lot_phys"
    id = Column(Integer, primary_key=True)
    lot_id = Column(Integer, ForeignKey("lot.id"))
    compressibility = Column(String)
    lot = relationship("Lot", backref="phys")


class LotPSD(Base):
    _tablename_ = "lot_psd"
    id = Column(Integer, primary_key=True)
    lot_id = Column(Integer, ForeignKey("lot.id"))
    d10 = Column(String); d50 = Column(String); d90 = Column(String)
    lot = relationship("Lot", backref="psd")

# ============================================================
# LIGHTWEIGHT MIGRATIONS (idempotent, prevents 500s)
# ============================================================

def apply_simple_migrations():
    stmts = [
        # GRN
        "ALTER TABLE grn ADD COLUMN IF NOT EXISTS item VARCHAR",
        "ALTER TABLE grn ADD COLUMN IF NOT EXISTS unit_cost DOUBLE PRECISION",

        # HEAT
        "ALTER TABLE heat ADD COLUMN IF NOT EXISTS grade VARCHAR",
        "ALTER TABLE heat ADD COLUMN IF NOT EXISTS qty DOUBLE PRECISION",
        "ALTER TABLE heat ADD COLUMN IF NOT EXISTS power_kwh DOUBLE PRECISION",
        "ALTER TABLE heat ADD COLUMN IF NOT EXISTS total_inputs DOUBLE PRECISION",
        "ALTER TABLE heat ADD COLUMN IF NOT EXISTS created_at TIMESTAMP",
        "ALTER TABLE heat ADD COLUMN IF NOT EXISTS qa_status VARCHAR DEFAULT 'PENDING'",
        "ALTER TABLE heat ADD COLUMN IF NOT EXISTS qa_remarks VARCHAR",

        # LOT
        "ALTER TABLE lot ADD COLUMN IF NOT EXISTS grade VARCHAR",
        "ALTER TABLE lot ADD COLUMN IF NOT EXISTS qty DOUBLE PRECISION",
        "ALTER TABLE lot ADD COLUMN IF NOT EXISTS status VARCHAR",
        "ALTER TABLE lot ADD COLUMN IF NOT EXISTS qa_status VARCHAR DEFAULT 'PENDING'",
        "ALTER TABLE lot ADD COLUMN IF NOT EXISTS qa_remarks VARCHAR",
        "ALTER TABLE lot ADD COLUMN IF NOT EXISTS unit_cost DOUBLE PRECISION",
        "ALTER TABLE lot ADD COLUMN IF NOT EXISTS total_cost DOUBLE PRECISION",
        "ALTER TABLE lot ADD COLUMN IF NOT EXISTS created_at TIMESTAMP",
        "ALTER TABLE lot ADD COLUMN IF NOT EXISTS date TIMESTAMP",

        # RAPLot
        "ALTER TABLE raplot ADD COLUMN IF NOT EXISTS grade VARCHAR",
        "ALTER TABLE raplot ADD COLUMN IF NOT EXISTS qty DOUBLE PRECISION",
        "ALTER TABLE raplot ADD COLUMN IF NOT EXISTS available_qty DOUBLE PRECISION",
        "ALTER TABLE raplot ADD COLUMN IF NOT EXISTS status VARCHAR",
        "ALTER TABLE raplot ADD COLUMN IF NOT EXISTS qa_status VARCHAR DEFAULT 'PENDING'",
        "ALTER TABLE raplot ADD COLUMN IF NOT EXISTS qa_remarks VARCHAR",
        "ALTER TABLE raplot ADD COLUMN IF NOT EXISTS unit_cost DOUBLE PRECISION",
        "ALTER TABLE raplot ADD COLUMN IF NOT EXISTS total_cost DOUBLE PRECISION",
        "ALTER TABLE raplot ADD COLUMN IF NOT EXISTS date TIMESTAMP",

        # AnnealLot
        "ALTER TABLE anneal_lot ADD COLUMN IF NOT EXISTS lot_no VARCHAR",
        "ALTER TABLE anneal_lot ADD COLUMN IF NOT EXISTS grade VARCHAR",
        "ALTER TABLE anneal_lot ADD COLUMN IF NOT EXISTS qty DOUBLE PRECISION",
        "ALTER TABLE anneal_lot ADD COLUMN IF NOT EXISTS available_qty DOUBLE PRECISION",
        "ALTER TABLE anneal_lot ADD COLUMN IF NOT EXISTS ammonia_kg DOUBLE PRECISION",
        "ALTER TABLE anneal_lot ADD COLUMN IF NOT EXISTS status VARCHAR",
        "ALTER TABLE anneal_lot ADD COLUMN IF NOT EXISTS qa_status VARCHAR DEFAULT 'PENDING'",
        "ALTER TABLE anneal_lot ADD COLUMN IF NOT EXISTS qa_remarks VARCHAR",
        "ALTER TABLE anneal_lot ADD COLUMN IF NOT EXISTS unit_cost DOUBLE PRECISION",
        "ALTER TABLE anneal_lot ADD COLUMN IF NOT EXISTS total_cost DOUBLE PRECISION",
        "ALTER TABLE anneal_lot ADD COLUMN IF NOT EXISTS created_at TIMESTAMP",
        "ALTER TABLE anneal_lot ADD COLUMN IF NOT EXISTS date TIMESTAMP",

        "ALTER TABLE anneal_qa ADD COLUMN IF NOT EXISTS oxygen VARCHAR",
        "ALTER TABLE anneal_qa ADD COLUMN IF NOT EXISTS decision VARCHAR DEFAULT 'PENDING'",
        "ALTER TABLE anneal_qa ADD COLUMN IF NOT EXISTS remarks VARCHAR",

        "ALTER TABLE anneal_downtime ADD COLUMN IF NOT EXISTS date TIMESTAMP",
        "ALTER TABLE anneal_downtime ADD COLUMN IF NOT EXISTS minutes INTEGER",
        "ALTER TABLE anneal_downtime ADD COLUMN IF NOT EXISTS kind VARCHAR",
        "ALTER TABLE anneal_downtime ADD COLUMN IF NOT EXISTS remarks VARCHAR",

        # ScreenLot
        "ALTER TABLE screen_lot ADD COLUMN IF NOT EXISTS lot_no VARCHAR",
        "ALTER TABLE screen_lot ADD COLUMN IF NOT EXISTS grade VARCHAR",
        "ALTER TABLE screen_lot ADD COLUMN IF NOT EXISTS qty DOUBLE PRECISION",
        "ALTER TABLE screen_lot ADD COLUMN IF NOT EXISTS available_qty DOUBLE PRECISION",
        "ALTER TABLE screen_lot ADD COLUMN IF NOT EXISTS oversize_40 DOUBLE PRECISION",
        "ALTER TABLE screen_lot ADD COLUMN IF NOT EXISTS oversize_80 DOUBLE PRECISION",
        "ALTER TABLE screen_lot ADD COLUMN IF NOT EXISTS status VARCHAR",
        "ALTER TABLE screen_lot ADD COLUMN IF NOT EXISTS qa_status VARCHAR DEFAULT 'PENDING'",
        "ALTER TABLE screen_lot ADD COLUMN IF NOT EXISTS qa_remarks VARCHAR",
        "ALTER TABLE screen_lot ADD COLUMN IF NOT EXISTS unit_cost DOUBLE PRECISION",
        "ALTER TABLE screen_lot ADD COLUMN IF NOT EXISTS total_cost DOUBLE PRECISION",
        "ALTER TABLE screen_lot ADD COLUMN IF NOT EXISTS created_at TIMESTAMP",
        "ALTER TABLE screen_lot ADD COLUMN IF NOT EXISTS date TIMESTAMP",

        "ALTER TABLE screen_qa ADD COLUMN IF NOT EXISTS c VARCHAR",
        "ALTER TABLE screen_qa ADD COLUMN IF NOT EXISTS si VARCHAR",
        "ALTER TABLE screen_qa ADD COLUMN IF NOT EXISTS s VARCHAR",
        "ALTER TABLE screen_qa ADD COLUMN IF NOT EXISTS p VARCHAR",
        "ALTER TABLE screen_qa ADD COLUMN IF NOT EXISTS cu VARCHAR",
        "ALTER TABLE screen_qa ADD COLUMN IF NOT EXISTS ni VARCHAR",
        "ALTER TABLE screen_qa ADD COLUMN IF NOT EXISTS mn VARCHAR",
        "ALTER TABLE screen_qa ADD COLUMN IF NOT EXISTS fe VARCHAR",
        "ALTER TABLE screen_qa ADD COLUMN IF NOT EXISTS o VARCHAR",
        "ALTER TABLE screen_qa ADD COLUMN IF NOT EXISTS compressibility VARCHAR",
        "ALTER TABLE screen_qa ADD COLUMN IF NOT EXISTS decision VARCHAR DEFAULT 'PENDING'",
        "ALTER TABLE screen_qa ADD COLUMN IF NOT EXISTS remarks VARCHAR",

        "ALTER TABLE screen_downtime ADD COLUMN IF NOT EXISTS date TIMESTAMP",
        "ALTER TABLE screen_downtime ADD COLUMN IF NOT EXISTS minutes INTEGER",
        "ALTER TABLE screen_downtime ADD COLUMN IF NOT EXISTS kind VARCHAR",
        "ALTER TABLE screen_downtime ADD COLUMN IF NOT EXISTS remarks VARCHAR",
    ]

    with engine.begin() as conn:
        for stmt in stmts:
            try:
                conn.exec_driver_sql(stmt)
            except Exception as e:
                print("Migration skipped:", stmt, str(e))

    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            adj = []
            for s in stmts:
                s = s.replace("DOUBLE PRECISION", "FLOAT")
                adj.append(s)
            stmts = adj
        for sql in stmts:
            conn.execute(text(sql))

# ============================================================
# DB DIAG & STARTUP
# ============================================================

@app.get("/db/upgrade")
def db_upgrade():
    apply_simple_migrations()
    return {"ok": True, "msg": "Schema upgraded (idempotent)."}

@app.get("/db/columns")
def db_columns():
    insp = inspect(engine)
    out = {}
    for tbl in [
        "grn","heat","lot","raplot","rap_alloc","rap_transfer","rap_dispatch","rap_dispatch_item",
        "anneal_lot","anneal_lot_item","anneal_qa","anneal_downtime",
        "screen_lot","gs_lot_item","screen_qa","screen_downtime",
        "lot_heat","rm_consumption","lot_chem","lot_phys","lot_psd"
    ]:
        try:
            out[tbl] = [c["name"] for c in insp.get_columns(tbl)]
        except Exception as e:
            out[tbl] = f"error: {e}"
    return out

@app.on_event("startup")
def _startup_create():
    Base.metadata.create_all(bind=engine)
    apply_simple_migrations()

@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "db": str(DATABASE_URL).split("://", 1)[0],
        "static_exists": STATIC_DIR.exists(),
        "templates_exists": TEMPLATES_DIR.exists(),
    }

@app.get("/sanity")
def sanity(db: Session = Depends(get_db)):
    # light read checks to ensure columns are present
    db.execute(text("SELECT id, heat_no, grade FROM heat LIMIT 1"))
    db.execute(text("SELECT id, grn_no, item, unit_cost FROM grn LIMIT 1"))
    db.execute(text("SELECT id, lot_no, status, unit_cost FROM lot LIMIT 1"))
    db.execute(text("SELECT id, grade, status FROM raplot LIMIT 1"))
    db.execute(text("SELECT id, lot_no, status FROM anneal_lot LIMIT 1"))
    db.execute(text("SELECT id, lot_no, status FROM screen_lot LIMIT 1"))

    counts = {
        "grn": db.query(func.count(GRN.id)).scalar() or 0,
        "heat": db.query(func.count(Heat.id)).scalar() or 0,
        "lot": db.query(func.count(Lot.id)).scalar() or 0,
        "raplot": db.query(func.count(RAPLot.id)).scalar() or 0,
    }
    return {"ok": True, "counts": counts}

# ============================================================
# BASIC ROUTES (home + auth)
# ============================================================

@app.get("/setup")
def setup():
    Base.metadata.create_all(bind=engine)
    apply_simple_migrations()
    return HTMLResponse('Tables ensured. Go to <a href="/">Home</a>.')

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "user": current_username(request), "role": current_role(request)}
    )

@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return templates.TemplateResponse("login.html", {
        "request": request,
        "err": request.query_params.get("err", "")
    })

@app.post("/login")
async def login_post(request: Request, username: str = Form(...), password: str = Form(...)):
    u = USER_DB.get((username or "").strip().lower())
    if not u or u.get("password") != (password or ""):
        return RedirectResponse("/login?err=Invalid+credentials", status_code=303)
    request.session["user"] = (username or "").strip().lower()
    request.session["role"] = u.get("role", "guest")
    return RedirectResponse("/", status_code=303)

@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=303)

# ==== PART A END ====

# ==== PART B1 START ====

# ============================================================
# DASHBOARD
# ============================================================

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    today = dt.date.today()
    yest = today - dt.timedelta(days=1)

    # GRN totals
    grn_today = db.query(func.sum(GRN.qty)).filter(func.date(GRN.date) == today).scalar() or 0
    grn_month = db.query(func.sum(GRN.qty)).filter(func.date(GRN.date) >= today.replace(day=1)).scalar() or 0

    # Heat
    melt_yest = db.query(func.sum(Heat.qty)).filter(func.date(Heat.created_at) == yest).scalar() or 0
    melt_month = db.query(func.sum(Heat.qty)).filter(func.date(Heat.created_at) >= today.replace(day=1)).scalar() or 0
    melt_power = db.query(func.sum(Heat.power_kwh)).filter(func.date(Heat.created_at) >= today.replace(day=1)).scalar() or 0
    melt_inputs = db.query(func.sum(Heat.total_inputs)).filter(func.date(Heat.created_at) >= today.replace(day=1)).scalar() or 0
    melt_yield = (melt_month / melt_inputs * 100) if melt_inputs else 0
    melt_kwh_per_ton = (melt_power / (melt_month / 1000)) if melt_month else 0

    # Atomization (simple aggregates)
    atom_yest = db.query(func.sum(Lot.qty)).filter(func.date(Lot.created_at) == yest).scalar() or 0
    atom_month = db.query(func.sum(Lot.qty)).filter(func.date(Lot.created_at) >= today.replace(day=1)).scalar() or 0

    # RAP stock
    rap_stock = db.query(func.sum(RAPLot.qty)).filter(RAPLot.status == "Approved").scalar() or 0

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "grn_today": grn_today, "grn_month": grn_month,
        "melt_yest": melt_yest, "melt_month": melt_month,
        "melt_kwh_per_ton": round(melt_kwh_per_ton, 2),
        "melt_yield": round(melt_yield, 1),
        "atom_yest": atom_yest, "atom_month": atom_month,
        "rap_stock": rap_stock,
        "read_only": _is_read_only(request)
    })


# ============================================================
# GRN
# ============================================================

@app.get("/grn", response_class=HTMLResponse)
def grn_list(request: Request, db: Session = Depends(get_db)):
    grn = db.query(GRN).order_by(GRN.date.desc()).all()
    return templates.TemplateResponse("grn.html", {
        "request": request,
        "grn": grn,
        "read_only": _is_read_only(request)
    })


@app.get("/grn/new", response_class=HTMLResponse)
def grn_new_form(request: Request):
    return templates.TemplateResponse("grn_new.html", {
        "request": request,
        "read_only": _is_read_only(request)
    })


@app.post("/grn/new")
def grn_new_save(
    request: Request,
    db: Session = Depends(get_db),
    grn_no: str = Form(...),
    supplier: str = Form(...),
    item: str = Form(...),
    qty: float = Form(...),
    unit_cost: float = Form(...),
    _guard= require_roles("admin", "stores")
):
    db.add(GRN(grn_no=grn_no, supplier=supplier, item=item, qty=qty, unit_cost=unit_cost))
    db.commit()
    return RedirectResponse("/grn", status_code=303)


# ============================================================
# MELTING
# ============================================================

@app.get("/melting", response_class=HTMLResponse)
def melting_list(request: Request, db: Session = Depends(get_db)):
    heats = db.query(Heat).order_by(Heat.created_at.desc()).all()
    return templates.TemplateResponse("melting.html", {
        "request": request,
        "Heat": Heat,
        "read_only": _is_read_only(request)
    })


@app.get("/melting/new", response_class=HTMLResponse)
def melting_new_form(request: Request):
    return templates.TemplateResponse("melting_new.html", {
        "request": request,
        "read_only": _is_read_only(request)
    })


@app.post("/melting/new")
def melting_new_save(
    request: Request,
    db: Session = Depends(get_db),
    heat_no: str = Form(...),
    grade: str = Form(...),
    qty: float = Form(...),
    power_kwh: float = Form(...),
    total_inputs: float = Form(...),
    _guard = require_roles("admin", "melt")
):
    db.add(Heat(
        heat_no=heat_no,
        grade=grade,
        qty=qty,
        power_kwh=power_kwh,
        total_inputs=total_inputs
    ))
    db.commit()
    return RedirectResponse("/melting", status_code=303)

# ==== PART B1 END ====

# ==== PART B2 START ====

# -------------------------------------------------
# Atomization (list, create, exports, downtime)
# -------------------------------------------------

def _redir_err(msg: str) -> RedirectResponse:
    return RedirectResponse(f"/atomization?err={quote(msg)}", status_code=303)

@app.get("/atomization", response_class=HTMLResponse)
def atom_page(
    request: Request,
    start: Optional[str] = None,
    end: Optional[str] = None,
    db: Session = Depends(get_db)
):
    if not role_allowed(request, {"admin", "atom"}):
        return RedirectResponse("/login", status_code=303)

    Heat_all = (
        db.query(Heat)
        .filter(Heat.qa_status == "APPROVED")
        .order_by(Heat.id.desc())
        .all()
    )

    available_map = {h.id: heat_available(db, h) for h in Heat_all}
    grades = {h.id: heat_grade(h) for h in Heat_all}
    heats = [h for h in Heat_all if (available_map.get(h.id) or 0.0) > 0.0001]

    lots = db.query(Lot).order_by(Lot.id.desc()).all()

    today = dt.date.today()
    Lot_with_dates = [(lot, lot_date_from_no(lot.lot_no) or today) for lot in Lot]
    prod_today = sum((lot.weight or 0.0) for lot, d in Lot_with_dates if d == today)
    eff_today = (100.0 * prod_today / DAILY_CAPACITY_ATOM_KG) if DAILY_CAPACITY_ATOM_KG > 0 else 0.0

    last5 = []
    for i in range(4, -1, -1):
        d = today - dt.timedelta(days=i)
        actual = sum((lot.weight or 0.0) for lot, dd in Lot_with_dates if dd == d)
        target = atom_day_target_kg(db, d)
        last5.append({"date": d.isoformat(), "actual": actual, "target": target})

    # RAP allocations impact live stock display
    rap_alloc_by_lot: Dict[int, float] = {}
    rap_pairs = (
        db.query(RAPLot.lot_id, func.coalesce(func.sum(RAPAlloc.qty), 0.0))
          .join(RAPAlloc, RAPAlloc.rap_lot_id == RAPLot.id, isouter=True)
          .group_by(RAPLot.lot_id)
          .all()
    )
    for lid, s in rap_pairs:
        rap_alloc_by_lot[int(lid)] = float(s or 0.0)

    stock = {"KRIP_qty": 0.0, "KRIP_val": 0.0, "KRFS_qty": 0.0, "KRFS_val": 0.0}
    for lot in Lot:
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

    Lot_stock = {
        "krip_qty": stock.get("KRIP_qty", 0.0),
        "krip_val": stock.get("KRIP_val", 0.0),
        "krfs_qty": stock.get("KRFS_qty", 0.0),
        "krfs_val": stock.get("KRFS_val", 0.0),
    }

    s = start or today.isoformat()
    e = end or today.isoformat()
    try:
        _ = dt.date.fromisoformat(s); _ = dt.date.fromisoformat(e)
    except Exception:
        s, e = today.isoformat(), today.isoformat()

    err = request.query_params.get("err")

    from types import SimpleNamespace
    month_start = today.replace(day=1)
    month_end = (month_start + dt.timedelta(days=32)).replace(day=1)
    try:
        atom_bal = _get_atomization_balance(db, month_start, month_end)  # if present
        tot_feed = (atom_bal.feed_kg or 0.0)
        tot_prod = (atom_bal.produced_kg or 0.0)
        atom_bal.conv_pct = (100.0 * tot_prod / tot_feed) if tot_feed > 0 else 0.0
    except Exception:
        atom_bal = SimpleNamespace(feed_kg=0.0, produced_kg=0.0, oversize_kg=0.0, conv_pct=0.0)

    return templates.TemplateResponse(
        "atomization.html",
        {
            "request": request,
            "role": current_role(request),
            "Heat": Heat,
            "Lot": Lot,
            "heat_grades": grades,
            "available_map": available_map,
            "today_iso": today.isoformat(),
            "start": s,
            "end": e,
            "atom_eff_today": eff_today,
            "atom_last5": last5,
            "atom_capacity": DAILY_CAPACITY_ATOM_KG,
            "atom_stock": stock,
            "Lot_stock": Lot_stock,
            "atom_bal": atom_bal,
            "error_msg": err,
        }
    )


@app.post("/atomization/new")
async def atom_new(
    request: Request,
    lot_weight: float = Form(3000.0),
    db: Session = Depends(get_db)
):
    try:
        form = await request.form()
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

        heats = db.query(Heat).filter(Heat.id.in_(allocs.keys())).all()
        if not Heat:
            return _alert_redirect("Selected Heat not found.")

        grades = {("KRFS" if heat_grade(h) == "KRFS" else "KRIP") for h in Heat}
        if len(grades) > 1:
            return _alert_redirect("Mixing KRIP and KRFS in the same lot is not allowed.")

        for h in Heat:
            avail = heat_available(db, h)
            take = allocs.get(h.id, 0.0)
            if take > avail + 1e-6:
                return _alert_redirect(f"Over-allocation from heat {h.heat_no}. Available {avail:.1f} kg.")

        total_alloc = sum(allocs.values())
        tol = 0.05
        if abs(total_alloc - float(lot_weight or 0.0)) > tol:
            return _alert_redirect(
                f"Allocated total ({total_alloc:.1f} kg) must equal Lot Weight ({float(lot_weight or 0):.1f} kg)."
            )

        any_fesi = any(heat_grade(h) == "KRFS" for h in Heat)
        grade = "KRFS" if any_fesi else "KRIP"

        today = dt.date.today().strftime("%Y%m%d")
        seq = (db.query(func.count(Lot.id)).filter(Lot.lot_no.like(f"KR%{today}%")).scalar() or 0) + 1
        lot_no = f"{grade}-{today}-{seq:03d}"

        lot = Lot(lot_no=lot_no, weight=float(lot_weight or 0.0), grade=grade)
        db.add(lot)
        db.flush()

        for h in Heat:
            q = allocs.get(h.id, 0.0)
            if q > 0:
                db.add(LotHeat(lot_id=lot.id, heat_id=h.id, qty=q))
                h.alloc_used = float(h.alloc_used or 0.0) + q

        weighted_cost = sum((h.unit_cost or 0.0) * allocs.get(h.id, 0.0) for h in Heat)
        avg_heat_unit_cost = (weighted_cost / total_alloc) if total_alloc > 1e-9 else 0.0
        lot.unit_cost = avg_heat_unit_cost + ATOMIZATION_COST_PER_KG + SURCHARGE_PER_KG
        lot.total_cost = lot.unit_cost * (lot.weight or 0.0)

        sums = {k: 0.0 for k in ["c", "si", "s", "p", "cu", "ni", "mn", "fe"]}
        for h in Heat:
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

    except Exception:
        db.rollback()
        return _alert_redirect("Unexpected error while creating lot.")


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
    for lot in Lot:
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
# QA Dashboard (unchanged structure, kept for completeness)
# -------------------------------------------------
@app.get("/qa-dashboard", response_class=HTMLResponse)
def qa_dashboard(
    request: Request,
    start: Optional[str] = None,
    end: Optional[str] = None,
    db: Session = Depends(get_db),
):
    if not role_allowed(request, {"admin", "qa"}):
        return RedirectResponse("/login", status_code=303)

    today = dt.date.today()
    s_iso = start or today.isoformat()
    e_iso = end or today.isoformat()
    try:
        s_date = dt.date.fromisoformat(s_iso)
        e_date = dt.date.fromisoformat(e_iso)
    except Exception:
        s_date = e_date = today
        s_iso = e_iso = today.isoformat()

    Heat_all = db.query(Heat).order_by(Heat.id.desc()).all()
    Lot_all  = db.query(Lot).order_by(Lot.id.desc()).all()

    def _hd(h: Heat) -> dt.date:
        return heat_date_from_no(h.heat_no) or today
    def _ld(l: Lot) -> dt.date:
        return lot_date_from_no(l.lot_no) or today

    Heat_vis = [h for h in Heat_all if s_date <= _hd(h) <= e_date]
    Lot_vis  = [l for l in Lot_all  if s_date <= _ld(l) <= e_date]

    month_start = e_date.replace(day=1)
    month_end   = (month_start + dt.timedelta(days=32)).replace(day=1) - dt.timedelta(days=1)
    Lot_this_month = [l for l in Lot_all if month_start <= _ld(l) <= month_end]

    def _sum_Lot(status: str) -> float:
        s = status.upper()
        return sum(float(l.weight or 0.0) for l in Lot_this_month if (l.qa_status or "").upper() == s)

    kpi = {
        "approved_kg": _sum_Lot("APPROVED"),
        "hold_kg": _sum_Lot("HOLD"),
        "rejected_kg": _sum_Lot("REJECTED"),
    }

    pending_count = (
        sum(1 for h in Heat_all if (h.qa_status or "").upper() == "PENDING") +
        sum(1 for l in Lot_all  if (l.qa_status or "").upper() == "PENDING")
    )
    todays_count = (
        sum(1 for h in Heat_all if _hd(h) == today and (h.qa_status or "").upper() != "PENDING") +
        sum(1 for l in Lot_all  if _ld(l) == today and (l.qa_status or "").upper() != "PENDING")
    )

    heat_grades = {h.id: heat_grade(h) for h in Heat_vis}

    return templates.TemplateResponse(
        "qa_dashboard.html",
        {
            "request": request,
            "role": current_role(request),
            "Heat": Heat_vis,
            "Lot": Lot_vis,
            "heat_grades": heat_grades,
            "kpi_approved_month": float(kpi.get("approved_kg", 0.0)),
            "kpi_hold_month":     float(kpi.get("hold_kg", 0.0)),
            "kpi_rejected_month": float(kpi.get("rejected_kg", 0.0)),
            "kpi_pending_count":  int(pending_count),
            "kpi_today_count":    int(todays_count),
            "start": s_iso,
            "end": e_iso,
            "today_iso": today.isoformat(),
        },
    )


# -------------------------------------------------
# RAP (page, allocate, exports, PDFs, multi-lot dispatch)
# -------------------------------------------------
@app.get("/rap", response_class=HTMLResponse)
def rap_page(request: Request, db: Session = Depends(get_db)):
    if not role_allowed(request, {"admin", "rap"}):
        return RedirectResponse("/login", status_code=303)

    today = dt.date.today()
    lots = (
        db.query(Lot)
        .filter(Lot.qa_status == "APPROVED")
        .order_by(Lot.id.desc())
        .all()
    )

    rap_rows: List[RAPLot] = []
    for lot in Lot:
        rap_rows.append(ensure_rap_lot(db, lot))
    db.commit()

    kpi = {"KRIP_qty": 0.0, "KRIP_val": 0.0, "KRFS_qty": 0.0, "KRFS_val": 0.0}
    for rap in rap_rows:
        lot = rap.lot
        qty = float(rap.available_qty or 0.0)
        if qty <= 0:
            continue
        val = qty * float(lot.unit_cost or 0.0)
        if (lot.grade or "KRIP") == "KRFS":
            kpi["KRFS_qty"] += qty; kpi["KRFS_val"] += val
        else:
            kpi["KRIP_qty"] += qty; kpi["KRIP_val"] += val

    month_start = today.replace(day=1)
    trend = (
        db.query(RAPAlloc.kind, Lot.grade, func.sum(RAPAlloc.qty))
        .join(RAPLot, RAPAlloc.rap_lot_id == RAPLot.id)
        .join(Lot, RAPLot.lot_id == Lot.id)
        .filter(RAPAlloc.date >= month_start)
        .group_by(RAPAlloc.kind, Lot.grade)
        .all()
    )
    kpi_trends = {"DISPATCH": {"KRIP": 0, "KRFS": 0}, "PLANT2": {"KRIP": 0, "KRFS": 0}}
    for kind, grade, qty in trend:
        kpi_trends[kind][(grade or "KRIP")] = float(qty or 0)

    return templates.TemplateResponse(
        "rap.html",
        {
            "request": request,
            "role": current_role(request),
            "rows": rap_rows,
            "kpi": kpi,
            "kpi_trends": kpi_trends,
            "today": today.isoformat(),
            "min_date": (today - dt.timedelta(days=3)).isoformat(),
        }
    )


@app.post("/rap/allocate")
def rap_allocate(
    rap_lot_id: int = Form(...),
    date: str = Form(...),
    kind: str = Form(...),
    qty: float = Form(...),
    dest: str = Form(""),
    remarks: str = Form(""),
    db: Session = Depends(get_db),
):
    rap = db.get(RAPLot, rap_lot_id)
    if not rap:
        return _alert_redirect("RAP lot not found.", url="/rap")
    try:
        d = dt.date.fromisoformat(date)
    except Exception:
        return _alert_redirect("Invalid date.", url="/rap")
    today = dt.date.today()
    if d > today or d < (today - dt.timedelta(days=3)):
        return _alert_redirect("Date must be today or within the last 3 days.", url="/rap")

    try:
        qty = float(qty or 0.0)
    except Exception:
        qty = 0.0
    if qty <= 0:
        return _alert_redirect("Quantity must be > 0.", url="/rap")

    lot = db.get(Lot, rap.lot_id)
    if not lot or (lot.qa_status or "") != "APPROVED":
        return _alert_redirect("Underlying lot is not APPROVED.", url="/rap")

    total_alloc = rap_total_alloc_qty_for_lot(db, lot.id)
    avail = max(float(lot.weight or 0.0) - float(total_alloc or 0.0), 0.0)
    if qty > avail + 1e-6:
        return _alert_redirect(f"Over-allocation. Available {avail:.1f} kg.", url="/rap")

    kind = (kind or "").upper()
    if kind not in ("DISPATCH", "PLANT2"):
        kind = "DISPATCH"
    if kind == "DISPATCH":
        if not (dest and dest.strip()):
            return _alert_redirect("Customer name is required for Dispatch.", url="/rap")
    else:
        dest = "Plant 2"

    rec = RAPAlloc(
        rap_lot_id=rap.id,
        date=d,
        kind=kind,
        qty=qty,
        remarks=remarks,
        dest=dest,
    )
    db.add(rec); db.flush()

    rap.available_qty = max(avail - qty, 0.0)
    rap.status = "CLOSED" if rap.available_qty <= 1e-6 else "OPEN"
    db.add(rap); db.commit()

    if rec.kind == "DISPATCH":
        return RedirectResponse(f"/rap/dispatch/{rec.id}/pdf", status_code=303)
    return RedirectResponse("/rap", status_code=303)


# --- Exports (Dispatch / Plant2) ---
@app.get("/rap/dispatch/export")
def export_rap_dispatch(db: Session = Depends(get_db)):
    import csv, io
    from fastapi.responses import StreamingResponse
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Date","Lot","Grade","Qty (kg)","Unit Cost (₹/kg)","Value (₹)","Customer","Remarks","Alloc ID"])
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
            lot.lot_no or "", lot.grade or "",
            f"{qty:.1f}", f"{unit:.2f}", f"{val:.2f}",
            alloc.dest or "", (alloc.remarks or "").replace(",", " "), alloc.id
        ])
    buf.seek(0)
    return StreamingResponse(io.StringIO(buf.getvalue()), media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="dispatch_movements.csv"'})

@app.get("/rap/transfer/export")
def export_rap_transfers(db: Session = Depends(get_db)):
    import csv, io
    from fastapi.responses import StreamingResponse
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Date","Lot","Grade","Qty (kg)","Unit Cost (₹/kg)","Value (₹)","Remarks","Alloc ID"])
    rows = (
        db.query(RAPAlloc, RAPLot, Lot)
          .join(RAPLot, RAPAlloc.rap_lot_id == RAPLot.id)
          .join(Lot, RAPLot.lot_id == Lot.id)
          .filter(RAPAlloc.kind == "PLANT2")
          .order_by(RAPAlloc.date.asc(), RAPAlloc.id.asc())
          .all()
    )
    for alloc, rl, lot in rows:
        qty  = float(alloc.qty or 0.0)
        unit = float(lot.unit_cost or 0.0)
        val  = qty * unit
        writer.writerow([
            (alloc.date or dt.date.today()).isoformat(),
            lot.lot_no or "", lot.grade or "",
            f"{qty:.1f}", f"{unit:.2f}", f"{val:.2f}",
            (alloc.remarks or "").replace(",", " "), alloc.id
        ])
    buf.seek(0)
    return StreamingResponse(io.StringIO(buf.getvalue()), media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="plant2_transfers.csv"'})

# --- Single-alloc dispatch PDF (with annexure) ---
@app.get("/rap/dispatch/{alloc_id}/pdf")
def rap_dispatch_pdf(alloc_id: int, db: Session = Depends(get_db)):
    alloc = db.get(RAPAlloc, alloc_id)
    if not alloc:
        return PlainTextResponse("Dispatch allocation not found.", status_code=404)
    rap = db.get(RAPLot, alloc.rap_lot_id)
    if not rap:
        return PlainTextResponse("RAP lot not found.", status_code=404)
    lot = db.get(Lot, rap.lot_id)
    if not lot:
        return PlainTextResponse("Lot not found.", status_code=404)

    Heat, fifo_rows = [], []
    for lh in lot.Heat:
        h = db.get(Heat, lh.heat_id)
        if not h:
            continue
        Heat.append((h, float(lh.qty or 0.0)))
        for cons in h.rm_consumptions:
            g = cons.grn
            fifo_rows.append({
                "heat_no": h.heat_no, "rm_type": cons.rm_type, "grn_id": cons.grn_id,
                "supplier": g.supplier if g else "", "qty": float(cons.qty or 0.0)
            })

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    draw_header(c, "Dispatch Note")

    y = height - 4 * cm
    c.setFont("Helvetica", 11)
    c.drawString(2*cm, y, f"Date/Time: {(alloc.date or dt.date.today()).isoformat()}  {dt.datetime.now().strftime('%H:%M')}"); y -= 14
    c.drawString(2*cm, y, f"Customer: {alloc.dest or ''}"); y -= 14
    c.drawString(2*cm, y, f"Note Type: DISPATCH"); y -= 14

    y -= 6
    c.setFont("Helvetica-Bold", 11); c.drawString(2*cm, y, "Lot Details"); y -= 14
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, y, f"Lot: {lot.lot_no}    Grade: {lot.grade or '-'}"); y -= 12
    c.drawString(2*cm, y, f"Allocated Qty (kg): {float(alloc.qty or 0):.1f}"); y -= 12
    c.drawString(2*cm, y, f"Cost / kg (₹): {float(lot.unit_cost or 0):.2f}    Cost Value (₹): {float(lot.unit_cost or 0) * float(alloc.qty or 0):.2f}"); y -= 16

    c.setFont("Helvetica-Oblique", 10)
    c.drawString(2*cm, y, "SELL RATE (₹/kg): _________________________     Amount (₹): _________________________"); y -= 18

    c.setFont("Helvetica-Bold", 11); c.drawString(2*cm, y, "Annexure: QA Certificates & GRN Trace"); y -= 14
    c.setFont("Helvetica", 10)

    c.drawString(2*cm, y, "Heat used in this lot (lot allocation vs. heat out / QA):"); y -= 12
    for h, qalloc in Heat:
        c.drawString(2.2*cm, y, f"{h.heat_no}  | Alloc to lot: {qalloc:.1f} kg  | Heat Out: {float(h.actual_output or 0):.1f} kg  | QA: {h.qa_status or ''}")
        y -= 12
        if y < 3*cm:
            c.showPage(); draw_header(c, "Dispatch Note"); y = height - 4*cm

    y -= 6
    c.setFont("Helvetica-Bold", 11); c.drawString(2*cm, y, "GRN Consumption (FIFO)"); y -= 14
    c.setFont("Helvetica", 10)
    for r in fifo_rows:
        c.drawString(2.2*cm, y, f"Heat {r['heat_no']} | {r['rm_type']} | GRN #{r['grn_id']} | {r['supplier']} | {r['qty']:.1f} kg")
        y -= 12
        if y < 3*cm:
            c.showPage(); draw_header(c, "Dispatch Note"); y = height - 4*cm

    y -= 8
    c.setFont("Helvetica-Bold", 10)
    c.drawString(2 * cm, y, "This document is a Dispatch Note for Invoice purpose only."); y -= 16

    c.setFont("Helvetica", 10)
    c.drawString(2 * cm, y, "Authorized Sign: ____________________________"); y -= 24

    # Optional: annexure per lot
    try:
        draw_lot_qa_annexure(c, lot)
    except Exception:
        pass

    c.showPage(); c.save()
    buf.seek(0)
    filename = f"Dispatch_{lot.lot_no}_{alloc.id}.pdf"
    return StreamingResponse(buf, media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'})


# -------------------------------------------------
# Annealing (page, create, downtime, export, QA)
# -------------------------------------------------
@app.get("/annealing", response_class=HTMLResponse)
def anneal_page(request: Request, start: str|None=None, end: str|None=None,
                db: Session = Depends(get_db),
                _guard = require_roles("admin","anneal")):

    today = dt.date.today()
    start_d = dt.date.fromisoformat(start) if start else today
    end_d = dt.date.fromisoformat(end) if end else today

    produced_today = (
        db.query(func.coalesce(func.sum(AnnealLot.weight), 0.0))
          .filter(AnnealLot.date == today)
          .scalar() or 0.0
    )
    cap = ANNEAL_CAPACITY_KG * (day_available_minutes_anneal(db, today) / 1440.0)
    eff_today = (produced_today / cap * 100.0) if cap > 0 else 0.0

    last5 = []
    for i in range(5):
        d = today - dt.timedelta(days=i)
        p = db.query(func.coalesce(func.sum(AnnealLot.weight), 0.0)).filter(AnnealLot.date == d).scalar() or 0.0
        t = day_target_kg_anneal(db, d)
        last5.append({"date": d.isoformat(), "actual": p, "target": t})
    last5 = list(reversed(last5))

    q = (
        db.query(
            Lot.grade,
            func.coalesce(func.sum(RAPLot.available_qty), 0.0).label("qty"),
            func.coalesce(func.sum(RAPLot.available_qty * Lot.unit_cost), 0.0).label("value"),
        )
        .join(Lot, Lot.id == RAPLot.lot_id)
        .filter((Lot.qa_status == "APPROVED") | (Lot.qa_status == None))
        .group_by(Lot.grade)
    )
    rap_live = {"KIP_qty": 0.0, "KIP_val": 0.0, "KFS_qty": 0.0, "KFS_val": 0.0}
    for g, qty, val in q.all():
        g2 = grade_after_rap(g)
        if g2 == "KFS":
            rap_live["KFS_qty"] += float(qty or 0)
            rap_live["KFS_val"] += float(val or 0)
        else:
            rap_live["KIP_qty"] += float(qty or 0)
            rap_live["KIP_val"] += float(val or 0)

    lots = (
        db.query(AnnealLot)
        .filter(AnnealLot.date >= start_d, AnnealLot.date <= end_d)
        .order_by(AnnealLot.date.desc(), AnnealLot.id.desc())
        .all()
    )

    ctx = {
        "request": request,
        "role": current_role(request),
        "today_iso": today.isoformat(),
        "start": start_d.isoformat(),
        "end": end_d.isoformat(),
        "ann_capacity": ANNEAL_CAPACITY_KG,
        "ann_eff_today": eff_today,
        "ann_last5": last5,
        "ann_live": rap_live,
        "ann_Lot": Lot,
        "atom_capacity": ANNEAL_CAPACITY_KG,
        "atom_eff_today": eff_today,
        "atom_last5": last5,
        "Lot": Lot,
        "Lot_stock": {"krip_qty": rap_live["KIP_qty"], "krip_val": rap_live["KIP_val"],
                       "krfs_qty": rap_live["KFS_qty"], "krfs_val": rap_live["KFS_val"]},
    }
    return templates.TemplateResponse("annealing.html", ctx)


@app.post("/annealing/new")
def anneal_new(
    request: Request,
    db: Session = Depends(get_db),
    lot_weight: float = Form(...),
    _guard = require_roles("admin", "anneal"),
    **allocs,
):
    pairs: List[Tuple[int, float]] = []
    for k, v in allocs.items():
        if not k.startswith("alloc_"):
            continue
        try:
            rid = int(k.split("_", 1)[1])
            q = float(v or 0)
            if q > 0:
                pairs.append((rid, q))
        except Exception:
            pass

    if not pairs:
        return _alert_redirect("Select at least one RAP lot with a positive allocation.", "/annealing")

    total_alloc = sum(q for _, q in pairs)
    if abs(total_alloc - float(lot_weight or 0)) > 0.05:
        return _alert_redirect("Allocated qty must equal the new lot weight.", "/annealing")

    rap_rows = db.query(RAPLot, Lot).join(Lot, Lot.id == RAPLot.lot_id).filter(RAPLot.id.in_([rid for rid,_ in pairs])).all()
    avail_map = {row.RAPLot.id: float(row.RAPLot.available_qty or 0) for row in rap_rows}
    for rid, q in pairs:
        if q > avail_map.get(rid, 0.0) + 1e-6:
            return _alert_redirect("Allocation exceeds available qty for a RAP lot.", "/annealing")

    by_grade: Dict[str, float] = {}
    cost_sum = 0.0
    for rid, q in pairs:
        rap, lot = next((x for x in rap_rows if x.RAPLot.id == rid), (None, None))
        if not rap or not lot:
            continue
        g = grade_after_rap(lot.grade or "KRIP")
        by_grade[g] = by_grade.get(g, 0.0) + q
        cost_sum += q * float(lot.unit_cost or 0.0)

    dom_grade = max(by_grade.items(), key=lambda x: x[1])[0] if by_grade else "KIP"

    input_unit_cost = (cost_sum / total_alloc) if total_alloc > 0 else 0.0
    unit_cost = input_unit_cost + ANNEAL_COST_PER_KG
    total_cost = unit_cost * float(lot_weight or 0)

    today = dt.date.today()
    seq = (db.query(func.count(AnnealLot.id)).filter(AnnealLot.date == today).scalar() or 0) + 1
    lot_no = next_anneal_lot_no(today, seq)
    newlot = AnnealLot(
        lot_no=lot_no,
        date=today,
        weight=float(lot_weight or 0),
        grade=dom_grade,
        unit_cost=unit_cost,
        total_cost=total_cost,
        available_qty=float(lot_weight or 0),
        qa_status="PENDING",
    )
    db.add(newlot); db.flush()

    for rid, q in pairs:
        db.add(AnnealLotItem(anneal_lot_id=newlot.id, rap_lot_id=rid, qty=q))
        rap = db.query(RAPLot).get(rid)
        rap.available_qty = max((rap.available_qty or 0.0) - q, 0.0)
        rap.status = "CLOSED" if rap.available_qty <= 1e-6 else "OPEN"
        db.add(rap)

    db.commit()
    return RedirectResponse("/annealing", status_code=303)


@app.get("/annealing/downtime", response_class=HTMLResponse)
def anneal_downtime_page(request: Request, db: Session = Depends(get_db),
                         _guard = require_roles("admin","anneal")):
    rows = db.query(AnnealDowntime).order_by(AnnealDowntime.date.desc(), AnnealDowntime.id.desc()).limit(50).all()
    return templates.TemplateResponse("anneal_downtime.html", {
        "request": request, "rows": rows, "today": dt.date.today().isoformat()
    })

@app.post("/annealing/downtime")
def anneal_downtime_save(
    request: Request,
    db: Session = Depends(get_db),
    date: str = Form(...),
    minutes: int = Form(...),
    kind: Optional[str] = Form(None),
    remarks: Optional[str] = Form(None),
    _guard = require_roles("admin", "anneal"),
):
    d = dt.date.fromisoformat(date)
    db.add(AnnealDowntime(date=d, minutes=int(minutes or 0), kind=kind, remarks=remarks))
    db.commit()
    return RedirectResponse("/annealing/downtime", status_code=303)

@app.get("/annealing/downtime/export")
def anneal_downtime_csv(db: Session = Depends(get_db)):
    import csv
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["date","minutes","kind","remarks"])
    for r in db.query(AnnealDowntime).order_by(AnnealDowntime.date).all():
        w.writerow([r.date.isoformat(), r.minutes, r.kind or "", r.remarks or ""])
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="anneal_downtime.csv"'}
    )

# --- Annealing QA (oxygen only; carry forward rest) ---
@app.get("/qa/anneal/lot/{lot_id}", response_class=HTMLResponse)
def qa_anneal_lot_form(lot_id: int,
                       request: Request,
                       db: Session = Depends(get_db),
                       _guard = require_roles("admin","anneal","qa","view")):
    lot = db.get(Lot, lot_id)
    if not lot:
        return _alert_redirect("Lot not found", "/annealing")
    grade = (lot.grade or "KIP")
    read_only = _is_read_only(request) or (current_role(request) not in ("admin","anneal","qa"))
    oxygen_val = _prefill_anneal_o(db, lot)
    return templates.TemplateResponse("qa_anneal_lot.html", {
        "request": request,
        "lot": lot, "grade": grade, "oxygen": oxygen_val,
        "read_only": read_only, "role": current_role(request),
    })

@app.post("/qa/anneal/lot/{lot_id}")
def qa_anneal_lot_save(lot_id: int,
                       request: Request,
                       oxygen: str = Form(""),
                       decision: str = Form(...),
                       remarks: str = Form(""),
                       db: Session = Depends(get_db),
                       _guard = require_roles("admin","anneal","qa")):
    lot = db.get(Lot, lot_id)
    if not lot:
        return _alert_redirect("Lot not found", "/annealing")
    row = db.query(AnnealQA).filter(AnnealQA.lot_id == lot.id).first()
    if not row:
        row = AnnealQA(lot_id=lot.id, oxygen=oxygen)
    else:
        row.oxygen = oxygen
    db.add(row)
    lot.qa_status = decision or "PENDING"
    lot.qa_remarks = remarks or ""
    db.add(lot); db.commit()
    return RedirectResponse(f"/qa/anneal/lot/{lot.id}", status_code=303)


# -------------------------------------------------
# Screening (page, create, downtime, export, QA)
# -------------------------------------------------
@app.get("/screening", response_class=HTMLResponse)
def gs_page(request: Request, start: str|None=None, end: str|None=None,
            db: Session = Depends(get_db),
            _guard = require_roles("admin","screening")):
    today = dt.date.today()
    start_d = dt.date.fromisoformat(start) if start else today
    end_d = dt.date.fromisoformat(end) if end else today

    produced_today = (
        db.query(func.coalesce(func.sum(ScreenLot.weight), 0.0))
          .filter(ScreenLot.date == today)
          .scalar() or 0.0
    )
    cap = SCREEN_CAPACITY_KG * (day_available_minutes_gs(db, today) / 1440.0)
    eff_today = (produced_today / cap * 100.0) if cap > 0 else 0.0

    last5 = []
    for i in range(5):
        d = today - dt.timedelta(days=i)
        p = db.query(func.coalesce(func.sum(ScreenLot.weight), 0.0)).filter(ScreenLot.date == d).scalar() or 0.0
        t = day_target_kg_gs(db, d)
        last5.append({"date": d.isoformat(), "actual": p, "target": t})
    last5 = list(reversed(last5))

    q = (
        db.query(
            AnnealLot.grade,
            func.coalesce(func.sum(AnnealLot.available_qty), 0.0).label("qty"),
            func.coalesce(func.sum(AnnealLot.available_qty * AnnealLot.unit_cost), 0.0).label("value"),
        )
        .group_by(AnnealLot.grade)
    )
    live = {"KIP_qty": 0.0, "KIP_val": 0.0, "KFS_qty": 0.0, "KFS_val": 0.0}
    for g, qty, val in q.all():
        g2 = (g or "KIP").upper()
        if g2 == "KFS":
            live["KFS_qty"] += float(qty or 0); live["KFS_val"] += float(val or 0)
        else:
            live["KIP_qty"] += float(qty or 0); live["KIP_val"] += float(val or 0)

    lots = (
        db.query(ScreenLot)
        .filter(ScreenLot.date >= start_d, ScreenLot.date <= end_d)
        .order_by(ScreenLot.date.desc(), ScreenLot.id.desc())
        .all()
    )

    ctx = {
        "request": request,
        "role": current_role(request),
        "today_iso": today.isoformat(),
        "start": start_d.isoformat(),
        "end": end_d.isoformat(),
        "gs_capacity": SCREEN_CAPACITY_KG,
        "gs_eff_today": eff_today,
        "gs_last5": last5,
        "gs_live": live,
        "gs_Lot": Lot,
        "atom_capacity": SCREEN_CAPACITY_KG,
        "atom_eff_today": eff_today,
        "atom_last5": last5,
        "Lot": Lot,
        "Lot_stock": {"krip_qty": live["KIP_qty"], "krip_val": live["KIP_val"],
                       "krfs_qty": live["KFS_qty"], "krfs_val": live["KFS_val"]},
    }
    return templates.TemplateResponse("screening.html", ctx)


@app.post("/screening/new")
def gs_new(
    request: Request,
    db: Session = Depends(get_db),
    lot_weight: float = Form(...),
    _guard = require_roles("admin", "screening"),
    **allocs,
):
    pairs: List[Tuple[int, float]] = []
    for k, v in allocs.items():
        if not k.startswith("alloc_"):
            continue
        try:
            aid = int(k.split("_", 1)[1])
            q = float(v or 0)
            if q > 0:
                pairs.append((aid, q))
        except Exception:
            pass

    if not pairs:
        return _alert_redirect("Select at least one Anneal lot with a positive allocation.", "/screening")

    total_alloc = sum(q for _, q in pairs)
    if abs(total_alloc - float(lot_weight or 0)) > 0.05:
        return _alert_redirect("Allocated qty must equal the new lot weight.", "/screening")

    a_rows = db.query(AnnealLot).filter(AnnealLot.id.in_([aid for aid,_ in pairs])).all()
    avail_map = {a.id: float(a.available_qty or 0) for a in a_rows}
    for aid, q in pairs:
        if q > avail_map.get(aid, 0.0) + 1e-6:
            return _alert_redirect("Allocation exceeds available qty for an Anneal lot.", "/screening")

    by_grade: Dict[str, float] = {}
    cost_sum = 0.0
    for aid, q in pairs:
        a = next((x for x in a_rows if x.id == aid), None)
        if not a:
            continue
        g = (a.grade or "KIP").upper()
        by_grade[g] = by_grade.get(g, 0.0) + q
        cost_sum += q * float(a.unit_cost or 0.0)
    dom_grade = max(by_grade.items(), key=lambda x: x[1])[0] if by_grade else "KIP"

    input_unit_cost = (cost_sum / total_alloc) if total_alloc > 0 else 0.0
    unit_cost = input_unit_cost + SCREEN_COST_PER_KG
    total_cost = unit_cost * float(lot_weight or 0)

    today = dt.date.today()
    seq = (db.query(func.count(ScreenLot.id)).filter(ScreenLot.date == today).scalar() or 0) + 1
    lot_no = next_gs_lot_no(today, seq)
    newlot = ScreenLot(
        lot_no=lot_no,
        date=today,
        weight=float(lot_weight or 0),
        grade=dom_grade,
        unit_cost=unit_cost,
        total_cost=total_cost,
        available_qty=float(lot_weight or 0),
        qa_status="PENDING",
    )
    db.add(newlot); db.flush()

    for aid, q in pairs:
        db.add(GSLotItem(gs_lot_id=newlot.id, anneal_lot_id=aid, qty=q))
        a = db.query(AnnealLot).get(aid)
        a.available_qty = max((a.available_qty or 0.0) - q, 0.0)
        db.add(a)

    db.commit()
    return RedirectResponse("/screening", status_code=303)


@app.get("/screening/downtime", response_class=HTMLResponse)
def gs_downtime_page(request: Request, db: Session = Depends(get_db),
                     _guard = require_roles("admin","screening")):
    rows = db.query(GSDowntime).order_by(GSDowntime.date.desc(), GSDowntime.id.desc()).limit(50).all()
    return templates.TemplateResponse("gs_downtime.html", {
        "request": request, "rows": rows, "today": dt.date.today().isoformat()
    })

@app.post("/screening/downtime")
def gs_downtime_save(
    request: Request,
    db: Session = Depends(get_db),
    date: str = Form(...),
    minutes: int = Form(...),
    kind: Optional[str] = Form(None),
    remarks: Optional[str] = Form(None),
    _guard = require_roles("admin", "screening"),
):
    d = dt.date.fromisoformat(date)
    db.add(GSDowntime(date=d, minutes=int(minutes or 0), kind=kind, remarks=remarks))
    db.commit()
    return RedirectResponse("/screening/downtime", status_code=303)

@app.get("/screening/downtime/export")
def gs_downtime_csv(db: Session = Depends(get_db)):
    import csv
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["date","minutes","kind","remarks"])
    for r in db.query(GSDowntime).order_by(GSDowntime.date).all():
        w.writerow([r.date.isoformat(), r.minutes, r.kind or "", r.remarks or ""])
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="screening_downtime.csv"'}
    )

# --- Screening QA (chem incl. O, plus compressibility) ---
@app.get("/qa/screen/lot/{lot_id}", response_class=HTMLResponse)
def qa_screen_lot_form(lot_id: int,
                       request: Request,
                       db: Session = Depends(get_db),
                       _guard = require_roles("admin","screening","qa","view")):
    lot = db.get(Lot, lot_id)
    if not lot:
        return _alert_redirect("Lot not found", "/screening")
    grade = (lot.grade or "KIP")
    read_only = _is_read_only(request) or (current_role(request) not in ("admin","screening","qa"))
    data = _prefill_screen_chem_from_upstream(db, lot)
    return templates.TemplateResponse("qa_screen_lot.html", {
        "request": request,
        "lot": lot,
        "grade": grade,
        "chem": {k: data.get(k,"") for k in ["c","si","s","p","cu","ni","mn","fe","o"]},
        "phys": {"compressibility": data.get("compressibility","")},
        "read_only": read_only,
        "role": current_role(request),
    })

@app.post("/qa/screen/lot/{lot_id}")
def qa_screen_lot_save(lot_id: int,
                       request: Request,
                       c: str = Form(""), si: str = Form(""), s: str = Form(""), p: str = Form(""),
                       cu: str = Form(""), ni: str = Form(""), mn: str = Form(""), fe: str = Form(""),
                       o: str = Form(""),
                       compressibility: str = Form(""),
                       decision: str = Form(...),
                       remarks: str = Form(""),
                       db: Session = Depends(get_db),
                       _guard = require_roles("admin","screening","qa")):
    lot = db.get(Lot, lot_id)
    if not lot:
        return _alert_redirect("Lot not found", "/screening")
    row = db.query(ScreenQA).filter(ScreenQA.lot_id == lot.id).first()
    if not row:
        row = ScreenQA(lot_id=lot.id)
    row.c, row.si, row.s, row.p = c, si, s, p
    row.cu, row.ni, row.mn, row.fe = cu, ni, mn, fe
    row.o = o
    row.compressibility = compressibility
    db.add(row)
    lot.qa_status = decision or "PENDING"
    lot.qa_remarks = remarks or ""
    db.add(lot); db.commit()
    return RedirectResponse(f"/qa/screen/lot/{lot.id}", status_code=303)

# ==== PART B2 END ====

# ==== PART B3 START ====

# -------------------------------------------------
# Traceability – LOT (full chain: Heat + FIFO grn)
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

    # FIFO GRN rows
    rows = []
    for h in Heat:
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
            "Heat": Heat,
            "alloc_map": alloc_map,
            "grn_rows": rows
        }
    )


# -------------------------------------------------
# Traceability – HEAT (RM → GRN FIFO breakdown)
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
# Lot quick views for RAP “Docs” column (Trace + QA)
# -------------------------------------------------
@app.get("/lot/{lot_id}/trace", response_class=HTMLResponse)
def lot_trace_view(lot_id: int, db: Session = Depends(get_db)):
    lot = db.get(Lot, lot_id)
    if not lot:
        return HTMLResponse("<p>Lot not found</p>", status_code=404)

    rows = []
    for lh in getattr(lot, "Heat", []):
        h = db.get(Heat, lh.heat_id)
        if not h:
            continue
        rows.append({
            "heat_no": h.heat_no,
            "alloc": float(lh.qty or 0.0),
            "out": float(h.actual_output or 0.0),
            "qa": h.qa_status or "",
            "rm_type": "", "grn_id": "", "supplier": "", "rm_qty": ""
        })
        for cons in getattr(h, "rm_consumptions", []):
            g = cons.grn
            rows.append({
                "heat_no": "", "alloc": "", "out": "", "qa": "",
                "rm_type": cons.rm_type, "grn_id": cons.grn_id,
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

    def v(x): return ("—" if x in (None, "",) else x)

    rows = []
    for lh in getattr(lot, "Heat", []):
        h = db.get(Heat, lh.heat_id)
        if not h:
            continue
        rows.append({
            "heat_no": h.heat_no,
            "qa": getattr(h, "qa_status", "") or "—",
            "notes": getattr(h, "qa_notes", "") if hasattr(h, "qa_notes") else "—",
        })

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
        html.append("<tr><td colspan='3'>No Heat recorded.</td></tr>")
    html.append("</table><br>")

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

    html += ["<h3>Physical</h3>",
             "<table border='1' cellpadding='6' cellspacing='0'>",
             "<tr><th>AD (g/cc)</th><th>Flow (s/50g)</th></tr>"]
    if phys:
        html.append(f"<tr><td>{v(getattr(phys,'ad',None))}</td><td>{v(getattr(phys,'flow',None))}</td></tr>")
    else:
        html.append("<tr><td colspan='2'>—</td></tr>")
    html.append("</table><br>")

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


# -------------------------------------------------
# QA Export (Heat + Lot) – by date range
# -------------------------------------------------
@app.get("/qa/export")
def qa_export(
    start: Optional[str] = None,
    end: Optional[str] = None,
    db: Session = Depends(get_db),
):
    heats = db.query(Heat).order_by(Heat.id.asc()).all()
    Lot  = db.query(Lot ).order_by(Lot.id.asc()).all()

    today = dt.date.today()
    s = dt.date.fromisoformat(start) if start else today
    e = dt.date.fromisoformat(end)   if end   else today

    def _hdate(h: Heat) -> dt.date:
        return heat_date_from_no(h.heat_no) or today
    def _ldate(l: Lot) -> dt.date:
        return lot_date_from_no(l.lot_no) or today

    Heat_in = [h for h in Heat if s <= _hdate(h) <= e]
    Lot_in  = [l for l in Lot  if s <= _ldate(l) <= e]

    out = io.StringIO()
    w = out.write
    w("Type,ID,Date,Grade/Type,Weight/Output (kg),QA Status,C,Si,S,P,Cu,Ni,Mn,Fe\n")

    for h in Heat_in:
        chem = h.chemistry
        w(
            f"HEAT,{h.heat_no},{_hdate(h).isoformat()},"
            f"{('KRFS' if heat_grade(h)=='KRFS' else 'KRIP')},"
            f"{float(h.actual_output or 0.0):.1f},{h.qa_status or ''},"
            f"{(chem.c  if chem else '')},"
            f"{(chem.si if chem else '')},"
            f"{(chem.s  if chem else '')},"
            f"{(chem.p  if chem else '')},"
            f"{(chem.cu if chem else '')},"
            f"{(chem.ni if chem else '')},"
            f"{(chem.mn if chem else '')},"
            f"{(chem.fe if chem else '')}\n"
        )

    for l in Lot_in:
        chem = l.chemistry
        w(
            f"LOT,{l.lot_no},{_ldate(l).isoformat()},{l.grade or ''},"
            f"{float(l.weight or 0.0):.1f},{l.qa_status or ''},"
            f"{(chem.c  if chem else '')},"
            f"{(chem.si if chem else '')},"
            f"{(chem.s  if chem else '')},"
            f"{(chem.p  if chem else '')},"
            f"{(chem.cu if chem else '')},"
            f"{(chem.ni if chem else '')},"
            f"{(chem.mn if chem else '')},"
            f"{(chem.fe if chem else '')}\n"
        )

    data = out.getvalue().encode("utf-8")
    fname = f"qa_export_{s.isoformat()}_{e.isoformat()}.csv"
    return StreamingResponse(
        io.BytesIO(data),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'}
    )


# -------------------------------------------------
# Lot PDFs (1) quick summary used anywhere  (2) generic /pdf/lot
# -------------------------------------------------
@app.get("/lot/{lot_id}/pdf")
def lot_pdf_view(lot_id: int, db: Session = Depends(get_db)):
    lot = db.get(Lot, lot_id)
    if not lot:
        return PlainTextResponse("Lot not found", status_code=404)

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    # header
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

    # section title
    c.setFont("Helvetica-Bold", 11)
    c.drawString(2 * cm, y, "Heat & GRN (FIFO)"); y -= 14
    c.setFont("Helvetica", 10)

    def new_page():
        nonlocal y
        c.showPage()
        try:
            draw_header(c, "Lot Summary")
        except Exception:
            c.setFont("Helvetica-Bold", 18)
            c.drawString(2 * cm, h - 2.5 * cm, "KRN Alloys Pvt. Ltd")
            c.setFont("Helvetica", 12)
            c.drawString(2 * cm, h - 3.2 * cm, "Lot Summary")
        y = h - 4 * cm
        c.setFont("Helvetica-Bold", 11)
        c.drawString(2 * cm, y, "Heat & GRN (FIFO)"); y -= 14
        c.setFont("Helvetica", 10)

    # parent rows (Heat) + child rows (GRN FIFO)
    for lh in getattr(lot, "Heat", []):
        hobj = db.get(Heat, lh.heat_id)
        if not hobj:
            continue
        if y < 3 * cm:
            new_page()
        c.drawString(2.2 * cm, y, f"{hobj.heat_no} | Alloc: {float(lh.qty or 0):.1f} kg | QA: {hobj.qa_status or ''} | Out: {float(hobj.actual_output or 0):.1f} kg")
        y -= 12
        for cons in getattr(hobj, "rm_consumptions", []):
            if y < 3 * cm:
                new_page()
            g = cons.grn
            supplier = (g.supplier if g else "")
            c.drawString(2.8 * cm, y, f"– {cons.rm_type} | GRN #{cons.grn_id} | {supplier} | {float(cons.qty or 0):.1f} kg")
            y -= 12

    c.showPage(); c.save()
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="lot_{lot.lot_no}.pdf"'}
    )


@app.get("/pdf/lot/{lot_id}")
def pdf_lot(lot_id: int, db: Session = Depends(get_db)):
    lot = db.get(Lot, lot_id)
    if not lot:
        return PlainTextResponse("Lot not found", status_code=404)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    draw_header(c, f"Traceability Report – Lot {lot.lot_no}")

    y = height - 4 * cm
    c.setFont("Helvetica", 11)
    c.drawString(2 * cm, y, f"Grade: {lot.grade or '-'}"); y -= 14
    c.drawString(2 * cm, y, f"Weight: {float(lot.weight or 0):.1f} kg"); y -= 14
    c.drawString(2 * cm, y, f"Lot QA: {lot.qa_status or '-'}"); y -= 18

    c.setFont("Helvetica-Bold", 11)
    c.drawString(2 * cm, y, "Heat (Allocation)"); y -= 14
    c.setFont("Helvetica", 10)
    for lh in lot.Heat:
        h = lh.heat
        c.drawString(2.2 * cm, y,
            f"{h.heat_no}  | Alloc to lot: {float(lh.qty or 0):.1f} kg  | Heat Out: {float(h.actual_output or 0):.1f} kg  | QA: {h.qa_status or '-'}")
        y -= 12
        if y < 3 * cm:
            c.showPage(); draw_header(c, f"Traceability Report – Lot {lot.lot_no}"); y = height - 4 * cm

    y -= 6
    c.setFont("Helvetica-Bold", 11); c.drawString(2 * cm, y, "GRN Consumption (FIFO)"); y -= 14
    c.setFont("Helvetica", 10)
    for lh in lot.Heat:
        h = lh.heat
        for cons in h.rm_consumptions:
            g = cons.grn
            c.drawString(2.2 * cm, y, f"Heat {h.heat_no} | {cons.rm_type} | GRN #{cons.grn_id} | {g.supplier if g else ''} | {float(cons.qty or 0):.1f} kg")
            y -= 12
            if y < 3 * cm:
                c.showPage(); draw_header(c, f"Traceability Report – Lot {lot.lot_no}"); y = height - 4 * cm

    # Optional: attach QA annexure
    try:
        draw_lot_qa_annexure(c, lot)
    except Exception:
        pass

    c.showPage(); c.save()
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="trace_{lot.lot_no}.pdf"'}
    )

# ==== PART B3 END ====

# ===========================
# B4 — Branding + PDFs (drop-in)
# ===========================

# --- KRN static logo loader ---
def _krn_logo_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    # try ./static then ../static
    c1 = os.path.join(here, "static", "KRN_Logo.png")
    c2 = os.path.join(here, "..", "static", "KRN_Logo.png")
    return c1 if os.path.exists(c1) else c2

# --- Always-on watermark on every page ---
def add_watermark(c: canvas.Canvas, text: str = "KRN Confidential"):
    width, height = A4
    c.saveState()
    c.setFont("Helvetica-Bold", 60)
    c.setFillGray(0.85)     # light
    c.translate(width/2, height/2)
    c.rotate(45)
    c.drawCentredString(0, 0, text)
    c.restoreState()

# --- Branded header & footer ---
def draw_header(c: canvas.Canvas, title: str):
    width, height = A4
    # watermark is ALWAYS on every page
    add_watermark(c)

    # header strip
    c.setLineWidth(1)
    c.setStrokeColorRGB(0.96, 0.62, 0.04)  # KRN orange-ish rule
    c.line(1.5*cm, height-3.35*cm, width-1.5*cm, height-3.35*cm)

    # logo
    lp = _krn_logo_path()
    if os.path.exists(lp):
        # preserve aspect & mask
        c.drawImage(lp, 1.5*cm, height-3.0*cm, width=4.0*cm, height=2.4*cm,
                    preserveAspectRatio=True, mask='auto')

    # title
    c.setFont("Helvetica-Bold", 18)
    c.setFillGray(0.1)
    c.drawString(7.0*cm, height-2.2*cm, "KRN Alloys Pvt. Ltd – Plant 1")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(7.0*cm, height-2.8*cm, title)

def draw_footer(c: canvas.Canvas):
    width, _ = A4
    c.setFont("Helvetica", 9)
    c.setFillGray(0.3)
    c.drawString(1.5*cm, 1.2*cm, "Generated by KRN MRP/QA Suite")
    c.drawRightString(width-1.5*cm, 1.2*cm, dt.datetime.now().strftime("%Y-%m-%d %H:%M"))
    # bottom rule
    c.setStrokeColorRGB(0.9, 0.9, 0.9)
    c.line(1.5*cm, 1.5*cm, width-1.5*cm, 1.5*cm)

# Utility: start a new branded page (header+footer+watermark)
def _start_page(c: canvas.Canvas, title: str):
    draw_header(c, title)
    draw_footer(c)
    # content should begin below header area:
    _, height = A4
    return height - 4.0*cm  # suggested top Y for content

# -------- QA Annexure (stage-agnostic) ----------
def draw_lot_qa_annexure(c: canvas.Canvas, lot: Lot, db: Session, start_y: float | None = None):
    """
    Adds QA annexure sections for the given ATOM lot:
      - Chemistry (LotChem if present)
      - Physical (LotPhys if present)
      - PSD (LotPSD if present)
      - If Screening QA exists (ScreenQA), also prints O and Compressibility
      - If Anneal QA exists (AnnealQA), prints Oxygen
    Automatically paginates with branded header/footer/watermark.
    """
    width, height = A4
    y = start_y or _start_page(c, f"QA Certificate – {lot.lot_no}")

    def ensure_room(lines: int, title: str | None = None):
        nonlocal y
        if y - 12*lines < 2.2*cm:
            c.showPage()
            y = _start_page(c, f"QA Certificate – {lot.lot_no}")
            if title:
                c.setFont("Helvetica-Bold", 12); c.drawString(2*cm, y, title); y -= 16

    # Title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2*cm, y, f"QA Certificate – Lot {lot.lot_no}"); y -= 20

    # Chemistry
    c.setFont("Helvetica-Bold", 12); c.drawString(2*cm, y, "Chemistry (Atomization):"); y -= 16
    c.setFont("Helvetica", 10)
    chem = getattr(lot, "chemistry", None)
    if chem:
        rows = [("C", chem.c),("Si", chem.si),("S", chem.s),("P", chem.p),
                ("Cu", chem.cu),("Ni", chem.ni),("Mn", chem.mn),("Fe", chem.fe)]
        for k, v in rows:
            ensure_room(1)
            c.drawString(2.5*cm, y, f"{k}: {v or ''}"); y -= 12
    else:
        ensure_room(1)
        c.drawString(2.5*cm, y, "No chemistry data"); y -= 14

    # Physical
    c.setFont("Helvetica-Bold", 12); c.drawString(2*cm, y, "Physical (Atomization):"); y -= 16
    c.setFont("Helvetica", 10)
    phys = getattr(lot, "phys", None)
    if phys:
        for k, v in (("AD (g/cc)", phys.ad),("Flow (s/50g)", phys.flow)):
            ensure_room(1)
            c.drawString(2.5*cm, y, f"{k}: {v or ''}"); y -= 12
    else:
        ensure_room(1)
        c.drawString(2.5*cm, y, "No physical data"); y -= 14

    # PSD
    c.setFont("Helvetica-Bold", 12); c.drawString(2*cm, y, "Particle Size Distribution (Atomization):"); y -= 16
    c.setFont("Helvetica", 10)
    psd = getattr(lot, "psd", None)
    if psd:
        rows = [("+212", psd.p212),("+180", psd.p180),("-180+150", psd.n180p150),
                ("-150+75", psd.n150p75),("-75+45", psd.n75p45),("-45", psd.n45)]
        for k, v in rows:
            ensure_room(1)
            c.drawString(2.5*cm, y, f"{k}: {v or ''}"); y -= 12
    else:
        ensure_room(1)
        c.drawString(2.5*cm, y, "No PSD data"); y -= 14

    # Annealing QA (Oxygen)
    aqa = db.query(AnnealQA).filter(AnnealQA.lot_id == lot.id).first()
    c.setFont("Helvetica-Bold", 12); c.drawString(2*cm, y, "Annealing QA (Oxygen):"); y -= 16
    c.setFont("Helvetica", 10)
    ensure_room(1)
    c.drawString(2.5*cm, y, f"Oxygen: {(aqa.oxygen if aqa else '')}"); y -= 14

    # Screening QA (Chem+Compressibility)
    sqa = db.query(ScreenQA).filter(ScreenQA.lot_id == lot.id).first()
    c.setFont("Helvetica-Bold", 12); c.drawString(2*cm, y, "Screening QA (Final):"); y -= 16
    c.setFont("Helvetica", 10)
    if sqa:
        rows = [("C", sqa.c),("Si", sqa.si),("S", sqa.s),("P", sqa.p),
                ("Cu", sqa.cu),("Ni", sqa.ni),("Mn", sqa.mn),("Fe", sqa.fe),
                ("O", sqa.o),("Compressibility (g/cc)", sqa.compressibility)]
        for k, v in rows:
            ensure_room(1)
            c.drawString(2.5*cm, y, f"{k}: {v or ''}"); y -= 12
    else:
        ensure_room(1)
        c.drawString(2.5*cm, y, "No screening QA data"); y -= 14

# -------- Lot PDF (Traceability) — REPLACE existing versions ----------
@app.get("/lot/{lot_id}/pdf")
def lot_pdf_view(lot_id: int, request: Request, db: Session = Depends(get_db)):
    lot = db.get(Lot, lot_id)
    if not lot:
        return PlainTextResponse("Lot not found", status_code=404)

    role = current_role(request)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    # Page 1: Summary
    y = _start_page(c, "Traceability Report – Lot Summary")
    c.setFont("Helvetica", 11)
    c.drawString(2*cm, y, f"Lot: {lot.lot_no}    Grade: {lot.grade or '-'}"); y -= 14
    c.drawString(2*cm, y, f"Lot Weight: {float(lot.weight or 0):.1f} kg"); y -= 14
    if role == "admin":
        c.drawString(2*cm, y, f"Unit Cost: ₹{float(lot.unit_cost or 0):.2f}   Total Cost: ₹{float(lot.total_cost or 0):.2f}")
        y -= 14
    c.drawString(2*cm, y, f"Lot QA: {lot.qa_status or '-'}"); y -= 18

    # Heat & allocations
    c.setFont("Helvetica-Bold", 11); c.drawString(2*cm, y, "Heat used in this Lot"); y -= 14
    c.setFont("Helvetica", 10)

    def need_new_page(lines=1):
        nonlocal y
        if y - 12*lines < 2.4*cm:
            c.showPage()
            y = _start_page(c, "Traceability Report – Lot Summary")
            c.setFont("Helvetica-Bold", 11); c.drawString(2*cm, y, "Heat used in this Lot"); y -= 14
            c.setFont("Helvetica", 10)

    for lh in getattr(lot, "Heat", []):
        hobj = db.get(Heat, lh.heat_id)
        if not hobj:
            continue
        need_new_page()
        c.drawString(2.2*cm, y,
            f"{hobj.heat_no} | Alloc: {float(lh.qty or 0):.1f} kg | Heat Out: {float(hobj.actual_output or 0):.1f} kg | QA: {hobj.qa_status or ''}"
        ); y -= 12
        # GRN FIFO children
        for cons in getattr(hobj, "rm_consumptions", []):
            need_new_page()
            g = cons.grn
            supplier = (g.supplier if g else "")
            c.drawString(2.8*cm, y, f"– {cons.rm_type} | GRN #{cons.grn_id} | {supplier} | {float(cons.qty or 0):.1f} kg"); y -= 12

    # Annexure page(s)
    c.showPage()
    _start_page(c, "Traceability Report – QA Annexure")
    draw_lot_qa_annexure(c, lot, db)

    c.showPage(); c.save()
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="lot_{lot.lot_no}.pdf"'}
    )

# Backward-compatible alias (REPLACE if you also have /pdf/lot/{id})
@app.get("/pdf/lot/{lot_id}")
def pdf_lot(lot_id: int, request: Request, db: Session = Depends(get_db)):
    return lot_pdf_view(lot_id, request, db)

# ----- Patch for rap_dispatch_pdf loop bug (call this where your function lives) -----
# In your existing `rap_dispatch_pdf(alloc_id)` (Part 6), replace:
#     for item in dispatch.items:
# with:
#     items = db.query(RAPDispatchItem).filter(RAPDispatchItem.dispatch_id == disp.id).all()
#     for item in items:
#         draw_lot_qa_annexure(c, item.lot, db)
#
# (And feel free to add `_start_page(c, "Dispatch Note")` at the top + use draw_footer() on each page
# if you want dispatch PDFs branded the same way.)
