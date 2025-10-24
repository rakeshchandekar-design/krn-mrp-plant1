import os, io, datetime as dt
from typing import Optional, Dict, List, Tuple
from urllib.parse import quote
from enum import Enum
from starlette.middleware.sessions import SessionMiddleware
import secrets

import os
from uuid import uuid4
from fastapi import File, UploadFile, Form, Request
from starlette.staticfiles import StaticFiles

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

# ---- File uploads config ----
BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

# SQLAlchemy
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey, func, text
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session
from sqlalchemy.orm import joinedload  # eager load
from sqlalchemy import func

def date_field(model):
    """
    Return a SQLAlchemy column on `model` that represents creation/production time.
    Adjust the priority order if your column names differ.
    """
    for name in ("date", "day", "prod_date", "production_date",
                 "created_at", "created_on", "timestamp", "ts"):
        if hasattr(model, name):
            return getattr(model, name)
    raise AttributeError(f"{model.__name__} has no date-like column")

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

                # --- Annealing tables ---
        if str(engine.url).startswith("sqlite"):
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS anneal_lots(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lot_no TEXT UNIQUE NOT NULL,
                    date DATE NOT NULL,
                    src_alloc_json TEXT NOT NULL,
                    grade TEXT NOT NULL,
                    weight_kg REAL NOT NULL,
                    rap_cost_per_kg REAL NOT NULL DEFAULT 0,
                    cost_per_kg REAL NOT NULL DEFAULT 0,
                    ammonia_kg REAL NOT NULL DEFAULT 0,
                    qa_status TEXT NOT NULL DEFAULT 'PENDING',
                    c_pct REAL, si_pct REAL, mn_pct REAL, s_pct REAL, p_pct REAL,
                    o_pct REAL, compressibility REAL,
                    remarks TEXT,
                    created_by TEXT,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS anneal_downtime(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    minutes INTEGER NOT NULL,
                    area TEXT NOT NULL,
                    reason TEXT NOT NULL
                )
            """))
        else:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS anneal_lots(
                    id SERIAL PRIMARY KEY,
                    lot_no TEXT UNIQUE NOT NULL,
                    date DATE NOT NULL,
                    src_alloc_json TEXT NOT NULL,
                    grade TEXT NOT NULL,
                    weight_kg DOUBLE PRECISION NOT NULL,
                    rap_cost_per_kg DOUBLE PRECISION NOT NULL DEFAULT 0,
                    cost_per_kg DOUBLE PRECISION NOT NULL DEFAULT 0,
                    ammonia_kg DOUBLE PRECISION NOT NULL DEFAULT 0,
                    qa_status TEXT NOT NULL DEFAULT 'PENDING',
                    c_pct DOUBLE PRECISION, si_pct DOUBLE PRECISION, mn_pct DOUBLE PRECISION,
                    s_pct DOUBLE PRECISION, p_pct DOUBLE PRECISION,
                    o_pct DOUBLE PRECISION, compressibility DOUBLE PRECISION,
                    remarks TEXT, created_by TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS anneal_downtime(
                    id SERIAL PRIMARY KEY,
                    date DATE NOT NULL,
                    minutes INT NOT NULL,
                    area TEXT NOT NULL,
                    reason TEXT NOT NULL
                )
            """))

        # --- Safety: ensure anneal_lots has expected columns ---
        for coldef in [
            "rap_cost_per_kg REAL DEFAULT 0",
            "cost_per_kg REAL DEFAULT 0",
            "ammonia_kg REAL DEFAULT 0",
            "o_pct REAL",
            "compressibility REAL"
        ]:
            col = coldef.split()[0]
            if not _table_has_column(conn, "anneal_lots", col):
                if str(engine.url).startswith("sqlite"):
                    conn.execute(text(f"ALTER TABLE anneal_lots ADD COLUMN {coldef}"))
                else:
                    sql = coldef.replace("REAL", "DOUBLE PRECISION")
                    conn.execute(text(f"ALTER TABLE anneal_lots ADD COLUMN IF NOT EXISTS {col} {sql.split(' ',1)[1]}"))

        # --- Safety: ensure anneal_downtime has expected columns ---
        for coldef in [
            "area TEXT",
            "reason TEXT"
        ]:
            col = coldef.split()[0]
            if not _table_has_column(conn, "anneal_downtime", col):
                if str(engine.url).startswith("sqlite"):
                    conn.execute(text(f"ALTER TABLE anneal_downtime ADD COLUMN {coldef}"))
                else:
                    conn.execute(text(f"ALTER TABLE anneal_downtime ADD COLUMN IF NOT EXISTS {col} {coldef.split(' ',1)[1]}"))

# ================================
# --- Grinding & Screening (DDL + safety) ---
# Matches krn_mrp_app/grinding/routes.py
# ================================
with engine.begin() as conn:
    if str(engine.url).startswith("sqlite"):
        # Main header table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS grinding_lots(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lot_no TEXT UNIQUE NOT NULL,
                date DATE NOT NULL,
                anneal_lot_id INTEGER,                 -- optional FK to anneal_lots.id
                src_alloc_json TEXT NOT NULL,          -- {"ANL-20251011-001": 500.0, ...}
                grade TEXT NOT NULL,                   -- KIP / KFS family
                weight_kg REAL NOT NULL,               -- final lot weight after G&S
                oversize_p80_kg REAL NOT NULL DEFAULT 0,  -- +80 kg
                oversize_p40_kg REAL NOT NULL DEFAULT 0,  -- +40 kg
                input_cost_per_kg REAL NOT NULL DEFAULT 0,   -- weighted input (anneal) cost
                process_cost_per_kg REAL NOT NULL DEFAULT 0, -- fixed process cost (₹6/kg)
                cost_per_kg REAL NOT NULL DEFAULT 0,          -- final: input + process
                qa_status TEXT NOT NULL DEFAULT 'PENDING',
                compressibility REAL,                  -- extra QA parameter
                remarks TEXT,
                created_by TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """))
        # QA header
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS grinding_qa(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                grinding_lot_id INTEGER NOT NULL,
                decision TEXT NOT NULL,
                oxygen REAL,
                compressibility REAL,
                remarks TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """))
        # QA parameters (name/value pairs)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS grinding_qa_params(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                grinding_qa_id INTEGER NOT NULL,
                param_name TEXT NOT NULL,
                param_value TEXT
            )
        """))
        # Downtime
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS grinding_downtime(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                minutes INTEGER NOT NULL,
                area TEXT NOT NULL,
                reason TEXT NOT NULL
            )
        """))
    else:
        # Postgres
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS grinding_lots(
                id SERIAL PRIMARY KEY,
                lot_no TEXT UNIQUE NOT NULL,
                date DATE NOT NULL,
                anneal_lot_id INT,
                src_alloc_json TEXT NOT NULL,
                grade TEXT NOT NULL,
                weight_kg DOUBLE PRECISION NOT NULL,
                oversize_p80_kg DOUBLE PRECISION NOT NULL DEFAULT 0,
                oversize_p40_kg DOUBLE PRECISION NOT NULL DEFAULT 0,
                input_cost_per_kg DOUBLE PRECISION NOT NULL DEFAULT 0,
                process_cost_per_kg DOUBLE PRECISION NOT NULL DEFAULT 0,
                cost_per_kg DOUBLE PRECISION NOT NULL DEFAULT 0,
                qa_status TEXT NOT NULL DEFAULT 'PENDING',
                compressibility DOUBLE PRECISION,
                remarks TEXT,
                created_by TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS grinding_qa(
                id SERIAL PRIMARY KEY,
                grinding_lot_id INT NOT NULL,
                decision TEXT NOT NULL,
                oxygen DOUBLE PRECISION,
                compressibility DOUBLE PRECISION,
                remarks TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS grinding_qa_params(
                id SERIAL PRIMARY KEY,
                grinding_qa_id INT NOT NULL,
                param_name TEXT NOT NULL,
                param_value TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS grinding_downtime(
                id SERIAL PRIMARY KEY,
                date DATE NOT NULL,
                minutes INT NOT NULL,
                area TEXT NOT NULL,
                reason TEXT NOT NULL
            )
        """))
        # Helpful indexes (no-op if already present)
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_grinding_lots_date ON grinding_lots(date)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_grinding_lots_qa ON grinding_lots(qa_status)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_grinding_qa_lot ON grinding_qa(grinding_lot_id)"))

    # --- Safety: ensure grinding_lots has expected columns (SQLite + Postgres)
    for coldef in [
        "anneal_lot_id INT",
        "src_alloc_json TEXT",
        "oversize_p80_kg REAL DEFAULT 0",
        "oversize_p40_kg REAL DEFAULT 0",
        "input_cost_per_kg REAL DEFAULT 0",
        "process_cost_per_kg REAL DEFAULT 0",
        "cost_per_kg REAL DEFAULT 0",
        "compressibility REAL",
    ]:
        col = coldef.split()[0]
        if not _table_has_column(conn, "grinding_lots", col):
            if str(engine.url).startswith("sqlite"):
                conn.execute(text(f"ALTER TABLE grinding_lots ADD COLUMN {coldef}"))
            else:
                sql = coldef.replace("REAL", "DOUBLE PRECISION")  # map types for PG
                conn.execute(text(f"ALTER TABLE grinding_lots ADD COLUMN IF NOT EXISTS {col} {sql.split(' ',1)[1]}"))

    # --- Safety: ensure grinding_downtime has expected columns (area, reason)
    for coldef in [
        "area TEXT",
        "reason TEXT",
    ]:
        col = coldef.split()[0]
        if not _table_has_column(conn, "grinding_downtime", col):
            if str(engine.url).startswith("sqlite"):
                conn.execute(text(f"ALTER TABLE grinding_downtime ADD COLUMN {coldef}"))
            else:
                conn.execute(text(f"ALTER TABLE grinding_downtime ADD COLUMN IF NOT EXISTS {col} {coldef.split(' ',1)[1]}"))
# ================================
# --- Packing & FG (DDL + safety)
# ================================
with engine.begin() as conn:

    if str(engine.url).startswith("sqlite"):
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS fg_lots(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lot_no TEXT UNIQUE NOT NULL,
                date DATE NOT NULL,
                family TEXT NOT NULL,                  -- 'KIP' or 'KFS' or 'OVERSIZE'
                fg_grade TEXT NOT NULL,                -- e.g. 'KIP 80.29', 'KFS 15/45', 'Premixes 01.01', etc.
                weight_kg REAL NOT NULL,
                -- costing
                base_cost_per_kg REAL NOT NULL DEFAULT 0,   -- weighted avg source Grinding cost/kg
                surcharge_per_kg REAL NOT NULL DEFAULT 0,   -- per-grade surcharge
                cost_per_kg REAL NOT NULL DEFAULT 0,        -- base + surcharge
                -- linkage
                src_alloc_json TEXT NOT NULL,          -- {"GRD-20251011-001": 500.0, ...}
                qa_status TEXT NOT NULL DEFAULT 'PENDING',
                remarks TEXT,
                created_by TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS fg_qa(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fg_lot_id INTEGER NOT NULL,
                decision TEXT NOT NULL,
                remarks TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS fg_qa_params(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fg_qa_id INTEGER NOT NULL,
                param_name TEXT NOT NULL,
                param_value TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS fg_downtime(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                minutes INTEGER NOT NULL,
                area TEXT NOT NULL,
                reason TEXT NOT NULL
            )
        """))

    else:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS fg_lots(
                id SERIAL PRIMARY KEY,
                lot_no TEXT UNIQUE NOT NULL,
                date DATE NOT NULL,
                family TEXT NOT NULL,
                fg_grade TEXT NOT NULL,
                weight_kg DOUBLE PRECISION NOT NULL,
                base_cost_per_kg DOUBLE PRECISION NOT NULL DEFAULT 0,
                surcharge_per_kg DOUBLE PRECISION NOT NULL DEFAULT 0,
                cost_per_kg DOUBLE PRECISION NOT NULL DEFAULT 0,
                src_alloc_json TEXT NOT NULL,
                qa_status TEXT NOT NULL DEFAULT 'PENDING',
                remarks TEXT,
                created_by TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS fg_qa(
                id SERIAL PRIMARY KEY,
                fg_lot_id INT NOT NULL,
                decision TEXT NOT NULL,
                remarks TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS fg_qa_params(
                id SERIAL PRIMARY KEY,
                fg_qa_id INT NOT NULL,
                param_name TEXT NOT NULL,
                param_value TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS fg_downtime(
                id SERIAL PRIMARY KEY,
                date DATE NOT NULL,
                minutes INT NOT NULL,
                area TEXT NOT NULL,
                reason TEXT NOT NULL
            )
        """))
        # helpful indexes (no-op if exist)
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fg_lots_date ON fg_lots(date)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fg_lots_qa ON fg_lots(qa_status)"))

    # --- Safety: add any missing columns (both engines) ---
    for coldef in [
        "family TEXT",
        "fg_grade TEXT",
        "weight_kg REAL",
        "base_cost_per_kg REAL DEFAULT 0",
        "surcharge_per_kg REAL DEFAULT 0",
        "cost_per_kg REAL DEFAULT 0",
        "src_alloc_json TEXT",
        "qa_status TEXT",
        "remarks TEXT",
    ]:
        col = coldef.split()[0]
        if not _table_has_column(conn, "fg_lots", col):
            if str(engine.url).startswith("sqlite"):
                conn.execute(text(f"ALTER TABLE fg_lots ADD COLUMN {coldef}"))
            else:
                sql = coldef.replace("REAL", "DOUBLE PRECISION")
                conn.execute(text(f"ALTER TABLE fg_lots ADD COLUMN IF NOT EXISTS {col} {sql.split(' ',1)[1]}"))

# --- FG schema bootstrap ---
def _ensure_fg_schema(conn):
    is_sqlite = (conn.dialect.name == "sqlite")

    if is_sqlite:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS fg_lots(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                lot_no TEXT UNIQUE NOT NULL,
                source_grind_id INTEGER,             -- optional FK to grinding_lots.id
                src_grind_json TEXT,                 -- {"GRD-20241014-001": 500.0, ...}
                fg_grade TEXT NOT NULL,              -- e.g. KIP 80.29, Premixes 01.01, KFS 15/45
                weight_kg REAL NOT NULL DEFAULT 0,
                input_cost_per_kg REAL NOT NULL DEFAULT 0,   -- from grinding
                surcharge_per_kg REAL NOT NULL DEFAULT 0,    -- FG surcharge by grade
                cost_per_kg REAL NOT NULL DEFAULT 0,         -- input + surcharge
                qa_status TEXT NOT NULL DEFAULT 'PENDING',
                remarks TEXT,
                created_by TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS fg_qa(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fg_lot_id INTEGER NOT NULL,
                decision TEXT NOT NULL,
                oxygen REAL,
                compressibility REAL,
                remarks TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS fg_qa_params(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fg_qa_id INTEGER NOT NULL,
                param_name TEXT NOT NULL,
                param_value TEXT
            )
        """))
    else:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS fg_lots(
                id SERIAL PRIMARY KEY,
                date DATE NOT NULL,
                lot_no TEXT UNIQUE NOT NULL,
                source_grind_id INT,
                src_grind_json TEXT,
                fg_grade TEXT NOT NULL,
                weight_kg DOUBLE PRECISION NOT NULL DEFAULT 0,
                input_cost_per_kg DOUBLE PRECISION NOT NULL DEFAULT 0,
                surcharge_per_kg DOUBLE PRECISION NOT NULL DEFAULT 0,
                cost_per_kg DOUBLE PRECISION NOT NULL DEFAULT 0,
                qa_status TEXT NOT NULL DEFAULT 'PENDING',
                remarks TEXT,
                created_by TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS fg_qa(
                id SERIAL PRIMARY KEY,
                fg_lot_id INT NOT NULL,
                decision TEXT NOT NULL,
                oxygen DOUBLE PRECISION,
                compressibility DOUBLE PRECISION,
                remarks TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS fg_qa_params(
                id SERIAL PRIMARY KEY,
                fg_qa_id INT NOT NULL,
                param_name TEXT NOT NULL,
                param_value TEXT
            )
        """))
        # helpful indexes
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fg_lots_date ON fg_lots(date)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fg_lots_qa ON fg_lots(qa_status)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fg_qa_lot ON fg_qa(fg_lot_id)"))

    # Safety: add any columns that might be missing (works for both backends)
    for coldef in [
        "source_grind_id INT",
        "src_grind_json TEXT",
        "input_cost_per_kg REAL DEFAULT 0",
        "surcharge_per_kg REAL DEFAULT 0",
        "cost_per_kg REAL DEFAULT 0",
        "remarks TEXT",
        "created_by TEXT",
    ]:
        col = coldef.split()[0]
        if not _table_has_column(conn, "fg_lots", col):
            if conn.dialect.name == "sqlite":
                conn.execute(text(f"ALTER TABLE fg_lots ADD COLUMN {coldef}"))
            else:
                sql_tail = coldef.replace("REAL", "DOUBLE PRECISION").split(' ', 1)[1]
                conn.execute(text(f"ALTER TABLE fg_lots ADD COLUMN IF NOT EXISTS {col} {sql_tail}"))

# ================================
# --- Dispatch (DDL + safety)
# ================================
with engine.begin() as conn:

    if str(engine.url).startswith("sqlite"):
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS dispatch_orders(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_no TEXT UNIQUE NOT NULL,
                date DATE NOT NULL,
                customer_name TEXT NOT NULL,
                customer_gstin TEXT,
                customer_address TEXT,
                transporter TEXT,
                vehicle_no TEXT,
                lr_no TEXT,
                contact TEXT,
                remarks TEXT,
                created_by TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS dispatch_items(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dispatch_id INTEGER NOT NULL,
                fg_lot_id INTEGER NOT NULL,
                fg_lot_no TEXT NOT NULL,
                fg_grade TEXT NOT NULL,
                qty_kg REAL NOT NULL,
                -- costing snapshot (admin-only UI)
                cost_per_kg REAL NOT NULL DEFAULT 0,
                value REAL NOT NULL DEFAULT 0
            )
        """))
    else:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS dispatch_orders(
                id SERIAL PRIMARY KEY,
                order_no TEXT UNIQUE NOT NULL,
                date DATE NOT NULL,
                customer_name TEXT NOT NULL,
                customer_gstin TEXT,
                customer_address TEXT,
                transporter TEXT,
                vehicle_no TEXT,
                lr_no TEXT,
                contact TEXT,
                remarks TEXT,
                created_by TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS dispatch_items(
                id SERIAL PRIMARY KEY,
                dispatch_id INT NOT NULL,
                fg_lot_id INT NOT NULL,
                fg_lot_no TEXT NOT NULL,
                fg_grade TEXT NOT NULL,
                qty_kg DOUBLE PRECISION NOT NULL,
                cost_per_kg DOUBLE PRECISION NOT NULL DEFAULT 0,
                value DOUBLE PRECISION NOT NULL DEFAULT 0
            )
        """))
        # Helpful indexes
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_dispatch_orders_date ON dispatch_orders(date)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_dispatch_items_dispatch ON dispatch_items(dispatch_id)"))

    # --- Safety: add any missing columns
    for coldef in [
        "customer_gstin TEXT",
        "customer_address TEXT",
        "transporter TEXT",
        "vehicle_no TEXT",
        "lr_no TEXT",
        "contact TEXT",
        "remarks TEXT",
    ]:
        col = coldef.split()[0]
        if not _table_has_column(conn, "dispatch_orders", col):
            if str(engine.url).startswith("sqlite"):
                conn.execute(text(f"ALTER TABLE dispatch_orders ADD COLUMN {coldef}"))
            else:
                conn.execute(text(f"ALTER TABLE dispatch_orders ADD COLUMN IF NOT EXISTS {col} {coldef.split(' ',1)[1]}"))

    for coldef in [
        "cost_per_kg REAL DEFAULT 0",
        "value REAL DEFAULT 0",
    ]:
        col = coldef.split()[0]
        if not _table_has_column(conn, "dispatch_items", col):
            if str(engine.url).startswith("sqlite"):
                conn.execute(text(f"ALTER TABLE dispatch_items ADD COLUMN {coldef}"))
            else:
                sql = coldef.replace("REAL", "DOUBLE PRECISION")
                conn.execute(text(f"ALTER TABLE dispatch_items ADD COLUMN IF NOT EXISTS {col} {sql.split(' ',1)[1]}"))

    # ---- GRN extra columns (safe migrations) ----
    for coldef in [
        "transporter TEXT",
        "vehicle_no TEXT",
        "invoice_file TEXT",
        "ewaybill_file TEXT",
    ]:
        col = coldef.split()[0]
        if not _table_has_column(conn, "grn", col):
            if str(engine.url).startswith("sqlite"):
                conn.execute(text(f"ALTER TABLE grn ADD COLUMN {coldef}"))
            else:
                # Postgres types map cleanly from TEXT
                conn.execute(text(f"ALTER TABLE grn ADD COLUMN IF NOT EXISTS {col} TEXT"))

# ================================
# --- Dispatch helpers (read-only)
# ================================

def _is_admin_request(request: Request) -> bool:
    s = getattr(request, "state", None)
    if not s: return False
    if getattr(s, "is_admin", False): return True
    role = getattr(s, "role", None)
    if isinstance(role, str) and role.lower() == "admin": return True
    roles = getattr(s, "roles", None)
    return isinstance(roles, (list, set, tuple)) and "admin" in roles

def _fg_available_rows() -> list[dict]:
    """
    Returns APPROVED FG lots with available balance:
    avail = fg_lots.weight_kg - SUM(dispatch_items.qty_kg for that fg_lot_id)
    """
    with engine.begin() as conn:
        lots = conn.execute(text("""
            SELECT id, lot_no, date, family, fg_grade,
                   COALESCE(weight_kg,0)::float AS weight_kg,
                   COALESCE(cost_per_kg,0)::float AS cost_per_kg,
                   qa_status, remarks
            FROM fg_lots
            WHERE UPPER(COALESCE(qa_status,''))='APPROVED'
            ORDER BY date DESC, id DESC
        """)).mappings().all()
        used_by_fg = dict(conn.execute(text("""
            SELECT fg_lot_id, COALESCE(SUM(qty_kg),0) AS used
            FROM dispatch_items
            GROUP BY fg_lot_id
        """)).all() or [])
    out = []
    for r in lots:
        used = float(used_by_fg.get(r["id"], 0.0))
        avail = float(r["weight_kg"] or 0.0) - used
        if avail > 0.0001:
            out.append({
                "fg_lot_id": r["id"],
                "fg_lot_no": r["lot_no"],
                "date": r["date"],
                "family": r["family"],
                "fg_grade": r["fg_grade"],
                "available_kg": avail,
                "cost_per_kg": float(r["cost_per_kg"] or 0.0),
                "remarks": r.get("remarks",""),
            })
    return out

def _dispatch_rows_in_range(start: dt.date, end: dt.date) -> list[dict]:
    with engine.begin() as conn:
        orders = conn.execute(text("""
            SELECT id, order_no, date, customer_name, transporter, vehicle_no, lr_no, remarks
            FROM dispatch_orders
            WHERE date BETWEEN :s AND :e
            ORDER BY date DESC, id DESC
        """), {"s": start, "e": end}).mappings().all()
        # attach totals
        out = []
        for o in orders:
            totals = conn.execute(text("""
                SELECT COALESCE(SUM(qty_kg),0) AS qty,
                       COALESCE(SUM(value),0) AS val
                FROM dispatch_items WHERE dispatch_id=:d
            """), {"d": o["id"]}).mappings().first() or {"qty":0,"val":0}
            dct = dict(o); dct["total_qty_kg"] = float(totals["qty"] or 0.0); dct["total_value"] = float(totals["val"] or 0.0)
            out.append(dct)
    return out

def _dispatch_fetch_order(order_id: int) -> tuple[dict|None, list[dict]]:
    with engine.begin() as conn:
        head = conn.execute(text("""
            SELECT * FROM dispatch_orders WHERE id=:id
        """), {"id": order_id}).mappings().first()
        items = conn.execute(text("""
            SELECT * FROM dispatch_items WHERE dispatch_id=:id ORDER BY id
        """), {"id": order_id}).mappings().all()
    return (dict(head) if head else None, [dict(i) for i in items])

def _fg_latest_coa_for_lot(fg_lot_id: int) -> dict|None:
    """
    Compose a CoA payload for an FG lot:
    - Prefer FG QA+params (fg_qa / fg_qa_params)
    - Fallback: dominant Grinding lot in fg_lots.src_alloc_json -> (grinding_qa, grinding_qa_params)
    Returns:
      {
        "header": {"decision": "...", "remarks": "..."},
        "params": [{"name":"C","value":"0.02","unit":"","spec_min":"","spec_max":""}, ...]
      }  or None
    """
    with engine.begin() as conn:
        # FG QA (latest)
        fgqa = conn.execute(text("""
            SELECT id, decision, remarks
            FROM fg_qa WHERE fg_lot_id=:id
            ORDER BY id DESC LIMIT 1
        """), {"id": fg_lot_id}).mappings().first()
        if fgqa:
            params = conn.execute(text("""
                SELECT param_name, param_value
                FROM fg_qa_params WHERE fg_qa_id=:qid ORDER BY id
            """), {"qid": fgqa["id"]}).mappings().all()
            return {
                "header": {"decision": fgqa["decision"], "remarks": fgqa.get("remarks","")},
                "params": [{"name":p["param_name"], "value":p["param_value"], "unit":"","spec_min":"","spec_max":""} for p in params]
            }

        # fallback to Grinding from src_alloc_json
        fg = conn.execute(text("SELECT src_alloc_json FROM fg_lots WHERE id=:id"), {"id": fg_lot_id}).scalar()
        if not fg: return None
        try:
            amap = json.loads(fg)
        except Exception:
            amap = {}
        if not amap: return None
        # pick the grinding lot with largest allocation
        top = max(amap.items(), key=lambda kv: float(kv[1] or 0.0))
        grd_lot_no = top[0]
        grd = conn.execute(text("SELECT id FROM grinding_lots WHERE lot_no=:ln"), {"ln": grd_lot_no}).mappings().first()
        if not grd: return None

        grqa = conn.execute(text("""
            SELECT id, decision, remarks, oxygen, compressibility
            FROM grinding_qa WHERE grinding_lot_id=:gid
            ORDER BY id DESC LIMIT 1
        """), {"gid": grd["id"]}).mappings().first()
        if not grqa: return None
        params = conn.execute(text("""
            SELECT param_name, param_value
            FROM grinding_qa_params WHERE grinding_qa_id=:qid ORDER BY id
        """), {"qid": grqa["id"]}).mappings().all()
        payload = {
            "header": {"decision": grqa["decision"], "remarks": grqa.get("remarks","")},
            "params": [{"name":p["param_name"], "value":p["param_value"], "unit":"","spec_min":"","spec_max":""} for p in params]
        }
        # include O2 and compressibility if present
        if grqa.get("oxygen") is not None:
            payload["params"].insert(0, {"name":"Oxygen","value":f"{float(grqa['oxygen']):.3f}","unit":"","spec_min":"","spec_max":""})
        if grqa.get("compressibility") is not None:
            payload["params"].insert(1, {"name":"Compressibility","value":f"{float(grqa['compressibility']):.2f}","unit":"","spec_min":"","spec_max":""})
        return payload

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
    transporter   = Column(String, nullable=True)
    vehicle_no    = Column(String, nullable=True)
    invoice_file  = Column(String, nullable=True)
    ewaybill_file = Column(String, nullable=True)

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
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
# session middleware (keep the secret; regenerate for production)
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET", secrets.token_hex(16)))
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

from krn_mrp_app.deps import engine  # if main.py needs engine for migrate_schema
from krn_mrp_app.annealing.routes import router as anneal_router
# ✅ Register Annealing router here
from krn_mrp_app.annealing import router as anneal_router
app.include_router(anneal_router, prefix="/anneal", tags=["Annealing"])
from krn_mrp_app.grinding import router as grind_router
app.include_router(grind_router, prefix="/grind", tags=["Grinding & Screening"])
from krn_mrp_app.fg.routes import router as fg_router
app.include_router(fg_router, prefix="/fg", tags=["FG"])
# -------------- include Dispatch router --------------
from krn_mrp_app.dispatch import routes as dispatch_routes
app.include_router(dispatch_routes.router, prefix="/dispatch", tags=["Dispatch"])

# expose python builtins to Jinja
templates.env.globals.update(max=max, min=min, round=round, int=int, float=float)

# ---- Global role helpers for templates + hard read-only enforcement ----
from fastapi.responses import PlainTextResponse

def _role_of(request: Request) -> str:
    """Read role from session; 'guest' if not logged in."""
    try:
        return (request.session or {}).get("role", "guest")
    except Exception:
        return "guest"

def _is_read_only(request: Request) -> bool:
    """Viewer (role='view') is read-only everywhere."""
    return _role_of(request) == "view"

# Make these helpers available in ALL templates:
templates.env.globals.update(role_of=_role_of, is_read_only=_is_read_only)

@app.middleware("http")
async def attach_role_flags(request: Request, call_next):
    """
    Attach role + read_only flags to request.state so templates
    can use:  request.state.role  /  request.state.read_only
    """
    request.state.role = _role_of(request)
    request.state.read_only = _is_read_only(request)
    return await call_next(request)

@app.middleware("http")
async def block_writes_for_view(request: Request, call_next):
    """
    One gate to block every write for the 'view' role across the app.
    No need to change individual routes.
    """
    if _is_read_only(request) and request.method in ("POST", "PUT", "PATCH", "DELETE"):
        return PlainTextResponse("Read-only account: action blocked", status_code=403)
    return await call_next(request)

# ---- Users: username = department; default password = same as username ----
# Change passwords here later.
USER_DB = {
    "admin":   {"password": "admin",   "role": "admin"},
    "store":   {"password": "store",   "role": "store"},
    "melting": {"password": "melting", "role": "melting"},
    "atom":    {"password": "atom",    "role": "atom"},
    "rap":     {"password": "rap",     "role": "rap"},
    "anneal":  {"password": "anneal", "role": "anneal"},
    "grind":   {"password": "grind", "role": "grind"},
    "fg":      {"password": "fg", "role": "fg"},
    "qa":      {"password": "qa",      "role": "qa"},
    "krn":     {"password": "krn",    "role": "view"},
}

def current_username(request: Request) -> str:
    return (getattr(request, "session", {}) or {}).get("user", "") or ""

def current_role(request: Request) -> str:
    return (getattr(request, "session", {}) or {}).get("role", "guest") or "guest"

def role_allowed(request: Request, allowed: set[str]) -> bool:
    role = current_role(request)
    if role == "view":
        # allow GET/HEAD/OPTIONS everywhere
        return request.method in ("GET", "HEAD", "OPTIONS")
    return role in allowed
    
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
def _lot_default_chemistry(db: Session, lot: Lot) -> Dict[str, Optional[float]]:
    """
    Default chemistry for a lot:
      - If any single heat contributes > 60% of the lot weight, use THAT heat's chemistry.
      - Else use a WEIGHTED AVERAGE of all heats' chemistry by their allocation qty.
    Returns dict with numeric floats or None for keys: c, si, s, p, cu, ni, mn, fe
    """
    # gather allocations
    allocs = list(getattr(lot, "heats", []) or [])
    total = sum(float(getattr(a, "qty", 0.0) or 0.0) for a in allocs) or 0.0
    if total <= 0:
        return {k: None for k in ["c","si","s","p","cu","ni","mn","fe"]}

    # check 60% rule
    for a in allocs:
        share = float(a.qty or 0.0) / total
        if share > 0.60:
            h = db.get(Heat, a.heat_id)
            chem = getattr(h, "chemistry", None)
            if not chem:
                continue
            out = {}
            for k in ["c","si","s","p","cu","ni","mn","fe"]:
                try:
                    out[k] = float(getattr(chem, k) or "")
                except Exception:
                    out[k] = None
            return out

    # weighted average
    sums = {k: 0.0 for k in ["c","si","s","p","cu","ni","mn","fe"]}
    have_any = False
    for a in allocs:
        if (a.qty or 0) <= 0:
            continue
        h = db.get(Heat, a.heat_id)
        chem = getattr(h, "chemistry", None)
        if not chem:
            continue
        w = float(a.qty or 0.0)
        for k in list(sums.keys()):
            try:
                v = float(getattr(chem, k) or "")
                sums[k] += v * w
                have_any = True
            except Exception:
                pass
    if not have_any or total <= 0:
        return {k: None for k in ["c","si","s","p","cu","ni","mn","fe"]}
    return {k: (sums[k] / total) for k in sums.keys()}

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


def _save_upload(file: UploadFile | None) -> str | None:
    if not file or not file.filename:
        return None
    ext = os.path.splitext(file.filename)[1].lower()
    name = f"grn_{uuid4().hex}{ext}"
    dest = os.path.join(UPLOAD_DIR, name)
    with open(dest, "wb") as f:
        f.write(file.file.read())
    return name  # store this in DB

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

@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "err": request.query_params.get("err", "")})

@app.post("/login")
async def login_post(request: Request, username: str = Form(...), password: str = Form(...)):
    uname = (username or "").strip().lower()
    u = USER_DB.get(uname) or {}

    if not u or u.get("password") != (password or ""):
        return RedirectResponse("/login?err=Invalid+credentials", status_code=303)

    role = u.get("role", "guest")

    # New unified session format (dict for Annealing)
    request.session["user"] = {"username": uname, "role": role}

    # Legacy keys so older code continues working
    request.session["username"] = uname
    request.session["role"] = role

    return RedirectResponse("/", status_code=303)

@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=303)

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
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "user": current_username(request), "role": current_role(request)}
    )

# -------------------------------
# Dashboard (full replacement)
# -------------------------------
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    role = current_role(request)
    if role not in {"admin", "view"}:
        return RedirectResponse("/", status_code=303)

    today = dt.date.today()
    yest  = today - dt.timedelta(days=1)
    month_start = today.replace(day=1)
    ymd_y = yest.strftime("%Y%m%d")
    ymd_m = month_start.strftime("%Y%m")

    # ---------- RM / GRN ----------
    grn_live_rows = (
        db.query(
            GRN.rm_type,
            func.coalesce(func.sum(GRN.remaining_qty), 0.0),
            func.coalesce(func.sum(GRN.remaining_qty * GRN.price), 0.0),
        )
        .group_by(GRN.rm_type)
        .all()
    )
    grn_live = [
        {"rm_type": r[0], "qty": float(r[1] or 0), "val": float(r[2] or 0)}
        for r in grn_live_rows
    ]
    grn_live_tot_qty = sum(r["qty"] for r in grn_live)
    grn_live_tot_val = sum(r["val"] for r in grn_live)

    # Yesterday’s inward by RM type
    grn_yday_rows = (
        db.query(GRN.rm_type, func.coalesce(func.sum(GRN.qty), 0.0))
        .filter(GRN.date == yest)
        .group_by(GRN.rm_type)
        .all()
    )
    grn_yday_by_type = [{"rm_type": r[0], "qty": float(r[1] or 0)} for r in grn_yday_rows]
    grn_yday_total = sum(r["qty"] for r in grn_yday_by_type)

    grn_mtd_inward = (
        db.query(func.coalesce(func.sum(GRN.qty), 0.0))
        .filter(GRN.date >= month_start, GRN.date <= today)
        .scalar()
        or 0.0
    )

    # ---------- Heats with available stock (grade-wise) ----------
    alloc_map = {
        hid: float(qty or 0)
        for (hid, qty) in db.query(LotHeat.heat_id, func.coalesce(func.sum(LotHeat.qty), 0.0)).group_by(LotHeat.heat_id)
    }
    heats_all = db.query(Heat).options(joinedload(Heat.rm_consumptions)).all()
    avail_by_grade = {"KRIP": {"qty": 0.0, "count": 0}, "KRFS": {"qty": 0.0, "count": 0}}
    for h in heats_all:
        used = float(alloc_map.get(h.id, 0.0))
        avail = max(float(h.actual_output or 0.0) - used, 0.0)
        if avail > 0.0001:
            g = heat_grade(h)
            if g not in avail_by_grade:
                avail_by_grade[g] = {"qty": 0.0, "count": 0}
            avail_by_grade[g]["qty"] += avail
            avail_by_grade[g]["count"] += 1

    # ---------- Melting KPIs ----------
    heats_yday = db.query(Heat).filter(Heat.heat_no.like(f"{ymd_y}-%")).all()
    heats_mtd  = db.query(Heat).filter(Heat.heat_no.like(f"{ymd_m}%")).all()

    def _agg_melting(heats, for_day: dt.date | None = None):
        total_out = sum(float(h.actual_output or 0) for h in heats)
        total_in  = sum(float(h.total_inputs  or 0) for h in heats)
        total_kwh = sum(float(h.power_kwh    or 0) for h in heats)

        kwhpt = (total_kwh / total_out * 1000.0) if total_out > 0 else 0.0
        yield_pct = (total_out / total_in * 100.0) if total_in > 0 else 0.0

        if for_day is not None:
            tgt = day_target_kg(db, for_day)
            eff_pct = (total_out / tgt * 100.0) if tgt > 0 else 0.0
            return dict(actual_kg=total_out, kwhpt=kwhpt, yield_pct=yield_pct, eff_pct=eff_pct, target=tgt)
        else:
            # month: sum of daily targets
            d = month_start
            tgt_sum = 0.0
            while d <= today:
                tgt_sum += day_target_kg(db, d)
                d += dt.timedelta(days=1)
            eff_pct = (total_out / tgt_sum * 100.0) if tgt_sum > 0 else 0.0
            return dict(actual_kg=total_out, kwhpt=kwhpt, yield_pct=yield_pct, eff_pct=eff_pct, target=tgt_sum)

    melt_yday = _agg_melting(heats_yday, for_day=yest)
    melt_mtd  = _agg_melting(heats_mtd,  for_day=None)

    # ---------- Atomization ----------
    # Lots "created" yesterday/month: inferred from lot_no date part
    lots_all = db.query(Lot).all()
    lots_yday = [l for l in lots_all if lot_date_from_no(l.lot_no) == yest]
    lots_mtd  = [l for l in lots_all if (month_start <= (lot_date_from_no(l.lot_no) or today) <= today)]

    atom_yday_prod = sum(float(l.weight or 0) for l in lots_yday)
    atom_mtd_prod  = sum(float(l.weight or 0) for l in lots_mtd)

    atom_tgt_yday = atom_day_target_kg(db, yest)
    atom_tgt_msum = 0.0
    d = month_start
    while d <= today:
        atom_tgt_msum += atom_day_target_kg(db, d)
        d += dt.timedelta(days=1)

    atom_yday_eff = (atom_yday_prod / atom_tgt_yday * 100.0) if atom_tgt_yday > 0 else 0.0
    atom_mtd_eff  = (atom_mtd_prod  / atom_tgt_msum * 100.0) if atom_tgt_msum  > 0 else 0.0

    # Non-approved lots (Pending/Hold/Rejected): qty & cost
    lots_pending  = db.query(Lot).filter((Lot.qa_status.is_(None)) | (Lot.qa_status == "PENDING")).all()
    lots_hold     = db.query(Lot).filter(Lot.qa_status == "HOLD").all()
    lots_rejected = db.query(Lot).filter(Lot.qa_status == "REJECTED").all()
    def _sum_qty_cost(rows): 
        return (
            sum(float(x.weight or 0) for x in rows),
            sum(float(x.total_cost or (x.unit_cost or 0)* (x.weight or 0)) for x in rows)
        )
    pend_qty, pend_cost = _sum_qty_cost(lots_pending)
    hold_qty, hold_cost = _sum_qty_cost(lots_hold)
    rej_qty,  rej_cost  = _sum_qty_cost(lots_rejected)

    # ---------- RAP availability & movements ----------
    # Available qty & cost for APPROVED lots
    lots_approved = db.query(Lot).filter(Lot.qa_status == "APPROVED").all()
    rap_grade_qty  = {"KRIP": 0.0, "KRFS": 0.0}
    rap_grade_cost = {"KRIP": 0.0, "KRFS": 0.0}
    for lot in lots_approved:
        total_alloc = (
            db.query(func.coalesce(func.sum(RAPAlloc.qty), 0.0))
            .join(RAPLot, RAPAlloc.rap_lot_id == RAPLot.id)
            .filter(RAPLot.lot_id == lot.id)
            .scalar()
            or 0.0
        )
        avail = max(float(lot.weight or 0.0) - float(total_alloc), 0.0)
        g = (lot.grade or "KRIP").upper()
        rap_grade_qty[g]  = rap_grade_qty.get(g, 0.0)  + avail
        rap_grade_cost[g] = rap_grade_cost.get(g, 0.0) + avail * float(lot.unit_cost or 0.0)

    # Dispatch & Plant-2 movements: yesterday + MTD
    rap_y_rows = (
        db.query(RAPAlloc.kind, func.coalesce(func.sum(RAPAlloc.qty), 0.0))
        .filter(RAPAlloc.date == yest)
        .group_by(RAPAlloc.kind)
        .all()
    )
    rap_m_rows = (
        db.query(RAPAlloc.kind, func.coalesce(func.sum(RAPAlloc.qty), 0.0))
        .filter(RAPAlloc.date >= month_start, RAPAlloc.date <= today)
        .group_by(RAPAlloc.kind)
        .all()
    )
    rap_y = {k: float(v or 0) for (k, v) in rap_y_rows}
    rap_m = {k: float(v or 0) for (k, v) in rap_m_rows}

    # ---------- QA breakdown ----------
    # Pending (grade-wise)
    heats_pending_rows = [h for h in heats_all if (h.qa_status is None) or (h.qa_status == "PENDING")]
    heat_pending_by_grade = {"KRIP": 0, "KRFS": 0}
    for h in heats_pending_rows:
        heat_pending_by_grade[heat_grade(h)] = heat_pending_by_grade.get(heat_grade(h), 0) + 1

    lot_pending_by_grade_rows = (
        db.query(Lot.grade, func.count(Lot.id))
        .filter((Lot.qa_status.is_(None)) | (Lot.qa_status == "PENDING"))
        .group_by(Lot.grade)
        .all()
    )
    lot_pending_by_grade = { (g or "KRIP"): int(c or 0) for (g, c) in lot_pending_by_grade_rows }

    # Approved qty MTD by grade (lots)
    lots_approved_mtd = (
        l for l in lots_all
        if (l.qa_status == "APPROVED") and (month_start <= (lot_date_from_no(l.lot_no) or today) <= today)
    )
    approved_qty_by_grade = {"KRIP": 0.0, "KRFS": 0.0}
    for l in lots_approved_mtd:
        approved_qty_by_grade[(l.grade or "KRIP")] += float(l.weight or 0)

    # Hold/Reject MTD counts
    hold_mtd = sum(1 for l in lots_all if (l.qa_status == "HOLD") and (month_start <= (lot_date_from_no(l.lot_no) or today) <= today))
    rej_mtd  = sum(1 for l in lots_all if (l.qa_status == "REJECTED") and (month_start <= (lot_date_from_no(l.lot_no) or today) <= today))

    # ---------- Downtime: Yesterday + MTD with kind breakdown ----------
    def _dt_breakdown(model, start: dt.date, end: dt.date):
        total = db.query(func.coalesce(func.sum(model.minutes), 0)).filter(model.date >= start, model.date <= end).scalar() or 0
        by_kind_rows = (
            db.query(model.kind, func.coalesce(func.sum(model.minutes), 0))
            .filter(model.date >= start, model.date <= end)
            .group_by(model.kind)
            .all()
        )
        by_kind = { (k or "OTHER").upper(): int(m or 0) for (k, m) in by_kind_rows }
        # ensure all keys present
        for k in ("PRODUCTION","MAINTENANCE","POWER","OTHER"):
            by_kind.setdefault(k, 0)
        return int(total), by_kind

    melt_dt_y,  melt_by_y  = _dt_breakdown(Downtime, yest, yest)
    melt_dt_m,  melt_by_m  = _dt_breakdown(Downtime, month_start, today)
    atom_dt_y,  atom_by_y  = _dt_breakdown(AtomDowntime, yest, yest)
    atom_dt_m,  atom_by_m  = _dt_breakdown(AtomDowntime, month_start, today)

    ctx = {
        "request": request,
        "today": today.isoformat(),
        "yesterday": yest.isoformat(),
        "power_target": POWER_TARGET_KWH_PER_TON,

        # RM
        "grn_live": grn_live,
        "grn_live_tot_qty": grn_live_tot_qty,
        "grn_live_tot_val": grn_live_tot_val,
        "grn_yday_by_type": grn_yday_by_type,
        "grn_yday_total": grn_yday_total,
        "grn_mtd_inward": grn_mtd_inward,

        # Heats availability
        "avail_by_grade": avail_by_grade,

        # Melting
        "melt_yday": melt_yday,      # includes 'target'
        "melt_mtd":  melt_mtd,       # includes 'target' (sum of day targets)

        # Atomization production
        "atom_yday_prod": atom_yday_prod,
        "atom_tgt_yday":  atom_tgt_yday,
        "atom_yday_eff":  atom_yday_eff,
        "atom_mtd_prod":  atom_mtd_prod,
        "atom_mtd_eff":   atom_mtd_eff,

        # Non-approved lots
        "pend_qty": pend_qty, "pend_cost": pend_cost,
        "hold_qty": hold_qty, "hold_cost": hold_cost,
        "rej_qty":  rej_qty,  "rej_cost":  rej_cost,

        # RAP availability & movements
        "rap_grade_qty":  rap_grade_qty,
        "rap_grade_cost": rap_grade_cost,
        "rap_y": rap_y,  # yesterday; keys DISPATCH/PLANT2
        "rap_m": rap_m,  # MTD

        # QA
        "heat_pending_by_grade": heat_pending_by_grade,
        "lot_pending_by_grade":  lot_pending_by_grade,
        "approved_qty_by_grade": approved_qty_by_grade,
        "hold_mtd": hold_mtd, "rej_mtd": rej_mtd,

        # Downtime
        "melt_dt_y": melt_dt_y, "melt_by_y": melt_by_y,
        "melt_dt_m": melt_dt_m, "melt_by_m": melt_by_m,
        "atom_dt_y": atom_dt_y, "atom_by_y": atom_by_y,
        "atom_dt_m": atom_dt_m, "atom_by_m": atom_by_m,
    }
    return templates.TemplateResponse("dashboard.html", ctx)


# -------------------------------------------------
# GRN (standardized UI hooks; business logic unchanged)
# -------------------------------------------------
@app.get("/grn", response_class=HTMLResponse)
def grn_list(
    request: Request,
    report: Optional[int] = 0,
    start: Optional[str] = None,
    end: Optional[str] = None,
    db: Session = Depends(get_db),
):
    # Allow read-only viewers to see GRN; only admin/store can create
    if not role_allowed(request, {"admin", "store", "view"}):
        return RedirectResponse("/login", status_code=303)

    q = db.query(GRN)
    if report:
        if start:
            q = q.filter(GRN.date >= dt.date.fromisoformat(start))
        if end:
            q = q.filter(GRN.date <= dt.date.fromisoformat(end))
    else:
        # default view = only rows with remaining_qty > 0 (live stock)
        q = q.filter(GRN.remaining_qty > 0)
    grns = q.order_by(GRN.id.desc()).all()

    # Live stock summary per RM type
    live = db.query(GRN).filter(GRN.remaining_qty > 0).all()
    rm_summary = []
    for rm in RM_TYPES:
        subset = [r for r in live if r.rm_type == rm]
        avail = sum(r.remaining_qty or 0.0 for r in subset)
        cost = sum((r.remaining_qty or 0.0) * (r.price or 0.0) for r in subset)
        rm_summary.append({"rm_type": rm, "available": avail, "cost": cost})

    # Totals for live stock (for footer row in template)
    rm_total_qty = sum(r["available"] for r in rm_summary)
    rm_total_cost = sum(r["cost"] for r in rm_summary)

    today = dt.date.today()
    return templates.TemplateResponse(
        "grn.html",
        {
            "request": request,
            "role": current_role(request),
            "read_only": is_read_only(request),   # <-- lets template disable create buttons
            "grns": grns,
            "prices": rm_price_defaults(),
            "report": report,
            "start": start or "",
            "end": end or "",
            "rm_summary": rm_summary,
            "rm_total_qty": rm_total_qty,
            "rm_total_cost": rm_total_cost,
            "today": today,
            "today_iso": today.isoformat(),
        },
    )


@app.get("/grn/new", response_class=HTMLResponse)
def grn_new(request: Request):
    # Only admin/store can create GRNs
    if not role_allowed(request, {"admin", "store"}):
        return RedirectResponse("/login", status_code=303)

    today = dt.date.today()
    min_date = (today - dt.timedelta(days=4)).isoformat()
    max_date = today.isoformat()
    return templates.TemplateResponse(
        "grn_new.html",
        {
            "request": request,
            "role": current_role(request),
            "read_only": is_read_only(request),
            "rm_types": RM_TYPES,
            "min_date": min_date,
            "max_date": max_date,
            "error_text": "",  # inline error slot
        },
    )

@app.post("/grn/new")
def grn_new_post(
    request: Request,
    date: str = Form(...),
    supplier: str = Form(...),
    rm_type: str = Form(...),
    qty: float = Form(...),
    price: float = Form(...),
    # NEW required fields:
    transporter: str = Form(...),
    vehicle_no: str = Form(...),
    # NEW optional uploads (make sure form has enctype="multipart/form-data")
    invoice_file: UploadFile = File(None),
    ewaybill_file: UploadFile = File(None),
    db: Session = Depends(get_db),
):
    # Guard: only admin/store may post
    if not role_allowed(request, {"admin", "store"}):
        return RedirectResponse("/login", status_code=303)

    today = dt.date.today()
    d = dt.date.fromisoformat(date)

    # Same rule as before: today to 4 days back; no future
    if d > today or d < (today - dt.timedelta(days=4)):
        min_date = (today - dt.timedelta(days=4)).isoformat()
        max_date = today.isoformat()
        return templates.TemplateResponse(
            "grn_new.html",
            {
                "request": request,
                "role": current_role(request),
                "read_only": is_read_only(request),
                "rm_types": RM_TYPES,
                "min_date": min_date,
                "max_date": max_date,
                "error_text": "Date must be today or within the last 4 days.",
            },
            status_code=400,
        )

    # Save files (helper returns saved filename or None)
    inv_name = _save_upload(invoice_file)
    ewb_name = _save_upload(ewaybill_file)

    # --- business logic unchanged; just store new fields & filenames ---
    g = GRN(
        grn_no=next_grn_no(db, d),
        date=d,
        supplier=supplier,
        rm_type=rm_type,
        qty=qty,
        remaining_qty=qty,
        price=price,
        transporter=transporter,    # NEW
        vehicle_no=vehicle_no,      # NEW
        invoice_file=inv_name,      # NEW (can be None)
        ewaybill_file=ewb_name,     # NEW (can be None)
    )
    db.add(g)
    db.commit()
    return RedirectResponse("/grn", status_code=303)

# ---------- CSV export for report range (unchanged) ----------
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
        headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'}
    )


# -------------------------------------------------
# Stock helpers (FIFO)  (unchanged)
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
    if not role_allowed(request, {"admin", "melting"}):
        return RedirectResponse("/login", status_code=303)

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
            "role": current_role(request),
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

    ro = not role_allowed(request, {"admin", "qa"})

    return templates.TemplateResponse(
        "qa_heat.html",
        {
            "request": request,
            "read_only": ro,
            "role": current_role(request),
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
    heat_id: int,
    C: str = Form(""), Si: str = Form(""), S: str = Form(""), P: str = Form(""),
    Cu: str = Form(""), Ni: str = Form(""), Mn: str = Form(""), Fe: str = Form(""),
    decision: str = Form("APPROVED"), remarks: str = Form(""),
    db: Session = Depends(get_db),
):
    heat = db.get(Heat, heat_id)
    if not heat:
        return PlainTextResponse("Heat not found", status_code=404)
    
    if not role_allowed(request, {"admin", "qa"}):
        return PlainTextResponse("Forbidden", status_code=403)


    # strict numeric, non-blank, > 0
    fields = {"C": C, "Si": Si, "S": S, "P": P, "Cu": Cu, "Ni": Ni, "Mn": Mn, "Fe": Fe}
    parsed: Dict[str, float] = {}
    for k, v in fields.items():
        s = (v or "").strip()
        try:
            val = float(s)
        except Exception:
            return PlainTextResponse(f"{k} must be a number > 0.", status_code=400)
        if val <= 0:
            return PlainTextResponse(f"{k} must be > 0.", status_code=400)
        parsed[k] = val

    chem = heat.chemistry or HeatChem(heat=heat)
    chem.c  = f"{parsed['C']:.4f}";   chem.si = f"{parsed['Si']:.4f}"
    chem.s  = f"{parsed['S']:.4f}";   chem.p  = f"{parsed['P']:.4f}"
    chem.cu = f"{parsed['Cu']:.4f}";  chem.ni = f"{parsed['Ni']:.4f}"
    chem.mn = f"{parsed['Mn']:.4f}";  chem.fe = f"{parsed['Fe']:.4f}"

    heat.qa_status = (decision or "APPROVED").upper()
    heat.qa_remarks = remarks or ""

    db.add_all([chem, heat]); db.commit()
    return RedirectResponse("/qa-dashboard", status_code=303)


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
    if not role_allowed(request, {"admin", "atom"}):
        return RedirectResponse("/login", status_code=303)

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

    from types import SimpleNamespace

    today = dt.date.today()
    month_start = today.replace(day=1)
    month_end = (month_start + dt.timedelta(days=32)).replace(day=1)

    try:
        atom_bal = _get_atomization_balance(db, month_start, month_end)
        # add a convenience field for % conversion if you want it:
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
            "atom_bal": atom_bal,
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

# ---------- Anneal QA helper (robust date filtering + safe oxygen cast) ----------
from sqlalchemy import text

def _anneal_rows_in_range(db, start_date, end_date):
    # Figure out which FK column exists in anneal_qa: anneal_lot_id or anneal_id
    qa_fk = db.execute(text("""
        SELECT CASE
          WHEN EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema='public' AND table_name='anneal_qa' AND column_name='anneal_lot_id'
          ) THEN 'anneal_lot_id'
          WHEN EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema='public' AND table_name='anneal_qa' AND column_name='anneal_id'
          ) THEN 'anneal_id'
          ELSE NULL
        END
    """)).scalar()

    if not qa_fk:
        # Fail early with a clear message instead of a cryptic f405
        raise RuntimeError("anneal_qa must have anneal_lot_id or anneal_id")

    # If the table uses anneal_id, alias it out as anneal_lot_id so the rest of the code can stay the same
    latest_sql = f"""
        WITH latest AS (
            SELECT DISTINCT ON ({qa_fk})
                   id,
                   {qa_fk} AS anneal_lot_id,
                   decision,
                   oxygen,
                   remarks
            FROM anneal_qa
            ORDER BY {qa_fk}, id DESC
        )
        SELECT
            al.id,
            al.lot_no,
            al.grade,
            COALESCE(al.weight_kg, 0)::double precision      AS weight_kg,
            (NULLIF(al.date::text, ''))::date                AS lot_date,
            COALESCE(lat.decision, 'PENDING')                AS qa_status,
            COALESCE(NULLIF(lat.oxygen::text,'')::float8,0)  AS oxygen,
            COALESCE(lat.remarks, '')                        AS remarks
        FROM anneal_lots al
        LEFT JOIN latest lat
               ON lat.anneal_lot_id = al.id
        WHERE (NULLIF(al.date::text, ''))::date BETWEEN :s AND :e
        ORDER BY lot_date DESC, al.id DESC
    """

    rows = db.execute(text(latest_sql), {"s": start_date, "e": end_date}).mappings().all()
    return [{
        "id": r["id"],
        "lot_no": r["lot_no"],
        "grade": r["grade"],
        "lot_date": r["lot_date"],
        "weight_kg": float(r["weight_kg"] or 0.0),
        "qa_status": r["qa_status"] or "PENDING",
        "oxygen": float(r["oxygen"] or 0.0),
        "remarks": r["remarks"] or "",
    } for r in rows]
    
def _anneal_latest_params_map(db, anneal_ids: list[int]) -> dict[int, dict[str, str]]:
    """
    For a list of anneal lot IDs, returns the param snapshot of the
    latest anneal_qa for each lot, as {anneal_lot_id: {name: value}}.
    No LATERAL; uses a max(id) per anneal_lot_id subquery.
    """
    if not anneal_ids:
        return {}

    rows = db.execute(text("""
        WITH mx AS (
            SELECT anneal_lot_id, MAX(id) AS max_id
            FROM anneal_qa
            WHERE anneal_lot_id = ANY(:ids)
            GROUP BY anneal_lot_id
        )
        SELECT
            mx.anneal_lot_id,
            p.param_name,
            p.param_value
        FROM mx
        JOIN anneal_qa aq
          ON aq.anneal_lot_id = mx.anneal_lot_id
         AND aq.id = mx.max_id
        LEFT JOIN anneal_qa_params p
          ON p.anneal_qa_id = aq.id
        ORDER BY mx.anneal_lot_id, p.id
    """), {"ids": anneal_ids}).mappings().all()

    out: dict[int, dict[str, str]] = {}
    for r in rows:
        lot_id = int(r["anneal_lot_id"])
        nm = str(r["param_name"] or "")
        val = "" if r["param_value"] is None else str(r["param_value"])
        if lot_id not in out:
            out[lot_id] = {}
        if nm:
            out[lot_id][nm] = val
    return out

# --- Compatibility shims so older calls still work ---
# Your real helper is `_grind_rows_in_range(start, end)`. The dashboard is calling
# `_grinding_rows_in_range(db, start, end)`. Make it an alias with the same signature.
def _grinding_rows_in_range(db, start, end):
    # `db` is unused on purpose (to match the caller's signature)
    return _grind_rows_in_range(start, end)

# If your CSV/export or other code ever calls `_grinding_latest_params_map(db, ids)`,
# point it at the actual `_grind_latest_params_map(ids)` if you have it.
def _grinding_latest_params_map(db, grind_ids):
    try:
        return _grind_latest_params_map(grind_ids)
    except NameError:
        # If you don't have _grind_latest_params_map yet, return empty so it still works.
        return {}

# ---------- Grinding helpers used by QA dashboard / export ----------
from sqlalchemy import text
from datetime import date as _date

def _grind_rows_in_range(start_d: _date, end_d: _date):
    """Return simple rows for dashboard tables (portable SQL: works on SQLite/PG)."""
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT id, date, lot_no, grade, COALESCE(weight_kg,0) AS weight_kg,
                   COALESCE(qa_status,'') AS qa_status
            FROM grinding_lots
            WHERE date >= :s AND date <= :e
            ORDER BY date, id
        """), {"s": start_d, "e": end_d}).mappings().all()
        return [dict(r) for r in rows]

def _grind_all_rows():
    """All grinding rows (for queue counts when 'all' is requested)."""
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT id, date, lot_no, grade, COALESCE(weight_kg,0) AS weight_kg,
                   COALESCE(qa_status,'') AS qa_status
            FROM grinding_lots
            ORDER BY id DESC
        """)).mappings().all()
        return [dict(r) for r in rows]

def _grind_latest_qa_header_map(lot_ids: list[int]):
    """
    Get latest Grinding QA header (oxygen, compressibility, decision) per grinding lot.
    Portable two-step (no DISTINCT ON / ANY required).
    """
    out = {}
    if not lot_ids:
        return out
    with engine.begin() as conn:
        for gid in lot_ids:
            row = conn.execute(text("""
                SELECT id, grinding_lot_id, decision, oxygen, compressibility, remarks
                FROM grinding_qa
                WHERE grinding_lot_id = :gid
                ORDER BY id DESC
                LIMIT 1
            """), {"gid": gid}).mappings().first()
            if row:
                out[gid] = dict(row)
    return out

def _fg_rows_in_range(db, s_date, e_date):
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT id, lot_no, date, family, fg_grade, weight_kg, qa_status
            FROM fg_lots
            WHERE date BETWEEN :s AND :e
            ORDER BY date DESC, id DESC
        """), {"s": s_date, "e": e_date}).mappings().all()
        return [dict(r) for r in rows]

# -------------------------------------------------
# FG Helpers (non-destructive)
# -------------------------------------------------
def _fg_rows_in_range(db, s_date, e_date):
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT id, lot_no, date, family, fg_grade, weight_kg, qa_status
            FROM fg_lots
            WHERE date BETWEEN :s AND :e
            ORDER BY date DESC, id DESC
        """), {"s": s_date, "e": e_date}).mappings().all()
    return [dict(r) for r in rows]

def _fg_latest_params_map(db, fg_ids: list[int]):
    if not fg_ids: return {}
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT fgp.fg_qa_id, fgp.param_name, fgp.param_value, fq.fg_lot_id
            FROM fg_qa_params fgp
            JOIN fg_qa fq ON fq.id = fgp.fg_qa_id
            WHERE fq.fg_lot_id = ANY(:ids)
        """), {"ids": fg_ids}).mappings().all()
    out: dict[int, dict[str, Any]] = {}
    for r in rows:
        out.setdefault(r["fg_lot_id"], {})[r["param_name"]] = r["param_value"]
    return out

# -------------------------------------------------
# QA Dashboard (Expanded, includes Grinding + FG)
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
    show_all = (request.query_params.get("all") == "1")

    # ---------------- Date Range Setup ----------------
    s_iso = start or today.isoformat()
    e_iso = end or today.isoformat()
    try:
        s_date = dt.date.fromisoformat(s_iso)
        e_date = dt.date.fromisoformat(e_iso)
    except Exception:
        s_date = e_date = today
        s_iso = e_iso = today.isoformat()

    # ---------------- Data Loading ----------------
    heats_all = db.query(Heat).order_by(Heat.id.desc()).all()
    lots_all  = db.query(Lot).order_by(Lot.id.desc()).all()
    grinds_all = _grinding_rows_in_range(db, dt.date(1970, 1, 1), dt.date(2100, 1, 1))
    fgs_all    = _fg_rows_in_range(db, dt.date(1970, 1, 1), dt.date(2100, 1, 1))

    def _hd(h: Heat) -> dt.date:
        return heat_date_from_no(h.heat_no) or today
    def _ld(l: Lot) -> dt.date:
        return lot_date_from_no(l.lot_no) or today
    def _gd(g: dict) -> dt.date:
        return g.get("date") or today
    def _fd(f: dict) -> dt.date:
        return f.get("date") or today

    # ---------------- Visible Rows ----------------
    if show_all:
        heats_vis = heats_all
        lots_vis  = lots_all
        anneals_vis = _anneal_rows_in_range(db, dt.date(1900,1,1), dt.date(3000,1,1))
        grinds_vis  = _grinding_rows_in_range(db, dt.date(1900,1,1), dt.date(3000,1,1))
        fgs_vis     = _fg_rows_in_range(db, dt.date(1900,1,1), dt.date(3000,1,1))
    else:
        heats_vis = [h for h in heats_all if s_date <= _hd(h) <= e_date]
        lots_vis  = [l for l in lots_all if s_date <= _ld(l) <= e_date]
        anneals_vis = _anneal_rows_in_range(db, s_date, e_date)
        grinds_vis  = _grinding_rows_in_range(db, s_date, e_date)
        fgs_vis     = _fg_rows_in_range(db, s_date, e_date)

    # ---------------- Monthly KPIs ----------------
    month_start = e_date.replace(day=1)
    month_end = (month_start + dt.timedelta(days=32)).replace(day=1) - dt.timedelta(days=1)
    lots_this_month = [l for l in lots_all if month_start <= _ld(l) <= month_end]
    grinds_this_month = _grinding_rows_in_range(db, month_start, month_end)
    fgs_this_month    = _fg_rows_in_range(db, month_start, month_end)

    def _sum_lots(status: str) -> float:
        s = status.upper()
        return sum(float(l.weight or 0.0) for l in lots_this_month if (l.qa_status or "").upper() == s)
    def _sum_grinds(status: str) -> float:
        s = status.upper()
        return sum(float(g["weight_kg"] or 0.0)
                   for g in grinds_this_month
                   if (g["qa_status"] or "").upper() == s)
    def _sum_fgs(status: str) -> float:
        s = status.upper()
        return sum(float(f["weight_kg"] or 0.0)
                   for f in fgs_this_month
                   if (f["qa_status"] or "").upper() == s)
    def _sum_heats(status: str) -> float:
        s = status.upper()
        return sum(float(h.actual_output or 0.0)
                   for h in heats_all
                   if month_start <= (_hd(h) or today) <= month_end and (h.qa_status or "").upper() == s)
    def _sum_anneals(status: str) -> float:
        s = status.upper()
        return sum(float(a["weight_kg"] or 0.0)
                   for a in _anneal_rows_in_range(db, month_start, month_end)
                   if (a["qa_status"] or "").upper() == s)

    # ---------------- KPI Buckets ----------------
    kpi = {
        "approved_kg": _sum_lots("APPROVED"),
        "hold_kg": _sum_lots("HOLD"),
        "rejected_kg": _sum_lots("REJECTED"),
    }
    kpi_heats = {
        "approved": _sum_heats("APPROVED"),
        "hold": _sum_heats("HOLD"),
        "rejected": _sum_heats("REJECTED"),
    }
    kpi_lots = {
        "approved": float(kpi.get("approved_kg", 0.0)),
        "hold": float(kpi.get("hold_kg", 0.0)),
        "rejected": float(kpi.get("rejected_kg", 0.0)),
    }
    kpi_anneal = {
        "approved": _sum_anneals("APPROVED"),
        "hold": _sum_anneals("HOLD"),
        "rejected": _sum_anneals("REJECTED"),
    }
    kpi_grind = {
        "approved": _sum_grinds("APPROVED"),
        "hold": _sum_grinds("HOLD"),
        "rejected": _sum_grinds("REJECTED"),
    }
    kpi_fg = {
        "approved": _sum_fgs("APPROVED"),
        "hold": _sum_fgs("HOLD"),
        "rejected": _sum_fgs("REJECTED"),
    }

    # ---------------- QA Queue Counts ----------------
    anneals_all   = _anneal_rows_in_range(db, dt.date(1970,1,1), dt.date(2100,1,1))
    anneals_today = _anneal_rows_in_range(db, today, today)
    grinds_today  = _grinding_rows_in_range(db, today, today)
    fgs_today     = _fg_rows_in_range(db, today, today)

    pending_heats  = sum(1 for h in heats_all if (h.qa_status or "").upper() == "PENDING")
    pending_lots   = sum(1 for l in lots_all if (l.qa_status or "").upper() == "PENDING")
    pending_anneal = sum(1 for a in anneals_all if (a["qa_status"] or "").upper() == "PENDING")
    pending_grind  = sum(1 for g in grinds_all if (g["qa_status"] or "").upper() == "PENDING")
    pending_fg     = sum(1 for f in fgs_all if (f["qa_status"] or "").upper() == "PENDING")

    today_heats  = sum(1 for h in heats_all if _hd(h) == today and (h.qa_status or "").upper() != "PENDING")
    today_lots   = sum(1 for l in lots_all if _ld(l) == today and (l.qa_status or "").upper() != "PENDING")
    today_anneal = sum(1 for a in anneals_today if (a["qa_status"] or "").upper() != "PENDING")
    today_grind  = sum(1 for g in grinds_today if (g["qa_status"] or "").upper() != "PENDING")
    today_fg     = sum(1 for f in fgs_today if (f["qa_status"] or "").upper() != "PENDING")

    pending_count = pending_heats + pending_lots + pending_anneal + pending_grind + pending_fg
    todays_count  = today_heats + today_lots + today_anneal + today_grind + today_fg

    heat_grades = {h.id: heat_grade(h) for h in heats_vis}

    queue_pending = {
        "heats": int(pending_heats),
        "lots": int(pending_lots),
        "anneal": int(pending_anneal),
        "grind": int(pending_grind),
        "fg": int(pending_fg),
    }
    queue_today = {
        "heats": int(today_heats),
        "lots": int(today_lots),
        "anneal": int(today_anneal),
        "grind": int(today_grind),
        "fg": int(today_fg),
    }

    queue_pending["total"] = sum(queue_pending.values())
    queue_today["total"]   = sum(queue_today.values())

    return templates.TemplateResponse(
        "qa_dashboard.html",
        {
            "request": request,
            "role": current_role(request),
            "heats": heats_vis,
            "lots": lots_vis,
            "heat_grades": heat_grades,
            "anneals": anneals_vis,
            "grinds": grinds_vis,
            "fgs": fgs_vis,
            "kpi_approved_month": float(kpi.get("approved_kg", 0.0)),
            "kpi_hold_month": float(kpi.get("hold_kg", 0.0)),
            "kpi_rejected_month": float(kpi.get("rejected_kg", 0.0)),
            "kpi_pending_count": int(pending_count),
            "kpi_today_count": int(todays_count),
            "kpi_heats": kpi_heats,
            "kpi_lots": kpi_lots,
            "kpi_anneal": kpi_anneal,
            "kpi_grind": kpi_grind,
            "kpi_fg": kpi_fg,
            "queue_pending": queue_pending,
            "queue_today": queue_today,
            "start": "" if show_all else s_iso,
            "end": "" if show_all else e_iso,
            "today_iso": today.isoformat(),
        },
    )


# -------------------------------------------------
# QA Export (Expanded, includes Grinding + FG)
# -------------------------------------------------
@app.get("/qa/export")
def qa_export(
    start: Optional[str] = None,
    end: Optional[str] = None,
    db: Session = Depends(get_db),
):
    heats = db.query(Heat).order_by(Heat.id.asc()).all()
    lots  = db.query(Lot).order_by(Lot.id.asc()).all()
    today = dt.date.today()
    s = dt.date.fromisoformat(start) if start else today
    e = dt.date.fromisoformat(end)   if end   else today

    heats_in = [h for h in heats if s <= (heat_date_from_no(h.heat_no) or today) <= e]
    lots_in  = [l for l in lots if s <= (lot_date_from_no(l.lot_no) or today) <= e]
    anneals_in = _anneal_rows_in_range(db, s, e)
    grinds_in  = _grinding_rows_in_range(db, s, e)
    fgs_in     = _fg_rows_in_range(db, s, e)

    anneal_ids = [a["id"] for a in anneals_in]
    grind_ids  = [g["id"] for g in grinds_in]
    fg_ids     = [f["id"] for f in fgs_in]

    anneal_params = _anneal_latest_params_map(db, anneal_ids)
    grind_params  = _grind_latest_params_map(db, grind_ids)
    fg_params     = _fg_latest_params_map(db, fg_ids)

    out = io.StringIO()
    w = out.write
    w("Type,ID,Date,Grade/Type,Weight/Output (kg),QA Status,Oxygen,Compressibility,C,Si,S,P,Cu,Ni,Mn,Fe\n")

    for h in heats_in:
        w(
            f"HEAT,{h.heat_no},{(heat_date_from_no(h.heat_no) or today).isoformat()},"
            f"{heat_grade(h) or ''},{float(h.actual_output or 0.0):.1f},{h.qa_status or ''},,,,,,,,\n"
        )

    for l in lots_in:
        w(
            f"LOT,{l.lot_no},{(lot_date_from_no(l.lot_no) or today).isoformat()},{l.grade or ''},"
            f"{float(l.weight or 0.0):.1f},{l.qa_status or ''},,,,,,,,\n"
        )

    for a in anneals_in:
        p = anneal_params.get(a["id"], {})
        oxygen = f"{float(a.get('oxygen', 0) or 0):.3f}" if a.get("oxygen") else ""
        w(
            f"ANNEAL,{a['lot_no']},{(a['lot_date'] or today).isoformat()},{a.get('grade','')},"
            f"{float(a['weight_kg'] or 0.0):.1f},{a.get('qa_status','')},{oxygen},,,"
            f"{p.get('C','')},{p.get('Si','')},{p.get('S','')},{p.get('P','')},{p.get('Cu','')},{p.get('Ni','')},{p.get('Mn','')},{p.get('Fe','')}\n"
        )

    for g in grinds_in:
        p = grind_params.get(g["id"], {})
        oxygen = f"{float(g.get('oxygen', 0) or 0):.3f}" if g.get("oxygen") else ""
        comp = f"{float(g.get('compressibility', 0) or 0):.2f}" if g.get("compressibility") else ""
        w(
            f"GRIND,{g['lot_no']},{(g.get('date') or today).isoformat()},{g.get('grade','')},"
            f"{float(g['weight_kg'] or 0.0):.1f},{g.get('qa_status','')},{oxygen},{comp},"
            f"{p.get('C','')},{p.get('Si','')},{p.get('S','')},{p.get('P','')},{p.get('Cu','')},{p.get('Ni','')},{p.get('Mn','')},{p.get('Fe','')}\n"
        )

    for f in fgs_in:
        p = fg_params.get(f["id"], {})
        w(
            f"FG,{f['lot_no']},{(f.get('date') or today).isoformat()},{f.get('fg_grade','')},"
            f"{float(f['weight_kg'] or 0.0):.1f},{f.get('qa_status','')},,,"
            f"{p.get('C','')},{p.get('Si','')},{p.get('S','')},{p.get('P','')},{p.get('Cu','')},{p.get('Ni','')},{p.get('Mn','')},{p.get('Fe','')}\n"
        )

    data = out.getvalue().encode("utf-8")
    fname = f"qa_export_{s.isoformat()}_{e.isoformat()}.csv"
    return StreamingResponse(
        io.BytesIO(data),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename=\"{fname}\"'}
    )


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
    if not role_allowed(request, {"admin", "rap"}):
        return RedirectResponse("/login", status_code=303)
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
            "role": current_role(request),
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
    writer.writerow([
        "Date", "Lot", "Grade", "Qty (kg)", "Unit Cost (₹/kg)", "Value (₹)", "Remarks", "Alloc ID"
    ])

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
            lot.lot_no or "",
            lot.grade or "",
            f"{qty:.1f}",
            f"{unit:.2f}",
            f"{val:.2f}",
            (alloc.remarks or "").replace(",", " "),
            alloc.id,
        ])

    buf.seek(0)
    return StreamingResponse(
        io.StringIO(buf.getvalue()),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="plant2_transfers.csv"'},)

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

@app.get("/qa/lot/{lot_id}", response_class=HTMLResponse)
def qa_lot_form(lot_id: int, request: Request, db: Session = Depends(get_db)):
    """
    Editable Lot QA form for QA/Admin; everyone else read-only.
    Prefills chemistry using 60% rule else weighted average.
    """
    lot = db.get(Lot, lot_id)
    if not lot:
        return PlainTextResponse("Lot not found", status_code=404)

    ro = not role_allowed(request, {"admin", "qa"})  # read-only for non QA/Admin

    # existing or default chemistry
    if lot.chemistry:
        chem_defaults = {}
        for k in ["c","si","s","p","cu","ni","mn","fe"]:
            try:
                chem_defaults[k] = float(getattr(lot.chemistry, k) or "")
            except Exception:
                chem_defaults[k] = None
    else:
        chem_defaults = _lot_default_chemistry(db, lot)

    # ensure phys/psd objects exist for form rendering
    phys = lot.phys or LotPhys(lot=lot, ad="", flow="")
    psd  = lot.psd  or LotPSD(lot=lot, p212="", p180="", n180p150="", n150p75="", n75p45="", n45="")

    return templates.TemplateResponse(
        "qa_lot.html",
        {
            "request": request,
            "read_only": ro,
            "role": current_role(request),
            "lot": lot,
            "grade": (lot.grade or "KRIP"),
            "chem": {
                "C":  "" if chem_defaults["c"]  is None else f"{chem_defaults['c']:.4f}",
                "Si": "" if chem_defaults["si"] is None else f"{chem_defaults['si']:.4f}",
                "S":  "" if chem_defaults["s"]  is None else f"{chem_defaults['s']:.4f}",
                "P":  "" if chem_defaults["p"]  is None else f"{chem_defaults['p']:.4f}",
                "Cu": "" if chem_defaults["cu"] is None else f"{chem_defaults['cu']:.4f}",
                "Ni": "" if chem_defaults["ni"] is None else f"{chem_defaults['ni']:.4f}",
                "Mn": "" if chem_defaults["mn"] is None else f"{chem_defaults['mn']:.4f}",
                "Fe": "" if chem_defaults["fe"] is None else f"{chem_defaults['fe']:.4f}",
            },
            "phys": {"ad": phys.ad or "", "flow": phys.flow or ""},
            "psd": {
                "p212": psd.p212 or "", "p180": psd.p180 or "",
                "n180p150": psd.n180p150 or "", "n150p75": psd.n150p75 or "",
                "n75p45": psd.n75p45 or "", "n45": psd.n45 or "",
            },
        },
    )

@app.post("/qa/lot/{lot_id}")
def qa_lot_save(
    lot_id: int,
    # chemistry
    C: str = Form(""), Si: str = Form(""), S: str = Form(""), P: str = Form(""),
    Cu: str = Form(""), Ni: str = Form(""), Mn: str = Form(""), Fe: str = Form(""),
    # physical
    ad: str = Form(""), flow: str = Form(""),
    # psd
    p212: str = Form(""), p180: str = Form(""), n180p150: str = Form(""),
    n150p75: str = Form(""), n75p45: str = Form(""), n45: str = Form(""),
    # decision
    decision: str = Form("APPROVED"), remarks: str = Form(""),
    request: Request = None,
    db: Session = Depends(get_db),
):
    # access gate: only admin/qa can save
    if not role_allowed(request, {"admin", "qa"}):
        return PlainTextResponse("Forbidden", status_code=403)

    lot = db.get(Lot, lot_id)
    if not lot:
        return PlainTextResponse("Lot not found", status_code=404)

    # strict numeric validations (non-blank, numeric, > 0) for ALL fields
    def _req_pos_float(name: str, val: str) -> float:
        s = (val or "").strip()
        try:
            f = float(s)
        except Exception:
            raise ValueError(f"{name} must be a number > 0.")
        if f <= 0:
            raise ValueError(f"{name} must be > 0.")
        return f

    try:
        chem_fields = {
            "C": _req_pos_float("C", C), "Si": _req_pos_float("Si", Si),
            "S": _req_pos_float("S", S), "P": _req_pos_float("P", P),
            "Cu": _req_pos_float("Cu", Cu), "Ni": _req_pos_float("Ni", Ni),
            "Mn": _req_pos_float("Mn", Mn), "Fe": _req_pos_float("Fe", Fe),
        }
        phys_ad   = _req_pos_float("AD", ad)
        phys_flow = _req_pos_float("Flow", flow)
        psd_fields = {
            "p212": _req_pos_float("+212", p212),
            "p180": _req_pos_float("+180", p180),
            "n180p150": _req_pos_float("-180+150", n180p150),
            "n150p75":  _req_pos_float("-150+75", n150p75),
            "n75p45":   _req_pos_float("-75+45", n75p45),
            "n45":      _req_pos_float("-45", n45),
        }
    except ValueError as ve:
        return PlainTextResponse(str(ve), status_code=400)

    # upsert LotChem / LotPhys / LotPSD
    lchem = lot.chemistry or LotChem(lot=lot)
    lchem.c  = f"{chem_fields['C']:.4f}";   lchem.si = f"{chem_fields['Si']:.4f}"
    lchem.s  = f"{chem_fields['S']:.4f}";   lchem.p  = f"{chem_fields['P']:.4f}"
    lchem.cu = f"{chem_fields['Cu']:.4f}";  lchem.ni = f"{chem_fields['Ni']:.4f}"
    lchem.mn = f"{chem_fields['Mn']:.4f}";  lchem.fe = f"{chem_fields['Fe']:.4f}"

    lphys = lot.phys or LotPhys(lot=lot)
    lphys.ad = f"{phys_ad:.4f}"
    lphys.flow = f"{phys_flow:.4f}"

    lpsd = lot.psd or LotPSD(lot=lot)
    lpsd.p212 = f"{psd_fields['p212']:.4f}"
    lpsd.p180 = f"{psd_fields['p180']:.4f}"
    lpsd.n180p150 = f"{psd_fields['n180p150']:.4f}"
    lpsd.n150p75  = f"{psd_fields['n150p75']:.4f}"
    lpsd.n75p45   = f"{psd_fields['n75p45']:.4f}"
    lpsd.n45      = f"{psd_fields['n45']:.4f}"

    lot.qa_status  = (decision or "APPROVED").upper()
    lot.qa_remarks = remarks or ""

    db.add_all([lchem, lphys, lpsd, lot]); db.commit()
    return RedirectResponse("/qa-dashboard", status_code=303)


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
