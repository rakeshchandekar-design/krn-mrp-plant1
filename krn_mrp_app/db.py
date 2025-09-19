import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

def _normalize_db_url(url: str) -> str:
    # Accept Heroku-style postgres:// and fix to SQLAlchemy’s postgresql+psycopg://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if url.startswith("postgresql://") and "+psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    DATABASE_URL = _normalize_db_url(DATABASE_URL)
else:
    DATABASE_URL = "sqlite:///./local.db"

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
