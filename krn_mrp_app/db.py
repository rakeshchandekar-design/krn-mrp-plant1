import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

def _normalize_db_url(url: str) -> str:
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if url.startswith("postgresql://") and "+psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url

DATABASE_URL = os.getenv("DATABASE_URL")

# 🚨 STRICT: No fallback allowed
if not DATABASE_URL:
    raise Exception("DATABASE_URL is not set. App cannot start.")

DATABASE_URL = _normalize_db_url(DATABASE_URL)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
