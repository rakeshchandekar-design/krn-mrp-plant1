# krn_mrp_app/deps.py
import os
from typing import Callable, List
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import create_engine


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _normalize_db_url(url: str) -> str:
    """
    Normalize DB URL for SQLAlchemy:
    - Convert 'postgres://' → 'postgresql://'
    - Ensure psycopg v3 driver: 'postgresql+psycopg://'
    """
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if url.startswith("postgresql://") and "+psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise Exception("DATABASE_URL is not set. Cannot start app.")

DATABASE_URL = _normalize_db_url(DATABASE_URL)

engine = create_engine(
    DATABASE_URL,
    connect_args={},
    pool_pre_ping=True,
    pool_recycle=_env_int("DB_POOL_RECYCLE", 1800),
    pool_size=_env_int("DB_POOL_SIZE", 10),
    max_overflow=_env_int("DB_MAX_OVERFLOW", 20),
    pool_timeout=_env_int("DB_POOL_TIMEOUT", 30),
    pool_use_lifo=True,
    query_cache_size=_env_int("DB_QUERY_CACHE_SIZE", 1200),
    future=True,
)

# ---------- Role helper (FastAPI dependency style) ----------
def get_current_user(request: Request):
    """Return the user dict stored in session by your login flow, or None."""
    return request.session.get("user")

def require_roles(*roles: List[str]) -> Callable:
    def wrapper(user=Depends(get_current_user)):
        # accept both dict {"username","role"} and legacy "role" string
        role = None
        if isinstance(user, dict):
            role = user.get("role")
        elif isinstance(user, str):
            role = user  # legacy sessions stored just the role string

        if not role or role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden",
            )
        return user
    return wrapper
