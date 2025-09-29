# krn_mrp_app/deps.py
from sqlalchemy import create_engine
from fastapi import Depends, HTTPException, Request, status
from typing import Callable, List

# ⚙️ Adjust DATABASE_URL to match your actual connection
DATABASE_URL = "postgresql+psycopg2://..."  
engine = create_engine(DATABASE_URL, future=True)

# ---- Role helper for FastAPI routes ----
def get_current_user(request: Request):
    return request.session.get("user")

def require_roles(*roles: List[str]) -> Callable:
    """FastAPI dependency to enforce allowed roles."""
    def wrapper(user=Depends(get_current_user)):
        if not user or user.get("role") not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden"
            )
        return user
    return wrapper
