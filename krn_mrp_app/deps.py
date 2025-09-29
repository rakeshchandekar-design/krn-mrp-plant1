# krn_mrp_app/deps.py
from sqlalchemy import create_engine
from fastapi import Depends, HTTPException, Request, status
from typing import Callable, List

# ⚙️ Adjust DATABASE_URL to how you already set it in main.py
DATABASE_URL = "postgresql+psycopg2://..."  # replace with your existing URL
engine = create_engine(DATABASE_URL, future=True)


# ---- Role system ----
USER_DB = {
    "admin": {"password": "admin123", "role": "admin"},
    "anneal": {"password": "anneal123", "role": "anneal"},
    "qa": {"password": "qa123", "role": "qa"},
    "view": {"password": "view123", "role": "view"},
}

def get_current_user(request: Request):
    return request.session.get("user")

def require_roles(*roles: List[str]) -> Callable:
    """Dependency to enforce allowed roles."""
    def wrapper(user=Depends(get_current_user)):
        if not user or user.get("role") not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, 
                detail="Forbidden"
            )
        return user
    return wrapper
