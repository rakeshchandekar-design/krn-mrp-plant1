from functools import wraps
from flask import request, abort
from krn_mrp_app.main import role_allowed  # uses your existing helper

def require_roles(*roles):
    allowed = set(roles)
    def _dec(fn):
        @wraps(fn)
        def _wrap(*a, **k):
            if not role_allowed(request, allowed):
                abort(403)
            return fn(*a, **k)
        return _wrap
    return _dec
