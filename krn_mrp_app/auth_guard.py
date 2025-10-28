# krn_mrp_app/auth_guard.py
import time

ACTIVE_SESSIONS: dict[str, dict] = {}  # username → {sid, last_seen}
IDLE_SECONDS = 15 * 60  # 15 min

def can_login(username: str) -> tuple[bool, str]:
    """Check if this user can log in or already has an active session."""
    now = int(time.time())
    record = ACTIVE_SESSIONS.get(username)
    if not record:
        return True, ""
    last = int(record.get("last_seen") or 0)
    if (now - last) > IDLE_SECONDS:
        ACTIVE_SESSIONS.pop(username, None)
        return True, ""
    return False, "User already logged in on another system."

def register_login(username: str, sid: str):
    ACTIVE_SESSIONS[username] = {"sid": sid, "last_seen": int(time.time())}

def unregister_login(username: str, sid: str):
    cur = ACTIVE_SESSIONS.get(username)
    if cur and cur.get("sid") == sid:
        ACTIVE_SESSIONS.pop(username, None)

def update_heartbeat(username: str, sid: str):
    cur = ACTIVE_SESSIONS.get(username)
    if cur and cur.get("sid") == sid:
        cur["last_seen"] = int(time.time())
