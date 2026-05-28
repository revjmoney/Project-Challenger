"""
Project Challenger — Web GUI authentication
=============================================
Single-user password auth with PBKDF2-HMAC-SHA256 hashing and
in-memory session tokens.

Credentials are stored in  data/auth.json  (gitignored — never committed).
Sessions are kept in memory; they are cleared on server restart,
which requires the user to log in again.

Typical flow
------------
  First run (no auth.json):
    1. GET  /api/auth/status  →  {"configured": false}
    2. POST /api/auth/setup   →  {"status": "ok"}  + Set-Cookie: auth_token=…
    3. Browser hides the setup overlay and loads the main UI.

  Subsequent runs (auth.json exists):
    1. GET  /api/auth/status  →  {"configured": true}
    2. POST /api/auth/login   →  {"status": "ok"}  + Set-Cookie: auth_token=…
    3. POST /api/auth/logout  →  clears cookie + destroys session.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from pathlib import Path

_ROOT     = Path(__file__).parent
AUTH_FILE = _ROOT / "data" / "auth.json"


# ── Credential storage ────────────────────────────────────────────────────────

def has_credentials() -> bool:
    """Return True if a username/password has been configured."""
    return AUTH_FILE.exists()


def _hash(password: str, salt: str) -> str:
    """Derive a hex key from password + salt using PBKDF2-HMAC-SHA256."""
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260_000)
    return dk.hex()


def save_credentials(username: str, password: str) -> None:
    """
    Hash and persist credentials to data/auth.json.
    Raises ValueError if username is blank or password is too short.
    Calling this a second time *replaces* existing credentials.
    """
    username = username.strip()
    if not username:
        raise ValueError("Username cannot be empty.")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")
    salt = secrets.token_hex(32)
    AUTH_FILE.write_text(
        json.dumps({"username": username, "hash": _hash(password, salt), "salt": salt},
                   indent=2),
        encoding="utf-8",
    )


def check_credentials(username: str, password: str) -> bool:
    """
    Return True if username + password match stored credentials.
    Uses a constant-time comparison to prevent timing attacks.
    """
    if not has_credentials():
        return False
    try:
        creds = json.loads(AUTH_FILE.read_text(encoding="utf-8"))
    except Exception:
        return False
    if creds.get("username", "").strip() != username.strip():
        return False
    expected = _hash(password, creds.get("salt", ""))
    return hmac.compare_digest(expected, creds.get("hash", ""))


def get_stored_username() -> str:
    """Return the stored username, or empty string if no credentials exist."""
    if not has_credentials():
        return ""
    try:
        return json.loads(AUTH_FILE.read_text(encoding="utf-8")).get("username", "")
    except Exception:
        return ""


# ── Session store (in-memory) ─────────────────────────────────────────────────

_sessions: dict[str, str] = {}   # token → username


def create_session(username: str) -> str:
    """Create a new cryptographically random session token and return it."""
    token = secrets.token_urlsafe(32)
    _sessions[token] = username
    return token


def validate_session(token: str) -> bool:
    """Return True if token is a valid active session."""
    return bool(token) and token in _sessions


def destroy_session(token: str) -> None:
    """Invalidate a session token (logout)."""
    _sessions.pop(token, None)


def get_username(token: str) -> str | None:
    """Return the username bound to a session token, or None if invalid."""
    return _sessions.get(token)


def destroy_all_sessions() -> None:
    """Invalidate every active session (e.g. after a password change)."""
    _sessions.clear()
