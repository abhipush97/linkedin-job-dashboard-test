from __future__ import annotations

import hmac
import os
import secrets
import time
from dataclasses import dataclass
from threading import Lock

DEFAULT_LOGIN_USERNAME = "recruiter"
DEFAULT_LOGIN_PASSWORD = "linkedin123"
DEFAULT_TOKEN_TTL_SECONDS = 12 * 60 * 60


@dataclass(frozen=True)
class SessionInfo:
    username: str
    expires_at: float


_SESSIONS: dict[str, SessionInfo] = {}
_SESSIONS_LOCK = Lock()


def _resolve_auth_value(key: str, fallback: str) -> str:
    value = os.getenv(key, "").strip()
    return value or fallback


def get_configured_credentials() -> tuple[str, str]:
    username = _resolve_auth_value("APP_LOGIN_USERNAME", DEFAULT_LOGIN_USERNAME)
    password = _resolve_auth_value("APP_LOGIN_PASSWORD", DEFAULT_LOGIN_PASSWORD)
    return username, password


def get_token_ttl_seconds() -> int:
    raw = os.getenv("APP_TOKEN_TTL_SECONDS", str(DEFAULT_TOKEN_TTL_SECONDS)).strip()
    try:
        parsed = int(raw)
    except ValueError:
        return DEFAULT_TOKEN_TTL_SECONDS
    return max(parsed, 300)


def validate_credentials(username: str, password: str) -> bool:
    expected_username, expected_password = get_configured_credentials()
    return (
        hmac.compare_digest(username.strip(), expected_username)
        and hmac.compare_digest(password, expected_password)
    )


def _clear_expired_sessions_locked(now_ts: float) -> None:
    expired = [token for token, session in _SESSIONS.items() if session.expires_at <= now_ts]
    for token in expired:
        _SESSIONS.pop(token, None)


def create_session(username: str) -> tuple[str, int]:
    token = secrets.token_urlsafe(32)
    ttl_seconds = get_token_ttl_seconds()
    expiry = time.time() + ttl_seconds

    with _SESSIONS_LOCK:
        _clear_expired_sessions_locked(time.time())
        _SESSIONS[token] = SessionInfo(username=username.strip(), expires_at=expiry)

    return token, ttl_seconds


def validate_session(token: str) -> SessionInfo | None:
    cleaned = token.strip()
    if not cleaned:
        return None

    now_ts = time.time()
    with _SESSIONS_LOCK:
        _clear_expired_sessions_locked(now_ts)
        session = _SESSIONS.get(cleaned)
        if not session:
            return None
        return session


def destroy_session(token: str) -> None:
    with _SESSIONS_LOCK:
        _SESSIONS.pop(token.strip(), None)
