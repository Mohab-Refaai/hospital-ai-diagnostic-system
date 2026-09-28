"""Data layer: user credentials (hashed) + prediction audit log, stored in SQLite.

Passwords are hashed with PBKDF2-HMAC-SHA256 (random salt, 200k iterations) using only the
standard library -- plain-text passwords are never stored (BRD FR-1.3).
Prediction logs deliberately store NO patient data / images: only user, tab, a short result and latency.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.environ.get("HOSPITAL_DB_PATH", os.path.join(ROOT, "data", "hospital.db"))
ITERATIONS = 200_000
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
ROLES = ("staff", "admin")

# Created ONLY when the DB is empty and no [users] are defined in Streamlit secrets.
# Change them (see README) before any public deployment.
DEMO_USERS = {
    "admin": {"password": "Admin@12345", "role": "admin"},
    "staff_ahmed": {"password": "Staff@12345", "role": "staff"},
}


@contextmanager
def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    with _conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS users (
                 username TEXT PRIMARY KEY, password_hash TEXT NOT NULL,
                 role TEXT NOT NULL DEFAULT 'staff', created_at TEXT NOT NULL)"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS predictions (
                 id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, username TEXT NOT NULL,
                 tab TEXT NOT NULL, output TEXT, latency_ms INTEGER)"""
        )


# ------------------------------------------------------------------ passwords
def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return f"pbkdf2_sha256${ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iters, salt_hex, hash_hex = stored.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(iters))
        return hmac.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False


_DUMMY_HASH = hash_password("dummy-password-for-timing")


# ---------------------------------------------------------------------- users
def create_user(username: str, password: str, role: str = "staff") -> None:
    if not USERNAME_RE.match(username or ""):
        raise ValueError("Username must be 3-32 characters: letters, digits, _ . -")
    if len(password or "") < 8:
        raise ValueError("Password must be at least 8 characters.")
    if role not in ROLES:
        raise ValueError(f"Role must be one of {ROLES}.")
    with _conn() as c:
        if c.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
            raise ValueError("This username already exists.")
        c.execute(
            "INSERT INTO users VALUES (?, ?, ?, ?)",
            (username, hash_password(password), role, datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )


def authenticate(username: str, password: str) -> Optional[str]:
    """Return the user's role if the credentials are valid, else None."""
    with _conn() as c:
        row = c.execute("SELECT password_hash, role FROM users WHERE username = ?", (username.strip(),)).fetchone()
    if row is None:
        verify_password(password, _DUMMY_HASH)  # keep timing similar for unknown users
        return None
    return row["role"] if verify_password(password, row["password_hash"]) else None


def seed_users(mapping: Dict[str, dict]) -> None:
    """Create/update users defined in Streamlit secrets: {name: {password, role}}."""
    for name, cfg in mapping.items():
        pw, role = cfg.get("password", ""), cfg.get("role", "staff")
        with _conn() as c:
            row = c.execute("SELECT password_hash FROM users WHERE username = ?", (name,)).fetchone()
            if row is not None:
                if not verify_password(pw, row["password_hash"]):
                    c.execute("UPDATE users SET password_hash = ?, role = ? WHERE username = ?", (hash_password(pw), role, name))
                continue
        create_user(name, pw, role)


def ensure_default_users() -> bool:
    """Seed demo accounts if there are no users at all. Returns True if seeded."""
    with _conn() as c:
        n = c.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
    if n:
        return False
    for name, cfg in DEMO_USERS.items():
        create_user(name, cfg["password"], cfg["role"])
    return True


def uses_default_password(username: str = "admin") -> bool:
    d = DEMO_USERS.get(username)
    return bool(d) and authenticate(username, d["password"]) is not None


def list_users() -> List[dict]:
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT username, role, created_at FROM users ORDER BY username")]


# ----------------------------------------------------------------------- logs
def log_prediction(username: str, tab: str, output: str, latency_s: float) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO predictions (ts, username, tab, output, latency_ms) VALUES (?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), username, tab, output[:200], int(latency_s * 1000)),
        )


def recent_logs(limit: int = 50) -> List[dict]:
    with _conn() as c:
        return [
            dict(r)
            for r in c.execute(
                "SELECT ts, username, tab, output, latency_ms FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
            )
        ]


def usage_summary() -> List[dict]:
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT tab, COUNT(*) AS predictions FROM predictions GROUP BY tab")]
