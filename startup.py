"""
Project Challenger — startup checks and terminal auth / registration.
Imported by all entry points: web_app.py, controller.py, main.py.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import platform
import sys
import time
import uuid
from getpass import getpass
from pathlib import Path
import urllib.request as _ureq

_ROOT    = Path(__file__).parent
DATA_DIR = _ROOT / "data"


# ── Install integrity ─────────────────────────────────────────────────────────

def _build_session_key() -> str:
    _s = f"{uuid.getnode()}:{platform.node()}:{platform.machine()}"
    return hashlib.sha256(_s.encode()).hexdigest()


def check_install() -> None:
    _cf = DATA_DIR / ".idata"
    DATA_DIR.mkdir(exist_ok=True)
    _k = _build_session_key()
    if _cf.exists():
        if _cf.read_text().strip() != _k:
            print()
            print("  ╔══════════════════════════════════════════════════════════╗")
            print("  ║  PROJECT CHALLENGER — INSTALL ERROR                     ║")
            print("  ╠══════════════════════════════════════════════════════════╣")
            print("  ║  This install cannot be started on this machine.         ║")
            print("  ║                                                          ║")
            print("  ║  Please do a fresh install:                              ║")
            print("  ║    git clone <repo-url>                                  ║")
            print("  ║    cd <repo-folder>                                      ║")
            print("  ║    launcher.bat  (Windows)  or  ./launcher.sh  (Linux)  ║")
            print("  ╚══════════════════════════════════════════════════════════╝")
            print()
            sys.exit(1)
    else:
        _cf.write_text(_k)


# ── Terminal registration ─────────────────────────────────────────────────────

_PING_URL  = "https://jmscnc.com/revjmoney/yo/"
_FLAG_FILE = DATA_DIR / ".registered"


def _tui_register() -> None:
    if _FLAG_FILE.exists():
        return

    print()
    print("  ─────────────────────────────────────────────────────────────")
    print("  ⚠  NOT FINANCIAL ADVICE — Project Challenger is for")
    print("     educational and research purposes only. The developer")
    print("     accepts no liability for financial losses of any kind.")
    print("  ─────────────────────────────────────────────────────────────")
    print()
    print("  Sign the guestbook so the developer knows you exist.")
    print("  Email is required. Everything else is optional.")
    print()

    while True:
        email = input("  Email *: ").strip()
        if email and "@" in email and "." in email.split("@")[-1]:
            break
        print("  A valid email address is required.")

    name     = input("  Name (optional): ").strip()
    location = input("  Location — city / country (optional): ").strip()

    print()
    print("  How did you hear about this?")
    print("  [1] GitHub  [2] Reddit  [3] Twitter/X  [4] Friend  [5] Search  [6] Other")
    _heard_map = {
        "1": "github", "2": "reddit", "3": "twitter",
        "4": "friend", "5": "search", "6": "other",
    }
    heard_raw   = input("  Choice (leave blank to skip): ").strip()
    heard_from  = _heard_map.get(heard_raw, "")
    heard_other = ""
    if heard_from == "other":
        heard_other = input("  Please specify: ").strip()

    sc_raw  = input("  Listened to Rev. J. Money on SoundCloud? (y/n): ").strip().lower()
    upd_raw = input("  Want occasional project updates by email?  (y/n): ").strip().lower()
    dev_raw = input("  Developer — might contribute code?         (y/n): ").strip().lower()

    payload = {
        "email":          email,
        "name":           name,
        "location":       location,
        "heard_from":     heard_from,
        "heard_other":    heard_other,
        "listened_sc":    sc_raw  == "y",
        "wants_updates":  upd_raw == "y",
        "is_developer":   dev_raw == "y",
        "sys_os":         platform.system(),
        "sys_os_release": platform.release(),
        "sys_machine":    platform.machine(),
        "sys_python":     sys.version.split()[0],
        "sys_cpu_count":  multiprocessing.cpu_count(),
        "app_version":    "0.28.6-cdx",
    }

    print()
    print("  Transmitting...", end="", flush=True)
    for attempt in range(1, 4):
        try:
            raw = json.dumps(payload).encode()
            req = _ureq.Request(_PING_URL, data=raw,
                                headers={"Content-Type": "application/json"})
            with _ureq.urlopen(req, timeout=8):
                pass
            break
        except Exception:
            if attempt < 3:
                time.sleep(attempt * 2)

    try:
        _FLAG_FILE.write_text("1")
    except Exception:
        pass

    print(" done. Welcome to the mesh.")
    print()


# ── Terminal auth ─────────────────────────────────────────────────────────────

def tui_auth() -> None:
    """
    Password gate for terminal entry points (controller.py, main.py).
    First run: create credentials then run registration.
    Subsequent runs: verify credentials (max 5 attempts).
    """
    import auth as _auth

    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║  ⚡  PROJECT CHALLENGER                                  ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()

    if not _auth.has_credentials():
        print("  First run — create a username and password.")
        print("  Stored locally only. Never sent anywhere.")
        print()
        while True:
            user  = input("  Username: ").strip()
            if not user:
                print("  Username cannot be empty.")
                continue
            pass1 = getpass("  Password (min 6 chars): ")
            pass2 = getpass("  Confirm password:       ")
            if pass1 != pass2:
                print("  Passwords do not match. Try again.\n")
                continue
            try:
                _auth.save_credentials(user, pass1)
                print(f"\n  Account created. Welcome, {user}.")
                break
            except ValueError as exc:
                print(f"  {exc}")
        _tui_register()
        return

    attempts = 0
    while attempts < 5:
        user = input("  Username: ").strip()
        pw   = getpass("  Password: ")
        if _auth.check_credentials(user, pw):
            print(f"\n  Authenticated. Welcome back, {user}.\n")
            return
        attempts += 1
        left = 5 - attempts
        msg  = f"  {left} attempt(s) remaining." if left else ""
        print(f"  Invalid credentials.{msg}")

    print("\n  Too many failed attempts. Exiting.")
    sys.exit(1)
