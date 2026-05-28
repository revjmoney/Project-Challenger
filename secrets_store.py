"""
Local secret storage helpers.

On Windows, values are protected with DPAPI for the current user before they
are written under data/.secrets.json. On other platforms we fall back to a
mode-0600 JSON file so the app remains portable.

Copyright (c) 2026 Rev. J. Money.
Non-commercial learning/research use only. See LICENSE and NOTICE.
"""
from __future__ import annotations

import base64
import ctypes
import json
import os
import platform
from ctypes import wintypes
from pathlib import Path
from typing import Any


class SecretStoreError(RuntimeError):
    pass


def _is_windows() -> bool:
    return platform.system().lower() == "windows"


if _is_windows():
    class _DATA_BLOB(ctypes.Structure):
        _fields_ = [
            ("cbData", wintypes.DWORD),
            ("pbData", ctypes.POINTER(ctypes.c_char)),
        ]


def _blob_from_bytes(data: bytes) -> "_DATA_BLOB":
    buf = ctypes.create_string_buffer(data)
    return _DATA_BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))


def _dpapi_protect(value: str) -> str:
    raw = value.encode("utf-8")
    in_blob = _blob_from_bytes(raw)
    out_blob = _DATA_BLOB()
    crypt32 = ctypes.windll.crypt32
    if not crypt32.CryptProtectData(
        ctypes.byref(in_blob), None, None, None, None, 0, ctypes.byref(out_blob)
    ):
        raise SecretStoreError("CryptProtectData failed")
    try:
        encrypted = ctypes.string_at(out_blob.pbData, out_blob.cbData)
        return base64.b64encode(encrypted).decode("ascii")
    finally:
        ctypes.windll.kernel32.LocalFree(out_blob.pbData)


def _dpapi_unprotect(value: str) -> str:
    encrypted = base64.b64decode(value.encode("ascii"))
    in_blob = _blob_from_bytes(encrypted)
    out_blob = _DATA_BLOB()
    crypt32 = ctypes.windll.crypt32
    if not crypt32.CryptUnprotectData(
        ctypes.byref(in_blob), None, None, None, None, 0, ctypes.byref(out_blob)
    ):
        raise SecretStoreError("CryptUnprotectData failed")
    try:
        raw = ctypes.string_at(out_blob.pbData, out_blob.cbData)
        return raw.decode("utf-8")
    finally:
        ctypes.windll.kernel32.LocalFree(out_blob.pbData)


def _restrict_file(path: Path) -> None:
    if _is_windows():
        return
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def load_secrets(path: str | Path) -> dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

    backend = data.get("backend")
    items = data.get("items", {})
    if not isinstance(items, dict):
        return {}

    out: dict[str, str] = {}
    for key, stored in items.items():
        if not isinstance(key, str) or not isinstance(stored, str):
            continue
        try:
            if backend == "dpapi":
                if not _is_windows():
                    continue
                out[key] = _dpapi_unprotect(stored)
            elif backend == "plain":
                out[key] = stored
        except Exception:
            continue
    return out


def save_secrets(path: str | Path, secrets: dict[str, str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in secrets.items() if isinstance(k, str) and v}
    if _is_windows():
        data = {
            "backend": "dpapi",
            "items": {k: _dpapi_protect(v) for k, v in sorted(clean.items())},
        }
    else:
        data = {"backend": "plain", "items": dict(sorted(clean.items()))}
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _restrict_file(p)


def update_secrets(path: str | Path, updates: dict[str, str | None]) -> None:
    current = load_secrets(path)
    for key, value in updates.items():
        if value:
            current[key] = value
        else:
            current.pop(key, None)
    save_secrets(path, current)
