"""
Guardrails for loading local model artifacts.

Pickle/joblib and legacy torch checkpoints can execute code while loading. This
module does not make those formats inherently safe, but it limits loads to the
project's expected artifact directories and records SHA-256 hashes for locally
trained files so replacements are visible.

Copyright (c) 2026 Rev. J. Money.
Non-commercial learning/research use only. See LICENSE and NOTICE.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import joblib


_ROOT = Path(__file__).resolve().parent
_TRUSTED_ROOTS = ((_ROOT / "models").resolve(), (_ROOT / "data").resolve())
_MANIFEST_PATH = (_ROOT / "models" / "artifact_manifest.json").resolve()
_STRICT = os.environ.get("CHALLENGER_STRICT_MODEL_HASH", "").lower() in {
    "1",
    "true",
    "yes",
}


def _resolve_trusted(path: str | os.PathLike[str]) -> Path:
    p = Path(path).resolve()
    if not any(p == root or root in p.parents for root in _TRUSTED_ROOTS):
        raise ValueError(f"Refusing to load model artifact outside trusted dirs: {p}")
    return p


def _rel(path: Path) -> str:
    try:
        return path.relative_to(_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest() -> dict[str, Any]:
    if not _MANIFEST_PATH.exists():
        return {}
    try:
        data = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_manifest(data: dict[str, Any]) -> None:
    _MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    _MANIFEST_PATH.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def sign_artifact(path: str | os.PathLike[str]) -> None:
    p = _resolve_trusted(path)
    if not p.exists():
        return
    manifest = _load_manifest()
    manifest[_rel(p)] = {
        "sha256": _sha256(p),
        "size": p.stat().st_size,
    }
    _save_manifest(manifest)


def verify_artifact(path: str | os.PathLike[str]) -> None:
    p = _resolve_trusted(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    entry = _load_manifest().get(_rel(p))
    if not entry:
        print(f"[MODEL SECURITY] No local hash recorded for {_rel(p)}; load allowed.")
        return
    actual = _sha256(p)
    expected = entry.get("sha256")
    if actual != expected:
        msg = f"[MODEL SECURITY] Hash mismatch for {_rel(p)}; artifact may have changed."
        if _STRICT:
            raise RuntimeError(msg)
        print(msg)


def load_joblib(path: str | os.PathLike[str]) -> Any:
    p = _resolve_trusted(path)
    verify_artifact(p)
    return joblib.load(p)


def load_torch(path: str | os.PathLike[str], **kwargs: Any) -> Any:
    p = _resolve_trusted(path)
    verify_artifact(p)
    import torch

    kwargs.setdefault("weights_only", True)
    try:
        return torch.load(p, **kwargs)
    except Exception as exc:
        if kwargs.get("weights_only") is True:
            print(
                "[MODEL SECURITY] Safe torch load failed; falling back for "
                f"legacy checkpoint {_rel(p)}: {type(exc).__name__}"
            )
            kwargs["weights_only"] = False
            return torch.load(p, **kwargs)
        raise
