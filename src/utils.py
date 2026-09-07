from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def require_package(package_name: str, install_hint: str | None = None) -> None:
    try:
        __import__(package_name)
    except ImportError as exc:
        hint = install_hint or f"установите `{package_name}` из requirements.txt."
        raise RuntimeError(hint) from exc
