"""Safe helpers for project artifacts and JSON reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def _prepare_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def save_joblib(value: Any, path: str | Path) -> Path:
    destination = _prepare_parent(path)
    joblib.dump(value, destination)
    return destination


def load_joblib(path: str | Path) -> Any:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Required model artifact is missing: {source}. Run the training pipeline first."
        )
    return joblib.load(source)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(value: Any, path: str | Path) -> Path:
    destination = _prepare_parent(path)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True, default=_json_default)
        file.write("\n")
    return destination


def load_json(path: str | Path) -> Any:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Required JSON artifact is missing: {source}")
    with source.open("r", encoding="utf-8") as file:
        return json.load(file)
