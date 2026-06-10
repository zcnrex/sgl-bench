"""Shared JSON/JSONL serialization helpers for run artifacts ([[RFC-0001:C-RUN-OUTPUT]])."""

from __future__ import annotations

import json
from pathlib import Path


def json_line(obj) -> str:
    """One JSONL line; non-JSON-native values are stringified."""
    return json.dumps(obj, default=str)


def parse_jsonl(text: str) -> list[dict]:
    """Records from JSONL text, skipping blank lines."""
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def read_jsonl(path) -> list[dict]:
    """Records from a JSONL file."""
    return parse_jsonl(Path(path).read_text())


def write_jsonl(records, path) -> Path:
    """Write records as JSONL, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json_line(rec) + "\n")
    return path


def write_json(obj, path) -> Path:
    """Write one indented JSON document, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    return path
