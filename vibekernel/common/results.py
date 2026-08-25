from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path


def write_results(payload: dict[str, object], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"created_at": datetime.now(timezone.utc).isoformat(), **payload}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
