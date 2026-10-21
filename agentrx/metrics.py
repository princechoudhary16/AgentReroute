"""Structured telemetry events written as append-only JSONL.

Set AGENTRX_METRICS=0 to disable, or AGENTRX_METRICS_FILE to redirect output.
Emission never raises: telemetry must not break a diagnostic run.
"""
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PATH = Path('runs/metrics/events.jsonl')
RUN_ID = os.environ.get('AGENTRX_RUN_ID') or uuid.uuid4().hex[:12]


def metrics_path():
    return Path(os.environ.get('AGENTRX_METRICS_FILE', DEFAULT_PATH))


def emit(event, **fields):
    """Append one event; returns the record, or None when disabled or unwritable."""
    if os.environ.get('AGENTRX_METRICS') == '0':
        return None
    record = {'ts': datetime.now(timezone.utc).isoformat(), 'run_id': RUN_ID, 'event': event, **fields}
    try:
        path = metrics_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
    except (OSError, TypeError, ValueError):
        return None
    return record


def read_events(path=None):
    path = Path(path) if path else metrics_path()
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events
