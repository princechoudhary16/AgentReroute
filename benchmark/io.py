"""Small JSON and hashing helpers shared by every module."""
import hashlib
import json
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def digest(value):
    data = value if isinstance(value, bytes) else canonical(value).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(path)
    return path


def _drop_partial_tail(path):
    """Discard a final line with no terminating newline.

    These files are append-only run logs. A process killed between the write and the newline leaves a
    partial record that was never committed, and appending after it would fuse the fragment and the next
    record into one unparseable line, costing a good record as well as the fragment.
    """
    size = path.stat().st_size
    if not size:
        return
    with path.open('rb') as stream:
        stream.seek(size - 1)
        if stream.read(1) == b'\n':
            return
        stream.seek(0)
        data = stream.read()
    with path.open('wb') as stream:
        stream.write(data[:data.rfind(b'\n') + 1])      # 0 when the only line is partial: drops it


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _drop_partial_tail(path)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def read_jsonl(path):
    """Rows of a JSONL log, tolerating an uncommitted final line.

    A generation or training run resumes by reading its own index, so a partial last line from an
    interrupted append must not make the whole log unreadable. Only the last line is forgiven: a
    malformed line anywhere earlier is real corruption and still raises, so this cannot quietly serve
    a damaged file as if it were complete.
    """
    path = Path(path)
    if not path.exists():
        return []
    lines = [line for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    rows = []
    for position, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if position != len(lines) - 1:
                raise
    return rows
