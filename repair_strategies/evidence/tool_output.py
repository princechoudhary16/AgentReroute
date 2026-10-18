"""Deterministic reading of tool outputs: parsing, error classes, field access, aggregates."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from decimal import Decimal, InvalidOperation
from typing import Any

JSON, TEXT, ERROR, EMPTY = 'json', 'text', 'error', 'empty'
TRANSIENT, PERSISTENT, UNKNOWN = 'transient', 'persistent', 'unknown'

_ERROR_START = re.compile(r'^\s*(error|exception|traceback|failed|failure)\b', re.I)
_TRANSIENT = re.compile(r'time(d)?[\s-]?out|\b50[234]\b|\b429\b|rate[\s-]?limit|temporar|try again|'
                        r'unavailable|connection (reset|refused|aborted)|overloaded|busy', re.I)
_PERSISTENT = re.compile(r'deprecat|no longer (supported|available)|not supported|unsupported|\b40[14]\b|'
                         r'unknown (tool|function|endpoint|argument|parameter)|unexpected (keyword|argument)|'
                         r'invalid (parameter|argument|schema)|removed|does not exist|not found', re.I)
_DRIFT = re.compile(r'deprecat|unknown (argument|parameter)|unexpected (keyword|argument)|renamed|'
                    r'invalid (parameter|schema)|schema', re.I)

_MISSING = object()


@dataclass(frozen=True)
class Leaf:
    path: tuple            # e.g. ('items', 0, 'price')
    value: Any

    @property
    def key(self) -> str:
        names = [p for p in self.path if isinstance(p, str)]
        return names[-1] if names else ''

    @property
    def dotted(self) -> str:
        return '.'.join(str(p) for p in self.path)


@dataclass(frozen=True)
class ParsedOutput:
    kind: str
    text: str
    data: Any = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.kind in (JSON, TEXT)

    def leaves(self) -> tuple[Leaf, ...]:
        return tuple(flatten(self.data)) if self.kind == JSON else ()


def parse_output(text) -> ParsedOutput:
    raw = text if isinstance(text, str) else ('' if text is None else json.dumps(text, default=str))
    stripped = raw.strip()
    if not stripped:
        return ParsedOutput(EMPTY, raw)
    if _ERROR_START.match(stripped):
        return ParsedOutput(ERROR, raw, error=stripped)
    if stripped[0] in '{[':
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            return ParsedOutput(TEXT, raw)
        if isinstance(data, dict) and data.get('error') and len(data) <= 3:
            return ParsedOutput(ERROR, raw, data, str(data['error']))
        return ParsedOutput(JSON, raw, data)
    return ParsedOutput(TEXT, raw)


def is_error(text) -> bool:
    return parse_output(text).kind == ERROR


def error_class(text) -> str:
    """``transient`` (retry may help), ``persistent`` (it will not), or ``unknown``."""
    message = parse_output(text).error or (text or '')
    if _TRANSIENT.search(message):
        return TRANSIENT
    if _PERSISTENT.search(message):
        return PERSISTENT
    return UNKNOWN


def indicates_drift(text) -> bool:
    """The error says the call shape no longer matches the tool (renamed/deprecated arguments)."""
    message = parse_output(text).error or (text or '')
    return bool(_DRIFT.search(message))


def flatten(value, path=()) -> list[Leaf]:
    if isinstance(value, dict):
        out = []
        for k, v in value.items():
            out.extend(flatten(v, path + (str(k),)))
        return out
    if isinstance(value, list):
        out = []
        for i, v in enumerate(value):
            out.extend(flatten(v, path + (i,)))
        return out
    return [Leaf(path, value)]


def get_path(data, dotted: str, default=None):
    """``a.0.b`` lookup through dicts and lists."""
    current = data
    for part in dotted.split('.') if dotted else ():
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.lstrip('-').isdigit() and -len(current) <= int(part) < len(current):
            current = current[int(part)]
        else:
            return default
    return current


def to_decimal(value) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float, Decimal)):
        try:
            return Decimal(str(value))
        except InvalidOperation:
            return None
    if isinstance(value, str):
        text = value.strip().replace(',', '').lstrip('$')
        if re.fullmatch(r'-?\d+(\.\d+)?', text):
            return Decimal(text)
    return None


@lru_cache(maxsize=65536)
def singular(word: str) -> str:
    """Crude English singular: 'ids' -> 'id', 'entries' -> 'entry'; 'status' and 'address' stay."""
    if len(word) > 4 and word.endswith('ies'):
        return word[:-3] + 'y'
    if len(word) > 2 and word.endswith('s') and not word.endswith(('ss', 'us', 'is')):
        return word[:-1]
    return word


def words(text: str) -> list[str]:
    return list(_words(str(text)))


@lru_cache(maxsize=65536)
def _words(text: str) -> tuple[str, ...]:
    return tuple(singular(w) for w in re.findall(r'[a-z]+', text.lower()))


@lru_cache(maxsize=65536)
def key_tokens(key: str) -> frozenset[str]:
    """'payment_method_ids' -> {'payment', 'method', 'id'}."""
    return frozenset(_words(key.replace('_', ' ')))


@dataclass(frozen=True)
class Aggregate:
    """A value computed in code from a structured output (count, sum)."""
    kind: str              # 'size' | 'count_true' | 'sum'
    path: str              # collection path
    key: str               # field the aggregate is over ('' for size)
    value: Decimal

    def describe(self) -> str:
        where = self.path or 'output'
        if self.kind == 'size':
            return f'{where} has {self.value} entries'
        if self.kind == 'count_true':
            return f'{self.value} entries of {where} have {self.key}=true'
        return f'sum of {self.key} over {where} is {self.value}'


def _collections(value, path=()):
    if isinstance(value, dict):
        items = list(value.values())
        if items and all(isinstance(v, dict) for v in items):
            yield path, items
        for k, v in value.items():
            yield from _collections(v, path + (str(k),))
    elif isinstance(value, list):
        if value and all(isinstance(v, dict) for v in value):
            yield path, value
        for i, v in enumerate(value):
            yield from _collections(v, path + (i,))


def aggregates(data) -> tuple[Aggregate, ...]:
    """Sizes of collections, counts of true flags and sums of numeric fields."""
    out = []
    for path, items in _collections(data):
        dotted = '.'.join(str(p) for p in path)
        out.append(Aggregate('size', dotted, '', Decimal(len(items))))
        keys = {k for item in items for k in item}
        for key in sorted(keys):
            values = [item.get(key) for item in items]
            if all(isinstance(v, bool) for v in values if v is not None) and any(isinstance(v, bool) for v in values):
                out.append(Aggregate('count_true', dotted, key, Decimal(sum(1 for v in values if v is True))))
            numbers = [to_decimal(v) for v in values]
            if len(numbers) > 1 and all(n is not None for n in numbers):
                out.append(Aggregate('sum', dotted, key, sum(numbers, Decimal(0))))
    return tuple(out)


def subset_sums(values: list[Decimal], max_size: int = 3, max_items: int = 12, min_size: int = 1) -> set[Decimal]:
    """Sums of ``min_size`` to ``max_size`` values (bounded, for refunds of a few items)."""
    values = values[:max_items]
    sums: set[Decimal] = set()
    frontier = {(): Decimal(0)}
    for _ in range(max_size):
        nxt = {}
        for chosen, total in frontier.items():
            start = chosen[-1] + 1 if chosen else 0
            for i in range(start, len(values)):
                key = chosen + (i,)
                nxt[key] = total + values[i]
                if len(key) >= min_size:
                    sums.add(nxt[key])
        frontier = nxt
    return sums
