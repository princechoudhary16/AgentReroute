"""Claim provenance: is every checkable claim of an agent grounded in evidence it had?

A *claim* is an identifier, an email, a money amount, or a quantified number
("3 available options") in an agent's message, handoff or tool arguments. A
claim is checked against the *evidence visible before the step*: user turns,
tool outputs, handoff contexts and the task instruction. The agent's own
earlier messages are not evidence, so an invention cannot ground itself.

Statuses:

* ``grounded``      the value appears in evidence (directly or as a computed aggregate);
* ``contradicted``  evidence has a value for the same field and it differs (``expected`` holds it);
* ``ungrounded``    an identifier, email or amount that no evidence contains;
* ``unverifiable``  a bare number with no field to compare against (not treated as a fault).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from decimal import Decimal

from ..trace import AGENT, TOOL, USER, Event, ToolCall, Trace
from .tool_output import JSON, Aggregate, aggregates, key_tokens, parse_output, singular, subset_sums, to_decimal, words

ID, EMAIL, MONEY, NUMBER = 'id', 'email', 'money', 'number'
GROUNDED, CONTRADICTED, UNGROUNDED, UNVERIFIABLE = 'grounded', 'contradicted', 'ungrounded', 'unverifiable'
FAULTY = (CONTRADICTED, UNGROUNDED)

_EMAIL = re.compile(r'\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b')
_IDS = re.compile(r'#[A-Za-z]{0,3}\d{3,}\b|\b[a-z]+(?:_[a-z0-9]+)*_\d{2,}\b|\b\d{6,}\b')
_MONEY = re.compile(r'\$\s?(\d[\d,]*(?:\.\d{1,2})?)|\b(\d[\d,]*\.\d{2})\s?(?:usd|dollars?)\b', re.I)
_NUMBER = re.compile(r'(?<![\w.$#/])(\d+(?:\.\d+)?)(?!\w|\.\d)')
_LIST_MARKER = re.compile(r'(?m)^\s*\d+[.)]\s')
_WORD = re.compile(r"[a-z][a-z'-]*")
_STOP = frozenset('a an the of for to in on at is are was were be been there this that these those and or with '
                  'your you my our we i it its by from as each per total new item'.split()) - {'total', 'item'}
_QUANTITY_WORDS = frozenset('option variant item order product unit piece result available count left remaining '
                            'stock size color colour type method card'.split())
_ID_ARGUMENT = re.compile(r'(^|_)(id|ids)$|email')
_MONEY_KEYS = frozenset({'price', 'amount', 'total', 'refund', 'cost', 'balance', 'fee', 'charge'})


@dataclass(frozen=True)
class Claim:
    kind: str
    raw: str
    value: str                          # normalized: lower-case id, Decimal string for amounts
    step_index: int
    qualifiers: tuple[str, ...] = ()    # nearby words ("available", "option")
    argument: str | None = None         # set for claims made in tool-call arguments
    tool: str | None = None

    @property
    def number(self) -> Decimal | None:
        return Decimal(self.value) if self.kind in (MONEY, NUMBER) else None


@dataclass(frozen=True)
class SourceRef:
    step_index: int
    kind: str                           # user | tool | handoff | instruction
    tool: str | None = None
    detail: str = ''                    # field path or aggregate description


@dataclass(frozen=True)
class ClaimCheck:
    claim: Claim
    status: str
    sources: tuple[SourceRef, ...] = ()
    expected: tuple[str, ...] = ()       # evidence values for contradicted claims

    @property
    def faulty(self) -> bool:
        return self.status in FAULTY

    def describe(self) -> str:
        where = f' ({self.claim.argument})' if self.claim.argument else ''
        if self.status == CONTRADICTED:
            return f'"{self.claim.raw}"{where} contradicts the evidence: {", ".join(self.expected)}'
        if self.status == UNGROUNDED:
            return f'"{self.claim.raw}"{where} does not appear in any user message or tool output'
        return f'"{self.claim.raw}"{where} is {self.status}'


# ---------------------------------------------------------------------- extraction
def _qualifiers(text: str, start: int, end: int) -> tuple[str, ...]:
    after = [w for w in _WORD.findall(text[end:end + 60].lower()) if w not in _STOP][:3]
    before = [w for w in _WORD.findall(text[max(0, start - 60):start].lower()) if w not in _STOP][-4:]
    tokens = []
    for word in before + after:
        token = singular(word)
        if token not in tokens:
            tokens.append(token)
    return tuple(tokens)


def extract_claims(text: str, step_index: int) -> list[Claim]:
    claims, taken = [], []

    def free(span):
        return all(span[1] <= a or span[0] >= b for a, b in taken)

    for m in _EMAIL.finditer(text):
        claims.append(Claim(EMAIL, m.group(0), m.group(0).lower(), step_index))
        taken.append(m.span())
    for m in _MONEY.finditer(text):
        if not free(m.span()):
            continue
        amount = to_decimal(m.group(1) or m.group(2))
        if amount is not None:
            claims.append(Claim(MONEY, m.group(0), str(amount.quantize(Decimal('0.01'))), step_index,
                                _qualifiers(text, *m.span())))
            taken.append(m.span())
    for m in _IDS.finditer(text):
        if free(m.span()):
            claims.append(Claim(ID, m.group(0), m.group(0).lower(), step_index))
            taken.append(m.span())
    markers = {m.start() + len(m.group(0)) - len(m.group(0).lstrip()) for m in _LIST_MARKER.finditer(text)}
    for m in _NUMBER.finditer(text):
        if not free(m.span()) or m.start() in markers:      # skip "1. " list numbering
            continue
        quals = _qualifiers(text, *m.span())
        claims.append(Claim(NUMBER, m.group(0), str(Decimal(m.group(1))), step_index, quals))
        taken.append(m.span())
    return claims


def argument_claims(call: ToolCall, step_index: int) -> list[Claim]:
    """Identifier-like arguments of a call; each is a claim that the value came from evidence."""
    claims = []
    for name, value in call.arguments.items():
        if name == '__raw__':
            continue
        values = value if isinstance(value, list) else [value]
        for v in values:
            if not isinstance(v, (str, int)) or isinstance(v, bool):
                continue
            text = str(v)
            if _ID_ARGUMENT.search(name) or _IDS.fullmatch(text) or _EMAIL.fullmatch(text):
                kind = EMAIL if _EMAIL.fullmatch(text) else ID
                claims.append(Claim(kind, text, text.lower(), step_index, argument=name, tool=call.name))
    return claims


def event_claims(event: Event) -> list[Claim]:
    claims = extract_claims(event.text, event.step_index) if event.text else []
    if event.handoff:
        for key, value in (event.handoff.get('context') or {}).items():
            values = value if isinstance(value, list) else [value]
            for v in values:
                if isinstance(v, (str, int)) and not isinstance(v, bool) and \
                        (_ID_ARGUMENT.search(str(key)) or _IDS.fullmatch(str(v))):
                    claims.append(Claim(ID, str(v), str(v).lower(), event.step_index, argument=f'context.{key}'))
    for call in event.tool_calls:
        claims.extend(argument_claims(call, event.step_index))
    return claims


# ---------------------------------------------------------------------- evidence index
@dataclass
class _Source:
    ref: SourceRef
    text: str
    data: object = None
    aggregates: tuple[Aggregate, ...] = ()
    numbers: list = field(default_factory=list)       # (Decimal, key, path)


class EvidenceIndex:
    """Everything an agent could legitimately rely on before a given step."""

    def __init__(self, sources: list[_Source]):
        self.sources = sources

    @classmethod
    def before(cls, trace: Trace, step_index: int) -> 'EvidenceIndex':
        sources = []
        if trace.instruction:
            sources.append(_Source(SourceRef(0, 'instruction'), trace.instruction))
        for event in trace.before(step_index):
            if event.kind == USER:
                sources.append(_Source(SourceRef(event.step_index, 'user'), event.content))
            elif event.kind == TOOL:
                sources.append(cls._structured(SourceRef(event.step_index, 'tool', event.tool_name), event.content))
            elif event.kind == AGENT and event.handoff:
                context = event.handoff.get('context') or {}
                sources.append(cls._structured(SourceRef(event.step_index, 'handoff'), context))
        return cls(sources)

    @staticmethod
    def _structured(ref: SourceRef, content) -> _Source:
        parsed = parse_output(content) if isinstance(content, str) else None
        data = parsed.data if parsed is not None and parsed.kind == JSON else (
            content if isinstance(content, (dict, list)) else None)
        text = content if isinstance(content, str) else _dump(content)
        source = _Source(ref, text, data)
        if data is not None:
            parsed = parse_output(_dump(data))
            for leaf in parsed.leaves():
                number = to_decimal(leaf.value)
                if number is not None:
                    source.numbers.append((number, leaf.key, leaf.dotted))
            source.aggregates = aggregates(data)
        else:
            for m in re.finditer(r'-?\d[\d,]*(?:\.\d+)?', text):
                number = to_decimal(m.group(0))
                if number is not None:
                    source.numbers.append((number, '', ''))
        return source

    # ------------------------------------------------------------------ lookups
    def find_text(self, value: str) -> list[SourceRef]:
        pattern = re.compile(r'(?<![\w#@.-])' + re.escape(value) + r'(?![\w@-])', re.I)
        return [s.ref for s in self.sources if pattern.search(s.text)]

    def find_number(self, number: Decimal) -> list[SourceRef]:
        refs = []
        for s in self.sources:
            if any(n == number for n, _, _ in s.numbers) or any(a.value == number for a in s.aggregates):
                refs.append(s.ref)
        return refs

    def find_sum(self, number: Decimal) -> list[SourceRef]:
        refs = []
        for s in self.sources:
            groups: dict[tuple, list[Decimal]] = {}
            for n, key, path in s.numbers:
                parent = tuple(p for p in path.split('.') if not p.isdigit())
                groups.setdefault(parent, []).append(n)
            if any(number in subset_sums(values) for values in groups.values() if len(values) > 1):
                refs.append(s.ref)
        return refs

    def anchored(self, qualifiers, keys) -> list[tuple[Decimal, SourceRef]]:
        """Numbers inside an object named by the claim ("the water bottle ... $45.04")."""
        wanted, found = set(qualifiers), []
        for s in self.sources:
            for path, obj in _objects(s.data):
                names = {w for v in obj.values() if isinstance(v, str) for w in words(v)}
                if not names & wanted:
                    continue
                for key, value in obj.items():
                    number = to_decimal(value)
                    if number is not None and key_tokens(key) & keys:
                        detail = '.'.join(str(p) for p in path + (key,))
                        found.append((number, SourceRef(s.ref.step_index, s.ref.kind, s.ref.tool, detail)))
        return found

    def keyed(self, qualifiers: tuple[str, ...]) -> list[tuple[Decimal, SourceRef]]:
        """Numbers and aggregates whose field matches a qualifier word."""
        wanted = set(qualifiers)
        found = []
        for s in self.sources:
            for n, key, path in s.numbers:
                if key and key_tokens(key) & wanted:
                    found.append((n, SourceRef(s.ref.step_index, s.ref.kind, s.ref.tool, path)))
            for a in s.aggregates:
                if a.kind == 'count_true' and key_tokens(a.key) & wanted:
                    found.append((a.value, SourceRef(s.ref.step_index, s.ref.kind, s.ref.tool, a.describe())))
                elif a.kind == 'size' and a.path and key_tokens(a.path.split('.')[-1]) & wanted:
                    found.append((a.value, SourceRef(s.ref.step_index, s.ref.kind, s.ref.tool, a.describe())))
        return found


def _objects(value, path=()):
    if isinstance(value, dict):
        yield path, value
        for k, v in value.items():
            yield from _objects(v, path + (str(k),))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _objects(v, path + (i,))


def expand(qualifiers: tuple[str, ...], synonyms=None) -> tuple[str, ...]:
    """Add field names a domain associates with a word (``refund`` -> ``price``, ``amount``)."""
    out = list(qualifiers)
    for q in qualifiers:
        for key in (synonyms or {}).get(q, ()):
            for token in key_tokens(key):
                if token not in out:
                    out.append(token)
    return tuple(out)


def _dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


# ---------------------------------------------------------------------- checking
def check_claim(claim: Claim, index: EvidenceIndex, synonyms=None) -> ClaimCheck:
    if claim.kind in (ID, EMAIL):
        refs = index.find_text(claim.raw)
        return ClaimCheck(claim, GROUNDED if refs else UNGROUNDED, tuple(refs))

    number = claim.number
    qualifiers = expand(claim.qualifiers, synonyms)
    comparable: list[tuple[Decimal, SourceRef]] = []
    if claim.kind == MONEY and claim.qualifiers:
        comparable = index.anchored(claim.qualifiers, _MONEY_KEYS | set(qualifiers))
    if not comparable and qualifiers:
        keyed = index.keyed(qualifiers)
        if claim.kind == MONEY:
            keyed = [(n, r) for n, r in keyed if not r.detail or key_tokens(r.detail.split('.')[-1]) & _MONEY_KEYS
                     or 'sum of' in r.detail]
        # A count qualified by a boolean flag ("3 available options") must match that flag's count.
        flags = [(n, r) for n, r in keyed if 'true' in r.detail]
        comparable = flags or keyed
    if comparable:
        matches = [r for n, r in comparable if n == number]
        if matches:
            return ClaimCheck(claim, GROUNDED, tuple(matches))
        summed = index.find_sum(number) if claim.kind == MONEY else []
        if summed:
            return ClaimCheck(claim, GROUNDED, tuple(summed))
        expected = tuple(dict.fromkeys(f'{r.detail or "value"} = {_fmt(n, claim.kind)}' for n, r in comparable))[:5]
        return ClaimCheck(claim, CONTRADICTED, tuple(r for _, r in comparable)[:5], expected)

    refs = index.find_number(number)
    if not refs and claim.kind == MONEY:
        refs = index.find_sum(number)
    if refs:
        return ClaimCheck(claim, GROUNDED, tuple(refs))
    if claim.kind == MONEY:
        return ClaimCheck(claim, UNGROUNDED)
    is_quantity = any(q in _QUANTITY_WORDS for q in claim.qualifiers)
    return ClaimCheck(claim, UNGROUNDED if is_quantity else UNVERIFIABLE)


def _fmt(number: Decimal, kind: str) -> str:
    return f'${number:.2f}' if kind == MONEY else (str(int(number)) if number == number.to_integral() else str(number))


@dataclass(frozen=True)
class ProvenanceReport:
    step_index: int
    checks: tuple[ClaimCheck, ...]

    @property
    def faulty(self) -> tuple[ClaimCheck, ...]:
        return tuple(c for c in self.checks if c.faulty)

    @property
    def clean(self) -> bool:
        return not self.faulty


def provenance(trace: Trace, step_index: int, include_arguments: bool = True, synonyms=None) -> ProvenanceReport:
    """Check every claim made by agent events at ``step_index`` against earlier evidence.

    ``synonyms`` maps a word to field names that hold its value (``TaskSpec.fact_keywords``).
    """
    index = EvidenceIndex.before(trace, step_index)
    checks = []
    for event in trace.at(step_index):
        if not event.is_agent:
            continue
        for claim in event_claims(event):
            if claim.argument and not include_arguments:
                continue
            checks.append(check_claim(claim, index, synonyms))
    return ProvenanceReport(step_index, tuple(checks))


def provenance_from(trace: Trace, first_step: int, synonyms=None) -> tuple[ProvenanceReport, ...]:
    """Provenance of every agent step from ``first_step`` on (used to check a repaired rollout)."""
    steps = sorted({e.step_index for e in trace.from_step(first_step) if e.is_agent})
    return tuple(provenance(trace, s, synonyms=synonyms) for s in steps)
