"""The repair plan: a backend-neutral, typed description of one intervention."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from typing import ClassVar, Mapping

ROLES = ('tool', 'system', 'user')


class PlanError(ValueError):
    pass


def _plain(value):
    """JSON-compatible copy (tuples -> lists, mappings -> dicts)."""
    return json.loads(json.dumps(value, default=lambda v: dict(v) if isinstance(v, Mapping) else list(v)))


# ---------------------------------------------------------------------- operations
_OPERATIONS: dict[str, type] = {}


def _operation(cls):
    _OPERATIONS[cls.op] = cls
    return cls


@dataclass(frozen=True)
class Operation:
    op: ClassVar[str] = ''

    def to_dict(self) -> dict:
        data = {'op': self.op}
        data.update({f.name: _plain(getattr(self, f.name)) for f in fields(self)})
        return data

    @staticmethod
    def from_dict(data: Mapping) -> 'Operation':
        kind = data.get('op')
        cls = _OPERATIONS.get(kind)
        if cls is None:
            raise PlanError(f'Unknown operation {kind!r}')
        names = {f.name for f in fields(cls)}
        unknown = set(data) - names - {'op'}
        if unknown:
            raise PlanError(f'{kind}: unknown fields {sorted(unknown)}')
        values = {}
        for f in fields(cls):
            if f.name in data:
                v = data[f.name]
                values[f.name] = tuple(v) if isinstance(v, list) and f.type.startswith('tuple') else v
        return cls(**values)


@_operation
@dataclass(frozen=True)
class InjectMessage(Operation):
    """Insert a message at the rollback point. ``tool``/``system`` deliver it under an external role."""
    op: ClassVar[str] = 'inject_message'
    role: str
    content: str
    name: str | None = None            # tool name when role == 'tool'
    persistent: bool = False           # re-applied on every turn of the replay

    def __post_init__(self):
        if self.role not in ROLES:
            raise PlanError(f'inject_message role must be one of {ROLES}')
        if not self.content.strip():
            raise PlanError('inject_message needs content')


@_operation
@dataclass(frozen=True)
class ReplaceEvent(Operation):
    """Patch a kept event before the rollback point (e.g. complete a handoff's context)."""
    op: ClassVar[str] = 'replace_event'
    step_index: int
    sub_index: int
    content: str
    reason: str = ''


@_operation
@dataclass(frozen=True)
class RequeryTool(Operation):
    """Re-execute a read-only tool at the rollback point and insert its fresh result."""
    op: ClassVar[str] = 'requery_tool'
    tool: str
    arguments: Mapping = field(default_factory=dict)
    reason: str = ''


@_operation
@dataclass(frozen=True)
class GuardCalls(Operation):
    """Pre-execution gate for every proposed call during the replay."""
    op: ClassVar[str] = 'guard_calls'
    rules: tuple = ()                  # rule specs (policy.rules format)
    validate_schema: bool = False
    contract: Mapping | None = None    # IntentContract.to_dict()
    allowed_tools: tuple = ()          # empty: no restriction


@_operation
@dataclass(frozen=True)
class BanCalls(Operation):
    op: ClassVar[str] = 'ban_calls'
    signatures: tuple = ()             # ToolCall.signature() values
    reason: str = ''


@_operation
@dataclass(frozen=True)
class SwitchTool(Operation):
    op: ClassVar[str] = 'switch_tool'
    from_tool: str
    to_tool: str
    reason: str = ''


@_operation
@dataclass(frozen=True)
class Resample(Operation):
    op: ClassVar[str] = 'resample'
    fresh_context: bool = False        # drop the abandoned branch from the agent's context
    seed_offset: int = 1


@_operation
@dataclass(frozen=True)
class AskUser(Operation):
    op: ClassVar[str] = 'ask_user'
    question: str
    fields: tuple = ()


@_operation
@dataclass(frozen=True)
class Retry(Operation):
    op: ClassVar[str] = 'retry'
    tool: str
    arguments: Mapping = field(default_factory=dict)
    backoff_s: float = 0.0


@_operation
@dataclass(frozen=True)
class Compensate(Operation):
    """Undo executed writes that have an inverse; list the ones that cannot be undone."""
    op: ClassVar[str] = 'compensate'
    calls: tuple = ()                  # ({'tool', 'arguments', 'undoes_step'}, ...)
    irreversible: tuple = ()           # ({'tool', 'step_index', 'reason'}, ...)


@_operation
@dataclass(frozen=True)
class Escalate(Operation):
    op: ClassVar[str] = 'escalate'
    reason: str
    summary: str = ''


@_operation
@dataclass(frozen=True)
class Abstain(Operation):
    op: ClassVar[str] = 'abstain'
    reason: str


TERMINAL_OPS = (Escalate, Abstain, Compensate)


# ---------------------------------------------------------------------- acceptance specs
CHECK_KINDS = ('claims_grounded', 'values_present', 'no_repeat', 'calls_valid', 'rules_pass', 'obligations_done',
               'contract_respected', 'asked_user', 'tool_not_called', 'tool_succeeded', 'no_error_results')


@dataclass(frozen=True)
class AcceptanceCheck:
    kind: str
    params: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in CHECK_KINDS:
            raise PlanError(f'Unknown acceptance check {self.kind!r}')
        object.__setattr__(self, 'params', _plain(dict(self.params)))

    def to_dict(self) -> dict:
        return {'kind': self.kind, 'params': dict(self.params)}

    @classmethod
    def from_dict(cls, data: Mapping) -> 'AcceptanceCheck':
        return cls(data['kind'], data.get('params') or {})


# ---------------------------------------------------------------------- plan
@dataclass(frozen=True)
class RepairPlan:
    strategy_id: str
    category: str
    rollback_step: int | None
    operations: tuple[Operation, ...]
    acceptance: tuple[AcceptanceCheck, ...] = ()
    rationale: str = ''
    subtype: str | None = None
    estimated_tokens: int = 0
    expected_success: float = 0.0
    evidence: tuple[str, ...] = ()
    lessons: tuple[str, ...] = ()
    hypothesis: Mapping = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, 'operations', tuple(self.operations))
        object.__setattr__(self, 'acceptance', tuple(self.acceptance))
        object.__setattr__(self, 'evidence', tuple(self.evidence))
        object.__setattr__(self, 'lessons', tuple(self.lessons))
        object.__setattr__(self, 'hypothesis', _plain(dict(self.hypothesis)))
        if not self.operations:
            raise PlanError(f'{self.strategy_id}: a plan needs at least one operation')
        if not all(isinstance(o, Operation) for o in self.operations):
            raise PlanError(f'{self.strategy_id}: operations must be Operation instances')
        if self.rollback_step is None:
            if not any(isinstance(o, TERMINAL_OPS) for o in self.operations):
                raise PlanError(f'{self.strategy_id}: a plan without a rollback step must escalate, abstain or compensate')
        elif type(self.rollback_step) is not int or self.rollback_step < 1:
            raise PlanError(f'{self.strategy_id}: rollback_step must be a positive integer')
        if not 0.0 <= self.expected_success <= 1.0:
            raise PlanError(f'{self.strategy_id}: expected_success must be in [0, 1]')
        if self.estimated_tokens < 0:
            raise PlanError(f'{self.strategy_id}: estimated_tokens must be non-negative')

    @property
    def terminal(self) -> bool:
        """Nothing to replay: the plan ends the attempt (escalate, abstain, compensate-and-escalate)."""
        return self.rollback_step is None

    @property
    def plan_id(self) -> str:
        """Content hash: identical plans share an id across processes and runs."""
        body = self.to_dict()
        body.pop('plan_id', None)
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]

    def find(self, kind: type) -> tuple[Operation, ...]:
        return tuple(o for o in self.operations if isinstance(o, kind))

    def to_dict(self) -> dict:
        data = {'strategy_id': self.strategy_id, 'category': self.category, 'subtype': self.subtype,
                'rollback_step': self.rollback_step, 'operations': [o.to_dict() for o in self.operations],
                'acceptance': [c.to_dict() for c in self.acceptance], 'rationale': self.rationale,
                'estimated_tokens': self.estimated_tokens, 'expected_success': self.expected_success,
                'evidence': list(self.evidence), 'lessons': list(self.lessons), 'hypothesis': dict(self.hypothesis)}
        return data

    def to_json(self, **kwargs) -> str:
        data = self.to_dict()
        data['plan_id'] = self.plan_id
        return json.dumps(data, ensure_ascii=False, **kwargs)

    @classmethod
    def from_dict(cls, data: Mapping) -> 'RepairPlan':
        try:
            plan = cls(strategy_id=data['strategy_id'], category=data['category'],
                       rollback_step=data.get('rollback_step'),
                       operations=tuple(Operation.from_dict(o) for o in data['operations']),
                       acceptance=tuple(AcceptanceCheck.from_dict(c) for c in data.get('acceptance', ())),
                       rationale=data.get('rationale', ''), subtype=data.get('subtype'),
                       estimated_tokens=int(data.get('estimated_tokens', 0)),
                       expected_success=float(data.get('expected_success', 0.0)),
                       evidence=tuple(data.get('evidence', ())), lessons=tuple(data.get('lessons', ())),
                       hypothesis=data.get('hypothesis') or {})
        except KeyError as exc:
            raise PlanError(f'Plan missing field {exc}') from None
        if 'plan_id' in data and data['plan_id'] != plan.plan_id:
            raise PlanError('plan_id does not match the plan content')
        return plan
