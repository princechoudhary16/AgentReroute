"""Inputs of a repair decision: the attributed hypothesis and the task specification."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .taxonomy import FaultType, normalize, normalize_agent

READ, WRITE = 'read', 'write'


class SpecError(ValueError):
    pass


# ---------------------------------------------------------------------- hypothesis
@dataclass(frozen=True)
class Hypothesis:
    """One attributed fault: where it happened, what kind, and how sure attribution is."""
    step_index: int | None
    category: str
    subtype: str | None = None
    agent: str | None = None
    confidence: float = 1.0
    rationale: str = ''
    evidence_steps: tuple[int, ...] = ()
    source: str = 'fa'

    def __post_init__(self):
        fault = normalize(self.category, self.subtype)
        object.__setattr__(self, 'category', fault.category)
        object.__setattr__(self, 'subtype', fault.subtype)
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise SpecError(f'confidence must be in [0, 1], got {self.confidence!r}')
        if self.step_index is not None and type(self.step_index) is not int:
            raise SpecError(f'step_index must be an integer, got {self.step_index!r}')

    @property
    def fault(self) -> FaultType:
        return FaultType(self.category, self.subtype)

    @classmethod
    def from_mapping(cls, data: Mapping) -> 'Hypothesis':
        """Accepts an FA prediction, an orchestrator ``top_k`` item or a benchmark hypothesis."""
        label = data.get('category', data.get('failure_case', data.get('mode')))
        subtype = data.get('subtype')
        if subtype is None and 'mode' in data and 'category' in data:
            subtype = data['mode']
        step = data.get('step_index', data.get('index'))
        refs = data.get('evidence_refs') or ()
        steps = tuple(sorted({int(r['step_index'] if isinstance(r, Mapping) else r[0]) for r in refs}))
        confidence = data.get('confidence', data.get('share', 1.0))
        return cls(step_index=int(step) if step is not None else None, category=label, subtype=subtype,
                   agent=normalize_agent(data.get('agent', data.get('failed_agent'))),
                   confidence=float(confidence if confidence is not None else 1.0),
                   rationale=str(data.get('rationale', data.get('reason_for_failure', '')) or ''),
                   evidence_steps=steps, source=str(data.get('source', 'fa')))

    @classmethod
    def from_object(cls, candidate) -> 'Hypothesis':
        """Duck-typed conversion of the orchestrator's ``Candidate``."""
        return cls.from_mapping({
            'step_index': getattr(candidate, 'step_index', None), 'category': getattr(candidate, 'category'),
            'agent': getattr(candidate, 'agent', None),
            'confidence': getattr(candidate, 'share', getattr(candidate, 'confidence', 1.0)),
            'rationale': getattr(candidate, 'rationale', ''),
            'evidence_refs': list(getattr(candidate, 'evidence_refs', ()) or ()),
            'source': getattr(candidate, 'source', 'fa')})

    def to_dict(self) -> dict:
        return {'step_index': self.step_index, 'category': self.category, 'subtype': self.subtype,
                'agent': self.agent, 'confidence': self.confidence, 'rationale': self.rationale,
                'evidence_steps': list(self.evidence_steps), 'source': self.source}


# ---------------------------------------------------------------------- task spec
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str = ''
    parameters: Mapping = field(default_factory=lambda: {'type': 'object', 'properties': {}})
    effect: str = READ

    def __post_init__(self):
        if not self.name:
            raise SpecError('Tool spec needs a name')
        if self.effect not in (READ, WRITE):
            raise SpecError(f'{self.name}: effect must be "read" or "write"')
        if not isinstance(self.parameters, Mapping):
            raise SpecError(f'{self.name}: parameters must be a JSON schema object')

    @property
    def properties(self) -> Mapping:
        return self.parameters.get('properties', {}) or {}

    @property
    def required(self) -> tuple[str, ...]:
        return tuple(self.parameters.get('required', ()) or ())

    @classmethod
    def from_schema(cls, schema: Mapping, write_tools=frozenset()) -> 'ToolSpec':
        fn = schema.get('function') if isinstance(schema.get('function'), Mapping) else schema
        name = fn.get('name')
        effect = fn.get('effect') or schema.get('effect') or (WRITE if name in write_tools else READ)
        return cls(name=name, description=fn.get('description', ''),
                   parameters=fn.get('parameters') or {'type': 'object', 'properties': {}}, effect=effect)

    def to_dict(self) -> dict:
        return {'name': self.name, 'description': self.description, 'effect': self.effect,
                'parameters': json.loads(json.dumps(self.parameters))}


def _frozen(mapping) -> Mapping:
    return MappingProxyType(dict(mapping or {}))


@dataclass(frozen=True)
class TaskSpec:
    """What a strategy may know about the task. Never contains gold actions or rewards."""
    instruction: str = ''
    policy: str = ''
    tools: Mapping[str, ToolSpec] = field(default_factory=dict)
    agent_tools: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    equivalents: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    rules: tuple[Mapping, ...] = ()
    compensations: Mapping[str, Mapping] = field(default_factory=dict)
    safe_defaults: Mapping[str, Mapping] = field(default_factory=dict)
    fact_keywords: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    name: str = 'custom'

    def __post_init__(self):
        object.__setattr__(self, 'tools', _frozen(self.tools))
        object.__setattr__(self, 'agent_tools', _frozen({a: tuple(t) for a, t in dict(self.agent_tools).items()}))
        object.__setattr__(self, 'equivalents', _frozen({a: tuple(t) for a, t in dict(self.equivalents).items()}))
        object.__setattr__(self, 'rules', tuple(self.rules))
        object.__setattr__(self, 'compensations', _frozen(self.compensations))
        object.__setattr__(self, 'safe_defaults', _frozen(self.safe_defaults))
        object.__setattr__(self, 'fact_keywords',
                           _frozen({k: tuple(v) for k, v in dict(self.fact_keywords).items()}))
        for name, spec in self.tools.items():
            if name != spec.name:
                raise SpecError(f'Tool key {name!r} does not match spec name {spec.name!r}')
        for tool, alternatives in self.equivalents.items():
            unknown = [t for t in (tool, *alternatives) if self.tools and t not in self.tools]
            if unknown:
                raise SpecError(f'Equivalents name unknown tools: {unknown}')

    # ------------------------------------------------------------------ queries
    @property
    def write_tools(self) -> frozenset[str]:
        return frozenset(n for n, s in self.tools.items() if s.effect == WRITE)

    def tool(self, name: str) -> ToolSpec | None:
        return self.tools.get(name)

    def is_write(self, name: str) -> bool:
        spec = self.tools.get(name)
        return spec is not None and spec.effect == WRITE

    def tools_of(self, agent: str | None) -> tuple[str, ...]:
        if agent is None:
            return ()
        return self.agent_tools.get(normalize_agent(agent) or '', ())

    def alternatives(self, tool: str) -> tuple[str, ...]:
        return self.equivalents.get(tool, ())

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_dict(cls, data: Mapping) -> 'TaskSpec':
        raw_tools = data.get('tools', [])
        if isinstance(raw_tools, Mapping):
            raw_tools = list(raw_tools.values())
        write = frozenset(data.get('write_tools', ()))
        tools = {}
        for raw in raw_tools:
            spec = ToolSpec.from_schema(raw, write)
            if spec.name in tools:
                raise SpecError(f'Duplicate tool {spec.name!r}')
            tools[spec.name] = spec
        unknown_write = write - set(tools)
        if unknown_write and tools:
            raise SpecError(f'write_tools names unknown tools: {sorted(unknown_write)}')
        return cls(instruction=data.get('instruction', ''), policy=data.get('policy', ''), tools=tools,
                   agent_tools=data.get('agent_tools', {}), equivalents=data.get('equivalents', {}),
                   rules=tuple(data.get('rules', ())), compensations=data.get('compensations', {}),
                   safe_defaults=data.get('safe_defaults', {}), fact_keywords=data.get('fact_keywords', {}),
                   name=data.get('name', 'custom'))

    @classmethod
    def load(cls, path) -> 'TaskSpec':
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))

    @classmethod
    def preset(cls, name: str) -> 'TaskSpec':
        resource = files('repair_strategies').joinpath(f'resources/{name}.json')
        if not resource.is_file():
            raise SpecError(f'Unknown preset {name!r}')
        return cls.from_dict(json.loads(resource.read_text(encoding='utf-8')))

    def with_task(self, instruction: str | None = None, policy: str | None = None) -> 'TaskSpec':
        """Same tools and rules, for a specific task instance."""
        return TaskSpec(instruction=self.instruction if instruction is None else instruction,
                        policy=self.policy if policy is None else policy, tools=dict(self.tools),
                        agent_tools=dict(self.agent_tools), equivalents=dict(self.equivalents), rules=self.rules,
                        compensations=dict(self.compensations), safe_defaults=dict(self.safe_defaults),
                        fact_keywords=dict(self.fact_keywords), name=self.name)

    def to_dict(self) -> dict:
        return {'name': self.name, 'instruction': self.instruction, 'policy': self.policy,
                'tools': [t.to_dict() for t in self.tools.values()],
                'agent_tools': {a: list(t) for a, t in self.agent_tools.items()},
                'equivalents': {a: list(t) for a, t in self.equivalents.items()},
                'rules': [dict(r) for r in self.rules], 'compensations': dict(self.compensations),
                'safe_defaults': dict(self.safe_defaults),
                'fact_keywords': {k: list(v) for k, v in self.fact_keywords.items()}}


# ---------------------------------------------------------------------- attempt history
VERIFIED, FAILED, REJECTED = 'verified', 'failed', 'rejected'


@dataclass(frozen=True)
class AttemptRecord:
    """A plan that was already tried on this run, and what happened."""
    strategy_id: str
    category: str
    rollback_step: int | None
    outcome: str                 # verified | failed (replayed, not accepted) | rejected (not executed)

    def __post_init__(self):
        if self.outcome not in (VERIFIED, FAILED, REJECTED):
            raise SpecError(f'Unknown attempt outcome {self.outcome!r}')
