"""Strategy registry: the contract with the Repair team.

One entry per repair strategy (schema in ``resources/registry_schema.md``).
Priors are fixed for now; learning them from verified outcomes (e.g. Thompson
sampling) is future work.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

from .taxonomy import CATEGORIES, CYCLE, normalize_category

SCHEMA_VERSION = 1
KINDS = ('reasoning_repair', 'retry', 'clarify', 'loop_break')
_STRATEGY_KEYS = {'id', 'name', 'kind', 'targets', 'cost', 'prior_success', 'reversible',
                  'description', 'template'}


@dataclass(frozen=True)
class Strategy:
    id: str
    name: str
    kind: str
    targets: frozenset[str]            # canonical category names; '*' = any repairable category
    tokens: int                        # estimated extra tokens to apply the repair (not the replay)
    latency_s: float
    prior_success: float               # P(verified recovery | correct attribution, category targeted)
    reversible: bool = True            # False: applying it has side effects outside the replay
    description: str = ''
    template: str = ''                 # guidance text; '{rationale}', '{step_index}' ... are filled in
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    def targets_category(self, category: str) -> bool:
        return category in self.targets or ('*' in self.targets and category != CYCLE)


class RegistryError(ValueError):
    pass


class StrategyRegistry:
    def __init__(self, strategies: list[Strategy], source: str = ''):
        ids = [s.id for s in strategies]
        if len(ids) != len(set(ids)):
            raise RegistryError('Duplicate strategy ids')
        self.strategies = list(strategies)
        self.source = source

    def __iter__(self):
        return iter(self.strategies)

    def __len__(self):
        return len(self.strategies)

    def get(self, strategy_id: str) -> Strategy:
        for s in self.strategies:
            if s.id == strategy_id:
                return s
        raise KeyError(strategy_id)

    def for_category(self, category: str, kind: str | None = None) -> list[Strategy]:
        """Strategies targeting ``category``, best prior first."""
        found = [s for s in self.strategies if s.targets_category(category)
                 and (kind is None or s.kind == kind)]
        return sorted(found, key=lambda s: (-s.prior_success, s.tokens, s.id))

    @classmethod
    def from_dict(cls, data: dict, source: str = '') -> 'StrategyRegistry':
        if data.get('schema_version') != SCHEMA_VERSION:
            raise RegistryError(f'Unsupported registry schema_version: {data.get("schema_version")!r}')
        return cls([parse_strategy(item) for item in data.get('strategies', [])], source)

    @classmethod
    def load(cls, path) -> 'StrategyRegistry':
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')), str(path))


def parse_strategy(item: dict) -> Strategy:
    unknown = set(item) - _STRATEGY_KEYS
    if unknown:
        raise RegistryError(f'Unknown strategy fields: {sorted(unknown)}')
    try:
        sid, kind = str(item['id']), item['kind']
        targets = item['targets']
        prior = float(item['prior_success'])
        cost = item.get('cost', {})
    except KeyError as exc:
        raise RegistryError(f'Strategy missing field {exc}') from None
    if kind not in KINDS:
        raise RegistryError(f'{sid}: kind must be one of {KINDS}')
    if not 0.0 <= prior <= 1.0:
        raise RegistryError(f'{sid}: prior_success must be in [0, 1]')
    if not isinstance(targets, list) or not targets:
        raise RegistryError(f'{sid}: targets must be a non-empty list')
    canonical = set()
    for target in targets:
        if target == '*':
            canonical.add('*')
            continue
        try:
            canonical.add(normalize_category(target))
        except ValueError as exc:
            raise RegistryError(f'{sid}: {exc}') from None
    if 'Inconclusive' in canonical:
        raise RegistryError(f'{sid}: Inconclusive is not a repair target')
    tokens, latency = int(cost.get('tokens', 0)), float(cost.get('latency_s', 0.0))
    if tokens < 0 or latency < 0:
        raise RegistryError(f'{sid}: costs must be non-negative')
    return Strategy(id=sid, name=item.get('name', sid), kind=kind, targets=frozenset(canonical),
                    tokens=tokens, latency_s=latency, prior_success=prior,
                    reversible=bool(item.get('reversible', True)),
                    description=item.get('description', ''), template=item.get('template', ''))


def _bundled(name: str) -> StrategyRegistry:
    text = files('recovery_orchestrator').joinpath(f'resources/{name}.json').read_text(encoding='utf-8')
    return StrategyRegistry.from_dict(json.loads(text), name)


def mock_registry() -> StrategyRegistry:
    """Small placeholder registry with the agreed schema; unit tests depend on it."""
    return _bundled('mock_registry')


def repair_registry() -> StrategyRegistry:
    """Snapshot of the Repair team's exported registry (see the file's ``_note``)."""
    return _bundled('repair_registry')


def default_registry() -> StrategyRegistry:
    """The Repair team's snapshot when bundled, else the mock."""
    try:
        return repair_registry()
    except FileNotFoundError:
        return mock_registry()


__all__ = ['Strategy', 'StrategyRegistry', 'RegistryError', 'mock_registry', 'repair_registry',
           'default_registry', 'parse_strategy',
           'CATEGORIES', 'KINDS']
