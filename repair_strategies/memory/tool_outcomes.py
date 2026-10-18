"""Per-tool outcome memory: how reliable is each tool, and is retrying worth it?

Each tool has a Beta(1 + successes, 1 + failures) posterior over its success
rate, plus counts of transient and persistent errors. A failed call is then
routed to retry (transient error, reliable tool, retries left), switch (an
equivalent tool that is more reliable) or abstain.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from ..evidence.tool_output import PERSISTENT, TRANSIENT, error_class, is_error
from ..trace import Trace
from .store import read_json, write_json_atomic

RETRY, SWITCH, ABSTAIN = 'retry', 'switch', 'abstain'


@dataclass
class ToolStats:
    successes: int = 0
    failures: int = 0
    transient: int = 0
    persistent: int = 0

    def success_rate(self, prior_a: float = 1.0, prior_b: float = 1.0) -> float:
        return (prior_a + self.successes) / (prior_a + prior_b + self.successes + self.failures)


@dataclass(frozen=True)
class RoutingDecision:
    action: str                 # retry | switch | abstain
    tool: str                   # the tool to call (the same one for retry)
    reason: str
    backoff_s: float = 0.0


class ToolOutcomeMemory:
    def __init__(self, path=None, prior=(1.0, 1.0), max_retries: int = 2, min_retry_rate: float = 0.3,
                 base_backoff_s: float = 1.0):
        self.path = path
        self.prior_a, self.prior_b = prior
        self.max_retries, self.min_retry_rate, self.base_backoff_s = max_retries, min_retry_rate, base_backoff_s
        self.stats: dict[str, ToolStats] = {}
        self.observed: set[str] = set()
        if path is not None:
            data = read_json(path, {})
            self.stats = {k: ToolStats(**v) for k, v in data.get('tools', {}).items()}
            self.observed = set(data.get('observed', ()))

    def copy(self) -> 'ToolOutcomeMemory':
        """An in-memory copy (no path), so planning can simulate observations without side effects."""
        clone = ToolOutcomeMemory(None, (self.prior_a, self.prior_b), self.max_retries, self.min_retry_rate,
                                  self.base_backoff_s)
        clone.stats = {k: ToolStats(**asdict(v)) for k, v in self.stats.items()}
        clone.observed = set(self.observed)
        return clone

    def record(self, tool: str, ok: bool, error_text: str = '') -> None:
        stats = self.stats.setdefault(tool, ToolStats())
        if ok:
            stats.successes += 1
            return
        stats.failures += 1
        kind = error_class(error_text)
        if kind == TRANSIENT:
            stats.transient += 1
        elif kind == PERSISTENT:
            stats.persistent += 1

    def observe(self, trace: Trace) -> int:
        """Record every answered call of a trace once (keyed by trajectory id). Returns calls recorded."""
        key = f'{trace.domain}/{trace.trajectory_id}'
        if trace.trajectory_id and key in self.observed:
            return 0
        count = 0
        for x in trace.tool_exchanges():
            if x.result is not None:
                self.record(x.call.name, not is_error(x.result.content), x.result.content)
                count += 1
        if trace.trajectory_id:
            self.observed.add(key)
        return count

    def success_rate(self, tool: str) -> float:
        return self.stats.get(tool, ToolStats()).success_rate(self.prior_a, self.prior_b)

    def decide(self, tool: str, error_text: str, alternatives=(), attempts: int = 0) -> RoutingDecision:
        kind = error_class(error_text)
        rate = self.success_rate(tool)
        best = max(alternatives, key=lambda t: (self.success_rate(t), t), default=None)
        best_rate = self.success_rate(best) if best else 0.0
        if kind == PERSISTENT:
            if best:
                return RoutingDecision(SWITCH, best, f'{tool} failed persistently; {best} is equivalent '
                                                     f'(success rate {best_rate:.2f})')
            return RoutingDecision(ABSTAIN, tool, f'{tool} failed persistently and has no equivalent tool')
        if attempts < self.max_retries and rate >= self.min_retry_rate:
            backoff = self.base_backoff_s * (2 ** attempts) if kind == TRANSIENT else 0.0
            return RoutingDecision(RETRY, tool, f'{kind} error; {tool} succeeds {rate:.2f} of the time', backoff)
        if best and best_rate > rate:
            return RoutingDecision(SWITCH, best, f'{tool} is unreliable ({rate:.2f}); {best} succeeds {best_rate:.2f}')
        return RoutingDecision(ABSTAIN, tool, f'{tool} keeps failing ({rate:.2f}, {attempts} retries) and no better '
                                              f'equivalent exists')

    def to_dict(self) -> dict:
        return {'tools': {k: asdict(v) for k, v in sorted(self.stats.items())}, 'observed': sorted(self.observed)}

    def save(self, path=None) -> None:
        target = path or self.path
        if target is None:
            raise ValueError('No path to save the tool outcome memory to')
        write_json_atomic(target, self.to_dict())
