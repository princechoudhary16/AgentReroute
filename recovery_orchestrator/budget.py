"""Per-task budget caps and loop guards."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BudgetCaps:
    max_tokens: int = 200_000          # repair + replay + verifier tokens
    max_seconds: float = 900.0
    max_attempts: int = 3              # repair attempts (rollback + replay + verify)
    max_rollbacks: int = 3             # distinct rollback points


class BudgetExceeded(Exception):
    pass


@dataclass
class BudgetManager:
    caps: BudgetCaps = field(default_factory=BudgetCaps)
    clock: callable = time.monotonic
    tokens: int = 0
    attempts: int = 0
    rollback_points: set = field(default_factory=set)
    verifier_tokens: int = 0

    def __post_init__(self):
        self._start = self.clock()

    @property
    def elapsed(self) -> float:
        return self.clock() - self._start

    def charge(self, tokens: int, verifier: bool = False) -> None:
        self.tokens += int(tokens)
        if verifier:
            self.verifier_tokens += int(tokens)

    def remaining_tokens(self) -> int:
        return max(0, self.caps.max_tokens - self.tokens)

    def exhausted(self) -> str | None:
        """Reason the budget is used up, or None."""
        if self.tokens >= self.caps.max_tokens:
            return 'tokens'
        if self.elapsed >= self.caps.max_seconds:
            return 'time'
        if self.attempts >= self.caps.max_attempts:
            return 'attempts'
        return None

    def can_afford(self, rollback_step: int, est_tokens: int) -> str | None:
        """Reason a new attempt at ``rollback_step`` is not allowed, or None."""
        reason = self.exhausted()
        if reason:
            return reason
        if rollback_step not in self.rollback_points and len(self.rollback_points) >= self.caps.max_rollbacks:
            return 'rollbacks'
        if est_tokens > self.remaining_tokens():
            return 'tokens'
        return None

    def start_attempt(self, rollback_step: int) -> None:
        self.attempts += 1
        self.rollback_points.add(rollback_step)

    def summary(self) -> dict:
        return {'tokens': self.tokens, 'verifier_tokens': self.verifier_tokens,
                'seconds': round(self.elapsed, 3), 'attempts': self.attempts,
                'rollbacks': len(self.rollback_points)}


@dataclass
class LoopGuard:
    """Never repeat a strategy on the same step; stop when a repair makes things worse."""
    allow_repeats: bool = False        # the retry-only baseline sets this
    stop_on_worse: bool = True
    tried: set = field(default_factory=set)
    tried_list: list = field(default_factory=list)     # every attempt, repeats included
    scores: list = field(default_factory=list)

    def allowed(self, rollback_step: int, strategy_id: str) -> bool:
        return self.allow_repeats or (rollback_step, strategy_id) not in self.tried

    def record(self, rollback_step: int, strategy_id: str, score: float | None,
               reference: float | None) -> str | None:
        """Register an attempt. Returns 'worsened' if the guard says stop."""
        self.tried.add((rollback_step, strategy_id))
        if score is None and reference is None:
            return None             # bookkeeping only: marks the pair tried, not an attempt
        self.tried_list.append((rollback_step, strategy_id))
        self.scores.append(score)
        if self.stop_on_worse and score is not None and reference is not None and score < reference:
            return 'worsened'
        return None
