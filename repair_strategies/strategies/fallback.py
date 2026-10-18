"""The last resort when no repair works: roll back side effects, then escalate.

Used for every fault category once its ladder is exhausted or nothing in it
applies. It never replays the run again. Instead it restores state by
compensating the writes made from the fault onward (writes before the fault are
the user's completed work and are kept), lists what cannot be undone, and hands
the case to a human with a record of everything that was tried.
"""
from __future__ import annotations

from ..context import FAILED, REJECTED
from ..evidence.tool_output import get_path, is_error, parse_output
from ..plan import Compensate, Escalate
from ..taxonomy import CATEGORIES
from ..trace import ToolExchange
from .base import RepairContext, Strategy, StrategySpec

ESCALATION_TOOLS = frozenset({'transfer_to_human_agents'})


def executed_writes(ctx: RepairContext, since: int | None = None) -> list[ToolExchange]:
    """Writes that succeeded, at or after ``since`` (all of them when ``since`` is None)."""
    return [x for x in ctx.exchanges if ctx.task.is_write(x.call.name) and x.call.name not in ESCALATION_TOOLS
            and x.result is not None and not is_error(x.result.content)
            and (since is None or x.step_index >= since)]


def inverse_call(ctx: RepairContext, exchange: ToolExchange) -> tuple[dict | None, str]:
    """The compensating call for an executed write, or None and the reason it cannot be undone."""
    if exchange.orphan:
        return None, 'the call arguments were not logged'
    spec = ctx.task.compensations.get(exchange.call.name)
    if not spec:
        return None, 'no inverse action exists'
    restore = spec.get('restore') or {}
    source = None
    for x in reversed(ctx.exchanges):
        if x.step_index >= exchange.step_index or x.orphan or x.call.name != restore.get('tool') \
                or x.result is None:
            continue
        match = restore.get('match') or {}
        if all(x.call.arguments.get(a) == exchange.call.arguments.get(b) for a, b in match.items()):
            parsed = parse_output(x.result.content)
            if parsed.ok and parsed.data is not None:
                source = parsed.data
                break
    if source is None:
        return None, f'the value before the change was never read ({restore.get("tool")})'
    arguments = {k: exchange.call.arguments[k] for k in spec.get('keep', ()) if k in exchange.call.arguments}
    for name, path in (restore.get('fields') or {}).items():
        value = get_path(source, path)
        if value is None:
            return None, f'previous value of {name} is unknown'
        arguments[name] = value
    return {'tool': spec['inverse'], 'arguments': arguments, 'undoes_step': exchange.step_index}, ''


def compensation(ctx: RepairContext, since: int | None = None):
    """(writes, compensating calls, irreversible items); the calls undo the latest write first."""
    writes = executed_writes(ctx, since)
    calls, irreversible = [], []
    for x in reversed(writes):
        inverse, reason = inverse_call(ctx, x)
        if inverse is not None:
            calls.append(inverse)
        else:
            irreversible.append({'tool': x.call.name, 'step_index': x.step_index, 'reason': reason})
    irreversible.reverse()
    return writes, calls, irreversible


def state_summary(writes, calls, irreversible, kept=()) -> str:
    parts = []
    if kept:
        parts.append('Kept (before the fault): ' + ', '.join(f'{x.call.name} (step {x.step_index})' for x in kept) + '.')
    if not writes:
        parts.append('No data was changed after the fault.' if kept else 'No data was changed.')
        return ' '.join(parts)
    parts.append(f'Changed: {", ".join(f"{x.call.name} (step {x.step_index})" for x in writes)}.')
    if calls:
        parts.append('Reverted: ' + ', '.join(f'{c["tool"]} undoing step {c["undoes_step"]}' for c in calls) + '.')
    if irreversible:
        parts.append('Cannot be reverted: ' + ', '.join(f'{i["tool"]} at step {i["step_index"]} ({i["reason"]})'
                                                       for i in irreversible) + '.')
    return ' '.join(parts)


class RollbackCompensateEscalate(Strategy):
    spec = StrategySpec(
        id='rollback_compensate_escalate', name='Fallback: roll back side effects and escalate', categories=CATEGORIES,
        rung=1, kind='clarify', tokens=200, prior_success=0.0, reversible=False, fallback=True,
        description='When every applicable strategy has failed or none applies, compensates the writes made from the '
                    'fault onward, lists the ones that cannot be undone, and escalates to a human with the full '
                    'repair history.')

    def plan(self, ctx: RepairContext):
        since = ctx.agent_step()
        writes, calls, irreversible = compensation(ctx, since)
        kept = [x for x in executed_writes(ctx) if since is not None and x.step_index < since]
        tried = [r for r in ctx.history if r.outcome in (FAILED, REJECTED)]
        where = f' at step {ctx.step}' if ctx.step is not None else ''
        if tried:
            attempts = ', '.join(f'{r.strategy_id} ({r.outcome})' for r in tried)
            reason = f'No repair restored the run after {len(tried)} attempt(s) for {ctx.hypothesis.category}{where}.'
            history = f'Tried: {attempts}.'
        else:
            reason = f'No repair strategy applies to {ctx.hypothesis.category}{where}.'
            history = 'No repair was attempted.'
        diagnosis = f'Diagnosis: {ctx.hypothesis.rationale.strip()}' if ctx.hypothesis.rationale.strip() else ''
        summary = ' '.join(p for p in (history, state_summary(writes, calls, irreversible, kept), diagnosis) if p)
        operations = []
        if calls or irreversible:
            operations.append(Compensate(tuple(calls), tuple(irreversible)))
        operations.append(Escalate(reason, summary))
        return self.build(ctx, None, operations, (),
                          'When repairs fail, restore the state changed after the fault and escalate with the history.',
                          [summary])
