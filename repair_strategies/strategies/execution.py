"""Invalid invocations, loops and system failures: repairs at the tool-call level."""
from __future__ import annotations

import json

from ..evidence import cycles, schema
from ..evidence.tool_output import indicates_drift, is_error
from ..memory.tool_outcomes import ABSTAIN, RETRY, ToolOutcomeMemory
from ..plan import BanCalls, GuardCalls, InjectMessage, Resample, Retry, SwitchTool
from ..taxonomy import CYCLE, INVALID_INVOCATION, SYSTEM_FAILURE
from .base import RepairContext, Skip, Strategy, StrategySpec, check, compact, failed_exchange, grounded, record


def _target_call(ctx: RepairContext):
    """The invalid call: the one at the attributed step, else the latest failed or invalid call before it."""
    step = ctx.agent_step()
    if step is None:
        return None
    for event in ctx.trace.at(step):
        for call in event.tool_calls:
            return event, call
    failed = failed_exchange(ctx.exchanges, ctx.step)
    if failed is not None:
        return ctx.trace.primary(failed.step_index), failed.call
    return None


def _result_of(ctx: RepairContext, step: int, call):
    for x in ctx.exchanges:
        if x.step_index == step and x.call == call:
            return x.result
    return None


class HardCallRepair(Strategy):
    spec = StrategySpec(
        id='hard_call_repair', name='Block the invalid call and request the corrected one',
        categories=(INVALID_INVOCATION,), rung=1, kind='retry', tokens=400, prior_success=0.7,
        description='Validates the call against the tool schema, returns the exact violations as a tool error and '
                    'asks for the corrected call only; invalid calls are blocked before execution on replay.')

    def plan(self, ctx: RepairContext):
        found = _target_call(ctx)
        if found is None:
            return Skip('no tool call at or before the attributed step')
        event, call = found
        allowed = ctx.task.tools_of(event.actor) or None
        issues = schema.validate_call(call, ctx.task, allowed)
        result = _result_of(ctx, event.step_index, call)
        error = result.content if result is not None and is_error(result.content) else ''
        if not issues and not error:
            return Skip(f'{call.name} at step {event.step_index} is schema-valid and did not fail')
        lines = [i.describe() for i in issues] or [f'the tool rejected it: {compact(error, 200)}']
        target = call.name if ctx.task.tool(call.name) else (issues[0].suggestion if issues else None)
        if target and ctx.task.tool(target):
            example = json.dumps({'name': target, 'arguments': schema.example_arguments(ctx.task.tool(target))})
            lines.append(f'required format: {example}')
        body = (f'Call rejected before execution: {call.name}({compact(call.arguments, 200)})\n'
                + '\n'.join(f'- {line}' for line in lines)
                + '\nSend only the corrected call, using values from the conversation.')
        operations = [GuardCalls(validate_schema=True, allowed_tools=tuple(allowed or ())),
                      InjectMessage('tool', body, name=call.name)]
        acceptance = [check('calls_valid'), check('no_error_results', tools=[target or call.name])]
        return self.build(ctx, event.step_index, operations, acceptance,
                          f'Validate {call.name} arguments against its schema before calling it.',
                          lines)


class SwitchOnSchemaDrift(Strategy):
    spec = StrategySpec(
        id='switch_on_schema_drift', name='Switch to an equivalent tool on schema drift',
        categories=(INVALID_INVOCATION,), rung=2, kind='retry', tokens=400, prior_success=0.5,
        description='When the error says the tool or its arguments changed (deprecated, unknown argument), routes '
                    'the call to an equivalent tool.')

    def plan(self, ctx: RepairContext):
        found = _target_call(ctx)
        if found is None:
            return Skip('no tool call at or before the attributed step')
        event, call = found
        result = _result_of(ctx, event.step_index, call)
        unknown = ctx.task.tool(call.name) is None
        if not unknown and not (result is not None and indicates_drift(result.content)):
            return Skip('the failure does not indicate schema drift')
        alternatives = ctx.task.alternatives(call.name)
        if unknown:
            suggestion = schema.suggest(call.name, ctx.task.tools_of(event.actor) or ctx.task.tools)
            alternatives = (suggestion,) if suggestion else ()
        if not alternatives:
            return Skip(f'{call.name} has no equivalent tool')
        to = alternatives[0]
        example = json.dumps(schema.example_arguments(ctx.task.tool(to)))
        operations = [SwitchTool(call.name, to, 'schema drift'),
                      InjectMessage('tool', f'{call.name} is not usable as called'
                                            f'{": " + compact(result.content, 160) if result is not None else ""}. '
                                            f'Use {to} instead with arguments {example}.', name=call.name),
                      GuardCalls(validate_schema=True)]
        return self.build(ctx, event.step_index, operations,
                          (check('tool_succeeded', tool=to), check('calls_valid')),
                          f'Switch to {to} when {call.name} reports a changed interface.', [f'{call.name} -> {to}'])


# ---------------------------------------------------------------------- cycle ladder
def _loop(ctx: RepairContext):
    loop = cycles.loop_at(ctx.trace, ctx.step, ctx.task.write_tools)
    return loop or cycles.loop_at(ctx.trace, ctx.step, ctx.task.write_tools, min_repeats=2)


def _loop_call(ctx: RepairContext, loop):
    if loop.kind != cycles.CALL:
        return None
    for event in ctx.trace.at(loop.start):
        for call in event.tool_calls:
            if call.signature() == loop.signature:
                return call
    return None


class _LoopRung(Strategy):
    def plan(self, ctx: RepairContext):
        loop = _loop(ctx)
        if loop is None:
            return Skip('no repeated call, handoff or message found')
        call = _loop_call(ctx, loop)
        rollback = self.rollback(ctx, loop)
        if rollback is None:
            return Skip('no agent step to roll back to for this rung')
        signatures = [loop.signature] if call is not None else []
        acceptance = [check('no_repeat', signatures=signatures)]
        return self.build(ctx, rollback, self.operations(ctx, loop, call), acceptance, self.rationale, [loop.describe()])

    def rollback(self, ctx, loop):
        return ctx.trace.rollback_point(loop.first_repeat)

    def _note(self, ctx, loop, call) -> str:
        if call is not None:
            result = next((x.result.content for x in ctx.exchanges if x.step_index == loop.start and x.call == call
                           and x.result is not None), '')
            return (f'{call.name}({compact(call.arguments, 120)}) already returned the same result '
                    f'{len(loop.steps)} times: {compact(result, 160)}')
        return loop.describe()


class LoopNudge(_LoopRung):
    spec = StrategySpec(id='loop_nudge', name='Point out the repetition', categories=(CYCLE,), rung=1,
                        kind='loop_break', tokens=200, prior_success=0.45,
                        description='Tells the agent the call already returned the same result and repeating it '
                                    'will not change the outcome.')
    rationale = 'Stop repeating a call that keeps returning the same result; use it or change approach.'

    def operations(self, ctx, loop, call):
        return [InjectMessage('system', self._note(ctx, loop, call) + '. Repeating it will not change the outcome: '
                                                                       'use the result you have or take a different action.')]


class LoopForceChange(_LoopRung):
    spec = StrategySpec(id='loop_force_change', name='Ban the repeated call and force a different approach',
                        categories=(CYCLE,), rung=2, kind='loop_break', tokens=300, prior_success=0.5,
                        description='Blocks the repeated call and lists the other actions available to the agent.')
    rationale = 'When a call loops, block it and choose a different tool or report the problem.'

    def operations(self, ctx, loop, call):
        actor = ctx.trace.primary(loop.start).actor
        others = [t for t in ctx.task.tools_of(actor) if call is None or t != call.name]
        options = f' Available alternatives: {", ".join(others)}.' if others else ''
        ops = [InjectMessage('system', self._note(ctx, loop, call) + '. This exact action is now blocked.' + options +
                             ' If none applies, tell the user what is blocking the request.')]
        if call is not None:
            ops.insert(0, BanCalls((loop.signature,), 'repeated without progress'))
        return ops


class LoopReplan(_LoopRung):
    spec = StrategySpec(id='loop_replan', name='Replan from before the loop', categories=(CYCLE,), rung=3,
                        kind='loop_break', tokens=800, prior_success=0.4,
                        description='Rolls back to where the loop started and regenerates in a fresh context with the '
                                    'repeated call banned.')
    rationale = 'Plan a different approach from the point where the loop started.'

    def rollback(self, ctx, loop):
        return ctx.trace.rollback_point(loop.start)

    def operations(self, ctx, loop, call):
        ops = [Resample(fresh_context=True, seed_offset=1),
               InjectMessage('system', f'A previous attempt looped: {self._note(ctx, loop, call)}. Plan a different '
                                       'approach.')]
        if call is not None:
            ops.insert(1, BanCalls((loop.signature,), 'looped in an earlier attempt'))
        return ops


class LoopRollback(LoopReplan):
    spec = StrategySpec(id='loop_rollback', name='Roll back to the last user turn', categories=(CYCLE,), rung=4,
                        kind='loop_break', tokens=1000, prior_success=0.3,
                        description='Restarts the agent turn that led into the loop, from the last user message.')
    rationale = 'Restart the turn that led into a loop instead of continuing it.'

    def rollback(self, ctx, loop):
        last_user = ctx.trace.last_user_step_before(loop.start)
        step = ctx.trace.next_agent_step(last_user + 1) if last_user is not None else ctx.trace.next_agent_step(1)
        return None if step == ctx.trace.rollback_point(loop.start) else step


# ---------------------------------------------------------------------- system failure
class ToolOutcomeRouting(Strategy):
    spec = StrategySpec(
        id='tool_outcome_routing', name='Retry, switch or abstain by tool reliability',
        categories=(SYSTEM_FAILURE,), rung=1, kind='retry', tokens=200, prior_success=0.55,
        description='Classifies the error as transient or persistent and uses per-tool outcome memory to retry '
                    'with backoff, switch to an equivalent tool, or abstain.')

    def plan(self, ctx: RepairContext):
        failed = failed_exchange(ctx.exchanges, ctx.step)
        if failed is None:
            return self._silent(ctx)
        memory = ctx.tool_memory.copy() if ctx.tool_memory is not None else ToolOutcomeMemory()
        memory.observe(ctx.trace)                       # on a copy: planning never mutates the shared memory
        actor = ctx.trace.primary(failed.step_index).actor
        alternatives = [t for t in ctx.task.alternatives(failed.call.name)
                        if not ctx.task.tools_of(actor) or t in ctx.task.tools_of(actor)]
        decision = memory.decide(failed.call.name, failed.result.content, alternatives, ctx.attempts(self.spec.id))
        evidence = [f'{failed.call.name} at step {failed.step_index}: {compact(failed.result.content, 160)}',
                    decision.reason]
        if decision.action == ABSTAIN:
            # No retry or equivalent tool is worth trying: leave it to the fallback (compensate, escalate).
            return Skip(decision.reason)
        if decision.action == RETRY:
            operations = [Retry(failed.call.name, dict(failed.call.arguments), decision.backoff_s),
                          InjectMessage('tool', f'The previous {failed.call.name} call failed with a transient error '
                                                f'({compact(failed.result.content, 120)}). It is being retried; '
                                                'check the new result before continuing.', name=failed.call.name)]
            target = failed.call.name
        else:
            example = json.dumps(schema.example_arguments(ctx.task.tool(decision.tool)))
            operations = [SwitchTool(failed.call.name, decision.tool, decision.reason),
                          InjectMessage('tool', f'{failed.call.name} is failing ({decision.reason}). Use '
                                                f'{decision.tool} with arguments {example}.', name=failed.call.name)]
            target = decision.tool
        acceptance = (check('no_error_results', tools=sorted({failed.call.name, target})),)
        return self.build(ctx, failed.step_index, operations, acceptance,
                          f'On a tool failure, {decision.action} based on how reliable the tool has been.', evidence)

    def _silent(self, ctx: RepairContext):
        """No explicit error: treat as a silent failure and re-query the output the agent relied on."""
        step = ctx.agent_step()
        exchange = ctx.consumed_exchange(step, requery=True) if step is not None else None
        if exchange is None or ctx.task.is_write(exchange.call.name):
            return Skip('no failed or re-queryable tool call before the attributed step')
        body = record(f'a check of {exchange.call.name}', [
            f'the output at step {exchange.result.step_index} may be stale or wrong',
            'a fresh result follows; compare it with the earlier one and use the fresh values'])
        operations = [Retry(exchange.call.name, dict(exchange.call.arguments)), InjectMessage('tool', body,
                                                                                          name=exchange.call.name)]
        return self.build(ctx, ctx.trace.rollback_point(exchange.step_index), operations,
                          (grounded(ctx, step),),
                          f'Re-query {exchange.call.name} when its output may be silently wrong.',
                          [f'silent failure suspected in {exchange.call.name}'], confidence_factor=0.7)
