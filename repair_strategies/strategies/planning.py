"""Intent-plan misalignment and instruction/plan adherence.

Asking an agent to fix a misunderstanding it already acted on rarely works.
For misalignment the run is rolled back to before the plan was formed, an
explicit intent contract is given in a fresh context, and every action is
validated against it. For adherence, policy is enforced by a pre-execution
gate and a persistent plan-state checklist rather than a reminder.
"""
from __future__ import annotations

from ..plan import GuardCalls, InjectMessage, Resample
from ..policy.intent import LLMContractExtractor, RuleBasedExtractor
from ..policy.plan_state import track
from ..taxonomy import INSTRUCTION_ADHERENCE, INTENT_MISALIGNMENT
from .base import RepairContext, Skip, Strategy, StrategySpec, check, compact, rule_specs

CONFIRM_KINDS = ('requires_confirmation',)


def plan_formation_step(ctx: RepairContext) -> int | None:
    """First agent step after the user turn that the attributed action responds to."""
    step = ctx.agent_step()
    if step is None:
        return None
    last_user = ctx.trace.last_user_step_before(step)
    if last_user is None:
        return ctx.trace.next_agent_step(1)
    return ctx.trace.next_agent_step(last_user + 1)


def abandoned_branch(ctx: RepairContext, rollback: int) -> str:
    """A short, distilled account of what the abandoned branch did after ``rollback``."""
    actions = []
    for event in ctx.trace.from_step(rollback):
        if not event.is_agent:
            continue
        for call in event.tool_calls:
            actions.append(f'{call.name}({compact(call.arguments, 80)})')
        if event.handoff:
            actions.append(f'handoff to {event.handoff["to"]}')
    if not actions:
        return ''
    reason = f' A reviewer found: {ctx.hypothesis.rationale.strip()}' if ctx.hypothesis.rationale.strip() else ''
    return f'An earlier attempt from this point did: {"; ".join(actions[:6])}. It did not achieve the goal.{reason}'


class IntentContractReplan(Strategy):
    spec = StrategySpec(
        id='intent_contract_replan', name='Replan against an explicit intent contract',
        categories=(INTENT_MISALIGNMENT,), rung=1, kind='reasoning_repair', tokens=1200, prior_success=0.5,
        description='Extracts the goal, entities in scope, required and forbidden actions; rolls back to before the '
                    'plan was formed; regenerates in a fresh context with a lesson from the abandoned branch; gates '
                    'every action against the contract.')

    def plan(self, ctx: RepairContext):
        rollback = plan_formation_step(ctx)
        if rollback is None:
            return Skip('no agent step to replan from')
        extractor = LLMContractExtractor(ctx.llm) if ctx.llm is not None else RuleBasedExtractor()
        contract = extractor.extract(ctx.trace, ctx.task, rollback)
        if not contract.goal:
            return Skip('no user request to build an intent contract from')
        lesson = abandoned_branch(ctx, rollback)
        guidance = 'Intent contract for this request:\n' + contract.render()
        if lesson:
            guidance += '\n' + lesson
        confirm = rule_specs(ctx.task, kinds=CONFIRM_KINDS)
        operations = [Resample(fresh_context=True, seed_offset=1), InjectMessage('system', guidance),
                      GuardCalls(rules=confirm, validate_schema=True, contract=contract.to_dict())]
        acceptance = [check('contract_respected', contract=contract.to_dict()), check('calls_valid')]
        if confirm:
            acceptance.append(check('rules_pass', rule_ids=[r['id'] for r in confirm]))
        evidence = [f'contract source: {contract.source}', *contract.notes]
        if lesson:
            evidence.append(lesson)
        return self.build(ctx, rollback, operations, acceptance,
                          'Restate the user goal as a contract and check every planned action against it before acting.',
                          evidence)


class ConfirmAmbiguousWrites(Strategy):
    spec = StrategySpec(
        id='confirm_ambiguous_writes', name='Confirm the target of a write with the user',
        categories=(INTENT_MISALIGNMENT,), rung=2, kind='clarify', tokens=400, prior_success=0.45,
        description='When the action changes data and its target is not the one the user named, has the agent '
                    'state the exact action and target and ask the user to confirm before executing it.')

    def plan(self, ctx: RepairContext):
        step = ctx.agent_step()
        if step is None:
            return Skip('the attributed step is not in the trace')
        writes = [(e.step_index, c) for e in ctx.trace.from_step(step) if e.is_agent for c in e.tool_calls
                  if ctx.task.is_write(c.name)]
        if not writes:
            return Skip('no data-changing action at or after the attributed step')
        write_step, call = writes[0]
        contract = RuleBasedExtractor().extract(ctx.trace, ctx.task, write_step)
        issues = contract.validate(call, ctx.task)
        targets = sorted(contract.entities)
        message = (f'Before calling {call.name}, state the exact action and its arguments to the user and ask them '
                   'to confirm the target' + (f' (in scope: {", ".join(targets[:8])})' if targets else '') +
                   '. Execute only after an explicit yes.')
        confirm = rule_specs(ctx.task, kinds=CONFIRM_KINDS)
        key_fields = [k for k in call.arguments if k.endswith(('_id', '_ids'))] or list(call.arguments)[:2]
        acceptance = [check('asked_user', fields=key_fields)]
        if confirm:
            acceptance.append(check('rules_pass', rule_ids=[r['id'] for r in confirm]))
        return self.build(ctx, ctx.trace.rollback_point(write_step),
                          [InjectMessage('system', message), GuardCalls(rules=confirm)], acceptance,
                          'Confirm the exact target of a data-changing action with the user when it is ambiguous.',
                          issues or [f'{call.name} at step {write_step}'],
                          confidence_factor=1.0 if issues else 0.7)


class PolicyGate(Strategy):
    spec = StrategySpec(
        id='policy_gate', name='Enforce policy before each call with persistent plan state',
        categories=(INSTRUCTION_ADHERENCE,), rung=1, kind='reasoning_repair', tokens=800, prior_success=0.6,
        description='Compiles the policy into pre-execution rules applied to every call, and carries a done/pending '
                    'checklist of confirmed actions through the whole re-execution.')
    fresh = False

    def plan(self, ctx: RepairContext):
        step = ctx.agent_step()
        if step is None:
            return Skip('the attributed step is not in the trace')
        violations = ctx.gate.audit(ctx.trace)
        rollback = min([step] + [ctx.trace.rollback_point(v.step_index) or step for v in violations
                                 if v.step_index >= (ctx.trace.last_user_step_before(step) or 1)])
        obligations = ctx.obligations
        state = track(ctx.trace, obligations)
        lines = [f'Rule: {v.message} (violated at step {v.step_index}: {v.detail})' for v in violations]
        if obligations:
            lines.append(state.render())
        if not lines:
            lines.append(f'A reviewer found a policy or plan deviation: {ctx.hypothesis.rationale or "see policy"}.')
        operations = [GuardCalls(rules=tuple(dict(r) for r in ctx.task.rules), validate_schema=True),
                      InjectMessage('system', '\n'.join(lines), persistent=True)]
        if self.fresh:
            operations.insert(0, Resample(fresh_context=True, seed_offset=2))
        acceptance = [check('rules_pass'), check('calls_valid')]
        if obligations:
            acceptance.append(check('obligations_done', obligations=[o.to_dict() for o in obligations]))
        evidence = [v.describe() for v in violations] + [f'{p.obligation.label()}: {p.status}' for p in state.items]
        factor = 1.0 if violations or state.pending or state.failed else 0.6
        return self.build(ctx, rollback, operations, acceptance,
                          'Check each call against the policy rules before executing it and track confirmed actions '
                          'until they are done.', evidence, confidence_factor=factor)


class PolicyGateFresh(PolicyGate):
    spec = StrategySpec(
        id='policy_gate_fresh', name='Policy gate after a fresh resample', categories=(INSTRUCTION_ADHERENCE,),
        rung=2, kind='reasoning_repair', tokens=900, prior_success=0.45,
        description='The same gate and plan state, regenerating from the rollback point in a fresh context.')
    fresh = True
