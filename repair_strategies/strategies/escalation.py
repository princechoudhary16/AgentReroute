"""Faults the agent cannot repair by itself: missing user input, out-of-scope requests, no diagnosis."""
from __future__ import annotations

from ..evidence import claims
from ..evidence.tool_output import words
from ..plan import AskUser, Compensate, Escalate, GuardCalls, InjectMessage, Resample
from ..taxonomy import GUARDRAILS, INCONCLUSIVE, INTENT_NOT_SUPPORTED, UNDERSPECIFIED_INTENT
from .base import RepairContext, Skip, Strategy, StrategySpec, check, compact
from .fallback import ESCALATION_TOOLS as _ESCALATION_TOOLS, compensation, state_summary


class TargetedClarification(Strategy):
    spec = StrategySpec(
        id='targeted_clarification', name='Ask only for the missing fields', categories=(UNDERSPECIFIED_INTENT,),
        rung=1, kind='clarify', tokens=300, prior_success=0.6,
        description='Determines which required arguments of the intended action no evidence supplies, fills '
                    'declared low-risk fields with safe defaults, and asks the user only for the rest.')

    def plan(self, ctx: RepairContext):
        step = ctx.agent_step()
        if step is None:
            return Skip('the attributed step is not in the trace')
        found = self._intended(ctx, step)
        if found is None:
            return Skip('cannot tell which action the user intended')
        tool, arguments, at = found
        spec = ctx.task.tool(tool)
        index = claims.EvidenceIndex.before(ctx.trace, at)
        defaults = dict(ctx.task.safe_defaults.get(tool, {}))
        said = set(words(' '.join(e.content for e in ctx.trace.before(at) if e.kind == 'user')))
        missing, defaulted = [], {}
        for name in spec.required:
            value = arguments.get(name)
            if name in defaults and value in (None, '', defaults[name]):
                defaulted[name] = defaults[name]
                continue
            if value in (None, '', []):
                missing.append(name)
                continue
            values = value if isinstance(value, list) else [value]
            choices = spec.properties.get(name, {}).get('enum') or self._enum(ctx, tool, name)
            if choices and value in choices and set(words(str(value))) & said:
                continue                     # a paraphrase mapped onto an allowed value
            if not all(index.find_text(str(v)) for v in values):
                missing.append(name)
        if not missing:
            return Skip(f'every required argument of {tool} is supplied by the conversation or a safe default')
        readable = ', '.join(m.replace('_', ' ') for m in missing)
        note = f'Ask the user only for: {readable}. Do not guess these values.'
        if defaulted:
            note += ' Use these defaults without asking: ' + ', '.join(f'{k}={v!r}' for k, v in defaulted.items()) + '.'
        grounding = {'id': 'clarified_fields', 'kind': 'argument_grounded', 'tools': [tool], 'arguments': missing,
                     'message': 'Use only values the user provided for these fields.'}
        operations = [AskUser(f'To {spec.description.rstrip(".").lower() or tool}, could you tell me your {readable}?',
                              tuple(missing)),
                      InjectMessage('system', note), GuardCalls(rules=(grounding,))]
        return self.build(ctx, step, operations,
                          (check('asked_user', fields=missing), check('rules_pass', rules=[grounding])),
                          'Ask the user for required details that are missing instead of guessing them.',
                          [f'{tool} needs {", ".join(missing)}'])

    @staticmethod
    def _enum(ctx: RepairContext, tool: str, argument: str):
        for rule in ctx.task.rules:
            if rule.get('kind') == 'argument_enum' and rule.get('tool') == tool and rule.get('argument') == argument:
                return rule.get('values')
        return None

    @staticmethod
    def _intended(ctx: RepairContext, step: int):
        for event in ctx.trace.from_step(step):
            for call in (event.tool_calls if event.is_agent else ()):
                if ctx.task.is_write(call.name) and call.name not in _ESCALATION_TOOLS and ctx.task.tool(call.name):
                    return call.name, call.arguments, event.step_index
        for obligation in ctx.obligations:
            if ctx.task.tool(obligation.tool):
                return obligation.tool, dict(obligation.arguments), step
        return None


class CompensateAndEscalate(Strategy):
    spec = StrategySpec(
        id='compensate_and_escalate', name='Undo what can be undone, then escalate',
        categories=(GUARDRAILS, INTENT_NOT_SUPPORTED), rung=1, kind='clarify', tokens=200, prior_success=0.9,
        reversible=False,
        description='The request is outside policy or capability: compensates every executed write that has an '
                    'inverse, lists those that cannot be undone, and escalates with a summary.')

    def plan(self, ctx: RepairContext):
        writes, calls, irreversible = compensation(ctx)
        reason = (ctx.hypothesis.rationale.strip() or f'{ctx.hypothesis.category}: the request cannot be completed '
                                                       'within policy')
        summary = state_summary(writes, calls, irreversible)
        operations = []
        if calls or irreversible:
            operations.append(Compensate(tuple(calls), tuple(irreversible)))
        operations.append(Escalate(reason, summary))
        return self.build(ctx, None, operations, (), 'Leave a consistent state before escalating an unsupported '
                                                     'request.', [summary])


class VerifiedUnmodifiedReplay(Strategy):
    spec = StrategySpec(
        id='verified_unmodified_replay', name='One bounded replay, accepted only by the verifier',
        categories=(INCONCLUSIVE,), rung=1, kind='retry', tokens=0, prior_success=0.15,
        description='Without a diagnosis there is nothing to target; a single resample from the attributed point '
                    '(or the first agent turn) is tried and counts only if the verifier passes it.')

    def plan(self, ctx: RepairContext):
        if ctx.attempts(self.spec.id):
            return Skip('the single unmodified replay was already used')
        step = ctx.agent_step() or ctx.trace.next_agent_step(1)
        if step is None:
            return Skip('the trace has no agent step')
        return self.build(ctx, step, [Resample(fresh_context=False, seed_offset=1)], (),
                          'Without a diagnosis, try one resample and rely on independent verification.',
                          [compact(ctx.hypothesis.rationale or 'attribution was inconclusive', 200)])

