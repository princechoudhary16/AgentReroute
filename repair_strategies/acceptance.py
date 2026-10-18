"""Deterministic post-conditions evaluated on a repaired rollout.

These are cheap pre-checks that reject a repair which visibly did not take
effect. They never declare a task solved: the orchestrator's fresh-context
verifier remains the gate for a recovery.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .context import TaskSpec
from .evidence import claims, cycles, schema
from .evidence.tool_output import is_error, to_decimal
from .plan import CHECK_KINDS, AcceptanceCheck, RepairPlan
from .policy.intent import IntentContract, scoped_entities
from .policy.plan_state import Obligation, track
from .policy.rules import RuleGate
from .trace import Trace


@dataclass(frozen=True)
class CheckResult:
    kind: str
    passed: bool
    detail: str = ''


@dataclass(frozen=True)
class AcceptanceReport:
    results: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(r for r in self.results if not r.passed)

    def to_dict(self) -> dict:
        return {'passed': self.passed,
                'results': [{'kind': r.kind, 'passed': r.passed, 'detail': r.detail} for r in self.results]}


def evaluate(plan: RepairPlan, rollout: Trace, task: TaskSpec) -> AcceptanceReport:
    """Run every acceptance check of ``plan`` on ``rollout`` (the replayed run)."""
    if plan.terminal:
        return AcceptanceReport(())
    start = plan.rollback_step
    return AcceptanceReport(tuple(run_check(check, rollout, task, start) for check in plan.acceptance))


def run_check(check: AcceptanceCheck, rollout: Trace, task: TaskSpec, start: int) -> CheckResult:
    passed, detail = _CHECKS[check.kind](check.params, rollout, task, start)
    return CheckResult(check.kind, passed, detail)


def _new_calls(rollout: Trace, start: int):
    return [x for x in rollout.tool_exchanges() if x.step_index >= start and not x.orphan]


def _claims_grounded(params, rollout, task, start):
    """No unsupported claim after the rollback point, except values the original run already had
    outside the repaired steps (``tolerate``): a repair must not regress, and is not held to a stricter
    standard than the agent's behaviour elsewhere."""
    tolerated = {str(v).lower() for v in params.get('tolerate', ())}
    reports = claims.provenance_from(rollout, start, task.fact_keywords)
    faulty = [c.describe() for r in reports for c in r.faulty if c.claim.value not in tolerated]
    return not faulty, '; '.join(faulty[:3])


def _present(value, text: str, numbers: set) -> bool:
    number = to_decimal(value)
    if number is not None:
        return number in numbers
    return re.sub(r'\s+', ' ', str(value).lower()) in text


def _values_present(params, rollout, task, start):
    texts = [e.text for e in rollout.from_step(start) if e.is_agent and e.text]
    final = rollout.final_reply()
    scope = ' '.join([final.text] if params.get('final_only', True) and final is not None else texts)
    numbers = {to_decimal(m.group(0)) for m in re.finditer(r'\d[\d,]*(?:\.\d+)?', scope)}
    text = re.sub(r'\s+', ' ', scope.lower())
    missing = [v for v in dict.fromkeys(map(str, params.get('values', []))) if not _present(v, text, numbers)]
    return not missing, f'missing from the reply: {", ".join(missing)}' if missing else ''


def _no_repeat(params, rollout, task, start):
    banned = set(params.get('signatures', []))
    repeated = [x.step_index for x in _new_calls(rollout, start) if x.call.signature() in banned]
    loops = [loop for loop in cycles.detect(rollout, task.write_tools) if loop.steps[-1] >= start]
    if repeated:
        return False, f'banned call repeated at steps {repeated}'
    if loops:
        return False, loops[0].describe()
    return True, ''


def _calls_valid(params, rollout, task, start):
    issues = []
    for x in _new_calls(rollout, start):
        allowed = task.tools_of(x.actor) or None
        issues += [f'step {x.step_index}: {i.describe()}' for i in schema.validate_call(x.call, task, allowed)]
    return not issues, '; '.join(issues[:3])


def _rules_pass(params, rollout, task, start):
    gate = RuleGate.for_task(task, extra=params.get('rules', ()))
    wanted = set(params.get('rule_ids', ())) | {r['id'] for r in params.get('rules', ())}
    violations = [v for v in gate.audit(rollout, start) if not wanted or v.rule_id in wanted]
    return not violations, '; '.join(v.describe() for v in violations[:3])


def _obligations_done(params, rollout, task, start):
    state = track(rollout, [Obligation.from_dict(o) for o in params.get('obligations', ())])
    open_items = [p.obligation.label() for p in state.items if p.status != 'done']
    return not open_items, f'not completed: {", ".join(open_items)}' if open_items else ''


def _contract_respected(params, rollout, task, start):
    contract = IntentContract.from_dict(params.get('contract') or {})
    issues = [f'step {x.step_index}: {issue}' for x in _new_calls(rollout, start)
              for issue in contract.validate(x.call, task, scoped_entities(rollout, x.step_index))]
    return not issues, '; '.join(issues[:3])


def _asked_user(params, rollout, task, start):
    fields = [f.lower() for f in params.get('fields', ())]
    for event in rollout.from_step(start):
        if not event.is_agent or not event.text or event.tool_calls or event.handoff:
            continue
        text = event.text.lower()
        if '?' in text and (not fields or any(f.replace('_', ' ') in text or f in text for f in fields)):
            return True, ''
    return False, f'the agent did not ask for {", ".join(fields) or "clarification"}'


def _tool_not_called(params, rollout, task, start):
    tool = params['tool']
    called = [x.step_index for x in _new_calls(rollout, start) if x.call.name == tool]
    return not called, f'{tool} called at steps {called}' if called else ''


def _tool_succeeded(params, rollout, task, start):
    tool = params['tool']
    for x in _new_calls(rollout, start):
        if x.call.name == tool and x.result is not None and not is_error(x.result.content):
            return True, ''
    return False, f'{tool} did not succeed after the rollback point'


def _no_error_results(params, rollout, task, start):
    tools = set(params.get('tools', ()))
    errors = [x.step_index for x in _new_calls(rollout, start)
              if (not tools or x.call.name in tools) and x.result is not None and is_error(x.result.content)]
    return not errors, f'tool errors at steps {errors}' if errors else ''


_CHECKS = {'claims_grounded': _claims_grounded, 'values_present': _values_present, 'no_repeat': _no_repeat,
           'calls_valid': _calls_valid, 'rules_pass': _rules_pass, 'obligations_done': _obligations_done,
           'contract_respected': _contract_respected, 'asked_user': _asked_user,
           'tool_not_called': _tool_not_called, 'tool_succeeded': _tool_succeeded,
           'no_error_results': _no_error_results}
if set(_CHECKS) != set(CHECK_KINDS):
    raise RuntimeError(f'acceptance kinds without an implementation: {set(CHECK_KINDS) ^ set(_CHECKS)}')
