"""Lower a backend-neutral RepairPlan to what a specific replay backend can execute.

Operations a backend supports are kept as native instructions; the rest become
explicit text in the guidance, and each such step is recorded in ``degraded``
so evaluation can tell strategy quality apart from backend limits.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .plan import (ROLES, Abstain, AskUser, BanCalls, Compensate, Escalate, GuardCalls, InjectMessage, RepairPlan,
                   ReplaceEvent, RequeryTool, Resample, Retry, SwitchTool)
from .policy.intent import IntentContract

GUIDANCE, RESAMPLE, RETRY, NONE, TERMINAL = 'guidance', 'resample', 'retry', 'none', 'terminal'


@dataclass(frozen=True)
class Capabilities:
    name: str
    message_roles: frozenset = frozenset({'system'})   # roles a message can be delivered under
    replace_events: bool = False
    requery: bool = False
    guards: bool = False
    ban_calls: bool = False
    switch_tool: bool = False
    persistent: bool = False
    fresh_context: bool = False
    resample: bool = False
    guidance_with_resample: bool = False
    retry: bool = False
    compensate: bool = False


FULL = Capabilities('full', frozenset(ROLES), True, True, True, True, True, True, True, True, True, True, True)
# recovery_orchestrator RepairDirective: inject_guidance | retry | none
ORCHESTRATOR = Capabilities('orchestrator', retry=True)
# benchmark Directive: guidance (supervisor note) | resample (new seed) | none
BENCHMARK = Capabilities('benchmark', resample=True)


@dataclass(frozen=True)
class Lowered:
    plan_id: str
    strategy_id: str
    kind: str                                # guidance | resample | retry | none | terminal
    rollback_step: int | None
    text: str = ''
    seed_offset: int = 1
    resample: bool = False                   # also resample (only with guidance_with_resample)
    terminal_action: str | None = None       # escalate | abstain
    native: tuple[str, ...] = ()
    degraded: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict:
        return dict(self.__dict__, native=list(self.native), degraded=list(self.degraded))


def _args(arguments) -> str:
    return json.dumps(dict(arguments), ensure_ascii=False, sort_keys=True)


def lower(plan: RepairPlan, caps: Capabilities) -> Lowered:
    records, notes, rules, actions = [], [], [], []
    retries = []                               # instructions only needed if the retry is not native
    native, degraded = [], []
    want_resample = want_retry = False
    seed = 1
    terminal_action = None

    for op in plan.operations:
        name = op.op
        if isinstance(op, InjectMessage):
            if op.role in caps.message_roles:
                native.append(f'{name}({op.role})')
            else:
                degraded.append(f'{name}: {op.role} message delivered as a supervisor note')
            if op.persistent and not caps.persistent:
                degraded.append(f'{name}: persistent message applied once')
            label = f'[Tool record: {op.name}]' if op.role == 'tool' else '[Note]' if op.role == 'system' else '[User]'
            (records if op.role == 'tool' else notes).append(f'{label}\n{op.content}')
        elif isinstance(op, ReplaceEvent):
            if caps.replace_events:
                native.append(name)
            else:
                degraded.append(f'{name}: step {op.step_index} correction given as text')
                actions.append(f'Treat step {op.step_index} as corrected ({op.reason or "patched"}): {op.content}')
        elif isinstance(op, RequeryTool):
            if caps.requery:
                native.append(name)
            else:
                degraded.append(f'{name}: {op.tool} re-query turned into an instruction')
                actions.append(f'Call {op.tool} with {_args(op.arguments)} again and use its fresh result.')
        elif isinstance(op, GuardCalls):
            if caps.guards:
                native.append(name)
            else:
                degraded.append(f'{name}: pre-execution gate stated as rules')
                rules += [r.get('message') or r['id'] for r in op.rules]
                if op.validate_schema:
                    rules.append('Check every call\'s tool name and arguments against the tool schema before calling.')
                if op.allowed_tools:
                    rules.append(f'Use only these tools: {", ".join(op.allowed_tools)}.')
                if op.contract:
                    rules.append('Every action must match this contract:\n' + IntentContract.from_dict(op.contract).render())
        elif isinstance(op, BanCalls):
            if caps.ban_calls:
                native.append(name)
            else:
                degraded.append(f'{name}: ban stated as a rule')
                rules += [f'Do not repeat this call: {s}' for s in op.signatures]
        elif isinstance(op, SwitchTool):
            if caps.switch_tool:
                native.append(name)
            else:
                degraded.append(f'{name}: routing stated as an instruction')
                actions.append(f'Use {op.to_tool} instead of {op.from_tool}.')
        elif isinstance(op, AskUser):
            native.append(name)
            actions.append(f'Ask the user: {op.question}')
        elif isinstance(op, Resample):
            want_resample, seed = True, op.seed_offset
            if op.fresh_context and not caps.fresh_context:
                degraded.append(f'{name}: fresh context not available; the abandoned branch stays visible')
        elif isinstance(op, Retry):
            want_retry = True
            if not (caps.retry or caps.resample):
                degraded.append(f'{name}: retry stated as an instruction')
            elif op.backoff_s and not caps.retry:
                degraded.append(f'{name}: backoff of {op.backoff_s}s not applied')
            retries.append(f'Call {op.tool} with {_args(op.arguments)} again.')
        elif isinstance(op, Compensate):
            if caps.compensate:
                native.append(name)
            else:
                degraded.append(f'{name}: compensation must be executed by the caller')
            for call in op.calls:
                actions.append(f'Undo step {call.get("undoes_step")}: call {call["tool"]} with {_args(call["arguments"])}.')
            for item in op.irreversible:
                actions.append(f'Cannot undo {item["tool"]} at step {item["step_index"]}: {item["reason"]}.')
        elif isinstance(op, (Escalate, Abstain)):
            terminal_action = 'escalate' if isinstance(op, Escalate) else 'abstain'
            actions.append(op.reason + (f' {op.summary}' if isinstance(op, Escalate) and op.summary else ''))

    if plan.terminal:
        text = _compose(records, notes, rules, actions + retries)
        return Lowered(plan.plan_id, plan.strategy_id, TERMINAL, None, text, seed, False, terminal_action or 'escalate',
                       tuple(native), tuple(degraded))

    resample = False
    text = ''
    if records or notes or rules or actions:
        kind, text = GUIDANCE, _compose(records, notes, rules, actions + retries)
        if want_resample:
            if caps.guidance_with_resample:
                resample = True
                native.append('resample')
            else:
                degraded.append('resample: backend cannot combine guidance and resampling; guidance kept')
    elif want_resample or want_retry:
        if want_retry and caps.retry:
            kind = RETRY
        elif caps.resample:
            kind = RESAMPLE
        elif caps.retry:
            kind = RETRY
        else:
            kind, text = GUIDANCE, _compose([], [], [], retries or ['Retry the last step.'])
        native.append(kind)
    else:
        kind = NONE
    return Lowered(plan.plan_id, plan.strategy_id, kind, plan.rollback_step, text, seed, resample, None,
                   tuple(native), tuple(degraded))


def _compose(records, notes, rules, actions) -> str:
    parts = []
    if records:
        parts.append('\n\n'.join(records))
    if notes:
        parts.append('\n\n'.join(notes))
    if rules:
        parts.append('Rules for every tool call:\n' + '\n'.join(f'- {r}' for r in dict.fromkeys(rules)))
    if actions:
        parts.append('Do this now:\n' + '\n'.join(f'- {a}' for a in actions))
    return '\n\n'.join(parts)
