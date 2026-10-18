"""Misinterpretation of tool output, and handoffs that drop or alter context.

A restatement reuses the reasoning that failed. These strategies remove that
step: the output is parsed and aggregated in code, read a second time in a
fresh context, or re-queried, and the correct reading is delivered as a tool
message. A handoff that dropped fields is patched from earlier evidence.
"""
from __future__ import annotations

import json

from ..evidence import claims
from ..evidence.tool_output import JSON, aggregates, key_tokens, parse_output, to_decimal
from ..llm import LLMError, parse_json_object
from ..plan import GuardCalls, InjectMessage, ReplaceEvent, RequeryTool
from ..taxonomy import HANDOFF, MISINTERPRETATION
from .base import RepairContext, Skip, Strategy, StrategySpec, check, compact, record, rule_specs

_MAX_FIELDS = 25


def _structured_lines(content: str, qualifiers=()) -> list[str]:
    """Fields of a parsed output plus computed counts and sums, most relevant first."""
    parsed = parse_output(content)
    if parsed.kind != JSON:
        return [f'output: {compact(content, 300)}']
    wanted = set(qualifiers)
    leaves = parsed.leaves()
    ranked = sorted(leaves, key=lambda leaf: (not key_tokens(leaf.key) & wanted, leaf.dotted))
    lines = [f'{leaf.dotted} = {json.dumps(leaf.value, ensure_ascii=False)}' for leaf in ranked[:_MAX_FIELDS]]
    if len(leaves) > _MAX_FIELDS:
        lines.append(f'... {len(leaves) - _MAX_FIELDS} more fields')
    lines += [f'computed: {a.describe()}' for a in aggregates(parsed.data)
              if not wanted or key_tokens(a.key or a.path.split('.')[-1]) & wanted or a.kind == 'count_true']
    return lines


class RepairHandoff(Strategy):
    spec = StrategySpec(
        id='repair_handoff', name='Patch the handoff with the missing context', categories=(MISINTERPRETATION,),
        rung=1, kind='reasoning_repair', tokens=300, prior_success=0.65, subtypes=(HANDOFF, None),
        description='Finds the fields the receiving agent needs that the handoff omitted or altered, recovers them '
                    'from earlier evidence, patches the handoff and re-runs only the receiver.')

    def plan(self, ctx: RepairContext):
        found = self._handoff(ctx)
        if found is None:
            return Skip('no handoff at or before the attributed step')
        event = found
        handoff = event.handoff
        context = dict(handoff.get('context') or {})
        needed = self._needed(ctx, handoff['to'])
        if not needed:
            return Skip(f'cannot tell which fields {handoff["to"]} needs')
        index = claims.EvidenceIndex.before(ctx.trace, event.step_index)
        recovered, missing, evidence = {}, [], []
        for name in needed:
            current = context.get(name)
            if current not in (None, '', []) and self._grounded(index, current):
                continue
            value = self._recover(ctx, name, event.step_index)
            if value is not None and value == current:
                continue                     # evidence agrees with the handoff; nothing to patch
            if value is None:
                missing.append(name)
                evidence.append(f'{name} is missing from the handoff and from the evidence')
            else:
                recovered[name] = value
                evidence.append(f'{name}: handoff had {current!r}, evidence has {value!r}')
        if not recovered and not missing:
            return Skip('the handoff already carries every needed field')
        receiver_rules = rule_specs(ctx.task, kinds=('argument_grounded',), tools=ctx.task.tools_of(handoff['to']))
        if recovered:
            context.update(recovered)
            patched = {'handoff_to': handoff['to'], 'message': handoff['message'], 'context': context}
            rollback = ctx.trace.next_agent_step(event.step_index + 1)
            if rollback is None:
                return Skip('the receiving agent never ran after the handoff')
            operations = [ReplaceEvent(event.step_index, event.sub_index, json.dumps(patched, ensure_ascii=False),
                                       'complete the handoff context')]
            if missing:
                operations.append(InjectMessage('system', f'The handoff does not include {", ".join(missing)}; '
                                                          'obtain them with your tools before acting.'))
        else:
            rollback = event.step_index
            operations = [InjectMessage('system', f'The handoff to {handoff["to"]} must include '
                                                  f'{", ".join(missing)}. Collect them first and pass them in the '
                                                  'handoff context.')]
        if receiver_rules:
            operations.append(GuardCalls(rules=receiver_rules))
        acceptance = [check('rules_pass', rule_ids=[r['id'] for r in receiver_rules])] if receiver_rules else []
        acceptance.append(check('claims_grounded'))
        return self.build(ctx, rollback, operations, acceptance,
                          'Pass every identifier the receiving agent needs in the handoff context.', evidence)

    @staticmethod
    def _handoff(ctx: RepairContext):
        if ctx.step is None:
            return None
        candidates = [e for e in ctx.trace.events if e.handoff and e.step_index <= ctx.step]
        at_step = [e for e in candidates if e.step_index == ctx.step]
        if at_step:
            return at_step[0]
        if not candidates:
            return None
        last_user = ctx.trace.last_user_step_before(ctx.step) or 0
        recent = [e for e in candidates if e.step_index > last_user]
        return recent[-1] if recent else None

    @staticmethod
    def _needed(ctx: RepairContext, receiver: str) -> list[str]:
        tools = ctx.task.tools_of(receiver)
        intended = [o for o in ctx.obligations if o.tool in tools]
        names: list[str] = []
        specs = [ctx.task.tool(o.tool) for o in intended] or \
                [ctx.task.tool(t) for t in tools if ctx.task.is_write(t)]
        for spec in specs:
            for name in spec.required if spec else ():
                is_identifier = name.endswith(('_id', '_ids'))
                if (intended or is_identifier) and name not in names:
                    names.append(name)
        return names

    @staticmethod
    def _grounded(index, value) -> bool:
        values = value if isinstance(value, list) else [value]
        return all(index.find_text(str(v)) for v in values if str(v))

    @staticmethod
    def _recover(ctx: RepairContext, name: str, before: int):
        for obligation in ctx.obligations:
            if name in obligation.arguments:
                return obligation.arguments[name]
        singular = name[:-1] if name.endswith('_ids') else name
        for x in reversed([x for x in ctx.exchanges if x.result is not None and x.result.step_index < before]):
            if singular in x.call.arguments and not name.endswith('_ids'):
                return x.call.arguments[singular]
            parsed = parse_output(x.result.content)
            if parsed.kind == JSON:
                hits = [leaf.value for leaf in parsed.leaves() if leaf.key == singular]
                if len(hits) == 1 and not name.endswith('_ids'):
                    return hits[0]
            elif parsed.ok and name == 'user_id' and x.call.name.startswith('find_user_id'):
                return parsed.text.strip()
        return None


class StructuredReading(Strategy):
    spec = StrategySpec(
        id='structured_reading', name='Read the tool output in code', categories=(MISINTERPRETATION,), rung=1,
        kind='reasoning_repair', tokens=600, prior_success=0.6, subtypes=(None,),
        description='Parses the consumed output into fields, computes counts and sums in code, and delivers the '
                    'authoritative reading as a tool message together with the corrected values.')

    def plan(self, ctx: RepairContext):
        step = ctx.agent_step()
        if step is None:
            return Skip('the attributed step is not in the trace')
        exchange = ctx.consumed_exchange(step)
        if exchange is None:
            return Skip('no tool output was consumed before the attributed step')
        faulty = ctx.provenance(step).faulty
        qualifiers = tuple(q for c in faulty for q in c.claim.qualifiers)
        lines = _structured_lines(exchange.result.content, qualifiers)
        corrections = [c.describe() for c in faulty]
        guarded = {c.claim.tool for c in faulty if c.claim.tool}
        body = record(f'{exchange.call.name} (step {exchange.result.step_index})', lines)
        if corrections:
            body += '\nCorrections to the previous reading:\n' + '\n'.join(f'- {c}' for c in corrections)
        operations = [InjectMessage('tool', body, name=exchange.call.name)]
        if guarded:
            operations.append(GuardCalls(rules=rule_specs(ctx.task, kinds=('argument_grounded',), tools=guarded)))
        return self.build(ctx, step, operations, (check('claims_grounded'),),
                          f'Use the parsed fields and computed totals of {exchange.call.name} instead of re-reading '
                          'its raw output.', corrections or [f'consumed output of {exchange.call.name}'],
                          confidence_factor=1.0 if corrections else 0.7)


class IndependentReading(Strategy):
    spec = StrategySpec(
        id='independent_reading', name='Second reading in a fresh context', categories=(MISINTERPRETATION,), rung=2,
        kind='reasoning_repair', tokens=1500, prior_success=0.5, subtypes=(None, HANDOFF), requires_llm=True,
        description='A separate model reads the raw output with only the user request; values it reports are '
                    'confirmed against the parsed output before they are delivered.')

    def plan(self, ctx: RepairContext):
        if ctx.llm is None:
            return Skip('needs an LLM client')
        step = ctx.agent_step()
        exchange = ctx.consumed_exchange(step) if step is not None else None
        if exchange is None:
            return Skip('no tool output was consumed before the attributed step')
        users = [e.content for e in ctx.trace.before(step) if e.kind == 'user']
        messages = [{'role': 'system', 'content': 'Read the tool output and extract the values that answer the '
                                                  'request. Use only the output. Return JSON: {"values": {"<field>": '
                                                  '<value>}, "answer": "<one sentence>"}.'},
                    {'role': 'user', 'content': f'Request: {users[-1] if users else ctx.trace.instruction}\n'
                                                f'Tool {exchange.call.name} output:\n{exchange.result.content[:8000]}'}]
        try:
            data = parse_json_object(ctx.llm.complete(messages, purpose='independent_reading').text)
        except LLMError as exc:
            return Skip(f'second reading failed: {exc}')
        values = data.get('values') if isinstance(data.get('values'), dict) else {}
        index = claims.EvidenceIndex.before(ctx.trace, exchange.result.step_index + 1)
        confirmed, rejected = [], []
        for name, value in values.items():
            number = to_decimal(value)
            ok = bool(index.find_number(number) or index.find_sum(number)) if number is not None \
                else bool(index.find_text(str(value)))
            (confirmed if ok else rejected).append(f'{name} = {json.dumps(value, ensure_ascii=False)}')
        if not confirmed:
            return Skip('the second reading reported no value the output confirms')
        body = record(f'an independent reading of {exchange.call.name}', confirmed)
        evidence = [f'confirmed: {c}' for c in confirmed] + [f'discarded (not in output): {r}' for r in rejected]
        return self.build(ctx, step, [InjectMessage('tool', body, name=exchange.call.name)],
                          (check('claims_grounded'),),
                          f'Compare your reading of {exchange.call.name} with an independent one before acting.',
                          evidence)


class CrossSourceCheck(Strategy):
    spec = StrategySpec(
        id='cross_source_check', name='Re-query the source to catch silent corruption',
        categories=(MISINTERPRETATION,), rung=3, kind='retry', tokens=400, prior_success=0.35,
        subtypes=(None, HANDOFF),
        description='Re-queries the output the agent relied on, and an equivalent tool when one accepts the same '
                    'arguments, so stale or corrupted data is replaced by a fresh reading.')

    def plan(self, ctx: RepairContext):
        step = ctx.agent_step()
        exchange = ctx.consumed_exchange(step) if step is not None else None
        if exchange is None or ctx.task.is_write(exchange.call.name):
            return Skip('no read-only tool output to re-query')
        operations = [RequeryTool(exchange.call.name, dict(exchange.call.arguments), 'refresh possibly stale data')]
        for alt in ctx.task.alternatives(exchange.call.name):
            spec = ctx.task.tool(alt)
            if spec and set(spec.required) <= set(exchange.call.arguments):
                arguments = {k: v for k, v in exchange.call.arguments.items() if k in spec.properties}
                operations.append(RequeryTool(alt, arguments, 'independent source'))
        operations.append(InjectMessage('system', f'The earlier {exchange.call.name} result may be stale. Fresh '
                                                  'results follow; use them and point out any difference.'))
        return self.build(ctx, step, operations, (check('claims_grounded'),),
                          f'Re-query {exchange.call.name} before relying on a value that looks inconsistent.',
                          [f'relied on {exchange.call.name} at step {exchange.result.step_index}'])
