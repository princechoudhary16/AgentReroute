"""Invention of new information: replace unsupported claims with verified values, or abstain/ask.

A critique ("you made that up") tends to be miscorrected. These strategies
instead supply the verified value, delivered under an external (tool) role,
and re-query the tool that can verify a claim when no evidence holds it yet.
"""
from __future__ import annotations

from decimal import Decimal

from ..evidence import claims
from ..evidence.tool_output import JSON, key_tokens, parse_output, to_decimal, words
from ..plan import AskUser, GuardCalls, InjectMessage, RequeryTool
from ..taxonomy import INVENTION, OMISSION
from .base import RepairContext, Skip, Strategy, StrategySpec, check, record, rule_specs

USER_FIELDS = frozenset({'email', 'first_name', 'last_name', 'zip', 'address1', 'address2', 'city', 'state',
                         'country', 'phone'})
_MONEY_KEYS = frozenset({'price', 'amount', 'total', 'refund', 'cost', 'balance'})


def _faulty_turn(ctx: RepairContext) -> tuple[int, tuple[claims.ClaimCheck, ...]] | None:
    """The first agent step at or after the attributed one (same turn) with an unsupported claim."""
    start = ctx.agent_step()
    if start is None:
        return None
    for step in ctx.turn_steps(start)[:4]:
        faulty = ctx.provenance(step).faulty
        if faulty:
            return step, faulty
    return None


def _verifying_call(ctx: RepairContext, check_: claims.ClaimCheck, step: int):
    """A read call that can verify an ungrounded claim: one about the same entities, else the latest read."""
    event_text = ' '.join(e.text + ' ' + str([c.arguments for c in e.tool_calls]) for e in ctx.trace.at(step))
    reads = [x for x in ctx.exchanges if x.step_index < step and not ctx.task.is_write(x.call.name)
             and x.result is not None and parse_output(x.result.content).ok]
    entity_words = set(words(event_text))
    for x in reversed(reads):
        values = {str(v).lower() for v in x.call.arguments.values() if isinstance(v, (str, int))}
        if values and any(v in event_text.lower() for v in values):
            return x
        if check_.claim.qualifiers and set(check_.claim.qualifiers) & entity_words and \
                any(key_tokens(leaf.key) & set(check_.claim.qualifiers) for leaf in parse_output(x.result.content).leaves()):
            return x
    return reads[-1] if reads else None


class GroundedReplacement(Strategy):
    spec = StrategySpec(
        id='grounded_replacement', name='Replace unsupported claims with verified values', categories=(INVENTION,),
        rung=1, kind='reasoning_repair', tokens=700, prior_success=0.55, subtypes=(None,),
        description='Checks every identifier, amount and count against user turns and tool outputs; replaces '
                    'contradicted values with the verified ones and re-queries the tool that can verify the rest, '
                    'delivered as a tool message.')

    def plan(self, ctx: RepairContext):
        found = _faulty_turn(ctx)
        if found is None:
            return Skip('no contradicted or ungrounded claim at the attributed step')
        step, faulty = found
        lines, requery, guarded_tools, evidence = [], {}, set(), []
        for c in faulty:
            evidence.append(c.describe())
            if c.status == claims.CONTRADICTED:
                lines.append(f'"{c.claim.raw}" is not supported; the record says {", ".join(c.expected)}')
                continue
            if c.claim.tool:
                guarded_tools.add(c.claim.tool)
            source = _verifying_call(ctx, c, step)
            if source is None:
                continue
            requery.setdefault(source.call.signature(), source)
            lines.append(f'"{c.claim.raw}" is not in any record; check it against the fresh {source.call.name} result')
        if not lines:
            return Skip('unsupported claims have no source to verify against (abstain_or_ask applies)')
        operations = [RequeryTool(x.call.name, dict(x.call.arguments), f'verify claims made at step {step}')
                      for x in requery.values()]
        operations.append(InjectMessage('tool', record('the tool outputs', lines) +
                                        '\nState only values that appear in a tool output.', name='verified_record'))
        if guarded_tools:
            operations.append(GuardCalls(rules=rule_specs(ctx.task, kinds=('argument_grounded',), tools=guarded_tools)))
        rationale = ('Replace claims that no tool output supports with the verified values, and re-query the source '
                     'before stating identifiers or amounts.')
        return self.build(ctx, ctx.trace.rollback_point(step), operations, (check('claims_grounded'),), rationale,
                          evidence)


class CompleteOmittedDetails(Strategy):
    spec = StrategySpec(
        id='complete_omitted_details', name='Supply requested facts the answer left out', categories=(INVENTION,),
        rung=1, kind='reasoning_repair', tokens=500, prior_success=0.6, subtypes=(OMISSION,),
        description='Finds facts the user asked for that were retrieved but missing from the final answer and '
                    'supplies them as verified values to include.')

    def plan(self, ctx: RepairContext):
        final = ctx.trace.final_reply()
        if final is None:
            return Skip('the run has no final answer to complete')
        requested = self._requested(ctx)
        if not requested:
            return Skip('the user did not ask for a fact this domain can look up')
        reply = final.text.lower().replace(',', '').replace('$', '')
        missing, lines = [], []
        for keyword, value, path in self._values(ctx, requested, final.step_index):
            plain = format(value, 'f') if isinstance(value, Decimal) else str(value)
            if plain.lower() not in reply:
                shown = f'${value:.2f}' if isinstance(value, Decimal) and key_tokens(path.split('.')[-1]) & _MONEY_KEYS \
                    else plain
                missing.append(shown)
                lines.append(f'{keyword}: {path} = {shown}')
        if not missing:
            return Skip('every requested fact already appears in the final answer')
        operations = [InjectMessage('tool', record('the tool outputs', lines) +
                                    '\nThe user asked for these values. Include each of them in your answer.',
                                    name='verified_record')]
        rationale = 'Before the final answer, check that every fact the user asked for is stated, using the verified values.'
        return self.build(ctx, final.step_index, operations,
                          (check('values_present', values=missing), check('claims_grounded')), rationale, lines)

    @staticmethod
    def _requested(ctx):
        asked = ' '.join([ctx.trace.instruction] + [e.content for e in ctx.trace.user_events()])
        tokens = set(words(asked))
        if 'how much' in asked.lower():
            tokens |= {'price', 'total'}
        return [k for k in ctx.task.fact_keywords if k in tokens]

    @staticmethod
    def _values(ctx, requested, before):
        user_words = set(words(' '.join(e.content for e in ctx.trace.user_events())))
        out, seen = [], set()
        for keyword in requested:
            keys = set().union(*(key_tokens(k) for k in ctx.task.fact_keywords[keyword]))
            for event in ctx.trace.before(before):
                if event.kind != 'tool':
                    continue
                parsed = parse_output(event.content)
                if parsed.kind != JSON:
                    continue
                for path, obj in _objects(parsed.data):
                    names = {w for v in obj.values() if isinstance(v, str) for w in words(v)}
                    anchored = bool(names & user_words)
                    for key, raw in obj.items():
                        if not key_tokens(key) & keys:
                            continue
                        value = to_decimal(raw)
                        value = value if value is not None else raw
                        if isinstance(value, (dict, list)) or (path and not anchored):
                            continue
                        dotted = '.'.join(str(p) for p in path + (key,))
                        if (keyword, dotted) not in seen:
                            seen.add((keyword, dotted))
                            out.append((keyword, value, dotted))
        return out[:6]


def _objects(value, path=()):
    if isinstance(value, dict):
        yield path, value
        for k, v in value.items():
            yield from _objects(v, path + (str(k),))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _objects(v, path + (i,))


class AbstainOrAsk(Strategy):
    spec = StrategySpec(
        id='abstain_or_ask', name='Abstain or ask when no source exists', categories=(INVENTION,), rung=2,
        kind='clarify', tokens=400, prior_success=0.4,
        description='When a claimed fact has no source, asks the user if it is theirs to give, otherwise forbids '
                    'the claim and has the agent say the information is unavailable.')

    def plan(self, ctx: RepairContext):
        start = ctx.agent_step()
        if start is None:
            return Skip('the attributed step is not in the trace')
        found = _faulty_turn(ctx)
        if found is None:
            if not ctx.hypothesis.rationale:
                return Skip('no unsupported claim found and no rationale to act on')
            message = ('A reviewer found an unsupported statement here: ' + ctx.hypothesis.rationale.strip() +
                       '\nState only facts present in user messages or tool outputs. If a fact is not available, '
                       'say so or look it up with an available tool.')
            return self.build(ctx, start, [InjectMessage('system', message)], (check('claims_grounded'),),
                              'Say that information is unavailable instead of stating unsupported facts.',
                              confidence_factor=0.6)
        step, faulty = found
        ask = [c for c in faulty if c.claim.kind == claims.EMAIL or (c.claim.argument or '') in USER_FIELDS]
        lines = [c.describe() for c in faulty]
        operations = []
        if ask:
            fields = tuple(dict.fromkeys(c.claim.argument or c.claim.kind for c in ask))
            operations.append(AskUser(f'Could you confirm your {", ".join(f.replace("_", " ") for f in fields)}?',
                                      fields))
        unsupported = ', '.join(f'"{c.claim.raw}"' for c in faulty)
        operations.append(InjectMessage('system', f'No user message or tool output supports {unsupported}. Do not '
                                                  'state or use these values. '
                                        + ('Ask the user for them.' if ask else
                                           'Tell the user this information is not available.')))
        acceptance = [check('claims_grounded')]
        if ask:
            acceptance.append(check('asked_user', fields=list(operations[0].fields)))
        return self.build(ctx, ctx.trace.rollback_point(step), operations, acceptance,
                          'Ask for or decline to state values that no source supports.', lines)
