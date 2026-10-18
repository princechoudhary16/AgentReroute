"""Shared helpers for the strategy tests."""
import json

from repair_strategies.context import Hypothesis
from repair_strategies.strategies import default_strategies
from repair_strategies.strategies.base import RepairContext

from .helpers import ORDER, RunBuilder, authenticated, retail

STRATEGIES = {s.spec.id: s for s in default_strategies()}
RETURN = {'order_id': '#W1000005', 'item_ids': ['2000000001'], 'payment_method_id': 'credit_card_4421'}
VARIANTS = {'variants': {'a': {'available': True}, 'b': {'available': False}, 'c': {'available': True}}}


def run(strategy_id, builder, step, category, subtype=None, rationale='', history=(), llm=None, memory=None):
    trace = builder.trace()
    hypothesis = Hypothesis(step, category, subtype=subtype, rationale=rationale)
    ctx = RepairContext(trace, hypothesis, retail(trace.instruction), tuple(history), memory, None, llm)
    return STRATEGIES[strategy_id].plan(ctx)


def op(plan, kind):
    found = plan.find(kind)
    assert found, f'{plan.strategy_id} has no {kind.__name__}'
    return found[0]


def kinds(plan):
    return [c.kind for c in plan.acceptance]


def proposal(builder, tool, arguments):
    builder.reply('planner', f'Please confirm the following action: {tool} {json.dumps(arguments)}')
    builder.user('yes')


def order_run(reply):
    """An authenticated run whose returns agent read the order and then replied."""
    b = authenticated(RunBuilder())
    b.tool('returns_agent', 'get_order_details', ORDER, order_id='#W1000005')
    return b, b.reply('returns_agent', reply)
