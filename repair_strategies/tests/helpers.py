"""Trace builders used across the test suite.

``RunBuilder`` writes traces exactly as the benchmark exports them: agent turns
are JSON actions, tool results are separate steps, handoffs carry a dedicated
``handoff`` substep.
"""
from __future__ import annotations

import json

from repair_strategies.context import TaskSpec
from repair_strategies.trace import Trace


class RunBuilder:
    def __init__(self, trajectory_id='t-1', domain='retail', instruction=''):
        self.trajectory_id, self.domain, self.instruction = trajectory_id, domain, instruction
        self.steps: list[dict] = []

    @property
    def last(self) -> int:
        return len(self.steps)

    def _add(self, *substeps) -> int:
        index = len(self.steps) + 1
        self.steps.append({'index': index, 'substeps': [dict(s, sub_index=i) for i, s in enumerate(substeps, 1)]})
        return index

    def user(self, text) -> int:
        if not self.instruction:
            self.instruction = text
        return self._add({'role': 'user', 'content': text})

    def reply(self, actor, text) -> int:
        return self._add({'role': actor, 'content': json.dumps({'thought': 'reply', 'action': 'reply',
                                                                'content': text})})

    def call(self, actor, tool, **arguments) -> int:
        action = {'thought': f'call {tool}', 'action': 'tool', 'tool': tool, 'arguments': arguments}
        index = len(self.steps) + 1
        return self._add({'role': actor, 'content': json.dumps(action),
                          'tool_calls': [{'id': f'call-{index}', 'name': tool, 'arguments': arguments}]})

    def result(self, tool, output) -> int:
        content = output if isinstance(output, str) else json.dumps(output)
        return self._add({'role': 'tool', 'name': tool, 'content': content})

    def tool(self, actor, tool, output, **arguments) -> tuple[int, int]:
        return self.call(actor, tool, **arguments), self.result(tool, output)

    def handoff(self, actor, to, message, context=None) -> int:
        payload = {'handoff_to': to, 'message': message, 'context': context or {}}
        action = {'thought': 'delegate', 'action': 'handoff', 'to': to, 'message': message,
                  'context': context or {}}
        return self._add({'role': actor, 'content': json.dumps(action)},
                         {'role': actor, 'name': 'handoff', 'content': json.dumps(payload)})

    def ir(self) -> dict:
        return {'trajectory_id': self.trajectory_id, 'domain': self.domain, 'instruction': self.instruction,
                'steps': json.loads(json.dumps(self.steps))}

    def trace(self) -> Trace:
        return Trace.from_ir(self.ir())


def retail(instruction='') -> TaskSpec:
    return TaskSpec.preset('retail').with_task(instruction=instruction)


ORDER = {'order_id': '#W1000005', 'user_id': 'ava_lee_2203', 'status': 'delivered',
         'address': {'address1': '12 Hillcrest Road', 'address2': '', 'city': 'Seattle', 'country': 'USA',
                     'state': 'WA', 'zip': '98101'},
         'items': [{'item_id': '2000000001', 'name': 'Water Bottle', 'price': 45.04, 'product_id': '8310926033'},
                   {'item_id': '2000000002', 'name': 'Desk Lamp', 'price': 120.5, 'product_id': '6817146515'}],
         'payment_history': [{'transaction_type': 'payment', 'amount': 165.54,
                              'payment_method_id': 'credit_card_4421'}]}

USER = {'user_id': 'ava_lee_2203', 'email': 'ava.lee5521@example.com',
        'address': {'address1': '12 Hillcrest Road', 'address2': '', 'city': 'Seattle', 'country': 'USA',
                    'state': 'WA', 'zip': '98101'},
        'payment_methods': {'credit_card_4421': {'source': 'credit_card', 'last_four': '4421'},
                            'gift_card_7781': {'source': 'gift_card', 'balance': 60.0}},
        'orders': ['#W1000005', '#W1000009']}


def authenticated(builder: RunBuilder, request='Return the water bottle from order #W1000005 and tell me the refund.'):
    """User turn, email lookup and user details: the usual opening of a retail run."""
    builder.user(f'Hi, my email is ava.lee5521@example.com. {request}')
    builder.tool('planner', 'find_user_id_by_email', 'ava_lee_2203', email='ava.lee5521@example.com')
    builder.tool('planner', 'get_user_details', USER, user_id='ava_lee_2203')
    return builder
