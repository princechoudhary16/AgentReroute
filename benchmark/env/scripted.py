"""Deterministic scripted agents and user for offline tests and the smoke run.

OFFLINE FIXTURE, NOT A MODEL. The scripted planner reads the task's gold
actions to know what the user wants, so its runs exercise the pipeline
(workflow, fault hooks, labelling, scoring) without any LLM. They are marked
``measurement_kind = offline_fixture`` and must never be reported as results.

The scripted specialists do read tool outputs (order status, prices, product
availability), so silent tool faults and misread values change their behaviour
the way they would change a real agent's.
"""
import json
import re

from .llm import ScriptedLLM
from .workflow import STOP

ROUTE = {
    'cancel_pending_order': 'order_agent', 'modify_pending_order_address': 'order_agent',
    'modify_pending_order_payment': 'order_agent', 'modify_pending_order_items': 'order_agent',
    'modify_user_address': 'order_agent', 'return_delivered_order_items': 'returns_agent',
    'exchange_delivered_order_items': 'returns_agent',
}
SPECIALIST_TOOLS = {
    'order_agent': {'cancel_pending_order', 'modify_pending_order_address', 'modify_pending_order_payment',
                    'modify_pending_order_items', 'modify_user_address'},
    'returns_agent': {'return_delivered_order_items', 'exchange_delivered_order_items'},
    'product_agent': set(),
}


def _history(messages):
    """(kind, payload) events from a conversation, oldest first."""
    events = []
    for m in messages[1:]:
        c = m['content']
        if m['role'] == 'assistant':
            events.append(('self', c))
        elif c.startswith('USER: '):
            events.append(('user', c[6:]))
        elif c.startswith('TOOL_RESULT '):
            name, _, result = c[len('TOOL_RESULT '):].partition(': ')
            events.append(('tool', (name, result)))
        elif c.startswith('AGENT_RESULT from '):
            events.append(('agent_result', c.partition(': ')[2]))
        elif c.startswith('HANDOFF from planner: '):
            body, _, ctx = c[len('HANDOFF from planner: '):].partition('\nCONTEXT: ')
            try:
                context = json.loads(ctx)
            except json.JSONDecodeError:
                context = {}
            events.append(('handoff', (body, context)))
        elif c.startswith('SUPERVISOR NOTE: '):
            events.append(('note', c[len('SUPERVISOR NOTE: '):]))
        else:
            events.append(('system', c))
    return events


def _tool(name, **arguments):
    return {'thought': f'call {name}', 'action': 'tool', 'tool': name, 'arguments': arguments}


def _reply(text):
    return {'thought': 'reply', 'action': 'reply', 'content': text}


class ScriptedBrain:
    """Responder for ``ScriptedLLM`` that plays every agent and the user for one task."""

    def __init__(self, domain, task):
        self.domain, self.task = domain, task
        self.user = domain.data['users'][task['user_id']]
        self.writes = [a for a in task['actions'] if a['name'] in ROUTE]

    def __call__(self, req):
        agent = req['meta']['agent']
        if agent == 'user':
            return self.user_turn(req['messages'])
        if agent == 'planner':
            return self.planner(_history(req['messages']))
        return self.specialist(agent, _history(req['messages']))

    # ------------------------------------------------------------------ user
    def user_turn(self, messages):
        agent_text = messages[-1]['content'].lower()
        if len(messages) == 2:
            return f"Hi, my email is {self.user['email']}. " + self.task['instruction'].split('). ', 1)[-1]
        if 'confirm' in agent_text or 'proceed' in agent_text:
            return 'yes'
        return STOP

    # ------------------------------------------------------------------ planner
    def planner(self, events):
        tools = [p for k, p in events if k == 'tool']
        user_id = next((r for n, r in tools if n == 'find_user_id_by_email' and not r.startswith('Error')), None)
        if user_id is None:
            first = next(p for k, p in events if k == 'user')
            email = re.search(r'[\w.+-]+@[\w-]+(?:\.[\w-]+)+', first)
            return _tool('find_user_id_by_email', email=email.group(0) if email else '')
        results = [p for k, p in events if k == 'agent_result']
        if results:
            return _reply(results[-1])
        if not self.writes:
            return {'thought': 'route question', 'action': 'handoff', 'to': 'product_agent',
                    'message': self.task['instruction'].split('). ', 1)[-1],
                    'context': {'user_id': user_id, 'question': 'available_variants'}}
        users = [p for k, p in events if k == 'user']
        asked = any('confirm' in p.lower() for k, p in events if k == 'self')
        if not asked:
            summary = '; '.join(f"{a['name']} {json.dumps(a['kwargs'])}" for a in self.writes)
            return _reply(f'Please confirm the following action: {summary}. Shall I proceed (yes/no)?')
        if users and users[-1].strip().lower() != 'yes':
            return _reply('Okay, I will not make any change.')
        action = self.writes[0]
        context = {'user_id': user_id, 'action': action['name'], **action['kwargs']}
        return {'thought': 'delegate', 'action': 'handoff', 'to': ROUTE[action['name']],
                'message': f"User confirmed: {action['name']} for order {action['kwargs'].get('order_id', '')}.",
                'context': context}

    # ------------------------------------------------------------------ specialists
    def specialist(self, agent, events):
        body, context = next(p for k, p in events if k == 'handoff')
        tools = [p for k, p in events if k == 'tool']
        if agent == 'product_agent':
            return self._product(context, tools)
        request = context.get('action')
        if request not in SPECIALIST_TOOLS[agent]:
            return _reply(f'I cannot handle this request ({request}).')
        order_id = context.get('order_id')
        if not order_id:
            return _reply('The handoff did not include an order id, so I cannot proceed.')
        reads = [r for n, r in tools if n == 'get_order_details']
        if not reads or _retryable(reads[-1], ('items',)):
            if len(reads) >= 2 and reads[-1].startswith('Error') and reads[-2].startswith('Error'):
                return _reply(f'The order service keeps failing: {reads[-1]}')
            return _tool('get_order_details', order_id=order_id)
        order = json.loads(reads[-1]) if not reads[-1].startswith('Error') else None
        if order is None:
            return _reply(f'Could not read the order: {reads[-1]}')
        writes = [(n, r) for n, r in tools if n == request]
        if not writes:
            kwargs = {k: v for k, v in context.items() if k not in ('user_id', 'action')}
            return _tool(request, **kwargs)
        result = writes[-1][1]
        if result.startswith('Error'):
            return _reply(f'The request failed: {result}')
        if request == 'return_delivered_order_items':
            prices = {i['item_id']: i['price'] for i in order['items']}
            amount = round(sum(prices.get(i, 0) for i in context.get('item_ids', [])), 2)
            return _reply(f'Return requested. Refund amount: {amount}.')
        return _reply(f'Done: {request} completed for order {order_id}.')

    def _product(self, context, tools):
        if context.get('question') != 'available_variants':
            return _reply('I cannot handle this request.')
        listing = [r for n, r in tools if n == 'list_all_product_types']
        if not listing or _retryable(listing[-1], ('T-Shirt',)):
            return _tool('list_all_product_types')
        product_id = json.loads(listing[-1]).get('T-Shirt')
        details = [r for n, r in tools if n == 'get_product_details']
        if not details or _retryable(details[-1], ('variants',)):
            return _tool('get_product_details', product_id=product_id)
        variants = json.loads(details[-1])['variants']
        count = sum(1 for v in variants.values() if v['available'])
        return _reply(f'There are {count} T-shirt variants available.')


def _retryable(result, needs=()):
    """True when a tool result cannot be used: an error, or a payload missing the fields the caller needs.

    Shape-based on purpose. This used to test ``value.get('status') == 'processing'``, which was the fault
    injector's own stuck payload, so rewording that payload silently stopped the scripted agent from
    retrying and it then read an in-flight response as an order. A fixture agent that recognises a stuck
    tool by its wording is the same leak the seed checks had, one layer down.
    """
    if result.startswith('Error'):
        return True
    try:
        value = json.loads(result)
    except json.JSONDecodeError:
        return False
    if not isinstance(value, dict):
        return False
    return any(key not in value for key in needs)


def scripted_llm(domain, task):
    return ScriptedLLM(ScriptedBrain(domain, task))
