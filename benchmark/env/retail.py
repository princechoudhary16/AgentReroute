"""Retail database and tools with tau-bench retail semantics.

The data files (users, orders, products) and tasks come from tau-bench
(sierra-research/tau-bench, MIT) or from the bundled fixture, which uses the
same format. The tool logic is a stdlib reimplementation: results are only
compared against gold actions executed with this same code, so the outcome
check (state hash) is internally consistent.

Tool errors are returned as strings starting with ``Error:``, as in tau-bench.
"""
import ast
import copy
import json
import operator

from ..io import digest

WRITE_TOOLS = frozenset({
    'cancel_pending_order', 'exchange_delivered_order_items', 'modify_pending_order_address',
    'modify_pending_order_items', 'modify_pending_order_payment', 'modify_user_address',
    'return_delivered_order_items', 'transfer_to_human_agents',
})

CANCEL_REASONS = ('no longer needed', 'ordered by mistake')


class ToolError(Exception):
    pass


def _schema(name, description, **params):
    required = [k for k, v in params.items() if not v.pop('optional', False)]
    return {'name': name, 'description': description,
            'parameters': {'type': 'object', 'properties': params, 'required': required}}


_S, _L = {'type': 'string'}, {'type': 'array', 'items': {'type': 'string'}}

TOOL_SCHEMAS = {s['name']: s for s in [
    _schema('calculate', 'Evaluate an arithmetic expression with + - * / and parentheses.', expression=dict(_S)),
    _schema('cancel_pending_order', "Cancel a pending order. Reason is 'no longer needed' or 'ordered by mistake'. "
            'Refunds go to the original payment methods.', order_id=dict(_S), reason=dict(_S)),
    _schema('exchange_delivered_order_items', 'Exchange items of a delivered order for other variants of the same '
            'products. The price difference is charged or refunded to payment_method_id.',
            order_id=dict(_S), item_ids=dict(_L), new_item_ids=dict(_L), payment_method_id=dict(_S)),
    _schema('find_user_id_by_email', 'Find a user id by email.', email=dict(_S)),
    _schema('find_user_id_by_name_zip', 'Find a user id by first name, last name and zip code.',
            first_name=dict(_S), last_name=dict(_S), zip=dict(_S)),
    _schema('get_order_details', 'Get the status and details of an order.', order_id=dict(_S)),
    _schema('get_product_details', 'Get the inventory details of a product and its variants.', product_id=dict(_S)),
    _schema('get_user_details', 'Get the details of a user, including orders and payment methods.', user_id=dict(_S)),
    _schema('list_all_product_types', 'List the name and product id of every product type.'),
    _schema('modify_pending_order_address', 'Modify the shipping address of a pending order.',
            order_id=dict(_S), address1=dict(_S), address2=dict(_S), city=dict(_S), state=dict(_S),
            country=dict(_S), zip=dict(_S)),
    _schema('modify_pending_order_items', 'Modify items of a pending order to other variants of the same products. '
            'Can be done once per order.', order_id=dict(_S), item_ids=dict(_L), new_item_ids=dict(_L),
            payment_method_id=dict(_S)),
    _schema('modify_pending_order_payment', 'Change the payment method of a pending order.',
            order_id=dict(_S), payment_method_id=dict(_S)),
    _schema('modify_user_address', 'Modify the default address of a user.', user_id=dict(_S), address1=dict(_S),
            address2=dict(_S), city=dict(_S), state=dict(_S), country=dict(_S), zip=dict(_S)),
    _schema('return_delivered_order_items', 'Return items of a delivered order. The refund goes to the original '
            'payment method or an existing gift card.', order_id=dict(_S), item_ids=dict(_L),
            payment_method_id=dict(_S)),
    _schema('transfer_to_human_agents', 'Transfer the user to a human agent with a summary of the issue.',
            summary=dict(_S)),
]}

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def _arith(node):
    if isinstance(node, ast.Expression):
        return _arith(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_arith(node.left), _arith(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_arith(node.operand))
    raise ToolError('invalid characters in expression')


class RetailDB:
    """Mutable copy of the retail data with the tau-bench tool set."""

    def __init__(self, data):
        self.data = copy.deepcopy(data)

    @property
    def users(self):
        return self.data['users']

    @property
    def orders(self):
        return self.data['orders']

    @property
    def products(self):
        return self.data['products']

    def state_hash(self):
        return digest(self.data)

    # ---------------------------------------------------------------- dispatch
    def call(self, name, arguments):
        """Run a tool; returns the observation string (errors included)."""
        if name not in TOOL_SCHEMAS:
            return f'Error: unknown tool {name}'
        if not isinstance(arguments, dict):
            return 'Error: arguments must be an object'
        schema = TOOL_SCHEMAS[name]['parameters']
        missing = [k for k in schema['required'] if k not in arguments]
        unknown = [k for k in arguments if k not in schema['properties']]
        if missing or unknown:
            return f'Error: invalid arguments for {name}: missing {missing}, unexpected {unknown}'
        try:
            result = getattr(self, name)(**arguments)
        except ToolError as exc:
            return f'Error: {exc}'
        except (TypeError, ValueError, KeyError, ZeroDivisionError) as exc:
            return f'Error: {type(exc).__name__}: {exc}'
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

    # ---------------------------------------------------------------- helpers
    def _order(self, order_id):
        if order_id not in self.orders:
            raise ToolError('order not found')
        return self.orders[order_id]

    def _user(self, user_id):
        if user_id not in self.users:
            raise ToolError('user not found')
        return self.users[user_id]

    def _variant(self, product_id, item_id):
        variants = self.products.get(product_id, {}).get('variants', {})
        if item_id not in variants:
            raise ToolError(f'new item {item_id} not found')
        return variants[item_id]

    def _payment(self, user_id, payment_method_id):
        methods = self._user(user_id)['payment_methods']
        if payment_method_id not in methods:
            raise ToolError('payment method not found')
        return methods[payment_method_id]

    def _swap(self, order, item_ids, new_item_ids):
        """Validate an item swap; returns (old items, new variants, price difference)."""
        if len(item_ids) != len(new_item_ids):
            raise ToolError('the number of items to be exchanged should match')
        available = [i['item_id'] for i in order['items']]
        for item_id in item_ids:
            if item_ids.count(item_id) > available.count(item_id):
                raise ToolError(f'{item_id} not found')
        old, new, diff = [], [], 0.0
        for item_id, new_id in zip(item_ids, new_item_ids):
            item = next(i for i in order['items'] if i['item_id'] == item_id)
            variant = self._variant(item['product_id'], new_id)
            if not variant['available']:
                raise ToolError(f'new item {new_id} not found or available')
            old.append(item)
            new.append(variant)
            diff += variant['price'] - item['price']
        return old, new, round(diff, 2)

    def _charge(self, user_id, payment_method_id, amount):
        method = self._payment(user_id, payment_method_id)
        if method['source'] == 'gift_card' and method.get('balance', 0) < amount:
            raise ToolError('insufficient gift card balance to pay for the price difference')
        if method['source'] == 'gift_card':
            method['balance'] = round(method['balance'] - amount, 2)

    # ---------------------------------------------------------------- read tools
    def calculate(self, expression):
        if not all(c in '0123456789+-*/(). ' for c in expression):
            raise ToolError('invalid characters in expression')
        return str(round(float(_arith(ast.parse(expression, mode='eval'))), 2))

    def find_user_id_by_email(self, email):
        for user_id, user in self.users.items():
            if user['email'].lower() == str(email).lower():
                return user_id
        raise ToolError('user not found')

    def find_user_id_by_name_zip(self, first_name, last_name, zip):
        for user_id, user in self.users.items():
            name = user['name']
            if (name['first_name'].lower(), name['last_name'].lower(), user['address']['zip']) == \
                    (str(first_name).lower(), str(last_name).lower(), str(zip)):
                return user_id
        raise ToolError('user not found')

    def get_order_details(self, order_id):
        return self._order(order_id)

    def get_product_details(self, product_id):
        if product_id not in self.products:
            raise ToolError('product not found')
        return self.products[product_id]

    def get_user_details(self, user_id):
        return self._user(user_id)

    def list_all_product_types(self):
        return dict(sorted((p['name'], pid) for pid, p in self.products.items()))

    def transfer_to_human_agents(self, summary):
        return 'Transfer successful'

    # ---------------------------------------------------------------- write tools
    def cancel_pending_order(self, order_id, reason):
        order = self._order(order_id)
        if order['status'] != 'pending':
            raise ToolError('non-pending order cannot be cancelled')
        if reason not in CANCEL_REASONS:
            raise ToolError('invalid reason')
        refunds = []
        for payment in order['payment_history']:
            refunds.append({'transaction_type': 'refund', 'amount': payment['amount'],
                            'payment_method_id': payment['payment_method_id']})
            method = self._payment(order['user_id'], payment['payment_method_id'])
            if method['source'] == 'gift_card':
                method['balance'] = round(method['balance'] + payment['amount'], 2)
        order['status'] = 'cancelled'
        order['cancel_reason'] = reason
        order['payment_history'].extend(refunds)
        return order

    def modify_pending_order_address(self, order_id, address1, address2, city, state, country, zip):
        order = self._order(order_id)
        if order['status'] != 'pending':
            raise ToolError('non-pending order cannot be modified')
        order['address'] = {'address1': address1, 'address2': address2, 'city': city, 'state': state,
                            'country': country, 'zip': zip}
        return order

    def modify_pending_order_items(self, order_id, item_ids, new_item_ids, payment_method_id):
        order = self._order(order_id)
        if order['status'] != 'pending':
            raise ToolError('non-pending order cannot be modified')
        old, new, diff = self._swap(order, item_ids, new_item_ids)
        self._charge(order['user_id'], payment_method_id, diff)
        if diff != 0:
            order['payment_history'].append({'transaction_type': 'payment' if diff > 0 else 'refund',
                                             'amount': abs(diff), 'payment_method_id': payment_method_id})
        for item, variant in zip(old, new):
            item['item_id'], item['price'] = variant['item_id'], variant['price']
            item['options'] = copy.deepcopy(variant['options'])
        order['status'] = 'pending (item modified)'
        return order

    def modify_pending_order_payment(self, order_id, payment_method_id):
        order = self._order(order_id)
        if order['status'] != 'pending':
            raise ToolError('non-pending order cannot be modified')
        payments = [p for p in order['payment_history'] if p['transaction_type'] == 'payment']
        if len(payments) != 1:
            raise ToolError('there should be exactly one payment for a pending order')
        current = payments[0]
        if current['payment_method_id'] == payment_method_id:
            raise ToolError('the new payment method should be different from the current one')
        amount = current['amount']
        self._charge(order['user_id'], payment_method_id, amount)
        old = self._payment(order['user_id'], current['payment_method_id'])
        if old['source'] == 'gift_card':
            old['balance'] = round(old['balance'] + amount, 2)
        order['payment_history'].extend([
            {'transaction_type': 'payment', 'amount': amount, 'payment_method_id': payment_method_id},
            {'transaction_type': 'refund', 'amount': amount, 'payment_method_id': current['payment_method_id']},
        ])
        return order

    def modify_user_address(self, user_id, address1, address2, city, state, country, zip):
        user = self._user(user_id)
        user['address'] = {'address1': address1, 'address2': address2, 'city': city, 'state': state,
                           'country': country, 'zip': zip}
        return user

    def return_delivered_order_items(self, order_id, item_ids, payment_method_id):
        order = self._order(order_id)
        if order['status'] != 'delivered':
            raise ToolError('non-delivered order cannot be returned')
        method = self._payment(order['user_id'], payment_method_id)
        original = order['payment_history'][0]['payment_method_id']
        if method['source'] != 'gift_card' and payment_method_id != original:
            raise ToolError('payment method should be either the original payment method or a gift card')
        available = [i['item_id'] for i in order['items']]
        for item_id in item_ids:
            if item_ids.count(item_id) > available.count(item_id):
                raise ToolError('some item not found')
        order['status'] = 'return requested'
        order['return_items'] = sorted(item_ids)
        order['return_payment_method_id'] = payment_method_id
        return order

    def exchange_delivered_order_items(self, order_id, item_ids, new_item_ids, payment_method_id):
        order = self._order(order_id)
        if order['status'] != 'delivered':
            raise ToolError('non-delivered order cannot be exchanged')
        _, _, diff = self._swap(order, item_ids, new_item_ids)
        method = self._payment(order['user_id'], payment_method_id)
        if method['source'] == 'gift_card' and method.get('balance', 0) < diff:
            raise ToolError('insufficient gift card balance to pay for the price difference')
        order['status'] = 'exchange requested'
        order['exchange_items'] = sorted(item_ids)
        order['exchange_new_items'] = sorted(new_item_ids)
        order['exchange_payment_method_id'] = payment_method_id
        order['exchange_price_difference'] = diff
        return order


def gold_state_hash(data, actions):
    """State hash after executing a task's gold actions on a fresh database."""
    db = RetailDB(data)
    for action in actions:
        db.call(action['name'], action['kwargs'])
    return db.state_hash()
