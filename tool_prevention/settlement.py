"""Shared reference business effects, independent of approval enforcement.

This is an integer-cent simulated ledger, not payment-network processing.
"""
import copy
from .contracts import Denied


def settle(user,original,order,tool,quote,*,unchecked=False):
    allocations=copy.deepcopy(original.get('payment_allocations',[
        {'payment_method_id':original['payment_method_id'],
         'amount_cents':sum(item['price_cents'] for item in original['items'])}]))
    flows=[]
    def flow(kind,method,amount):
        if amount:flows.append({'kind':kind,'payment_method_id':method,'amount_cents':amount,'order_id':order['order_id']})
    def reduce_allocations(amount):
        for allocation in allocations:
            deduction=min(allocation['amount_cents'],amount)
            allocation['amount_cents']-=deduction;amount-=deduction
        if amount and not unchecked:raise Denied('invalid_funding','Refund exceeds recorded funding')
    if tool in ('cancel_pending_order','modify_pending_order_payment'):
        for allocation in allocations:flow('refund',allocation['payment_method_id'],allocation['amount_cents'])
        quote['refunds']=copy.deepcopy(allocations);allocations=[]
        if tool=='modify_pending_order_payment':
            amount=sum(item['price_cents'] for item in order['items'])
            flow('charge',order['payment_method_id'],amount)
            allocations=[{'payment_method_id':order['payment_method_id'],'amount_cents':amount}]
    elif tool in ('modify_pending_order_items','return_delivered_order_items','exchange_delivered_order_items'):
        amount=quote['amount_cents'];method=quote['payment_method_id']
        refund=tool=='return_delivered_order_items' or amount<0
        flow('refund' if refund else 'charge',method,abs(amount))
        if refund:reduce_allocations(abs(amount))
        elif amount:allocations.append({'payment_method_id':method,'amount_cents':amount})
    for entry in flows:
        method=user['payment_methods'].get(entry['payment_method_id'])
        if not method and not unchecked:raise Denied('payment_ownership','Funding method is not owned by customer')
        if method and method['type']=='gift_card':
            delta=entry['amount_cents']*(1 if entry['kind']=='refund' else -1)
            balance=method.get('balance_cents',0)+delta
            if balance<0 and not unchecked:raise Denied('insufficient_funds','Gift-card balance is insufficient')
            method['balance_cents']=balance
        user.setdefault('transactions',[]).append(entry)
    order['payment_allocations']=[allocation for allocation in allocations if allocation['amount_cents']]
    if flows:quote['transactions']=copy.deepcopy(flows)
    return bool(flows)
