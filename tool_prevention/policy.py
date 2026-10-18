"""Explicit retail policy compilation and authoritative action planning."""
import copy
import json
from importlib.resources import files
from .contracts import CompatibilityError,Denied,ActionPlan
from .retail import READ_FIELDS,WRITE_FIELDS,ADDRESS,schema
from .settlement import settle


def default_policy():
    return json.loads(files('tool_prevention').joinpath('resources/retail_policy.json').read_text())


def validate_arguments(parameters,args):
    if not isinstance(args,dict) or set(args)!=set(parameters['required']):
        raise Denied('invalid_arguments','Arguments must contain exactly the declared fields')
    for key,definition in parameters['properties'].items():
        value=args[key]
        if definition['type']=='string':
            if not isinstance(value,str) or (not value and key!='address2'):
                raise Denied('invalid_arguments',f'{key} must be a string')
        elif definition['type']=='array':
            if not isinstance(value,list) or not value or any(not isinstance(v,str) or not v for v in value):
                raise Denied('invalid_arguments',f'{key} must be a nonempty string list')
        else: raise CompatibilityError('Unsupported schema type')


def compile_policy(system,manifest):
    methods=('backend_identity','tool_specs','read','snapshot','execute_atomic','lookup_execution')
    if any(not callable(getattr(system,name,None)) for name in methods):
        raise CompatibilityError('System needs a pre-execution tool adapter; replay-only environments cannot prevent effects')
    if not {'conditional_writes','durable_idempotency'}<=set(getattr(system,'capabilities',())):
        raise CompatibilityError('Backend must declare conditional writes and durable idempotency')
    if not isinstance(manifest,dict) or set(manifest)!={'version','approval_ttl_seconds','reads','writes'}:
        raise CompatibilityError('Policy requires version, approval_ttl_seconds, reads and writes only')
    if not isinstance(manifest['version'],str) or not manifest['version']:
        raise CompatibilityError('Policy needs a version')
    ttl=manifest['approval_ttl_seconds']
    if type(ttl) is not int or not 1<=ttl<=86400:raise CompatibilityError('Approval TTL must be 1..86400 seconds')
    if (not isinstance(manifest['reads'],list) or any(not isinstance(n,str) for n in manifest['reads'])
        or len(set(manifest['reads']))!=len(manifest['reads']) or not isinstance(manifest['writes'],dict)):
        raise CompatibilityError('Reads must be unique tool names; writes must map tool names to rules')
    specs=copy.deepcopy(system.tool_specs())
    if not isinstance(specs,dict):raise CompatibilityError('Adapter tool_specs must be a name-to-spec mapping')
    names=set(manifest['reads'])|set(manifest['writes'])
    if names!=set(specs) or set(manifest['reads'])&set(manifest['writes']):
        raise CompatibilityError('Every exposed tool needs exactly one explicit policy classification')
    for name,spec in specs.items():
        known=WRITE_FIELDS if name in manifest['writes'] else READ_FIELDS
        if (not isinstance(spec,dict) or spec.get('name')!=name or not isinstance(spec.get('description'),str)
            or name not in known or spec.get('parameters')!=schema(known[name])
            or type(spec.get('mutates')) is not bool or spec['mutates']!=(name in WRITE_FIELDS)):
            raise CompatibilityError(f'Incompatible executable schema/effect declaration for {name}')
    for name,rule in manifest['writes'].items():
        if (not isinstance(rule,dict) or set(rule)!={'statuses','once'} or type(rule['once']) is not bool
            or not isinstance(rule['statuses'],list) or any(not isinstance(s,str) for s in rule['statuses'])):
            raise CompatibilityError(f'Unsupported mutation policy {name}')
        required=({'delivered'} if name.startswith(('return_','exchange_')) else
                  set() if name=='modify_user_address' else {'pending'})
        if set(rule['statuses'])!=required: raise CompatibilityError(f'Unsafe status policy {name}')
        if name in ('modify_pending_order_items','return_delivered_order_items','exchange_delivered_order_items') and not rule['once']:
            raise CompatibilityError(f'{name} must enforce once-per-order semantics')
    return CompiledPolicy(system,copy.deepcopy(manifest),specs)


class CompiledPolicy:
    def __init__(self,system,manifest,specs):
        self.system,self.manifest,self.specs=system,manifest,specs
        self.version=manifest['version'];self.ttl=manifest['approval_ttl_seconds']
        self.coverage={'read_tools':list(manifest['reads']),'protected_mutations':list(manifest['writes']),
                       'unsupported_tools':[], 'policy_version':self.version}

    def validate(self,context,tool,args):
        context.validate()
        if tool not in self.specs: raise Denied('unknown_tool','Tool not exposed by compiled registry')
        validate_arguments(self.specs[tool]['parameters'],args)

    def plan(self,context,tool,args):
        self.validate(context,tool,args)
        if tool not in self.manifest['writes']: raise Denied('read_only','Read operations do not issue mutation plans')
        snapshot=self.system.snapshot();data=snapshot['resources'];versions=snapshot['versions']
        userkey='user:'+context.customer_id
        if userkey not in data:raise Denied('ownership','Unknown trusted customer')
        user=copy.deepcopy(data[userkey]);used={userkey:versions[userkey]};effects={}
        quote={'amount_cents':0,'currency':'USD'}
        if tool=='modify_user_address':
            if args['user_id']!=context.customer_id:raise Denied('ownership','Cannot modify another customer')
            user['address']={k:args[k] for k in ADDRESS};effects[userkey]=user
        else:
            orderkey='order:'+args['order_id'];order=copy.deepcopy(data.get(orderkey))
            if not order or order['user_id']!=context.customer_id:raise Denied('ownership','Order does not belong to trusted customer')
            original_order=copy.deepcopy(order)
            used[orderkey]=versions[orderkey]
            rule=self.manifest['writes'][tool]
            if order['status'] not in rule['statuses']:raise Denied('order_status','Operation is not allowed in current order status')
            if rule['once'] and tool in order.get('operations',[]):raise Denied('already_processed','This operation was already performed')
            if tool=='cancel_pending_order':
                if args['reason'] not in ('no longer needed','ordered by mistake'):raise Denied('invalid_reason','Unsupported cancellation reason')
                quote.update(amount_cents=sum(i['price_cents'] for i in order['items']),payment_method_id=order['payment_method_id'])
                order['status']='cancelled'
            elif tool=='modify_pending_order_address':order['address']={k:args[k] for k in ADDRESS}
            elif tool=='modify_pending_order_payment':
                self._payment(user,args['payment_method_id'])
                if args['payment_method_id']==order['payment_method_id']:
                    raise Denied('same_payment_method','New payment method must differ from the existing method')
                quote.update(amount_cents=sum(i['price_cents'] for i in order['items']),payment_method_id=args['payment_method_id'])
                order['payment_method_id']=args['payment_method_id']
            else:
                selected,remaining=self._select(order['items'],args['item_ids'])
                payment=self._payment(user,args['payment_method_id'])
                old_total=sum(i['price_cents'] for i in selected)
                if tool=='return_delivered_order_items':
                    if args['payment_method_id']!=order['payment_method_id'] and payment['type']!='gift_card':
                        raise Denied('refund_destination','Refund must use original method or an owned gift card')
                    quote.update(amount_cents=old_total,payment_method_id=args['payment_method_id'])
                    order['status']='return requested';order['return_items']=args['item_ids']
                else:
                    if len(args['new_item_ids'])!=len(selected):raise Denied('item_count','Replacement count must match selected items')
                    replacements=[]
                    for old,new_id in zip(selected,args['new_item_ids']):
                        match=next(((k,p['variants'][new_id]) for k,p in data.items() if k.startswith('product:') and new_id in p['variants']),None)
                        if not match:raise Denied('unknown_variant','Replacement variant does not exist')
                        productkey,new=match;used[productkey]=versions[productkey]
                        if new['product_id']!=old['product_id']:raise Denied('product_mismatch','Replacement must have the same product type')
                        if new.get('available') is not True:raise Denied('unavailable','Replacement is unavailable')
                        replacements.append({k:new[k] for k in ('item_id','product_id','price_cents')})
                    difference=sum(i['price_cents'] for i in replacements)-old_total
                    if difference<0 and args['payment_method_id']!=order['payment_method_id'] and payment['type']!='gift_card':
                        raise Denied('refund_destination','Price-difference refund requires original method or owned gift card')
                    if difference>0 and payment['type']=='gift_card' and payment.get('balance_cents',0)<difference:
                        raise Denied('insufficient_funds','Gift-card balance is insufficient')
                    quote.update(amount_cents=difference,payment_method_id=args['payment_method_id'])
                    if tool=='exchange_delivered_order_items':
                        order['status']='exchange requested';order['exchange_items']=args['new_item_ids']
                    else:order['items']=remaining+replacements
            order.setdefault('operations',[]).append(tool);effects[orderkey]=order
            if settle(user,original_order,order,tool,quote):effects[userkey]=user
        return ActionPlan(tool,copy.deepcopy(args),context.customer_id,context.session_id,self.version,quote,used,effects)

    @staticmethod
    def _payment(user,identifier):
        payment=user.get('payment_methods',{}).get(identifier)
        if not payment:raise Denied('payment_ownership','Payment method is not owned by this customer')
        return payment

    @staticmethod
    def _select(items,identifiers):
        remaining=copy.deepcopy(items);selected=[]
        for identifier in identifiers:
            position=next((i for i,v in enumerate(remaining) if v['item_id']==identifier),None)
            if position is None:raise Denied('item_ownership','Selected item/quantity is not in order')
            selected.append(remaining.pop(position))
        return selected,remaining
