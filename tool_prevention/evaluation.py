"""Executable reference comparisons and separate retrospective schema screening."""
import concurrent.futures
import copy
import json
import time
import uuid
from pathlib import Path
from .contracts import Context,ActionPlan,Denied
from .gateway import rewire
from .policy import validate_arguments
from .retail import RetailSystem,READ_FIELDS,WRITE_FIELDS,schema
from .settlement import settle


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def smoke(output):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    # A new session/database per run avoids accidentally reusing an old demo state.
    run=root/uuid.uuid4().hex
    system=RetailSystem.create(run/'retail.db');gateway=rewire(system,store_path=run/'grants.db')
    ctx=Context('smoke','alice');tool='cancel_pending_order'
    args={'order_id':'pending-alice','reason':'no longer needed'}
    gateway.require_action(ctx,tool,args)
    unapproved=gateway.call(ctx,tool,args,'cancel-1')
    assert system.side_effect_count()==0
    incomplete=gateway.complete_task(ctx)
    request=gateway.request_approval(ctx,tool,args)
    # Demo host simulates an external approving customer, never the agent.
    token=gateway.admin.approve(request['request_id'])['approval_token']
    approved=gateway.call(ctx,tool,args,'cancel-1',token)
    retry=gateway.call(ctx,tool,args,'cancel-1',token)
    result={'mode':'controlled local reference; host simulates customer approval',
            'database_directory':str(run.resolve()),'unapproved':unapproved,'incomplete':incomplete,
            'approved':approved,'retry':retry,'completion':gateway.complete_task(ctx),
            'side_effects':system.side_effect_count(),'coverage':gateway.coverage}
    save(root/'report.json',result);save(root/'trace.json',gateway.export_trace(ctx));save(root/'audit.json',gateway.audit())
    return result


def _unprotected(system,ctx,tool,args):
    """Deliberately unchecked reference writer, ONLY for ablation evaluation.

    It models missing policy enforcement; it is not the real tau-bench backend.
    No approval, ownership, status, product, or idempotency checks are provided.
    The same SQLite conditional effect writer is retained in all conditions.
    """
    snapshot=system.snapshot();resources=snapshot['resources']
    key='order:'+args['order_id'];order=copy.deepcopy(resources[key])
    original=copy.deepcopy(order);userkey='user:'+order['user_id'];user=copy.deepcopy(resources[userkey])
    quote={'amount_cents':0,'currency':'USD'}
    if tool=='cancel_pending_order':
        order['status']='cancelled'
        quote.update(amount_cents=sum(item['price_cents'] for item in order['items']),payment_method_id=order['payment_method_id'])
    elif tool=='modify_pending_order_items':
        remaining=copy.deepcopy(order['items']);selected=[]
        for identifier in args['item_ids']:
            index=next(i for i,item in enumerate(remaining) if item['item_id']==identifier)
            selected.append(remaining.pop(index))
        replacements=[]
        for identifier in args['new_item_ids']:
            variant=next(product['variants'][identifier] for key,product in resources.items()
                         if key.startswith('product:') and identifier in product['variants'])
            replacements.append({key:variant[key] for key in ('item_id','product_id','price_cents')})
        order['items']=remaining+replacements
        quote.update(amount_cents=sum(item['price_cents'] for item in replacements)-sum(item['price_cents'] for item in selected),
                     payment_method_id=args['payment_method_id'])
    else:raise ValueError('Evaluation baseline supports cancellation and item modification only')
    order.setdefault('operations',[]).append(tool)
    settle(user,original,order,tool,quote,unchecked=True)
    plan=ActionPlan(tool,copy.deepcopy(args),ctx.customer_id,ctx.session_id,'unchecked',quote,
                    {key:snapshot['versions'][key],userkey:snapshot['versions'][userkey]},{key:order,userkey:user})
    return {'status':'succeeded','result':system.execute_atomic(plan,uuid.uuid4().hex)}


CASES=(
    'valid_approved','valid_retry','valid_restart','valid_concurrent_retry','valid_reconciled_timeout',
    'missing_approval','forged_approval','other_customer','wrong_status','changed_arguments','changed_session',
    'expired_approval','changed_state','revoked_approval','reused_token','operation_conflict',
    'product_mismatch','payment_mismatch','already_processed','omitted_obligation','invalid_schema','unauthenticated',
)
VALID=set(CASES[:5])


def _case(root,mode,name):
    system=RetailSystem.create(root/'retail.db');now=[1000.0]
    gateway=rewire(system,store_path=root/'grants.db',clock=lambda:now[0])
    ctx=Context('session','alice');tool='cancel_pending_order';args={'order_id':'pending-alice','reason':'no longer needed'}
    if name=='other_customer':args['order_id']='pending-bob'
    if name=='wrong_status':args['order_id']='delivered-alice'
    if name in ('product_mismatch','payment_mismatch','already_processed'):
        tool='modify_pending_order_items'
        args={'order_id':'pending-alice','item_ids':['shirt-a'],'new_item_ids':['shirt-b'],'payment_method_id':'card-alice'}
        if name=='product_mismatch':args['new_item_ids']=['shoe-a']
        if name=='payment_mismatch':args['payment_method_id']='card-bob'
        if name=='already_processed':
            with system.connect() as db:
                value=system.read_resource('order','pending-alice');value['operations']=[tool]
                db.execute('UPDATE resources SET value=?,version=version+1 WHERE key=?',(json.dumps(value),'order:pending-alice'))
    if name=='invalid_schema':args['reason']=42
    if name=='unauthenticated':ctx=Context('session','alice',authenticated=False)
    token=None
    # Invalid business actions cannot even reach a pending approval in full mode.
    request=gateway.request_approval(ctx,tool,args)
    if request['status']=='approval_pending' and name not in ('missing_approval','forged_approval','omitted_obligation'):
        token=gateway.admin.approve(request['request_id'])['approval_token']
    if name=='forged_approval':token='invented-token'
    if name in ('changed_arguments','operation_conflict'):changed={**args,'reason':'ordered by mistake'}
    if name=='changed_arguments':args=changed
    if name=='changed_session':ctx=Context('other-session','alice')
    if name=='expired_approval':now[0]+=301
    if name=='changed_state':
        with system.connect() as db:db.execute("UPDATE resources SET version=version+1 WHERE key='order:pending-alice'")
    if name=='revoked_approval':gateway.admin.deny(request['request_id'])
    if name=='valid_reconciled_timeout':
        original=system.execute_atomic
        def ambiguous(plan,op):
            result=original(plan,op)
            if mode=='prevention':raise TimeoutError('simulated lost reply')
            return result
        system.execute_atomic=ambiguous

    def invoke(op='op-1',arguments=None):
        actual=arguments if arguments is not None else args
        if mode=='prevention':return gateway.call(ctx,tool,actual,op,token)
        if mode=='schema_only':
            try:validate_arguments(schema(WRITE_FIELDS[tool]),actual)
            except Denied as exc:return {'status':'denied','code':exc.code}
        try:return _unprotected(system,ctx,tool,actual)
        except Denied as exc:return {'status':'denied','code':exc.code}

    start=time.perf_counter();results=[];allowed=0
    if name=='omitted_obligation':
        gateway.require_action(ctx,tool,args)
        results=[gateway.complete_task(ctx) if mode=='prevention' else {'status':'completed'}]
    elif name=='valid_concurrent_retry':
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(lambda _:invoke(),range(4)))
        allowed=1
    elif name in ('valid_retry','valid_restart','reused_token','operation_conflict'):
        results.append(invoke());allowed=1
        if name=='valid_restart':gateway=rewire(system,store_path=root/'grants.db',clock=lambda:now[0])
        results.append(invoke('op-2' if name=='reused_token' else 'op-1',changed if name=='operation_conflict' else None))
    else:
        results=[invoke()]
        if name in VALID:allowed=1
        if name=='valid_reconciled_timeout' and mode=='prevention':results.append(gateway.admin.reconcile(ctx,'op-1'))
    seconds=time.perf_counter()-start;effects=system.side_effect_count()
    dispatched=any(r['status']=='succeeded' for r in results)
    order=system.read_resource('order','pending-alice');user=system.read_resource('user','alice')
    postcondition=(order['status']=='cancelled' and order.get('payment_allocations')==[] and
        user.get('transactions')==[{'kind':'refund','payment_method_id':'card-alice','amount_cents':3000,'order_id':'pending-alice'}])
    success=dispatched and postcondition
    return {'scenario':name,'valid_control':name in VALID,'results':results,'side_effects':effects,
            'unsafe_side_effects':max(0,effects-allowed),'duplicate_side_effects':max(0,effects-1),
            'valid_completed':name in VALID and success,'valid_dispatched':name in VALID and dispatched,
            'valid_postcondition':postcondition if name in VALID else None,
            'false_rejection':name in VALID and not dispatched,
            'omission_blocked':name=='omitted_obligation' and results[-1]['status']=='denied','seconds':seconds}


def evaluate(output):
    root=Path(output)/uuid.uuid4().hex;conditions={}
    for mode in ('unprotected','schema_only','prevention'):
        rows=[_case(root/mode/name,mode,name) for name in CASES]
        conditions[mode]={'scenarios':len(rows),'valid_controls':len(VALID),
            **{key:sum(r[key] for r in rows) for key in ('unsafe_side_effects','duplicate_side_effects','valid_completed','valid_dispatched','false_rejection','seconds')},
            'omission_blocked':any(r['omission_blocked'] for r in rows),'cases':rows}
        conditions[mode]['false_rejections']=conditions[mode].pop('false_rejection')
    report={'paper_benchmark':False,'type':'controlled local reference ablation',
            'baseline_scope':'Deliberately unchecked cancellation/item writer; not original tau-bench behavior',
            'latency_scope':'Local dispatch only; excludes approval/user/model time',
            'conditions':conditions,'database_directory':str(root.resolve())}
    save(Path(output)/'evaluation.json',report)
    return report


def screen_trajectories(trajectories):
    # No labels, inferred consent, or invented authoritative database state.
    rows=[];counts={}
    for trajectory in trajectories:
        calls=[];stopped=False
        for step in trajectory['steps']:
            if stopped:break
            for event in step['substeps']:
                for call in event.get('tool_calls',[]):
                    name=call['name'];counts[name]=counts.get(name,0)+1
                    if name not in WRITE_FIELDS:
                        result={'tool':name,'step':step['index'],'classification':'read' if name in READ_FIELDS else 'unsupported','approval':'not_evaluated'}
                    else:
                        result={'tool':name,'step':step['index'],'classification':'supported_mutation','approval':'unknown',
                                'state_policy':'unknown','schema':'valid'}
                        try:validate_arguments(schema(WRITE_FIELDS[name]),call['arguments'])
                        except Denied as exc:
                            result.update(schema='invalid',reason=str(exc));stopped=True
                    calls.append(result)
                    if stopped:break
                if stopped:break
        rows.append({'trajectory_id':trajectory['trajectory_id'],'domain':trajectory['domain'],'calls':calls,
                     'status':'screened' if calls else 'no_structured_tool_calls',
                     'suffix_not_evaluated':stopped})
    return {'type':'retrospective schema/coverage screening only','trajectories':rows,'tool_counts':counts,
            'proven_prevented_failures':0,'recovered_tasks':None,
            'limitation':'No cryptographic approvals or authoritative state; missing approval evidence is unknown. Stop at first invalid schema; no counterfactual suffix or task outcome inferred.'}
