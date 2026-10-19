import asyncio
import copy
import concurrent.futures
import unittest
from tool_middleware import Tool, Middleware, Blocked, PlanError, catalog_hash


def obj(**properties):
    return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}


def shipping():
    calls=[]
    def validate(shipment_number):
        calls.append(('validate',shipment_number));return {'valid':shipment_number=='SHIP-123'}
    result={'status':'in transit'}
    def status(shipment_number):
        calls.append(('status',shipment_number));return result
    schema=obj(shipment_number={'type':'string'})
    tools=[Tool('validate_shipment_number','Validate that this shipment number exists.',schema,validate,
                obj(valid={'type':'boolean'}),'read_only'),
           Tool('get_shipment_status','Requires a valid shipment number.',schema,status,obj(status={'type':'string'}))]
    plan={'version':1,'catalog_hash':catalog_hash(tools),'rules':[{
        'id':'shipment_valid','target':'get_shipment_status','validator':'validate_shipment_number',
        'bindings':{'shipment_number':{'from':'/shipment_number'}},
        'success':[{'pointer':'/valid','equals':True}],
        'confidence':'high','rationale':'Status requires a valid shipment number.'}], 'abstentions':[]}
    return tools,plan,calls,result


class MiddlewareTests(unittest.TestCase):
    def make(self):
        tools,plan,calls,result=shipping()
        return Middleware(tools,plan,validators=['validate_shipment_number']),tools,plan,calls,result

    def test_automatically_validates_before_forwarding_original_arguments_and_result(self):
        layer,tools,plan,calls,result=self.make()
        self.assertIs(layer.dispatch('get_shipment_status',{'shipment_number':'SHIP-123'}),result)
        self.assertEqual(calls,[('validate','SHIP-123'),('status','SHIP-123')])
        self.assertEqual(layer.definitions(),[t.definition() for t in tools])
        self.assertIs(layer.wrapped_tools()['get_shipment_status'](shipment_number='SHIP-123'),result)

    def test_invalid_validation_blocks_before_target_is_called(self):
        layer,_,_,calls,_=self.make()
        with self.assertRaises(Blocked) as error:layer.dispatch('get_shipment_status',{'shipment_number':'BAD'})
        self.assertEqual(error.exception.code,'validation_failed')
        self.assertEqual(calls,[('validate','BAD')])

    def test_no_cross_call_evidence_cache(self):
        layer,_,_,calls,_=self.make()
        for _ in range(2):layer.dispatch('get_shipment_status',{'shipment_number':'SHIP-123'})
        self.assertEqual(sum(name=='validate' for name,_ in calls),2)

    def test_shadow_forwards_without_automatic_validator_calls(self):
        tools,plan,calls,result=shipping()
        layer=Middleware(tools,plan,validators=['validate_shipment_number'],mode='shadow')
        self.assertIs(layer.dispatch('get_shipment_status',{'shipment_number':'BAD'}),result)
        self.assertEqual(calls,[('status','BAD')])
        self.assertEqual(layer.audit()[-1]['decision'],'shadow')

    def test_validator_exception_and_wrong_evidence_type_fail_closed(self):
        for value in ({'valid':1},{},{'valid':True,'unexpected':object()}):
            tools,plan,calls,_=shipping();tools[0].handler=lambda **kw:value
            layer=Middleware(tools,plan,validators=['validate_shipment_number'])
            with self.subTest(value=value),self.assertRaises(Blocked):layer.dispatch('get_shipment_status',{'shipment_number':'SHIP-123'})
            self.assertFalse(any(name=='status' for name,_ in calls))
        def unavailable(**kw):raise TimeoutError('validator unavailable')
        tools[0].handler=unavailable
        layer=Middleware(tools,plan,validators=['validate_shipment_number'])
        with self.assertRaises(Blocked) as error:layer.dispatch('get_shipment_status',{'shipment_number':'SHIP-123'})
        self.assertEqual(error.exception.code,'validator_error')

    def test_unknown_tools_pass_through_and_original_target_exceptions_are_preserved(self):
        tools,plan,calls,_=shipping()
        failure=RuntimeError('original backend failure')
        def failing(**kwargs):raise failure
        extra=Tool('unrelated','Unrelated endpoint',{},failing)
        tools.append(extra);plan['catalog_hash']=catalog_hash(tools)
        layer=Middleware(tools,plan,validators=['validate_shipment_number'])
        with self.assertRaises(RuntimeError) as error:layer.dispatch('unrelated',{'original_field':42})
        self.assertIs(error.exception,failure)
        self.assertEqual(calls,[])

    def test_registry_and_plans_are_snapshotted(self):
        tools,plan,calls,_=shipping();layer=Middleware(tools,plan,validators=['validate_shipment_number'])
        plan['rules'].clear();tools[0].input_schema.clear()
        with self.assertRaises(Blocked):layer.dispatch('get_shipment_status',{'shipment_number':'BAD'})

    def test_compiler_rejects_unsafe_and_unverifiable_rules(self):
        for mode in ('write','not_allowlisted','missing_output','invented_tool','missing_binding','unknown_pointer','cycle','stale_catalog','low_confidence'):
            tools,plan,_,_=shipping();validators=['validate_shipment_number']
            if mode=='write':tools[0].effect='write';plan['catalog_hash']=catalog_hash(tools)
            if mode=='not_allowlisted':validators=[]
            if mode=='missing_output':tools[0].output_schema=None;plan['catalog_hash']=catalog_hash(tools)
            if mode=='invented_tool':plan['rules'][0]['validator']='invented'
            if mode=='missing_binding':plan['rules'][0]['bindings']={}
            if mode=='unknown_pointer':plan['rules'][0]['bindings']['shipment_number']={'from':'/invented'}
            if mode=='cycle':
                plan['rules'][0]['target']='validate_shipment_number'
            if mode=='stale_catalog':plan['catalog_hash']='wrong'
            if mode=='low_confidence':plan['rules'][0]['confidence']='low'
            with self.subTest(mode=mode),self.assertRaises(PlanError):Middleware(tools,plan,validators=validators)

    def test_unrelated_domain_uses_the_same_engine(self):
        calls=[];schema=obj(employee_id={'type':'integer'})
        tools=[Tool('verify_employee','Check employee exists',schema,lambda employee_id:{'exists':employee_id==7},obj(exists={'type':'boolean'}),'read_only'),
               Tool('retrieve_employee_devices','Fetch devices for an existing employee',schema,lambda employee_id:calls.append(employee_id) or ['laptop'])]
        plan={'version':1,'catalog_hash':catalog_hash(tools),'rules':[{'id':'employee_exists','target':'retrieve_employee_devices','validator':'verify_employee',
             'bindings':{'employee_id':{'from':'/employee_id'}},'success':[{'pointer':'/exists','equals':True}],
             'confidence':'high','rationale':'Employee must exist'}],'abstentions':[]}
        layer=Middleware(tools,plan,validators=['verify_employee'])
        self.assertEqual(layer.dispatch('retrieve_employee_devices',{'employee_id':7}),['laptop'])
        with self.assertRaises(Blocked):layer.dispatch('retrieve_employee_devices',{'employee_id':8})
        self.assertEqual(calls,[7])

    def test_sync_target_with_async_prerequisite_gets_an_async_wrapper(self):
        tools,plan,calls,result=shipping()
        async def validate_async(shipment_number):
            await asyncio.sleep(0)
            calls.append(('validate',shipment_number));return {'valid':True}
        tools[0].handler=validate_async
        layer=Middleware(tools,plan,validators=['validate_shipment_number'])
        self.assertIs(asyncio.run(layer.wrapped_tools()['get_shipment_status'](shipment_number='SHIP-123')),result)
        with self.assertRaises(Blocked):layer.dispatch('get_shipment_status',{'shipment_number':'SHIP-123'})

    def test_scope_bindings_cannot_drop_or_replace_a_shared_tenant(self):
        tools,plan,_,_=shipping()
        for tool in tools:
            tool.input_schema=obj(shipment_number={'type':'string'},tenant_id={'type':'string'})
        plan['catalog_hash']=catalog_hash(tools)
        for binding in (None,{'literal':'another-tenant'},{'from':'/shipment_number'}):
            if binding is not None:plan['rules'][0]['bindings']['tenant_id']=binding
            with self.subTest(binding=binding),self.assertRaises(PlanError):Middleware(tools,plan,validators=['validate_shipment_number'])
        plan['rules'][0]['bindings']['tenant_id']={'from':'/tenant_id'}
        self.assertIsNotNone(Middleware(tools,plan,validators=['validate_shipment_number']))

    def test_validator_cannot_mutate_original_nested_arguments(self):
        original={'reference':{'id':'good'}}
        def validate_reference(reference):
            reference['id']='tampered';return {'valid':True}
        seen=[]
        schema=obj(reference=obj(id={'type':'string'}))
        tools=[Tool('check','Validate reference',schema,validate_reference,obj(valid={'type':'boolean'}),'read_only'),
               Tool('fetch','Fetch reference',schema,lambda reference:seen.append(copy.deepcopy(reference)) or reference)]
        plan={'version':1,'catalog_hash':catalog_hash(tools),'rules':[{'id':'r','target':'fetch','validator':'check',
              'bindings':{'reference':{'from':'/reference'}},'success':[{'pointer':'/valid','equals':True}],
              'confidence':'high','rationale':'Check reference first'}],'abstentions':[]}
        layer=Middleware(tools,plan,validators=['check']);layer.dispatch('fetch',original)
        self.assertEqual(original,{'reference':{'id':'good'}})
        self.assertEqual(seen,[{'id':'good'}])

    def test_chains_reuse_identical_evidence_only_within_one_dispatch_and_obey_budget(self):
        tools,plan,calls,_=shipping()
        extra=Tool('check_account','Check account',copy.deepcopy(tools[0].input_schema),
                   lambda shipment_number:calls.append(('account',shipment_number)) or {'ok':True},obj(ok={'type':'boolean'}),'read_only')
        tools.append(extra);plan['catalog_hash']=catalog_hash(tools)
        rule=copy.deepcopy(plan['rules'][0]);rule.update(id='account',target='validate_shipment_number',validator='check_account',success=[{'pointer':'/ok','equals':True}])
        plan['rules'].append(rule)
        duplicate=copy.deepcopy(plan['rules'][0]);duplicate['id']='shipment_valid_second_check';plan['rules'].append(duplicate)
        validators=['validate_shipment_number','check_account']
        layer=Middleware(tools,plan,validators=validators)
        layer.dispatch('get_shipment_status',{'shipment_number':'SHIP-123'})
        self.assertEqual(calls,[('account','SHIP-123'),('validate','SHIP-123'),('status','SHIP-123')])
        calls.clear();limited=Middleware(tools,plan,validators=validators,max_validations=1)
        with self.assertRaises(Blocked) as error:limited.dispatch('get_shipment_status',{'shipment_number':'SHIP-123'})
        self.assertEqual(error.exception.code,'validation_budget');self.assertEqual(calls,[])

    def test_concurrent_top_level_calls_do_not_share_evidence(self):
        layer,_,_,calls,_=self.make()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _:layer.dispatch('get_shipment_status',{'shipment_number':'SHIP-123'}),range(8)))
        self.assertEqual(sum(name=='validate' for name,_ in calls),8)

    def test_malformed_model_fields_are_plan_errors(self):
        for field in ('target','validator'):
            tools,plan,_,_=shipping();plan['rules'][0][field]=[]
            with self.subTest(field=field),self.assertRaises(PlanError):Middleware(tools,plan,validators=['validate_shipment_number'])
