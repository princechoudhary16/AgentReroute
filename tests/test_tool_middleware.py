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
        layer=Middleware(tools,plan,validators=['check'])
        with self.assertRaises(Blocked) as error:layer.dispatch('fetch',original)
        self.assertEqual(error.exception.code,'validator_input_mutated')
        self.assertEqual(original,{'reference':{'id':'good'}})
        self.assertEqual(seen,[])

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

    def test_existing_dispatcher_can_be_rewired_from_unchanged_openai_or_mcp_metadata(self):
        from tool_middleware.adapters import from_dispatcher
        tools,_,calls,_=shipping()
        definitions=[{'type':'function','function':{'name':t.name,'description':t.description,'parameters':copy.deepcopy(t.input_schema)}} for t in tools]
        outputs={t.name:t.output_schema for t in tools}
        original=copy.deepcopy(definitions)
        handlers={t.name:t.handler for t in tools}
        adapted=from_dispatcher(definitions,lambda name,args:handlers[name](**args),
                                effects={'validate_shipment_number':'read_only'},output_schemas=outputs)
        _,plan,_,_=shipping();plan['catalog_hash']=catalog_hash(adapted)
        layer=Middleware(adapted,plan,validators=['validate_shipment_number'])
        self.assertEqual(layer.dispatch('get_shipment_status',{'shipment_number':'SHIP-123'}),{'status':'in transit'})
        self.assertEqual(definitions,original)
        self.assertEqual(calls,[('validate','SHIP-123'),('status','SHIP-123')])

    def test_async_callable_objects_are_supported_by_wrapped_registry(self):
        tools,plan,calls,result=shipping()
        class AsyncValidator:
            async def __call__(self,shipment_number):return {'valid':True}
        tools[0].handler=AsyncValidator()
        layer=Middleware(tools,plan,validators=['validate_shipment_number'])
        self.assertIs(asyncio.run(layer.wrapped_tools()['get_shipment_status'](shipment_number='SHIP-123')),result)

    def test_excessively_deep_model_graph_is_rejected_without_recursion_failure(self):
        schema=obj(identifier={'type':'string'});output=obj(ok={'type':'boolean'})
        tools=[Tool(f'tool_{n}','Declared tool',schema,lambda **kw:{'ok':True},output,'read_only') for n in range(513)]
        rules=[{'id':f'r{n}','target':f'tool_{n}','validator':f'tool_{n+1}',
                'bindings':{'identifier':{'from':'/identifier'}},'success':[{'pointer':'/ok','equals':True}],
                'confidence':'high','rationale':'Declared prerequisite'} for n in range(512)]
        plan={'version':1,'catalog_hash':catalog_hash(tools),'rules':rules,'abstentions':[]}
        with self.assertRaises(PlanError):Middleware(tools,plan,validators=[t.name for t in tools])

    def test_original_positional_and_positional_only_invocations_are_preserved(self):
        layer,tools,plan,calls,result=self.make()
        self.assertIs(layer.wrapped_tools()['get_shipment_status']('SHIP-123'),result)
        def positional_only(shipment_number, /):return result
        tools[1].handler=positional_only
        layer=Middleware(tools,plan,validators=['validate_shipment_number'])
        self.assertIs(layer.wrapped_tools()['get_shipment_status']('SHIP-123'),result)
        tools[1].handler=lambda reference:reference
        plan['rules']=[]
        layer=Middleware(tools,plan,validators=['validate_shipment_number'])
        self.assertEqual(layer.wrapped_tools()['get_shipment_status']('unchanged'),'unchanged')

    def test_optional_validator_tenant_cannot_default_to_a_different_scope(self):
        tools,plan,_,_=shipping()
        for tool in tools:tool.input_schema=obj(shipment_number={'type':'string'},tenant_id={'type':'string'})
        tools[0].input_schema['required']=['shipment_number']
        tools[0].input_schema['properties']['tenant_id']['default']='A'
        plan['catalog_hash']=catalog_hash(tools)
        with self.assertRaises(PlanError):Middleware(tools,plan,validators=['validate_shipment_number'])
        plan['rules'][0]['bindings']['tenant_id']={'from':'/tenant_id'}
        self.assertIsNotNone(Middleware(tools,plan,validators=['validate_shipment_number']))

    def test_bound_values_must_match_source_constraints_before_validation(self):
        tools,plan,calls,_=shipping()
        tools[1].input_schema=obj(shipment_number={'type':'string','const':'SHIP-123'})
        tools[0].handler=lambda **kwargs:{'valid':True};plan['catalog_hash']=catalog_hash(tools)
        layer=Middleware(tools,plan,validators=['validate_shipment_number'])
        with self.assertRaises(Blocked):layer.dispatch('get_shipment_status',{'shipment_number':'BAD'})
        self.assertEqual(calls,[])

    def test_async_changes_to_protected_arguments_block_the_target(self):
        async def exercise():
            tools,plan,calls,_=shipping();started=asyncio.Event();release=asyncio.Event()
            async def validate(shipment_number):
                started.set();await release.wait();return {'valid':True}
            tools[0].handler=validate
            layer=Middleware(tools,plan,validators=['validate_shipment_number'])
            args={'shipment_number':'SHIP-123'}
            task=asyncio.create_task(layer.adispatch('get_shipment_status',args))
            await started.wait();args['shipment_number']='BAD';release.set()
            with self.assertRaises(Blocked) as error:await task
            self.assertEqual(error.exception.code,'arguments_changed')
            self.assertEqual(calls,[])
        asyncio.run(exercise())

    def test_valid_large_json_integer_in_validator_output_does_not_overflow(self):
        tools,plan,_,result=shipping()
        tools[0].output_schema=obj(valid={'type':'boolean'},count={'type':'integer'})
        tools[0].handler=lambda **kwargs:{'valid':True,'count':10**400}
        plan['catalog_hash']=catalog_hash(tools)
        layer=Middleware(tools,plan,validators=['validate_shipment_number'])
        self.assertIs(layer.dispatch('get_shipment_status',{'shipment_number':'SHIP-123'}),result)

    def test_shadow_does_not_require_inspectable_positional_signatures(self):
        tools,plan,calls,result=shipping()
        class Uninspectable:
            @property
            def __signature__(self):raise ValueError('Extension callable has no signature')
            def __call__(self,*args,**kwargs):return result
        tools[1].handler=Uninspectable()
        layer=Middleware(tools,plan,validators=['validate_shipment_number'],mode='shadow')
        self.assertIs(layer.wrapped_tools()['get_shipment_status']('BAD'),result)
        self.assertEqual(calls,[])
