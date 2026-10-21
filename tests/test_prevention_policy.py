import copy
import tempfile
import unittest
from pathlib import Path
from tool_prevention.contracts import Context, CompatibilityError, Denied
from tool_prevention.policy import compile_policy, default_policy
from tool_prevention.retail import RetailSystem


class PreventionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.system=RetailSystem.create(Path(self.temp.name)/'retail.db')
        self.context=Context('session-alice','alice')
        self.policy=compile_policy(self.system,default_policy())

    def test_replay_only_environment_cannot_be_rewired(self):
        class ReplayOnly:
            def rollout(self,*args): pass
        with self.assertRaisesRegex(CompatibilityError,'pre-execution'):
            compile_policy(ReplayOnly(),default_policy())

    def test_compiler_rejects_unknown_tools_and_incompatible_schemas(self):
        p=default_policy();p['writes']['unknown']={'kind':'cancel'}
        with self.assertRaises(CompatibilityError): compile_policy(self.system,p)
        original=self.system.tool_specs
        specs=copy.deepcopy(original())
        specs['return_delivered_order_items']['parameters']['properties'].pop('payment_method_id')
        self.system.tool_specs=lambda:specs
        with self.assertRaises(CompatibilityError): compile_policy(self.system,default_policy())

    def test_plan_binds_customer_exact_arguments_amount_and_versions(self):
        plan=self.policy.plan(self.context,'cancel_pending_order',{'order_id':'pending-alice','reason':'no longer needed'})
        self.assertEqual(plan.quote['amount_cents'],3000)
        self.assertEqual(plan.customer_id,'alice')
        self.assertIn('order:pending-alice',plan.versions)
        changed=self.policy.plan(self.context,'cancel_pending_order',{'order_id':'pending-alice','reason':'ordered by mistake'})
        self.assertNotEqual(plan.binding_hash,changed.binding_hash)

    def test_authentication_ownership_status_and_schema_fail_before_mutation(self):
        examples=[(Context('unauth','alice',False),'pending-alice','no longer needed'),
                  (self.context,'pending-bob','no longer needed'),
                  (self.context,'delivered-alice','no longer needed'),
                  (self.context,'pending-alice',42)]
        for ctx,order,reason in examples:
            with self.subTest(order=order,reason=reason):
                with self.assertRaises(Denied):
                    self.policy.plan(ctx,'cancel_pending_order',{'order_id':order,'reason':reason})
        self.assertEqual(self.system.read_resource('order','pending-alice')['status'],'pending')

    def test_product_and_payment_rules_are_deterministic(self):
        base={'order_id':'delivered-alice','item_ids':['shirt-a'],'payment_method_id':'card-alice'}
        with self.assertRaises(Denied):
            self.policy.plan(self.context,'exchange_delivered_order_items',{**base,'new_item_ids':['shoe-a']})
        with self.assertRaises(Denied):
            self.policy.plan(self.context,'return_delivered_order_items',{**base,'payment_method_id':'card-bob'})
        good=self.policy.plan(self.context,'exchange_delivered_order_items',{**base,'new_item_ids':['shirt-b']})
        self.assertEqual(good.quote['amount_cents'],500)

    def test_backend_conditional_execution_and_idempotent_retries(self):
        args={'order_id':'pending-alice','reason':'no longer needed'}
        p=self.policy.plan(self.context,'cancel_pending_order',args)
        result=self.system.execute_atomic(p,'op-1')
        self.assertEqual(self.system.execute_atomic(p,'op-1'),result)
        self.assertEqual(self.system.side_effect_count(),1)
        with self.assertRaises(Denied): self.system.execute_atomic(p,'different-op')
        self.assertEqual(self.system.lookup_execution('op-1')['status'],'succeeded')

class PreventionPolicyCoverageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.system=RetailSystem.create(Path(self.tmp.name)/'state.db')
        self.policy=compile_policy(self.system,default_policy());self.ctx=Context('s','alice')

    def test_malformed_manifests_and_renamed_tools_fail_with_compatibility_error(self):
        import copy
        for field,value in [('reads',None),('writes',[]),('reads',['get_order_details','get_order_details'])]:
            manifest=default_policy();manifest[field]=value
            with self.subTest(field=field,value=value),self.assertRaises(CompatibilityError):compile_policy(self.system,manifest)
        original=self.system.tool_specs
        specs=original();specs['get_order_details']['name']='cancel_pending_order'
        self.system.tool_specs=lambda:specs
        with self.assertRaises(CompatibilityError):compile_policy(self.system,default_policy())

    def test_every_mutation_has_a_valid_control_and_changes_expected_resource(self):
        from tool_prevention.retail import ADDRESS
        address={k:('' if k=='address2' else 'new-'+k) for k in ADDRESS}
        cases=[
          ('cancel_pending_order',{'order_id':'pending-alice','reason':'no longer needed'},'order:pending-alice'),
          ('modify_pending_order_address',{'order_id':'pending-alice',**address},'order:pending-alice'),
          ('modify_user_address',{'user_id':'alice',**address},'user:alice'),
          ('modify_pending_order_items',{'order_id':'pending-alice','item_ids':['shirt-a'],'new_item_ids':['shirt-b'],'payment_method_id':'card-alice'},'order:pending-alice'),
          ('modify_pending_order_payment',{'order_id':'pending-alice','payment_method_id':'gift-alice'},'order:pending-alice'),
          ('return_delivered_order_items',{'order_id':'delivered-alice','item_ids':['shirt-a'],'payment_method_id':'card-alice'},'order:delivered-alice'),
          ('exchange_delivered_order_items',{'order_id':'delivered-alice','item_ids':['shirt-a'],'new_item_ids':['shirt-b'],'payment_method_id':'card-alice'},'order:delivered-alice'),
        ]
        for number,(tool,args,key) in enumerate(cases):
            with self.subTest(tool=tool):
                system=RetailSystem.create(Path(self.tmp.name)/f'case{number}.db')
                policy=compile_policy(system,default_policy());before=system.snapshot()['resources'][key]
                plan=policy.plan(self.ctx,tool,args);system.execute_atomic(plan,'test')
                self.assertNotEqual(before,system.snapshot()['resources'][key])
                self.assertEqual(system.side_effect_count(),1)

    def test_payment_change_refunds_old_method_and_charges_new_method(self):
        plan=self.policy.plan(self.ctx,'modify_pending_order_payment',{'order_id':'pending-alice','payment_method_id':'gift-alice'})
        self.system.execute_atomic(plan,'payment')
        user=self.system.read_resource('user','alice')
        self.assertEqual(user['payment_methods']['gift-alice']['balance_cents'],7000)
        self.assertEqual([(t['kind'],t['payment_method_id'],t['amount_cents']) for t in user['transactions']],
                         [('refund','card-alice',3000),('charge','gift-alice',3000)])

    def test_all_reads_require_owner_and_do_not_mutate(self):
        valid=[('find_user_id_by_email',{'email':'alice@example.test'}),
               ('find_user_id_by_name_zip',{'first_name':'Alice','last_name':'Example','zip':'12345'}),
               ('list_all_product_types',{}),('get_product_details',{'product_id':'shirt'}),
               ('get_user_details',{'user_id':'alice'}),('get_order_details',{'order_id':'pending-alice'})]
        for tool,args in valid:
            self.policy.validate(self.ctx,tool,args);self.assertIsNotNone(self.system.read(self.ctx,tool,args))
        self.assertEqual(self.system.side_effect_count(),0)
        with self.assertRaises(Denied):self.system.read(self.ctx,'get_user_details',{'user_id':'bob'})

    def test_mixed_tender_item_change_then_cancel_reverses_actual_funding(self):
        args={'order_id':'pending-alice','item_ids':['shirt-a'],'new_item_ids':['shirt-b'],'payment_method_id':'gift-alice'}
        self.system.execute_atomic(self.policy.plan(self.ctx,'modify_pending_order_items',args),'modify')
        self.system.execute_atomic(self.policy.plan(self.ctx,'cancel_pending_order',{'order_id':'pending-alice','reason':'no longer needed'}),'cancel')
        user=self.system.read_resource('user','alice')
        self.assertEqual(user['payment_methods']['gift-alice']['balance_cents'],10000)
        refunds=[(t['payment_method_id'],t['amount_cents']) for t in user['transactions'] if t['kind']=='refund']
        self.assertEqual(refunds,[('card-alice',3000),('gift-alice',500)])

    def test_mixed_tender_payment_replacement_refunds_all_old_allocations(self):
        args={'order_id':'pending-alice','item_ids':['shirt-a'],'new_item_ids':['shirt-b'],'payment_method_id':'gift-alice'}
        self.system.execute_atomic(self.policy.plan(self.ctx,'modify_pending_order_items',args),'modify')
        self.system.execute_atomic(self.policy.plan(self.ctx,'modify_pending_order_payment',{'order_id':'pending-alice','payment_method_id':'gift-alice'}),'payment')
        user=self.system.read_resource('user','alice')
        self.assertEqual(user['payment_methods']['gift-alice']['balance_cents'],6500)
        self.assertEqual(self.system.read_resource('order','pending-alice')['payment_allocations'],[{'payment_method_id':'gift-alice','amount_cents':3500}])
