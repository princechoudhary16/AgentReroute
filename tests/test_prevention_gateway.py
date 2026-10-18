import concurrent.futures
import json
import tempfile
import unittest
from pathlib import Path
from tool_prevention.contracts import Context
from tool_prevention.gateway import rewire
from tool_prevention.retail import RetailSystem


class PreventionGatewayTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.system=RetailSystem.create(self.root/'retail.db')
        self.now=[1000.0]
        self.gateway=rewire(self.system,store_path=self.root/'grants.db',clock=lambda:self.now[0])
        self.ctx=Context('alice-session','alice')
        self.tool='cancel_pending_order';self.args={'order_id':'pending-alice','reason':'no longer needed'}

    def grant(self,args=None,ctx=None):
        request=self.gateway.request_approval(ctx or self.ctx,self.tool,args or self.args)
        self.assertEqual(request['status'],'approval_pending')
        return self.gateway.admin.approve(request['request_id'])['approval_token']

    def invoke(self,token=None,args=None,ctx=None,op='op-1',gateway=None):
        return (gateway or self.gateway).call(ctx or self.ctx,self.tool,args or self.args,op,token)

    def test_registry_denies_missing_forged_tokens_and_exposes_no_approval_authority(self):
        tools=self.gateway.tools(self.ctx)
        self.assertNotIn('approve',tools)
        self.assertIn('request_action_approval',tools)
        self.assertEqual(self.invoke()['status'],'denied')
        self.assertEqual(self.invoke('invented')['status'],'denied')
        self.assertEqual(self.system.side_effect_count(),0)
        self.assertEqual(self.gateway.call(self.ctx,'approve',{},'bad')['code'],'unknown_tool')

    def test_exact_approval_executes_once_and_retry_uses_cached_result(self):
        token=self.grant()
        first=self.invoke(token)
        self.assertEqual(first['status'],'succeeded')
        second=self.invoke(token)
        self.assertEqual(second['status'],'succeeded')
        self.assertTrue(second['cached'])
        self.assertEqual(self.system.side_effect_count(),1)
        self.assertEqual(self.system.read_resource('order','pending-alice')['status'],'cancelled')
        audit=json.dumps(self.gateway.audit())
        self.assertNotIn(token,audit)
        self.assertNotIn(token,json.dumps(self.gateway.export_trace(self.ctx)))

    def test_approval_is_bound_to_session_arguments_expiry_and_current_versions(self):
        for mode in ('session','arguments','expiry','state'):
            with self.subTest(mode=mode):
                token=self.grant()
                kwargs={}
                if mode=='session':kwargs['ctx']=Context('other-session','alice')
                if mode=='arguments':kwargs['args']={**self.args,'reason':'ordered by mistake'}
                if mode=='expiry':self.now[0]+=301
                if mode=='state':
                    with self.system.connect() as db:
                        db.execute("UPDATE resources SET version=version+1 WHERE key='order:pending-alice'")
                self.assertEqual(self.invoke(token,op=mode,**kwargs)['status'],'denied')
                self.assertEqual(self.system.side_effect_count(),0)

    def test_pending_request_cannot_be_modified_or_self_approved(self):
        args=dict(self.args)
        request=self.gateway.request_approval(self.ctx,self.tool,args)
        args['reason']='ordered by mistake'
        token=self.gateway.admin.approve(request['request_id'])['approval_token']
        self.assertEqual(self.invoke(token,args=args)['status'],'denied')
        self.assertEqual(self.invoke(token)['status'],'succeeded')

    def test_concurrent_claims_across_gateway_instances_allow_one_effect(self):
        token=self.grant()
        def attempt(number):
            other=rewire(self.system,store_path=self.root/'grants.db',clock=lambda:self.now[0])
            return self.invoke(token,op=f'parallel-{number}',gateway=other)
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            results=list(pool.map(attempt,range(6)))
        self.assertEqual(sum(r['status']=='succeeded' for r in results),1)
        self.assertEqual(self.system.side_effect_count(),1)

    def test_restart_preserves_consumption_and_operation_identity(self):
        token=self.grant();self.assertEqual(self.invoke(token)['status'],'succeeded')
        restarted=rewire(self.system,store_path=self.root/'grants.db',clock=lambda:self.now[0])
        self.assertTrue(self.invoke(token,gateway=restarted)['cached'])
        changed={**self.args,'reason':'ordered by mistake'}
        self.assertEqual(self.invoke(token,args=changed,gateway=restarted)['code'],'operation_conflict')
        self.assertEqual(self.system.side_effect_count(),1)

    def test_unknown_completion_blocks_retry_until_trusted_reconciliation(self):
        original=self.system.execute_atomic
        def ambiguous(plan,operation_id):
            original(plan,operation_id)
            raise TimeoutError('reply lost after commit')
        self.system.execute_atomic=ambiguous
        token=self.grant();first=self.invoke(token)
        self.assertEqual(first['status'],'indeterminate')
        self.assertEqual(self.invoke(token)['status'],'indeterminate')
        self.assertEqual(self.system.side_effect_count(),1)
        reconciled=self.gateway.admin.reconcile(self.ctx,'op-1')
        self.assertEqual(reconciled['status'],'succeeded')
        self.assertTrue(self.invoke(token)['cached'])
        self.assertEqual(self.system.side_effect_count(),1)

    def test_stale_state_between_check_and_execute_is_conditionally_rejected(self):
        original=self.system.execute_atomic
        def change_first(plan,operation_id):
            with self.system.connect() as db:
                db.execute("UPDATE resources SET version=version+1 WHERE key='order:pending-alice'")
            return original(plan,operation_id)
        self.system.execute_atomic=change_first
        token=self.grant();result=self.invoke(token)
        self.assertEqual(result['status'],'denied')
        self.assertEqual(result['code'],'stale_state')
        self.assertEqual(self.system.side_effect_count(),0)

    def test_explicit_host_obligation_blocks_completion_until_action_succeeds(self):
        self.gateway.require_action(self.ctx,self.tool,self.args)
        self.assertEqual(self.gateway.complete_task(self.ctx)['status'],'denied')
        token=self.grant();self.invoke(token)
        self.assertEqual(self.gateway.complete_task(self.ctx)['status'],'completed')
        self.assertEqual(self.gateway.complete_task(Context('other','alice'))['status'],'completed')

    def test_tokens_are_redacted_even_in_invalid_nested_arguments(self):
        token=self.grant()
        invalid={**self.args,'nested':{'approval_token':token,'unexpected_value':token}}
        self.assertEqual(self.invoke(token,args=invalid)['status'],'denied')
        self.assertNotIn(token,json.dumps(self.gateway.audit()))

    def test_approved_request_can_be_revoked_before_execution(self):
        request=self.gateway.request_approval(self.ctx,self.tool,self.args)
        token=self.gateway.admin.approve(request['request_id'])['approval_token']
        self.gateway.admin.deny(request['request_id'])
        self.assertEqual(self.invoke(token)['status'],'denied')
        self.assertEqual(self.system.side_effect_count(),0)

    def test_cached_retries_and_revocations_are_audited(self):
        token=self.grant();self.invoke(token);self.invoke(token)
        self.assertEqual(self.gateway.audit()[-1]['decision'],'cached')
        other={'order_id':'pending-alice','payment_method_id':'gift-alice'}
        # New request before any mutation on this separate customer's address.
        from tool_prevention.retail import ADDRESS
        args={'user_id':'alice',**{k:('' if k=='address2' else 'x') for k in ADDRESS}}
        request=self.gateway.request_approval(self.ctx,'modify_user_address',args)
        self.gateway.admin.deny(request['request_id'])
        self.assertEqual(self.gateway.audit()[-1]['decision'],'approval_denied')

    def test_grant_journal_cannot_be_reused_with_another_backend(self):
        from tool_prevention.contracts import CompatibilityError
        token=self.grant()
        other=RetailSystem.create(self.root/'different-retail.db')
        with self.assertRaises(CompatibilityError):rewire(other,store_path=self.root/'grants.db')
        self.assertEqual(other.side_effect_count(),0)
        self.assertEqual(self.invoke(token)['status'],'succeeded')

    def test_late_timeout_cannot_overwrite_reconciled_success(self):
        original=self.system.execute_atomic
        def reconciled_before_timeout(plan,operation_id):
            original(plan,operation_id)
            self.assertEqual(self.gateway.admin.reconcile(self.ctx,'op-1')['status'],'succeeded')
            raise TimeoutError('reply arrived after reconciliation')
        self.system.execute_atomic=reconciled_before_timeout
        token=self.grant();self.invoke(token)
        self.assertEqual(self.invoke(token)['status'],'succeeded')
        self.assertEqual(self.system.side_effect_count(),1)

    def test_tokens_are_redacted_when_used_as_unknown_tool_names_or_argument_keys(self):
        token=self.grant()
        self.gateway.call(self.ctx,token,{token:'value'})
        self.assertNotIn(token,json.dumps(self.gateway.audit()))
