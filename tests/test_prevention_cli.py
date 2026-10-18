import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from agentrx.data import normalize
from tool_prevention.contracts import Context, CompatibilityError
from tool_prevention.gateway import rewire
from tool_prevention.retail import RetailSystem


class PreventionEndToEndTests(unittest.TestCase):
    def test_cli_smoke_produces_usable_diagnostic_trace_and_exactly_one_effect(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=subprocess.run([sys.executable,'-m','tool_prevention','smoke','--out',tmp],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads((Path(tmp)/'report.json').read_text())
            self.assertEqual(report['unapproved']['status'],'denied')
            self.assertEqual(report['approved']['status'],'succeeded')
            self.assertTrue(report['retry']['cached'])
            self.assertEqual(report['side_effects'],1)
            trace=json.loads((Path(tmp)/'trace.json').read_text())
            self.assertTrue(normalize(trace,'tau')['steps'])

    def test_controlled_evaluation_separates_valid_controls_and_unsafe_effects(self):
        from tool_prevention.evaluation import evaluate
        with tempfile.TemporaryDirectory() as tmp:
            report=evaluate(tmp)
        self.assertEqual(set(report['conditions']),{'unprotected','schema_only','prevention'})
        full=report['conditions']['prevention']
        self.assertEqual(full['unsafe_side_effects'],0)
        self.assertEqual(full['false_rejections'],0)
        self.assertGreater(full['valid_completed'],0)
        self.assertGreater(report['conditions']['schema_only']['unsafe_side_effects'],0)
        self.assertFalse(report['paper_benchmark'])

    def test_historical_screen_treats_absent_capabilities_as_unknown(self):
        from tool_prevention.evaluation import screen_trajectories
        trace={'trajectory_id':'old','domain':'tau','steps':[{'index':1,'substeps':[{'tool_calls':[
            {'name':'cancel_pending_order','arguments':{'order_id':'123','reason':'no longer needed'}}]}]}]}
        report=screen_trajectories([trace])
        self.assertEqual(report['trajectories'][0]['calls'][0]['approval'],'unknown')
        self.assertEqual(report['proven_prevented_failures'],0)
        self.assertEqual(report['recovered_tasks'],None)

    def test_project_dispatcher_checks_proposed_actions_before_backend(self):
        from tool_prevention.adapters import ProtectedDispatcher
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            system=RetailSystem.create(Path(tmp)/'state.db')
            protected=rewire(system,store_path=Path(tmp)/'grants.db')
            dispatch=ProtectedDispatcher(protected,Context('s','alice'))
            action=SimpleNamespace(name='cancel_pending_order',kwargs={'order_id':'pending-alice','reason':'no longer needed','operation_id':'op'})
            self.assertEqual(dispatch.step(action)['status'],'denied')
            self.assertEqual(system.side_effect_count(),0)
            with self.assertRaises(CompatibilityError):dispatch.step(SimpleNamespace(steps=[]))


if __name__=='__main__':unittest.main()

class PreventionOperatorCliTests(unittest.TestCase):
    def test_persistent_request_preview_approve_execute_audit_flow(self):
        import os
        with tempfile.TemporaryDirectory() as tmp:
            paths=['--state',str(Path(tmp)/'retail.db'),'--store',str(Path(tmp)/'grants.db')]
            identity=['--session','operator-test','--customer','alice']
            arguments=['--arguments',json.dumps({'order_id':'pending-alice','reason':'no longer needed'})]
            def run(command,*args,env=None):
                result=subprocess.run([sys.executable,'-m','tool_prevention',command,*paths,*args],capture_output=True,text=True,env=env)
                self.assertEqual(result.returncode,0,result.stderr)
                return json.loads(result.stdout)
            request=run('request','cancel_pending_order',*identity,*arguments)
            preview=run('approve',request['request_id'])
            self.assertEqual(preview['state'],'pending')
            approved=run('approve',request['request_id'],'--decision','approve')
            token=approved['approval_token']
            env={**os.environ,'PREVENTION_APPROVAL_TOKEN':token}
            result=run('call','cancel_pending_order',*identity,*arguments,'--operation-id','op',env=env)
            self.assertEqual(result['status'],'succeeded')
            self.assertNotIn(token,json.dumps(run('audit')))
            self.assertTrue(run('audit','--trace',*identity)['steps'])
