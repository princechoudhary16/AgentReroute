import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class MiddlewareCliTests(unittest.TestCase):
    def run_cli(self,*arguments):
        result=subprocess.run([sys.executable,'-m','tool_middleware',*arguments],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        return json.loads(result.stdout)

    def test_offline_demo_automatically_checks_and_preserves_the_original_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            report=self.run_cli('smoke','--out',tmp)
            self.assertEqual(report['discovery'],'fixture; not a model measurement')
            self.assertEqual(report['valid_result'],{'status':'in transit'})
            self.assertEqual(report['invalid_result']['status'],'blocked')
            self.assertEqual(report['calls'],[['validate','SHIP-123'],['status','SHIP-123'],['validate','BAD']])
            self.assertTrue((Path(tmp)/'plan.json').is_file())
            self.assertTrue((Path(tmp)/'catalog.json').is_file())
            inspected=self.run_cli('inspect','--catalog',str(Path(tmp)/'catalog.json'),'--plan',str(Path(tmp)/'plan.json'),'--validators','validate_shipment_number')
            self.assertEqual(inspected['guarded_tools'],['get_shipment_status'])

    def test_evaluation_uses_the_same_engine_on_two_unrelated_catalogs(self):
        with tempfile.TemporaryDirectory() as tmp:
            report=self.run_cli('evaluate','--out',tmp)
            self.assertFalse(report['paper_benchmark'])
            self.assertEqual(set(report['domains']),{'shipping','employee_devices'})
            for domain,row in report['domains'].items():
                self.assertEqual(row['wrapped']['invalid_target_invocations'],0,domain)
                self.assertGreater(row['original']['invalid_target_invocations'],0,domain)
                self.assertEqual(row['wrapped']['false_blocks'],0,domain)
                self.assertEqual(row['wrapped']['valid_completed'],row['original']['valid_completed'],domain)

    def test_discovery_cli_requires_explicit_validator_designation(self):
        result=subprocess.run([sys.executable,'-m','tool_middleware','discover','--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0)
        self.assertIn('--validators',result.stdout)
