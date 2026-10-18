import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from repair_strategies import cli

from .helpers import ORDER, RunBuilder, authenticated

ROOT = Path(__file__).resolve().parents[2]


def invoke(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class CliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        b = authenticated(RunBuilder('cli-1'))
        b.tool('returns_agent', 'get_order_details', ORDER, order_id='#W1000005')
        self.step = b.reply('returns_agent', 'The refund for the water bottle is $54.04.')
        self.trace = self.dir / 'trace.json'
        self.trace.write_text(json.dumps(b.ir()))

    def tearDown(self):
        self.tmp.cleanup()

    def test_catalog_and_registry(self):
        code, out, _ = invoke('catalog', '--category', 'cycles')
        self.assertEqual([r['id'] for r in json.loads(out)],
                         ['loop_nudge', 'loop_force_change', 'loop_replan', 'loop_rollback'])
        target = self.dir / 'registry.json'
        code, out, _ = invoke('registry', '--out', str(target))
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(target.read_text())['schema_version'], 1)

    def test_plan_from_an_fa_result_file_and_lower_it(self):
        result = self.dir / 'result.json'
        result.write_text(json.dumps({'status': 'ok', 'prediction': {
            'step_index': self.step, 'agent': 'returns_agent', 'category': 'InventionOfNewInformation',
            'rationale': 'wrong refund'}}))
        plan_file = self.dir / 'plan.json'
        code, out, _ = invoke('plan', '--trace', str(self.trace), '--hypothesis', str(result),
                              '--target', 'benchmark', '--out', str(plan_file))
        data = json.loads(out)
        self.assertEqual((code, data['plan']['strategy_id']), (0, 'grounded_replacement'))
        self.assertEqual(data['lowered']['kind'], 'guidance')
        self.assertIn('$45.04', data['lowered']['text'])
        code, out, _ = invoke('check', '--plan', str(plan_file), '--rollout', str(self.trace))
        self.assertEqual(code, 1)                               # the original run still has the wrong claim
        self.assertFalse(json.loads(out)['passed'])

    def test_explain_history_and_exhausted_ladder(self):
        hypothesis = json.dumps({'hypotheses': [{'step_index': self.step, 'mode': 'missing_details'}]})
        code, out, _ = invoke('plan', '--trace', str(self.trace), '--hypothesis', hypothesis, '--explain')
        ladder = json.loads(out)['ladder']
        self.assertEqual([r['strategy_id'] for r in ladder], ['complete_omitted_details', 'abstain_or_ask'])
        history = json.dumps([{'strategy_id': s, 'category': 'InventionOfNewInformation', 'rollback_step': 1,
                               'outcome': 'failed'} for s in ('complete_omitted_details', 'abstain_or_ask')])
        code, out, _ = invoke('plan', '--trace', str(self.trace), '--hypothesis', hypothesis, '--history', history)
        self.assertEqual((code, json.loads(out)['plan']), (3, None))

    def test_audit(self):
        code, out, _ = invoke('audit', '--trace', str(self.trace))
        report = json.loads(out)
        self.assertEqual(report['unsupported_claims'][0]['step_index'], self.step)
        self.assertEqual(report['loops'], [])

    def test_usage_errors_exit_2(self):
        cases = [('plan', '--trace', str(self.trace), '--hypothesis', '{not json'),
                 ('plan', '--trace', str(self.trace), '--hypothesis', '{"step_index": 1, "category": "Nope"}'),
                 ('plan', '--trace', str(self.trace), '--hypothesis', '{"hypotheses": []}'),
                 ('plan', '--trace', str(self.dir / 'missing.json'), '--hypothesis', '{}'),
                 ('plan', '--trace', str(self.trace), '--hypothesis', '{"step_index": 1, "category": 3}',
                  '--llm', 'gemini'),
                 ('audit', '--trace', str(self.trace), '--preset', 'airline')]
        for argv in cases:
            code, _, err = invoke(*argv)
            self.assertEqual(code, 2, argv)
            self.assertTrue(err.startswith('error:'), argv)

    def test_llm_presets(self):
        args = cli.build_parser().parse_args(['plan', '--trace', 'x', '--hypothesis', '{}', '--llm',
                                              'ollama:qwen2.5:7b'])
        client = cli.llm_from(args)
        self.assertEqual((client.url, client.model), ('http://127.0.0.1:11434/v1/chat/completions', 'qwen2.5:7b'))
        args.llm_base_url = 'http://gpu-box:11434/v1'
        self.assertEqual(cli.llm_from(args).url, 'http://gpu-box:11434/v1/chat/completions')

    def test_module_entry_point(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        proc = subprocess.run([sys.executable, '-m', 'repair_strategies', 'catalog'], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(len(json.loads(proc.stdout)), 22)


if __name__ == '__main__':
    unittest.main()
