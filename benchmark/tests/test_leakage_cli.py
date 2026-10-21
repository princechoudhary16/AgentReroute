"""Leakage boundary and command-line checks."""
import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
# Code that runs on the submission side must never read gold labels, rewards or injection logs.
SUBMISSION_SIDE = ['scoring/baselines.py', 'env/workflow.py', 'env/llm.py', 'env/retail.py', 'schema.py', 'taxonomy.py']
FORBIDDEN_NAMES = {'load_gold', 'gold_record', 'outcome', 'gold_state_hash', 'oracle_hypotheses', 'load_episodes'}


class LeakageTest(unittest.TestCase):
    def test_submission_side_does_not_touch_gold(self):
        for rel in SUBMISSION_SIDE:
            tree = ast.parse((PACKAGE / rel).read_text())
            imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
            modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
            self.assertFalse(imported & (FORBIDDEN_NAMES - ({'outcome', 'gold_state_hash'} if rel.startswith('env/') else set())),
                             rel)
            self.assertFalse({m for m in modules if m.endswith(('corpus', 'generate', 'recovery', 'attribution'))}, rel)
            strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
            self.assertFalse(any(s.startswith('gold/') or s == 'gold' for s in strings), rel)

    def test_agents_never_receive_task_gold(self):
        source = (PACKAGE / 'env' / 'workflow.py').read_text()
        self.assertNotIn("task['actions']", source)
        self.assertNotIn("task['outputs']", source.split('def outcome')[0])


class CliTest(unittest.TestCase):
    def test_smoke_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'smoke'
            result = subprocess.run([sys.executable, '-m', 'benchmark', 'smoke', '--out', str(out)],
                                    cwd=PACKAGE.parent, capture_output=True, text=True, timeout=300)
            self.assertEqual(result.returncode, 0, result.stderr)
            page = (out / 'dashboard.html').read_text()
            self.assertIn('<script id="data" type="application/json">', page)
            data = page.split('<script id="data" type="application/json">')[1].split('</script>')[0]
            parsed = json.loads(data)
            self.assertTrue(parsed['meta']['offline'])
            self.assertIn('OFFLINE FIXTURE', (out / 'report.md').read_text())
            sub = Path(tmp) / 'h.json'
            for args in (['baseline', '--corpus', str(out / 'corpus'), '--method', 'heuristic', '--out', str(sub)],
                         ['score', '--corpus', str(out / 'corpus'), '--submission', str(sub), '--out',
                          str(Path(tmp) / 's.json')],
                         ['recover', '--corpus', str(out / 'corpus'), '--policy', 'fixed_rules', '--predictions',
                          str(sub), '--backend', 'scripted', '--out', str(Path(tmp) / 'r.json')],
                         ['report', '--attribution', str(Path(tmp) / 's.json'), '--recovery', str(Path(tmp) / 'r.json')]):
                result = subprocess.run([sys.executable, '-m', 'benchmark', *args], cwd=PACKAGE.parent,
                                        capture_output=True, text=True, timeout=300)
                self.assertEqual(result.returncode, 0, f'{args[0]}: {result.stderr}')

    def test_check_ollama_reports_unreachable_server(self):
        result = subprocess.run([sys.executable, '-m', 'benchmark', 'check-ollama', '--host', 'http://127.0.0.1:9'],
                                cwd=PACKAGE.parent, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 2)
        self.assertIn('not reachable', result.stderr)


if __name__ == '__main__':
    unittest.main()
