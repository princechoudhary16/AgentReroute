"""Leakage boundary and command-line checks."""
import ast
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from benchmark import cli
from benchmark.env.llm import OllamaClient

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

    def test_mp_rl_training_path_does_not_import_gold_readers(self):
        """The mp_rl policy/training path may call the harness (run_recovery) but must not import a gold
        reader itself. Gold-reading reporting lives in agentrx/rl_reporting.py, off this path."""
        repo = PACKAGE.parent
        for rel in ('agentrx/rl_cross_branch.py', 'agentrx/pipeline.py', 'agentrx/rl_policy.py'):
            tree = ast.parse((repo / rel).read_text(encoding='utf-8'))
            imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
            self.assertFalse(imported & FORBIDDEN_NAMES, f'{rel} imports a gold reader: {imported & FORBIDDEN_NAMES}')
            strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
            self.assertFalse(any(s.startswith('gold/') or s == 'gold' for s in strings), rel)


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

    def test_check_ollama_probes_a_live_completion(self):
        """The model listing alone passed against a server that cannot complete; the probe must exercise /api/chat."""
        class _Resp:
            def __init__(self, data): self._data = json.dumps(data).encode()
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return self._data

        def opener_for(chat_message):
            def opener(req, timeout=None):
                url = req.full_url if hasattr(req, 'full_url') else req
                if url.endswith('/api/tags'):
                    return _Resp({'models': [{'name': 'qwen2.5:7b'}]})
                if url.endswith('/api/show'):
                    return _Resp({'digest': 'sha256:abcd'})
                return _Resp({'message': {'content': chat_message}})
            return opener

        def run(chat_message):
            args = cli.parser().parse_args(['check-ollama', '--model', 'qwen2.5:7b'])
            orig = cli.OllamaClient
            cli.OllamaClient = lambda host: OllamaClient(host, opener=opener_for(chat_message))
            try:
                with redirect_stdout(io.StringIO()):
                    return cli.cmd_check_ollama(args)
            finally:
                cli.OllamaClient = orig

        self.assertEqual(run('ok'), 0)       # a real reply -> healthy
        self.assertEqual(run('   '), 2)      # blank reply -> fail the gate


if __name__ == '__main__':
    unittest.main()
