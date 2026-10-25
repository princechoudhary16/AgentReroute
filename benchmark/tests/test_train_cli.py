"""The `train` subcommand: nothing in this repo invoked train_and_evaluate_benchmark before it.

Runs against the scripted fixture corpus, so there is no model and no network. The checks that matter
are the ones that protect something: the checkpoint guard (a published checkpoint is what measured
numbers were taken against), and the per-epoch gap between observable acceptance and the harness's own
recovery rate, which is the reward-hacking alarm.
"""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from benchmark.cli import main

from .helpers import corpus


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class TrainCliTest(unittest.TestCase):
    def test_trains_writes_checkpoint_and_logs_the_acceptance_gap(self):
        with tempfile.TemporaryDirectory() as d:
            state = Path(d) / 'rl_state_v2.json'
            code, text, _ = run(['train', '--corpus', str(corpus()), '--out', str(Path(d) / 'run'),
                                 '--checkpoint', str(state), '--backend', 'scripted',
                                 '--epochs', '2', '--limit', '2'])
            self.assertEqual(code, 0, text)
            self.assertTrue(state.exists())
            summary = json.loads((Path(d) / 'run' / 'train_val_summary.json').read_text())
            self.assertEqual([h['epoch'] for h in summary['train_history']], [1, 2])
            for row in summary['train_history']:
                self.assertAlmostEqual(row['acceptance_minus_ledger_gap'],
                                       row['acceptance_rate'] - row['harness_recovery_rate'])
            self.assertEqual(summary['test_report']['split'], 'test')
            self.assertEqual(summary['splits'], {'train': 'train', 'val': 'calibration', 'test': 'test'})
            self.assertIn('gap=', text)

    def test_refuses_to_train_over_an_existing_checkpoint(self):
        """checkpoints/rl_state.json is the frozen baseline; overwriting it silently would be worse
        than failing, because the published numbers were measured against exactly those weights."""
        with tempfile.TemporaryDirectory() as d:
            state = Path(d) / 'rl_state.json'
            frozen = {'weights': [0.5] * 10, 'episodes': 7}        # a plausible trained checkpoint
            state.write_text(json.dumps(frozen))
            argv = ['train', '--corpus', str(corpus()), '--out', str(Path(d) / 'run'),
                    '--checkpoint', str(state), '--backend', 'scripted', '--epochs', '1', '--limit', '1']
            code, _, err = run(argv)
            self.assertEqual(code, 2)
            self.assertIn('--resume', err)
            self.assertEqual(json.loads(state.read_text()), frozen)               # untouched
            code, _, _ = run(argv + ['--resume'])
            self.assertEqual(code, 0)
            resumed = json.loads(state.read_text())
            self.assertEqual(resumed['episodes'], 8)        # continued from 7, not restarted at 0
            self.assertNotEqual(resumed['weights'], frozen['weights'])

    def test_synthesis_requires_a_model_backend(self):
        with tempfile.TemporaryDirectory() as d:
            code, _, err = run(['train', '--corpus', str(corpus()), '--out', str(Path(d) / 'run'),
                                '--checkpoint', str(Path(d) / 's.json'), '--backend', 'scripted',
                                '--use-llm-synthesis', '--epochs', '1', '--limit', '1'])
            self.assertEqual(code, 2)
            self.assertIn('ollama or gemini', err)


if __name__ == '__main__':
    unittest.main()
