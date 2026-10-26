"""Training can stop as soon as validation clears a target, and reports progress while it runs.

A full training pass is hours long. Without a stopping rule you pay for every episode even after the
policy is good enough, and without per-episode output a corrupted reward signal is discovered at the
end rather than in the first minutes. Both matter here because the reward was measured disagreeing
with gold outcomes (acceptance 0.20 against harness recovery 1.00).

The stop writes the checkpoint and the consumed ledger exactly as a normal episode does, so a run
halted on target is still resumable rather than a dead end.
"""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from agentrx.rl_cross_branch import train_and_evaluate_benchmark

from .helpers import corpus

COMMON = dict(train_split='train', test_split='test', val_split='calibration', epochs=1, limit=3)


def train(checkpoint, **kwargs):
    err = io.StringIO()
    with redirect_stderr(err):
        result = train_and_evaluate_benchmark(corpus(), rl_state_path=str(checkpoint), **COMMON, **kwargs)
    return result, err.getvalue()


class TrainEarlyStopTest(unittest.TestCase):
    def test_a_reachable_target_stops_the_run_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            state = Path(d) / 'stop.json'
            result, _ = train(state, stop_above=-1.0, val_every=1)   # any score clears -1.0
            row = result['train_history'][0]
            self.assertIsNotNone(row['stopped_on_target'], 'run did not stop on target')
            self.assertEqual(row['stopped_on_target']['after_episodes'], 1)
            self.assertEqual(row['val_attribution_exact_step'], row['stopped_on_target']['score'])

    def test_stopping_leaves_a_resumable_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            state = Path(d) / 'stop.json'
            train(state, stop_above=-1.0, val_every=1)
            saved = json.loads(state.read_text())
            consumed = saved['progress']['epoch_consumed']['1']
            self.assertEqual(len(consumed), 1, 'the ledger should hold exactly the episodes that trained')
            self.assertEqual(len(consumed), len(set(consumed)))
            self.assertEqual(saved['progress']['history'][0]['epoch'], 1)

    def test_an_unreachable_target_runs_the_whole_split(self):
        with tempfile.TemporaryDirectory() as d:
            state = Path(d) / 'go.json'
            result, _ = train(state, stop_above=2.0, val_every=1)    # exact-step can never exceed 1.0
            self.assertIsNone(result['train_history'][0]['stopped_on_target'])
            consumed = json.loads(state.read_text())['progress']['epoch_consumed']['1']
            self.assertGreater(len(consumed), 1)

    def test_no_target_means_no_validation_passes_and_no_stop(self):
        with tempfile.TemporaryDirectory() as d:
            result, text = train(Path(d) / 'plain.json', val_every=5)
            self.assertIsNone(result['train_history'][0]['stopped_on_target'])
            self.assertNotIn('[val @', text, 'ran a mid-epoch validation pass with no target to compare to')

    def test_progress_reports_each_episode_and_can_be_silenced(self):
        with tempfile.TemporaryDirectory() as d:
            _, loud = train(Path(d) / 'loud.json')
            self.assertIn('[ep', loud)
            self.assertIn('s/ep', loud)
            self.assertIn('|dtheta|', loud)
            _, quiet = train(Path(d) / 'quiet.json', show_progress=False)
            self.assertNotIn('[ep', quiet)


if __name__ == '__main__':
    unittest.main()
