"""Resuming training must not apply an episode's gradient twice.

The session this corpus is trained in is time-boxed, so training will be interrupted and restarted.
theta is saved after every episode, but before the consumed-episode ledger existed nothing recorded
WHICH episodes produced it, so a restart replayed them and folded their gradients in a second time.
That is silent: the weights are plausible, just wrong, and no output says so.

The property under test is the one that matters operationally: an interrupted-then-resumed run must
land on exactly the weights an uninterrupted run would have reached. The last test deletes the ledger
before resuming to show the comparison is sensitive, so these cannot pass vacuously.
"""
import json
import tempfile
import unittest
from pathlib import Path

from agentrx import rl_cross_branch
from agentrx.rl_cross_branch import MPRLRecoveryPolicy, train_and_evaluate_benchmark

from .helpers import corpus

LIMIT = 2


def train(checkpoint, stop_after=None):
    """Run training against the scripted fixture, optionally aborting after `stop_after` episodes."""
    kwargs = dict(train_split='train', test_split='test', val_split=None, epochs=1,
                  rl_state_path=str(checkpoint), limit=LIMIT)
    if stop_after is None:
        return train_and_evaluate_benchmark(corpus(), **kwargs)
    original, seen = MPRLRecoveryPolicy.recover, {'n': 0}

    def aborting(self, env):                 # abort AFTER the per-episode save, as a real kill would
        result = original(self, env)
        seen['n'] += 1
        if seen['n'] >= stop_after:
            raise KeyboardInterrupt('simulated session end')
        return result

    MPRLRecoveryPolicy.recover = aborting
    try:
        train_and_evaluate_benchmark(corpus(), **kwargs)
        raise AssertionError('expected the simulated interruption to fire')
    except KeyboardInterrupt:
        pass
    finally:
        MPRLRecoveryPolicy.recover = original


class TrainResumeTest(unittest.TestCase):
    def test_an_interrupted_run_resumes_to_the_same_weights(self):
        with tempfile.TemporaryDirectory() as d:
            uninterrupted = train(Path(d) / 'a.json')['learned_weights']

            broken = Path(d) / 'b.json'
            train(broken, stop_after=1)
            mid = json.loads(broken.read_text())
            self.assertEqual(len(mid['progress']['epoch_consumed']['1']), 1,
                             'the interrupted run should have banked exactly one episode')
            resumed = train_and_evaluate_benchmark(corpus(), train_split='train', test_split='test',
                                                   val_split=None, epochs=1, rl_state_path=str(broken),
                                                   limit=LIMIT)['learned_weights']

            for i, (want, got) in enumerate(zip(uninterrupted, resumed)):
                self.assertAlmostEqual(want, got, places=12, msg=f'weight {i} diverged after resume')

    def test_each_episode_is_counted_once_across_the_interruption(self):
        with tempfile.TemporaryDirectory() as d:
            straight = Path(d) / 'a.json'
            train(straight)
            expected = json.loads(straight.read_text())['episodes']

            broken = Path(d) / 'b.json'
            train(broken, stop_after=1)
            train_and_evaluate_benchmark(corpus(), train_split='train', test_split='test', val_split=None,
                                         epochs=1, rl_state_path=str(broken), limit=LIMIT)
            state = json.loads(broken.read_text())
            self.assertEqual(state['episodes'], expected)
            consumed = state['progress']['epoch_consumed']['1']
            self.assertEqual(len(consumed), len(set(consumed)), 'an episode was banked twice')

    def test_a_completed_epoch_is_not_retrained(self):
        with tempfile.TemporaryDirectory() as d:
            state = Path(d) / 'a.json'
            first = train(state)
            weights = list(first['learned_weights'])
            episodes = json.loads(state.read_text())['episodes']
            again = train_and_evaluate_benchmark(corpus(), train_split='train', test_split='test',
                                                 val_split=None, epochs=1, rl_state_path=str(state),
                                                 limit=LIMIT)
            self.assertEqual(again['learned_weights'], weights, 'epoch 1 was trained a second time')
            self.assertEqual(json.loads(state.read_text())['episodes'], episodes)

    def test_resuming_onto_a_different_corpus_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            state = Path(d) / 'a.json'
            train(state, stop_after=1)
            with self.assertRaises(ValueError) as caught:
                train_and_evaluate_benchmark(corpus(), train_split='calibration', test_split='test',
                                             val_split=None, epochs=1, rl_state_path=str(state), limit=LIMIT)
            self.assertIn('different settings', str(caught.exception))

    def test_without_the_ledger_the_resume_double_counts(self):
        """Sensitivity check: if the ledger is discarded, the resumed weights must NOT match.

        Without this, the equality assertions above would still pass if skipping silently stopped
        working and both runs simply trained everything twice.
        """
        with tempfile.TemporaryDirectory() as d:
            uninterrupted = train(Path(d) / 'a.json')['learned_weights']

            broken = Path(d) / 'b.json'
            train(broken, stop_after=1)
            state = json.loads(broken.read_text())
            state['progress']['epoch_consumed'] = {}            # pretend the ledger never existed
            broken.write_text(json.dumps(state))
            resumed = train_and_evaluate_benchmark(corpus(), train_split='train', test_split='test',
                                                   val_split=None, epochs=1, rl_state_path=str(broken),
                                                   limit=LIMIT)['learned_weights']
            self.assertNotEqual([round(w, 12) for w in uninterrupted], [round(w, 12) for w in resumed])


if __name__ == '__main__':
    unittest.main()
