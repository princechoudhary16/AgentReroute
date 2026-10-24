"""The lenient counter must register a NON-ZERO count through the full chain (run -> episode -> manifest).

A zero check cannot tell a working counter from a dead one, so this forces a lenient response (as a
MAX_TOKENS reply would) and asserts it reaches both the episode JSON and the build-corpus manifest.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmark.corpus import build
from benchmark.env.scripted import scripted_llm as real_scripted_llm
from benchmark.generate import generate


class _LenientOnce:
    """Wraps the scripted actor and flags its first completion lenient, as a MAX_TOKENS reply would."""
    measurement_kind = 'offline_fixture'

    def __init__(self, inner):
        self.inner, self.fired = inner, False

    def complete(self, req):
        completion = self.inner.complete(req)
        if not self.fired:
            self.fired = True
            completion.lenient = True
        return completion


class LenientCounterTest(unittest.TestCase):
    def test_nonzero_lenient_count_reaches_episode_and_manifest(self):
        factory = lambda domain, task: _LenientOnce(real_scripted_llm(domain, task))  # noqa: E731
        with tempfile.TemporaryDirectory() as d, mock.patch('benchmark.generate.scripted_llm', factory):
            gen = Path(d) / 'gen'
            generate({'backend': 'scripted', 'model': 'scripted', 'seeds': [0], 'max_tasks': 1}, gen)
            episodes = [json.loads(p.read_text()) for p in (gen / 'episodes').glob('*.json')]
            self.assertTrue(any(e['lenient_responses'] >= 1 for e in episodes), 'no episode recorded a lenient response')
            manifest = build([gen], Path(d) / 'corpus')
        self.assertGreaterEqual(manifest['lenient_episodes'], 1)


if __name__ == '__main__':
    unittest.main()
