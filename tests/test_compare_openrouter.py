"""Offline checks of the OpenRouter comparison script's budget layers and episode selection (no network)."""
import importlib.util
import json
import unittest
from pathlib import Path
from unittest import mock

from agentrx.providers import ProviderError

SCRIPT = Path(__file__).resolve().parents[1]/'scripts'/'compare_openrouter_agentrx_vs_mprl.py'
spec = importlib.util.spec_from_file_location('compare_openrouter', SCRIPT)
cmp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmp)


class FakeHTTP:
    identity, measurement_kind, last_raw, last_telemetry = {'provider': 'fake'}, 'model_measurement', None, {}

    def __init__(self, fail=None):
        self.http_requests, self.retries, self.fail = 0, 3, fail

    def complete(self, stage, payload):
        self.http_requests += 1
        if self.fail:
            raise ProviderError(self.fail)
        return {'stage': stage}


class BudgetLayerTests(unittest.TestCase):
    def test_live_gate_ceiling_clamps_retries_and_flags_429(self):
        http = FakeHTTP()
        gate = cmp.LiveGate(http, max_calls=2, min_interval=0)
        gate.complete('judge', {})
        self.assertEqual(http.retries, 1)               # 2 left before the call: one try + one retry at most
        gate.complete('judge', {})
        self.assertEqual(http.retries, 0)
        with self.assertRaisesRegex(ProviderError, 'budget exhausted'):
            gate.complete('judge', {})
        self.assertTrue(gate.exhausted)
        limited = cmp.LiveGate(FakeHTTP(fail='Model HTTP 429; response body omitted'), 5, 0)
        with self.assertRaises(ProviderError):
            limited.complete('judge', {})
        self.assertTrue(limited.rate_limited)

    def test_ledger_caps_semantic_checks_per_pass(self):
        ledger = cmp.Ledger(FakeHTTP(), semantic_cap=2)
        for _ in range(2):
            ledger.complete('semantic', {})
        with self.assertRaisesRegex(ProviderError, 'per-trace call cap'):
            ledger.complete('semantic', {})
        ledger.complete('judge', {})                    # other stages are never capped
        ledger.begin()
        ledger.complete('semantic', {})
        self.assertEqual((ledger.calls, ledger.refused), (5, 1))

    def test_min_interval_spaces_live_requests(self):
        gate = cmp.LiveGate(FakeHTTP(), max_calls=5, min_interval=3.2)
        with mock.patch.object(cmp.time, 'sleep') as sleep:
            gate.complete('judge', {})
            gate.complete('judge', {})
        self.assertGreater(sleep.call_args.args[0], 3.0)


@unittest.skipUnless((cmp.CORPUS/'gold'/'test.json').exists(), 'benchmark smoke corpus not generated')
class SelectionTests(unittest.TestCase):
    def test_retail_selection_is_stratified_over_every_mode(self):
        items = cmp.retail_episodes(6)
        self.assertEqual([i['gold']['critical']['mode'] for i in items], list(cmp.MODES))
        self.assertTrue(all(i['gold']['status'] == 'failed' for i in items))
        for i in items:     # attribution sees the compact view; replays keep the full one, step for step
            self.assertNotIn('input', json.dumps(i['trace_attr']['steps'][1]))
            self.assertIn('input', i['trace']['steps'][1])
            self.assertEqual([s['index'] for s in i['trace_attr']['steps']], [s['index'] for s in i['trace']['steps']])
            self.assertEqual(cmp.compact(i['trace'])['steps'][1].keys(), i['trace_attr']['steps'][1].keys())


if __name__ == '__main__':
    unittest.main()
