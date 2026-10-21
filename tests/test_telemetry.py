import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agentrx import metrics
from agentrx.testrunner import ProgressResult


class MetricsTests(unittest.TestCase):
    def test_emit_appends_jsonl_and_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'm' / 'events.jsonl'
            with mock.patch.dict(os.environ, {'AGENTRX_METRICS_FILE': str(path)}):
                metrics.emit('smoke', result_rows=1)
                metrics.emit('audit', validated_pairs=73)
                events = metrics.read_events()
                self.assertEqual([e['event'] for e in events], ['smoke', 'audit'])
                self.assertEqual(events[1]['validated_pairs'], 73)
                self.assertTrue(all('run_id' in e and 'ts' in e for e in events))
            with mock.patch.dict(os.environ, {'AGENTRX_METRICS_FILE': str(path), 'AGENTRX_METRICS': '0'}):
                self.assertIsNone(metrics.emit('ignored'))
                self.assertEqual(len(metrics.read_events()), 2)

    def test_emit_never_raises_on_unwritable_path(self):
        with tempfile.TemporaryDirectory() as d:
            blocker = Path(d) / 'file'
            blocker.write_text('x')
            with mock.patch.dict(os.environ, {'AGENTRX_METRICS_FILE': str(blocker / 'events.jsonl')}):
                self.assertIsNone(metrics.emit('x'))


class RunnerTests(unittest.TestCase):
    def test_runner_reports_failures_by_name(self):
        class Sample(unittest.TestCase):
            def test_ok(self): pass
            def test_bad(self): self.fail('boom')
        stream = io.StringIO()
        result = ProgressResult(stream, 2, color=False)
        unittest.defaultTestLoader.loadTestsFromTestCase(Sample).run(result)
        self.assertEqual([o[2] for o in sorted(result.outcomes)], ['fail', 'pass'])
        self.assertEqual(len(result.failures), 1)


if __name__ == '__main__':
    unittest.main()
