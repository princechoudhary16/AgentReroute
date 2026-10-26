"""An interrupted append must not make a run's own index unreadable.

Generation and training both resume by reading an append-only index they wrote themselves. A process
killed between the record and its newline leaves a partial final line, and the reader used to raise on
it, which would block the resume of a corpus that cannot be regenerated identically. Worse, appending
after such a fragment fused it with the next record, destroying a good record too.

Only the final line is forgiven. A malformed line earlier in the file is real corruption and must still
raise, or this hardening would quietly serve a damaged log as a complete one.
"""
import json
import tempfile
import unittest
from pathlib import Path

from benchmark.io import append_jsonl, read_jsonl


class JsonlResumeTest(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / 'index.jsonl'

    def test_a_partial_final_line_is_dropped_not_raised(self):
        self.path.write_text('{"a": 1}\n{"b": 2}\n{"c":')
        self.assertEqual(read_jsonl(self.path), [{'a': 1}, {'b': 2}])

    def test_appending_after_a_partial_line_keeps_both_records_readable(self):
        """The fragment is removed rather than fused with the next record."""
        self.path.write_text('{"a": 1}\n{"c":')
        append_jsonl(self.path, {'d': 4})
        self.assertEqual(read_jsonl(self.path), [{'a': 1}, {'d': 4}])
        self.assertTrue(self.path.read_text().endswith('\n'))

    def test_a_malformed_line_that_is_not_last_still_raises(self):
        self.path.write_text('{"a": 1}\n{BROKEN}\n{"c": 3}\n')
        with self.assertRaises(json.JSONDecodeError):
            read_jsonl(self.path)

    def test_a_file_whose_only_line_is_partial_reads_as_empty_and_recovers(self):
        self.path.write_text('{"only":')
        self.assertEqual(read_jsonl(self.path), [])
        append_jsonl(self.path, {'e': 5})
        self.assertEqual(read_jsonl(self.path), [{'e': 5}])

    def test_a_complete_log_is_untouched(self):
        for row in ({'a': 1}, {'b': 2}):
            append_jsonl(self.path, row)
        before = self.path.read_text()
        append_jsonl(self.path, {'c': 3})
        self.assertTrue(self.path.read_text().startswith(before), 'a healthy log was rewritten')
        self.assertEqual(read_jsonl(self.path), [{'a': 1}, {'b': 2}, {'c': 3}])

    def test_a_missing_file_is_empty(self):
        self.assertEqual(read_jsonl(self.path), [])


if __name__ == '__main__':
    unittest.main()
