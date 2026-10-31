"""Sliced generation: N processes over disjoint task slices, merged by corpus.build().

This is the whole of the generation parallelism, chosen over threads because it needs no shared mutable
state and no locks: each process owns its own output directory, its own response cache and its own
index.jsonl, so each stays independently resumable.

The test that matters is the equivalence one: a corpus merged from two slices must be identical to one
built in a single pass, or slicing is quietly changing the benchmark.
"""
import io
import tempfile
import unittest
from pathlib import Path

from benchmark.corpus import SPLITS, build, load_gold, load_public
from benchmark.generate import generate, load_episodes, validate_config

BASE = {'backend': 'scripted', 'model': 'scripted', 'seeds': [0]}


def generate_into(root, name, **extra):
    out = Path(root) / name
    generate({**BASE, **extra}, out, log=io.StringIO())
    return out


class TaskSliceTest(unittest.TestCase):
    def test_slices_partition_the_task_list_without_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            dirs = [generate_into(d, f'gen{k}', task_slice=f'{k}/2') for k in range(2)]
            per_dir = [{e['task_id'] for e in load_episodes(x)} for x in dirs]
            self.assertTrue(per_dir[0] and per_dir[1])
            self.assertFalse(per_dir[0] & per_dir[1], 'a task was generated twice')

    def test_a_merged_corpus_equals_a_single_pass_corpus(self):
        with tempfile.TemporaryDirectory() as d:
            whole = generate_into(d, 'whole')
            sliced = [generate_into(d, f'part{k}', task_slice=f'{k}/2') for k in range(2)]
            one = build([whole], Path(d) / 'corpus_one')
            many = build(sliced, Path(d) / 'corpus_many')
            for split in SPLITS:
                self.assertEqual(load_public(Path(d) / 'corpus_one', split),
                                 load_public(Path(d) / 'corpus_many', split), split)
                self.assertEqual(load_gold(Path(d) / 'corpus_one', split),
                                 load_gold(Path(d) / 'corpus_many', split), split)
            self.assertEqual(one['gold_hash'], many['gold_hash'])
            self.assertEqual(one['splits'], many['splits'])
            self.assertEqual(len(many['sources']), 2)

    def test_refuses_to_merge_generations_made_under_different_settings(self):
        """The flag invites multi-directory builds, so a settings mismatch has to fail loudly."""
        with tempfile.TemporaryDirectory() as d:
            a = generate_into(d, 'a', task_slice='0/2')
            b = generate_into(d, 'b', task_slice='1/2', num_ctx=4096)
            with self.assertRaises(ValueError) as caught:
                build([a, b], Path(d) / 'corpus_mixed')
            self.assertIn('num_ctx', str(caught.exception))

    def test_slice_is_absent_from_the_settings_lock(self):
        """Slices must record identical identities, or they could never be merged."""
        with tempfile.TemporaryDirectory() as d:
            from benchmark.io import read_json
            dirs = [generate_into(d, f'g{k}', task_slice=f'{k}/2') for k in range(2)]
            identities = [read_json(x / 'generation.json')['identity'] for x in dirs]
            self.assertEqual(identities[0], identities[1])

    def test_validation(self):
        for bad in ('4/4', '1/0', 'x/4', '1', '0/-1'):
            with self.assertRaises(ValueError, msg=bad):
                validate_config({**BASE, 'task_slice': bad})
        self.assertEqual(validate_config({**BASE, 'task_slice': '2/3'})['task_slice'], '2/3')


if __name__ == '__main__':
    unittest.main()
