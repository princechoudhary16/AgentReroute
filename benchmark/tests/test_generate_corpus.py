import io
import json
import unittest
from pathlib import Path

from benchmark.corpus import DEFAULT_HELDOUT, assign_split, load_gold, load_public
from benchmark.generate import generate, load_episodes
from benchmark.io import digest, read_json, read_jsonl

from .helpers import ROOT, corpus, generated

GOLD_KEYS = ('critical', 'fault_step', 'injection', 'label', 'operator', 'acceptable_steps', 'faults', 'outcome')


class GenerateTest(unittest.TestCase):
    def test_statuses_and_labels(self):
        episodes = load_episodes(generated())
        statuses = {e['status'] for e in episodes}
        self.assertTrue({'clean', 'failed', 'tolerated'} <= statuses)
        for e in episodes:
            if e['status'] == 'failed':
                label = e['label']
                self.assertIn(label['critical_step'], label['acceptable_steps'])
                self.assertIn(label['fault_step'], label['acceptable_steps'])
                self.assertGreaterEqual(label['critical_step'], label['fault_step'])
                self.assertEqual(e['outcome']['reward'], 0.0)
            if e['status'] == 'tolerated':
                self.assertEqual(e['outcome']['reward'], 1.0)
                self.assertTrue(e['activation']['fired'])
        failed_modes = {e['label']['mode'] for e in episodes if e['status'] == 'failed'}
        self.assertEqual(len(failed_modes), 7)

    def test_resume_is_idempotent_and_settings_are_locked(self):
        before = read_jsonl(generated() / 'index.jsonl')
        generate({'backend': 'scripted', 'model': 'scripted', 'seeds': [0]}, generated(), log=io.StringIO())
        self.assertEqual(len(before), len(read_jsonl(generated() / 'index.jsonl')))
        with self.assertRaises(ValueError):
            generate({'backend': 'scripted', 'model': 'scripted', 'seeds': [0], 'temperature': 0.7}, generated(),
                     log=io.StringIO())

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            generate({'backend': 'scripted', 'operators': ['nope']}, ROOT / 'bad', log=io.StringIO())


class CorpusTest(unittest.TestCase):
    def test_public_traces_carry_no_gold(self):
        for path in (corpus() / 'public').rglob('tr-*.json'):
            text = path.read_text()
            trace = json.loads(text)
            for key in GOLD_KEYS:
                self.assertNotIn(key, trace)
                for step in trace['steps']:
                    self.assertNotIn(key, step)
            self.assertNotIn('GOLD', text)
            for word in ('skip_write', 'stale_output', 'misroute', 'drop_context'):
                self.assertNotIn(word, text)

    def test_splits_are_disjoint_by_task_and_heldout_only_in_test(self):
        templates = {}
        for split in ('train', 'calibration', 'test'):
            gold = load_gold(corpus(), split)
            self.assertEqual(set(gold), set(load_public(corpus(), split)))
            templates[split] = {g['task_template'] for g in gold.values()}
            for g in gold.values():
                self.assertEqual(assign_split(g['task_template']), split)
                if split != 'test' and g.get('injection'):
                    self.assertNotIn(g['injection']['variant'], DEFAULT_HELDOUT)
        self.assertFalse(templates['train'] & templates['test'])
        self.assertFalse(templates['calibration'] & templates['test'])

    def test_agentrx_export_matches_faultattribution_contract(self):
        for split in ('train', 'calibration', 'test'):
            cases = read_json(corpus() / 'agentrx' / split / 'cases.json')
            manifest = read_json(corpus() / 'agentrx' / split / 'manifest.json')
            self.assertEqual(manifest['corpus_hash'], digest(cases))
            for case in cases:
                self.assertEqual(case['trajectory']['domain'], 'tau')
                indices = {s['index'] for s in case['trajectory']['steps']}
                self.assertIn(case['gold']['critical']['step_index'], indices)
                self.assertEqual(case['gold']['failures'][0], case['gold']['critical'])

    def test_manifest_counts(self):
        manifest = read_json(corpus() / 'manifest.json')
        total = sum(v['total'] for v in manifest['splits'].values())
        self.assertEqual(total, sum(len(load_gold(corpus(), s)) for s in ('train', 'calibration', 'test')))
        self.assertGreater(manifest['dropped']['not_activated'], 0)


if __name__ == '__main__':
    unittest.main()
