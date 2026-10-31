"""Research claims must follow the pinned per-case evidence, including degradation."""
import csv
from pathlib import Path
import tempfile
import unittest
from demo import story_data as data


class StoryEvidenceTests(unittest.TestCase):
    def test_current_evidence_api_exists(self):
        self.assertTrue(callable(getattr(data, 'load_evidence', None)), 'Load current CSV evidence instead of archived summaries')

    def test_headline_counts_intervals_and_paired_tests(self):
        expected = {('tau', 'agentrx'): (9, 29, 31.0, 17.3, 49.2),
                    ('tau', 'seeds_select'): (10, 29, 34.5, 19.9, 52.7),
                    ('tau', 'synth_select_p10'): (11, 29, 37.9, 22.7, 56.0),
                    ('magentic', 'agentrx'): (8, 44, 18.2, 9.5, 32.0),
                    ('magentic', 'seeds_select'): (11, 44, 25.0, 14.6, 39.4)}
        for (domain, variant), values in expected.items():
            arm = data.summarize(data.FINAL_ROWS, domain, variant)
            self.assertEqual((arm['hits'], arm['n'], round(arm['accuracy'], 1), *[round(x, 1) for x in arm['ci']]), values)
        self.assertEqual(data.paired(data.FINAL_ROWS, 'agentrx', 'synth_select_p10', 'tau')['p'], .6875)
        self.assertEqual(data.paired(data.FINAL_ROWS, 'agentrx', 'seeds_select', 'magentic')['p'], .25)

    def test_exclusions_are_by_selector_errors_not_fallback_flag(self):
        tau = data.summarize(data.FINAL_ROWS, 'tau', 'seeds_select')
        self.assertEqual((tau['fallbacks'], tau['selector_errors'], tau['reportable']), (3, 0, True))
        bad = data.summarize(data.FINAL_ROWS, 'magentic', 'synth_select_p5')
        self.assertEqual((bad['n'], bad['selector_errors'], bad['reportable']), (44, 44, False))
        self.assertFalse(data.summarize(data.FINAL_ROWS, 'magentic', 'synth_select_p10')['reportable'])
        baseline = data.summarize(data.FINAL_ROWS, 'magentic', 'agentrx')
        self.assertEqual((baseline['statuses']['error'], baseline['statuses']['partial_checker_errors']), (2, 41))

    def test_clean_ablation_and_recall(self):
        result = data.paired(data.SELECTOR_ROWS, 'mp_rl_top1', 'mp_rl_select_trace')
        self.assertEqual((result['wins'], result['losses'], result['n']), (9, 1, 73))
        self.assertAlmostEqual(result['p'], .021484375)
        tau = data.summarize(data.FINAL_ROWS, 'tau', 'seeds_select')
        self.assertEqual((tau['in_pool_n'], tau['in_pool_hits'], tau['out_pool_hits']), (12, 8, 2))
        self.assertAlmostEqual(tau['recall'], 100 * 12 / 29)
        pool = data.summarize(data.SELECTOR_ROWS, 'tau', 'mp_rl_select_pool')
        self.assertEqual((pool['hits'], pool['fallbacks'], pool['selector_errors']), (7, 12, 0))

    def test_rejects_changed_source_duplicate_and_inconsistent_scoring(self):
        source = Path(data.RESULTS_DIR) / 'final-evidence.csv'
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'evidence.csv'
            dest.write_bytes(source.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'source digest'):
                data.load_evidence(dest, expected_sha256=data.SOURCE_DIGESTS['final'])
            with source.open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            for changed, message in [([rows[0], rows[0]], 'duplicate'),
                                     ([dict(rows[0], hit='0')], 'scoring')]:
                with dest.open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(changed)
                with self.assertRaisesRegex(ValueError, message):
                    data.load_evidence(dest)

    def test_rejects_incomplete_pairing_and_gold_mismatch(self):
        shortened = [r for r in data.FINAL_ROWS if not (r['variant']=='seeds_select' and r['domain']=='tau' and r['trajectory_id']==next(x['trajectory_id'] for x in data.FINAL_ROWS if x['domain']=='tau'))]
        with self.assertRaisesRegex(ValueError, 'case sets'):
            data.paired(shortened, 'agentrx', 'seeds_select', 'tau')
        changed = [dict(r) for r in data.FINAL_ROWS]
        next(r for r in changed if r['variant']=='seeds_select')['gold_step'] = '-999'
        with self.assertRaisesRegex(ValueError, 'gold'):
            data.paired(changed, 'agentrx', 'seeds_select')


if __name__ == '__main__':
    unittest.main()
