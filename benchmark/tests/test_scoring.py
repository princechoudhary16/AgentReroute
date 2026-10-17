import unittest

from benchmark.corpus import load_gold, load_public
from benchmark.schema import SchemaError, validate_submission
from benchmark.scoring.attribution import score_attribution
from benchmark.scoring.baselines import heuristic_baseline, random_baseline
from benchmark.scoring.stats import bootstrap_ci, expected_calibration_error, macro_f1, mcnemar

from .helpers import corpus


def oracle_submission(gold, drop=()):
    rows = []
    for tid, g in gold.items():
        if g['status'] != 'failed' or tid in drop:
            continue
        c = g['critical']
        rows.append({'trajectory_id': tid, 'hypotheses': [{'step_index': c['step_index'], 'agent': c['agent'],
                                                           'mode': c['mode'], 'confidence': 0.9}]})
    return {'system': 'oracle', 'predictions': rows}


class AttributionScoringTest(unittest.TestCase):
    def setUp(self):
        self.gold = load_gold(corpus(), 'test')

    def test_oracle_is_perfect(self):
        m = score_attribution(oracle_submission(self.gold), self.gold)['metrics']
        for key in ('step_accuracy', 'agent_accuracy', 'mode_accuracy', 'joint_accuracy', 'acceptable_step_accuracy'):
            self.assertEqual(m[key]['value'], 1.0, key)
        self.assertEqual(m['mode_macro_f1'], 1.0)

    def test_missing_predictions_keep_the_denominator(self):
        failed = [t for t, g in self.gold.items() if g['status'] == 'failed']
        report = score_attribution(oracle_submission(self.gold, drop=failed[:2]), self.gold)
        self.assertEqual(report['denominator'], len(failed))
        self.assertEqual(report['states']['missing'], 2)
        self.assertAlmostEqual(report['metrics']['step_accuracy']['value'], (len(failed) - 2) / len(failed))

    def test_baselines_are_valid_and_ordered(self):
        traces = load_public(corpus(), 'test')
        rand = score_attribution(random_baseline(traces), self.gold)
        heur = score_attribution(heuristic_baseline(traces), self.gold)
        self.assertGreater(heur['metrics']['step_within_1']['value'], rand['metrics']['step_within_1']['value'])
        self.assertIn('detection', heur)
        d = heur['detection']
        self.assertEqual(d['n_failed'] + d['n_controls'], len(self.gold))

    def test_submission_validation(self):
        with self.assertRaises(SchemaError):
            validate_submission({'system': 'x', 'predictions': [{'trajectory_id': 'a'}, {'trajectory_id': 'a'}]})
        with self.assertRaises(SchemaError):
            validate_submission({'system': 'x', 'predictions': [
                {'trajectory_id': 'a', 'hypotheses': [{'step_index': 1, 'mode': 'drift', 'confidence': 2}]}]})
        rows = validate_submission({'system': 'x', 'predictions': [{'trajectory_id': 'a', 'hypotheses': [
            {'step_index': 1, 'mode': 'drift', 'confidence': 0.2}, {'step_index': 3, 'mode': 'Cycles', 'confidence': 0.7}]}]})
        self.assertEqual(rows['a']['hypotheses'][0]['mode'], 'cycles')


class StatsTest(unittest.TestCase):
    def test_mcnemar_exact(self):
        result = mcnemar([1] * 8 + [0] * 2, [0] * 8 + [0] * 2)
        self.assertEqual((result['only_a'], result['only_b']), (8, 0))
        self.assertAlmostEqual(result['p_value'], 2 / 256)
        self.assertEqual(mcnemar([1, 0], [1, 0])['p_value'], 1.0)

    def test_bootstrap_is_seeded_and_brackets_mean(self):
        values = [1, 0, 1, 1, 0, 1, 0, 1]
        self.assertEqual(bootstrap_ci(values, seed=3), bootstrap_ci(values, seed=3))
        low, high = bootstrap_ci(values)
        self.assertLessEqual(low, 5 / 8)
        self.assertGreaterEqual(high, 5 / 8)

    def test_ece_and_macro_f1(self):
        ece, _ = expected_calibration_error([0.9, 0.9, 0.1], [True, True, False])
        self.assertAlmostEqual(ece, 0.1)
        f1, per = macro_f1(['a', 'a', 'b'], ['a', 'b', 'b'], ['a', 'b', 'c'])
        self.assertNotIn('c', per)
        self.assertAlmostEqual(f1, (2 / 3 + 2 / 3) / 2)


if __name__ == '__main__':
    unittest.main()
