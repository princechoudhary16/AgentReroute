import unittest

from benchmark.corpus import load_gold
from benchmark.env.scripted import scripted_llm
from benchmark.scoring.recovery import (BudgetCaps, FixedRules, FullRestart, NoRecovery, RecoveryEnv,
                                        RecoveryPolicy, compare, run_recovery)
from benchmark.env.workflow import Directive

from .helpers import corpus


class Greedy(RecoveryPolicy):
    """Spends attempts until the budget runs out, to test enforcement."""
    name = 'greedy'

    def recover(self, env):
        last = None
        for i in range(10):
            last = env.replay(Directive(rollback_step=2, kind='resample', seed_offset=i + 1))
        return {'delivered': last.rollout_id, 'status': 'intervened'}


class Peeker(RecoveryPolicy):
    name = 'peeker'

    def recover(self, env):
        self.seen = [a for a in dir(env) if 'ledger' in a.lower() or 'reward' in a.lower() or 'gold' in a.lower()]
        return {'delivered': None, 'status': 'abstained'}


class RecoveryTest(unittest.TestCase):
    def test_baselines_and_oracle(self):
        none = run_recovery(corpus(), 'test', NoRecovery(), scripted_llm)
        restart = run_recovery(corpus(), 'test', FullRestart(), scripted_llm)
        oracle = run_recovery(corpus(), 'test', FixedRules(), scripted_llm, oracle=True)
        self.assertEqual(none['metrics']['recovery_rate']['value'], 0.0)
        self.assertEqual(none['metrics']['harmful_intervention_rate']['value'], 0.0)
        self.assertGreater(oracle['metrics']['recovery_rate']['value'], restart['metrics']['recovery_rate']['value'])
        self.assertLess(oracle['metrics']['tokens_per_episode'], restart['metrics']['tokens_per_episode'])
        self.assertEqual(oracle['metrics']['false_intervention_rate']['value'], 0.0)
        paired = compare(oracle, restart)
        self.assertEqual(paired['n_paired'], oracle['n_failed'])
        self.assertLess(paired['p_value'], 0.05)

    def test_budget_is_enforced(self):
        report = run_recovery(corpus(), 'test', Greedy(), scripted_llm, BudgetCaps(attempts=2), include_controls=False,
                              limit=3)
        for row in report['per_episode']:
            self.assertLessEqual(row['attempts'], 2)
            self.assertEqual(row['decision'], 'aborted')

    def test_policy_cannot_see_rewards_or_gold(self):
        peeker = Peeker()
        run_recovery(corpus(), 'test', peeker, scripted_llm, include_controls=False, limit=1)
        self.assertEqual(peeker.seen, [])
        self.assertFalse(hasattr(RecoveryEnv, 'ledger'))

    def test_rows_cover_failed_and_controls(self):
        gold = load_gold(corpus(), 'test')
        report = run_recovery(corpus(), 'test', NoRecovery(), scripted_llm)
        self.assertEqual(report['n_failed'] + report['n_controls'], len(gold))


if __name__ == '__main__':
    unittest.main()
