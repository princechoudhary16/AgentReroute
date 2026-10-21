import contextlib
import io
import json
import unittest

from recovery_orchestrator import evaluation as ev
from recovery_orchestrator.cli import main
from recovery_orchestrator.orchestrator import Orchestrator
from recovery_orchestrator.policy import EVConfig, EVPolicy, UnmodifiedReplayPolicy
from recovery_orchestrator.rca import load_payload
from recovery_orchestrator.registry import mock_registry
from recovery_orchestrator.tau_retail import WRITE_TOOLS

from .helpers import CASES, SMOKE_ID, SMOKE_RUN, SYN_ID, SYN_RUN, TAU_RAW, make_env, make_verifier, syn_trace

SIM = EVConfig(respect_irreversible=False)


def ev_policy():
    return EVPolicy(mock_registry(), SIM, write_tools=WRITE_TOOLS)


class GoldTest(unittest.TestCase):
    def test_gold_and_controls(self):
        gold = ev.load_gold(CASES)[SYN_ID]
        self.assertEqual((gold.critical.step_index, gold.critical.category), (7, 'MisinterpretationOfToolOutput'))
        self.assertEqual(ev.oracle_payload(gold).top.step_index, 7)
        self.assertEqual(ev.misattributed_payload(gold).top.step_index, 9)
        self.assertEqual(ev.load_tau_rewards(TAU_RAW), {SYN_ID: 0.0})

    def test_offline_decision_eval(self):
        gold = ev.load_gold(CASES)
        rows = ev.offline_decision_eval([(syn_trace(), load_payload(SYN_RUN, SYN_ID, 'tau'))], gold, ev_policy())
        row = rows[0]
        self.assertEqual((row['rollback_step'], row['exact'], row['covers_critical'], row['category_match']),
                         (7, True, True, True))
        summary = ev.summarize_decisions(rows)['all']
        self.assertEqual((summary['intervene'], summary['top1_step_accuracy']), (1, 1.0))
        table = ev.calibration_table(rows, bins=2)
        self.assertEqual(table[1], {'bin': [0.5, 1.0], 'n': 1, 'mean_confidence': 0.5, 'accuracy': 1.0})
        self.assertEqual(ev.binned_calibrator(table)(0.6), 1.0)
        self.assertEqual(ev.binned_calibrator(table)(0.2), 0.2)


class RecoveryMetricsTest(unittest.TestCase):
    def run_conditions(self, control_fixes):
        trace, payload = syn_trace(), load_payload(SYN_RUN, SYN_ID, 'tau')
        gold = ev.load_gold(CASES)[SYN_ID]
        rewards = ev.load_tau_rewards(TAU_RAW)
        records = []
        env = make_env(fix_with=('reground_tool_output',) + control_fixes)
        for condition, policy, p in [('orchestrator', ev_policy(), payload),
                                     ('oracle', ev_policy(), ev.oracle_payload(gold)),
                                     ('misattribution', ev_policy(), ev.misattributed_payload(gold))]:
            verifier, _ = make_verifier()
            records += ev.run_condition(condition, [(trace, p)], Orchestrator(env, verifier, policy),
                                        env.ledger, rewards)
        verifier, _ = make_verifier()
        control = ev.run_condition('unmodified', [(trace, payload)],
                                   Orchestrator(env, verifier, UnmodifiedReplayPolicy(ev_policy())), env.ledger, rewards)
        return records, control

    def test_conditions(self):
        records, control = self.run_conditions(())
        m = ev.recovery_metrics(records + control, control)
        self.assertEqual(m['orchestrator']['recovery_rate'], 1.0)
        self.assertEqual(m['oracle']['recovery_rate'], 1.0)
        self.assertEqual(m['misattribution']['recovery_rate'], 0.0)   # policy_reminder at step 9 doesn't fix it
        self.assertEqual(m['unmodified']['recovery_rate'], 0.0)
        self.assertEqual(m['orchestrator']['unnecessary_intervention_rate'], 0.0)
        self.assertEqual(m['orchestrator']['false_recovery_rate'], 0.0)
        self.assertIsNotNone(m['orchestrator']['tokens_per_recovered'])

    def test_unnecessary_intervention_when_control_succeeds(self):
        records, control = self.run_conditions(('unmodified_replay',))
        m = ev.recovery_metrics(records, control)
        self.assertEqual(m['orchestrator']['unnecessary_intervention_rate'], 1.0)
        self.assertEqual(m['misattribution']['harm_rate'], 1.0)     # failed where plain replay would succeed

    def test_metrics_by_hand(self):
        base = {'condition': 'c', 'attempt_rewards': [], 'tokens': 10, 'seconds': 1.0, 'attempts': 1}
        records = [
            {**base, 'trajectory_id': 'a', 'original_reward': 0.0, 'final_reward': 1.0, 'verified': True, 'intervened': True},
            {**base, 'trajectory_id': 'b', 'original_reward': 0.0, 'final_reward': 0.0, 'verified': True, 'intervened': True},
            {**base, 'trajectory_id': 'c', 'original_reward': 1.0, 'final_reward': 0.0, 'verified': False, 'intervened': True},
            {**base, 'trajectory_id': 'd', 'original_reward': 0.0, 'final_reward': 0.0, 'verified': False, 'intervened': False},
        ]
        m = ev.recovery_metrics(records)['c']
        self.assertAlmostEqual(m['recovery_rate'], 1 / 3)
        self.assertAlmostEqual(m['verified_rate'], 2 / 3)
        self.assertEqual(m['false_recovery_rate'], 0.5)
        self.assertAlmostEqual(m['harm_rate'], 1 / 3)
        self.assertEqual(m['tokens_per_recovered'], 40)
        self.assertIsNone(m['unnecessary_intervention_rate'])


class CLITest(unittest.TestCase):
    def run_cli(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main([str(a) for a in argv]), 0)
        return json.loads(out.getvalue())

    def test_payload_and_decide(self):
        p = self.run_cli('payload', '--fa-run', SMOKE_RUN, '--trajectory-id', SMOKE_ID)
        self.assertEqual((p['confidence'], p['top_k'][0]['step_index']), (1.0, 3))
        self.assertEqual(len(self.run_cli('payload', '--fa-run', SYN_RUN)), 1)
        d = self.run_cli('decide', '--fa-run', SYN_RUN, '--trajectory-id', SYN_ID)
        self.assertEqual(d['action'], 'escalate')
        d = self.run_cli('decide', '--fa-run', SYN_RUN, '--trajectory-id', SYN_ID, '--allow-irreversible')
        self.assertEqual(d['option']['rollback_step'], 7)

    def test_eval_decisions(self):
        r = self.run_cli('eval-decisions', '--fa-run', SYN_RUN, '--cases', CASES, '--allow-irreversible')
        self.assertEqual(r['summary']['all']['rollback_exact'], 1.0)
        r = self.run_cli('eval-decisions', '--oracle', '--cases', CASES, '--allow-irreversible')
        self.assertEqual(r['summary']['all']['top1_step_accuracy'], 1.0)


if __name__ == '__main__':
    unittest.main()
