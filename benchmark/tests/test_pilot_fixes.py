"""Regression tests for the problems found in the 2026-10-20 pilot (docs/pilot-2026-10-20.md)."""
import unittest

from benchmark.corpus import load_gold
from benchmark.env.retail import gold_state_hash
from benchmark.env.scripted import scripted_llm
from benchmark.env.workflow import AGENTS, WORKFLOW_VERSION, Directive, Workflow, outcome, system_prompt
from benchmark.faults.operators import (OPERATORS, OPERATORS_VERSION, build_operator, candidate_sites,
                                        completion_claim, pick_site)
from benchmark.generate import load_episodes
from benchmark.io import read_json
from benchmark.scoring.recovery import BudgetCaps, FixedRules, FullRestart, run_recovery
from benchmark.scoring.robustness import robustness

from .helpers import corpus, generated, run_task


def _inject(task_id, name, **params):
    domain, task, clean = run_task(task_id)
    site = pick_site(candidate_sites(name, params, clean), 'early')
    spec = {'name': name, 'ordinal': site['ordinal'], 'params': params}
    if 'target_agent' in site:
        spec['target_agent'] = site['target_agent']
    op = build_operator(spec)
    _, _, run = run_task(task_id, faults=[op])
    gold = gold_state_hash(domain.data, task['actions'])
    return task, op, run, gold


def _replay(task_id, op, directive):
    _, task, run = run_task(task_id, faults=[op.replay_copy()], directive=directive)
    return run


class ReplaySemanticsTest(unittest.TestCase):
    """Problem 1: agent faults must not disappear just because the context changed somewhere."""

    def setUp(self):
        self.task, self.op, self.run, self.gold = _inject('fixture-cancel', 'skip_write')
        self.site = self.run.faults_fired[0]['step_index']
        self.assertEqual(outcome(self.task, self.run, self.gold)['reward'], 0.0)

    def reward(self, directive):
        return outcome(self.task, _replay('fixture-cancel', self.op, directive), self.gold)['reward']

    def test_full_restart_does_not_remove_an_agent_fault(self):
        self.assertEqual(self.reward(Directive(1, 'resample', seed_offset=1)), 0.0)

    def test_guidance_to_another_agent_does_not_remove_it(self):
        planner_turn = next(s.index for s in self.run.steps if s.actor == 'planner')
        self.assertEqual(self.reward(Directive(planner_turn, 'guidance', 'Be careful.')), 0.0)

    def test_guidance_after_the_faulty_turn_does_not_remove_it(self):
        self.assertEqual(self.reward(Directive(self.site + 1, 'guidance', 'Call the write tool.')), 0.0)

    def test_guidance_to_the_responsible_agent_at_the_site_removes_it(self):
        self.assertEqual(self.reward(Directive(self.site, 'guidance', 'Call the write tool.')), 1.0)

    def test_guidance_before_the_site_in_the_same_conversation_removes_it(self):
        earlier = max(s.index for s in self.run.steps if s.actor == 'order_agent' and s.index < self.site)
        self.assertEqual(self.reward(Directive(earlier, 'guidance', 'Remember to call the write tool.')), 1.0)

    def test_planner_fault_is_removed_by_planner_guidance_before_it(self):
        task, op, run, gold = _inject('fixture-cancel', 'misroute', persistence='persistent')
        self.assertEqual(outcome(task, run, gold)['reward'], 0.0)
        first = next(s.index for s in run.steps if s.actor == 'planner')
        guided = _replay('fixture-cancel', op, Directive(first, 'guidance', 'Route to the right specialist.'))
        self.assertEqual(outcome(task, guided, gold)['reward'], 1.0)
        restarted = _replay('fixture-cancel', op, Directive(1, 'resample'))
        self.assertEqual(outcome(task, restarted, gold)['reward'], 0.0)

    def test_specialist_guidance_ends_with_its_conversation(self):
        domain, task, _ = run_task('fixture-cancel')
        wf = Workflow(domain, task, scripted_llm(domain, task), 'scripted',
                      directive=Directive(self.site, 'guidance', 'note'))
        wf.run()
        self.assertEqual(wf.guided, {})

    def test_prompt_mutation_targets_the_agent_turn(self):
        domain, task, clean = run_task('fixture-cancel')
        site = pick_site(candidate_sites('policy_contradiction', {'directive': 'no_writes'}, clean), 'early')
        self.assertEqual(site['target_agent'], 'order_agent')
        writer_turns = [s.index for s in clean.steps if s.actor == 'order_agent' and s.kind == 'agent']
        self.assertEqual(writer_turns[site['ordinal'] - 1], site['step_index'])


class PersistentFaultTest(unittest.TestCase):
    """Problem 2: drift / context faults need persistent variants a replanning agent cannot route around."""

    def test_variants_exist(self):
        for name in ('misroute', 'irrelevant_tool', 'drop_context'):
            self.assertEqual(OPERATORS[name].PARAMS['persistence'], ('once', 'persistent'))

    def test_persistent_fires_on_every_later_match_and_once_fires_once(self):
        handoff = {'action': 'handoff', 'to': 'order_agent', 'message': 'cancel #W1000001',
                   'context': {'order_id': '#W1000001'}}
        for persistence, expected in (('once', [False, True, False, False]), ('persistent', [False, True, True, True])):
            op = OPERATORS['misroute'](ordinal=2, persistence=persistence)
            fired = []
            for i in range(4):
                out, record = op.on_action({'agent': 'planner', 'step_index': i + 1}, dict(handoff))
                fired.append(record is not None)
            self.assertEqual(fired, expected, persistence)

    def test_label_points_at_the_first_firing(self):
        task, op, run, _ = _inject('fixture-cancel', 'drop_context', variant='drop_ids', persistence='persistent')
        label = op.label(run, run.faults_fired[0])
        self.assertEqual(label['critical_step'], run.faults_fired[0]['step_index'])
        self.assertEqual(label['responsible_agent'], 'planner')


class RewriteTextTest(unittest.TestCase):
    """Problem 3: injected success claims must not be one canned sentence."""

    def test_claim_names_the_specifics_and_is_deterministic(self):
        action = {'action': 'tool', 'tool': 'cancel_pending_order',
                  'arguments': {'order_id': '#W1000001', 'reason': 'no longer needed'}}
        claim = completion_claim(action)
        self.assertIn('#W1000001', claim)
        self.assertEqual(claim, completion_claim(dict(action)))
        claims = {completion_claim({'action': 'tool', 'tool': t, 'arguments': {'order_id': '#W1', 'item_ids': ['1']}})
                  for t in ('return_delivered_order_items', 'exchange_delivered_order_items',
                            'modify_pending_order_items', 'cancel_pending_order')}
        self.assertEqual(len(claims), 4)
        self.assertNotIn('Done. The request has been completed as asked.', claims)

    def test_skip_write_episode_uses_a_specific_claim(self):
        task, op, run, _ = _inject('fixture-return', 'skip_write')
        injected = run.faults_fired[0]['injected']['content']
        self.assertIn('#W1000002', injected)


class BudgetAndPolicyTest(unittest.TestCase):
    """Problem 4: the token cap scales with the episode; the judge-driven policy can abstain."""

    def test_relative_cap(self):
        self.assertEqual(BudgetCaps().for_episode(30000).tokens, 60000)
        self.assertEqual(BudgetCaps(token_multiple=3).for_episode(10).tokens, 3000)
        self.assertEqual(BudgetCaps(tokens=500).for_episode(30000).tokens, 500)

    def test_rows_record_cap_and_cost_ratio(self):
        report = run_recovery(corpus(), 'test', FullRestart(), scripted_llm, include_controls=False, limit=4)
        for row in report['per_episode']:
            self.assertEqual(row['token_cap'], int(2 * max(1000, row['original_tokens'])))
            self.assertAlmostEqual(row['cost_vs_original'], row['tokens'] / row['original_tokens'], places=3)
        self.assertIsNotNone(report['metrics']['cost_vs_original'])

    def test_min_confidence_abstains(self):
        gold = load_gold(corpus(), 'test')
        weak = {t: [{'step_index': 2, 'agent': 'planner', 'mode': 'drift', 'confidence': 0.1}] for t in gold}
        report = run_recovery(corpus(), 'test', FixedRules(min_confidence=0.5), scripted_llm, hypotheses=weak,
                              limit=5)
        self.assertTrue(all(r['decision'] == 'abstained' and r['attempts'] == 0 for r in report['per_episode']))


class RobustnessAndLockTest(unittest.TestCase):
    def test_robustness_matches_episode_counts(self):
        episodes = load_episodes(generated())
        report = robustness(episodes)
        injected = [e for e in episodes if e['operator'] and e['status'] in ('failed', 'tolerated')]
        self.assertEqual(report['injected'], len(injected))
        for name, row in report['by_variant'].items():
            self.assertEqual(row['n'] - row['failed'], round(row['robustness_score']['value'] * row['n']), name)
            self.assertTrue(0.0 <= row['recovery_cost'] <= 1.0)
        cells = report['toolmaze']['cells']
        self.assertEqual(set(cells), {'P1 explicit-transient', 'P2 explicit-permanent', 'P3 implicit-transient',
                                      'P4 implicit-permanent'})
        self.assertEqual(cells['P2 explicit-permanent']['recovery_cost'], 1.0)
        self.assertIn('transient', report['toolmaze']['explicit_minus_implicit'])

    def test_generation_lock_covers_workflow_prompts_and_operators(self):
        identity = read_json(generated() / 'generation.json')['identity']
        self.assertEqual(identity['workflow_version'], WORKFLOW_VERSION)
        self.assertEqual(identity['operators_version'], OPERATORS_VERSION)
        self.assertIn('prompts_digest', identity)

    def test_planner_can_read_orders_and_specialists_are_told_confirmation_is_done(self):
        self.assertIn('get_order_details', AGENTS['planner']['tools'])
        self.assertTrue(all(w not in AGENTS['planner']['tools'] for w in ('cancel_pending_order',
                                                                          'return_delivered_order_items')))
        for agent in ('order_agent', 'returns_agent', 'product_agent'):
            self.assertIn('already satisfied', system_prompt(agent, 'policy'))
        self.assertNotIn('already satisfied', system_prompt('planner', 'policy'))


if __name__ == '__main__':
    unittest.main()
