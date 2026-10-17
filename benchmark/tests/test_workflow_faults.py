import unittest

from benchmark.env.retail import gold_state_hash
from benchmark.env.workflow import Directive, export_trace, outcome
from benchmark.faults.operators import OPERATORS, build_operator, candidate_sites, pick_site, variants
from benchmark.schema import validate_trace

from .helpers import run_task


def _inject(task_id, name, timing='early', **params):
    domain, task, clean = run_task(task_id)
    site = pick_site(candidate_sites(name, params, clean), timing)
    spec = {'name': name, 'ordinal': site['ordinal'], 'params': params}
    if name == 'policy_contradiction':
        spec['site_step'] = site['step_index']
    op = build_operator(spec)
    _, _, run = run_task(task_id, faults=[op])
    return domain, task, clean, op, run, outcome(task, run, gold_state_hash(domain.data, task['actions']))


class WorkflowTest(unittest.TestCase):
    def test_every_fixture_task_passes_clean(self):
        from benchmark.env.tasks import load_fixture
        for task in load_fixture().tasks:
            domain, task, run = run_task(task['task_id'])
            self.assertEqual(outcome(task, run, gold_state_hash(domain.data, task['actions']))['reward'], 1.0,
                             task['task_id'])
            self.assertEqual(run.termination, 'user_stop')

    def test_trace_views(self):
        _, _, run = run_task('fixture-refund-amount')
        full = validate_trace(export_trace(run, 't1', 'retail', 'hi', 'full'))
        lean = validate_trace(export_trace(run, 't1', 'retail', 'hi', 'output_only'))
        self.assertTrue(any('input' in s for s in full['steps']))
        self.assertFalse(any('input' in s or 'state_version' in s for s in lean['steps']))
        handoffs = [e for s in lean['steps'] for e in s['substeps'] if e.get('name') == 'handoff']
        self.assertEqual(len(handoffs), 1)

    def test_unmodified_replay_reproduces_and_charges_only_new_steps(self):
        _, _, clean = run_task('fixture-cancel')
        _, _, replay = run_task('fixture-cancel', directive=Directive(rollback_step=6, kind='none'))
        self.assertEqual([s.content for s in clean.steps], [s.content for s in replay.steps])
        self.assertTrue(all(s.replayed for s in replay.steps if s.index < 6))
        self.assertLess(replay.new_tokens, replay.tokens)


class FaultOperatorTest(unittest.TestCase):
    def test_grid_covers_all_seven_modes(self):
        self.assertEqual({OPERATORS[name].mode for name, _ in variants()},
                         {'instruction_adherence', 'drift', 'cycles', 'tool_failure',
                          'misinterpretation_of_tool_output', 'context_propagation', 'missing_details'})

    def test_skip_write_fails_with_exact_label(self):
        _, _, clean, op, run, result = _inject('fixture-cancel', 'skip_write')
        self.assertEqual(result['reward'], 0.0)
        label = op.label(run, run.faults_fired[0])
        self.assertEqual(label['critical_step'], label['fault_step'])
        self.assertEqual(run.steps[label['critical_step'] - 1].actor, 'order_agent')
        self.assertEqual([s.content for s in clean.steps[:label['fault_step'] - 1]],
                         [s.content for s in run.steps[:label['fault_step'] - 1]])

    def test_transient_tool_error_is_tolerated_persistent_fails(self):
        *_, transient = _inject('fixture-cancel', 'tool_error', cell='explicit_transient')
        *_, persistent = _inject('fixture-cancel', 'tool_error', cell='explicit_persistent')
        self.assertEqual(transient['reward'], 1.0)
        self.assertEqual(persistent['reward'], 0.0)

    def test_tool_fault_label_points_to_consuming_agent(self):
        _, _, _, op, run, result = _inject('fixture-refund-amount', 'stale_output', cell='silent_transient')
        self.assertEqual(result['reward'], 0.0)
        label = op.label(run, run.faults_fired[0])
        self.assertEqual(run.steps[label['fault_step'] - 1].kind, 'tool')
        self.assertEqual(run.steps[label['critical_step'] - 1].actor, 'returns_agent')
        self.assertGreater(label['critical_step'], label['fault_step'])

    def test_cycle_label_is_a_repeated_call(self):
        _, _, _, op, run, result = _inject('fixture-cancel', 'stuck_processing')
        self.assertEqual(result['reward'], 0.0)
        label = op.label(run, run.faults_fired[0])
        call = run.steps[label['critical_step'] - 1].action
        self.assertEqual(call['tool'], 'get_order_details')

    def test_context_and_routing_faults_blame_planner(self):
        for name in ('drop_context', 'misroute'):
            _, _, _, op, run, result = _inject('fixture-exchange', name)
            self.assertEqual(result['reward'], 0.0, name)
            self.assertEqual(op.label(run, run.faults_fired[0])['responsible_agent'], 'planner')

    def test_omit_detail_removes_required_output(self):
        *_, result = _inject('fixture-tshirt-options', 'omit_detail')
        self.assertEqual(result['missing_outputs'], ['3'])

    def test_agent_fault_recurs_on_unchanged_context_and_clears_with_guidance(self):
        domain, task, _, op, run, _ = _inject('fixture-cancel', 'skip_write')
        site = run.faults_fired[0]['step_index']
        gold = gold_state_hash(domain.data, task['actions'])
        _, _, same = run_task('fixture-cancel', faults=[op.replay_copy()], directive=Directive(site, 'none'))
        self.assertEqual(outcome(task, same, gold)['reward'], 0.0)
        _, _, guided = run_task('fixture-cancel', faults=[op.replay_copy()],
                                directive=Directive(site, 'guidance', 'Call the write tool.'))
        self.assertEqual(outcome(task, guided, gold)['reward'], 1.0)

    def test_environment_fault_persists_through_guidance(self):
        domain, task, _, op, run, _ = _inject('fixture-cancel', 'tool_error', cell='explicit_persistent')
        site = run.faults_fired[0]['step_index']
        _, _, guided = run_task('fixture-cancel', faults=[op.replay_copy()],
                                directive=Directive(site + 1, 'guidance', 'Retry the tool.'))
        self.assertEqual(outcome(task, guided, gold_state_hash(domain.data, task['actions']))['reward'], 0.0)

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            OPERATORS['tool_error'](cell='explicit_sometimes')
        with self.assertRaises(ValueError):
            OPERATORS['skip_write'](colour='red')


if __name__ == '__main__':
    unittest.main()
