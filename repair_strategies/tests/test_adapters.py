import importlib.util
import itertools
import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from repair_strategies.adapters.benchmark import RepairStrategyPolicy, task_from_view
from repair_strategies.adapters.orchestrator import OrchestratorRepairer, RepairDirective, export_registry
from repair_strategies.engine import RepairEngine
from repair_strategies.memory.lessons import LessonMemory
from repair_strategies.taxonomy import CATEGORIES, INCONCLUSIVE

from .helpers import RunBuilder, authenticated, retail

ORCHESTRATOR_KINDS = ('reasoning_repair', 'retry', 'clarify', 'loop_break')
ORCHESTRATOR_KEYS = {'id', 'name', 'kind', 'targets', 'cost', 'prior_success', 'reversible', 'description', 'template'}


def loop_builder():
    b = authenticated(RunBuilder('ep-1'))
    for _ in range(3):
        b.tool('order_agent', 'get_order_details', 'Error: still processing, retry', order_id='#W1000005')
    return b


def recovered_builder():
    b = authenticated(RunBuilder('ep-1'))
    b.tool('order_agent', 'get_order_details', '{"status": "pending"}', order_id='#W1000005')
    b.reply('planner', 'Your order is pending.')
    b.user('###STOP###')
    return b


class RegistryTest(unittest.TestCase):
    def test_export_matches_the_orchestrator_schema(self):
        registry = export_registry()
        self.assertEqual(registry['schema_version'], 1)
        ids = [s['id'] for s in registry['strategies']]
        self.assertEqual(len(ids), len(set(ids)))
        covered = set()
        for entry in registry['strategies']:
            self.assertLessEqual(set(entry), ORCHESTRATOR_KEYS)
            self.assertIn(entry['kind'], ORCHESTRATOR_KINDS)
            self.assertTrue(entry['targets'])
            self.assertNotIn(INCONCLUSIVE, entry['targets'])
            self.assertTrue(0 <= entry['prior_success'] <= 1)
            self.assertIn('{rationale}', entry['template'])
            self.assertNotEqual(entry['id'], 'rollback_compensate_escalate')
            covered |= set(entry['targets'])
        self.assertEqual(covered, set(CATEGORIES) - {INCONCLUSIVE})


class OrchestratorRepairerTest(unittest.TestCase):
    def setUp(self):
        self.engine = RepairEngine(retail())
        self.repairer = OrchestratorRepairer(self.engine)
        self.candidate = SimpleNamespace(step_index=8, agent='order_agent', category='Cycle', share=0.8,
                                         rationale='looping on get_order_details', evidence_refs=(), source='fa')
        self.trace = loop_builder().trace()

    def test_build_uses_the_named_strategy_then_climbs_the_ladder(self):
        first = self.repairer.build(SimpleNamespace(id='loop_nudge'), self.candidate, self.trace)
        self.assertEqual((first.kind, first.strategy_id), ('inject_guidance', 'loop_nudge'))
        self.assertIn('Repeating it will not change the outcome', first.message)
        self.assertEqual(first.as_step(8).role, 'system')
        second = self.repairer.build(SimpleNamespace(id='loop_nudge'), self.candidate, self.trace)
        self.assertEqual(second.strategy_id, 'loop_force_change')            # nudge failed verification
        self.assertEqual(self.repairer.last_plan('ep-1').strategy_id, 'loop_force_change')
        self.assertIn('ban_calls: ban stated as a rule', self.repairer.last_lowered('ep-1').degraded)

    def test_unknown_strategy_falls_back_to_the_ladder_or_template(self):
        directive = self.repairer.build(SimpleNamespace(id='mock_reflect'), self.candidate, self.trace)
        self.assertEqual(directive.strategy_id, 'loop_nudge')
        exhausted = RepairEngine(retail(), strategies=())
        repairer = OrchestratorRepairer(exhausted)
        strategy = SimpleNamespace(id='mock', template='Fix step {step_index}: {rationale}')
        directive = repairer.build(strategy, self.candidate, self.trace)
        self.assertEqual((directive.kind, directive.message), ('inject_guidance',
                                                               'Fix step 8: looping on get_order_details'))
        self.assertEqual(repairer.build(SimpleNamespace(id='mock'), self.candidate, self.trace).kind, 'retry')

    def test_terminal_plans_replay_unchanged(self):
        candidate = SimpleNamespace(step_index=2, agent='planner', category='GuardrailsTriggered', share=1.0,
                                    rationale='forbidden', evidence_refs=(), source='fa')
        directive = self.repairer.build(SimpleNamespace(id='compensate_and_escalate'), candidate, self.trace)
        self.assertEqual(directive.kind, 'none')
        self.assertTrue(self.repairer.last_plan('ep-1').terminal)

    def test_custom_factory_and_verified_outcome(self):
        made = []

        def factory(kind, strategy_id, message=''):
            made.append(kind)
            return RepairDirective(kind, strategy_id, message)

        lessons = LessonMemory()
        repairer = OrchestratorRepairer(RepairEngine(retail(), lessons=lessons), factory)
        repairer.build(SimpleNamespace(id='loop_nudge'), self.candidate, self.trace)
        repairer.record_outcome('ep-1', verified=True, rollout=self.trace)
        self.assertEqual(made, ['inject_guidance'])
        self.assertEqual(len(lessons.lessons), 1)
        repairer.record_outcome('unknown-trajectory', verified=True)


@dataclass
class FakeDirective:
    rollback_step: int
    kind: str = 'guidance'
    text: str = ''
    seed_offset: int = 1


class FakeEnv:
    """The benchmark RecoveryEnv surface: trace, task_view, hypotheses, remaining(), replay()."""

    def __init__(self, trace, hypotheses, rollouts, attempts=3):
        self.trajectory_id = trace['trajectory_id']
        self.trace, self.hypotheses = trace, hypotheses
        self.task_view = {'first_user_message': trace['instruction'], 'policy': 'be nice',
                          'tools': {a: [t.to_dict() for t in retail().tools.values() if t.name in names]
                                    for a, names in retail().agent_tools.items()}}
        self._rollouts, self._attempts, self.directives = list(rollouts), attempts, []
        self._ids = itertools.count(1)

    def remaining(self):
        return {'tokens': 1000, 'attempts': self._attempts - len(self.directives), 'seconds': 60}

    def replay(self, directive):
        self.directives.append(directive)
        trace, termination = self._rollouts.pop(0)
        return SimpleNamespace(rollout_id=f'r{next(self._ids)}', trace=trace, termination=termination)


class BenchmarkPolicyTest(unittest.TestCase):
    def policy(self, **kwargs):
        return RepairStrategyPolicy(directive_cls=FakeDirective, **kwargs)

    def test_escalates_the_ladder_until_a_rollout_is_accepted(self):
        env = FakeEnv(loop_builder().ir(), [{'step_index': 8, 'agent': 'order_agent', 'mode': 'cycles'}],
                      [(loop_builder().ir(), 'max_steps'), (recovered_builder().ir(), 'user_stop')])
        policy = self.policy()
        decision = policy.recover(env)
        self.assertEqual(decision, {'delivered': 'r2', 'status': 'intervened',
                                    'notes': 'loop_force_change accepted at step 8'})
        self.assertEqual([d.kind for d in env.directives], ['guidance', 'guidance'])
        attempts = policy.log['ep-1'].attempts
        self.assertEqual([a['accepted'] for a in attempts], [False, True])
        self.assertTrue(attempts[0]['failures'])

    def test_budget_and_unaccepted_delivery(self):
        rollouts = [(loop_builder().ir(), 'max_steps')] * 4
        env = FakeEnv(loop_builder().ir(), [{'step_index': 8, 'mode': 'cycles'}], rollouts, attempts=2)
        decision = self.policy().recover(env)                  # default: deliver the last replay
        self.assertEqual((decision['delivered'], decision['status'], len(env.directives)), ('r2', 'intervened', 2))
        env = FakeEnv(loop_builder().ir(), [{'step_index': 8, 'mode': 'cycles'}], rollouts, attempts=2)
        policy = self.policy(deliver_unaccepted=False)         # opt-in: escalate instead
        decision = policy.recover(env)
        self.assertEqual((decision['delivered'], decision['status']), (None, 'escalated'))
        fallback = policy.log['ep-1'].attempts[-1]
        self.assertEqual(fallback['strategy'], 'rollback_compensate_escalate')
        self.assertIn('Tried: loop_nudge (failed), loop_force_change (failed)', fallback['summary'])

    def test_fallback_when_nothing_could_be_replayed(self):
        b = RunBuilder('ep-1')
        b.user('hello')
        b.reply('planner', 'Hi, how can I help?')
        env = FakeEnv(b.ir(), [{'step_index': 2, 'category': 'UnderspecifiedUserIntent'}], [])
        decision = self.policy().recover(env)
        self.assertEqual((decision['delivered'], decision['status'], env.directives), (None, 'escalated', []))
        self.assertTrue(decision['notes'].startswith('rollback_compensate_escalate'))

    def test_second_hypothesis_is_tried_before_falling_back(self):
        hypotheses = [{'step_index': 2, 'category': 'IntentNotSupported'}, {'step_index': 8, 'mode': 'cycles'}]
        env = FakeEnv(loop_builder().ir(), hypotheses, [(recovered_builder().ir(), 'user_stop')])
        decision = self.policy().recover(env)
        self.assertEqual((decision['delivered'], decision['status']), ('r1', 'intervened'))

    def test_no_or_invalid_hypotheses(self):
        env = FakeEnv(loop_builder().ir(), [{'step_index': 8, 'mode': 'telepathy'}], [])
        policy = self.policy()
        self.assertEqual(policy.recover(env)['status'], 'abstained')
        self.assertIn('telepathy', policy.log['ep-1'].skipped_hypotheses[0])

    def test_terminal_plan_escalates_without_replay(self):
        env = FakeEnv(loop_builder().ir(), [{'step_index': 2, 'category': 'IntentNotSupported'}], [])
        decision = self.policy().recover(env)
        self.assertEqual((decision['status'], env.directives), ('escalated', []))
        self.assertTrue(decision['notes'].startswith('compensate_and_escalate'))

    def test_task_from_view(self):
        view = FakeEnv(loop_builder().ir(), [], []).task_view
        task = task_from_view(view)
        self.assertEqual(set(task.tools_of('returns_agent')), set(retail().tools_of('returns_agent')))
        self.assertEqual(task.policy, 'be nice')
        view['tools']['planner'].append({'name': 'new_tool', 'parameters': {'type': 'object', 'properties': {}}})
        self.assertIn('new_tool', task_from_view(view).tools)

    @unittest.skipIf(importlib.util.find_spec('benchmark') is not None, 'benchmark package is installed')
    def test_missing_benchmark_package_is_reported(self):
        with self.assertRaisesRegex(RuntimeError, 'directive_cls'):
            RepairStrategyPolicy().directive_cls


if __name__ == '__main__':
    unittest.main()
