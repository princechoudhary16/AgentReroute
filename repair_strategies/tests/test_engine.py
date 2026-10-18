import unittest

from repair_strategies.context import AttemptRecord, Hypothesis
from repair_strategies.engine import ERROR, PLANNED, SKIPPED, TRIED, RepairEngine
from repair_strategies.llm import ScriptedLLM
from repair_strategies.memory.lessons import LessonMemory
from repair_strategies.memory.tool_outcomes import ToolOutcomeMemory
from repair_strategies.plan import InjectMessage
from repair_strategies.strategies import Skip, Strategy, StrategySpec, default_strategies

from .helpers import RunBuilder, authenticated, retail
from .strategy_helpers import VARIANTS


def loop_run():
    b = authenticated(RunBuilder())
    for _ in range(3):
        b.tool('order_agent', 'get_order_details', 'Error: still processing, retry', order_id='#W1000005')
    return b.trace()


def misread_run():
    b = RunBuilder()
    b.user('How many T-shirt options are available?')
    b.tool('product_agent', 'get_product_details', VARIANTS, product_id='9523456873')
    step = b.reply('product_agent', 'There are 3 available options.')
    return b.trace(), step


class Exploding(Strategy):
    spec = StrategySpec(id='exploding', name='x', categories=('Cycle',), rung=1, kind='loop_break', description='x')

    def plan(self, ctx):
        raise RuntimeError('bug in a strategy')


class LadderTest(unittest.TestCase):
    def setUp(self):
        self.engine = RepairEngine(retail())
        self.cycle = Hypothesis(8, 'Cycle')

    def test_ladder_advances_with_failed_attempts(self):
        trace, history, order = loop_run(), [], []
        while True:
            plan = self.engine.propose(trace, self.cycle, history)
            if plan is None:
                break
            order.append(plan.strategy_id)
            history.append(AttemptRecord(plan.strategy_id, plan.category, plan.rollback_step, 'failed'))
        self.assertEqual(order, ['loop_nudge', 'loop_force_change', 'loop_replan', 'loop_rollback'])

    def test_verified_fault_needs_no_more_plans(self):
        done = [AttemptRecord('loop_nudge', 'Cycle', 8, 'verified')]
        self.assertIsNone(self.engine.propose(loop_run(), self.cycle, done))

    def test_inapplicable_rungs_are_skipped_with_reasons(self):
        trace, step = misread_run()
        candidates = self.engine.evaluate(trace, Hypothesis(step, 'MisinterpretationOfToolOutput'))
        status = {c.strategy_id: (c.status, c.reason) for c in candidates}
        self.assertEqual(status['repair_handoff'][0], SKIPPED)
        self.assertEqual(status['structured_reading'][0], PLANNED)
        self.assertEqual(status['independent_reading'], (SKIPPED, 'needs an LLM client'))
        self.assertEqual(self.engine.propose(trace, Hypothesis(step, 'MisinterpretationOfToolOutput')).strategy_id,
                         'structured_reading')
        self.assertEqual(candidates[0].to_dict()['status'], SKIPPED)

    def test_llm_rungs_run_when_a_client_is_given(self):
        trace, step = misread_run()
        engine = RepairEngine(retail(), llm=ScriptedLLM(['{"values": {"available": 2}}']))
        history = [AttemptRecord('structured_reading', 'MisinterpretationOfToolOutput', step, 'failed')]
        plan = engine.propose(trace, Hypothesis(step, 'MisinterpretationOfToolOutput'), history)
        self.assertEqual(plan.strategy_id, 'independent_reading')

    def test_tried_and_failing_strategies_are_reported(self):
        engine = RepairEngine(retail(), strategies=(Exploding(),) + default_strategies())
        history = [AttemptRecord('loop_nudge', 'Cycle', 8, 'rejected')]
        candidates = engine.evaluate(loop_run(), self.cycle, history)
        status = {c.strategy_id: c.status for c in candidates}
        self.assertEqual((status['exploding'], status['loop_nudge'], status['loop_force_change']),
                         (ERROR, TRIED, PLANNED))
        self.assertIn('bug in a strategy', candidates[0].reason)
        self.assertEqual(engine.propose(loop_run(), self.cycle, history).strategy_id, 'loop_force_change')

    def test_plan_for_named_strategy(self):
        plan = self.engine.plan_for('loop_replan', loop_run(), self.cycle)
        self.assertEqual(plan.rollback_step, 6)
        self.assertIsInstance(self.engine.plan_for('hard_call_repair', loop_run(), self.cycle), Skip)
        self.assertIsInstance(self.engine.plan_for('independent_reading', loop_run(),
                                                   Hypothesis(8, 'MisinterpretationOfToolOutput')), Skip)
        with self.assertRaises(KeyError):
            self.engine.plan_for('nope', loop_run(), self.cycle)
        failed = [AttemptRecord('loop_replan', 'Cycle', 6, 'failed')]
        self.assertIsInstance(self.engine.plan_for('loop_replan', loop_run(), self.cycle, failed), Skip)

    def test_duplicate_strategy_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            RepairEngine(retail(), strategies=default_strategies() + default_strategies()[:1])


class LearningTest(unittest.TestCase):
    def test_verified_repairs_become_lessons_that_fade_on_failure(self):
        lessons = LessonMemory()
        engine = RepairEngine(retail(), lessons=lessons)
        trace, hypothesis = loop_run(), Hypothesis(8, 'Cycle', agent='order_agent')
        first = engine.propose(trace, hypothesis)
        self.assertEqual(first.lessons, ())
        record = engine.record_outcome(first, verified=True)
        self.assertEqual(record.outcome, 'verified')
        [lesson] = lessons.lessons.values()
        self.assertEqual((lesson.category, lesson.agents), ('Cycle', ('order_agent',)))

        second = engine.propose(trace, hypothesis)
        self.assertEqual(second.lessons, (lesson.lesson_id,))
        self.assertIn(lesson.text, second.operations[-1].content)
        self.assertIsInstance(second.operations[-1], InjectMessage)
        engine.record_outcome(second, verified=True)
        self.assertEqual(lesson.verified, 2)                     # one count per verified repair

        for _ in range(3):
            engine.record_outcome(engine.propose(trace, hypothesis), verified=False)
        self.assertEqual(engine.propose(trace, hypothesis).lessons, ())

    def test_terminal_plans_get_no_lessons_and_teach_none(self):
        lessons = LessonMemory()
        lessons.add_verified('IntentNotSupported', 'something', 's')
        engine = RepairEngine(retail(), lessons=lessons)
        plan = engine.propose(loop_run(), Hypothesis(2, 'IntentNotSupported'))
        self.assertTrue(plan.terminal)
        self.assertEqual(plan.lessons, ())
        engine.record_outcome(plan, verified=True)
        self.assertEqual(len(lessons.lessons), 1)

    def test_rollouts_update_tool_memory_once_per_plan(self):
        memory = ToolOutcomeMemory()
        engine = RepairEngine(retail(), tool_memory=memory)
        trace = loop_run()
        plan = engine.propose(trace, Hypothesis(8, 'Cycle'))
        engine.record_outcome(plan, verified=False, rollout=trace)
        engine.record_outcome(plan, verified=False, rollout=trace)
        self.assertEqual(memory.stats['get_order_details'].failures, 3)
        other = engine.propose(trace, Hypothesis(8, 'Cycle'), [engine.record_outcome(plan, False)])
        engine.record_outcome(other, verified=False, rollout=trace)
        self.assertEqual(memory.stats['get_order_details'].failures, 6)


if __name__ == '__main__':
    unittest.main()
