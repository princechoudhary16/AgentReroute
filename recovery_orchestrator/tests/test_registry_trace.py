import unittest

from recovery_orchestrator.detectors import RuntimeMonitor, detect_loops, detect_tool_errors
from recovery_orchestrator.registry import RegistryError, StrategyRegistry, mock_registry
from recovery_orchestrator.tau_retail import WRITE_TOOLS
from recovery_orchestrator.trace import Step, ToolCall, Trace, build_prefix, rollback_point

from .helpers import syn_trace


def strategy(**over):
    item = {'id': 's', 'kind': 'reasoning_repair', 'targets': ['InvalidInvocation'],
            'cost': {'tokens': 10, 'latency_s': 1}, 'prior_success': 0.5}
    item.update(over)
    return {'schema_version': 1, 'strategies': [item]}


class RegistryTest(unittest.TestCase):
    def test_mock_registry(self):
        reg = mock_registry()
        self.assertGreaterEqual(len(reg), 5)
        self.assertEqual(reg.for_category('MisinterpretationOfToolOutput')[0].id, 'reground_tool_output')
        self.assertEqual(reg.for_category('Cycle')[0].id, 'break_loop')
        self.assertNotIn('generic_reflect', [s.id for s in reg.for_category('Cycle')])

    def test_repair_team_snapshot(self):
        from recovery_orchestrator.registry import default_registry, repair_registry
        reg = repair_registry()
        self.assertEqual(default_registry().source, 'repair_registry')
        self.assertEqual(len(reg), 20)
        from recovery_orchestrator.taxonomy import CATEGORIES, CYCLE
        uncovered = [c for c in list(CATEGORIES) + [CYCLE] if c != 'Inconclusive' and not reg.for_category(c)]
        self.assertEqual(uncovered, [])

    def test_validation(self):
        StrategyRegistry.from_dict(strategy())
        for bad in [strategy(prior_success=1.5), strategy(kind='magic'), strategy(targets=[]),
                    strategy(targets=['Drift']), strategy(targets=['Inconclusive']), strategy(colour='red'),
                    {'schema_version': 2, 'strategies': []}]:
            with self.assertRaises(RegistryError):
                StrategyRegistry.from_dict(bad)
        with self.assertRaises(RegistryError):
            StrategyRegistry.from_dict({'schema_version': 1, 'strategies': strategy()['strategies'] * 2})


class TraceTest(unittest.TestCase):
    def test_rollback_points(self):
        t = syn_trace()
        self.assertEqual(rollback_point(t, 7), 7)      # agent step
        self.assertEqual(rollback_point(t, 6), 5)      # tool output -> the call that produced it
        self.assertEqual(rollback_point(t, 2), 3)      # user turn -> agent's reply
        self.assertEqual(rollback_point(t, 1), 3)      # system prompt -> first agent step
        self.assertIsNone(rollback_point(t, 99))
        t.steps = t.steps[:8]                          # trace ends on a user turn (cut off)
        self.assertEqual(rollback_point(t, 8), 9)      # resume point
        self.assertEqual(build_prefix(t, 9).steps[-1].index, 8)

    def test_prefix_and_irreversible(self):
        p = build_prefix(syn_trace(), 7, WRITE_TOOLS)
        self.assertEqual([s.index for s in p.steps], list(range(1, 7)))
        self.assertEqual([c.name for c in p.tool_calls], ['find_user_id_by_email', 'get_order_details'])
        self.assertEqual(p.irreversible_downstream, (9,))
        self.assertEqual(build_prefix(syn_trace(), 11, WRITE_TOOLS).irreversible_in_prefix, (9,))


def looping_trace(n=3, write_between=False):
    steps, i = [Step(1, 'user', 'where is my order?')], 2
    for k in range(n):
        if write_between and k == 1:
            steps += [Step(i, 'assistant', '', (ToolCall('modify_user_address', {'a': 1}),)),
                      Step(i + 1, 'tool', 'ok', name='modify_user_address')]
            i += 2
        steps += [Step(i, 'assistant', '', (ToolCall('get_order_details', {'order_id': '#W1'}),)),
                  Step(i + 1, 'tool', '{"status": "pending"}', name='get_order_details')]
        i += 2
    return Trace('loop', 'tau', steps)


class DetectorTest(unittest.TestCase):
    def test_loop(self):
        found = detect_loops(looping_trace(), write_tools=WRITE_TOOLS)
        self.assertEqual([(f.kind, f.step_index, f.category) for f in found], [('loop', 4, 'Cycle')])
        self.assertEqual(detect_loops(looping_trace(2)), [])
        self.assertEqual(detect_loops(looping_trace(3, write_between=True), write_tools=WRITE_TOOLS), [])

    def test_repeated_message(self):
        steps = [Step(i, 'Orchestrator (-> WebSurfer)', 'Search again') for i in range(1, 4)]
        self.assertEqual(detect_loops(Trace('m', 'magentic', steps))[0].step_index, 2)

    def test_tool_errors(self):
        t = Trace('e', 'tau', [Step(1, 'assistant', '', (ToolCall('cancel_pending_order', {'order_id': '#1'}),)),
                               Step(2, 'tool', 'Error: non-pending order cannot be cancelled')])
        self.assertEqual([(f.step_index, f.category) for f in detect_tool_errors(t)], [(1, 'InvalidInvocation')])
        self.assertEqual(detect_tool_errors(syn_trace()), [])

    def test_runtime_monitor_fires_once(self):
        monitor = RuntimeMonitor('loop', 'tau', write_tools=WRITE_TOOLS)
        fired = [f for s in looping_trace(4).steps for f in monitor.observe(s)]
        self.assertEqual(len(fired), 1)


if __name__ == '__main__':
    unittest.main()
