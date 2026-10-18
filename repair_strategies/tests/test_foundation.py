import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from repair_strategies import taxonomy as tx
from repair_strategies.context import AttemptRecord, Hypothesis, SpecError, TaskSpec, ToolSpec
from repair_strategies.trace import AGENT, TOOL, USER, Trace, TraceError

from .helpers import ORDER, RunBuilder, authenticated


class TaxonomyTest(unittest.TestCase):
    def test_agentrx_names_aliases_and_numbers(self):
        self.assertEqual(tx.normalize('Invalid Invocation').category, tx.INVALID_INVOCATION)
        self.assertEqual(tx.normalize('Plan Adherence Failure').category, tx.INSTRUCTION_ADHERENCE)
        self.assertEqual(tx.normalize(3).category, tx.INVALID_INVOCATION)
        self.assertEqual(tx.normalize('10').category, tx.INCONCLUSIVE)
        self.assertEqual(tx.normalize('cycles').category, tx.CYCLE)

    def test_benchmark_modes_keep_subtype(self):
        self.assertEqual(tx.normalize('context_propagation'), tx.FaultType(tx.MISINTERPRETATION, tx.HANDOFF))
        self.assertEqual(tx.normalize('missing_details'), tx.FaultType(tx.INVENTION, tx.OMISSION))
        self.assertEqual(tx.normalize('drift').category, tx.INTENT_MISALIGNMENT)

    def test_explicit_subtype_must_refine_category(self):
        fault = tx.normalize('MisinterpretationOfToolOutput', 'context_propagation')
        self.assertEqual(fault.subtype, tx.HANDOFF)
        with self.assertRaises(tx.UnknownCategory):
            tx.normalize('SystemFailure', 'context_propagation')

    def test_unknown_labels_are_rejected(self):
        for bad in ('Hallucination', 0, 11, None, ''):
            with self.assertRaises(tx.UnknownCategory):
                tx.normalize(bad)

    def test_agent_names(self):
        self.assertEqual(tx.normalize_agent('Orchestrator (-> WebSurfer)'), 'orchestrator')
        self.assertEqual(tx.normalize_agent('Human'), 'user')
        self.assertEqual(tx.normalize_agent('returns_agent'), 'returns_agent')
        self.assertIsNone(tx.normalize_agent(None))


class TraceTest(unittest.TestCase):
    def test_benchmark_actions_are_unpacked(self):
        b = authenticated(RunBuilder())
        reply = b.reply('planner', 'Your refund will be $45.04.')
        trace = b.trace()
        call = trace.primary(2)
        self.assertEqual(call.kind, AGENT)
        self.assertEqual(call.tool_calls[0].name, 'find_user_id_by_email')
        self.assertEqual(call.text, '')
        self.assertEqual(trace.primary(reply).text, 'Your refund will be $45.04.')
        self.assertEqual(trace.final_reply().step_index, reply)
        self.assertEqual(trace.primary(1).kind, USER)
        self.assertEqual(trace.primary(3).kind, TOOL)

    def test_handoff_is_kept_once(self):
        b = RunBuilder()
        b.user('hi')
        step = b.handoff('planner', 'returns_agent', 'Return item', {'order_id': '#W1'})
        trace = b.trace()
        handoffs = [e for e in trace.at(step) if e.handoff]
        self.assertEqual(len(handoffs), 1)
        self.assertEqual(handoffs[0].handoff['to'], 'returns_agent')
        self.assertEqual(handoffs[0].handoff['context'], {'order_id': '#W1'})
        self.assertIsNone(trace.final_reply())

    def test_tool_exchanges_pair_calls_with_results(self):
        b = authenticated(RunBuilder())
        b.call('returns_agent', 'get_order_details', order_id='#W1000005')
        b.reply('returns_agent', 'no result came back')
        b.tool('returns_agent', 'get_order_details', ORDER, order_id='#W1000005')
        exchanges = b.trace().tool_exchanges()
        self.assertEqual([x.call.name for x in exchanges],
                         ['find_user_id_by_email', 'get_user_details', 'get_order_details', 'get_order_details'])
        self.assertIsNone(exchanges[2].result)
        self.assertEqual(exchanges[3].result.tool_name, 'get_order_details')
        self.assertEqual(exchanges[3].actor, 'returns_agent')

    def test_openai_messages_with_ids_and_malformed_arguments(self):
        messages = [
            {'role': 'user', 'content': 'cancel'},
            {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': 'a', 'function': {'name': 'get_order_details', 'arguments': '{"order_id": "#W1"}'}},
                {'id': 'b', 'function': {'name': 'get_user_details', 'arguments': '{bad json'}}]},
            {'role': 'tool', 'tool_call_id': 'b', 'name': 'get_user_details', 'content': 'Error: bad'},
            {'role': 'tool', 'tool_call_id': 'a', 'name': 'get_order_details', 'content': '{}'},
        ]
        trace = Trace.from_messages(messages, 'x', 'tau')
        calls = trace.primary(2).tool_calls
        self.assertEqual(calls[0].arguments, {'order_id': '#W1'})
        self.assertEqual(calls[1].arguments, {'__raw__': '{bad json'})
        pairs = {x.call.id: x.result.step_index for x in trace.tool_exchanges()}
        self.assertEqual(pairs, {'a': 4, 'b': 3})
        self.assertEqual(Trace.from_ir(trace.to_ir()), trace)

    def test_rollback_points(self):
        b = authenticated(RunBuilder())
        trace = b.trace()
        self.assertEqual(trace.rollback_point(2), 2)          # agent step
        self.assertEqual(trace.rollback_point(3), 2)          # tool result -> caller
        self.assertEqual(trace.rollback_point(1), 2)          # user turn -> next agent turn
        self.assertIsNone(trace.rollback_point(99))
        self.assertEqual(trace.last_user_step_before(5), 1)

    def test_from_object_and_round_trip(self):
        call = SimpleNamespace(name='get_order_details', arguments={'order_id': '#W1'}, id='c1')
        obj = SimpleNamespace(trajectory_id='7', domain='tau', instruction='i', steps=[
            SimpleNamespace(index=1, role='user', content='hi', tool_calls=()),
            SimpleNamespace(index=2, role='assistant', content='', tool_calls=(call,)),
            SimpleNamespace(index=3, role='tool', content='{}', tool_calls=(), name='get_order_details')])
        trace = Trace.from_any(obj)
        self.assertEqual(trace.tool_exchanges()[0].result.step_index, 3)
        again = Trace.from_ir(json.loads(json.dumps(trace.to_ir())))
        self.assertEqual(again, trace)

    def test_load_accepts_fa_case_lists_and_tau_records(self):
        b = authenticated(RunBuilder())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cases.json'
            path.write_text(json.dumps([b.ir()]))
            self.assertEqual(Trace.load(path).trajectory_id, 't-1')
            path.write_text(json.dumps([b.ir(), b.ir()]))
            with self.assertRaises(TraceError):
                Trace.load(path)
        tau = Trace.from_any({'task_id': 5, 'traj': [{'role': 'user', 'content': 'x'}]})
        self.assertEqual((tau.trajectory_id, tau.domain), ('5', 'tau'))

    def test_invalid_traces_are_rejected(self):
        with self.assertRaises(TraceError):
            Trace.from_ir({'steps': [{'index': 1, 'substeps': [{'role': 'user'}]},
                                     {'index': 1, 'substeps': [{'role': 'user'}]}]})
        with self.assertRaises(TraceError):
            Trace.from_ir({'steps': [{'index': 1, 'substeps': [{'content': 'no role'}]}]})
        with self.assertRaises(TraceError):
            Trace.from_ir({'steps': [{'index': 'x', 'substeps': [{'role': 'user'}]}]})
        with self.assertRaises(TraceError):
            Trace.from_any(42)


class ContextTest(unittest.TestCase):
    def test_hypothesis_from_fa_prediction(self):
        h = Hypothesis.from_mapping({'step_index': 7, 'agent': 'Assistant', 'category': 'Invalid Invocation',
                                     'rationale': 'bad args', 'evidence_refs': [{'step_index': 6, 'substep_index': 1}]})
        self.assertEqual((h.category, h.agent, h.evidence_steps), (tx.INVALID_INVOCATION, 'assistant', (6,)))

    def test_hypothesis_from_benchmark_and_orchestrator(self):
        h = Hypothesis.from_mapping({'step_index': 9, 'agent': 'planner', 'mode': 'context_propagation',
                                     'confidence': 0.7})
        self.assertEqual((h.category, h.subtype, h.confidence), (tx.MISINTERPRETATION, tx.HANDOFF, 0.7))
        candidate = SimpleNamespace(step_index=4, agent='assistant', category='SystemFailure', share=0.5,
                                    rationale='timeout', evidence_refs=((3, 1),), source='fa')
        h = Hypothesis.from_object(candidate)
        self.assertEqual((h.step_index, h.confidence, h.evidence_steps), (4, 0.5, (3,)))

    def test_hypothesis_validation(self):
        with self.assertRaises(SpecError):
            Hypothesis(1, 'SystemFailure', confidence=1.5)
        with self.assertRaises(SpecError):
            Hypothesis('3', 'SystemFailure')
        with self.assertRaises(tx.UnknownCategory):
            Hypothesis(1, 'nonsense')

    def test_retail_preset(self):
        task = TaskSpec.preset('retail')
        self.assertIn('cancel_pending_order', task.write_tools)
        self.assertNotIn('get_order_details', task.write_tools)
        self.assertEqual(task.tool('get_order_details').required, ('order_id',))
        self.assertIn('return_delivered_order_items', task.tools_of('returns_agent'))
        self.assertEqual(task.alternatives('find_user_id_by_email'), ('find_user_id_by_name_zip',))
        self.assertEqual(TaskSpec.from_dict(task.to_dict()), task)
        self.assertEqual(task.with_task('do x').instruction, 'do x')
        with self.assertRaises(SpecError):
            TaskSpec.preset('airline-not-shipped')

    def test_task_spec_validation(self):
        tool = {'name': 'a', 'parameters': {'type': 'object', 'properties': {}}}
        with self.assertRaises(SpecError):
            TaskSpec.from_dict({'tools': [tool, tool]})
        with self.assertRaises(SpecError):
            TaskSpec.from_dict({'tools': [tool], 'equivalents': {'a': ['missing']}})
        with self.assertRaises(SpecError):
            TaskSpec.from_dict({'tools': [tool], 'write_tools': ['missing']})
        with self.assertRaises(SpecError):
            ToolSpec('x', effect='delete')
        spec = TaskSpec.from_dict({'tools': [{'type': 'function', 'function': tool}], 'write_tools': ['a']})
        self.assertTrue(spec.is_write('a'))

    def test_attempt_outcomes(self):
        AttemptRecord('s', tx.CYCLE, 3, 'failed')
        with self.assertRaises(SpecError):
            AttemptRecord('s', tx.CYCLE, 3, 'maybe')


if __name__ == '__main__':
    unittest.main()
