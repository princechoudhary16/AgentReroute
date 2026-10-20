"""Shared fixtures: synthetic tau trace, scripted environment and verifier."""
import json
from pathlib import Path

from recovery_orchestrator.llm import ScriptedLLM
from recovery_orchestrator.replay import ScriptedEnvironment
from recovery_orchestrator.tau_retail import WRITE_TOOLS, load_tasks, load_traces
from recovery_orchestrator.trace import Step, ToolCall
from recovery_orchestrator.verifier import FreshContextVerifier

FIXTURES = Path(__file__).parent / 'fixtures'
SMOKE_RUN = FIXTURES / 'smoke_run'
SYN_RUN = FIXTURES / 'synthetic_run'
CASES = FIXTURES / 'cases.json'
TAU_RAW = FIXTURES / 'tau_raw.json'
SMOKE_ID, SYN_ID = 'fixture-count-mismatch', 'syn-1'

GOOD_REPLY = 'Order #W1 contains Headphones and a Watch. Shall I return the Headphones (111)?'


def syn_trace():
    return load_traces(TAU_RAW)[SYN_ID]


def good_continuation(start):
    return [Step(start, 'assistant', GOOD_REPLY), Step(start + 1, 'user', 'Yes please.'),
            Step(start + 2, 'assistant', '', (ToolCall('return_delivered_order_items',
                                                       {'order_id': '#W1', 'item_ids': ['111']}),)),
            Step(start + 3, 'tool', '{"status": "return requested"}', name='return_delivered_order_items'),
            Step(start + 4, 'assistant', 'Done, the headphones return is requested.')]


def bad_continuation(start):
    return [Step(start, 'assistant', 'Order #W1 only contains a Watch. Shall I return the Watch?'),
            Step(start + 1, 'user', 'Ok.')]


def make_env(fix_with=('reground_tool_output',), tokens=2000, script=None):
    """Replay succeeds (reward 1) only for strategies in ``fix_with``."""
    def default(trace, prefix, directive):
        start = prefix.rollback_step
        if directive.strategy_id in fix_with:
            return good_continuation(start), 1.0, tokens
        return bad_continuation(start), 0.0, tokens
    return ScriptedEnvironment(load_tasks(TAU_RAW), script or default, WRITE_TOOLS)


def verdict_llm(pass_marker=GOOD_REPLY, fail_score=0.2):
    def reply(messages, purpose):
        text = messages[-1]['content']
        if pass_marker in text:
            return json.dumps({'verdict': 'PASS', 'score': 0.9, 'gaps': []})
        return json.dumps({'verdict': 'FAIL', 'score': fail_score, 'gaps': ['wrong item']})
    return ScriptedLLM(reply, tokens_per_call=700)


def make_verifier(**kwargs):
    llm = verdict_llm(**kwargs)
    return FreshContextVerifier(llm), llm
