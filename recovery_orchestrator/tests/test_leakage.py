"""Gold actions, outputs, rewards and failure labels must never reach the decision/repair path."""
import ast
import json
import unittest
from pathlib import Path

from recovery_orchestrator.orchestrator import Orchestrator
from recovery_orchestrator.policy import EVConfig, EVPolicy
from recovery_orchestrator.rca import load_payload
from recovery_orchestrator.registry import mock_registry
from recovery_orchestrator.tau_retail import WRITE_TOOLS, load_tasks, load_traces

from .helpers import SYN_ID, SYN_RUN, TAU_RAW, make_env, make_verifier, syn_trace

PACKAGE = Path(__file__).resolve().parents[1]
GOLD_READERS = {'evaluation.py', 'cli.py'}   # cli imports evaluation lazily, only for eval commands
SENTINEL = 'GOLDSENTINEL'


def imported_modules(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or '')
            names.update(a.name for a in node.names)
    return names


class NoLedger:
    """Environment proxy that fails if the orchestrator reaches for ground truth."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        if name == 'ledger':
            raise AssertionError('orchestrator accessed the score ledger')
        return getattr(self._env, name)


class LeakageTest(unittest.TestCase):
    def test_decision_path_never_imports_evaluation_or_agentrx(self):
        paths = [p for p in PACKAGE.rglob('*.py') if not {'tests', 'integration'} & set(p.relative_to(PACKAGE).parts)]
        self.assertIn('benchmark.py', {p.name for p in paths})
        for path in paths:
            names = imported_modules(path)
            for other in ('agentrx', 'benchmark', 'repair_strategies'):
                self.assertFalse(any(n == other or n.startswith(other + '.') for n in names), f'{path.name}: {other}')
            if path.name not in GOLD_READERS:
                self.assertFalse({'evaluation', 'recovery_orchestrator.evaluation'} & names, path.name)

    def test_tau_loaders_drop_gold(self):
        dumped = repr(load_tasks(TAU_RAW)) + repr(load_traces(TAU_RAW))
        self.assertNotIn(SENTINEL, dumped)
        self.assertNotIn('reward', dumped)

    def test_full_run_sees_no_gold(self):
        env = make_env()
        verifier, llm = make_verifier()
        policy = EVPolicy(mock_registry(), EVConfig(respect_irreversible=False), write_tools=WRITE_TOOLS)
        outcome = Orchestrator(NoLedger(env), verifier, policy).run(syn_trace(), load_payload(SYN_RUN, SYN_ID, 'tau'))
        self.assertEqual(outcome.status, 'recovered')
        seen = json.dumps(llm.requests) + json.dumps([c['directive'].message for c in env.calls]) \
            + json.dumps(outcome.to_dict(), default=str)
        self.assertNotIn(SENTINEL, seen)
        self.assertNotIn('reward', seen)

    def test_verifier_never_sees_repair_or_rationale(self):
        env = make_env()
        verifier, llm = make_verifier()
        policy = EVPolicy(mock_registry(), EVConfig(respect_irreversible=False), write_tools=WRITE_TOOLS)
        payload = load_payload(SYN_RUN, SYN_ID, 'tau')
        Orchestrator(env, verifier, policy).run(syn_trace(), payload)
        directive = env.calls[0]['directive'].message
        self.assertIn(payload.top.rationale, directive)          # the repair does use the RCA rationale
        requests = json.dumps(llm.requests)
        self.assertNotIn(payload.top.rationale, requests)
        self.assertNotIn('re-read every tool output', requests)


if __name__ == '__main__':
    unittest.main()
