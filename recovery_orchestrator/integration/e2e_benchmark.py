"""End-to-end run on the benchmark: attribution -> repair -> orchestrator -> scored recovery.

All conditions go through the same harness (``benchmark.scoring.recovery.run_recovery``),
the same corpus and the same reward ledger, so the numbers are directly comparable.

    python3 -m benchmark smoke --out runs/e2e/benchmark            # corpus + heuristic submission
    python3 recovery_orchestrator/integration/e2e_benchmark.py \
        --corpus runs/e2e/benchmark/corpus --submission runs/e2e/benchmark/submissions/heuristic.json \
        --out runs/e2e/recovery

``--backend scripted`` (default) uses the benchmark's scripted fixture agents. No LLM is
called, so the run checks the pipeline, not any model. The orchestrator then uses
``CallbackVerifier`` with the baselines' own observable check (``looks_ok``), so every
policy is accepted by the same rule. With a live backend, use ``FreshContextVerifier``.

This is an integration harness (it imports all three packages); the packages
themselves still never import one another.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark.env.scripted import scripted_llm                               # noqa: E402
from benchmark.env.workflow import Directive                                  # noqa: E402
from benchmark.io import read_json                                            # noqa: E402
from benchmark.scoring.recovery import POLICIES, BudgetCaps, looks_ok, run_recovery  # noqa: E402
from benchmark.scoring.report import markdown                                 # noqa: E402
from recovery_orchestrator.adapters.benchmark import CallbackVerifier, OrchestratorPolicy  # noqa: E402
from recovery_orchestrator.budget import BudgetCaps as OrchCaps               # noqa: E402
from recovery_orchestrator.policy import BaselinePolicy, EVConfig, EVPolicy, UnmodifiedReplayPolicy  # noqa: E402
from recovery_orchestrator.registry import repair_registry                    # noqa: E402
from recovery_orchestrator.replay import RepairDirective                      # noqa: E402
from recovery_orchestrator.tau_retail import WRITE_TOOLS                      # noqa: E402
from repair_strategies.adapters.benchmark import RepairStrategyPolicy, task_from_view  # noqa: E402
from repair_strategies.adapters.orchestrator import OrchestratorRepairer      # noqa: E402
from repair_strategies.engine import RepairEngine                             # noqa: E402

# Replay rebuilds the environment, so rolling back across writes is safe here.
SIM = EVConfig(respect_irreversible=False)


def orchestrator(name, policy_factory, repairer='templates', **kwargs):
    registry = repair_registry()

    def verifier_factory(environment):
        return CallbackVerifier(looks_ok, lambda: environment)

    def repairer_factory(task_view):
        if repairer == 'repair_engine':
            return OrchestratorRepairer(RepairEngine(task_from_view(task_view)), RepairDirective)
        return None                                   # registry templates (TemplateRepairer)

    def caps_factory(remaining):                      # stay inside the harness's own caps
        return OrchCaps(max_tokens=max(0, int(remaining['tokens'])), max_seconds=remaining['seconds'],
                        max_attempts=max(0, int(remaining['attempts'])), max_rollbacks=3)

    return OrchestratorPolicy(Directive, lambda: policy_factory(registry), verifier_factory,
                              repairer_factory, caps_factory, name=name, **kwargs)


def conditions():
    ev = lambda reg: EVPolicy(reg, SIM, write_tools=WRITE_TOOLS)                  # noqa: E731
    base = lambda reg: BaselinePolicy(reg, retries=3, config=SIM, write_tools=WRITE_TOOLS)  # noqa: E731
    yield 'no_recovery', POLICIES['no_recovery'](), False
    yield 'full_restart', POLICIES['full_restart'](), False
    yield 'generic_reflection', POLICIES['generic_reflection'](), False
    yield 'fixed_rules', POLICIES['fixed_rules'](), None
    yield 'repair_strategies', RepairStrategyPolicy(Directive), None
    yield 'orch_baseline', orchestrator('orch_baseline', base), None
    yield 'orch_ev', orchestrator('orch_ev', ev), None
    yield 'orch_ev+repair_engine', orchestrator('orch_ev+repair_engine', ev, repairer='repair_engine'), None
    yield 'orch_unmodified', orchestrator('orch_unmodified', lambda reg: UnmodifiedReplayPolicy(ev(reg))), None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--corpus', required=True, type=Path)
    parser.add_argument('--submission', required=True, type=Path, help='attribution submission (hypotheses)')
    parser.add_argument('--split', default='test')
    parser.add_argument('--out', type=Path, default=Path('runs/e2e/recovery'))
    parser.add_argument('--backend', choices=['scripted'], default='scripted')
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    hypotheses = {row['trajectory_id']: row.get('hypotheses', [])
                  for row in read_json(args.submission)['predictions']}
    attribution = args.submission.stem
    reports, details = [], {}
    for name, policy, uses_attribution in conditions():
        runs = [('', {})] if uses_attribution is False else \
            [(f'+{attribution}', {'hypotheses': hypotheses}), (':oracle', {'oracle': True})]
        for suffix, kwargs in runs:
            report = run_recovery(args.corpus, args.split, policy, scripted_llm, BudgetCaps(), **kwargs)
            report['policy'] = name + suffix
            reports.append(report)
            if isinstance(policy, OrchestratorPolicy):
                details[report['policy']] = {tid: {'status': o.status, 'reason': o.reason,
                                                   'attempts': [a.applied_strategy for a in o.attempts]}
                                             for tid, o in policy.outcomes.items()}
                policy.outcomes = {}
            m = report['metrics']
            val = lambda k: m[k]['value'] if isinstance(m[k], dict) else m[k]          # noqa: E731
            print(f"{report['policy']:<36} recovery {val('recovery_rate'):.2f}  "
                  f"false-int {val('false_intervention_rate'):.2f}  harm {val('harmful_intervention_rate'):.2f}  "
                  f"cost x{val('cost_vs_original'):.2f}", flush=True)
            (args.out / f"recovery-{report['policy'].replace(':', '-').replace('+', '_')}.json").write_text(
                json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (args.out / 'orchestrator_outcomes.json').write_text(json.dumps(details, indent=2) + '\n', encoding='utf-8')
    note = (f'OFFLINE FIXTURE ({args.backend} agents). Attribution: `{attribution}` submission and gold (oracle). '
            'Orchestrator verifier = the baselines\' observable check (looks_ok). Pipeline check, not model results.')
    (args.out / 'report.md').write_text(markdown(recovery=reports, title='End-to-end recovery', note=note),
                                        encoding='utf-8')
    print(f'report -> {args.out / "report.md"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
