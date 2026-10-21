"""Scoring and controls. The ONLY module that reads gold labels or rewards.

Nothing on the decision, repair, replay or verification path may import this
module (enforced by ``tests/test_leakage.py``). Gold enters the decision path
only deliberately, through the oracle and mis-attribution payloads, which are
experimental conditions, and never as actions, outputs or rewards.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .policy import INTERVENE
from .rca import Candidate, RCAPayload
from .replay import ScoreLedger
from .taxonomy import normalize_agent, normalize_category
from .trace import Trace, rollback_point


# ----------------------------------------------------------------------- gold data

@dataclass(frozen=True)
class GoldFailure:
    failure_id: str
    step_index: int
    category: str
    agent: str | None


@dataclass(frozen=True)
class GoldLabel:
    trajectory_id: str
    domain: str
    critical: GoldFailure
    failures: tuple[GoldFailure, ...]


def _failure(raw: dict) -> GoldFailure:
    return GoldFailure(str(raw.get('failure_id', '')), int(raw['step_index']),
                       normalize_category(raw['category']), normalize_agent(raw.get('agent')))


def load_gold(cases_path) -> dict[str, GoldLabel]:
    """FA's prepared ``data/benchmark/cases.json`` -> gold labels by trajectory_id."""
    gold = {}
    for case in json.loads(Path(cases_path).read_text(encoding='utf-8-sig')):
        g = case['gold']
        tid = str(g['trajectory_id'])
        gold[tid] = GoldLabel(tid, case['trajectory']['domain'], _failure(g['critical']),
                              tuple(_failure(f) for f in g.get('failures', [])))
    return gold


def load_tau_rewards(raw_path) -> dict[str, float]:
    """Original reward of each tau trace (all 0.0 in the failed set)."""
    return {str(r['task_id']): float(r['reward'])
            for r in json.loads(Path(raw_path).read_text(encoding='utf-8-sig'))}


# ----------------------------------------------------------------------- controls

def _payload(label: GoldLabel, failure: GoldFailure, source: str) -> RCAPayload:
    cand = Candidate(failure.step_index, failure.agent, failure.category, 1, 1.0,
                     rationale=f'{failure.category} at step {failure.step_index}', source=source)
    return RCAPayload(label.trajectory_id, label.domain, 'ok', [cand], 1, 1, 0.0)


def oracle_payload(label: GoldLabel) -> RCAPayload:
    """Gold critical failure with confidence 1: upper bound on orchestrator performance."""
    return _payload(label, label.critical, 'oracle')


def misattributed_payload(label: GoldLabel) -> RCAPayload | None:
    """A non-critical gold failure at a different step, presented with confidence 1."""
    others = [f for f in label.failures if f.step_index != label.critical.step_index]
    return _payload(label, others[0], 'misattribution') if others else None


# -------------------------------------------------------- offline decision evaluation

def offline_decision_eval(items: list[tuple[Trace, RCAPayload]], gold: dict[str, GoldLabel],
                          policy) -> list[dict]:
    """First decision per trace vs gold (usable on Magentic-One, which can't be replayed)."""
    rows = []
    for trace, payload in items:
        label = gold[str(trace.trajectory_id)]
        decision = policy.decide(payload, trace, policy.make_guard())
        gold_point = rollback_point(trace, label.critical.step_index)
        row = {'trajectory_id': trace.trajectory_id, 'domain': trace.domain,
               'action': decision.action, 'reason': decision.reason,
               'gold_step': label.critical.step_index, 'gold_rollback': gold_point,
               'gold_category': label.critical.category,
               'top_step': payload.top.step_index if payload.top else None,
               'confidence': payload.confidence}
        if decision.action == INTERVENE:
            o = decision.option
            row.update(rollback_step=o.rollback_step, strategy=o.strategy.id,
                       category=o.primary.category, ev=o.ev,
                       exact=o.rollback_step == gold_point,
                       covers_critical=gold_point is not None and o.rollback_step <= gold_point,
                       distance=None if gold_point is None else gold_point - o.rollback_step,
                       category_match=o.primary.category == label.critical.category)
        rows.append(row)
    return rows


def summarize_decisions(rows: list[dict]) -> dict:
    out = {}
    for domain in sorted({r['domain'] for r in rows}) + ['all']:
        group = [r for r in rows if domain == 'all' or r['domain'] == domain]
        acted = [r for r in group if r['action'] == INTERVENE]
        out[domain] = {
            'n': len(group), 'intervene': len(acted),
            'abstain': sum(r['action'] == 'abstain' for r in group),
            'escalate': sum(r['action'] == 'escalate' for r in group),
            'top1_step_accuracy': _rate(sum(r['top_step'] == r['gold_step'] for r in group), len(group)),
            'rollback_exact': _rate(sum(r['exact'] for r in acted), len(acted)),
            'rollback_covers_critical': _rate(sum(r['covers_critical'] for r in acted), len(acted)),
            'category_match': _rate(sum(r['category_match'] for r in acted), len(acted)),
        }
    return out


def calibration_table(rows: list[dict], bins: int = 5) -> list[dict]:
    """Agreement-confidence vs top-1 step accuracy. Run this before trusting EV scores."""
    table = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        group = [r for r in rows if r['top_step'] is not None
                 and (lo <= r['confidence'] < hi or (b == bins - 1 and r['confidence'] == 1.0))]
        table.append({'bin': [round(lo, 3), round(hi, 3)], 'n': len(group),
                      'mean_confidence': _rate(sum(r['confidence'] for r in group), len(group)),
                      'accuracy': _rate(sum(r['top_step'] == r['gold_step'] for r in group), len(group))})
    return table


def binned_calibrator(table: list[dict]):
    """share -> empirical accuracy of its bin (identity for empty bins)."""
    def calibrate(share: float) -> float:
        for row in table:
            lo, hi = row['bin']
            if (lo <= share < hi or (hi == 1.0 and share == 1.0)) and row['accuracy'] is not None:
                return row['accuracy']
        return share
    return calibrate


# ------------------------------------------------------------------ recovery runs

CONDITIONS = ('orchestrator', 'baseline', 'unmodified', 'oracle', 'misattribution')


def run_condition(condition: str, items: list[tuple[Trace, RCAPayload | None]], orchestrator,
                  ledger: ScoreLedger, original_rewards: dict[str, float]) -> list[dict]:
    """Run one condition over traces and score each outcome from the ledger."""
    records = []
    for trace, payload in items:
        tid = str(trace.trajectory_id)
        if payload is None:          # e.g. no non-critical gold failure for mis-attribution
            continue
        outcome = orchestrator.run(trace, payload)
        original = original_rewards.get(tid, 0.0)
        delivered = outcome.final_rollout_id if outcome.status in ('recovered', 'partial') else None
        final = ledger.reward(delivered) if delivered else original
        records.append({
            'trajectory_id': tid, 'condition': condition, 'status': outcome.status,
            'intervened': outcome.intervened, 'verified': outcome.status == 'recovered',
            'original_reward': original, 'final_reward': final,
            'attempt_rewards': [ledger.reward(a.rollout_id) for a in outcome.attempts],
            'tokens': outcome.budget.get('tokens', 0), 'seconds': outcome.budget.get('seconds', 0.0),
            'attempts': outcome.budget.get('attempts', 0), 'reason': outcome.reason})
    return records


def recovery_metrics(records: list[dict], control: list[dict] | None = None) -> dict:
    """Metrics per condition. ``control`` = unmodified-replay records for the same traces."""
    control_success = {r['trajectory_id']: any((x or 0) >= 1.0 for x in r['attempt_rewards'])
                       for r in control or []}
    by_condition = defaultdict(list)
    for r in records:
        by_condition[r['condition']].append(r)
    out = {}
    for condition, group in by_condition.items():
        failed = [r for r in group if r['original_reward'] < 1.0]
        recovered = [r for r in failed if r['final_reward'] >= 1.0]
        verified = [r for r in group if r['verified']]
        intervened = [r for r in group if r['intervened']]
        reference = {r['trajectory_id']: max(r['original_reward'],
                                             1.0 if control_success.get(r['trajectory_id']) else 0.0)
                     for r in group}
        tokens = sum(r['tokens'] for r in group)
        out[condition] = {
            'n': len(group),
            'recovery_rate': _rate(len(recovered), len(failed)),               # ground truth
            'verified_rate': _rate(sum(r['verified'] for r in failed), len(failed)),
            'false_recovery_rate': _rate(sum(r['final_reward'] < 1.0 for r in verified), len(verified)),
            'harm_rate': _rate(sum(r['final_reward'] < reference[r['trajectory_id']] for r in intervened),
                               len(intervened)),
            'unnecessary_intervention_rate': (_rate(sum(control_success.get(r['trajectory_id'], False)
                                                        for r in intervened), len(intervened))
                                              if control is not None else None),
            'intervention_rate': _rate(len(intervened), len(group)),
            'tokens_per_recovered': tokens / len(recovered) if recovered else None,
            'seconds_per_recovered': sum(r['seconds'] for r in group) / len(recovered) if recovered else None,
            'attempts_per_recovered': sum(r['attempts'] for r in group) / len(recovered) if recovered else None,
        }
    return out


def _rate(num, den):
    return num / den if den else None


__all__ = ['GoldFailure', 'GoldLabel', 'load_gold', 'load_tau_rewards', 'oracle_payload',
           'misattributed_payload', 'offline_decision_eval', 'summarize_decisions',
           'calibration_table', 'binned_calibrator', 'run_condition', 'recovery_metrics',
           'CONDITIONS']
