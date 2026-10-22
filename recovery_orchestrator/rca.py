"""RCA payload: the orchestrator's only view of fault attribution (FA).

FA is consumed exclusively through the JSON files it writes. The adapter here
reads the Tier A files of every repetition/variant of one trace and computes
the Tier B aggregates (``confidence``, ``top_k``, ``n_runs`` ...). Tier C
fields are optional and are only filled when FA's official ``rca_payload.json``
provides them.

Run agreement measures *stability*, not correctness: calibrate ``confidence``
against gold labels (see ``evaluation.calibration_table``) before trusting it.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .taxonomy import INCONCLUSIVE, normalize_agent, normalize_category

MAX_TOP_K = 3


@dataclass(frozen=True)
class Candidate:
    """One hypothesised failure location, pooled over runs."""
    step_index: int
    agent: str | None
    category: str
    votes: int
    share: float                 # votes / total_runs (errors and abstentions dilute it)
    rationale: str = ''
    evidence_refs: tuple = ()
    source: str = 'fa'           # 'fa', 'oracle', 'misattribution', 'detector'


@dataclass
class RCAPayload:
    trajectory_id: str
    domain: str
    status: str                  # 'error' if every pooled run failed, 'partial_checker_errors' if
                                 # any run errored or had checker errors, else 'ok'
    candidates: list[Candidate]  # Tier B top_k, ranked by votes then earliest step
    n_runs: int                  # pooled predictions that localized a step (not Inconclusive/error)
    total_runs: int              # all pooled result files, including errors
    inconclusive_share: float
    violations_by_step: dict[int, dict] = field(default_factory=dict)
    fa_decision: str | None = None   # Track B gate (majority): repair_required | within_boundary | inconclusive
    confidence_kind: str = 'run_agreement'   # or 'uncalibrated_model_estimate' (Track B), 'oracle'
    prediction: dict | None = None   # Tier A prediction of a run agreeing with the top candidate
    experiment_identities: tuple = ()
    variants: tuple = ()
    # Tier C (optional; requires FA changes). The orchestrator works without them.
    ranked_probabilities: list[dict] | None = None
    all_failures: list[dict] | None = None
    causal_chain: list[dict] | None = None

    @property
    def confidence(self) -> float:
        """Fraction of pooled runs agreeing on the top step_index."""
        return self.candidates[0].share if self.candidates else 0.0

    @property
    def top(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def is_inconclusive(self) -> bool:
        return not self.candidates

    def to_dict(self) -> dict:
        data = asdict(self)
        data['confidence'] = self.confidence
        data['top_k'] = data.pop('candidates')
        data['violations_by_step'] = {str(k): v for k, v in self.violations_by_step.items()}
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'RCAPayload':
        """Read an official FA ``rca_payload.json`` (or our own ``to_dict`` output)."""
        total = int(data.get('total_runs', data.get('n_runs', 0)) or 0)
        candidates = []
        for item in data.get('top_k', [])[:MAX_TOP_K]:
            votes = int(item.get('votes', 0))
            share = float(item['share']) if 'share' in item else (votes / total if total else 0.0)
            candidates.append(Candidate(
                step_index=int(item['step_index']), agent=item.get('agent'),
                category=normalize_category(item['category']), votes=votes, share=share,
                rationale=item.get('rationale', ''),
                evidence_refs=tuple(_ref(r) for r in item.get('evidence_refs', ())),
                source=item.get('source', 'fa')))
        prediction = data.get('prediction')
        if not candidates and prediction and prediction.get('step_index') is not None \
                and normalize_category(prediction['category']) != INCONCLUSIVE:
            # Tier A only: a single prediction with no aggregate.
            conf = float(data.get('confidence', 1.0))
            candidates.append(Candidate(int(prediction['step_index']), prediction.get('agent'),
                                        normalize_category(prediction['category']), 1, conf,
                                        prediction.get('rationale', '')))
        return cls(
            trajectory_id=str(data['trajectory_id']), domain=data.get('domain', ''),
            status=data.get('status', 'ok'), candidates=candidates,
            n_runs=int(data.get('n_runs', len(candidates))), total_runs=total,
            inconclusive_share=float(data.get('inconclusive_share', 0.0 if candidates else 1.0)),
            violations_by_step={int(k): v for k, v in data.get('violations_by_step', {}).items()},
            prediction=prediction,
            ranked_probabilities=data.get('ranked_probabilities'),
            all_failures=data.get('all_failures'), causal_chain=data.get('causal_chain'))


def _ref(ref):
    return (ref['step_index'], ref['substep_index']) if isinstance(ref, dict) else tuple(ref)


# --------------------------------------------------------------------------- FA files

def trace_dir_name(trajectory_id) -> str:
    """Folder name FA uses per trace: sha256 of the canonical JSON of the id."""
    return hashlib.sha256(json.dumps(trajectory_id, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _read(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _jobs_root(run_dir) -> Path:
    run_dir = Path(run_dir)
    return run_dir / 'jobs' if (run_dir / 'jobs').is_dir() else run_dir


HANDOFF_SCHEMA = 'agentrx-recovery-v1'


def is_full_agentrx(config: dict) -> bool:
    """Default pooling filter: variants that use both violations and the checklist
    (the joint judge, or the Track B ranked judge)."""
    return bool(config.get('evidence')) and bool(config.get('checklist')) \
        and config.get('protocol', 'joint') in ('joint', 'ranked')


def find_runs(run_dir, trajectory_id, domain: str, variants=None) -> list[dict]:
    """Every pooled repetition of one trace: dicts with variant, rep, dir, result."""
    runs = []
    name = trace_dir_name(trajectory_id)
    for variant_dir in sorted(p for p in _jobs_root(run_dir).iterdir() if p.is_dir()):
        if variants is not None and variant_dir.name not in variants:
            continue
        for rep_dir in sorted(variant_dir.glob('rep-*')):
            trace_dir = rep_dir / domain / name
            if not (trace_dir / 'result.json').is_file():
                continue
            result = _read(trace_dir / 'result.json')
            if variants is None and not is_full_agentrx(result.get('config', {})):
                continue
            runs.append({'variant': variant_dir.name, 'rep': rep_dir.name,
                         'dir': trace_dir, 'result': result})
    return runs


def list_trajectories(run_dir) -> list[tuple[str, str]]:
    """(domain, trajectory_id) of every trace diagnosed in an FA run directory."""
    found = set()
    for path in _jobs_root(run_dir).glob('*/rep-*/*/*/result.json'):
        result = _read(path)
        found.add((result['domain'], str(result['trajectory_id'])))
    return sorted(found)


def load_payload(run_dir, trajectory_id, domain: str, variants=None,
                 top_k: int = MAX_TOP_K) -> RCAPayload:
    """Tier A files of all pooled runs -> one RCAPayload with Tier B aggregates.

    ``variants=None`` pools the full AgentRx variants (see ``is_full_agentrx``).
    """
    runs = find_runs(run_dir, trajectory_id, domain, variants)
    if not runs:
        raise FileNotFoundError(f'No FA results for {domain}/{trajectory_id} in {run_dir}')
    official = runs[0]['dir'] / 'rca_payload.json'
    if official.is_file() and variants is None:
        return RCAPayload.from_dict(_read(official))
    handoffs = [run['dir'] / 'recovery_handoff.json' for run in runs]
    if all(h.is_file() for h in handoffs):
        payload = aggregate_handoffs([_read(h) for h in handoffs], top_k=top_k)
        payload.variants = tuple(sorted({r['variant'] for r in runs}))
        return payload
    logs = []
    for run in runs:
        path = run['dir'] / 'validation_log.json'
        logs.append(_read(path) if path.is_file() else run['result'].get('violations', []))
    payload = aggregate([r['result'] for r in runs], logs, top_k=top_k)
    payload.variants = tuple(sorted({r['variant'] for r in runs}))
    return payload


def aggregate(results: list[dict], validation_logs: list[list[dict]] | None = None,
              top_k: int = MAX_TOP_K) -> RCAPayload:
    """Pure Tier B aggregation over the ``result.json`` contents of pooled runs."""
    if not results:
        raise ValueError('aggregate needs at least one result')
    total = len(results)
    localized, inconclusive = [], 0
    for result in results:
        prediction = result.get('prediction')
        if result.get('status') == 'error' or not prediction:
            inconclusive += 1
            continue
        category = normalize_category(prediction.get('category', INCONCLUSIVE))
        if category == INCONCLUSIVE or prediction.get('step_index') is None:
            inconclusive += 1
            continue
        localized.append({**prediction, 'category': category})

    steps = Counter(p['step_index'] for p in localized)
    ranked = sorted(steps, key=lambda s: (-steps[s], s))[:top_k]
    candidates = []
    for step in ranked:
        group = [p for p in localized if p['step_index'] == step]
        agent = _majority([normalize_agent(p.get('agent')) for p in group])
        category = _majority([p['category'] for p in group])
        exemplar = next(p for p in group if p['category'] == category)
        candidates.append(Candidate(
            step_index=step, agent=agent, category=category, votes=len(group),
            share=len(group) / total, rationale=exemplar.get('rationale', ''),
            evidence_refs=tuple(_ref(r) for r in exemplar.get('evidence_refs', ()))))

    prediction = None
    if candidates:
        top = candidates[0]
        prediction = next(p for p in localized
                          if p['step_index'] == top.step_index and p['category'] == top.category)
    statuses = {r.get('status', 'error') for r in results}
    status = 'error' if statuses == {'error'} else 'ok' if statuses == {'ok'} else 'partial_checker_errors'
    first = results[0]
    return RCAPayload(
        trajectory_id=str(first['trajectory_id']), domain=first.get('domain', ''),
        status=status, candidates=candidates, n_runs=len(localized), total_runs=total,
        inconclusive_share=inconclusive / total,
        violations_by_step=violations_by_step(validation_logs or []),
        prediction=prediction,
        experiment_identities=tuple(sorted({r['experiment_identity'] for r in results
                                            if r.get('experiment_identity')})))


def _majority(values):
    counts = Counter(values)
    order = {}
    for i, v in enumerate(values):
        order.setdefault(v, i)
    return min(counts, key=lambda v: (-counts[v], order[v]))


def violations_by_step(logs: list[list[dict]]) -> dict[int, dict]:
    """Secondary signal: {step: {count, runs, categories}} over pooled validation logs."""
    table: dict[int, dict] = {}
    for log in logs:
        flagged = set()
        for item in log:
            if item.get('verdict', 'VIOL') != 'VIOL' or item.get('step_index') is None:
                continue
            step = int(item['step_index'])
            entry = table.setdefault(step, {'count': 0, 'runs': 0, 'categories': []})
            entry['count'] += 1
            for target in item.get('taxonomy_targets', []):
                try:
                    target = normalize_category(target)
                except ValueError:
                    continue
                if target not in entry['categories']:
                    entry['categories'].append(target)
            flagged.add(step)
        for step in flagged:
            table[step]['runs'] += 1
    for entry in table.values():
        entry['categories'].sort()
    return dict(sorted(table.items()))


def aggregate_handoffs(handoffs: list[dict], top_k: int = MAX_TOP_K) -> RCAPayload:
    """Track B ``recovery_handoff.json`` files of pooled runs -> RCAPayload.

    A run contributes candidates only if FA's own gate says act on it: decision
    ``repair_required``, attribution ``ranked``, status not ``error``. Every other
    run counts as inconclusive. A step's share is the mean of its confidence over
    all runs (0 where a run did not list it), so with one run it is the model's
    confidence. These are uncalibrated model estimates, not probabilities.
    """
    if not handoffs:
        raise ValueError('aggregate_handoffs needs at least one handoff')
    for h in handoffs:
        if h.get('schema_version') != HANDOFF_SCHEMA:
            raise ValueError(f'Unsupported handoff schema {h.get("schema_version")!r}')
    total = len(handoffs)
    usable = [h for h in handoffs if h.get('decision') == 'repair_required'
              and h.get('attribution_status') == 'ranked' and h.get('status') != 'error'
              and h.get('candidates')]
    weight, votes, members, first = Counter(), Counter(), {}, {}
    for h in usable:
        for c in h['candidates']:
            step = int(c['step_index'])
            conf = float(c.get('confidence', 0.0))
            weight[step] += conf
            votes[step] += 1
            members.setdefault(step, []).append({**c, 'category': normalize_category(c['category'])})
            first.setdefault(step, (int(c.get('rank', 99)), len(first)))
    ranked = sorted(weight, key=lambda s: (-weight[s], s))[:top_k]
    candidates = []
    for step in ranked:
        group = members[step]
        by_cat, by_agent = Counter(), Counter()
        for c in group:
            by_cat[c['category']] += float(c.get('confidence', 0.0))
            by_agent[normalize_agent(c.get('agent'))] += float(c.get('confidence', 0.0))
        category = max(by_cat, key=lambda k: (by_cat[k], -[c['category'] for c in group].index(k)))
        agent = max(by_agent, key=lambda k: (by_agent[k], -[normalize_agent(c.get('agent')) for c in group].index(k)))
        exemplar = max((c for c in group if c['category'] == category), key=lambda c: float(c.get('confidence', 0)))
        candidates.append(Candidate(
            step_index=step, agent=agent, category=category, votes=votes[step],
            share=weight[step] / total, rationale=exemplar.get('rationale', ''),
            evidence_refs=tuple(_ref(r) for r in exemplar.get('evidence_refs', ()))))
    decisions = Counter(h.get('decision') for h in handoffs)
    decision = max(decisions, key=lambda d: (decisions[d], d == 'repair_required'))
    statuses = {h.get('status', 'error') for h in handoffs}
    status = 'error' if statuses == {'error'} else 'ok' if statuses == {'ok'} else 'partial_checker_errors'
    top = candidates[0] if candidates else None
    prediction = None
    if top is not None:
        prediction = {'step_index': top.step_index, 'agent': top.agent, 'category': top.category,
                      'rationale': top.rationale, 'evidence_refs': [
                          {'step_index': a, 'substep_index': b} for a, b in top.evidence_refs]}
    return RCAPayload(
        trajectory_id=str(handoffs[0]['trajectory_id']), domain=handoffs[0].get('domain', ''),
        status=status, candidates=candidates, n_runs=len(usable), total_runs=total,
        inconclusive_share=(total - len(usable)) / total, prediction=prediction,
        ranked_probabilities=[{'run': i, 'decision': h.get('decision'), 'candidates': h.get('candidates', [])}
                              for i, h in enumerate(handoffs)],
        fa_decision=decision, confidence_kind='uncalibrated_model_estimate')
