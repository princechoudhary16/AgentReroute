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


def is_full_agentrx(config: dict) -> bool:
    """Default pooling filter: variants that use both violations and the checklist."""
    return bool(config.get('evidence')) and bool(config.get('checklist')) \
        and config.get('protocol', 'joint') == 'joint'


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
