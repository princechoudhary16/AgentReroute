"""Import external labelled corpora into a built corpus as extra, attribution-only splits.

* AgentRx release: the ``cases.json`` written by the faultattribution branch
  (``python -m agentrx prepare``). Categories map to the seven modes with
  ``taxonomy.from_agentrx``; the fit (exact / approximate / none) is kept.
* Who&When: the released JSON logs (``history``, ``mistake_agent``,
  ``mistake_step``). Who&When has no failure category, so ``mode`` is null and
  those traces count only for agent and step metrics.

External traces are not replayable, so they never enter recovery evaluation.
"""
import json
from pathlib import Path

from .io import digest, read_json, write_json
from .schema import validate_gold, validate_trace
from .taxonomy import from_agentrx


def _write_split(corpus_dir, split, traces, golds, source):
    corpus_dir = Path(corpus_dir)
    for trace in traces:
        validate_trace(trace)
        for view in ('full', 'output_only'):
            write_json(corpus_dir / 'public' / view / split / f'{trace["trajectory_id"]}.json', {**trace, 'view': view})
    for gold, trace in zip(golds, traces):
        validate_gold(gold, trace)
    write_json(corpus_dir / 'public' / f'{split}.json', sorted(t['trajectory_id'] for t in traces))
    write_json(corpus_dir / 'gold' / f'{split}.json', sorted(golds, key=lambda g: g['trajectory_id']))
    manifest_path = corpus_dir / 'manifest.json'
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    manifest.setdefault('external', {})[split] = {'source': source, 'total': len(golds), 'gold_hash': digest(golds),
                                                  'replayable': False}
    write_json(manifest_path, manifest)
    return len(golds)


def import_agentrx(cases_path, corpus_dir, split='external_agentrx'):
    cases = read_json(cases_path)
    traces, golds = [], []
    for case in cases:
        trajectory = case['trajectory']
        domain = trajectory['domain']
        tid = f'agentrx-{domain}-{trajectory["trajectory_id"]}'
        trace = {'trajectory_id': tid, 'domain': f'agentrx_{domain}', 'instruction': trajectory.get('instruction', ''),
                 'instruction_source': trajectory.get('instruction_source', ''),
                 'steps': [{'index': s['index'], 'substeps': s['substeps']} for s in trajectory['steps']]}
        critical = case['gold']['critical']
        mode, fit = from_agentrx(critical['category'])
        gold = {'trajectory_id': tid, 'status': 'failed', 'split': split, 'source': 'agentrx_release',
                'domain': trace['domain'], 'task_template': str(trajectory['trajectory_id']),
                'measurement_kind': 'human_annotation',
                'critical': {'step_index': critical['step_index'], 'agent': critical['agent'],
                             'mode': None if mode == 'inconclusive' else mode},
                'acceptable_steps': [critical['step_index']],
                'label_mapping': {'from': critical['category'], 'fit': fit},
                'all_failures': [{'step_index': f['step_index'], 'agent': f['agent'],
                                  'mode': from_agentrx(f['category'])[0]} for f in case['gold']['failures']]}
        traces.append(trace)
        golds.append(gold)
    return _write_split(corpus_dir, split, traces, golds, {'kind': 'agentrx_release', 'path': str(cases_path)})


def _whowhen_trace(raw, tid):
    history = raw.get('history', [])
    steps = []
    for i, event in enumerate(history):
        role = str(event.get('name') or event.get('role') or 'unknown')
        content = event.get('content', '')
        steps.append({'index': i, 'substeps': [{'sub_index': 1, 'role': role, 'content': content if isinstance(
            content, str) else json.dumps(content, ensure_ascii=False)}]})
    return {'trajectory_id': tid, 'domain': 'whowhen', 'instruction': str(raw.get('question', '')),
            'instruction_source': 'question', 'steps': steps}


def import_whowhen(directory, corpus_dir, split='external_whowhen'):
    traces, golds = [], []
    for path in sorted(Path(directory).rglob('*.json')):
        raw = read_json(path)
        if not isinstance(raw, dict) or 'history' not in raw or 'mistake_step' not in raw:
            continue
        tid = f'whowhen-{path.parent.name}-{path.stem}'
        trace = _whowhen_trace(raw, tid)
        if not trace['steps']:
            continue
        step = int(raw['mistake_step'])
        gold = {'trajectory_id': tid, 'status': 'failed', 'split': split, 'source': 'whowhen',
                'domain': 'whowhen', 'task_template': path.stem, 'measurement_kind': 'human_annotation',
                'critical': {'step_index': step, 'agent': str(raw.get('mistake_agent', '')), 'mode': None},
                'acceptable_steps': [step], 'reason': str(raw.get('mistake_reason', ''))}
        traces.append(trace)
        golds.append(gold)
    return _write_split(corpus_dir, split, traces, golds, {'kind': 'whowhen', 'path': str(directory)})
