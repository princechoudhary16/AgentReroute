"""Corpus build: splits, public traces (two views), hidden gold, and a faultattribution export.

Layout of a built corpus::

    corpus/
      manifest.json                      sources, counts, hashes, split policy
      public/<view>/<split>/<tid>.json   observable traces (view = full | output_only)
      public/<split>.json                trajectory ids per split (no labels)
      gold/<split>.json                  labels (scoring side only)
      gold/episodes.json                 tid -> generation episode (recovery harness only)
      agentrx/<split>/cases.json         faultattribution-compatible corpus (+ manifest.json)

Splits are disjoint by task template. Held-out fault variants appear only in
``test``; episodes using them on other splits are dropped (and counted).
"""
from pathlib import Path

from . import BENCHMARK_NAME, __version__
from .env.tasks import load_domain
from .env.workflow import AGENTS, Run, Step, export_trace
from .env.retail import TOOL_SCHEMAS
from .io import digest, read_json, write_json
from .schema import validate_gold, validate_trace
from .taxonomy import to_agentrx

SPLITS = ('train', 'calibration', 'test')
DEFAULT_RATIOS = {'train': 0.5, 'calibration': 0.2, 'test': 0.3}
DEFAULT_HELDOUT = ('tool_error[cell=explicit_persistent]', 'stale_output[cell=silent_persistent]',
                   'drop_context[variant=truncate]', 'misread_value[target=claim]',
                   'policy_contradiction[directive=skip_verification]')
INCLUDED_STATUSES = ('failed', 'tolerated', 'clean')


def variant_key(operator):
    params = ','.join(f'{k}={v}' for k, v in sorted(operator['params'].items()))
    return f'{operator["name"]}[{params}]' if params else operator['name']


def assign_split(task_id, ratios=None, salt=BENCHMARK_NAME):
    """Deterministic split by task template (stable across corpus rebuilds)."""
    ratios = ratios or DEFAULT_RATIOS
    point = int(digest(f'{salt}:{task_id}')[:8], 16) / 0xFFFFFFFF
    edge = 0.0
    for split in SPLITS:
        edge += ratios[split]
        if point <= edge:
            return split
    return SPLITS[-1]


def run_from_dict(data):
    steps = [Step(**s) for s in data['steps']]
    return Run(**{k: v for k, v in data.items() if k != 'steps'}, steps=steps)


def public_instruction(run):
    return next((s.content for s in run.steps if s.kind == 'user'), '')


def gold_record(episode, split):
    status = episode['status']
    label = episode['label']
    record = {'trajectory_id': episode['trajectory_id'], 'status': status, 'split': split,
              'source': 'injected' if episode['operator'] else ('natural' if status == 'failed' else 'clean_run'),
              'domain': episode['domain'], 'task_template': episode['task_id'],
              'measurement_kind': episode['measurement_kind'], 'critical': None}
    if status == 'failed':
        record['critical'] = {'step_index': label['critical_step'], 'agent': label['responsible_agent'],
                              'mode': label['mode']}
        record['fault_step'] = label.get('fault_step')
        record['acceptable_steps'] = label.get('acceptable_steps', [label['critical_step']])
    if episode['operator']:
        op = episode['operator']
        record['injection'] = {'operator': op['name'], 'mode': op['mode'], 'method': op['method'],
                               'level': op['level'], 'params': op['params'], 'timing': episode['timing'],
                               'variant': variant_key(op)}
    return record


def _agentrx_case(trace, gold, policy):
    critical = gold['critical']
    failure = {'failure_id': '1', 'step_index': critical['step_index'],
               'category': to_agentrx(critical['mode']), 'agent': critical['agent']}
    tools = {a: [TOOL_SCHEMAS[t] for t in spec['tools']] for a, spec in AGENTS.items()}
    return {'trajectory': {k: trace[k] for k in ('trajectory_id', 'instruction', 'instruction_source', 'steps')}
            | {'domain': 'tau'},
            'gold': {'trajectory_id': trace['trajectory_id'], 'failures': [failure], 'critical': failure},
            'policy': policy, 'tools': tools}


def build(gen_dirs, out_dir, ratios=None, heldout_variants=DEFAULT_HELDOUT, natural_labels=None):
    """Build a corpus from one or more generation directories."""
    from .generate import load_episodes
    out_dir = Path(out_dir)
    ratios = ratios or DEFAULT_RATIOS
    if abs(sum(ratios.values()) - 1) > 1e-9 or set(ratios) != set(SPLITS):
        raise ValueError(f'ratios must cover {SPLITS} and sum to 1')
    natural = {}
    if natural_labels:
        for row in read_json(natural_labels):
            natural[row['trajectory_id']] = row
    gold = {s: [] for s in SPLITS}
    public_ids = {s: [] for s in SPLITS}
    episodes_index, sources, dropped = {}, [], {'heldout_variant_outside_test': 0, 'unlabeled_natural': 0,
                                                 'not_activated': 0}
    policies = {}
    for gen_dir in gen_dirs:
        meta = read_json(Path(gen_dir) / 'generation.json')
        domain = load_domain(meta['config']['data_dir'])
        policies[domain.name] = domain.policy
        sources.append({'generation_dir': str(gen_dir), 'identity': meta['identity']})
        for episode in load_episodes(gen_dir):
            status = episode['status']
            if status == 'not_activated':
                dropped['not_activated'] += 1
                continue
            if status == 'natural_failure':
                label = natural.get(episode['trajectory_id'])
                if label is None:
                    dropped['unlabeled_natural'] += 1
                    continue
                episode = {**episode, 'status': 'failed', 'label': {
                    'critical_step': label['critical']['step_index'], 'responsible_agent': label['critical']['agent'],
                    'mode': label['critical']['mode'], 'fault_step': label.get('fault_step'),
                    'acceptable_steps': label.get('acceptable_steps', [label['critical']['step_index']])}}
            split = assign_split(episode['task_id'], ratios)
            if episode['operator'] and variant_key(episode['operator']) in heldout_variants and split != 'test':
                dropped['heldout_variant_outside_test'] += 1
                continue
            run = run_from_dict(episode['run'])
            tid = episode['trajectory_id']
            if tid in episodes_index:
                raise ValueError(f'duplicate trajectory {tid}')
            for view in ('full', 'output_only'):
                trace = export_trace(run, tid, episode['domain'], public_instruction(run), view)
                validate_trace(trace)
                write_json(out_dir / 'public' / view / split / f'{tid}.json', trace)
            record = gold_record(episode, split)
            validate_gold(record, trace)
            gold[split].append(record)
            public_ids[split].append(tid)
            episodes_index[tid] = {'generation_dir': str(Path(gen_dir).resolve()),
                                   'episode_id': episode['episode_id'], 'split': split}
    agentrx = {}
    for split in SPLITS:
        gold[split].sort(key=lambda g: g['trajectory_id'])
        write_json(out_dir / 'gold' / f'{split}.json', gold[split])
        write_json(out_dir / 'public' / f'{split}.json', sorted(public_ids[split]))
        cases = []
        for g in gold[split]:
            if g['status'] == 'failed':
                trace = read_json(out_dir / 'public' / 'output_only' / split / f'{g["trajectory_id"]}.json')
                cases.append(_agentrx_case(trace, g, policies.get(g['domain'], '')))
        write_json(out_dir / 'agentrx' / split / 'cases.json', cases)
        write_json(out_dir / 'agentrx' / split / 'manifest.json', {
            'format_version': 1, 'corpus_hash': digest(cases), 'available_total': len(cases),
            'source_url': BENCHMARK_NAME, 'measurement_scope': f'{BENCHMARK_NAME}:{split}',
            'note': "Exported for the faultattribution branch. Native domain is the multi-agent retail workflow, "
                    "labelled 'tau' because that loader accepts only AgentRx domains. Categories are mapped from "
                    "the seven modes with taxonomy.to_agentrx and are lossy."})
        agentrx[split] = len(cases)
    write_json(out_dir / 'gold' / 'episodes.json', episodes_index)
    counts = {}
    for split in SPLITS:
        by = {}
        for g in gold[split]:
            key = g['critical']['mode'] if g['critical'] else g['status']
            by[key] = by.get(key, 0) + 1
        counts[split] = {'total': len(gold[split]), 'by_label': dict(sorted(by.items()))}
    manifest = {'benchmark': BENCHMARK_NAME, 'version': __version__, 'splits': counts, 'ratios': ratios,
                'split_policy': 'disjoint by task template; held-out variants only in test',
                'heldout_variants': list(heldout_variants), 'dropped': dropped, 'sources': sources,
                'agentrx_export': agentrx,
                'gold_hash': digest(gold), 'views': ['full', 'output_only']}
    write_json(out_dir / 'manifest.json', manifest)
    return manifest


# ---------------------------------------------------------------------------- loading
def load_gold(corpus_dir, split):
    return {g['trajectory_id']: g for g in read_json(Path(corpus_dir) / 'gold' / f'{split}.json')}


def load_public(corpus_dir, split, view='output_only'):
    corpus_dir = Path(corpus_dir)
    ids = read_json(corpus_dir / 'public' / f'{split}.json')
    return {tid: read_json(corpus_dir / 'public' / view / split / f'{tid}.json') for tid in ids}


def natural_queue(gen_dirs, out_dir):
    """Write natural failures (clean runs that failed) for manual annotation, plus a label template."""
    from .generate import load_episodes
    out_dir = Path(out_dir)
    template = []
    for gen_dir in gen_dirs:
        for episode in load_episodes(gen_dir):
            if episode['status'] != 'natural_failure':
                continue
            run = run_from_dict(episode['run'])
            tid = episode['trajectory_id']
            write_json(out_dir / 'traces' / f'{tid}.json',
                       export_trace(run, tid, episode['domain'], public_instruction(run), 'full'))
            template.append({'trajectory_id': tid, 'task_template': episode['task_id'],
                             'outcome': episode['outcome'], 'annotator': '',
                             'critical': {'step_index': None, 'agent': '', 'mode': ''},
                             'fault_step': None, 'acceptable_steps': [], 'notes': ''})
    write_json(out_dir / 'labels_template.json', template)
    return len(template)
