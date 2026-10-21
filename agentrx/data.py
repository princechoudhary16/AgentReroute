"""Allowlisted observable IR and independently stored benchmark annotations."""
import hashlib
import json
import subprocess
from importlib.resources import files
from pathlib import Path
from .io import digest, read_json, write_json
from .taxonomy import category

REFERENCE_URL = 'https://github.com/microsoft/AgentRx.git'
REFERENCE_REVISION = '7a18c79708e7671be15124460f4f7296107c2a55'
EXPECTED = {'tau': 39, 'magentic': 44, 'flash': 42, 'relwork': 45}


def _index(value):
    if type(value) is not int or value < 0:
        raise ValueError(f'Invalid step index: {value!r}')
    return value


def _event(raw, sub_index=1):
    content = raw.get('content', raw.get('message', ''))
    event = {'sub_index': _index(raw.get('sub_index', sub_index)),
             'role': str(raw.get('role', raw.get('agent_name', 'unknown'))),
             'content': content if isinstance(content, str) else
                        ('' if content is None else json.dumps(content, ensure_ascii=False))}
    for field in ('name', 'tool_call_id'):
        if field in raw:
            event[field] = str(raw[field])
    calls = raw.get('tool_calls', [])
    if calls:
        event['tool_calls'] = []
        for call in calls:
            fn = call.get('function', call)
            args = fn.get('arguments', {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass  # Malformed arguments are evidence, not a loader failure.
            event['tool_calls'].append({'id': str(call.get('id', '')),
                                         'name': str(fn.get('name', '')),
                                         'arguments': args})
    return event


def normalize(raw, domain, trajectory_id=None):
    if domain not in EXPECTED:
        raise ValueError(f'Unsupported domain {domain}')
    if isinstance(raw, dict) and 'steps' in raw:
        tid = str(trajectory_id if trajectory_id is not None else raw.get('trajectory_id', 'unknown'))
        instruction = str(raw.get('instruction', ''))
        instruction_source = str(raw.get('instruction_source', 'provided_ir'))
        steps = [{'index': _index(s['index']),
                  'substeps': [_event(e, i + 1) for i, e in enumerate(s['substeps'])]}
                 for s in raw['steps']]
    else:
        if isinstance(raw, dict):
            tid = str(trajectory_id if trajectory_id is not None else raw.get('trajectory_id', raw.get('task_id', 'unknown')))
            events = raw.get('traj', raw.get('events', raw.get('messages', [])))
            instruction = raw.get('instruction', '')
            instruction_source = 'instruction'
            if domain == 'tau':
                instruction = raw.get('info', {}).get('task', {}).get('instruction', instruction)
                instruction_source = 'info.task.instruction'
        elif isinstance(raw, list):
            tid, events, instruction, instruction_source = str(trajectory_id or 'unknown'), raw, '', 'user_message'
        else:
            raise ValueError('Trace must be an object or event list')
        if not isinstance(events, list) or not events:
            raise ValueError('Trace has no events')
        if not instruction:
            instruction = next((e.get('content', e.get('message', '')) for e in events
                                if str(e.get('role', '')).lower() in ('human', 'user')), '')
            instruction_source = 'user_message'
        instruction = str(instruction or '')
        steps = [{'index': _index(e.get('index', i + 1)), 'substeps': [_event(e)]}
                 for i, e in enumerate(events)]
    indices = [s['index'] for s in steps]
    if not steps or indices != sorted(set(indices)):
        raise ValueError('Step indices must be unique and strictly increasing')
    for s in steps:
        sub = [e['sub_index'] for e in s['substeps']]
        if not sub or len(sub) != len(set(sub)):
            raise ValueError('Each step requires uniquely indexed substeps')
    return {'trajectory_id': tid, 'domain': domain, 'instruction': instruction,
            'instruction_source': instruction_source, 'steps': steps,
            'index_map': [{'offset': i, 'step_index': s['index'], 'raw_position': i}
                          for i, s in enumerate(steps)]}


def normalize_gold(raw):
    failures = []
    seen = set()
    for f in raw['failures']:
        fid = str(f['failure_id'])
        if fid in seen:
            raise ValueError(f'Duplicate failure ID {fid}')
        seen.add(fid)
        failures.append({'failure_id': fid, 'step_index': _index(f.get('step_number', f.get('step_index'))),
                         'category': category(f.get('failure_category', f.get('category'))),
                         'agent': str(f.get('failed_agent', f.get('agent', '')))})
    root_id = str(raw.get('root_cause', {}).get('failure_id', raw.get('root_cause_failure_id')))
    roots = [f for f in failures if f['failure_id'] == root_id]
    if len(roots) != 1:
        raise ValueError(f'Root failure {root_id} does not identify exactly one event')
    return {'trajectory_id': str(raw['trajectory_id']), 'failures': failures, 'critical': roots[0]}


def validate_case(case):
    trajectory, gold = case['trajectory'], case['gold']
    if trajectory['trajectory_id'] != gold['trajectory_id']:
        raise ValueError('Trajectory and annotation IDs differ')
    indices = {s['index'] for s in trajectory['steps']}
    for failure in gold['failures']:
        if failure['step_index'] not in indices:
            raise ValueError(f"Gold step {failure['step_index']} absent from {trajectory['trajectory_id']}")


def prepare_corpus(source, output):
    source, output = Path(source).resolve(), Path(output)
    # Both a repository checkout and its data directory are accepted.
    base = source/'data' if (source/'data').is_dir() else source
    try:
        revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                                           stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = None
    cases, inventory, exclusions = [], {}, []

    def load(path):
        inventory[str(path.relative_to(base))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return read_json(path)

    def add(raw, label, domain, tid, policy='', tools=None):
        try:
            case = {'trajectory': normalize(raw, domain, tid), 'gold': normalize_gold(label),
                    'policy': policy, 'tools': tools or {}}
            validate_case(case)
            cases.append(case)
        except (ValueError, TypeError, KeyError) as exc:
            exclusions.append({'domain': domain, 'trajectory_id': str(tid), 'reason': str(exc)})

    policy_file = base/'policies/retail_policy.txt'
    policy = policy_file.read_text(encoding='utf-8') if policy_file.exists() else ''
    if policy_file.exists():
        inventory[str(policy_file.relative_to(base))] = hashlib.sha256(policy_file.read_bytes()).hexdigest()
    # The release's data/tool_schemas/tools_structure.json contains example
    # conversations, not schemas. Use the actual domain-registry definitions.
    tools = json.loads(files('agentrx').joinpath('resources/tools.json').read_text(encoding='utf-8'))
    for domain, label_name in [('tau', 'tau_ground_truth.json'), ('magentic', 'magentic_one_ground_truth.json')]:
        labels_path = base/'ground_truth'/label_name
        if not labels_path.exists():
            continue
        labels = load(labels_path)
        if domain == 'tau':
            path = base/'tau_retail/tau_dataset_failed.json'
            traces = {str(t['task_id']): t for t in load(path)} if path.exists() else {}
        else:
            traces = None
        seen = set()
        for label in labels:
            tid = str(label['trajectory_id'])
            if tid in seen:
                raise ValueError(f'Duplicate annotation trajectory ID {domain}/{tid}')
            seen.add(tid)
            if domain == 'tau':
                raw = traces.get(tid)
            else:
                path = base/'magentic_dataset'/f'{tid}.json'
                raw = load(path) if path.exists() else None
            if raw is None:
                exclusions.append({'domain': domain, 'trajectory_id': tid, 'reason': 'missing trace'})
                continue
            add(raw, label, domain, tid, policy if domain == 'tau' else '', tools.get(domain, {}))
    # Native missing-domain data can be supplied as explicitly normalized cases.
    for domain in ('flash', 'relwork'):
        path = base/f'{domain}_cases.json'
        if path.exists():
            for case in load(path):
                add(case['trajectory'], case['annotation'], domain,
                    case['annotation']['trajectory_id'], case.get('policy', ''), case.get('tools', {}))
    domains = {}
    for domain, expected in EXPECTED.items():
        ids = [c['trajectory']['trajectory_id'] for c in cases if c['trajectory']['domain'] == domain]
        domains[domain] = {'expected': expected, 'available': len(ids), 'trajectory_ids': ids,
                           'status': 'complete_count_unverified_membership' if len(ids) == expected else
                                     ('available_subset' if ids else 'blocked_missing_data')}
    manifest = {'format_version': 1, 'source_url': REFERENCE_URL, 'source_revision': revision,
                'reference_revision': REFERENCE_REVISION, 'source_pinned': revision == REFERENCE_REVISION,
                'tool_schema_source': 'packaged_reference_domain_registry',
                'tool_schema_hash': digest(tools),
                'files': inventory, 'domains': domains, 'exclusions': exclusions,
                'corpus_hash': digest(cases), 'available_total': len(cases), 'paper_total': 170,
                'measurement_scope': 'available_release_subset'}
    write_json(output/'cases.json', cases)
    write_json(output/'manifest.json', manifest)
    return manifest


def load_corpus(directory):
    directory = Path(directory)
    cases, manifest = read_json(directory/'cases.json'), read_json(directory/'manifest.json')
    if digest(cases) != manifest['corpus_hash']:
        raise ValueError('Corpus hash mismatch: re-prepare after changes')
    seen = set()
    for case in cases:
        validate_case(case)
        key = (case['trajectory']['domain'], case['trajectory']['trajectory_id'])
        if key in seen:
            raise ValueError(f'Duplicate corpus case {key}')
        seen.add(key)
    return cases, manifest
