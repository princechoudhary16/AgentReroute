"""Retail tasks and data: bundled fixture, or tau-bench retail fetched at a pinned revision.

A task is ``{task_id, user_id, instruction, actions: [{name, kwargs}], outputs}``.
``actions`` and ``outputs`` are gold-side: only the outcome check and the
generator read them, never the agents.
"""
import ast
import json
import urllib.request
from importlib.resources import files
from pathlib import Path

from ..io import digest, read_json, write_json

TAU_REPO = 'sierra-research/tau-bench'
TAU_DEFAULT_REF = 'main'
TAU_FILES = {
    'users.json': 'tau_bench/envs/retail/data/users.json',
    'orders.json': 'tau_bench/envs/retail/data/orders.json',
    'products.json': 'tau_bench/envs/retail/data/products.json',
    'policy.md': 'tau_bench/envs/retail/wiki.md',
    'tasks_test.py': 'tau_bench/envs/retail/tasks_test.py',
}


class Domain:
    """Data, policy and tasks for one domain source."""

    def __init__(self, name, data, policy, tasks, source):
        self.name, self.data, self.policy, self.tasks, self.source = name, data, policy, tasks, source

    def task(self, task_id):
        for task in self.tasks:
            if task['task_id'] == task_id:
                return task
        raise KeyError(f'unknown task {task_id}')


def _load_dir(directory, name, source):
    directory = Path(directory)
    data = {k: read_json(directory / f'{k}.json') for k in ('users', 'orders', 'products')}
    policy = (directory / 'policy.md').read_text(encoding='utf-8')
    if (directory / 'tasks.json').exists():
        tasks = read_json(directory / 'tasks.json')
    else:
        tasks = parse_tau_tasks((directory / 'tasks_test.py').read_text(encoding='utf-8'))
    for task in tasks:
        task.setdefault('outputs', [])
    return Domain(name, data, policy, tasks, source)


def load_fixture():
    root = files('benchmark').joinpath('resources/fixture_retail')
    return _load_dir(Path(str(root)), 'retail', {'kind': 'fixture', 'note': 'hand-written, tau-bench format'})


def load_domain(data_dir=None):
    """Fixture when ``data_dir`` is None, otherwise a fetched tau-bench retail directory."""
    if data_dir is None:
        return load_fixture()
    manifest = Path(data_dir) / 'manifest.json'
    source = read_json(manifest) if manifest.exists() else {'kind': 'directory', 'path': str(data_dir)}
    return _load_dir(data_dir, 'retail', source)


# --------------------------------------------------------------------- tau-bench task parsing
def _literal(node):
    if isinstance(node, ast.Call):
        name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, 'attr', '')
        if name not in ('Task', 'Action'):
            raise ValueError(f'unexpected call {name} in tasks file')
        return {kw.arg: _literal(kw.value) for kw in node.keywords}
    if isinstance(node, ast.List):
        return [_literal(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        return {_literal(k): _literal(v) for k, v in zip(node.keys, node.values)}
    return ast.literal_eval(node)


def parse_tau_tasks(source):
    """Parse tau-bench's ``tasks_test.py`` without executing it."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.List) and node.value.elts \
                and isinstance(node.value.elts[0], ast.Call):
            tasks = []
            for i, element in enumerate(node.value.elts):
                raw = _literal(element)
                tasks.append({'task_id': f'retail-{i:03d}', 'user_id': raw['user_id'],
                              'instruction': raw['instruction'],
                              'actions': [{'name': a['name'], 'kwargs': a['kwargs']} for a in raw['actions']],
                              'outputs': raw.get('outputs', []) or []})
            return tasks
    raise ValueError('no task list found in tasks file')


def fetch_tau_retail(out_dir, ref=TAU_DEFAULT_REF, opener=urllib.request.urlopen):
    """Download tau-bench retail data and tasks; records the resolved commit and file hashes."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = ref
    try:
        with opener(f'https://api.github.com/repos/{TAU_REPO}/commits/{ref}', timeout=30) as response:
            commit = json.loads(response.read().decode('utf-8'))['sha']
    except Exception:  # noqa: BLE001 - an unresolved ref is recorded as given
        pass
    hashes = {}
    for local, remote in TAU_FILES.items():
        with opener(f'https://raw.githubusercontent.com/{TAU_REPO}/{commit}/{remote}', timeout=120) as response:
            body = response.read()
        (out_dir / local).write_bytes(body)
        hashes[local] = digest(body)
    tasks = parse_tau_tasks((out_dir / 'tasks_test.py').read_text(encoding='utf-8'))
    manifest = {'kind': 'tau-bench', 'repo': TAU_REPO, 'ref': ref, 'commit': commit, 'files': hashes,
                'n_tasks': len(tasks), 'license': 'MIT (tau-bench)'}
    write_json(out_dir / 'manifest.json', manifest)
    return manifest
