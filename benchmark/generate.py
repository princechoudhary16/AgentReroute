"""Episode generation: clean runs, eligibility, paired fault injection, causation check, labels.

For every task:

1. Run the clean workflow once per seed. The task is *eligible* only if every
   clean run succeeds (pass^k = 1), so later failures are attributable to the
   injected fault rather than to the model's baseline weakness. Clean runs that
   fail are natural failures and go to an annotation queue.
2. For every operator variant and timing, pick a site from the clean run of the
   primary seed and rerun with the fault (same seed, same cache).
3. Keep the pair only if the fault fired and was activated. Status is
   ``failed`` when the injected run fails the outcome check, ``tolerated`` when
   it still succeeds.
4. Failed episodes get labels: ``fault_step``, ``critical_step``,
   ``responsible_agent``, ``mode``, ``acceptable_steps``.

Everything written here is gold-side. ``corpus.build`` derives the public
traces from it.
"""
import sys
from pathlib import Path

from .env.llm import DEFAULT_NUM_CTX, CachedLLM, GeminiActor, OllamaClient
from .env.retail import gold_state_hash
from .env.scripted import scripted_llm
from .env.tasks import load_domain
from .env.workflow import AGENTS, WORKFLOW_VERSION, Workflow, outcome, system_prompt
from .faults.operators import (OPERATORS, OPERATORS_VERSION, TIMINGS, build_operator, candidate_sites,
                               pick_site, variants)
from .io import append_jsonl, digest, read_json, read_jsonl, write_json

DEFAULTS = {
    'name': 'generation', 'data_dir': None, 'backend': 'ollama', 'model': 'qwen2.5:7b',
    'host': 'http://localhost:11434', 'temperature': 0.0, 'seeds': [0, 1, 2], 'operators': None,
    'timings': list(TIMINGS), 'task_ids': None, 'max_tasks': None, 'max_steps': 60, 'cache_dir': None,
    'actor_response_schema': True, 'num_ctx': DEFAULT_NUM_CTX, 'task_slice': None,
}


def validate_config(config):
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'unknown config keys: {sorted(unknown)}')
    config = {**DEFAULTS, **config}
    if config['backend'] not in ('ollama', 'scripted', 'gemini'):
        raise ValueError('backend must be ollama, scripted or gemini')
    if not config['seeds'] or len(set(config['seeds'])) != len(config['seeds']):
        raise ValueError('seeds must be a non-empty list of distinct ints')
    if config['operators'] is not None and set(config['operators']) - set(OPERATORS):
        raise ValueError(f'unknown operators: {sorted(set(config["operators"]) - set(OPERATORS))}')
    if set(config['timings']) - set(TIMINGS):
        raise ValueError(f'timings must be from {TIMINGS}')
    if type(config['actor_response_schema']) is not bool:
        raise ValueError('actor_response_schema must be a boolean')
    if type(config['num_ctx']) is not int or config['num_ctx'] < 2048:
        raise ValueError('num_ctx must be an integer >= 2048')
    if config['task_slice'] is not None:
        parts = str(config['task_slice']).split('/')
        if len(parts) != 2 or not all(p.isdigit() for p in parts):
            raise ValueError('task_slice must look like K/N, e.g. 0/4')
        index, count = int(parts[0]), int(parts[1])
        if count < 1 or not 0 <= index < count:
            raise ValueError(f'task_slice {config["task_slice"]} needs 0 <= K < N and N >= 1')
    return config


class LLMFactory:
    """Builds the agent/user LLM per task: a shared cached Ollama or Gemini client, or the scripted fixture."""

    def __init__(self, config, out_dir):
        self.config = config
        self.shared = None
        cache = config['cache_dir'] or str(Path(out_dir) / 'llm_cache')
        if config['backend'] == 'ollama':
            self.shared = CachedLLM(OllamaClient(config['host'], num_ctx=config['num_ctx'],
                                                 use_response_schema=config['actor_response_schema']), cache)
        elif config['backend'] == 'gemini':
            actor = GeminiActor(config['model'], settings={'temperature': config['temperature']},
                                lenient_log=str(Path(out_dir) / 'lenient.jsonl'),
                                use_response_schema=config['actor_response_schema'])
            self.shared = CachedLLM(actor, cache)

    @property
    def measurement_kind(self):
        return 'offline_fixture' if self.config['backend'] == 'scripted' else 'model'

    def model_digest(self, model):
        """Weights digest, for the settings lock, so a corpus is pinned to the exact model that made it.
        Scripted runs have no weights."""
        return self.shared.inner.model_digest(model) if self.shared is not None else 'scripted'

    def __call__(self, domain, task):
        return self.shared if self.shared is not None else scripted_llm(domain, task)


def episode_id(task_id, seed, op_spec=None, timing=None):
    if op_spec is None:
        return f'{task_id}.clean.s{seed}'
    params = ','.join(f'{k}={v}' for k, v in sorted(op_spec['params'].items()))
    return f'{task_id}.{op_spec["name"]}[{params}].{timing}.s{seed}'


def trajectory_id(eid):
    return 'tr-' + digest(eid)[:16]


def _run(domain, task, factory, config, seed, faults=()):
    llm = factory(domain, task)
    return Workflow(domain, task, llm, config['model'], seed=seed, temperature=config['temperature'],
                    faults=faults, max_steps=config['max_steps']).run()


def divergence_step(clean, injected):
    for a, b in zip(clean.steps, injected.steps):
        if (a.actor, a.content) != (b.actor, b.content):
            return a.index
    return None if len(clean.steps) == len(injected.steps) else min(len(clean.steps), len(injected.steps)) + 1


def _episode(eid, task, config, factory, seed, run, result, status, op=None, timing=None, site=None, label=None,
             activation=None):
    return {'episode_id': eid, 'trajectory_id': trajectory_id(eid), 'task_id': task['task_id'],
            'domain': 'retail', 'seed': seed, 'model': config['model'], 'backend': config['backend'],
            'measurement_kind': factory.measurement_kind, 'temperature': config['temperature'],
            'max_steps': config['max_steps'], 'status': status, 'outcome': result,
            'operator': op.spec() if op else None, 'timing': timing, 'site': site, 'label': label,
            'activation': activation, 'lenient_responses': run.lenient_responses, 'run': run.to_dict()}


def generate(config, out_dir, log=sys.stderr):
    config = validate_config(config)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    domain = load_domain(config['data_dir'])
    tasks = domain.tasks
    if config['task_ids']:
        tasks = [domain.task(t) for t in config['task_ids']]
    if config['max_tasks']:
        tasks = tasks[:config['max_tasks']]
    if config['task_slice']:
        # Disjoint slices of the SAME task list, so N processes can generate side by side into N output
        # directories and corpus.build() merges them (it already takes several generation dirs). This is
        # the whole of the parallelism: no shared state, no locks, and each process stays independently
        # resumable from its own index.jsonl. task_slice is deliberately absent from `identity`, because
        # every slice must record identical generation settings for the merge to be accepted.
        index, count = (int(p) for p in str(config['task_slice']).split('/'))
        tasks = tasks[index::count]
        print(f'task slice {index}/{count}: {len(tasks)} task(s)', file=log, flush=True)
    factory = LLMFactory(config, out_dir)
    identity = {k: config[k] for k in ('backend', 'model', 'temperature', 'seeds', 'max_steps')}
    identity['domain_source'] = domain.source
    identity['workflow_version'] = WORKFLOW_VERSION
    identity['operators_version'] = OPERATORS_VERSION
    identity['prompts_digest'] = digest({a: system_prompt(a, domain.policy) for a in AGENTS})
    identity['actor_response_schema'] = config['actor_response_schema']   # changes the draws the actor makes
    identity['num_ctx'] = config['num_ctx']                               # ollama backend; what the actor could see
    identity['model_digest'] = factory.model_digest(config['model'])
    meta_path = out_dir / 'generation.json'
    if meta_path.exists() and read_json(meta_path)['identity'] != identity:
        raise ValueError(f'{out_dir} was generated with different settings; use a new output directory')
    write_json(meta_path, {'config': config, 'identity': identity, 'policy_digest': digest(domain.policy)})
    done = {row['episode_id'] for row in read_jsonl(out_dir / 'index.jsonl')}
    grid = variants(config['operators'])
    counts = {}

    def save(episode):
        write_json(out_dir / 'episodes' / f'{episode["episode_id"]}.json', episode)
        append_jsonl(out_dir / 'index.jsonl', {k: episode[k] for k in ('episode_id', 'trajectory_id', 'task_id',
                                                                         'status', 'timing', 'seed')}
                     | {'operator': episode['operator']['name'] if episode['operator'] else None,
                        'mode': episode['label']['mode'] if episode['label'] else None})
        counts[episode['status']] = counts.get(episode['status'], 0) + 1

    for n, task in enumerate(tasks, 1):
        gold_hash = gold_state_hash(domain.data, task['actions'])
        print(f'[{n}/{len(tasks)}] {task["task_id"]}: clean runs', file=log, flush=True)
        clean = {}
        for seed in config['seeds']:
            run = _run(domain, task, factory, config, seed)
            clean[seed] = (run, outcome(task, run, gold_hash))
            eid = episode_id(task['task_id'], seed)
            if eid not in done:
                # An unrecovered actor reply leaves an empty action in the trace, indistinguishable from a
                # real agent error, so any label on this run would be a guess. It is neither 'clean' nor a
                # 'natural_failure' (the model did not fail; our decoding did), so it gets its own status
                # and corpus.build drops it.
                status = ('degraded' if run.lenient_responses else
                          ('clean' if clean[seed][1]['reward'] == 1 else 'natural_failure'))
                save(_episode(eid, task, config, factory, seed, run, clean[seed][1], status))
        if any(r.lenient_responses for r, _ in clean.values()):
            # Skipping the task here also avoids paying for ~20 injected runs replaying a poisoned prefix.
            print('  not eligible (unrecovered actor replies in a clean run)', file=log, flush=True)
            continue
        if any(r['reward'] != 1 for _, r in clean.values()):
            print(f'  not eligible (clean pass^{len(clean)} < 1)', file=log, flush=True)
            continue
        seed = config['seeds'][0]
        clean_run = clean[seed][0]
        for name, params in grid:
            sites = candidate_sites(name, params, clean_run)
            used = set()
            for timing in config['timings']:
                site = pick_site(sites, timing)
                if site is None or site['ordinal'] in used:
                    continue
                used.add(site['ordinal'])
                spec = {'name': name, 'ordinal': site['ordinal'], 'params': params}
                if 'target_agent' in site:
                    spec['target_agent'] = site['target_agent']
                op = build_operator(spec)
                eid = episode_id(task['task_id'], seed, op.spec(), timing)
                if eid in done:
                    continue
                run = _run(domain, task, factory, config, seed, faults=[op])
                result = outcome(task, run, gold_hash)
                diverge = divergence_step(clean_run, run)
                fired = run.faults_fired
                activation = {'fired': bool(fired), 'divergence_step': diverge,
                              'site_step_clean': site['step_index']}
                if run.lenient_responses:
                    # The outcome check can fail from an unrecovered reply alone, so labelling this run
                    # would attribute our decoding failure to the injected operator.
                    status, label = 'degraded', None
                elif not fired or diverge is None:
                    status, label = 'not_activated', None
                elif result['reward'] == 1:
                    status, label = 'tolerated', None
                else:
                    status = 'failed'
                    label = {**op.label(run, fired[0]), 'mode': op.mode}
                    label['acceptable_steps'] = sorted(set(label['acceptable_steps'])
                                                       | {label['fault_step'], label['critical_step']})
                save(_episode(eid, task, config, factory, seed, run, result, status, op, timing, site, label,
                              activation))
        print(f'  done; totals so far {counts}', file=log, flush=True)
    summary = {'episodes_written': counts, 'tasks': len(tasks)}
    if factory.shared is not None:
        summary['cache'] = factory.shared.stats()
        actor_stats = getattr(factory.shared.inner, 'stats', None)
        if actor_stats is not None:
            summary['actor'] = actor_stats()
    write_json(out_dir / 'summary.json', summary)
    return summary


def load_episodes(gen_dir):
    gen_dir = Path(gen_dir)
    return [read_json(gen_dir / 'episodes' / f'{row["episode_id"]}.json') for row in read_jsonl(gen_dir / 'index.jsonl')]
