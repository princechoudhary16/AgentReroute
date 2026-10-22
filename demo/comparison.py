"""Real local-model comparisons on shared controlled input, with inspectable execution events."""
import copy
import json
import threading
import time
import uuid
from pathlib import Path
from urllib.request import urlopen

from agentrx.io import digest, write_json
from agentrx.pipeline import diagnose, diagnose_mp_rl
from agentrx.providers import OllamaProvider
from agentrx.rl_policy import DualLoopRLJudge
from agentrx.rl_cross_branch import candidates_to_benchmark_hypotheses, _repairable
from benchmark.env.llm import Completion
from benchmark.env.retail import gold_state_hash
from benchmark.env.scripted import scripted_llm
from benchmark.env.tasks import load_fixture
from benchmark.env.workflow import Workflow, export_trace, outcome
from benchmark.scoring.baselines import heuristic_findings
from benchmark.scoring.recovery import BudgetCaps, RecoveryEnv, looks_ok
from benchmark.taxonomy import from_agentrx
from repair_strategies.adapters.benchmark import RepairStrategyPolicy
from .scenarios import MAX_STEPS, SCENARIOS, _task_view, step_frames

OLLAMA = 'http://127.0.0.1:11434'
SYSTEMS = ('agentrx', 'agentreroute')
TERMINAL = ('completed', 'error', 'canceled')


class ComparisonCancelled(RuntimeError):
    pass


def installed_models():
    with urlopen(OLLAMA + '/api/tags', timeout=5) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError('Ollama model catalog is too large')
    data = json.loads(raw)
    return sorted({m['name'] for m in data['models'] if isinstance(m.get('name'), str) and m['name']})


def local_provider(model):
    return OllamaProvider(model, OLLAMA, num_ctx=16384, settings={'temperature': 0, 'num_predict': 2048},
                          timeout=180, retries=0)


class MeasuredProvider:
    """Observe transport without altering prompts or model responses. No response cache is used."""
    def __init__(self, provider, emit, output, cancel, max_calls=64):
        self.provider, self.emit, self.output, self.cancel = provider, emit, Path(output), cancel
        self.measurement_kind, self.identity = provider.measurement_kind, provider.identity
        self.max_calls, self.calls = max_calls, []
        self.last_telemetry = {}

    def _call(self, stage, payload, messages=False):
        if self.cancel.is_set():
            raise ComparisonCancelled('Stopped after the current model request.')
        if len(self.calls) >= self.max_calls:
            raise RuntimeError(f'Model-call budget exhausted ({self.max_calls} requests per system).')
        row = {'id': len(self.calls) + 1, 'stage': stage, 'status': 'running'}
        self.calls.append(row)
        self.emit({'kind': 'request_started', 'request_id': row['id'], 'stage': stage,
                   'perspective': payload.get('perspective') if isinstance(payload, dict) else None,
                   'steps': len(payload.get('trajectory', {}).get('steps', [])) if isinstance(payload, dict) else None})
        started = time.monotonic()
        try:
            answer = self.provider.complete_messages(payload) if messages else self.provider.complete(stage, payload)
            self.last_telemetry = copy.deepcopy(getattr(self.provider, 'last_telemetry', {}))
            row.update(status='completed', seconds=round(time.monotonic() - started, 3), telemetry=self.last_telemetry)
            write_json(self.output/'requests'/f'{row["id"]:03d}.json',
                       {'stage': stage, 'input': payload, 'output': answer, 'telemetry': self.last_telemetry})
            text = json.dumps(answer, ensure_ascii=False)
            self.emit({'kind': 'request_completed', 'request_id': row['id'], 'stage': stage,
                       'seconds': row['seconds'], 'telemetry': self.last_telemetry,
                       'response': text[:16000], 'response_truncated': len(text) > 16000,
                       'metrics': self.metrics()})
        except Exception as exc:
            row.update(status='error', seconds=round(time.monotonic() - started, 3), error=f'{type(exc).__name__}: {exc}')
            write_json(self.output/'requests'/f'{row["id"]:03d}.json', {'stage': stage, 'input': payload, **row})
            self.emit({'kind': 'request_failed', **row, 'metrics': self.metrics()})
            raise
        if self.cancel.is_set():
            raise ComparisonCancelled('Stopped after the current model request.')
        return answer

    def complete(self, stage, payload):
        return self._call(stage, payload)

    def complete_messages(self, messages, *, stage='replay_agent'):
        return self._call(stage, messages, messages=True)

    def metrics(self):
        completed = [r for r in self.calls if r['status'] == 'completed']
        result = {'model_calls': len(self.calls), 'completed_calls': len(completed),
                  'failed_calls': sum(r['status'] == 'error' for r in self.calls),
                  'model_seconds': round(sum(r.get('seconds', 0) for r in self.calls), 3)}
        for field, key in (('prompt_tokens', 'prompt_tokens'), ('completion_tokens', 'completion_tokens')):
            values = [r['telemetry'].get('usage', {}).get(key) for r in completed]
            result[field] = sum(values) if values and all(type(v) is int for v in values) else None
        return result


class ScenarioInput:
    """A single recorded starting trace; each comparison arm creates an independent recovery environment."""
    def __init__(self, sid):
        self.sid, self.spec = sid, SCENARIOS[sid]
        self.domain = load_fixture()
        self.task = self.domain.task(self.spec['task_id'])
        self.tid = 'comparison-input'
        self.original = self.execute()
        self.trace = self.export(self.original)
        self.view = _task_view(self.trace, self.domain)
        self.tools = {s['name']: s for schemas in self.view['tools'].values() for s in schemas}
        self.input_hash = digest(self.trace)

    def export(self, run):
        first = next((s.content for s in run.steps if s.kind == 'user'), '')
        return export_trace(run, self.tid, self.domain.name, first, 'full')

    def execute(self, directive=None, provider=None):
        faults = [self.spec['operator'][0](**self.spec['operator'][1])] if self.spec['operator'] else []
        scripted = scripted_llm(self.domain, self.task)
        client = ReplayClient(scripted, provider, directive.rollback_step) if provider and directive else scripted
        return Workflow(self.domain, self.task, client, 'controlled-input' if provider is None else provider.identity['model'],
                        user_llm=scripted, faults=faults, directive=directive, max_steps=MAX_STEPS).run()

    def recover(self, hypotheses, provider, emit):
        runs = {}
        def runner(rid, directive):
            if provider.cancel.is_set():
                raise ComparisonCancelled('Comparison canceled.')
            emit({'kind': 'repair_started', 'rollout': rid, 'directive': directive.__dict__.copy()})
            run = self.execute(directive, provider)
            runs[rid] = run
            trace = self.export(run)
            emit({'kind': 'repair_replayed', 'rollout': rid, 'termination': run.termination,
                  'frames': step_frames(run, 'recovery_replay', start=directive.rollback_step)})
            return run, trace
        env = RecoveryEnv(self.tid, copy.deepcopy(self.trace), copy.deepcopy(self.view), runner,
                          BudgetCaps(tokens=100000, attempts=3, seconds=600), hypotheses=hypotheses)
        policy = RepairStrategyPolicy(max_hypotheses=3, learn=False)
        decision = policy.recover(env)
        attempts = policy.log[self.tid].attempts
        delivered = decision.get('delivered')
        selected = runs.get(delivered, self.original)
        # Evaluation only, after the policy returned. Neither gold actions nor scores enter model inputs.
        expected = gold_state_hash(self.domain.data, self.task.get('actions', []))
        scores = {'before': outcome(self.task, self.original, expected), 'after': outcome(self.task, selected, expected)}
        return decision, attempts, scores, env


class ReplayClient:
    """Keep the original prefix identical; post-rollback agent turns use actual native Ollama inference."""
    def __init__(self, scripted, provider, rollback):
        self.scripted, self.provider, self.rollback = scripted, provider, rollback

    def complete(self, request):
        meta = request.get('meta', {})
        if meta.get('agent') == 'user' or meta.get('step_index', 0) < self.rollback:
            return self.scripted.complete(request)
        answer = self.provider.complete_messages(request['messages'], stage='replay_agent/' + str(meta.get('agent', 'agent')))
        usage = self.provider.last_telemetry.get('usage', {})
        return Completion(json.dumps(answer), usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0),
                          model=self.provider.identity['model'])


def run_system(system, case, raw_provider, output, emit, cancel):
    output = Path(output)
    provider = MeasuredProvider(raw_provider, emit, output, cancel)
    started = time.monotonic()
    result = {'system': system, 'status': 'running', 'input_hash': case.input_hash,
              'policy_state': 'initial weights, no checkpoint' if system == 'agentreroute' else 'single judge',
              'provider': copy.deepcopy(provider.identity)}
    emit({'kind': 'system_started', **result})
    try:
        raw_provider.prepare_for_run()
        result['provider'] = copy.deepcopy(raw_provider.identity)
        judge = DualLoopRLJudge() if system == 'agentreroute' else None
        config = {'evidence': True, 'checklist': True, 'generation': 'oneshot',
                  'protocol': 'mp_rl' if judge else 'joint', 'recovery_loop': False}
        if judge:
            diagnosed = diagnose_mp_rl(copy.deepcopy(case.trace), case.view['policy'], case.tools, config,
                                      provider, output/'diagnosis', rl_judge=judge)
        else:
            diagnosed = diagnose(copy.deepcopy(case.trace), case.view['policy'], case.tools, config, provider, output/'diagnosis')
        if cancel.is_set():
            raise ComparisonCancelled('Comparison canceled.')
        if diagnosed['status'] == 'error':
            raise RuntimeError(diagnosed.get('error', 'Diagnosis failed'))
        if provider.metrics()['completed_calls'] == 0:
            raise RuntimeError('No model requests succeeded; refusing a seed-only live comparison.')
        result['diagnosis_seconds'] = round(time.monotonic() - started, 3)
        result['prediction'] = diagnosed['prediction']
        checks = diagnosed['checks']
        counts = {v: sum(c['verdict'] == v for c in checks) for v in ('SAT', 'VIOL', 'SKIP', 'ERROR')}
        result['checks'] = counts
        result['seed_checks'] = sum(c.get('origin') == 'seed' for c in checks)
        emit({'kind': 'checks_completed', 'counts': counts, 'seed_checks': result['seed_checks'],
              'rows': [{k: c.get(k) for k in ('step_index','assertion_name','perspective','verdict','check_hint','error','origin')}
                       for c in checks]})
        if judge:
            ranked = diagnosed['ranked_candidates']
            hypotheses = candidates_to_benchmark_hypotheses(ranked, case.trace)[:3]
            if not any(c.get('violated_checks') for c in ranked) and not heuristic_findings(case.trace):
                hypotheses = []
                judge.pending.pop(case.tid, None)
            result['ranked_candidates'] = ranked[:5]
        else:
            prediction = diagnosed['prediction']
            hypotheses = [] if prediction.get('step_index') is None else [{**prediction, 'mode': from_agentrx(prediction['category'])[0]}]
        usable = [_repairable(h) for h in hypotheses if h.get('mode') != 'out_of_scope']
        result['hypotheses'] = hypotheses
        emit({'kind': 'attribution_completed', 'prediction': result['prediction'], 'hypotheses': hypotheses,
              'ranked_candidates': result.get('ranked_candidates', []), 'diagnosis_seconds': result['diagnosis_seconds']})
        if cancel.is_set():
            raise ComparisonCancelled('Comparison canceled.')
        decision, attempts, scores, env = case.recover(usable, provider, emit)
        if cancel.is_set():
            raise ComparisonCancelled('Comparison canceled.')
        result.update(decision=decision, attempts=attempts, task_outcome=scores,
                      verification={'accepted': any(a.get('accepted') for a in attempts),
                                    'source': 'shared repair acceptance and actual retail-state scoring'})
        if judge and case.tid in judge.pending:
            accepted = next((a for a in reversed(attempts) if a.get('accepted')), None)
            step = accepted['rollback_step'] if accepted else None
            failed = [a['rollback_step'] for a in attempts if not a.get('accepted')] if accepted else []
            before = list(judge.weights)
            telemetry = judge.record_recovery_outcome(case.tid, step, verified=bool(accepted),
                verifier_score_before=0.0, verifier_score_after=1.0 if accepted else max(
                    (0.5 if looks_ok(r) else 0.0 for r in env.rollouts.values()), default=0.0),
                stopped_early=bool(env.rollouts) and not accepted and list(env.rollouts.values())[-1].termination != 'user_stop',
                failed_steps=failed)
            result['policy_update'] = {'before': before, 'after': list(judge.weights), 'reward': telemetry['reward']}
            judge.save(output/'policy_after.json')
        result['status'] = 'partial' if diagnosed['status'] != 'ok' or provider.metrics()['failed_calls'] else 'completed'
        emit({'kind': 'verification_completed', **{k: result[k] for k in ('decision','task_outcome','verification')},
              'policy_update': result.get('policy_update')})
    except ComparisonCancelled as exc:
        result.update(status='canceled', error=str(exc))
    except Exception as exc:
        result.update(status='error', error=f'{type(exc).__name__}: {exc}')
    finally:
        result['metrics'] = {**provider.metrics(), 'elapsed_seconds': round(time.monotonic() - started, 3)}
        write_json(output/'result.json', result)
        emit({'kind': 'system_finished', 'result': result})
    return result


class ComparisonService:
    def __init__(self, output='runs/ui-comparisons', *, provider_factory=local_provider, model_catalog=installed_models,
                 max_jobs=8):
        self.output, self.provider_factory, self.model_catalog = Path(output), provider_factory, model_catalog
        self.max_jobs, self.jobs = max_jobs, {}
        self.lock, self.closed = threading.RLock(), False

    def start(self, scenario, model):
        if not isinstance(scenario, str) or scenario not in SCENARIOS:
            raise ValueError('Select a known scenario.')
        if not isinstance(model, str) or len(model) > 512 or model not in self.model_catalog():
            raise ValueError('Select an installed local Ollama model.')
        with self.lock:
            if self.closed:
                raise RuntimeError('Comparison service is closed.')
            if any(j['status'] not in TERMINAL for j in self.jobs.values()):
                raise RuntimeError('A comparison is already running. Wait or cancel it before starting another.')
            while len(self.jobs) >= self.max_jobs:
                self.jobs.pop(next(iter(self.jobs)))
            jid = uuid.uuid4().hex
            job = {'id': jid, 'scenario': scenario, 'model': model, 'status': 'running', 'events': [], 'results': {},
                   'cancel': threading.Event(), 'created': time.time()}
            self.jobs[jid] = job
            threading.Thread(target=self._run, args=(job,), daemon=True, name='comparison-' + jid[:8]).start()
            return {'id': jid, 'status': job['status']}

    def _emit(self, job, event):
        with self.lock:
            job['events'].append({**copy.deepcopy(event), 'seq': len(job['events']) + 1,
                                  'elapsed_seconds': round(time.time() - job['created'], 3)})

    def _run(self, job):
        output = self.output/job['id']
        try:
            case = ScenarioInput(job['scenario'])
            write_json(output/'input.json', case.trace)
            self._emit(job, {'kind': 'input_ready', 'trace': case.trace, 'input_hash': case.input_hash,
                             'frames': step_frames(case.original, 'initial_run'),
                             'notice': 'Controlled scripted input and user; actual Ollama diagnosis and post-rollback agent turns. Sequential systems; initial MP weights.'})
            for system in SYSTEMS:
                if job['cancel'].is_set():
                    break
                result = run_system(system, case, self.provider_factory(job['model']), output/system,
                                    lambda e, s=system: self._emit(job, {**e, 'system': s}), job['cancel'])
                with self.lock:
                    job['results'][system] = result
            terminal_status = 'canceled' if job['cancel'].is_set() else 'completed'
        except Exception as exc:
            terminal_status = 'error'
            self._emit(job, {'kind': 'job_error', 'error': f'{type(exc).__name__}: {exc}'})
        finally:
            with self.lock:
                self._emit(job, {'kind': 'job_finished', 'status': terminal_status})
                snapshot = self.snapshot(job['id'])
                snapshot['status'] = terminal_status
                try:
                    write_json(output/'comparison.json', snapshot)
                except Exception as exc:
                    terminal_status = 'error'
                    self._emit(job, {'kind': 'job_error', 'error': f'Could not save comparison: {exc}'})
                job['status'] = terminal_status

    def snapshot(self, job_id, after=0):
        if type(after) is not int or after < 0:
            raise ValueError('Event cursor must be a nonnegative integer.')
        with self.lock:
            job = self.jobs[job_id]
            return copy.deepcopy({k: job[k] for k in ('id','scenario','model','status','results')}) | {
                'cancel_requested': job['cancel'].is_set(), 'events': copy.deepcopy(job['events'][after:]),
                'artifact_dir': str((self.output/job_id).resolve())}

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs[job_id]
            if job['status'] not in TERMINAL:
                job['cancel'].set()
            return {'id': job_id, 'cancel_requested': job['cancel'].is_set()}

    def close(self):
        with self.lock:
            self.closed = True
            for job in self.jobs.values():
                job['cancel'].set()
