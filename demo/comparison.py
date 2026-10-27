"""Published-trace comparisons: blind inference, frozen policy, then exact attribution scoring."""
import copy
import json
import threading
import time
import uuid
from pathlib import Path

from agentrx.data import normalize
from agentrx.io import digest, write_json, read_json
from agentrx.pipeline import diagnose, diagnose_mp_rl
from agentrx.rl_policy import DualLoopRLJudge
from agentrx.taxonomy import agent, category
from . import model_settings

SYSTEMS = ('agentrx', 'agentreroute')
TERMINAL = ('completed', 'partial', 'error', 'canceled')
CORPUS = Path(__file__).resolve().parents[1] / 'data' / 'benchmark'
VARIANTS = {
    'seeds_select': {'evidence': False, 'protocol': 'mp_rl', 'llm_selector': 'trace', 'selector_pool': 5},
    'synth_select_p10': {'evidence': True, 'protocol': 'mp_rl', 'llm_selector': 'trace', 'selector_pool': 10},
}
MISSING_CORPUS = ('Published benchmark cases are unavailable. Prepare data/benchmark/cases.json '
                  'with python3 -m agentrx prepare --fetch, then refresh the case catalog.')


class ComparisonCancelled(RuntimeError):
    pass


class MeasuredProvider:
    """Observe transport without altering prompts or model responses. No response cache is used."""
    def __init__(self, provider, emit, output, cancel, max_calls=1024):
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
        self.provider.last_telemetry = {}
        self.last_telemetry = {}
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
            self.last_telemetry = copy.deepcopy(getattr(self.provider, 'last_telemetry', {}))
            row.update(status='error', seconds=round(time.monotonic() - started, 3),
                       telemetry=self.last_telemetry, error=f'{type(exc).__name__}: {exc}')
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
        finished = [r for r in self.calls if r['status'] in ('completed', 'error')]
        result = {'model_calls': len(self.calls), 'completed_calls': len(completed),
                  'failed_calls': sum(r['status'] == 'error' for r in self.calls),
                  'model_seconds': round(sum(r.get('seconds', 0) for r in self.calls), 3)}
        for field, key in (('prompt_tokens', 'prompt_tokens'), ('completion_tokens', 'completion_tokens')):
            values = [r['telemetry'].get('usage', {}).get(key) for r in finished]
            result[field] = sum(values) if values and all(type(v) is int and v >= 0 for v in values) else None
        return result


def synthesis_summary(path, checks):
    """Expose validation fallback separately from successful but empty model synthesis."""
    views = read_json(path)
    summary = {'llm_constraints': 0, 'seed_constraints': 0, 'rejected_constraints': 0,
               'failed_stages': 0, 'failed_perspectives': [], 'issues': []}
    for perspective, view in views.items():
        for constraint in view['constraints']:
            summary['seed_constraints' if constraint.get('origin') == 'seed' else 'llm_constraints'] += 1
        latest = {}
        for entry in view['synthesis_log']:
            latest[entry['stage']] = entry
            summary['rejected_constraints'] += len(entry['rejected'])
            for reason in ([entry['error']] if entry.get('error') else []) + [r['reason'] for r in entry['rejected']]:
                summary['issues'].append({'perspective': perspective, 'stage': entry['stage'],
                                          'attempt': entry['attempt'], 'reason': reason})
        failed = sum(bool(e.get('error') or e['rejected']) for e in latest.values())
        summary['failed_stages'] += failed
        if failed:
            summary['failed_perspectives'].append(perspective)
    model_checks = [c for c in checks if c.get('origin') == 'llm']
    summary['llm_check_evaluations'] = len(model_checks)
    summary['llm_conclusive_checks'] = sum(c['verdict'] == 'VIOL' or
        (c['verdict'] == 'SAT' and not c.get('unclear_count')) for c in model_checks)
    summary['state'] = ('invalid_synthesis_fallback' if not summary['llm_constraints'] else 'partial_synthesis') \
        if summary['failed_stages'] else ('model_and_seeds' if summary['llm_constraints'] else 'valid_empty_synthesis')
    return summary


class PublishedInput:
    """An allowlisted observable view. Annotations are never retained on the inference input."""
    def __init__(self, record):
        raw = record['trajectory']
        self.trace = normalize(raw, raw['domain'], raw['trajectory_id'])
        self.policy = str(record.get('policy', ''))
        self.tools = copy.deepcopy(record.get('tools', {}))
        self.input_hash = digest({'trajectory': self.trace, 'policy': self.policy, 'tools': self.tools})


def run_system(system, case, raw_provider, output, emit, cancel, variant='seeds_select'):
    """Finish a blind attribution arm. The service scores only after both arms have returned."""
    output = Path(output)
    provider = MeasuredProvider(raw_provider, emit, output, cancel)
    started = time.monotonic()
    result = {'system': system, 'status': 'running', 'input_hash': case.input_hash,
              'variant': variant if system == 'agentreroute' else 'agentrx',
              'policy_state': 'THETA0 frozen; inference only' if system == 'agentreroute' else 'joint judge',
              'provider': copy.deepcopy(provider.identity), 'degradation_reasons': []}
    emit({'kind': 'system_started', **result})
    try:
        if hasattr(raw_provider, 'prepare_for_run'):
            raw_provider.prepare_for_run()
        result['provider'] = copy.deepcopy(raw_provider.identity)
        config = ({**VARIANTS[variant], 'recovery_loop': False} if system == 'agentreroute' else
                  {'evidence': True, 'checklist': True, 'generation': 'oneshot',
                   'protocol': 'joint', 'recovery_loop': False})
        result['config'] = config
        if system == 'agentreroute':
            judge = DualLoopRLJudge(read_only=True)
            diagnosed = diagnose_mp_rl(copy.deepcopy(case.trace), case.policy, copy.deepcopy(case.tools), config,
                                      provider, output/'diagnosis', rl_judge=judge)
        else:
            diagnosed = diagnose(copy.deepcopy(case.trace), case.policy, copy.deepcopy(case.tools), config,
                                 provider, output/'diagnosis')
        if cancel.is_set():
            raise ComparisonCancelled('Comparison canceled.')
        result['diagnosis_seconds'] = round(time.monotonic() - started, 3)
        result['prediction'] = diagnosed['prediction']
        checks = diagnosed['checks']
        counts = {v: sum(c['verdict'] == v for c in checks) for v in ('SAT', 'VIOL', 'SKIP', 'ERROR')}
        if system == 'agentreroute' and (output/'diagnosis'/'perspectives.json').exists():
            result['synthesis'] = synthesis_summary(output/'diagnosis'/'perspectives.json', checks)
            if not config['evidence']:
                result['synthesis']['state'] = 'fixed_checks'
            emit({'kind': 'synthesis_summary', 'summary': result['synthesis']})
        result['checks'] = counts
        result['seed_checks'] = sum(c.get('origin') == 'seed' for c in checks)
        emit({'kind': 'checks_completed', 'counts': counts, 'seed_checks': result['seed_checks'],
              'rows': [{k: c.get(k) for k in ('step_index','assertion_name','perspective','verdict',
                                            'check_hint','error','origin','unclear_count')} for c in checks]})
        if diagnosed['status'] == 'error':
            raise RuntimeError(diagnosed.get('error', 'Diagnosis failed'))
        result['ranked_candidates'] = diagnosed.get('ranked_candidates', [])
        result['selection'] = diagnosed.get('selection')
        reasons = result['degradation_reasons']
        if diagnosed['status'] != 'ok' or counts['ERROR']:
            reasons.append('One or more checks failed.')
        if provider.metrics()['failed_calls']:
            reasons.append('One or more model requests failed.')
        if result.get('synthesis', {}).get('failed_stages'):
            reasons.append('Written-check synthesis failed or returned invalid constraints; seed fallback was used.')
        if (result.get('selection') or {}).get('fell_back'):
            reasons.append('The LLM selector failed; the displayed prediction is the frozen policy fallback.')
        if system == 'agentreroute' and (not result.get('selection') or result['selection'].get('status') != 'ok'):
            if not reasons:
                reasons.append('The LLM selector did not produce a clean attribution.')
        result['status'] = 'partial' if reasons else 'completed'
        emit({'kind': 'attribution_completed', 'prediction': result['prediction'],
              'selection': result['selection'], 'ranked_candidates': result['ranked_candidates'],
              'diagnosis_seconds': result['diagnosis_seconds'], 'status': result['status'],
              'degradation_reasons': reasons})
    except ComparisonCancelled as exc:
        result.update(status='canceled', error=str(exc))
    except Exception as exc:
        result.update(status='error', error=f'{type(exc).__name__}: {exc}')
    finally:
        result['metrics'] = {**provider.metrics(), 'elapsed_seconds': round(time.monotonic() - started, 3)}
        write_json(output/'result.json', result)
    return result


def score_prediction(prediction, gold):
    """Evaluation boundary; called only after both blind decision paths have finished."""
    predicted = prediction or {}
    critical = gold['critical']
    return {'step_correct': predicted.get('step_index') == critical['step_index'],
            'agent_correct': agent(predicted.get('agent')) == agent(critical['agent']),
            'category_correct': bool(predicted.get('category')) and category(predicted['category']) == category(critical['category']),
            'gold_step': critical['step_index'], 'predicted_step': predicted.get('step_index'),
            'gold_agent': critical['agent'], 'predicted_agent': predicted.get('agent'),
            'gold_category': critical['category'], 'predicted_category': predicted.get('category')}


class ComparisonService:
    def __init__(self, output='runs/ui-comparisons', *, provider_factory=None, model_catalog=None,
                 max_jobs=8, corpus=CORPUS):
        self.output, self.corpus = Path(output), Path(corpus)
        self.provider_factory = provider_factory or model_settings.gemini_provider
        self.model_catalog = model_catalog or model_settings.gemini_models
        self._injected_provider = provider_factory is not None or model_catalog is not None
        self.max_jobs, self.jobs = max_jobs, {}
        self.lock, self.closed = threading.RLock(), False
        self.catalog_error = None

    def _records(self):
        try:
            records = read_json(self.corpus/'cases.json')
            if not isinstance(records, list):
                raise ValueError('cases.json must contain a list')
            selected = {}
            for record in records:
                trace = record['trajectory']
                if trace['domain'] not in ('tau', 'magentic'):
                    continue
                cid = f"{trace['domain']}:{trace['trajectory_id']}"
                if cid in selected:
                    raise ValueError('Duplicate case identifier')
                selected[cid] = record
            self.catalog_error = None if selected else MISSING_CORPUS
            return selected
        except (OSError, ValueError, KeyError, TypeError):
            self.catalog_error = MISSING_CORPUS
            return {}

    def catalog(self):
        return [{'id': cid, 'title': f"{row['trajectory']['domain'].title()} · trace {row['trajectory']['trajectory_id']}",
                 'domain': row['trajectory']['domain'], 'trajectory_id': str(row['trajectory']['trajectory_id']),
                 'steps': len(row['trajectory']['steps'])} for cid, row in self._records().items()]

    def model_info(self):
        if not self._injected_provider:
            return model_settings.model_info()
        models = self.model_catalog()
        return {'models': models, 'provider': 'test', 'configured': True,
                'default_model': models[0] if models else None}

    def start(self, scenario, model, variant='seeds_select'):
        info = self.model_info()
        if not info['configured']:
            raise ValueError(info['message'])
        if not isinstance(model, str) or len(model) > 512 or model not in info['models']:
            raise ValueError('Select an available model.')
        if not isinstance(variant, str) or variant not in VARIANTS:
            raise ValueError('Select fixed checks (seeds_select) or written checks (synth_select_p10).')
        records = self._records()
        if not records:
            raise ValueError(self.catalog_error)
        if not isinstance(scenario, str) or scenario not in records:
            raise ValueError('Select a published case from the catalog.')
        # Snapshot observable input before the background run; scoring annotations stay in this
        # service and are not attached to PublishedInput or passed to either attribution pipeline.
        case = PublishedInput(records[scenario])
        with self.lock:
            if self.closed:
                raise RuntimeError('Comparison service is closed.')
            if any(j['status'] not in TERMINAL for j in self.jobs.values()):
                raise RuntimeError('A comparison is already running. Wait or cancel it before starting another.')
            while len(self.jobs) >= self.max_jobs:
                self.jobs.pop(next(iter(self.jobs)))
            jid = uuid.uuid4().hex
            job = {'id': jid, 'scenario': scenario, 'model': model, 'variant': variant,
                   'mode': 'published_attribution', 'legacy': False, 'status': 'running', 'events': [], 'results': {},
                   'provider': info['provider'], 'cancel': threading.Event(), 'created': time.time()}
            self.jobs[jid] = job
            threading.Thread(target=self._run, args=(job, case, records[scenario]), daemon=True,
                             name='comparison-' + jid[:8]).start()
            return {'id': jid, 'status': job['status']}

    def _emit(self, job, event):
        with self.lock:
            job['events'].append({**copy.deepcopy(event), 'seq': len(job['events']) + 1,
                                  'elapsed_seconds': round(time.time() - job['created'], 3)})

    def _run(self, job, case, record):
        output = self.output/job['id']
        terminal_status = 'error'
        try:
            write_json(output/'input.json', {'trajectory': case.trace, 'policy': case.policy, 'tools': case.tools})
            self._emit(job, {'kind': 'input_ready', 'trace': case.trace, 'input_hash': case.input_hash,
                             'notice': 'Published benchmark trace. Matched model settings; sequential attribution; THETA0 frozen. Labels are scored after both predictions.'})
            # One transport means one rolling token budget across BOTH systems, including retries.
            provider = self.provider_factory(job['model'])
            for system in SYSTEMS:
                if job['cancel'].is_set():
                    break
                result = run_system(system, case, provider, output/system,
                                    lambda e, s=system: self._emit(job, {**e, 'system': s}),
                                    job['cancel'], job['variant'])
                with self.lock:
                    job['results'][system] = result
            # Gold is first accessed here, after every model decision. It never enters a prompt.
            for system, result in job['results'].items():
                if result.get('prediction'):
                    result['score'] = score_prediction(result['prediction'], record['gold'])
                    self._emit(job, {'kind': 'attribution_scored', 'system': system, 'score': result['score']})
                write_json(output/system/'result.json', result)
                self._emit(job, {'kind': 'system_finished', 'system': system, 'result': result})
            terminal_status = ('canceled' if job['cancel'].is_set() else
                               'completed' if len(job['results']) == len(SYSTEMS) and
                               all(r['status'] == 'completed' for r in job['results'].values()) else 'partial')
        except Exception as exc:
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
            job = self._lookup(job_id)
            return copy.deepcopy({k: job[k] for k in
                ('id','scenario','model','provider','status','results','variant','mode','legacy')}) | {
                'cancel_requested': job['cancel'].is_set(), 'events': copy.deepcopy(job['events'][after:]),
                'artifact_dir': str((self.output/job_id).resolve())}

    def cancel(self, job_id):
        with self.lock:
            job = self._lookup(job_id)
            if job['status'] not in TERMINAL:
                job['cancel'].set()
            return {'id': job_id, 'cancel_requested': job['cancel'].is_set()}

    def _lookup(self, job_id):
        if job_id in self.jobs:
            return self.jobs[job_id]
        if not isinstance(job_id, str) or len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
            raise KeyError(job_id)
        path = self.output/job_id/'comparison.json'
        try:
            if path.stat().st_size > 16_000_000:
                raise KeyError(job_id)
            saved = read_json(path)
            if saved['id'] != job_id or saved['status'] not in TERMINAL or not isinstance(saved['events'], list):
                raise KeyError(job_id)
            job = {k: saved[k] for k in ('id','scenario','model','status','events','results')}
            job['provider'] = saved.get('provider') or next((r.get('provider', {}).get('provider')
                for r in saved['results'].values() if r.get('provider', {}).get('provider')), 'ollama')
            job['legacy'] = saved.get('mode') != 'published_attribution'
            job['mode'] = saved.get('mode', 'legacy_recovery')
            job['variant'] = saved.get('variant', 'legacy')
        except (OSError, ValueError, TypeError):
            raise KeyError(job_id) from None
        while len(self.jobs) >= self.max_jobs:
            old = next((k for k, v in self.jobs.items() if v['status'] in TERMINAL), None)
            if old is None:
                raise KeyError(job_id)
            self.jobs.pop(old)
        job.update(cancel=threading.Event(), created=path.stat().st_mtime)
        self.jobs[job_id] = job
        return job

    def close(self):
        with self.lock:
            self.closed = True
            for job in self.jobs.values():
                job['cancel'].set()
