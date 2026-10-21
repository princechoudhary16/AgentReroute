"""Attribution baselines. Each reads only public traces and returns a submission.

* ``random``        a random agent step and mode (seeded); a floor.
* ``last_step``     the last agent step; a common naive heuristic.
* ``heuristic``     rule checks over observable evidence (errors, stuck tools,
                    repeated calls, failed handoffs), with the earliest anomaly first.
                    Also raises an online alarm, so it is the detection baseline.
* ``llm_judge``     all-at-once LLM judge over the whole trace (Who&When "all-at-once").
* ``step_by_step``  LLM judges each step in order and stops at the first decisive error
                    (Who&When "step-by-step").

``from_agentrx_run`` converts the faultattribution branch's run directory into a
submission, mapping AgentRx categories to the seven modes.
"""
import json
import random
from pathlib import Path

from .. import BENCHMARK_NAME
from ..env.llm import parse_json, request
from ..io import read_json
from ..taxonomy import INCONCLUSIVE, MODES, describe, from_agentrx, mode as normalize_mode

AGENT_ROLES_EXCLUDED = ('user', 'tool')


def _agent_steps(trace):
    return [(s['index'], s['substeps'][0]['role']) for s in trace['steps']
            if s['substeps'][0]['role'] not in AGENT_ROLES_EXCLUDED]


def _submission(system, predictions, view, **extra):
    return {'benchmark': BENCHMARK_NAME, 'system': system, 'view': view, 'predictions': predictions, **extra}


def random_baseline(traces, view='output_only', seed=0):
    rng = random.Random(seed)
    rows = []
    for tid, trace in sorted(traces.items()):
        steps = _agent_steps(trace) or [(trace['steps'][-1]['index'], trace['steps'][-1]['substeps'][0]['role'])]
        index, agent = rng.choice(steps)
        rows.append({'trajectory_id': tid, 'hypotheses': [
            {'step_index': index, 'agent': agent, 'mode': rng.choice(MODES), 'confidence': round(1 / len(steps), 4)}]})
    return _submission('random', rows, view)


def last_step_baseline(traces, view='output_only'):
    rows = []
    for tid, trace in sorted(traces.items()):
        steps = _agent_steps(trace)
        if not steps:
            rows.append({'trajectory_id': tid, 'hypotheses': []})
            continue
        index, agent = steps[-1]
        rows.append({'trajectory_id': tid, 'hypotheses': [
            {'step_index': index, 'agent': agent, 'mode': 'missing_details', 'confidence': 0.3}]})
    return _submission('last_step', rows, view)


# ----------------------------------------------------------------------------- heuristic
def _calls(trace):
    out = []
    for step in trace['steps']:
        for event in step['substeps']:
            for call in event.get('tool_calls', []):
                out.append((step['index'], event['role'], call['name'], json.dumps(call.get('arguments', {}),
                                                                                    sort_keys=True)))
    return out


def heuristic_findings(trace):
    """Observable anomalies as ranked hypotheses (earliest first) plus an alarm step."""
    findings = []
    steps = trace['steps']
    by_index = {s['index']: s for s in steps}
    order = [s['index'] for s in steps]

    def next_agent_step(after, role):
        return next((i for i in order if i > after and by_index[i]['substeps'][0]['role'] == role), after)

    calls = {i: (role, name, args) for i, role, name, args in _calls(trace)}
    for pos, step in enumerate(steps):
        event = step['substeps'][0]
        if event['role'] != 'tool':
            continue
        caller = next((calls[i][0] for i in reversed(order[:pos]) if i in calls), 'unknown')
        content = event['content']
        if content.startswith('Error'):
            findings.append((step['index'], next_agent_step(step['index'], caller), caller, 'tool_failure', 0.6))
        elif '"status": "processing"' in content:
            findings.append((step['index'], next_agent_step(step['index'], caller), caller, 'cycles', 0.6))
    seen = {}
    for index, role, name, args in _calls(trace):
        key = (role, name, args)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 3:
            findings.append((index, index, role, 'cycles', 0.7))
    for step in steps:
        for event in step['substeps']:
            if event.get('name') != 'handoff':
                continue
            try:
                handoff = json.loads(event['content'])
            except json.JSONDecodeError:
                continue
            context = handoff.get('context') or {}
            if not any(k in context for k in ('order_id', 'user_id', 'item_ids')):
                findings.append((step['index'], step['index'], event['role'], 'context_propagation', 0.5))
    for step in steps:
        event = step['substeps'][0]
        text = event['content'].lower()
        if event['role'] not in AGENT_ROLES_EXCLUDED and ('cannot handle' in text or 'cannot proceed' in text):
            findings.append((step['index'], step['index'], event['role'], 'drift', 0.4))
    findings.sort(key=lambda f: (f[0], -f[4]))
    return findings


def heuristic_baseline(traces, view='output_only'):
    rows = []
    for tid, trace in sorted(traces.items()):
        findings = heuristic_findings(trace)
        hyps, used = [], set()
        for onset, step, agent, mode, conf in findings:
            if step in used:
                continue
            used.add(step)
            hyps.append({'step_index': step, 'agent': agent, 'mode': mode, 'confidence': conf,
                         'rationale': f'observable anomaly at step {onset}'})
        if not hyps:
            steps = _agent_steps(trace)
            if steps:
                index, agent = steps[-1]
                hyps.append({'step_index': index, 'agent': agent, 'mode': 'missing_details', 'confidence': 0.2,
                             'rationale': 'no anomaly found; final answer suspected'})
        rows.append({'trajectory_id': tid, 'hypotheses': hyps[:3],
                     'alarm_step': findings[0][0] if findings else None})
    return _submission('heuristic', rows, view, detection=True)


# ----------------------------------------------------------------------------- LLM judges
def render_trace(trace, max_chars=600):
    lines = [f'Task (first user message): {trace.get("instruction", "")}', '']
    for step in trace['steps']:
        for event in step['substeps']:
            content = event['content']
            if len(content) > max_chars:
                content = content[:max_chars] + ' ...[truncated]'
            name = f' ({event["name"]})' if event.get('name') else ''
            lines.append(f'[step {step["index"]}] {event["role"]}{name}: {content}')
    return '\n'.join(lines)


def _modes_text():
    return '\n'.join(f'- {m["mode"]}: {m["definition"]} Rules: {" ".join(m["decision_rules"])}' for m in describe())


JUDGE_SYSTEM = ('You diagnose failed runs of a multi-agent customer-service system. Find the FIRST critical, '
                'unrecoverable failure: the earliest step after which the task could no longer succeed without '
                'intervention. Use only evidence in the trace.')


def llm_judge_baseline(traces, llm, model, view='output_only', seed=0, temperature=0.0):
    rows = []
    for tid, trace in sorted(traces.items()):
        prompt = (f'Failure modes:\n{_modes_text()}\n\nTrace:\n{render_trace(trace)}\n\n'
                  'Return JSON: {"hypotheses": [{"step_index": <int>, "agent": "<role at that step>", '
                  '"mode": "<one mode id>", "confidence": <0..1>, "rationale": "<short>"}]} with up to 3 '
                  'hypotheses, most likely first.')
        messages = [{'role': 'system', 'content': JUDGE_SYSTEM}, {'role': 'user', 'content': prompt}]
        try:
            reply = parse_json(llm.complete(request(model, messages, temperature, seed, agent='judge',
                                                    trajectory_id=tid)).text)
            hyps = [_clean_hypothesis(h, trace) for h in reply.get('hypotheses', [])]
            hyps = [h for h in hyps if h is not None][:3]
        except ValueError:
            hyps = []
        rows.append({'trajectory_id': tid, 'hypotheses': hyps})
    return _submission(f'llm_judge:{model}', rows, view)


def step_by_step_baseline(traces, llm, model, view='output_only', seed=0, temperature=0.0, max_steps=40):
    rows = []
    for tid, trace in sorted(traces.items()):
        hyps = []
        agent_steps = [s for s in trace['steps'] if s['substeps'][0]['role'] not in AGENT_ROLES_EXCLUDED][:max_steps]
        for step in agent_steps:
            prefix = {**trace, 'steps': [s for s in trace['steps'] if s['index'] <= step['index']]}
            prompt = (f'Failure modes:\n{_modes_text()}\n\nTrace so far:\n{render_trace(prefix)}\n\n'
                      f'Is step {step["index"]} the first critical failure? Return JSON: '
                      '{"decisive": true|false, "mode": "<mode id or null>", "confidence": <0..1>, '
                      '"rationale": "<short>"}')
            messages = [{'role': 'system', 'content': JUDGE_SYSTEM}, {'role': 'user', 'content': prompt}]
            try:
                reply = parse_json(llm.complete(request(model, messages, temperature, seed, agent='judge',
                                                        trajectory_id=tid, step_index=step['index'])).text)
            except ValueError:
                continue
            if reply.get('decisive') is True:
                h = _clean_hypothesis({**reply, 'step_index': step['index'],
                                       'agent': step['substeps'][0]['role']}, trace)
                if h:
                    hyps.append(h)
                break
        rows.append({'trajectory_id': tid, 'hypotheses': hyps})
    return _submission(f'step_by_step:{model}', rows, view)


def _clean_hypothesis(h, trace):
    if not isinstance(h, dict):
        return None
    indices = {s['index'] for s in trace['steps']}
    step = h.get('step_index')
    if isinstance(step, str) and step.isdigit():
        step = int(step)
    if type(step) is not int or step not in indices:
        return None
    try:
        label = normalize_mode(h.get('mode'))
    except ValueError:
        label = INCONCLUSIVE
    if label == INCONCLUSIVE:
        return None
    try:
        confidence = min(1.0, max(0.0, float(h.get('confidence', 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5
    role = next(s['substeps'][0]['role'] for s in trace['steps'] if s['index'] == step)
    return {'step_index': step, 'agent': str(h.get('agent') or role), 'mode': label, 'confidence': confidence,
            'rationale': str(h.get('rationale', ''))[:500]}


# ----------------------------------------------------------------------------- faultattribution import
def from_agentrx_run(run_dir, variant=None, repetition=None, system=None):
    """Convert ``jobs/<variant>/rep-<n>/<domain>/<hash>/result.json`` files to a submission.

    AgentRx gives one prediction per run; when several repetitions are present,
    each distinct step becomes a hypothesis with confidence = vote share.
    """
    votes = {}
    for path in sorted(Path(run_dir).glob('jobs/*/rep-*/*/*/result.json')):
        parts = path.parts
        if variant and parts[-5] != variant:
            continue
        if repetition is not None and parts[-4] != f'rep-{repetition}':
            continue
        result = read_json(path)
        prediction = result.get('prediction')
        tid = str(result['trajectory_id'])
        bucket = votes.setdefault(tid, {'total': 0, 'steps': {}})
        bucket['total'] += 1
        if not prediction or prediction.get('category') == 'Inconclusive' or prediction.get('step_index') is None:
            continue
        mapped, _ = from_agentrx(prediction['category'])
        key = prediction['step_index']
        entry = bucket['steps'].setdefault(key, {'votes': 0, 'agent': prediction.get('agent', ''), 'modes': {}})
        entry['votes'] += 1
        entry['modes'][mapped] = entry['modes'].get(mapped, 0) + 1
    rows = []
    for tid, bucket in sorted(votes.items()):
        hyps = []
        for step, entry in sorted(bucket['steps'].items(), key=lambda kv: (-kv[1]['votes'], kv[0]))[:3]:
            label = max(entry['modes'].items(), key=lambda kv: kv[1])[0]
            hyps.append({'step_index': step, 'agent': entry['agent'],
                         'mode': label if label in MODES else 'out_of_scope',
                         'confidence': round(entry['votes'] / bucket['total'], 4)})
        rows.append({'trajectory_id': tid, 'hypotheses': hyps})
    return _submission(system or f'agentrx:{variant or "all"}', rows, 'output_only')
