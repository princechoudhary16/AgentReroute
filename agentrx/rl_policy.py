"""Phase 2 dual-loop RL judge: [C2] action, [C3] live repair, [C4] split-role verification, [C5] reward."""
import copy
import json
import math
from pathlib import Path
from .constraints import check_constraint
from .io import digest, read_json, write_json
from .perspectives import FEATURES, ORDER
from .taxonomy import agent

# Initial theta over the 10 candidate features, in order: blame per perspective (0-2), violation
# density per perspective (3-5), python-check share (6), temporal position (7), perspective breadth (8),
# max density (9).
#
# Two deliberate moves from the first vector (..., 1.5, 0.9, 0.7, ...):
#   f6 1.5 -> 0.85 so a python check cannot drown out an LLM nl_check. This has no effect on a seed-only
#     run, where every check is a python check and f6 is a constant across candidates, so it is
#     unmeasurable until constraint synthesis is on (it changes nothing in the external-trace ablation).
#   f7 0.9 -> 0.45 so temporal position cannot override violation density (1.15-1.35) or breadth (0.90).
#     On the 29 external tau traces this moves exact step 10.3% -> 13.8% and acc@3 34.5% -> 41.4%.
# f7 = 0.0 reaches the same 13.8% exact with worse acc@3 (37.9%), and is NOT adopted: picking a weight
# because it maximises the set we report on is the overfitting this evaluation exists to avoid. Training
# on corpus-v2 is where theta is supposed to move.
THETA0 = (0.85, 0.95, 0.90, 1.15, 1.35, 1.25, 0.85, 0.45, 0.90, 1.20)


def _snippet(value, limit=500):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text[:limit]


class ForbiddenPatternMemory:
    """Persistent negative memory: fingerprints of outputs Layer 1 rejected, injected as 'never emit' hints."""

    def __init__(self, patterns=None):
        self.patterns = dict(patterns or {})

    def __contains__(self, pattern_hash):
        return pattern_hash in self.patterns

    def record(self, perspective_id, value, reason):
        key = digest(value)
        entry = self.patterns.setdefault(key, {'pattern_hash': key, 'perspective_id': perspective_id,
                                               'snippet': _snippet(value), 'error_reason': reason, 'count': 0})
        entry['count'] += 1
        entry['error_reason'] = reason
        return key

    def retrieve(self, perspective_id, k=5):
        rows = sorted((p for p in self.patterns.values() if p['perspective_id'] == perspective_id),
                      key=lambda p: -p['count'])
        return [{'snippet': p['snippet'], 'error_reason': p['error_reason']} for p in rows[:k]]

    def to_dict(self):
        return {'patterns': copy.deepcopy(self.patterns)}

    @classmethod
    def from_dict(cls, data):
        return cls(data.get('patterns'))


class PromptMemoryBank:
    """Verbal loop M_i: exemplar constraints scored by UCB Q(e) = mean_R + c*sqrt(ln(N+1)/(n_e+1))."""

    def __init__(self, exemplars=None, forbidden=None, total_updates=0, ucb_c=0.5):
        self.exemplars = dict(exemplars or {})
        self.forbidden = forbidden or ForbiddenPatternMemory()
        self.total_updates, self.ucb_c = total_updates, ucb_c

    def score(self, entry):
        mean = entry['total_reward'] / entry['uses'] if entry['uses'] else 0.0
        return mean + self.ucb_c * math.sqrt(math.log(self.total_updates + 1) / (entry['uses'] + 1))

    def retrieve_exemplars(self, perspective_id, k=2):
        rows = [e for e in self.exemplars.values()
                if e['perspective_id'] == perspective_id and e['verified_count'] > e['failed_count']]
        return [e['exemplar'] for e in sorted(rows, key=lambda e: -self.score(e))[:k]]

    def retrieve_forbidden(self, perspective_id, k=5):
        return self.forbidden.retrieve(perspective_id, k)

    def record_forbidden(self, perspective_id, snippet, reason):
        return self.forbidden.record(perspective_id, snippet, reason)

    def record_episode_feedback(self, perspective_id, constraints_used, reward):
        self.total_updates += 1
        scores = {}
        for c in constraints_used:
            exemplar = {k: v for k, v in c.items() if k not in ('origin', 'relaxed')}
            key = digest(exemplar)
            entry = self.exemplars.setdefault(key, {'exemplar_hash': key, 'perspective_id': perspective_id,
                                                    'exemplar': exemplar, 'verified_count': 0, 'failed_count': 0,
                                                    'total_reward': 0.0, 'uses': 0})
            entry['uses'] += 1
            entry['total_reward'] += reward
            if reward > 0:
                entry['verified_count'] += 1
            elif reward < 0:
                entry['failed_count'] += 1
            scores[key] = self.score(entry)
        return scores

    def to_dict(self):
        return {'exemplars': copy.deepcopy(self.exemplars), 'forbidden': self.forbidden.to_dict(),
                'total_updates': self.total_updates, 'ucb_c': self.ucb_c}

    @classmethod
    def from_dict(cls, data):
        return cls(data.get('exemplars'), ForbiddenPatternMemory.from_dict(data.get('forbidden', {})),
                   data.get('total_updates', 0), data.get('ucb_c', 0.5))

    def save(self, path):
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path):
        return cls.from_dict(read_json(path)) if Path(path).exists() else cls()


def outcome_reward(accepted, verifier_score=None):
    """Observable (non-gold) recovery feedback -> R in [-1, 1], keeping the +/- floors of the intrinsic reward."""
    if verifier_score is None:
        return 1.0 if accepted else -1.0
    value = max(-1.0, min(1.0, 2.0 * float(verifier_score) - 1.0))
    return max(0.25, value) if accepted else min(-0.25, value)


def verifier_reward(verified, score_before=0.0, score_after=None, stopped_early=False):
    """Cross-branch verifier deltas -> R in [-1, 1]: +1 verified, -1 stopped early, else ΔV-scaled penalty.

    Unverified repairs stay negative (at most -0.1, the spec's step penalty) but rank by score_after - score_before.
    """
    if verified:
        return 1.0
    if stopped_early:
        return -1.0
    delta = max(-1.0, min(1.0, float(score_before if score_after is None else score_after) - float(score_before)))
    return min(-0.1, 0.5 * delta - 0.5)


class DualLoopRLJudge:
    """[C2] linear-softmax policy pi_theta(k, t) over candidate features h_{k,t}, plus prompt memory M_i."""

    def __init__(self, weights=None, learning_rate=0.05, temperature=0.5, ucb_alpha=0.1,
                 baseline_reward=0.0, action_counts=None, memory=None, episodes=0, pairwise_lr=0.15,
                 read_only=False):
        # read_only is a load-time guard, not serialized state: a checkpoint loaded for evaluation
        # refuses save() so a frozen run cannot overwrite the weights it is reading.
        self.read_only = bool(read_only)
        self.weights = list(weights or THETA0)
        if len(self.weights) != FEATURES:
            raise ValueError(f'RL weights must have {FEATURES} entries')
        self.learning_rate, self.temperature, self.ucb_alpha = learning_rate, temperature, ucb_alpha
        self.baseline_reward, self.episodes, self.pairwise_lr = baseline_reward, episodes, pairwise_lr
        self.action_counts = dict(action_counts or {})
        self.memory = memory or PromptMemoryBank()
        # ponytail: in-process only, not serialized; persist if record_recovery_outcome must survive restarts.
        self.pending = {}

    @staticmethod
    def action_key(candidate):
        return f"{agent(candidate['agent'])}@{candidate['step_index']}"

    def select_action(self, candidates, explore=False, trajectory_id=None):
        """Returns (top1, ranked candidates with probabilities, probabilities aligned to `candidates`)."""
        if not candidates:
            raise ValueError('No fault candidates to rank')
        total = sum(self.action_counts.values())
        logits = []
        for c in candidates:
            value = sum(w * h for w, h in zip(self.weights, c['features']))
            if explore:
                value += self.ucb_alpha * math.sqrt(math.log(total + 1) / (self.action_counts.get(self.action_key(c), 0) + 1))
            logits.append(value)
        top = max(z / self.temperature for z in logits)
        exps = [math.exp(z / self.temperature - top) for z in logits]
        probs = [e / sum(exps) for e in exps]
        order = sorted(range(len(candidates)), key=lambda i: (-probs[i], candidates[i]['offset']))
        ranked = [{**candidates[i], 'probability': probs[i], 'logit': logits[i], 'rank': rank + 1,
                   'candidate_index': i} for rank, i in enumerate(order)]
        if trajectory_id is not None:
            self.pending[str(trajectory_id)] = {'candidates': candidates, 'probs': probs, 'top1': order[0],
                                               'constraints_used': []}
        return ranked[0], ranked, probs

    def update_policy(self, chosen_idx, candidates, probs, reward):
        """REINFORCE: theta += lr * (R - baseline) * (h* - sum_j p_j h_j); baseline is an EMA of R."""
        reward = max(-1.0, min(1.0, float(reward)))
        chosen = candidates[chosen_idx]['features']
        expected = [sum(p * c['features'][j] for p, c in zip(probs, candidates)) for j in range(FEATURES)]
        gradient = [chosen[j] - expected[j] for j in range(FEATURES)]
        advantage = reward - self.baseline_reward
        self.weights = [w + self.learning_rate * advantage * g for w, g in zip(self.weights, gradient)]
        self.baseline_reward = 0.9 * self.baseline_reward + 0.1 * reward
        key = self.action_key(candidates[chosen_idx])
        self.action_counts[key] = self.action_counts.get(key, 0) + 1
        self.episodes += 1
        return {'reward': reward, 'Advantage': advantage, 'gradient': gradient, 'updated_weights': list(self.weights)}

    def record_recovery_outcome(self, trajectory_id, chosen_step=None, chosen_agent=None, accepted=None,
                                verifier_score=None, rollout_trace=None, *, verified=None, verifier_score_before=0.0,
                                verifier_score_after=None, stopped_early=False, failed_steps=()):
        """Cross-branch hook: observable verifier/acceptance outcome (never ScoreLedger gold) closes the loop.

        Feedback is either (accepted, verifier_score) or the verifier deltas (verified, verifier_score_before/after,
        stopped_early). chosen_step/chosen_agent name the repaired candidate; omitted, the top-1 action is credited.
        failed_steps are candidates tried and rejected before a rewarded one: each gets a pairwise ranking update
        theta += pairwise_lr * (h_chosen - h_failed), so the order in which hypotheses are tried is learned directly.
        rollout_trace is accepted for the orchestrator/RecoveryEnv contract; standalone mode does not inspect it.
        """
        if accepted is None and verified is None:
            raise ValueError('Pass accepted or verified')
        decision = self.pending.pop(str(trajectory_id), None)
        if decision is None:
            raise KeyError(f'No pending decision for trajectory {trajectory_id!r}; call select_action with trajectory_id')
        candidates, probs = decision['candidates'], decision['probs']
        if chosen_step is None:
            index = decision['top1']
        else:
            index = next((i for i, c in enumerate(candidates) if c['step_index'] == chosen_step
                          and (chosen_agent is None or agent(c['agent']) == agent(chosen_agent))), None)
        if index is None:
            raise ValueError('Recovery outcome does not match a ranked candidate')
        reward = (verifier_reward(verified, verifier_score_before, verifier_score_after, stopped_early)
                  if verified is not None else outcome_reward(accepted, verifier_score))
        telemetry = self.update_policy(index, candidates, probs, reward)
        telemetry['pairwise_steps'] = []
        for step in dict.fromkeys(failed_steps) if reward > 0 else ():
            j = next((i for i, c in enumerate(candidates) if c['step_index'] == step and i != index), None)
            if j is not None:
                self.weights = [w + self.pairwise_lr * (a - b) for w, a, b in
                                zip(self.weights, candidates[index]['features'], candidates[j]['features'])]
                telemetry['pairwise_steps'].append(step)
        telemetry['updated_weights'] = list(self.weights)
        telemetry['memory_scores'] = self.memory.record_episode_feedback(
            candidates[index]['dominant_perspective'], decision['constraints_used'], reward)
        return telemetry

    def to_dict(self):
        return {'version': 1, 'weights': list(self.weights), 'learning_rate': self.learning_rate,
                'temperature': self.temperature, 'ucb_alpha': self.ucb_alpha,
                'baseline_reward': self.baseline_reward, 'action_counts': dict(self.action_counts),
                'episodes': self.episodes, 'pairwise_lr': self.pairwise_lr, 'memory': self.memory.to_dict()}

    @classmethod
    def from_dict(cls, data):
        return cls(data['weights'], data.get('learning_rate', 0.05), data.get('temperature', 0.5),
                   data.get('ucb_alpha', 0.1), data.get('baseline_reward', 0.0), data.get('action_counts'),
                   PromptMemoryBank.from_dict(data.get('memory', {})), data.get('episodes', 0),
                   data.get('pairwise_lr', 0.15))

    def save(self, path):
        if self.read_only:
            raise RuntimeError(f'refusing to save a read-only RL checkpoint to {path} '
                               '(loaded for evaluation; train into a run directory instead)')
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path, read_only=False):
        judge = cls.from_dict(read_json(path)) if Path(path).exists() else cls()
        judge.read_only = bool(read_only)
        return judge


def _window(trajectory, steps):
    result = {k: v for k, v in trajectory.items() if k not in ('steps', 'index_map')}
    result['steps'] = steps
    return result


def _acting_substep(step, raw_agent):
    return next((e for e in step['substeps'] if e['role'] == raw_agent), step['substeps'][0])


def _layer1_reason(original, repaired, category):
    if not isinstance(repaired, dict) or not isinstance(repaired.get('content', ''), str):
        return 'repaired_substep must be an object with string content'
    calls = repaired.get('tool_calls') or []
    if not isinstance(calls, list) or any(not isinstance(c, dict) or not isinstance(c.get('name'), str) or not c['name']
                                          or not isinstance(c.get('arguments'), dict) for c in calls):
        return 'tool_calls must be a list of {"name": non-empty string, "arguments": object}'
    if original.get('tool_calls') and category == 'InvalidInvocation' and not calls:
        return 'Action-Structure Invariant: the original step required a tool call; emit corrected tool_calls, do not delete it'
    if not repaired.get('content', '').strip() and not calls:
        return 'Repaired step is empty'
    return None


def _sanitize(original, repaired):
    """Keep only observable fields, so no repair rationale leaks into the verifier's context."""
    result = {'sub_index': original['sub_index'], 'role': original['role'], 'content': repaired.get('content', '')}
    calls = repaired.get('tool_calls') or []
    if calls:
        result['tool_calls'] = [{'id': str(c.get('id', '')), 'name': c['name'], 'arguments': c['arguments']} for c in calls]
    return result


def _structural_repair(original):
    """Offline fallback: coerce malformed tool arguments into objects; otherwise keep the step."""
    result = {'sub_index': original['sub_index'], 'role': original['role'], 'content': original['content']}
    calls = []
    for call in original.get('tool_calls', []):
        args = call.get('arguments')
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        calls.append({'id': str(call.get('id', '')), 'name': call.get('name', ''),
                      'arguments': args if isinstance(args, dict) else {}})
    if calls:
        result['tool_calls'] = calls
    return result


def _sanitize_step(raw, index):
    if not isinstance(raw, dict) or not isinstance(raw.get('substeps'), list) or not raw['substeps']:
        raise ValueError('tool_sandbox must return next_step with a non-empty substeps list')
    substeps = []
    for i, event in enumerate(raw['substeps']):
        if not isinstance(event, dict) or not isinstance(event.get('role'), str) or not event['role']:
            raise ValueError('tool_sandbox substeps need a role')
        content = event.get('content', '')
        item = {'sub_index': event['sub_index'] if type(event.get('sub_index')) is int else i + 1,
                'role': event['role'],
                'content': content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)}
        if isinstance(event.get('name'), str) and event['name']:
            item['name'] = event['name']
        substeps.append(item)
    return {'index': index, 'substeps': substeps}


def live_dynamic_recover(trajectory, chosen_candidate, policy, tools, provider, memory=None, max_retries=2):
    """[C3] Regenerate step t* for agent k* on the prefix, gate it (Layer 1 + anti-repeat), then build (t*+1)' fresh."""
    steps = trajectory['steps']
    target = chosen_candidate.get('actor_offset', chosen_candidate['offset'])
    step = steps[target]
    original = _acting_substep(step, chosen_candidate['raw_agent'])
    perspective = chosen_candidate['dominant_perspective']
    category = chosen_candidate['dominant_category']
    forbidden = memory.retrieve_forbidden(perspective) if memory is not None else []
    seen, feedback, attempts = set(), [], []
    rung, repaired, reason, rationale, mode = 1, None, 'No repair attempt was made', '', 'live'
    for attempt in range(max_retries + 1):
        payload = {'trajectory': _window(trajectory, copy.deepcopy(steps[:target])), 'original_step': step,
                   'target': {'agent': chosen_candidate['raw_agent'], 'step_index': step['index'],
                              'category': category, 'perspective': perspective},
                   'violations': chosen_candidate.get('violated_checks', []), 'policy': policy, 'tools': tools,
                   'rung': rung, 'forbidden_patterns': forbidden}
        if feedback:
            payload['retry_feedback'] = feedback
        try:
            answer = provider.complete('perspective_repair', payload)
        except Exception as exc:
            attempts.append({'attempt': attempt, 'rung': rung, 'error': f'{type(exc).__name__}: {exc}'})
            if attempt == 0:  # Offline replay without repair fixtures: deterministic structural repair.
                mode, repaired = 'fallback', _structural_repair(original)
                reason = _layer1_reason(original, repaired, category)
            break
        candidate = answer.get('repaired_substep') if isinstance(answer, dict) else None
        key = digest(candidate)
        duplicate = key in seen or (memory is not None and key in memory.forbidden)
        reason = 'Duplicate of a rejected output (anti-repeat fingerprint)' if duplicate else \
            _layer1_reason(original, candidate, category)
        attempts.append({'attempt': attempt, 'rung': rung, 'fingerprint': key, 'rejected': reason})
        if reason is None:
            repaired = _sanitize(original, candidate)
            rationale = str(answer.get('repair_rationale', ''))
            break
        seen.add(key)
        if memory is not None:
            memory.record_forbidden(perspective, candidate, reason)
        feedback.append({'attempt': attempt, 'reason': reason,
                         'note': 'FORBIDDEN OUTPUT (DO NOT REPEAT): ' + _snippet(candidate)})
        if duplicate and rung == 2:
            break
        rung = 2 if duplicate or attempt + 1 >= 2 else 1
    layer1_passed = repaired is not None and reason is None
    if repaired is None:
        repaired = copy.deepcopy(original)
    repaired_step = {'index': step['index'],
                     'substeps': [copy.deepcopy(repaired if e is original else e) for e in step['substeps']]}
    next_step, source, next_error = None, 'none', None
    if layer1_passed and target + 1 < len(steps):
        following = steps[target + 1]
        if digest(_sanitize(original, original)) == digest(repaired):
            next_step, source = copy.deepcopy(following), 'original_unchanged'
        else:
            try:
                answer = provider.complete('tool_sandbox', {
                    'trajectory': _window(trajectory, copy.deepcopy(steps[:target]) + [repaired_step]),
                    'repaired_step': repaired_step, 'next_index': following['index'],
                    'expected_role': following['substeps'][0]['role'] if following['substeps'] else None,
                    'policy': policy, 'tools': tools})
                next_step, source = _sanitize_step(answer.get('next_step'), following['index']), 'tool_sandbox'
            except Exception as exc:  # Never reuse the stale response; the window just ends at t*'.
                source, next_error = 'unavailable', f'{type(exc).__name__}: {exc}'
    window = copy.deepcopy(steps[:target]) + [repaired_step] + ([next_step] if next_step else [])
    return {'target_offset': target, 'target_step_index': step['index'], 'agent': chosen_candidate['raw_agent'],
            'category': category, 'perspective': perspective, 'mode': mode, 'rung': rung, 'attempts': attempts,
            'layer1_passed': layer1_passed, 'layer1_reason': reason, 'seen_hashes': sorted(seen),
            'repaired_substep': repaired, 'repair_rationale': rationale, 'repaired_step': repaired_step,
            'next_step': next_step, 'next_step_source': source, 'next_step_error': next_error,
            'repaired_trajectory': _window(trajectory, window)}


def _viol(rows, perspective=None, kind=None):
    return sum(r['verdict'] == 'VIOL' and (perspective is None or r.get('perspective') == perspective)
               and (kind is None or r.get('check_type') == kind) for r in rows)


def verify_and_score_state_change(trajectory, checks, effective_constraints, recovery, chosen, candidates, probs,
                                  provider, context, rl_judge=None):
    """[C4] 3-layer split-role verification of s_{t+1}, then [C5] scaled reward and dual-loop update."""
    target, perspective = recovery['target_offset'], recovery['perspective']
    window = recovery['repaired_trajectory']
    length = len(window['steps'])
    before = [r for r in checks if r['offset'] < length]
    result = {'window_steps': [s['index'] for s in window['steps']], 'v_before': _viol(before)}
    if not recovery['layer1_passed']:
        result.update(layer1={'passed': False, 'reason': recovery['layer1_reason']},
                      layer2={'status': 'skipped'}, layer3={'status': 'skipped'},
                      v_after=None, delta_v=None, feedback='-', reward=-1.0)
    else:
        # The observed prefix before t*' is unchanged, so its verdicts are reused; only t*' and (t*+1)' are re-checked.
        after = [r for r in checks if r['offset'] < target]
        for pid in ORDER:
            for c in effective_constraints.get(pid, []):
                for offset in range(target, length):
                    row = check_constraint(c, window, offset, provider, context)
                    row.update(offset=offset, perspective=pid, origin=c.get('origin', 'llm'))
                    after.append(row)
        per = {pid: {'before': _viol(before, pid), 'after': _viol(after, pid)} for pid in ORDER}
        regressed = [pid for pid in ORDER if pid != perspective and per[pid]['after'] > per[pid]['before']]
        nl_before, nl_after = _viol(before, kind='nl_check'), _viol(after, kind='nl_check')
        v_before, v_after = _viol(before), _viol(after)
        delta = v_before - v_after
        intrinsic = max(-1.0, min(1.0, delta / max(v_before, 1)))
        if delta > 0 and not regressed:
            feedback, reward = '+', max(0.25, intrinsic)
        elif delta >= 0:
            feedback, reward = '=', -0.1  # includes a Layer 3 veto: total improved but another lens regressed
        else:
            feedback, reward = '-', min(-0.25, intrinsic)
        result.update(
            layer1={'passed': True, 'python_viol_before': _viol(before, kind='python_check'),
                    'python_viol_after': _viol(after, kind='python_check')},
            layer2={'nl_viol_before': nl_before, 'nl_viol_after': nl_after, 'passed': nl_after <= nl_before,
                    'context': 'repaired window only; no directive, rationale, or broken turn'},
            layer3={'per_perspective': per, 'target_perspective': perspective, 'regressed': regressed,
                    'passed': not regressed},
            v_after=v_after, delta_v=delta, feedback=feedback, reward=reward)
    if rl_judge is not None:
        names = {v['assertion_name'] for v in chosen.get('violated_checks', []) if v['perspective'] == perspective}
        used = [c for c in effective_constraints.get(perspective, [])
                if c.get('origin') == 'llm' and c['assertion_name'] in names]
        result['policy_update'] = rl_judge.update_policy(chosen['candidate_index'], candidates, probs, result['reward'])
        result['memory_scores'] = rl_judge.memory.record_episode_feedback(perspective, used, result['reward'])
        if result['reward'] < 0 and recovery['mode'] == 'live':
            rl_judge.memory.record_forbidden(perspective, recovery['repaired_substep'],
                                             f"Repair verified as {result['feedback']} (R={result['reward']:.2f})")
        rl_judge.pending.pop(str(trajectory['trajectory_id']), None)
    return result
