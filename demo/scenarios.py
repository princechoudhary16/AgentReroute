"""Four curated live runs of the 4-agent retail MAS, recovered by the real MPRLRecoveryPolicy.

Each scenario runs ``Workflow`` (scripted agents, offline fixture) with a real fault operator, hands the
observable trace to ``MPRLRecoveryPolicy`` through a ``RecoveryEnv``, and turns everything that happened into
an ordered list of propagation frames for the visualizer:

  initial_run      the original execution, one frame per dialog / tool call / handoff / reply
  debugging        AgentReroute milestones I-III (checks 25%, ranking 50%, rollback 75%)
  recovery_replay  the delivered rollout from its rollback step onward
  verified         milestone IV (scripted replay acceptance, frozen weights, 100%)

The policy only ever sees ``env.trace`` and ``env.task_view``. Fault records (``Step.faults``) are gold-side:
the frames show them so a viewer can see where the fault was injected, but they never reach the policy.
"""
import json

from agentrx.perspectives import PERSPECTIVES
from agentrx.rl_cross_branch import MPRLRecoveryPolicy
from agentrx.rl_policy import DualLoopRLJudge
from benchmark.env.retail import TOOL_SCHEMAS
from benchmark.env.scripted import scripted_llm
from benchmark.env.tasks import load_fixture
from benchmark.env.workflow import AGENTS, Workflow, export_trace
from benchmark.faults.operators import DropContext, Misroute, SkipWrite
from benchmark.scoring.recovery import BudgetCaps, RecoveryEnv

MAX_STEPS = 20      # keeps the unrepaired Case I loop short enough to watch

SCENARIOS = {
    'handoff_drop_context': {
        'title': 'Case I · Handoff Context Drop & Guided Rollback', 'task_id': 'fixture-cancel',
        'operator': (DropContext, {'variant': 'drop_ids'}),
        'summary': 'The planner authenticates the user, then hands off to order_agent without the order id. '
                   'AgentReroute blames the handoff (context_propagation) and replays it with guidance.'},
    'misroute_specialist': {
        'title': 'Case II · Misrouted Specialist Handoff & Reroute', 'task_id': 'fixture-cancel',
        'operator': (Misroute, {'persistence': 'once'}),
        'summary': 'The planner routes an order cancellation to product_agent. AgentReroute flags the '
                   'misrouted handoff (drift) and reroutes it to order_agent.'},
    'skip_write_claim': {
        'title': 'Case III · Unbacked Write Claim & Forced Tool Call', 'task_id': 'fixture-return',
        'operator': (SkipWrite, {}),
        'summary': 'The specialist claims the return was done without calling the write tool. AgentReroute '
                   'blames the specialist step (instruction_adherence) and forces the real write.'},
    'clean_control_flow': {
        'title': 'Case IV · Healthy Multi-Agent Control Flow (Abstained)', 'task_id': 'fixture-cancel',
        'operator': None,
        'summary': 'A clean run: user -> planner -> handoff -> specialist -> user. AgentReroute checks every '
                   'step, finds no violation and abstains.'},
}

MILESTONES = ['I. P1/P2/P3 Attribution Checks', 'II. Fixed-Check Candidate Ranking',
              'III. Rollback & Directive Replay', 'IV. Replay Acceptance · Frozen Policy']


class CapturingJudge(DualLoopRLJudge):
    """Keeps the last ranked candidate list (with violated checks) so the demo can show milestones I and II."""
    ranked = ()

    def select_action(self, candidates, explore=False, trajectory_id=None):
        top1, ranked, probs = super().select_action(candidates, explore, trajectory_id)
        self.ranked = ranked
        return top1, ranked, probs


# ------------------------------------------------------------------------------ frame helpers
def _short(value, limit=28):
    text = "'" + value + "'" if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit - 1] + '…'


def popper_text(name, arguments):
    """``tool(args)`` label for a tool mini-block popper, e.g. find_user_id_by_email('a@b.com')."""
    if len(arguments) == 1:
        inner = _short(next(iter(arguments.values())), 40)
    else:
        inner = ', '.join(f'{k}={_short(v)}' for k, v in arguments.items())
    return f'{name}({inner[:72]})'


def _frame(phase, **fields):
    pct = {'initial_run': 0, 'recovery_replay': 75}.get(phase, 0)
    frame = {'phase': phase, 'step_index': None, 'actor': None, 'kind': None, 'status': 'NORMAL', 'title': '',
             'narrative': '', 'edge': None, 'focus': None, 'tool': None, 'handoff': None, 'fault': None,
             'payload': {}, 'debug_pct': pct, 'active_phase': 3 if phase == 'recovery_replay' else 0}
    frame.update(fields)
    return frame


def _fault(step):
    return {'operator': step.faults[0]['operator'], 'mode': step.faults[0]['mode'],
            'note': 'injected by the benchmark; hidden from AgentReroute'} if step.faults else None


def step_frames(run, phase, start=1):
    """Frames for ``run.steps`` from step ``start`` on; an agent tool call and its result form one frame."""
    frames = []
    steps = run.steps
    by_index = {s.index: s for s in steps}
    for s in steps:
        if s.index < start or s.kind == 'tool':
            continue
        fault = _fault(s)
        base = {'step_index': s.index, 'actor': s.actor, 'fault': fault}
        good = 'RECOVERED' if phase == 'recovery_replay' else 'NORMAL'
        if s.kind == 'user':
            frames.append(_frame(phase, **base, kind='dialog', edge=['user', 'planner'], focus='planner',
                                 status=good, title='User → planner', narrative=s.content,
                                 payload={'role': 'user', 'content': s.content}))
            continue
        action = s.action
        if action is None:
            frames.append(_frame(phase, **base, kind='invalid', focus=s.actor, status='FAULT',
                                 title=f'{s.actor} · invalid action', narrative=s.content[:200],
                                 payload={'raw': s.content[:400]}))
            continue
        if action['action'] == 'tool':
            result = by_index.get(s.index + 1)
            content = result.content if result is not None and result.kind == 'tool' else ''
            name, args = action['tool'], action.get('arguments', {})
            error = content.startswith('Error')
            tool = {'name': name, 'agent': s.actor, 'arguments': args, 'result_preview': content[:180],
                    'popper_text': popper_text(name, args), 'error': error}
            frames.append(_frame(phase, **base, kind='tool', edge=[s.actor, 'tool:' + name], focus=s.actor,
                                 tool=tool, status='FAULT' if error or fault else good,
                                 title=f'{s.actor} → {name}', narrative=f'{s.actor} calls {tool["popper_text"]}',
                                 payload={'action': 'tool', 'tool': name, 'arguments': args,
                                          'result': content[:600]}))
        elif action['action'] == 'handoff':
            handoff = {'from': 'planner', 'to': action['to'], 'message': action.get('message', ''),
                       'context': action.get('context', {}), 'fault': bool(fault)}
            frames.append(_frame(phase, **base, kind='handoff', edge=['planner', action['to']], focus=action['to'],
                                 handoff=handoff, status='FAULT' if fault else good,
                                 title=f'Handoff planner → {action["to"]}', narrative=handoff['message'],
                                 payload={'action': 'handoff', 'to': action['to'], 'message': handoff['message'],
                                          'context': handoff['context']}))
        else:
            target = 'user' if s.actor == 'planner' else 'planner'
            frames.append(_frame(phase, **base, kind='reply', edge=[s.actor, target], focus=target,
                                 status='FAULT' if fault else good, title=f'{s.actor} → {target}',
                                 narrative=action['content'], payload={'action': 'reply',
                                                                       'content': action['content']}))
    return frames


# ------------------------------------------------------------------------------ scenario runner
def _task_view(trace, domain):
    """What a recovery policy may read: first user message, policy, agents and tool schemas (no gold)."""
    return {'first_user_message': trace['instruction'], 'policy': domain.policy, 'agents': sorted(AGENTS),
            'tools': {a: [TOOL_SCHEMAS[t] for t in spec['tools']] for a, spec in AGENTS.items()}}


def run_scenario(scenario_id):
    """Run one curated scenario end to end; returns its metadata, the policy's decision and its frames."""
    spec = SCENARIOS[scenario_id]
    domain = load_fixture()
    task = domain.task(spec['task_id'])
    runs = {}
    tid = f'demo-case-{list(SCENARIOS).index(scenario_id) + 1}'   # neutral: the scenario id names the fault

    def execute(directive=None):
        faults = [spec['operator'][0](**spec['operator'][1])] if spec['operator'] else []
        run = Workflow(domain, task, scripted_llm(domain, task), 'scripted', faults=faults, directive=directive,
                       max_steps=MAX_STEPS).run()
        first_user = next((s.content for s in run.steps if s.kind == 'user'), '')
        return run, export_trace(run, tid, domain.name, first_user, 'full')

    def runner(rollout_id, directive):
        run, trace = execute(directive)
        runs[rollout_id] = run
        return run, trace

    original, trace = execute()
    env = RecoveryEnv(tid, trace, _task_view(trace, domain), runner, BudgetCaps().for_episode(original.tokens))
    judge = CapturingJudge(read_only=True)
    policy = MPRLRecoveryPolicy(rl_judge=judge, learn=False)
    decision = policy.recover(env)
    delivered = decision.get('delivered')
    frames = step_frames(original, 'initial_run')
    frames += _debug_frames(policy, judge, env, decision, tid)
    if delivered:
        rollback = env.rollouts[delivered].directive['rollback_step']
        replay = step_frames(runs[delivered], 'recovery_replay', start=rollback)
        if replay:
            replay[0]['directive_applied'] = True
            replay[0]['narrative'] = 'Supervisor note applied. ' + replay[0]['narrative']
        frames += replay
    frames.append(_verify_frame(policy, judge, decision, tid))
    for seq, frame in enumerate(frames, 1):
        frame['seq'] = seq
    op = spec['operator']
    return {'id': scenario_id, 'measurement_kind': 'scripted_illustration', 'trajectory_id': tid, 'title': spec['title'], 'summary': spec['summary'],
            'task_id': spec['task_id'], 'operator': {'name': op[0].name, 'mode': op[0].mode, 'params': op[1]} if op else None,
            'original_termination': original.termination, 'decision': decision,
            'hypotheses': policy.hypotheses.get(tid, []), 'rl': policy.rl_log.get(tid),
            'agents': {a: list(s['tools']) for a, s in AGENTS.items()}, 'frames': frames}


def _checks_by_perspective(ranked):
    fired = {p: [] for p in PERSPECTIVES}
    for c in ranked:
        for v in c.get('violated_checks', ()):
            entry = {'check': v['assertion_name'], 'step_index': v['step_index'], 'agent': c['raw_agent']}
            if entry not in fired.setdefault(v['perspective'], []):
                fired[v['perspective']].append(entry)
    return fired


def _debug_frames(policy, judge, env, decision, tid):
    fired = _checks_by_perspective(judge.ranked)
    total = sum(len(v) for v in fired.values())
    lines = [f'{p}: ' + (', '.join(f"{e['check']} @ step {e['step_index']}" for e in v) or 'clear')
             for p, v in fired.items()]
    debug = {'phase': 'debugging', 'kind': 'debug', 'status': 'DEBUG'}
    frames = [_frame(**debug, debug_pct=25, active_phase=1, title=MILESTONES[0],
                     narrative=f'{total} violation(s) across P1/P2/P3. ' + ' · '.join(lines),
                     payload={'milestone': MILESTONES[0], 'violations': total, 'checks_fired': fired})]
    hypotheses = policy.hypotheses.get(tid, [])
    ranking = [{'agent': c['raw_agent'], 'step_index': c['step_index'], 'perspective': c['dominant_perspective'],
                'probability': round(c['probability'], 4), 'violated': bool(c['violated_checks'])}
               for c in list(judge.ranked)[:5]]
    if hypotheses:
        top = hypotheses[0]
        frames.append(_frame(**debug, debug_pct=50, active_phase=2, title=MILESTONES[1], focus=top['agent'],
                             step_index=top['step_index'], actor=top['agent'],
                             narrative=f"Top-1 blame: {top['agent']} at step {top['step_index']} "
                                       f"({top['mode']}, uncalibrated policy score {top['confidence']:.3f}).",
                             payload={'milestone': MILESTONES[1], 'top1': {k: top[k] for k in (
                                 'agent', 'step_index', 'mode', 'category', 'confidence')},
                                      'hypotheses': hypotheses, 'ranking': ranking}))
    else:
        frames.append(_frame(**debug, debug_pct=50, active_phase=2, title=MILESTONES[1],
                             narrative='No candidate carries a violated check: top-1 blame withheld.',
                             payload={'milestone': MILESTONES[1], 'top1': None, 'ranking': ranking}))
    attempts = policy.log[tid].attempts if tid in policy.log else []
    delivered = decision.get('delivered')
    if delivered:
        directive = env.rollouts[delivered].directive
        accepted = next((a for a in attempts if a.get('rollout') == delivered), {})
        actor = hypotheses[0]['agent'] if hypotheses else None
        frames.append(_frame(**debug, debug_pct=75, active_phase=3, title=MILESTONES[2], focus=actor,
                             step_index=directive['rollback_step'],
                             narrative=f"Rollback to step {directive['rollback_step']} with a {directive['kind']} "
                                       f"directive ({accepted.get('strategy', 'repair')}); the prefix is replayed "
                                       'from the same scripted simulator. No model is called.',
                             payload={'milestone': MILESTONES[2], 'rollback_step': directive['rollback_step'],
                                      'directive_kind': directive['kind'], 'directive': directive['text'],
                                      'strategy': accepted.get('strategy'), 'attempts': attempts}))
    else:
        frames.append(_frame(**debug, debug_pct=75, active_phase=3, title=MILESTONES[2],
                             narrative=f"No rollback: {decision.get('status')} ({decision.get('notes', '')}).",
                             payload={'milestone': MILESTONES[2], 'rollback_step': None, 'decision': decision}))
    return frames


def _verify_frame(policy, judge, decision, tid):
    rl = policy.rl_log.get(tid)
    theta = [round(w, 4) for w in judge.weights]
    if decision.get('status') == 'abstained' and not policy.hypotheses.get(tid):
        return _frame('verified', kind='verify', status='RECOVERED', debug_pct=100, active_phase=4,
                      title='IV. Clean Run — Abstained', verified=True, reward=None,
                      narrative='No check flagged this example, so the illustration takes no action '
                                'and leaves θ unchanged.',
                      payload={'milestone': MILESTONES[3], 'verified': True, 'abstained': True, 'reward': None,
                               'theta': theta})
    attempts = policy.log[tid].attempts if tid in policy.log else []
    verified = any(a.get('accepted') for a in attempts)
    reward = None
    return _frame('verified', kind='verify', status='RECOVERED' if verified else 'FAULT', debug_pct=100,
                  active_phase=4, title=MILESTONES[3], verified=verified, reward=reward,
                  narrative=('The scripted replay passed the repair acceptance checks. Policy weights stay frozen; '
                             'this illustration is not a published benchmark result.' if verified else
                             'No scripted replay passed acceptance. Policy weights stay frozen.'),
                  payload={'milestone': MILESTONES[3], 'verified': verified, 'reward': reward, 'rl': rl,
                           'decision': decision, 'theta': theta})


def scenario_meta():
    return [{'id': k, 'title': v['title'], 'summary': v['summary'], 'task_id': v['task_id'],
             'operator': v['operator'][0].name if v['operator'] else None} for k, v in SCENARIOS.items()]


def all_scenarios(ids=None):
    return [run_scenario(sid) for sid in (ids or SCENARIOS)]
