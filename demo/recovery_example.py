"""A separate, scripted decision illustration scored by the real EV policy."""
from recovery_orchestrator.policy import EVConfig, EVPolicy
from recovery_orchestrator.rca import Candidate, RCAPayload
from recovery_orchestrator.registry import Strategy, StrategyRegistry
from recovery_orchestrator.trace import Step, Trace
from recovery_orchestrator.orchestrator import Orchestrator
from recovery_orchestrator.replay import ScriptedEnvironment, TaskView
from recovery_orchestrator.verifier import FreshContextVerifier
from recovery_orchestrator.llm import ScriptedLLM
import json


def decision_example():
    category = 'MisinterpretationOfToolOutput'
    trace = Trace('example', 'retail', [Step(i, 'planner', 'Observed step') for i in range(1, 11)])
    candidates = [Candidate(6, 'planner', category, 1, .6), Candidate(8, 'planner', category, 1, .3)]
    payload = RCAPayload('example', 'retail', 'ok', candidates, 1, 1, 0)
    strategies = StrategyRegistry([
        Strategy('restore_context', 'Restore handoff context', 'reasoning_repair', frozenset({category}), 500, 0, .7),
        Strategy('reground', 'Reground from tool output', 'reasoning_repair', frozenset({category}), 800, 0, .5),
    ])
    class TrackingPolicy(EVPolicy):
        transition = None
        def observe_failure(self, option):
            before = self.belief(option.primary)
            super().observe_failure(option)
            self.transition = (before, self.belief(option.primary))

    policy = TrackingPolicy(strategies, EVConfig(), use_detectors=False)
    guard = policy.make_guard()
    options, _ = policy.options(payload, trace, guard)
    first = policy.decide(payload, trace, guard).option
    def replay(original, prefix, directive):
        content = ('Order identifier still missing.' if directive.strategy_id == 'restore_context'
                   else 'Order identifier restored: #W1000001.')
        return [Step(prefix.rollback_step, 'planner', content)], 0.0, 200
    def verdict(messages, purpose):
        transcript = messages[-1]['content'].split('## Transcript\n', 1)[-1]
        passed = 'Order identifier restored: #W1000001.' in transcript
        return json.dumps({'verdict': 'PASS' if passed else 'FAIL', 'score': .9 if passed else .2,
                           'gaps': [] if passed else ['Missing order identifier']})
    task = TaskView('example', 'retail', 'Restore order identifier #W1000001.', 'Use identifiers in the transcript.')
    env = ScriptedEnvironment({'example': task}, replay)
    verifier = FreshContextVerifier(ScriptedLLM(verdict))
    outcome = Orchestrator(env, verifier, policy, verify_original=False, monitor_loops=False).run(trace, payload)
    if len(outcome.attempts) < 2 or policy.transition is None:
        raise RuntimeError('The scripted retry example did not reach its second attempt.')
    return {'options': [o.describe() for o in options[:3]], 'chosen': first.describe(),
            'belief_before': policy.transition[0], 'belief_after': policy.transition[1],
            'first_verdict': outcome.attempts[0].verifier['verdict'],
            'second_verdict': outcome.attempts[1].verifier['verdict'],
            'next': outcome.attempts[1].option, 'next_action': 'intervene'}



def example_html():
    data = decision_example()
    options = ''.join(
        f'<li class="ev-option {"chosen" if row == data["chosen"] else ""}">'
        f'<strong>Step {row["rollback_step"]} · {row["strategy"].replace("_", " ")}</strong>'
        f'<span>Estimated success {row["p_success"]:.3f} − cost {row["cost"]:.3f} = EV {row["ev"]:.3f}</span>'
        f'{"<em>Selected</em>" if row == data["chosen"] else ""}</li>' for row in data['options'])
    nxt = data['next']
    return f'''<section class="decision-example panel" aria-label="Failed repair decision example">
      <div class="decision-head"><div><span class="eyebrow-label">SEPARATE SCRIPTED SCENARIO / REAL EV POLICY</span><h3>What if the first repair fails?</h3></div><span class="pill">Stand-in verifier · no model call</span></div>
      <p>Two hypothetical fault candidates and fixed strategy priors feed the real expected-value policy. This teaching example does not control the retail replay above.</p>
      <p class="ev-formula">EV = Σ (candidate belief × strategy prior × 0.97<sup>distance</sup>) − estimated cost</p>
      <ol class="ev-options">{options}</ol>
      <div class="example-ladder"><strong>Illustrated repair ladder</strong><span class="ladder-current">1 · Restore handoff context — selected</span><span class="ladder-next">2 · Reground from tool output — next</span></div>
      <p class="example-correction"><strong>Correction:</strong> Redo step 6 with order_id = #W1000001.</p>
      <button id="example-fail" class="button" type="button">Show failed first repair →</button>
      <div id="example-result" class="example-result" hidden><strong>Stand-in verifier: FAIL</strong><p>The scripted first replay did not provide the required evidence. The system does not declare success. After the failure, the real policy lowers the primary candidate’s belief from {data['belief_before']:.2f} to {data['belief_after']:.2f}, then selects step {nxt['rollback_step']} · {nxt['strategy'].replace('_', ' ')} (EV {nxt['ev']:.3f}). The second scripted replay receives a {data['second_verdict']} from the stand-in.</p><p>Its verifier is given the task, policy and transcript—not the repair, its rationale or strategy. The first replay and verifier are deterministic stand-ins, not a model judgment.</p></div>
    </section>'''
