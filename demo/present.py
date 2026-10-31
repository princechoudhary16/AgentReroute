"""The presentation route: one scroll, six chapters, eighteen beats.

Same evidence as the home page (story_data validates the committed CSVs), ordered the way the talk is
given. Seven beats are scroll-driven scenes: a pinned stage whose state follows how far into the
section the viewer has scrolled. Everything that carries a number stays still.
"""
import json
from html import escape

from .present_css import CSS
from .present_data import (CHAPTERS, CONTROLS, CORPUS, DISTANCE_DISCOUNT, HARM_P, ORCHESTRATOR_ABSTAINED,
                           ORCHESTRATOR_REJECTED, RECOVERY, RECOVERY_EPISODES, REPAIR_CONFIRMED_CLAIMS, RQ2_P,
                           STRATEGIES, TRAINING, VERIFIER_CONFIRMED_CLAIMS)
from .story_data import DOMAINS, FINAL_ROWS, HEADLINE, SELECTOR_ROWS, paired, summarize
from .studio_theme import MARK

BEATS_PER_CHAPTER = (3, 3, 4, 3, 3, 2)


# ------------------------------------------------------------------------------ building blocks
def beat(chapter, index, body, *, scene=None, length=3):
    """One full-viewport beat. A scene is ``length`` viewports tall with a pinned stage and a progress value."""
    scene_attrs = f' data-scene="{scene}" style="--len:{length}"' if scene else ''
    cls = 'beat scene' if scene else 'beat'
    return (f'<section class="{cls}" id="b{chapter}-{index}" data-chapter="{chapter}" data-beat="{index}"{scene_attrs}>'
            f'<div class="stage">{body}</div></section>')


def head(overline, title, *, tone='', lead='', width='min(900px,70vw)'):
    lead_html = f'<p class="lead rv" style="--d:2;max-width:640px;margin-top:18px">{lead}</p>' if lead else ''
    return (f'<div style="max-width:{width}"><div class="ol {tone} rv">{overline}</div>'
            f'<h2 class="h2 rv" style="--d:1;margin-top:14px">{title}</h2>{lead_html}</div>')


def foot(source, hint='→ next beat'):
    return f'<div class="foot"><span class="src">{source}</span><span class="hint">{hint}</span></div>'


def pct(value):
    return f'{value:.1f}%'


# ------------------------------------------------------------------------------ 01 introduction
def chapter_intro():
    title = beat(0, 1, f'''
<svg class="title-art" viewBox="0 0 1600 900" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
<path d="M-20 700 C 200 700, 260 560, 480 560 S 760 700, 980 700 S 1260 420, 1620 420" stroke="#1f2d33" stroke-width="2" fill="none"/>
<path d="M-20 700 C 200 700, 260 560, 480 560 S 760 700, 980 700" stroke="#bcf783" stroke-width="2" fill="none" stroke-dasharray="6 10" opacity=".8"/>
<circle cx="980" cy="700" r="6" fill="#bcf783"/><circle cx="980" cy="700" r="18" fill="none" stroke="#bcf783" opacity=".35"/><circle cx="1280" cy="520" r="5" fill="#ff9b9b"/>
</svg>
<div style="margin-top:12vh;position:relative;max-width:1200px" class="stack">
 <div class="ol rv">AI6101 · Introduction to AI and AI Ethics · Final presentation</div>
 <h1 class="h1 rv" style="--d:1">A wrong turn.<br><em>A trail of evidence.</em></h1>
 <p class="lead rv" style="--d:2;max-width:760px;font-size:clamp(17px,1.5vw,24px)">When a team of AI agents fails, which decision caused it? And who should be held to account?</p>
</div>
<div class="row rv" style="--d:3;margin-top:auto;position:relative"><span style="font-size:22px;font-weight:600;letter-spacing:-.02em">AgentReroute</span><span class="src" style="letter-spacing:1px;font-size:12px">William · Badr · Nischit · Prince · Ananya</span></div>
{foot('home › hero copy and route art', '→ or Space · next beat')}''')

    steps = ''.join(f'<div class="step" data-i="{i}"><b>{i}</b><i></i></div>' for i in range(1, 36))
    wrong_turn = beat(0, 2, f'''
{head('The problem', '<span class="hl" id="hl"><span class="a">The failure you see<br>is not the mistake that was made.</span><span class="b">Find the mistake,<br>not the symptom.</span></span>')}
<p class="lead rv" style="--d:2;max-width:620px;margin-top:16px">A planner misreads an order status at step 21. Nothing looks wrong until the cancellation fails at step 33. Every step in between inherits the mistake.</p>
<div class="trace-wrap" aria-label="A trace of 35 steps; step 21 is the cause, step 33 the visible failure">
 <div class="trace-rail" id="trace-rail">{steps}<div class="tag fail" id="tag-fail">THE SYMPTOM · cancellation fails</div><div class="tag cause" id="tag-cause">THE CAUSE · planner misread the order status</div></div>
</div>
{foot('scene · the rail pans to the symptom, then reverses to the cause', '→ next beat')}''', scene='trace', length=3)

    questions = beat(0, 3, f'''
{head('Why this is an ethics problem', 'An automated repair can cancel an order.<br>Agents report success without checking.', width='min(1200px,80vw)')}
<div class="grid3" style="margin-top:5vh">
 <div class="card rv" style="--d:2;padding:26px"><div class="ol mint">01 / Accountability</div><h3 class="h3" style="margin:14px 0 10px">Which agent was responsible?</h3><p style="font-size:14px;line-height:1.7;color:var(--muted)">Follow the handoff back to the decision that changed the outcome. Name the step, the agent and the obligation it broke.</p></div>
 <div class="card rv" style="--d:3;padding:26px"><div class="ol mint">02 / Honest reporting</div><h3 class="h3" style="margin:14px 0 10px">Did the repair really work?</h3><p style="font-size:14px;line-height:1.7;color:var(--muted)">A system’s own judgement of its success cannot be trusted. Success is declared only by something that cannot see the repair.</p></div>
 <div class="card rv" style="--d:4;padding:26px"><div class="ol mint">03 / Do no harm</div><h3 class="h3" style="margin:14px 0 10px">Leave healthy runs alone.</h3><p style="font-size:14px;line-height:1.7;color:var(--muted)">A policy that fixes failures but breaks runs that were fine has not improved anything. Measure harm, not just recoveries.</p></div>
</div>
<div class="grid3 rv" style="--d:5;margin-top:4vh;gap:clamp(20px,3vw,44px)">
 <div class="q"><b>RQ1</b><span>Does checking each step from several perspectives find the faulty step better than a single judge?<small>ANSWERED IN CH 03 · NISCHIT</small></span></div>
 <div class="q"><b>RQ2</b><span>Does a better diagnosis lead to more recoveries under the same repair machinery?<small>ANSWERED IN CH 05 · ANANYA</small></span></div>
 <div class="q"><b>RQ3</b><span>Can recovery be verified, budgeted and harmless to healthy runs?<small>ANSWERED IN CH 04 + 05 · PRINCE, ANANYA</small></span></div>
</div>
{foot('home › the three idea cards, relabelled to the ethics triad', '→ hand-off: Badr')}''')
    return title + wrong_turn + questions


# ------------------------------------------------------------------------------ 02 existing work
def chapter_existing():
    rx = {d: summarize(FINAL_ROWS, d, 'agentrx') for d in DOMAINS}
    agentrx = beat(1, 1, f'''
<div class="grid2">
 <div>
  {head('The baseline · Microsoft', 'AgentRx. One judge<br>reads the whole trace.', tone='violet', lead='It turns the system’s policy and tools into constraints, checks the trace against them, and asks one language model for the critical step: the earliest agent fault in the chain. Its output is a bare verdict.')}
  <div class="flow rv" style="--d:3;margin-top:4vh;max-width:520px" aria-label="AgentRx pipeline: trace, constraints, one judge, verdict">
   <div class="box"><div class="ol" style="font-size:9px">Input</div><strong>Recorded trajectory</strong></div><div class="arrow" aria-hidden="true">↓</div>
   <div class="box"><div class="ol" style="font-size:9px">Generate · check</div><strong>Constraints from policy and tools</strong></div><div class="arrow" aria-hidden="true">↓</div>
   <div class="judge"><div class="ol violet" style="font-size:9px">Joint attribution</div><strong>“Where did this go wrong?”</strong><p>One model, the whole trace, every kind of error at once.</p></div><div class="arrow" aria-hidden="true">↓</div>
   <div class="box" style="border-color:#4a3f78"><div class="ol" style="font-size:9px">Output</div><strong>Critical step + agent + explanation</strong></div>
  </div>
 </div>
 <div class="stack" style="padding-top:12vh;gap:34px">
  <div class="row rv" style="--d:2">
   <div class="stat"><b style="color:var(--violet)">{pct(rx['tau']['accuracy'])}</b><span>exact-step accuracy · tau-bench retail · {rx['tau']['hits']} / {rx['tau']['n']}</span></div>
   <div class="stat"><b style="color:var(--violet)">{pct(rx['magentic']['accuracy'])}</b><span>Magentic-One · {rx['magentic']['hits']} / {rx['magentic']['n']}</span></div>
  </div>
  <div class="rv" style="--d:3;border-top:1px solid var(--line);padding-top:24px;font-size:clamp(15px,1.2vw,18px);line-height:1.65">We reproduced both figures to within <span class="mono" style="color:var(--mint)">0.1 points</span>. That reproduction is the control for every comparison that follows.</div>
  <p class="rv" style="--d:4;font-size:14px;line-height:1.7;color:var(--muted)">What the verdict lacks: the obligation that was broken. It names a step without saying why, so nothing downstream can act on the diagnosis.</p>
 </div>
</div>
{foot('home › the method › AgentRx card · reproduced figures from results/final-evidence.csv')}''')

    gap = beat(1, 2, f'''
<div class="grid2">
 <div>
  {head('Two insights behind AgentReroute', 'Failures are silent,<br>and they come in three kinds.')}
  <div class="stack" style="margin-top:3vh;gap:0">
   <div class="insight rv" style="--d:2"><b>01</b><div><strong>Silent failures</strong><p>A retail agent fails to place an order because the previous step never supplied the order id. The visible failure is downstream of the cause.</p></div></div>
   <div class="insight rv" style="--d:3"><b>02</b><div><strong>Three kinds of failure</strong><p>Every agent failure lands in one of three bins. Each bin gets its own reviewer later.</p></div></div>
  </div>
  <div class="grid3 rv" style="--d:4;gap:10px;margin-top:10px">
   <div class="lens a"><div class="ol">Intent</div><strong>Did this step serve the goal?</strong><small>Plan and policy</small></div>
   <div class="lens b"><div class="ol">Tool execution</div><strong>Did the call behave?</strong><small>Well formed, succeeded, read right</small></div>
   <div class="lens c"><div class="ol">Communication</div><strong>Were the facts handed over?</strong><small>Handoffs and grounding</small></div>
  </div>
 </div>
 <div>
  <div class="ol rv">The gap</div>
  <div class="h3 rv" style="--d:1;font-size:clamp(22px,2vw,30px);margin-top:14px">Two lines of work exist,<br>and they do not meet.</div>
  <div class="gapviz" id="gapviz" aria-label="Attribution research stops at the diagnosis; recovery research assumes the fault is known; AgentReroute sits in the gap">
   <div class="line l"></div><div class="lab" style="left:0;top:80px;color:var(--violet)">ATTRIBUTION WORK · AGENTRX</div><div class="lab" style="left:0;top:136px">stops at the diagnosis</div>
   <div class="line r"></div><div class="lab" style="right:0;top:80px;color:var(--cyan);text-align:right">RECOVERY WORK</div><div class="lab" style="right:0;top:136px;text-align:right">assumes the fault is known</div>
   <div class="mark"><strong>AgentReroute</strong><p>an uncertain diagnosis, connected to a repair that is verified, budgeted and checked for harm</p></div>
  </div>
 </div>
</div>
{foot('home › the method › lens cards and approach note · scene: the two lines slide in and stop short')}''', scene='gap', length=2)

    network = beat(1, 3, f'''
<div style="max-width:40%;position:relative;z-index:2">
 {head('Our sandbox', 'Four agents.<br>One connected system.', lead='A planner and three specialists for orders, products and returns, each with real tools. We inject faults into this workflow and watch the diagnosis and the recovery happen step by step.', width='100%')}
 <div class="row rv" style="--d:3;margin-top:4vh;gap:30px">
  <div><div class="big">4</div><div class="src" style="margin-top:6px">AGENTS</div></div>
  <div><div class="big">3</div><div class="src" style="margin-top:6px">PERSPECTIVES</div></div>
  <div><div class="big">4</div><div class="src" style="margin-top:6px">SCENARIOS</div></div>
 </div>
</div>
<div class="net-wrap" aria-label="The agent network: a planner connected to order, product and returns specialists">
 <div class="net" id="net">
  <svg viewBox="0 0 900 844" width="900" height="844" aria-hidden="true" style="position:absolute;inset:0">
   <path d="M190 400 C 300 400, 330 180, 470 180" fill="none" stroke="#2b3a42" stroke-width="2"/>
   <path id="sig-path" d="M190 400 C 300 400, 330 400, 470 400" fill="none" stroke="#2b3a42" stroke-width="2"/>
   <path class="signal" d="M190 400 C 300 400, 330 400, 470 400" fill="none" stroke="#bcf783" stroke-width="2" stroke-dasharray="8 12"/>
   <path d="M190 400 C 300 400, 330 620, 470 620" fill="none" stroke="#2b3a42" stroke-width="2"/>
  </svg>
  <div class="signal-dot" style="position:absolute;width:10px;height:10px;border-radius:50%;background:#bcf783;box-shadow:0 0 14px #bcf783;offset-path:path('M190 400 C 300 400, 330 400, 470 400');offset-distance:0"></div>
  <div class="node p" style="left:0;top:350px"><div class="ol mint">Planner</div><strong>planner</strong><small>routes, authenticates, hands off</small></div>
  <div class="node spec" style="left:470px;top:130px"><div class="ol">Order operations</div><strong>order_agent</strong><small>8 tools</small><div class="tools" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div></div>
  <div class="node spec" style="left:470px;top:350px"><div class="ol">Product catalog</div><strong>product_agent</strong><small>5 tools</small><div class="tools" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i></div></div>
  <div class="node spec" style="left:470px;top:570px"><div class="ol">Returns</div><strong>returns_agent</strong><small>6 tools</small><div class="tools" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i></div></div>
 </div>
</div>
{foot('/system › the agent network · scene: the network grows out of one dot', '→ hand-off: Nischit')}''', scene='network', length=3)
    return agentrx + gap + network


# ------------------------------------------------------------------------------ 03 fault attribution
def chapter_attribution():
    mini = ''.join(f'<div class="step" data-i="{i}">{i}</div>' for i in range(1, 29))
    check = beat(2, 1, f'''
<div class="grid2" style="align-items:end">
 {head('01 / Check', 'One recorded trace.<br>Three sets of questions.')}
 <p class="lead rv" style="--d:2">Each reviewer looks for one kind of mistake. Each check sees the trace only up to the step it judges, so a verdict can never lean on what happened later.</p>
</div>
<div class="lens-track" id="lens-track">
 <div class="lens a" data-k="0"><div class="ol">Intent &amp; plan</div><strong>Did this step serve the request, within policy?</strong></div>
 <div class="lens b" data-k="1"><div class="ol">Tool execution</div><strong>Was the call well formed, did it succeed, was it read right?</strong></div>
 <div class="lens c" data-k="2"><div class="ol">Handoffs &amp; grounding</div><strong>Were the facts passed between agents actually observed?</strong></div>
</div>
<div class="trace-mini" id="trace-mini" aria-label="A 28-step trace; steps light up as the lenses pass">{mini}<div class="edge" id="edge" style="left:0"><span>PREFIX EDGE · nothing right of this line is visible to a check</span></div></div>
<div class="vios" id="vios">
 <div class="vio" data-step="5" style="color:var(--mint)"><i style="background:var(--mint)"></i>step 5 · intent</div>
 <div class="vio" data-step="10" style="color:var(--amber)"><i style="background:var(--amber)"></i>step 10 · handoff</div>
 <div class="vio" data-step="16" style="color:var(--cyan)"><i style="background:var(--cyan)"></i>step 16 · tool</div>
 <div class="vio" data-step="24" style="color:var(--amber)"><i style="background:var(--amber)"></i>step 24 · grounding</div>
</div>
{foot('home › follow the evidence › beat 01 and the lens cards · scene: three lenses sweep the pinned trace, prefix only')}''', scene='lenses', length=3)

    t = TRAINING
    gather = beat(2, 2, f'''
<div class="grid2">
 <div>
  {head('02 / Gather', 'Turn violations<br>into evidence.', lead='Violations go into a blame matrix: how often each agent breaks each reviewer’s rules. Every candidate step gets ten features, including its position, because a root cause comes before its symptoms.', width='100%')}
  <div class="card rv" style="--d:3;padding:22px;margin-top:4vh">
   <div class="ol" style="font-size:9px;margin-bottom:12px">Blame matrix · lost-context retail trace · violation rate</div>
   <table class="blame" aria-label="Violation rate per agent per perspective"><thead><tr><th>Agent</th><th>Intent</th><th>Tool</th><th>Grounding</th></tr></thead>
   <tbody><tr><td>Planner</td><td class="hot">30%</td><td>0%</td><td>8.3%</td></tr><tr><td>Order agent</td><td class="max">50%</td><td>0%</td><td>0%</td></tr></tbody></table>
   <p style="margin-top:14px;font-size:12px;color:var(--muted);line-height:1.6">Ten features per candidate: blame by perspective, position, pool size, violation count, agent role…</p>
  </div>
 </div>
 <div>
  <div class="ol red rv">What we tried first · a dead end we kept</div>
  <div class="h3 rv" style="--d:1;font-size:clamp(24px,2.2vw,32px);margin-top:14px">Training made it worse.</div>
  <p class="rv" style="--d:2;font-size:14px;line-height:1.65;color:var(--muted);margin-top:14px">A linear softmax policy ranked the candidates, trained with REINFORCE. The reward was whether the repair passed its acceptance check. Passing that check is easier than actually recovering, so the policy learned which steps produce repairs that get accepted. That is reward hacking, and we could measure it.</p>
  <div class="tflow rv" style="--d:3;margin-top:3vh" aria-label="Training flow: features, policy, proxy reward, held-out accuracy fell, weights frozen">
   <div><b>INPUT</b>Candidate features</div><div><b>POLICY</b>Linear softmax · REINFORCE</div><div><b>REWARD</b>Repair acceptance, no gold labels</div>
   <div class="bad"><b style="color:var(--red)">HELD OUT</b>{t['before']}% → {t['after']}%<br>{t['lost']} lost · {t['gained']} gained · p = {t['p']:.3f}</div><div><b>SHIPPED</b>Hand-set weights, frozen</div>
  </div>
  <div class="row rv" style="--d:4;margin-top:3vh">
   <div><div class="big" style="color:var(--red)">−{t['lost']}<small> pts</small></div><div style="font-size:12px;color:var(--muted);margin-top:6px">on {t['test_episodes']} held-out episodes</div></div>
   <div><div class="big" style="color:var(--amber)">+{t['reward_gap']}</div><div style="font-size:12px;color:var(--muted);margin-top:6px">proxy reward sat above real recovery</div></div>
  </div>
 </div>
</div>
{foot('home › evidence stage 02 and the training strip · results/README.md')}''')

    select = beat(2, 3, f'''
{head('03 / What we finalised', 'The evidence stops deciding.<br>One model call decides.', lead='The matrix and the features become an argument. The model gets the trace, the blame matrix and each candidate’s broken rule, and names the critical step and the responsible agent. It may pick a step our checks never flagged.')}
<div class="cands" id="cands" style="flex:1;margin-top:3vh;min-height:360px">
 <svg class="wires" id="wires" viewBox="0 0 1000 400" preserveAspectRatio="none" aria-hidden="true" style="position:absolute;inset:0;width:100%;height:100%">
  <path d="M250 40 C 450 40, 560 120, 640 120"/><path class="pick" d="M250 130 C 450 130, 560 130, 640 130"/><path d="M250 220 C 450 220, 560 140, 640 140"/><path d="M250 310 C 450 310, 560 150, 640 150"/><path class="out" d="M420 380 C 540 380, 600 200, 640 165"/>
 </svg>
 <div class="cand" style="left:0;top:0"><b>CANDIDATE · STEP 6 · PLANNER</b>Intent: routed the cancellation to the product specialist</div>
 <div class="cand pick" style="left:0;top:23%"><b>CANDIDATE · STEP 21 · PLANNER</b>Tool: order status read as “pending”, tool said “delivered”</div>
 <div class="cand" style="left:0;top:46%"><b>CANDIDATE · STEP 24 · ORDER AGENT</b>Grounding: order id not present in any earlier message</div>
 <div class="cand" style="left:0;top:69%"><b>CANDIDATE · STEP 31 · ORDER AGENT</b>Intent: claimed the write without calling the tool</div>
 <div class="cand outside" style="left:28%;bottom:0;border-color:#4a3f78;width:min(260px,24vw)"><b style="color:var(--violet)">OUTSIDE THE POOL · ANY STEP</b>The selector may also choose a step with no violation at all</div>
 <div class="selcard" id="selcard">
  <div class="ol mint">Full-trace selector · one call</div><strong>“Step 21, planner.”</strong>
  <p>Reads the whole trace, the blame matrix and the candidates. Returns the step, the agent, and the obligation it believes was broken, so the diagnosis carries its evidence into repair.</p>
  <div class="inputs" aria-label="Inputs to the selector"><span>trace</span><span>blame matrix</span><span>candidates + broken rules</span><span>frozen THETA0</span></div>
  <div class="nogold">GOLD LABELS NEVER ENTER THIS CALL · scored afterwards</div>
 </div>
</div>
{foot('home › follow the evidence › beat 03 · scene: the wires draw toward the selector as it arrives')}''', scene='select', length=2)

    def chart(domain, arms):
        rows = []
        for i, arm in enumerate(arms):
            cls = {'agentrx': 'rx', 'seeds_select': 'ar'}.get(arm['variant'], 'wr')
            if arm['variant'] == 'agentrx':
                note = f"{arm['hits']}/{arm['n']} · reproduced"
            else:
                pair = paired(FINAL_ROWS, 'agentrx', arm['variant'], domain)
                tau_only = ' · tau only' if arm['variant'] == 'synth_select_p10' else ''
                note = f"{arm['hits']}/{arm['n']} · {pair['wins']} gained, {pair['losses']} lost · p = {pair['p']:.2f}{tau_only}"
            rows.append(f'<div class="brow {cls}"><div class="lbl"><span>{escape(arm["label"])}<small>{escape(note)}</small></span>'
                        f'<strong>{pct(arm["accuracy"])}</strong></div><div class="track"><div class="bar" style="--d:{i};width:{arm["accuracy"]:.3f}%"></div></div></div>')
        if domain == 'magentic':
            rows.append('<p style="margin-top:20px;font-size:12px;color:var(--muted);line-height:1.6">Written checks: <span style="color:var(--amber)">not reportable</span>. Rate limiting failed the selector on 24 of 44 traces; the arm is excluded rather than published as a mixed number.</p>')
        return (f'<figure class="card chart rv" style="--d:2;margin:0"><figcaption><div class="ol" style="font-size:9px">{escape(DOMAINS[domain])} · n = {arms[0]["n"]}</div>'
                f'<h3>Exact fault-step accuracy</h3></figcaption>{"".join(rows)}</figure>')

    recall = {d: summarize(FINAL_ROWS, d, 'seeds_select') for d in DOMAINS}
    ablation = paired(SELECTOR_ROWS, 'mp_rl_top1', 'mp_rl_select_trace')
    total = sum(arms[0]['n'] for arms in HEADLINE.values())
    in_pool = {d: 100 * r['in_pool_hits'] / r['in_pool_n'] for d, r in recall.items()}
    out_pool = {d: 100 * r['out_pool_hits'] / (r['n'] - r['in_pool_n']) for d, r in recall.items()}
    rq1 = beat(2, 4, f'''
{head(f'RQ1 · {total} published traces · exact step match', 'It edges past AgentRx.<br>Not significantly. Recall is the limit.', tone='violet', width='min(1100px,80vw)')}
<div style="display:grid;grid-template-columns:minmax(0,2.4fr) minmax(0,1fr);gap:clamp(20px,3vw,50px);margin-top:3vh;align-items:start">
 <div class="grid2" style="gap:20px">{chart('tau', HEADLINE['tau'])}{chart('magentic', HEADLINE['magentic'])}</div>
 <div class="stack" style="gap:30px">
  <div class="stat rv" style="--d:3"><b style="color:var(--amber)">{recall['tau']['recall']:.0f}% <small style="font-size:.4em;color:var(--muted);letter-spacing:0">/ {recall['magentic']['recall']:.0f}%</small></b><span>Pool recall, tau / Magentic: how often the true step appears among the candidates at all. When it does, the selector is right {in_pool['tau']:.0f}% and {in_pool['magentic']:.0f}% of the time. When it does not: {out_pool['tau']:.0f}% and {out_pool['magentic']:.0f}%.</span></div>
  <div class="stat rv" style="--d:4"><b style="color:var(--mint)">{ablation['wins']} : {ablation['losses']}</b><span>Selector against our own ranker, paired over {ablation['n']} traces. p = {ablation['p']:.2f}. The one significant result, and it is internal.</span></div>
 </div>
</div>
{foot('home › the evidence charts and recall stats, same CSV-derived values · bars grow once, then hold', '→ hand-off: Prince')}''')
    return check + gather + select + rq1


# ------------------------------------------------------------------------------ 04 repair and orchestrator
def chapter_repair():
    milestones = ['Identify constraints', 'Rank candidates', 'Choose repair', 'Replay correction', 'Repair acceptance', 'Fresh-context verifier']
    ms = ''.join(f'<div class="ms"><i></i>{"I II III IV V VI".split()[i]} · {m}</div>' for i, m in enumerate(milestones))
    live = beat(3, 1, f'''
<div class="live">
 <div class="frame"><iframe title="Live retail system, Case II: a misrouted specialist handoff" data-widget="system" data-src="/widgets/system?scenario=misroute_specialist"></iframe></div>
 <div class="side">
  <div class="ol mint">Live system · Case II · Wrong specialist</div>
  <div class="h3">A diagnosis is a hypothesis.<br>Recovery has to act on it carefully.</div>
  <p style="font-size:13px;line-height:1.6;color:var(--muted)">The planner sends a cancellation to the catalog specialist. Attribution points back to the planner’s routing step, not to the specialist.</p>
  <div>{ms}</div>
  <div class="guide" aria-live="polite"><div class="ol" style="font-size:9px">What’s happening</div><strong data-guide="system-title">Follow the request through the agents.</strong><p data-guide="system-body">The moving signal shows who is acting. Press Play, or step with the arrows under the canvas.</p></div>
  <div class="foot" style="padding-top:12px"><span class="src">/widgets/system, scripted, no model calls</span><span class="hint">→ next beat</span></div>
 </div>
</div>''')

    repair = beat(3, 2, f'''
<div class="grid2">
 <div>
  {head('AgentReroute-Repair', 'A typed plan,<br>not free-text advice.', lead=f'{STRATEGIES} strategies arranged as escalation ladders, one ladder per failure type. Each rung has a deterministic acceptance check. We replace a wrong value instead of critiquing it: models fix errors far more reliably when handed the correct value.', width='100%')}
  <div class="stack rv" style="--d:3;gap:8px;margin-top:3vh" aria-label="One escalation ladder, for a lost-context failure">
   <div class="ol" style="font-size:9px">Ladder · context_propagation</div>
   <div class="rung on"><b>1</b>Inject the verified tool record at the faulty step</div><div class="rung"><b>2</b>Patch the handoff with the missing order id</div><div class="rung"><b>3</b>Re-query the tool and reground the agent</div><div class="rung"><b>4</b>Gate the next call by policy</div><div class="rung"><b>5</b>Ask the user</div><div class="rung esc"><b>6</b>Escalate to a human</div>
  </div>
 </div>
 <div>
  {head('The orchestrator · whether, where, how', 'It acts only when the<br>expected value is positive.', width='100%')}
  <div class="ev rv" style="--d:2;margin-top:3vh">EV = Σ (belief × strategy prior × {DISTANCE_DISCOUNT}<sup>distance</sup>) − cost</div>
  <div class="stack rv" style="--d:3;gap:8px;margin-top:16px" aria-label="Expected-value options scored by the real policy">
   <div class="opt sel"><strong>Step 6 · restore context</strong><span>success 0.618 − cost 0.055 = EV 0.563</span><em>Selected</em></div>
   <div class="opt"><strong>Step 6 · reground</strong><span>success 0.441 − cost 0.058 = EV 0.383</span></div>
   <div class="opt no"><strong>Step 8 · restore context</strong><span>success 0.210 − cost 0.047 = EV 0.163</span></div>
  </div>
  <div class="grid3 rv" style="--d:4;margin-top:3vh;font-size:12px;line-height:1.55;color:var(--muted);gap:18px">
   <div><div class="ol" style="font-size:9px;color:var(--text);margin-bottom:5px">On failure</div>Belief in the covered candidates drops by Bayes’ rule; the next rung is tried.</div>
   <div><div class="ol" style="font-size:9px;color:var(--text);margin-bottom:5px">Budgets</div>Tokens, time and attempts are capped. Never the same strategy twice at one step.</div>
   <div><div class="ol" style="font-size:9px;color:var(--text);margin-bottom:5px">Otherwise</div>EV ≤ 0: abstain, or escalate. No action is a valid action.</div>
  </div>
 </div>
</div>
{foot('/system › the EV decision example (same scripted values) · repair_strategies/strategies · policy.py EVConfig')}''')

    verifier = beat(3, 3, f'''
<div class="grid2">
 <div>
  {head('The key safeguard', 'A verifier that<br>cannot see the repair.', tone='cyan', lead='It gets the task, the policy and the replayed transcript. Never the repair, its rationale or the strategy. A repair cannot talk it into a success. If the verifier is unavailable, the orchestrator fails closed: success is never declared without it.', width='100%')}
  <div class="grid3 rv" style="--d:3;margin-top:4vh">
   <div class="lim"><b style="color:var(--mint)">SEES</b>task · policy · transcript</div>
   <div class="lim"><b style="color:var(--red)">NEVER SEES</b>repair · rationale · strategy · reward</div>
   <div class="lim"><b style="color:var(--amber)">ON OUTAGE</b>FAIL, closed</div>
  </div>
  <div class="stamp" id="stamp" style="margin-top:5vh"><b>PASS</b><span>the transcript contains the requested write and a non-error tool response</span></div>
 </div>
 <div class="vview" id="vview" aria-label="What the verifier reads: task, policy and transcript; the repair and its rationale stay out of its window">
  <div class="vdocs" id="vdocs">
   <div class="doc pending"><b>TASK</b>Cancel order #W1000001 for the authenticated user.</div>
   <div class="doc pending"><b>POLICY</b>Use identifiers present in the transcript. Cancellations require the order id and user confirmation.</div>
   <div class="doc pending"><b>TRANSCRIPT · replayed from step 6</b>user → planner: cancel my order<br>planner → tool get_order(#W1000001): {{"status": "delivered"}}<br>planner → order_agent: cancel #W1000001<br>order_agent → tool cancel_order(#W1000001)<br>tool → order_agent: {{"status": "cancelled"}}<br>order_agent → user: Your order is cancelled.</div>
   <div class="doc hid" data-stop><b>REPAIR · strategy 1 · inject verified record</b>Rollback to step 6, inject order_id = #W1000001 from get_order</div>
   <div class="doc hid"><b>RATIONALE</b>Candidate belief 0.62, prior 0.70, distance 3, EV 0.563</div>
   <div class="doc hid"><b>REWARD · hidden ground truth</b>ledger: order #W1000001 cancelled = true</div>
  </div>
  <div class="vwin" id="vwin"><span>VERIFIER VIEWPORT · fresh context</span></div>
 </div>
</div>
{foot('/system › milestone V copy · scene: the viewport scrolls the transcript and stops before the repair', '→ hand-off: Ananya')}''', scene='verifier', length=3)
    return live + repair + verifier


# ------------------------------------------------------------------------------ 05 benchmark and results
def _dots():
    """733 generated episodes as dots, grouped so the sorted field forms bands: seven fault labels, then
    clean, tolerated and dropped. The test set is marked: 100 fault dots, and 25 healthy controls drawn
    from the test split's clean and tolerated runs (13 clean, the rest tolerated)."""
    dots, test_left = [], CORPUS['test_failed']
    failed, healthy = ' data-t="f"', ' data-t="h"'
    for k, (_, _, count) in enumerate(CORPUS['labels']):
        share = min(test_left, round(CORPUS['test_failed'] * count / sum(c for _, _, c in CORPUS['labels'])))
        dots.extend(f'<i data-l="{k}"{failed if j < share else ""}></i>' for j in range(count))
        test_left -= share
    clean_controls = min(CORPUS['test_controls'], CORPUS['test_clean'])
    tolerated_controls = CORPUS['test_controls'] - clean_controls
    dots.extend(f'<i data-l="c"{healthy if j < clean_controls else ""}></i>' for j in range(CORPUS['clean']))
    dots.extend(f'<i data-l="t"{healthy if j < tolerated_controls else ""}></i>' for j in range(CORPUS['tolerated']))
    dots.extend('<i data-l="x"></i>' for _ in range(CORPUS['dropped']))
    assert len(dots) == CORPUS['generated'], len(dots)
    return ''.join(dots)


def chapter_benchmark():
    c = CORPUS
    legend = ''.join(f'<span><i style="background:{col}"></i>{escape(name)} · {n}</span>'
                     for (_, name, n), col in zip(c['labels'], ('#bcf783', '#7fdbed', '#f5cd88', '#b9a4ff', '#ff9b9b', '#8fb36a', '#d4a5ff')))
    legend += (f'<span><i style="background:#4e6068"></i>clean · {c["clean"]}</span>'
               f'<span><i style="background:#2b3a42"></i>tolerated the fault · {c["tolerated"]}</span>'
               f'<span><i style="background:#1c2a31"></i>dropped · {c["dropped"]}</span>')
    corpus = beat(4, 1, f'''
<div class="grid2" style="grid-template-columns:minmax(0,1fr) minmax(0,1.1fr)">
 <div>
  {head('Why we built our own', 'Published traces are recordings.<br>Ours can be replayed.', lead=f'A recording cannot tell you whether a repair works. So we reimplemented all {c["tasks"]} tau-bench retail tasks as the four-agent workflow, inject faults into tasks whose clean run succeeds, and keep only the runs that really fail. Every failure has a known step, agent and type.', width='100%')}
  <div class="row rv" style="--d:3;margin-top:4vh;gap:36px">
   <div><div class="big">{c['tasks']}</div><div class="src" style="margin-top:6px">TASKS</div></div>
   <div><div class="counter" id="dot-counter">0</div><div class="src" style="margin-top:6px">GENERATED</div></div>
   <div><div class="big">{len(c['labels'])}</div><div class="src" style="margin-top:6px">FAULT LABELS</div></div>
   <div><div class="big" style="color:var(--mint)">{c['test_failed']}<small> + {c['test_controls']}</small></div><div class="src" style="margin-top:6px">TEST: FAILED + HEALTHY</div></div>
  </div>
 </div>
 <div>
  <div class="field" id="field" aria-label="{c['generated']} generated episodes as dots, coloured by fault label">{_dots()}</div>
  <div class="leg" id="leg" style="margin-top:18px">{legend}</div>
 </div>
</div>
{foot('corpus-v2 manifest · scene: the field fills, sorts into labels, then the test set steps forward')}''', scene='dots', length=3)

    def claim_row(i, r):
        note = f'<small>{escape(r["note"])}</small>' if r['note'] else ''
        agentrx = r['id'] == 'verified_agentrx'
        fill = 'cf rx' if agentrx else 'cf'
        value = f'<b style="color:var(--violet)">{r["recovered"]}</b>' if agentrx else f'<b>{r["recovered"]}</b>'
        return (f'<div class="crow"><div class="n">{escape(r["label"])}{note}</div>'
                f'<div class="tr"><div class="cl" style="--d:{i};width:{r["claimed"]}%"></div><div class="{fill}" style="--d:{i};width:{r["recovered"]}%"></div></div>'
                f'<div class="v"><i>{r["claimed"]}</i> → {value}</div></div>')
    rows = ''.join(claim_row(i, r) for i, r in enumerate(RECOVERY))
    ours = next(r for r in RECOVERY if r['id'] == 'verified')
    theirs = next(r for r in RECOVERY if r['id'] == 'verified_agentrx')
    repair_row = next(r for r in RECOVERY if r['id'] == 'repair')
    overclaim = beat(4, 2, f'''
<div style="display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.6fr);gap:clamp(24px,4vw,70px)">
 <div>
  {head('Finding 1 · RQ2', 'Finding the right fault matters.', tone='violet', width='100%')}
  <div class="row rv" style="--d:2;margin-top:3vh;gap:14px"><span class="big" style="font-size:clamp(60px,6vw,96px);color:var(--mint)">{ours['recovered']}</span><span class="mono" style="font-size:26px;color:var(--dim)">vs</span><span class="big" style="font-size:clamp(60px,6vw,96px);color:var(--violet)">{theirs['recovered']}</span></div>
  <p class="rv" style="--d:3;font-size:14px;line-height:1.65;color:var(--muted);margin-top:12px">Recoveries out of {RECOVERY_EPISODES}, same orchestrator, same verifier. Our attribution against AgentRx’s. p = {RQ2_P}, short of significance, and the honest sentence is “more, under the same machinery”.</p>
 </div>
 <div>
  {head('Finding 2 · the one that matters most', 'Every system over-claims.<br>Ours included.', tone='amber', width='100%')}
  <div class="rv" style="--d:2;margin-top:3vh" aria-label="Claimed versus confirmed recoveries per policy, out of 100 episodes">
   <div style="display:flex;gap:18px;font:10px var(--mono);letter-spacing:1px;color:var(--muted);margin-bottom:6px"><span><i style="display:inline-block;width:14px;height:8px;border:1px solid var(--amber);border-radius:2px;vertical-align:middle;margin-right:6px"></i>CLAIMED by the system</span><span><i style="display:inline-block;width:14px;height:8px;background:var(--mint);border-radius:2px;vertical-align:middle;margin-right:6px"></i>CONFIRMED by the hidden ground truth</span></div>
   {rows}
  </div>
  <p class="rv" style="--d:6;font-size:clamp(13px,1.1vw,15px);line-height:1.65;margin-top:3vh">Our repair policy reported {repair_row['claimed']} recoveries; the hidden check confirmed {REPAIR_CONFIRMED_CLAIMS} of those claims. Even the strict verifier claimed {ours['claimed']} where {VERIFIER_CONFIRMED_CLAIMS} were real. <span style="color:var(--amber)">A system’s own judgement of its success cannot be trusted.</span></p>
 </div>
</div>
{foot('final report, Table 2 · replaces opening the PDF at page 5')}''')

    def harm_card(r, tone, highlight=False):
        broken = ''.join('<i class="x"></i>' if j < (r['harmed'] or 0) else '<i></i>' for j in range(CONTROLS))
        note = f'rejected {ORCHESTRATOR_REJECTED} unverifiable repairs · abstained on {ORCHESTRATOR_ABSTAINED} for cost' if r['id'] == 'verified' else (f'p = {HARM_P}' if r['id'] == 'fixed' else '&nbsp;')
        style = 'border-color:#3a5a3a;background:#13201a' if highlight else ''
        return (f'<div class="card rv" style="--d:{2 if not highlight else 4};padding:22px;{style}"><div class="ol {tone}" style="font-size:9px">{escape(r["label"])}</div><h3 style="font-size:16px;font-weight:600;margin-top:4px">Recovered {r["recovered"]} / {RECOVERY_EPISODES}</h3>'
                f'<div class="runs" aria-hidden="true">{broken}</div><div class="harmed"><span class="big" style="color:var(--{tone})">{r["harmed"]}</span><span class="ol">/ {CONTROLS} healthy runs broken</span></div>'
                f'<div style="font-size:12px;color:var(--muted);margin-top:6px">{note}</div><div class="cost"><i style="width:{int(r["cost"] * 100)}px"></i>cost {r["cost"]:.2f}× the original run</div></div>')

    fixed = next(r for r in RECOVERY if r['id'] == 'fixed')
    harm = beat(4, 3, f'''
<div style="display:grid;grid-template-columns:minmax(0,2fr) minmax(0,1fr);gap:clamp(24px,3vw,50px)">
 <div>
  {head('Finding 3 · RQ3', 'Simple rules recover the most.<br>They also break runs that were fine.', tone='red', width='100%')}
  <div class="grid3" style="margin-top:3vh">{harm_card(fixed, 'red')}{harm_card(repair_row, 'amber')}{harm_card(ours, 'mint', True)}</div>
  <p class="rv" style="--d:5;font-size:clamp(13px,1.1vw,15px);line-height:1.65;margin-top:3vh">It recovered fewer because it refuses to accept a repair it cannot verify. Which error is worse, a missed recovery or a broken healthy run, is a value choice. <span style="color:var(--mint)">We argue it should be set explicitly, with escalation to a human, not hidden in a threshold.</span></p>
 </div>
 <div>
  <div class="ol rv">Then, live</div>
  <div class="h3 rv" style="--d:1;margin-top:12px">Case IV · Healthy control.<br>Watch it check every step and abstain.</div>
  <div class="mini-live rv" style="--d:2;margin-top:3vh"><iframe title="Live retail system, Case IV: a healthy run the orchestrator leaves alone" data-widget="system" data-src="/widgets/system?scenario=clean_control_flow"></iframe></div>
  <p class="rv" style="--d:3;font-size:12px;line-height:1.6;color:var(--muted);margin-top:14px"><strong data-guide="system-title" style="display:block;color:var(--text);font-size:13px;margin-bottom:4px">Follow the request through the agents.</strong><span data-guide="system-body">Scripted, no model call. The last frame stays on screen through the hand-off.</span></p>
 </div>
</div>
{foot('final report, Table 2 harm and cost rows · /widgets/system with Case IV preselected', '→ hand-off: William')}''')
    return corpus + overclaim + harm


# ------------------------------------------------------------------------------ 06 closing
def chapter_closing():
    ours = {d: summarize(FINAL_ROWS, d, 'seeds_select') for d in DOMAINS}
    rx = {d: summarize(FINAL_ROWS, d, 'agentrx') for d in DOMAINS}
    verified = next(r for r in RECOVERY if r['id'] == 'verified')
    theirs = next(r for r in RECOVERY if r['id'] == 'verified_agentrx')
    fixed = next(r for r in RECOVERY if r['id'] == 'fixed')
    answers = beat(5, 1, f'''
{head('The three questions, answered', 'What we can claim.<br>And what we cannot.')}
<div style="margin-top:4vh">
 <div class="ans rv" style="--d:2"><b>RQ1</b><div><h3>Multi-perspective attribution edged past AgentRx.</h3><span class="v">{pct(ours['tau']['accuracy'])} / {pct(ours['magentic']['accuracy'])} vs {pct(rx['tau']['accuracy'])} / {pct(rx['magentic']['accuracy'])} · not significant</span></div><p>Recall limits it: the true step reaches the candidate pool only {ours['tau']['recall']:.0f}% and {ours['magentic']['recall']:.0f}% of the time. Training on an acceptance signal made it worse, a case of reward hacking we could measure.</p></div>
 <div class="ans rv" style="--d:3"><b>RQ2</b><div><h3>Better attribution led to more recoveries under the same orchestrator.</h3><span class="v">{verified['recovered']} vs {theirs['recovered']} of {RECOVERY_EPISODES} · p = {RQ2_P}</span></div><p>Short of significance. We compared attribution sources; an arm with no attribution at all is the next experiment, and the harness already contains it.</p></div>
 <div class="ans rv" style="--d:4"><b>RQ3</b><div><h3>Yes. Recovery can be verified, budgeted and kept free of label leakage.</h3><span class="v">{verified['harmed']} of {CONTROLS} healthy runs broken · {verified['cost']:.2f}× cost</span></div><p>Only the verified orchestrator harmed no healthy run. It paid for that in recoveries: {verified['recovered']} against {fixed['recovered']} for fixed rules.</p></div>
</div>
{foot('the closing script · home › three limits')}''')

    close = beat(5, 2, f'''
<svg class="title-art" viewBox="0 0 1600 900" preserveAspectRatio="xMidYMid slice" aria-hidden="true" style="opacity:.55">
<path d="M-20 820 C 300 820, 420 600, 760 600 S 1100 820, 1620 820" stroke="#1f2d33" stroke-width="2" fill="none"/>
<path d="M-20 820 C 300 820, 420 600, 760 600 S 1100 820, 1620 820" stroke="#bcf783" stroke-width="2" fill="none" stroke-dasharray="6 10" opacity=".6"/>
<circle cx="760" cy="600" r="6" fill="#bcf783"/>
</svg>
<div style="position:relative;margin-top:8vh;max-width:1200px">
 <div class="ol rv">The lesson we would carry beyond this project</div>
 <h1 class="h1 rv" style="--d:1;margin-top:20px;font-size:clamp(40px,5vw,78px)">Self-reported success<br>has to be checked by something<br><em>that cannot see the repair.</em></h1>
</div>
<div class="grid4 rv" style="--d:3;margin-top:auto;position:relative">
 <div class="lim"><b>LIMIT · 01</b>One injected benchmark, one actor model shared with the verifier.</div>
 <div class="lim"><b>LIMIT · 02</b>{CONTROLS} healthy controls; 73 attribution traces. Enough to separate 7 from 0, not to see small harm rates.</div>
 <div class="lim"><b>NEXT · 01</b>Higher-recall checks, so the true step reaches the pool more often.</div>
 <div class="lim"><b>NEXT · 02</b>An actor-independent verifier and a learned intervention threshold.</div>
</div>
<div class="row rv" style="--d:4;margin-top:4vh;position:relative;align-items:center"><span style="font-size:20px;font-weight:600;letter-spacing:-.02em">Thank you. We’re happy to take questions.</span><span class="src" style="font-size:11px">WILLIAM MODERATES · ANSWER FIRST · ONE NUMBER · 30 SECONDS</span></div>
{foot('home › three limits and the closing line · the route art returns, so the scroll ends where it began', 'Esc · exit')}''')
    return answers + close


# ------------------------------------------------------------------------------ chrome and page
def chrome():
    dots = ''.join(f'<a href="#b{i}-1" data-chapter="{i}" title="{escape(ch["title"])} · {escape(ch["speaker"])}"><i></i></a>'
                   for i, ch in enumerate(CHAPTERS))
    return f'''<div class="chrome">
<div class="progress" id="progress" aria-hidden="true"></div>
<nav class="rail" aria-label="Chapters">{dots}<span class="brand" aria-hidden="true">{MARK}</span></nav>
<header class="topbar"><span class="ol" id="chapter-label">Ch 01 · Introduction</span><span class="topbar-right"><span class="ol mint"><span id="speaker">William</span> · <span id="timer">1:15</span> · <span id="beat-count">beat 1 / 3</span></span><a class="exit ol" href="/">Esc · exit</a></span></header>
<div class="handoff" id="handoff" role="status" aria-live="polite"><div><span class="from" id="handoff-from"></span><strong class="to" id="handoff-to"></strong></div><span class="budget" id="handoff-budget"></span></div>
</div>'''


JS = r'''
(function(){
 'use strict';
 const CHAPTERS=__CHAPTERS__, BEATS=__BEATS__;
 const root=document.documentElement, reduced=matchMedia('(prefers-reduced-motion: reduce)');
 const motionOff=()=>reduced.matches;
 root.classList.toggle('rm',motionOff());reduced.addEventListener('change',()=>root.classList.toggle('rm',motionOff()));
 const $=id=>document.getElementById(id), clamp=(v,a,b)=>Math.min(b,Math.max(a,v)), lerp=(a,b,t)=>a+(b-a)*t;
 const ease=t=>1-Math.pow(1-t,3);
 const beats=[...document.querySelectorAll('.beat')];

 /* ---- scroll-driven scenes: each receives progress p in [0,1] through its own section ---- */
 const scenes={
  trace(el,p){
   const rail=el.querySelector('#trace-rail'),steps=[...rail.querySelectorAll('.step')],wrap=rail.parentElement;
   const centre=i=>steps[i-1].offsetLeft+steps[i-1].offsetWidth/2, half=wrap.clientWidth/2;
   const target=p<.5?lerp(centre(8),centre(33),ease(p/.5)):lerp(centre(33),centre(21),ease((p-.5)/.5));
   rail.style.transform=`translateX(${Math.round(half-target)}px)`;
   const lit=p<.5?Math.round(lerp(0,33,p/.5)):33;
   steps.forEach((s,i)=>{s.classList.toggle('lit',i<lit);});
   const fail=steps[32],cause=steps[20];
   fail.classList.toggle('fail',p>=.47);fail.classList.toggle('faded',p>=.9);
   cause.classList.toggle('cause',p>=.88);
   const tf=el.querySelector('#tag-fail'),tc=el.querySelector('#tag-cause');
   tf.style.left=(centre(33)-110)+'px';tc.style.left=(centre(21)-130)+'px';
   tf.classList.toggle('show',p>=.47&&p<.92);tc.classList.toggle('show',p>=.9);
   el.querySelector('#hl').classList.toggle('resolved',p>=.8);
  },
  gap(el,p){
   const t=ease(clamp(p/.65,0,1));
   el.querySelector('.line.l').style.transform=`translateX(${-110*(1-t)}%)`;
   el.querySelector('.line.r').style.transform=`translateX(${110*(1-t)}%)`;
   el.querySelector('#gapviz').classList.toggle('closed',p>=.72);
  },
  network(el,p){
   /* the drawing is 900×844; fit it to the stage and centre it, scaling around the planner (21%, 48%) */
   const net=el.querySelector('#net'),wrap=net.parentElement;
   const fit=Math.min(1,(wrap.clientHeight-24)/844,(wrap.clientWidth-24)/900);
   net.style.left=Math.round((wrap.clientWidth-900*fit)/2-189*(1-fit))+'px';
   net.style.top=Math.round((wrap.clientHeight-844*fit)/2-405*(1-fit))+'px';
   const s=lerp(.06,fit,ease(clamp(p/.6,0,1)));
   net.style.transform=`scale(${s.toFixed(4)})`;
   net.classList.toggle('show-specs',p>=.45);net.classList.toggle('show-tools',p>=.72);net.classList.toggle('show-signal',p>=.85);
  },
  lenses(el,p){
   const track=el.querySelector('#lens-track'),lenses=[...track.querySelectorAll('.lens')],w=track.clientWidth;
   const mini=el.querySelector('#trace-mini'),steps=[...mini.querySelectorAll('.step')];
   let lead=0;
   lenses.forEach((l,k)=>{const t=ease(clamp((p-k*.11)/.72,0,1));const x=lerp(-l.offsetWidth,w-l.offsetWidth,t);l.style.transform=`translateX(${Math.round(x)}px)`;lead=Math.max(lead,x+l.offsetWidth);});
   const edge=clamp(lead,0,mini.clientWidth),edgeEl=el.querySelector('#edge');
   edgeEl.style.left=edge+'px';edgeEl.classList.toggle('flip',edge>mini.clientWidth-460);
   steps.forEach(s=>s.classList.toggle('lit',s.offsetLeft+s.offsetWidth<=edge+1));
   el.querySelectorAll('.vio').forEach(v=>{const s=steps[Number(v.dataset.step)-1];v.style.left=(s.offsetLeft+s.offsetWidth/2-v.offsetWidth/2)+'px';v.classList.toggle('show',s.offsetLeft+s.offsetWidth<=edge+1);});
  },
  select(el,p){
   const draw=ease(clamp(p/.7,0,1));
   el.querySelectorAll('.wires path').forEach(path=>{path.style.strokeDashoffset=String(Math.round(1200*(1-draw)));});
   const card=el.querySelector('#selcard'),t=ease(clamp((p-.1)/.5,0,1));
   card.style.transform=`translateX(${Math.round(60*(1-t))}%)`;card.style.opacity=String(t);
   el.querySelector('#cands').classList.toggle('chosen',p>=.78);
  },
  verifier(el,p){
   const view=el.querySelector('#vview'),docs=el.querySelector('#vdocs'),win=el.querySelector('#vwin');
   const stop=docs.querySelector('[data-stop]'),maxY=Math.max(0,stop.offsetTop-win.offsetHeight-6);
   const y=lerp(0,maxY,ease(p));
   win.style.transform=`translateY(${Math.round(y)}px)`;
   const top=y, bottom=y+win.offsetHeight;
   docs.querySelectorAll('.doc:not(.hid)').forEach(d=>{const c=d.offsetTop+d.offsetHeight/2;d.classList.toggle('pending',c>bottom);});
   el.querySelector('#stamp').classList.toggle('show',p>=.92);
  },
  dots(el,p){
   const field=el.querySelector('#field'),dots=field.children,n=dots.length;
   const on=Math.round(n*clamp(p/.33,0,1));
   if(on!==field._on){const from=field._on||0;if(on>from)for(let i=from;i<on;i++)dots[i].classList.add('on');else for(let i=on;i<from;i++)dots[i].classList.remove('on');field._on=on;}
   el.querySelector('#dot-counter').firstChild.textContent=String(on);
   field.classList.toggle('sorted',p>=.4);field.classList.toggle('test',p>=.74);
   el.querySelector('#leg').classList.toggle('show',p>=.42);
  }
 };

 /* ---- navigation: stops are the start of every beat and the end of every scene ---- */
 let stops=[];
 function measure(){
  stops=[];
  beats.forEach(b=>{const top=b.offsetTop;stops.push(top);if(b.classList.contains('scene'))stops.push(top+b.offsetHeight-innerHeight);});
  stops=[...new Set(stops)].sort((a,b)=>a-b);
 }
 function go(y){scrollTo({top:y,behavior:motionOff()?'auto':'smooth'});}
 function next(){const y=scrollY+4;const s=stops.find(v=>v>y);if(s!==undefined)go(s);}
 function prev(){const y=scrollY-4;const s=[...stops].reverse().find(v=>v<y);go(s===undefined?0:s);}

 /* ---- chapter state, timer, hand-off ---- */
 let current=-1, chapter=-1, deadline=0, handoffTimer=null;
 const timer=$('timer');
 function fmt(sec){const s=Math.abs(sec),m=Math.floor(s/60),r=String(s%60).padStart(2,'0');return (sec<0?'+':'')+m+':'+r;}
 function tick(){if(!deadline)return;const left=Math.ceil((deadline-Date.now())/1000);timer.textContent=fmt(left);timer.className=left<0?'over':left<=20?'warn':'';}
 setInterval(tick,250);
 function resetTimer(){deadline=Date.now()+CHAPTERS[chapter].seconds*1000;tick();}
 function showHandoff(from,to){
  $('handoff-from').textContent=`${CHAPTERS[from].speaker} · ${CHAPTERS[from].title.toLowerCase()} · done`;
  $('handoff-to').innerHTML=`${CHAPTERS[to].speaker}<br><em>${CHAPTERS[to].title}</em>`;
  $('handoff-budget').innerHTML=`${fmt(CHAPTERS[to].seconds)}<small>RESTARTS ON HAND-OFF</small>`;
  const strip=$('handoff');strip.classList.add('show');clearTimeout(handoffTimer);handoffTimer=setTimeout(()=>strip.classList.remove('show'),1300);
 }
 function enter(index){
  if(index===current)return;
  const b=beats[index],ch=Number(b.dataset.chapter),beat=Number(b.dataset.beat);
  current=index;
  $('chapter-label').textContent=`Ch 0${ch+1} · ${CHAPTERS[ch].title}`;
  $('speaker').textContent=CHAPTERS[ch].speaker;
  $('beat-count').textContent=`beat ${beat} / ${BEATS[ch]}`;
  document.querySelectorAll('.rail a').forEach(a=>{if(Number(a.dataset.chapter)===ch)a.setAttribute('aria-current','true');else a.removeAttribute('aria-current');});
  if(ch!==chapter){const from=chapter;chapter=ch;resetTimer();if(from>=0&&ch===from+1)showHandoff(from,ch);}
 }

 /* ---- the scroll loop ---- */
 let queued=false;
 function frame(){
  queued=false;
  const y=scrollY, mid=y+innerHeight*.5;
  let active=0;
  beats.forEach((b,i)=>{if(b.offsetTop<=mid)active=i;});
  enter(active);
  beats.forEach(b=>{
   const scene=b.dataset.scene;if(!scene)return;
   const span=b.offsetHeight-innerHeight;
   const p=motionOff()?1:clamp(span>0?(y-b.offsetTop)/span:1,0,1);
   if(b._p!==p){b._p=p;scenes[scene](b,p);}
  });
  const travel=root.scrollHeight-innerHeight;$('progress').style.width=(travel>0?clamp(y/travel*100,0,100):0)+'%';
 }
 function queue(){if(!queued){queued=true;requestAnimationFrame(frame);}}
 addEventListener('scroll',queue,{passive:true});
 addEventListener('resize',()=>{measure();beats.forEach(b=>{b._p=undefined;});queue();});

 /* ---- reveal on arrival ---- */
 const seen=new IntersectionObserver(entries=>{for(const e of entries)if(e.isIntersecting){e.target.classList.add('in');seen.unobserve(e.target);}},{threshold:.3});
 document.querySelectorAll('.stage').forEach(s=>seen.observe(s));

 /* ---- the live widgets mount when they come near, and report what they show ---- */
 const frames=[...document.querySelectorAll('iframe[data-src]')];
 function mount(f){
  if(f.src)return;
  f.onload=()=>f.contentWindow.postMessage({type:'agentreroute:motion',paused:motionOff()},location.origin);
  f.src=f.dataset.src;
 }
 /* once the widget has reported its height, open its wrapper on the canvas rather than the scenario buttons */
 function openOnCanvas(f){
  if(f._opened)return;
  const canvas=f.contentDocument&&f.contentDocument.querySelector('.chart');
  if(!canvas)return;
  f._opened=true;f.parentElement.scrollTop=Math.max(0,canvas.offsetTop-8);
 }
 const near=new IntersectionObserver(entries=>{for(const e of entries)if(e.isIntersecting){mount(e.target);near.unobserve(e.target);}},{rootMargin:'120% 0px'});
 frames.forEach(f=>near.observe(f));
 addEventListener('message',e=>{
  if(e.origin!==location.origin||!e.data||typeof e.data!=='object')return;
  const frame=frames.find(f=>f.contentWindow===e.source);if(!frame)return;
  if(e.data.type==='agentreroute:resize'&&Number.isFinite(e.data.height)&&e.data.height>=100&&e.data.height<=20000){frame.style.height=Math.ceil(e.data.height)+'px';openOnCanvas(frame);return;}
  if(e.data.type!=='agentreroute:guide')return;
  const beat=frame.closest('.beat');
  const title=beat.querySelector('[data-guide="system-title"]'),body=beat.querySelector('[data-guide="system-body"]');
  if(title&&typeof e.data.title==='string')title.textContent=e.data.title.slice(0,200);
  if(body&&typeof e.data.body==='string')body.textContent=e.data.body.slice(0,1000);
 });

 /* ---- keys ---- */
 addEventListener('keydown',e=>{
  if(e.target instanceof Element&&e.target.closest('input,textarea,select,[contenteditable="true"]'))return;
  if(e.key==='ArrowRight'||e.key===' '||e.key==='PageDown'){e.preventDefault();next();}
  else if(e.key==='ArrowLeft'||e.key==='PageUp'){e.preventDefault();prev();}
  else if(e.key==='Home'){e.preventDefault();go(0);}
  else if(e.key==='End'){e.preventDefault();go(stops[stops.length-1]||0);}
  else if(e.key==='Escape'){location.href='/';}
  else if(e.key==='t'||e.key==='T'){resetTimer();}
  else if(e.key==='n'||e.key==='N'){root.classList.toggle('notes');}   /* rehearsal: show each beat's source line */
 });
 document.querySelectorAll('.rail a').forEach(a=>a.addEventListener('click',e=>{e.preventDefault();const t=document.querySelector(a.getAttribute('href'));if(t)go(t.offsetTop);}));

 measure();frame();
})();
'''


def render():
    body = chapter_intro() + chapter_existing() + chapter_attribution() + chapter_repair() + chapter_benchmark() + chapter_closing()
    script = JS.replace('__CHAPTERS__', json.dumps([{k: ch[k] for k in ('title', 'speaker', 'seconds')} for ch in CHAPTERS])) \
               .replace('__BEATS__', json.dumps(list(BEATS_PER_CHAPTER)))
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AgentReroute — Presentation</title>
<meta name="description" content="The AgentReroute talk as one scroll: six chapters, five speakers, the same evidence as the research story.">
<style>{CSS}</style></head>
<body>
{chrome()}
<main id="deck">
{body}
</main>
<script>{script}</script>
</body></html>'''


PAGE = render()
