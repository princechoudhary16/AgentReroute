"""Gemini-powered comparison studio with inspectable execution history."""
from .studio_theme import BASE_CSS, MARK, navigation
PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Compare systems · AgentReroute</title>
<style>__STUDIO_THEME__
:root{--paper:var(--bg);--card:var(--surface);--ink:var(--text);--line:var(--border);--blue:var(--violet);--green:var(--mint);--gold:var(--amber);--red:var(--danger)}
.run-setup{padding:20px 22px;background:linear-gradient(115deg,#172323,#11191f);border:1px solid #2b3e42;border-radius:14px;position:relative}.setup-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}.setup-top .eyebrow-label{color:#b1c5c7}.setup-top a{font-size:10px;color:var(--muted);text-decoration:none}.controls{display:grid;grid-template-columns:repeat(4,minmax(140px,1fr)) auto;gap:18px;align-items:end}.field{min-width:0}.field label{font-size:11px;color:var(--muted);display:block;margin-bottom:8px}.field select{width:100%;font-size:12px;height:42px;background:#0d171c}.run-actions{display:flex;align-items:center;gap:7px}.run-actions button{height:42px}.notice{font-size:10px;color:#8fa3aa;line-height:1.8;margin:13px 0 0}.provider-setup{display:flex;justify-content:space-between;align-items:center;gap:20px;margin-top:15px;padding:13px 15px;border:1px solid #5b533b;background:#2a261a55;border-radius:9px}.provider-setup strong{font-size:12px;color:var(--amber);font-weight:500}.provider-setup p{margin:4px 0 0;font-size:11px;color:var(--muted);line-height:1.7}.provider-setup code{color:var(--text)}.error{margin:14px 0;padding:13px 16px;border:1px solid #69404a;background:#2e1d26;color:var(--danger);border-radius:9px;font-size:12px;line-height:1.7;white-space:pre-wrap}.error:empty{display:none}.runbar{display:flex;align-items:center;justify-content:space-between;gap:18px;margin:25px 0 15px}.runbar h2{font-size:12px;font-weight:500;margin:0 0 6px}.status-line{font:10px var(--mono);color:var(--muted);margin:0}.playback{display:flex;align-items:center;gap:5px;padding:5px;background:#0c1418;border:1px solid var(--border);border-radius:9px}.playback .button{background:transparent;border-color:transparent}.playback .button:hover{background:var(--raised)}.playback input{width:120px;margin:0 5px}#cursor-label{font:9px var(--mono);color:var(--muted);min-width:78px;text-align:center}.columns{display:grid;grid-template-columns:1fr 1fr;gap:20px;position:relative}.system{--engine:var(--violet);--engine-soft:#211d34;background:linear-gradient(135deg,#1c1c2c55,transparent 35%),var(--surface);border:1px solid #3a354b;border-radius:15px;min-width:0;overflow:hidden;display:flex;flex-direction:column}.system.reroute{--engine:var(--mint);--engine-soft:#1c2b23;border-color:#344b3d;background:linear-gradient(135deg,#203c2755,transparent 35%),var(--surface)}.system-head{display:flex;align-items:center;gap:12px;padding:22px 23px 19px}.system-emblem{display:flex;align-items:center;justify-content:center;width:42px;height:42px;border:1px solid color-mix(in srgb,var(--engine) 35%,transparent);background:var(--engine-soft);color:var(--engine);border-radius:12px;font:15px var(--mono)}.system-emblem svg{width:26px;height:26px}.system h2{font-size:23px;letter-spacing:-.8px;font-weight:550;margin:0 0 5px}.kicker{color:var(--muted);font-size:10px}.badge{margin-left:auto;padding:5px 8px;background:#111c22;border:1px solid var(--border);border-radius:5px;font:9px var(--mono);color:var(--muted);white-space:nowrap}.badge.running{color:var(--cyan);background:#142a32;border-color:#325d6b}.badge.completed{color:var(--mint);background:#213225;border-color:#46653c}.badge.error,.badge.partial,.badge.canceled{color:var(--danger);background:#301f28;border-color:#69404a}.algorithm{margin:0 23px 20px;padding:17px 15px;border:1px solid var(--border);border-radius:11px;background:#0b1318}.flow-caption{font:9px var(--mono);text-transform:uppercase;letter-spacing:.9px;color:var(--engine);margin-bottom:13px}.flow-group{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;position:relative}.stage{min-width:0;display:flex;align-items:center;flex-direction:column;justify-content:center;gap:7px;position:relative;padding:11px 5px 10px;border:1px solid #28343f;border-radius:7px;font-size:10px;text-align:center;line-height:1.4;color:#8e9fa6;background:#121c23;min-height:66px}.stage i{width:20px;height:20px;display:flex;align-items:center;justify-content:center;border:1px solid #354550;border-radius:6px;font:9px var(--mono);font-style:normal;color:#708791}.stage.done{color:#d9e5e5;border-color:#3d5355;background:#172229}.stage.done i{color:var(--engine);border-color:color-mix(in srgb,var(--engine) 35%,transparent);background:var(--engine-soft)}.stage.active{border-color:var(--engine);background:var(--engine-soft);color:var(--engine);box-shadow:0 0 20px color-mix(in srgb,var(--engine) 9%,transparent)}.stage.active i{color:var(--engine);border-color:var(--engine)}.stage.failed{color:var(--danger);background:#2b1e27;border-color:#713e4c}.stage.failed i{color:var(--danger);border-color:#713e4c}.flow-connector{height:24px;position:relative;display:flex;justify-content:center;color:#4d6470;font-size:13px;align-items:center}.flow-connector:before,.flow-connector:after{content:'';height:1px;width:18%;background:#2b4249;position:absolute;top:50%}.flow-connector:before{right:55%}.flow-connector:after{left:55%}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;padding:16px 23px;background:#0d171b;border-block:1px solid var(--border)}.stat strong{font-size:23px;letter-spacing:-.7px;font-variant-numeric:tabular-nums;font-weight:500;display:block}.stat span{font-size:9px;color:var(--muted);display:block;margin-top:5px}.section{padding:18px 23px}.section-title{font:9px var(--mono);letter-spacing:.8px;text-transform:uppercase;font-weight:400;color:var(--muted);margin:0 0 12px}.attribution{flex:1}.candidates{display:grid;gap:7px}.candidate{position:relative;display:grid;grid-template-columns:20px minmax(0,1fr) auto;gap:8px;padding:12px;border:1px solid var(--border);background:#152027;border-radius:8px;overflow:hidden}.candidate:after{content:'';position:absolute;bottom:0;left:0;height:2px;width:var(--score,0%);background:var(--engine);opacity:.5}.rank{font:10px var(--mono);color:var(--engine);padding-top:2px}.candidate b{font-size:12px;font-weight:500;overflow-wrap:anywhere}.candidate small{display:block;color:var(--muted);font-size:9px;margin-top:5px;overflow-wrap:anywhere}.score{font:12px var(--mono);color:var(--engine);text-align:right}.outcome{margin:0 23px 19px;padding:13px 15px;border:1px solid var(--border);border-radius:9px;background:#0d171b;color:var(--muted);font-size:10px;line-height:1.8;min-height:89px}.outcome strong{display:block;font-size:14px;font-weight:500;margin-bottom:4px;color:var(--text)}.outcome .good{color:var(--mint)}.outcome .bad{color:var(--danger)}.outcome:has(.good){border-color:#3f6242;background:#16271d}.outcome:has(.bad){border-color:#5c3946;background:#251c24}.warning{padding:12px 23px;color:var(--amber);background:#2a261c;font-size:11px;line-height:1.7;border-top:1px solid #5c5137}.warning:empty{display:none}.evidence{border-top:1px solid var(--border)}.evidence>summary{display:flex;align-items:center;gap:10px;padding:15px 23px;color:var(--muted);font-size:11px}.evidence-count{font:9px var(--mono);margin-left:auto;color:#718a93}.evidence>summary:after{margin-left:5px}.evidence .section{padding-top:7px}.checks-line{display:flex;flex-wrap:wrap;gap:9px;color:var(--muted);font:10px/1.8 var(--mono)}.violation{color:var(--danger)}.log{max-height:220px;overflow:auto;display:grid;gap:5px}.log-row{width:100%;text-align:left;background:#0d171d;border:1px solid var(--border);padding:9px;border-radius:6px;color:var(--muted);font:10px/1.6 var(--mono);overflow-wrap:anywhere}.log-row:hover{border-color:#536b75}.log-row b{font-weight:400;color:var(--text)}.log-row.failed{color:var(--danger);border-color:#5b3643}.supporting{margin-top:20px;display:grid;grid-template-columns:1fr 1fr;gap:12px}.supporting>details{padding:0 19px;background:var(--surface);border:1px solid var(--border);border-radius:10px;min-width:0}.supporting>details[open]{grid-column:1/-1;padding-bottom:16px}.supporting summary{padding:17px 0;font-size:11px;font-weight:500}.supporting summary span{color:var(--muted);font-size:10px;margin-left:7px}.table-scroll{overflow:auto}.performance table{border-collapse:collapse;width:100%;font-size:12px}.performance td,.performance th{text-align:left;padding:12px 10px;border-bottom:1px solid var(--border)}.performance th{font:9px var(--mono);color:var(--muted)}.performance td:nth-child(2){color:var(--violet)}.performance td:nth-child(3){color:var(--mint)}.performance p,.method p{font-size:11px;line-height:1.8;color:var(--muted);margin:10px 0 0}.source-meta{color:var(--muted);font:10px/1.8 var(--mono);margin:0 0 12px}.trace{display:flex;gap:7px;overflow:auto;padding-bottom:10px}.trace-step{flex:0 0 125px;background:#111f26;border:1px solid var(--border);border-radius:8px;padding:11px;text-align:left;min-height:70px}.trace-step b{display:block;font:10px var(--mono);color:var(--cyan);margin-bottom:8px}.trace-step span{font-size:10px;color:var(--muted)}.trace-step.focus{border-color:var(--mint);background:#1c2c23}pre{max-height:300px;overflow:auto;padding:13px;border:1px solid var(--border);border-radius:8px;background:#0a1318}.footer{font:9px/1.8 var(--mono);color:#728992;overflow-wrap:anywhere;margin-top:20px}.record-context{font:10px var(--mono);color:var(--violet);margin:12px 0 0}.record-context:empty{display:none}
@media(max-width:1100px){.controls{grid-template-columns:1fr 1fr}.run-actions{grid-column:1/-1}.runbar{flex-direction:column;align-items:flex-start;gap:13px}.system-head,.section,.stats{padding-left:17px;padding-right:17px}.algorithm{margin-left:17px;margin-right:17px}.system h2{font-size:21px}.stat strong{font-size:21px}.system-emblem{width:36px;height:36px}.system-head{gap:9px}}
@media(max-width:760px){.run-setup{padding:17px}.controls{grid-template-columns:1fr;gap:14px}.run-actions{grid-column:auto}.run-actions .primary{flex:1}.setup-top{gap:10px}.setup-top a{font-size:9px}.columns{grid-template-columns:1fr}.playback{width:100%;flex-wrap:wrap}.playback input{flex:1;min-width:70px;width:80px}.playback .button{font-size:10px;padding:7px}.provider-setup{flex-direction:column;align-items:flex-start;gap:10px}.supporting{grid-template-columns:1fr}.system-head{padding:20px 17px}.algorithm{padding:14px 11px}.stats{gap:8px}.stage{font-size:10px}.outcome{margin-left:17px;margin-right:17px}.evidence>summary{padding:15px 17px}.evidence-count{font-size:8px}.notice{font-size:9px}}
</style></head><body><main id="comparison-app" class="studio-shell">
__NAV__
<header class="studio-hero"><div><div class="eyebrow">The comparison lab</div><h1>One failure.<br><em>Two attributions.</em></h1><p>See how AgentRx and AgentReroute reason through the same trace.<br>Follow the evidence, compare the selected steps, inspect exact-match scoring.</p></div><div class="hero-aside"><div><strong>Same input</strong>A shared starting point</div><i></i><div><strong>Real inference</strong>Every decision on record</div></div></header>
<section class="run-setup" aria-label="Configure comparison"><div class="setup-top"><span class="eyebrow-label">Configure an experiment</span><a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener noreferrer">Get a Gemini API key ↗</a></div><form id="run-form" class="controls"><div class="field"><label for="scenario">Published benchmark case</label><select id="scenario" required><option value="">Loading published cases…</option></select></div><div class="field"><label for="model">Model for both engines</label><select id="model" required><option value="">Loading Gemini settings…</option></select></div><div class="field"><label for="variant">AgentReroute checks</label><select id="variant"><option value="seeds_select">Fixed checks</option><option value="synth_select_p10">Written + fixed checks</option></select></div><div class="field"><label for="policy-mode">Candidate policy</label><select id="policy-mode"><option value="untrained">Without RL training · reported</option><option value="trained">With RL training · experimental</option></select></div><div class="run-actions"><button id="start-run" class="button primary" type="submit" disabled>Run comparison ↗</button><button id="cancel-run" class="button" type="button" disabled>Stop</button></div></form><p class="notice">Published trace · Frozen policy during inference · The trained checkpoint is experimental and predates the latest results · Gemini uses your project's quota</p><div class="provider-setup" id="provider-setup" hidden><div><strong>Connect Gemini to start</strong><p id="provider-message">Save GEMINI_API_KEY in the project's .env file.</p></div><button id="refresh-provider" class="button" type="button">Refresh settings ↻</button></div><p id="record-context" class="record-context"></p></section>
<div id="error" class="error" role="alert"></div>
<div class="runbar"><div><h2>Execution workspace</h2><p class="status-line" id="run-status" aria-live="polite">Ready when you are. Both systems receive the same recorded input.</p></div><div class="playback" aria-label="Shared event playback"><button class="button small" id="previous" type="button" aria-label="Previous event">←</button><button class="button small" id="play" type="button">Play events</button><button class="button small" id="next" type="button" aria-label="Next event">→</button><input id="cursor" type="range" min="0" max="0" value="0" aria-label="Event timeline"><span id="cursor-label">0 / 0 events</span><button class="button small" id="go-live" type="button">Go live</button></div></div>
<div class="columns" id="systems"></div>
<div class="supporting">
<details class="performance"><summary>Performance breakdown <span>Timings & token usage</span></summary><div class="table-scroll"><table><thead><tr><th>THIS EXECUTION</th><th>AGENTRX</th><th>AGENTREROUTE</th></tr></thead><tbody id="metrics"></tbody></table></div><p>Observed measurements for this run; policy scores are not calibrated failure probabilities.</p></details>
<details class="input-run"><summary>Shared input <span>Original messages & tools</span></summary><p class="source-meta" id="input-meta">The published trace appears here when a comparison starts.</p><div class="trace" id="trace"></div><pre id="trace-detail">Select a step to inspect its messages and tool calls.</pre></details>
<details class="details-panel" id="event-inspector"><summary>Event inspector <span>Selected model response</span></summary><pre id="event-detail">Start a comparison to inspect its events.</pre></details>
<details class="method"><summary>How to read this comparison</summary><p>Both systems receive the same published trace and matched Gemini 3.1 Flash Lite settings. AgentRx uses its unchanged joint judge. AgentReroute applies three perspectives, prefix-only checks and a frozen candidate ranking, then a full-trace model selector chooses the final attribution. The selector may choose a step outside the evidence shortlist.</p><p>Without RL training uses the untrained THETA0 weights from the latest reported method. With RL training loads an older trained checkpoint as an exploratory comparison; its results are not the reported headline and the selector still makes the final decision. Neither mode trains during a run. Fixed checks are the default; written checks add model-generated constraints. Gold labels are added only after both predictions for exact step, agent and category scoring. This comparison performs no repair or policy update. Failed calls and fallbacks remain visible. These single-run observations do not establish benchmark accuracy or significance.</p><p>Saved legacy recovery runs retain their original events and verification results, and are labeled historical.</p></details>
</div><footer class="footer" id="artifacts">Requests, attribution and scoring artifacts are saved locally for each run.</footer>
</main><script>
'use strict';
const $=id=>document.getElementById(id), SIDES=['agentrx','agentreroute'];
const NAMES={agentrx:'AgentRx',agentreroute:'AgentReroute'};
const LABELS={agentrx:['Global constraints','Dynamic constraints','Guarded checks','Joint judge','Exact-match scoring'],agentreroute:['Three perspectives','Prefix checks','Evidence candidates','Full-trace selector','Exact-match scoring']};
const LEGACY_LABELS={agentrx:['Global constraints','Dynamic constraints','Guarded checks','Single judge','Shared repair','Verify outcome'],agentreroute:['P1 · Intent & plan','P2 · Tool execution','P3 · Handoff & facts','Policy ranking','Shared repair','Verify + update']};
const TERMINAL=['completed','partial','error','canceled'];
const S={configured:false,provider:'gemini',events:[],cursor:0,following:true,playing:true,job:null,status:'idle',timer:null,loaded:false,catalogReady:false,legacy:false,variant:'seeds_select',policyMode:'untrained',trace:null};
function text(tag,value,cls){const el=document.createElement(tag);el.textContent=value??'';if(cls)el.className=cls;return el;}
function fmt(v,unit=''){return typeof v==='number'?v.toLocaleString(undefined,{maximumFractionDigits:2})+unit:'—';}
function showError(message){$('error').textContent=message||'';}
function stageFor(side,e){
 const stage=e.stage||'';
 if(S.legacy){if(e.kind.startsWith('repair_')||stage.startsWith('replay_agent'))return 4;if(e.kind==='verification_completed')return 5;if(e.kind==='attribution_completed')return 3;if(e.kind==='checks_completed')return side==='agentrx'?2:null;if(side==='agentrx')return {global:0,dynamic:1,semantic:2,judge:3}[stage]??null;return {P1_INTENT_PLAN:0,P2_TOOL_EXECUTION:1,P3_HANDOFF_GROUNDING:2}[e.perspective]??null;}
 if(e.kind==='attribution_scored')return 4;
 if(e.kind==='attribution_completed')return 3;
 if(e.kind==='checks_completed')return side==='agentrx'?2:1;
 if(e.kind==='synthesis_summary')return side==='agentreroute'?0:null;
 if(side==='agentrx')return {global:0,dynamic:1,semantic:2,judge:3}[stage]??null;
 if(stage==='mp_rl_select')return 3;
 if(stage==='semantic')return 1;
 if(e.perspective||stage.startsWith('perspective_'))return 0;
 return null;
}
for(const side of SIDES){
 const card=text('section','',`system ${side==='agentreroute'?'reroute':''}`);card.id=side;card.setAttribute('aria-label',NAMES[side]+' execution');
 card.innerHTML=`<div class="system-head"><div class="system-emblem">${side==='agentrx'?'Rx':'__MARK__'}</div><div><h2>${NAMES[side]}</h2><span class="kicker" id="${side}-kicker"></span></div><span class="badge" id="${side}-status">Ready</span></div>
 <div class="algorithm" id="${side}-stages" aria-label="Algorithm stages"><div class="flow-caption" id="${side}-caption"></div><div class="flow-group" id="${side}-inputs"></div><div class="flow-connector" aria-hidden="true">↓</div><div class="flow-group" id="${side}-outputs"></div></div>
 <div class="stats"><div class="stat"><strong id="${side}-calls">—</strong><span>Model calls</span></div><div class="stat"><strong id="${side}-time">—</strong><span>Model time</span></div><div class="stat"><strong id="${side}-tokens">—</strong><span>Output tokens</span></div><div class="stat"><strong id="${side}-violations">—</strong><span>Violations</span></div></div>
 <div class="section attribution"><h3 class="section-title" id="${side}-prediction-label">Selected attribution</h3><div class="outcome" id="${side}-prediction" style="margin:0 0 18px"></div><h3 class="section-title" id="${side}-candidates-label">Evidence shortlist</h3><div class="candidates" id="${side}-candidates"></div></div>
 <div class="outcome" id="${side}-outcome"></div><div class="warning" id="${side}-warning" role="status"></div>
 <details class="evidence" id="${side}-evidence"><summary>Inspect evidence <span class="evidence-count" id="${side}-evidence-count"></span></summary><div class="section"><h3 class="section-title">Checks & evidence</h3><div class="checks-line" id="${side}-checks">No checks completed yet.</div></div><div class="section"><h3 class="section-title">All model requests · select to inspect</h3><div class="log" id="${side}-log"></div></div></details>`;
 $('systems').append(card);
}
function renderStages(side,p){
 const holder=$(side+'-stages'),mode=S.legacy?'legacy':'current';
 if(holder.dataset.mode!==mode){
  holder.dataset.mode=mode;$(side+'-inputs').replaceChildren();$(side+'-outputs').replaceChildren();
  (S.legacy?LEGACY_LABELS:LABELS)[side].forEach((label,i)=>{const el=text('div','','stage');el.id=side+'-stage-'+i;el.append(text('i',String(i+1).padStart(2,'0')),text('span',label));$(side+(i<3?'-inputs':'-outputs')).append(el);});
 }
 $(side+'-kicker').textContent=S.legacy?'Historical recovery execution':side==='agentrx'?'Joint fault attribution':'Perspective evidence + full-trace selector';
 $(side+'-caption').textContent=S.legacy?'Historical stages · original run':side==='agentrx'?'Generate → check → judge → score':'Perspectives → checks → candidates → select → score';
 for(let i=0;i<(S.legacy?6:5);i++)$(side+'-stage-'+i).className='stage '+(p.stages[i]||'');
}
function fold(){
 const states=Object.fromEntries(SIDES.map(s=>[s,{status:'Ready',requests:[],metrics:{},checks:null,hypotheses:null,prediction:null,selection:null,score:null,verification:null,stages:{},result:null}]));
 for(const e of S.events.slice(0,S.cursor)){
  const p=states[e.system];if(!p)continue;
  if(e.kind==='system_started')p.status='running';
  if(e.kind==='request_started'){p.requests.push({...e,state:'running'});p.metrics.model_calls=(p.metrics.model_calls||0)+1;}
  if(e.kind==='request_completed'||e.kind==='request_failed'){const row=p.requests.find(r=>r.request_id===e.request_id);if(row)Object.assign(row,e,{state:e.kind==='request_failed'?'failed':'completed'});}
  if(e.metrics)p.metrics={...e.metrics};
  const observed=p.requests.find(r=>r.request_id===e.request_id),stage=stageFor(e.system,{...e,perspective:e.perspective||observed?.perspective});
  if(stage!==null){if(e.kind==='request_started'||e.kind==='repair_started')p.stages[stage]='active';else if(e.kind==='request_failed')p.stages[stage]='failed';else if(p.stages[stage]!=='failed')p.stages[stage]='done';}
  if(e.kind==='synthesis_summary'){p.synthesis=e.summary;if(e.summary?.failed_perspectives?.length){if(S.legacy){for(const perspective of e.summary.failed_perspectives){const i=stageFor(e.system,{kind:e.kind,perspective});if(i!==null)p.stages[i]='failed';}}else p.stages[0]='failed';}}
  if(e.kind==='checks_completed'){p.checks=e.counts;p.seed_checks=e.seed_checks;p.checkRows=e.rows;if(!S.legacy&&e.system==='agentreroute'&&p.stages[0]!=='failed')p.stages[0]='done';}
  if(e.kind==='attribution_completed'){p.hypotheses=e.ranked_candidates??e.hypotheses??[];p.prediction=e.prediction;p.selection=e.selection;p.diagnosis_seconds=e.diagnosis_seconds;if(!S.legacy&&e.system==='agentreroute')p.stages[2]='done';}
  if(e.kind==='attribution_scored')p.score=e.score;
  if(e.kind==='repair_started')p.directive=e.directive;
  if(e.kind==='verification_completed')p.verification=e;
  if(e.kind==='system_finished'){
   p.status=e.result.status;p.result=e.result;p.metrics=e.result.metrics||p.metrics;
   p.error=e.result.error;p.prediction=e.result.prediction??p.prediction;p.selection=e.result.selection??p.selection;p.score=e.result.score??p.score;p.synthesis=e.result.synthesis??p.synthesis;
   p.hypotheses=e.result.ranked_candidates??p.hypotheses;
   for(const i of Object.keys(p.stages))if(p.stages[i]==='active')p.stages[i]=p.status==='error'?'failed':'done';
  }
 }
 return states;
}
function renderPrediction(side,p){
 const box=$(side+'-prediction');box.replaceChildren();
 $(side+'-prediction-label').textContent=S.legacy?'Recorded attribution':'Selected attribution';
 $(side+'-candidates-label').textContent=S.legacy?'Recorded fault candidates':side==='agentreroute'?'Evidence shortlist · '+(S.policyMode==='trained'?'trained checkpoint (frozen)':'untrained THETA0 (frozen)'):'Judge evidence candidates';
 if(!p.prediction){box.append(text('p',p.hypotheses===null?'Waiting for attribution. No prediction has been made.':'No attribution returned.','empty'));return;}
 const v=p.prediction;box.append(text('strong',`${v.agent} · step ${v.step_index}`),text('div',v.category||v.mode||''));if(v.rationale)box.append(text('div',v.rationale));
 if(!S.legacy&&side==='agentreroute')box.append(text('div',p.selection?.fell_back?'Selector fallback · frozen-policy answer':p.selection?.in_pool===false?'Full-trace selector · selected outside the evidence shortlist':'Full-trace selector decision'));
}
function candidates(side,p){
 const box=$(side+'-candidates');box.replaceChildren();
 if(p.hypotheses===null){box.append(text('p','Waiting for attribution. No evidence shortlist yet.','empty'));return;}
 if(!p.hypotheses.length){box.append(text('p','No ranked evidence candidates recorded.','empty'));return;}
 p.hypotheses.slice(0,3).forEach((c,i)=>{
  const row=text('div','','candidate'),body=text('div');body.append(text('b',`${c.raw_agent||c.agent} · step ${c.step_index}`),text('small',c.mode||c.category||c.dominant_category));
  const value=c.probability??c.confidence,score=text('div',value==null?'—':fmt(value),'score');score.append(text('small',side==='agentreroute'?'policy weight':'reported score'));
  if(typeof value==='number')row.style.setProperty('--score',Math.max(0,Math.min(1,value))*100+'%');row.append(text('span',String(i+1).padStart(2,'0'),'rank'),body,score);box.append(row);
 });
 if(!S.legacy&&side==='agentreroute')box.append(text('p',`Top ${Math.min(3,p.hypotheses.length)} of ${p.hypotheses.length} evidence candidates. Policy weights are not confidence in the selected answer.`,'empty'));
}
function renderTrace(){
 const input=S.events.slice(0,S.cursor).find(e=>e.kind==='input_ready');
 if(!input){$('trace').replaceChildren();$('trace').dataset.hash='';$('input-meta').textContent='The published trace appears here when a comparison starts.';$('trace-detail').textContent='Select a step to inspect its messages and tool calls.';return;}
 S.trace=input.trace;const steps=input.trace.steps||input.frames||[],hash=input.input_hash||S.job;
 $('input-meta').textContent=`${steps.length} recorded steps · Input ${String(hash).slice(0,12)} · Identical input for both systems`;
 const holder=$('trace');if(holder.dataset.hash===hash)return;holder.dataset.hash=hash;holder.replaceChildren();
 for(const step of steps){const event=step.substeps?.[0]||step,b=text('button','','trace-step');b.type='button';b.append(text('b',`#${step.index??step.step_index} · ${event.role||step.agent||'event'}`),text('span',event.name||event.tool_calls?.[0]?.name||String(event.content||step.kind||'').slice(0,35)));b.onclick=()=>{holder.querySelectorAll('button').forEach(n=>n.classList.remove('focus'));b.classList.add('focus');$('trace-detail').textContent=JSON.stringify(step,null,2);};holder.append(b);}
}
function renderChecks(side,p){
 const checks=$(side+'-checks');checks.replaceChildren();
 if(p.checks){for(const key of ['SAT','VIOL','SKIP','ERROR'])checks.append(text('span',`${key} ${p.checks[key]??0}`,key==='VIOL'?'violation':''));checks.append(text('span',`${p.seed_checks||0} seed evaluations`));}else checks.append(text('span','No checks completed yet.'));
 const v=p.synthesis;
 if(v){checks.append(text('span',`${v.llm_constraints??0} model rules · ${v.llm_conclusive_checks??0} conclusive model checks`));if(v.state==='invalid_synthesis_fallback')checks.append(text('span','Seed-only fallback: model rules rejected or generation failed.','violation'));else if(v.state==='partial_synthesis')checks.append(text('span','Some model rules failed validation.','violation'));else if(v.state==='valid_empty_synthesis')checks.append(text('span','Model returned valid empty rule sets.'));if(v.rejected_constraints)checks.append(text('span',`${v.rejected_constraints} rejected rule outputs.`,'violation'));}
 const warnings=[];
 if(v?.state==='invalid_synthesis_fallback')warnings.push('Seed-only fallback: model rules rejected or generation failed.');
 if(v?.state==='partial_synthesis')warnings.push('Some model rules failed validation.');
 if(p.selection?.fell_back||['error','invalid_prediction','out_of_pool'].includes(p.selection?.status))warnings.push('Selector fallback: the frozen-policy answer was used. '+(p.selection.error||''));
 if(p.requests.some(r=>r.state==='failed'))warnings.push('A model request failed. Inspect the recorded response.');
 if(p.checks?.ERROR)warnings.push(`${p.checks.ERROR} checker errors; evidence is incomplete.`);
 const reasons=p.result?.degradation_reasons||[];for(const reason of reasons)if(reason)warnings.push(String(reason));
 if(p.status==='partial'&&!warnings.length)warnings.push('Partial result. Inspect the execution evidence before interpreting this attribution.');
 $(side+'-warning').textContent=[...new Set(warnings)].join(' ');
}
function renderOutcome(side,p){
 const out=$(side+'-outcome');out.replaceChildren();
 if(S.legacy&&p.verification){const v=p.verification,goal=v.task_outcome.after.reward===1;out.append(text('strong','Historical recovery · '+(goal?'Task met':'Task not met'),goal?'good':'bad'));out.append(text('div',`Repair: ${v.verification.accepted?'accepted':'not accepted / no intervention'} · Decision: ${v.decision.status}.`));out.append(text('div',`State ${v.task_outcome.after.state_ok?'matched':'not matched'} · Required outputs ${v.task_outcome.after.missing_outputs.length?'missing':'present'}.`));if(v.policy_update)out.append(text('div',`Historical policy update · Feedback reward ${fmt(v.policy_update.reward)}`));}
 else if(S.legacy&&p.directive)out.append(text('strong',`Historical rollback to step ${p.directive.rollback_step}`),text('div',p.directive.text));
 else if(p.score){const v=p.score;out.append(text('strong',v.step_correct?'Exact step match':'Step mismatch',v.step_correct?'good':'bad'));out.append(text('div',`Predicted step ${v.predicted_step} · Gold step ${v.gold_step}`));out.append(text('div',`Agent ${v.agent_correct?'correct':'incorrect'} · Category ${v.category_correct?'correct':'incorrect'}`));out.append(text('div','Gold revealed after both predictions.'));}
 else out.append(text('p',p.prediction?'Prediction recorded. Waiting for post-prediction scoring.':'Waiting for attribution.','empty'));
 if(p.error)out.append(text('div',p.error,'bad'));
}
function render(){
 const panels=fold();
 for(const side of SIDES){
  const p=panels[side],badge=$(side+'-status');badge.textContent=p.status==='Ready'&&S.job?(TERMINAL.includes(S.status)&&S.cursor===S.events.length?'Not run':'Queued'):p.status.replaceAll('_',' ');badge.className='badge '+p.status;renderStages(side,p);
  $(side+'-calls').textContent=fmt(p.metrics.model_calls);$(side+'-tokens').textContent=fmt(p.metrics.completion_tokens);$(side+'-time').textContent=fmt(p.metrics.model_seconds,'s');$(side+'-violations').textContent=p.checks?fmt(p.checks.VIOL):'—';
  renderPrediction(side,p);candidates(side,p);renderChecks(side,p);renderOutcome(side,p);
  $(side+'-evidence-count').textContent=p.requests.length?`${p.requests.length} calls · ${p.checks?Object.values(p.checks).reduce((a,b)=>a+b,0):'—'} checks`:'';
  const log=$(side+'-log');log.replaceChildren();
  for(const request of p.requests){const button=text('button','','log-row '+(request.state==='failed'?'failed':''));button.type='button';button.append(text('b',`${String(request.request_id).padStart(2,'0')}  ${request.stage}`),text('span',` · ${request.state}${request.seconds!=null?' · '+fmt(request.seconds,'s'):''}`));button.onclick=()=>{S.following=false;S.playing=false;S.cursor=S.events.findIndex(e=>e.seq===request.seq)+1;render();$('event-inspector').open=true;$('event-inspector').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'nearest'});};log.append(button);}
  if(!p.requests.length)log.append(text('p','Requests will appear here as they are sent.','empty'));
 }
 const rows=[['Attribution elapsed',p=>fmt(p.diagnosis_seconds,'s')],['Total elapsed',p=>fmt(p.result?.metrics?.elapsed_seconds,'s')],['Model requests',p=>fmt(p.metrics.model_calls)],['Input tokens',p=>fmt(p.metrics.prompt_tokens)],['Output tokens',p=>fmt(p.metrics.completion_tokens)],['Check violations',p=>p.checks?fmt(p.checks.VIOL):'—']];
 if(S.legacy)rows.push(['Historical task met after intervention',p=>p.verification?(p.verification.task_outcome.after.reward===1?'Yes':'No'):'—']);else rows.push(['Exact step match',p=>p.score?(p.score.step_correct?'Yes':'No'):'—'],['Agent match',p=>p.score?(p.score.agent_correct?'Yes':'No'):'—'],['Category match',p=>p.score?(p.score.category_correct?'Yes':'No'):'—']);
 $('metrics').replaceChildren();for(const [name,value] of rows){const tr=text('tr');tr.append(text('td',name),...SIDES.map(s=>text('td',value(panels[s]))));$('metrics').append(tr);}
 $('cursor').max=S.events.length;$('cursor').value=S.cursor;$('cursor-label').textContent=`${S.cursor} / ${S.events.length} events`;$('play').textContent=S.cursor===S.events.length&&S.events.length?'Replay':S.playing&&!S.following?'Pause':'Play events';$('play').disabled=!S.events.length;$('previous').disabled=S.cursor===0;$('next').disabled=S.cursor===S.events.length;
 $('run-status').textContent=S.job?`${S.status}${S.legacy?' · historical legacy recovery':''}${S.following?' · following recorded events':' · historical playback'}${S.status==='running'?' · model calls can take several minutes':''}`:'Ready when you are. Both systems receive the same published input.';
 const e=S.events[S.cursor-1];$('event-detail').textContent=e?JSON.stringify(e,null,2):'No event selected.';renderTrace();if(window.storyGuide)window.storyGuide(comparisonGuide(panels));
}
function comparisonGuide(panels){
 if(!S.cursor)return {title:'Same input. Two approaches.',body:'AgentRx uses its joint judge. AgentReroute checks three perspectives and lets a model select any step in the full trace. The selected candidate policy stays frozen.'};
 const viewed=S.events.slice(0,S.cursor),last=viewed.at(-1);
 if(last.kind==='job_error')return {title:'This run needs attention.',body:last.error||'The comparison encountered an error. Completed requests remain inspectable.'};
 if(last.kind==='job_finished')return last.status==='canceled'?{title:'The comparison was stopped.',body:'Completed requests are preserved. Unfinished engines have no result.'}:S.legacy?{title:'Inspect this historical recovery run.',body:'The saved verification results describe the original execution and its original method.'}:{title:'Compare the selected steps.',body:last.status==='partial'?'This comparison is partial. Inspect failed calls and fallback warnings before interpreting its scores.':'The final attribution is separate from the evidence shortlist. Gold labels score the predictions only after both decisions.'};
 const side=[...viewed].reverse().find(e=>e.system)?.system,p=panels[side];if(!p)return {title:'Start with the same published input.',body:'Inspect the original messages and tools under Shared input.'};
 const name=NAMES[side];
 if(p.error)return {title:name+' needs attention.',body:p.error};
 if(S.legacy)return {title:'Historical execution · '+name,body:'These saved events preserve the original attribution, repair and verification behavior. Inspect the event record for the original method.'};
 if(p.score)return {title:name+' has been scored.',body:'Exact step match is the primary outcome. Agent and category matches are shown separately. Gold labels were added after prediction.'};
 if(p.selection?.fell_back)return {title:'The selector fell back.',body:'The model selector could not supply a valid answer. The panel shows the frozen-policy fallback and the recorded error.'};
 if(p.synthesis?.failed_stages)return {title:'Some generated checks need attention.',body:'Rule generation failed for part of this run. Inspect the fallback or partial-result warning.'};
 if(p.prediction)return {title:name+' has selected its attribution.',body:'The prominent selected answer is the prediction. Evidence candidates and policy weights below it do not override that decision.'};
 const request=p.requests.at(-1),stage=request?.stage||'';
 if(request?.state==='failed')return {title:'A model request failed.',body:'The error remains visible in the evidence and contributes to the measured run.'};
 if(stage==='mp_rl_select')return {title:'The selector reads the full trace.',body:'The candidate pool is fallible evidence. The model can choose a step outside it; the weights remain frozen.'};
 if(stage==='judge')return {title:'AgentRx is weighing the evidence.',body:'The joint judge receives the trace and checks, then selects the fault step, agent and category.'};
 return {title:name+' is building its evidence.',body:'Constraints are checked against observed prefixes. The next decision uses this evidence to attribute the failure.'};
}
async function api(path,options={}){const response=await fetch(path,options);let data;try{data=await response.json();}catch{throw Error('The local server returned an unreadable response.');}if(!response.ok)throw Error(data.error||`Request failed (${response.status})`);return data;}
function busy(value){$('start-run').disabled=value||!S.loaded||!S.configured||!S.catalogReady||!!$('model').selectedOptions[0]?.disabled||!!$('scenario').selectedOptions[0]?.disabled;$('cancel-run').disabled=!value;$('scenario').disabled=value;$('model').disabled=value;$('variant').disabled=value;$('policy-mode').disabled=value;}
function recordedOption(id,value,label){if(value&&![...$(id).options].some(o=>o.value===value)){const o=text('option',label||value+' (recorded run)');o.value=value;o.disabled=true;$(id).append(o);}if(value)$(id).value=value;}
async function poll(){
 if(!S.job)return;
 try{
  const data=await api('/api/comparisons/'+S.job+'?after='+S.events.length);if(new URLSearchParams(location.search).get('run')!==data.id)history.replaceState(null,'',location.pathname+'?run='+data.id);
  S.status=data.status;S.legacy=data.legacy===true||data.mode==='legacy_recovery';S.variant=data.variant||'seeds_select';S.policyMode=data.policy_mode||'untrained';
  $('record-context').textContent=(S.legacy?'Historical legacy recovery · original events and metadata':('Published attribution · '+(S.policyMode==='trained'?'trained checkpoint (frozen)':'untrained THETA0 (frozen)')))+` · ${data.provider||'recorded'} · ${data.model}`;
  recordedOption('scenario',data.scenario);recordedOption('model',data.model);$('variant').value=S.variant;$('policy-mode').value=S.policyMode==='trained'?'trained':'untrained';
  for(const e of data.events)if(!S.events.some(x=>x.seq===e.seq))S.events.push(e);if(S.following)S.cursor=S.events.length;
  $('artifacts').textContent='Artifacts: '+data.artifact_dir;busy(!TERMINAL.includes(data.status));
  if(data.cancel_requested)showError(data.status==='canceled'?'Stopped. Completed requests remain available for inspection.':'Stop requested. The current model call will finish; no further model requests will be sent.');
  if(data.status==='error'){const error=S.events.findLast(e=>e.kind==='job_error');if(error)showError(error.error);}
  render();if(!TERMINAL.includes(data.status))S.timer=setTimeout(poll,900);
 }catch(error){showError('Could not retrieve this run: '+error.message);busy(false);}
}
$('run-form').onsubmit=async e=>{
 e.preventDefault();if($('start-run').disabled)return;busy(true);showError('');
 try{const job=await api('/api/comparisons',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({scenario:$('scenario').value,model:$('model').value,variant:$('variant').value,policy_mode:$('policy-mode').value})});clearTimeout(S.timer);S.job=job.id;S.events=[];S.cursor=0;S.following=true;S.playing=true;S.status='running';S.legacy=false;S.variant=$('variant').value;S.policyMode=$('policy-mode').value;$('trace').dataset.hash='';sessionStorage.setItem('comparison-job',job.id);history.replaceState(null,'',location.pathname+'?run='+job.id);render();poll();}catch(error){showError(error.message);busy(false);}
};
$('cancel-run').onclick=async()=>{try{await api('/api/comparisons/'+S.job+'/cancel',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});showError('Stop requested. Waiting for the current model call to finish.');}catch(error){showError(error.message);}};
function move(value){S.cursor=Math.max(0,Math.min(S.events.length,value));S.following=false;S.playing=false;render();}
$('previous').onclick=()=>move(S.cursor-1);$('next').onclick=()=>move(S.cursor+1);$('cursor').oninput=e=>move(Number(e.target.value));$('go-live').onclick=()=>{S.following=true;S.playing=true;S.cursor=S.events.length;render();};$('play').onclick=()=>{if(S.cursor===S.events.length&&S.events.length){S.cursor=0;S.playing=true;}else S.playing=S.following||!S.playing;S.following=false;render();};setInterval(()=>{if(S.playing&&!S.following&&S.cursor<S.events.length){S.cursor++;render();}},800);
async function loadModels(){
 try{const data=await api('/api/models');S.provider=data.provider||'gemini';S.configured=data.configured!==false;S.loaded=true;
  const selected=$('model').value,models=(data.models||[]).filter(name=>S.provider==='test'||name==='gemini-3.1-flash-lite');$('model').replaceChildren();for(const name of models){const o=text('option',name);o.value=name;$('model').append(o);}if(models.includes(selected))$('model').value=selected;else if(models.includes(data.default_model))$('model').value=data.default_model;
  if(!models.length){S.configured=false;showError('This comparison requires gemini-3.1-flash-lite. Check the local model configuration.');}
  $('provider-setup').hidden=S.configured;$('provider-message').textContent=data.message||'Save GEMINI_API_KEY in the project .env file, then refresh settings.';$('connection').textContent=S.provider==='test'?'Test transport':S.configured?'Gemini configured':'Gemini · key needed';busy(S.status==='running');
 }catch(error){S.configured=false;S.loaded=false;$('provider-setup').hidden=false;$('provider-message').textContent=error.message;$('connection').textContent='Settings unavailable';busy(false);}
}
$('refresh-provider').onclick=loadModels;
async function init(){
 render();
 try{const data=await api('/api/comparison-cases'),rows=Array.isArray(data)?data:(data.cases||[]);$('scenario').replaceChildren();for(const row of rows){const o=text('option',row.title);o.value=row.id;$('scenario').append(o);}S.catalogReady=rows.length>0;if(!rows.length){const o=text('option','Published corpus unavailable');o.value='';$('scenario').append(o);showError(data.message||'Published benchmark cases are unavailable. Install the authors’ published corpus at data/benchmark with cases.json, then reload. Saved runs can still be inspected.');}}
 catch(error){showError('Could not load published cases: '+error.message);}
 await loadModels();const saved=new URLSearchParams(location.search).get('run')||sessionStorage.getItem('comparison-job');if(saved){S.job=saved;await poll();}
}
for(const id of ['model','scenario'])$(id).onchange=()=>{if(S.status!=='running')busy(false);};init();
</script></body></html>'''

PAGE = PAGE.replace('__STUDIO_THEME__', BASE_CSS).replace('__NAV__', navigation('compare')).replace('__MARK__', MARK)
