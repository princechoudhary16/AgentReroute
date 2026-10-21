"""Self-contained HTML dashboard built from a corpus and scoring reports.

No server and no external scripts: data is embedded as JSON and drawn with
inline SVG, so the file opens offline in any browser and can be shared as-is.
The dashboard is built after scoring, from public traces, gold labels and
reports; nothing in it flows back into a system under test.
"""
import html
import json
from pathlib import Path

from . import BENCHMARK_NAME, __version__
from .io import read_json

MAX_TRACES = 400
MAX_CONTENT = 600


def _corpus_rows(corpus_dir):
    corpus_dir = Path(corpus_dir)
    rows, traces = [], {}
    manifest = read_json(corpus_dir / 'manifest.json')
    for split in [*manifest.get('splits', {}), *manifest.get('external', {})]:
        gold_path = corpus_dir / 'gold' / f'{split}.json'
        if not gold_path.exists():
            continue
        for g in read_json(gold_path):
            tid = g['trajectory_id']
            trace_path = corpus_dir / 'public' / 'output_only' / split / f'{tid}.json'
            trace = read_json(trace_path) if trace_path.exists() else {'steps': []}
            inj = g.get('injection') or {}
            rows.append({'tid': tid, 'split': split, 'status': g['status'],
                         'mode': (g['critical'] or {}).get('mode') if g['critical'] else None,
                         'source': g.get('source'), 'method': inj.get('method'), 'variant': inj.get('variant'),
                         'timing': inj.get('timing'), 'n_steps': len(trace['steps']),
                         'critical_step': (g['critical'] or {}).get('step_index') if g['critical'] else None,
                         'critical_agent': (g['critical'] or {}).get('agent') if g['critical'] else None,
                         'fault_step': g.get('fault_step'), 'acceptable': g.get('acceptable_steps', []),
                         'measurement_kind': g.get('measurement_kind')})
            if len(traces) < MAX_TRACES:
                traces[tid] = [{'i': s['index'], 'role': s['substeps'][0]['role'],
                                'name': s['substeps'][0].get('name'),
                                'text': '\n'.join(e['content'] for e in s['substeps'])[:MAX_CONTENT]}
                               for s in trace['steps']]
    return rows, traces


def _attribution(report):
    keep = ('trajectory_id', 'pred_step', 'pred_agent', 'pred_mode', 'confidence', 'step_exact', 'agent_correct',
            'mode_correct', 'state')
    return {'system': report['system'], 'view': report.get('view'), 'split': report.get('split'),
            'denominator': report['denominator'], 'metrics': report['metrics'], 'confusion': report['confusion'],
            'calibration': report['calibration'], 'detection': {k: v for k, v in report.get('detection', {}).items()
                                                                if k != 'per_trajectory'} or None,
            'rows': [{k: r.get(k) for k in keep} for r in report['per_trajectory']]}


def _recovery(report):
    return {'policy': report['policy'], 'split': report['split'], 'n_failed': report['n_failed'],
            'n_controls': report['n_controls'], 'metrics': report['metrics'], 'by_mode': report['by_mode']}


def build_dashboard(corpus_dir, attribution_reports=(), recovery_reports=(), out_path='dashboard.html', title=None,
                    robustness=None):
    manifest = read_json(Path(corpus_dir) / 'manifest.json')
    rows, traces = _corpus_rows(corpus_dir)
    offline = any(r['measurement_kind'] == 'offline_fixture' for r in rows)
    data = {'meta': {'benchmark': BENCHMARK_NAME, 'version': __version__, 'offline': offline,
                     'heldout_variants': manifest.get('heldout_variants', []), 'dropped': manifest.get('dropped', {})},
            'rows': rows, 'traces': traces,
            'attribution': [_attribution(read_json(p) if not isinstance(p, dict) else p) for p in attribution_reports],
            'recovery': [_recovery(read_json(p) if not isinstance(p, dict) else p) for p in recovery_reports],
            'robustness': robustness}
    payload = json.dumps(data, ensure_ascii=False).replace('</', '<\\/')
    page = TEMPLATE.replace('__TITLE__', html.escape(title or 'AgentReroute benchmark')).replace('__DATA__', payload)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding='utf-8')
    return out_path


TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f7f7f5;--surface:#ffffff;--surface2:#f0efec;--text:#1b1b1a;--muted:#6b6a65;--border:#e1e0d9;
--accent:#2a78d6;--good:#0f7b4f;--bad:#c23b3b;--warn:#a86b00;
--m1:#2a78d6;--m2:#eb6834;--m3:#1baf7a;--m4:#eda100;--m5:#e87ba4;--m6:#6250d6;--m7:#e34948;--m8:#8a8984;--m9:#a3a29b;--m10:#d6d5cf}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141413;--surface:#1d1d1c;--surface2:#262624;--text:#f0efec;
--muted:#a3a29b;--border:#343431;--accent:#3987e5;--good:#3fbf86;--bad:#e66767;--warn:#e0a52a;
--m1:#3987e5;--m2:#d95926;--m3:#199e70;--m4:#c98500;--m5:#d55181;--m6:#9085e9;--m7:#e66767;--m8:#77766f;--m9:#5a5954;--m10:#444441}}
:root[data-theme="dark"]{--bg:#141413;--surface:#1d1d1c;--surface2:#262624;--text:#f0efec;--muted:#a3a29b;--border:#343431;
--accent:#3987e5;--good:#3fbf86;--bad:#e66767;--warn:#e0a52a;--m1:#3987e5;--m2:#d95926;--m3:#199e70;--m4:#c98500;--m5:#d55181;
--m6:#9085e9;--m7:#e66767;--m8:#77766f;--m9:#5a5954;--m10:#444441}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;font-weight:600;margin:0 0 4px}h2{font-size:17px;font-weight:600;margin:32px 0 12px}
h3{font-size:14px;font-weight:600;margin:0 0 8px}
.sub{color:var(--muted);margin:0 0 16px}
.banner{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:8px 12px;margin:12px 0;color:var(--warn)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
.kpi,.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.kpi .l{color:var(--muted);font-size:12px}.kpi .v{font-size:24px;font-weight:600;font-variant-numeric:tabular-nums}.kpi .s{color:var(--muted);font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.grid2{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px}
@media (max-width:820px){.grid2{grid-template-columns:minmax(0,1fr)}}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--border);white-space:nowrap}
th{color:var(--muted);font-weight:500;font-size:12px}td.n,th.n{text-align:right}
.scroll{overflow-x:auto}
.legend{display:flex;flex-wrap:wrap;gap:10px;font-size:12px;color:var(--muted);margin:6px 0}
.legend span{display:inline-flex;align-items:center;gap:4px}.sw{width:10px;height:10px;border-radius:2px;display:inline-block}
select,input{background:var(--surface);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:5px 8px;font:inherit}
.controls{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px}
.strip{display:flex;gap:3px;flex-wrap:wrap;margin:8px 0}
.strip button{min-width:34px;height:40px;border:1px solid var(--border);background:var(--surface);color:var(--text);
border-radius:6px;cursor:pointer;font-size:11px;padding:2px 4px;position:relative}
.strip button.gold{border:2px solid var(--good)}.strip button.fault{outline:2px dashed var(--warn);outline-offset:1px}
.strip button.acc{background:var(--surface2)}.strip button .p{position:absolute;left:3px;right:3px;bottom:3px;height:4px;border-radius:2px;background:var(--accent)}
.strip button.sel{box-shadow:0 0 0 2px var(--accent)}
pre{white-space:pre-wrap;word-break:break-word;background:var(--surface2);border-radius:8px;padding:10px;margin:8px 0 0;font-size:12px;max-height:280px;overflow:auto}
.ok{color:var(--good)}.no{color:var(--bad)}.muted{color:var(--muted)}
svg text{fill:var(--muted);font-size:11px}
.theme{float:right}
</style>
</head>
<body>
<main>
<button class="theme" id="theme" type="button">Toggle theme</button>
<h1>__TITLE__</h1>
<p class="sub" id="subtitle"></p>
<div id="banner"></div>
<section class="kpis" id="kpis"></section>

<h2>Corpus</h2>
<div class="grid2">
  <div class="card"><h3>Trajectories by split and label</h3><div class="legend" id="labelLegend"></div><div id="splitChart"></div></div>
  <div class="card"><h3>Where the critical failure falls (relative position in the run)</h3><div id="posChart"></div></div>
</div>
<div class="card" style="margin-top:16px"><h3>Failed trajectories by injection method and timing</h3><div class="scroll" id="methodTable"></div></div>

<h2>Attribution leaderboard</h2>
<div class="card scroll" id="leaderboard"></div>
<div class="card" style="margin-top:16px"><div class="controls"><label>System <select id="sysSel"></select></label>
    <span class="muted">Selection also drives the trace explorer and per-trajectory table.</span></div>
  <h3>Confusion (gold mode × predicted mode)</h3><div class="scroll" id="confusion"></div></div>
<div class="grid2" style="margin-top:16px">
  <div class="card"><h3>Calibration (top-1 confidence vs step accuracy)</h3><div id="calib"></div></div>
  <div class="card"><h3>Online detection</h3><div id="detect" class="muted"></div></div>
</div>

<h2>Recovery</h2>
<div class="grid2">
  <div class="card"><h3>Recovery rate on failed episodes (95% CI)</h3><div id="recChart"></div></div>
  <div class="card"><h3>Recovery vs cost (tokens per failed episode)</h3><div id="costChart"></div></div>
</div>
<div class="card scroll" style="margin-top:16px" id="recTable"></div>

<h2>Agent robustness (before recovery)</h2>
<div class="card scroll" id="robust"></div>

<h2>Trace explorer</h2>
<div class="card">
  <div class="controls">
    <select id="exSplit"></select><select id="exMode"></select><select id="exTrace"></select>
  </div>
  <div class="legend"><span><span class="sw" style="border:2px solid var(--good)"></span>gold critical step</span>
    <span><span class="sw" style="outline:2px dashed var(--warn)"></span>fault entered</span>
    <span><span class="sw" style="background:var(--surface2)"></span>acceptable step</span>
    <span><span class="sw" style="background:var(--accent)"></span>prediction of selected system</span></div>
  <div id="exInfo" class="muted"></div>
  <div class="strip" id="strip"></div>
  <div id="stepDetail"></div>
</div>

<h2>Per-trajectory results</h2>
<div class="card">
  <div class="controls"><select id="tSplit"></select><select id="tStatus"></select><select id="tMode"></select>
    <input id="tSearch" placeholder="Search id or variant"></div>
  <div class="scroll" id="traceTable"></div>
</div>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const MODES = ['instruction_adherence','drift','cycles','tool_failure','misinterpretation_of_tool_output','context_propagation','missing_details'];
const LABELS = [...MODES,'out_of_scope','unlabeled','tolerated','clean'];
const COLOR = Object.fromEntries(LABELS.map((l,i)=>[l,`var(--m${Math.min(i+1,10)})`]));
COLOR.unlabeled='var(--m8)';COLOR.out_of_scope='var(--m8)';COLOR.tolerated='var(--m9)';COLOR.clean='var(--m10)';
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pct = v => v == null ? '–' : (100*v).toFixed(0)+'%';
const num = v => v == null ? '–' : (Number.isInteger(+v) || Math.abs(v) >= 100 ? Math.round(v).toLocaleString() : (+v).toFixed(2));
const val = m => m == null ? null : (typeof m === 'object' ? m.value : m);
const label = r => r.status === 'failed' ? (r.mode || 'unlabeled') : r.status;
const short = s => String(s).replace('misinterpretation_of_tool_output','misinterpretation').replace('instruction_adherence','instruction');

$('theme').onclick = () => { const r=document.documentElement; const dark = r.dataset.theme ? r.dataset.theme==='dark' : matchMedia('(prefers-color-scheme: dark)').matches; r.dataset.theme = dark ? 'light' : 'dark'; };
const splits = [...new Set(D.rows.map(r=>r.split))];
$('subtitle').textContent = `${D.meta.benchmark} v${D.meta.version} · ${D.rows.length} trajectories · splits: ${splits.join(', ')}`;
if (D.meta.offline) $('banner').innerHTML = '<div class="banner">Includes offline fixture runs (scripted agents). These verify the pipeline; they are not model measurements.</div>';

// ---------- KPIs
const failed = D.rows.filter(r=>r.status==='failed');
const best = (arr, f) => arr.length ? arr.reduce((a,b)=> (f(b)??-1) > (f(a)??-1) ? b : a) : null;
const bA = best(D.attribution, a=>val(a.metrics.step_accuracy)), bR = best(D.recovery, r=>val(r.metrics.recovery_rate));
const kpis = [['Trajectories', D.rows.length, splits.length + ' splits'], ['Failed (labelled)', failed.length, 'denominator for attribution'],
  ['Controls', D.rows.length - failed.length, 'tolerated + clean runs'], ['Modes covered', new Set(failed.map(r=>r.mode).filter(Boolean)).size + ' / 7', ''],
  ['Best step accuracy', bA ? pct(val(bA.metrics.step_accuracy)) : '–', bA ? bA.system + ' · ' + (bA.split||'') : ''],
  ['Best recovery rate', bR ? pct(val(bR.metrics.recovery_rate)) : '–', bR ? bR.policy : '']];
$('kpis').innerHTML = kpis.map(([l,v,sub])=>`<div class="kpi"><div class="l">${esc(l)}</div><div class="v">${esc(v)}</div><div class="s" title="${esc(sub)}">${esc(sub)}</div></div>`).join('');

// ---------- SVG helpers
function svg(w,h,inner){return `<svg viewBox="0 0 ${w} ${h}" width="100%" role="img">${inner}</svg>`;}
function hbars(items, max, opts={}){ // items: [{label, segs:[{v,color,title}]}]
  const w=520, rowH=26, left=opts.left||110, h=items.length*rowH+24; let s='';
  items.forEach((it,i)=>{ let x=left; const y=i*rowH+4;
    s+=`<text x="${left-6}" y="${y+14}" text-anchor="end">${esc(it.label)}</text>`;
    it.segs.forEach(sg=>{ const bw=max? (w-left-40)*sg.v/max:0; if(bw>0) s+=`<rect x="${x}" y="${y}" width="${Math.max(bw-1,0)}" height="18" rx="3" fill="${sg.color}"><title>${esc(sg.title)}</title></rect>`; x+=bw; });
    s+=`<text x="${x+4}" y="${y+14}">${esc(it.total ?? '')}</text>`; });
  return svg(w,h,s);
}

// ---------- corpus
const present = LABELS.filter(l => D.rows.some(r=>label(r)===l));
$('labelLegend').innerHTML = present.map(l=>`<span><span class="sw" style="background:${COLOR[l]}"></span>${esc(short(l))}</span>`).join('');
const maxSplit = Math.max(1,...splits.map(s=>D.rows.filter(r=>r.split===s).length));
$('splitChart').innerHTML = hbars(splits.map(s=>{ const rs=D.rows.filter(r=>r.split===s);
  return {label:s, total:rs.length, segs: present.map(l=>({v:rs.filter(r=>label(r)===l).length, color:COLOR[l], title:`${short(l)}: ${rs.filter(r=>label(r)===l).length}`}))}; }), maxSplit, {left:130});
const bins = Array(10).fill(0); failed.forEach(r=>{ if(r.n_steps && r.critical_step!=null){ bins[Math.min(9,Math.floor(10*(r.critical_step-1)/Math.max(1,r.n_steps)))]++; }});
const bmax=Math.max(1,...bins); let ps='';
bins.forEach((b,i)=>{ const h=150*b/bmax; ps+=`<rect x="${30+i*46}" y="${170-h}" width="40" height="${h}" rx="3" fill="var(--accent)"><title>${b} trajectories</title></rect><text x="${50+i*46}" y="186" text-anchor="middle">${i*10}%</text><text x="${50+i*46}" y="${165-h}" text-anchor="middle">${b||''}</text>`; });
$('posChart').innerHTML = svg(500,195,ps+`<line x1="28" x2="492" y1="170" y2="170" stroke="var(--border)"/>`);
const methods=[...new Set(failed.map(r=>r.method||'natural/external'))], timings=['early','middle','late',null];
$('methodTable').innerHTML = `<table><tr><th>Method</th>${timings.map(t=>`<th class="n">${t||'n/a'}</th>`).join('')}<th class="n">Total</th></tr>`+
  methods.map(m=>{const rs=failed.filter(r=>(r.method||'natural/external')===m);return `<tr><td>${esc(m)}</td>${timings.map(t=>`<td class="n">${rs.filter(r=>(r.timing||null)===t).length||''}</td>`).join('')}<td class="n">${rs.length}</td></tr>`;}).join('')+'</table>';

// ---------- attribution
const ACOLS=[['step_accuracy','Step'],['step_within_1','Step ±1'],['acceptable_step_accuracy','Acceptable'],['top3_step_accuracy','Top-3 step'],['agent_accuracy','Agent'],['mode_accuracy','Mode'],['mode_macro_f1','Mode F1'],['step_ece','ECE']];
const ci = m => (m && typeof m==='object' && m.ci95 && m.ci95[0]!=null && m.ci95[0]!==m.ci95[1]) ? `<span class="muted"> [${pct(m.ci95[0])}–${pct(m.ci95[1])}]</span>` : '';
const cell = (k,m) => k==='mode_macro_f1'||k==='step_ece' ? num(val(m)) : pct(val(m))+ci(m);
$('leaderboard').innerHTML = D.attribution.length ? `<table><tr><th>System</th><th>Split</th><th class="n">n</th>${ACOLS.map(c=>`<th class="n">${c[1]}</th>`).join('')}</tr>`+
  [...D.attribution].sort((a,b)=>(val(b.metrics.step_accuracy)||0)-(val(a.metrics.step_accuracy)||0)).map(a=>`<tr><td>${esc(a.system)}</td><td>${esc(a.split||'')}</td><td class="n">${a.denominator}</td>${ACOLS.map(([k])=>`<td class="n">${cell(k,a.metrics[k])}</td>`).join('')}</tr>`).join('')+'</table>'
  : '<p class="muted">No attribution reports given. Run <code>python3 -m benchmark score</code> and pass the reports with <code>--attribution</code>.</p>';
D.attribution.sort((a,b)=>(val(b.metrics.step_accuracy)||0)-(val(a.metrics.step_accuracy)||0));
$('sysSel').innerHTML = D.attribution.map((a,i)=>`<option value="${i}">${esc(a.system)}${a.split?' · '+esc(a.split):''}</option>`).join('');
function drawSystem(){
  const a = D.attribution[+$('sysSel').value]; if(!a){$('confusion').innerHTML='<p class="muted">–</p>';return;}
  const order = k => { const i = MODES.indexOf(k); return i < 0 ? 99 : i; };
  const gold = Object.keys(a.confusion).sort((x,y)=>order(x)-order(y)), preds=[...new Set(gold.flatMap(g=>Object.keys(a.confusion[g])))].sort((x,y)=>order(x)-order(y));
  const mx = Math.max(1,...gold.flatMap(g=>Object.values(a.confusion[g])));
  $('confusion').innerHTML = `<table><tr><th>gold \\ pred</th>${preds.map(p=>`<th class="n">${esc(short(p))}</th>`).join('')}</tr>`+gold.map(g=>`<tr><td>${esc(short(g))}</td>${preds.map(p=>{const v=a.confusion[g][p]||0;return `<td class="n" style="background:color-mix(in srgb, var(--accent) ${Math.round(70*v/mx)}%, transparent)">${v||''}</td>`;}).join('')}</tr>`).join('')+'</table>';
  let s=`<line x1="40" y1="180" x2="220" y2="0" stroke="var(--border)" stroke-dasharray="4 4"/><line x1="40" y1="180" x2="220" y2="180" stroke="var(--border)"/><line x1="40" y1="0" x2="40" y2="180" stroke="var(--border)"/>`;
  (a.calibration||[]).forEach(b=>{ const x=40+180*b.confidence, y=180-180*b.accuracy; s+=`<circle cx="${x}" cy="${y}" r="${4+Math.min(8,b.n)}" fill="var(--accent)" fill-opacity=".7"><title>confidence ${num(b.confidence)}, accuracy ${num(b.accuracy)}, n=${b.n}</title></circle>`; });
  s+=`<text x="130" y="198" text-anchor="middle">confidence</text><text x="12" y="90" transform="rotate(-90 12 90)" text-anchor="middle">accuracy</text>`;
  $('calib').innerHTML = (a.calibration||[]).length ? svg(240,205,s) : '<p class="muted">No calibrated predictions.</p>';
  const d=a.detection; $('detect').innerHTML = d ? `<table><tr><th>Precision</th><th>Recall</th><th>False alarms</th><th>Premature</th><th>Mean delay</th></tr><tr><td>${pct(d.precision)}</td><td>${pct(d.recall)}</td><td>${pct(d.false_alarm_rate)}</td><td>${pct(d.premature_rate)}</td><td>${num(d.mean_delay_steps)} steps</td></tr></table><p>Alarms at or after the fault step on failed runs count as detections; alarms on tolerated or clean runs are false alarms.</p>` : 'This system does not raise online alarms (no <code>alarm_step</code> in its submission).';
  drawTrace(); drawTable();
}
$('sysSel').onchange = drawSystem;

// ---------- recovery
if (D.recovery.length){
  const rs=[...D.recovery].sort((a,b)=>(val(b.metrics.recovery_rate)||0)-(val(a.metrics.recovery_rate)||0));
  let s=''; const left=150,w=520;
  rs.forEach((r,i)=>{ const m=r.metrics.recovery_rate, v=val(m)||0, y=i*28+4, x=left+(w-left-50)*v;
    s+=`<text x="${left-6}" y="${y+14}" text-anchor="end">${esc(r.policy)}</text><rect x="${left}" y="${y}" width="${Math.max(0,x-left)}" height="18" rx="3" fill="var(--accent)"/>`;
    if(m && m.ci95 && m.ci95[0]!=null){ const a=left+(w-left-50)*m.ci95[0], b=left+(w-left-50)*m.ci95[1]; s+=`<line x1="${a}" x2="${b}" y1="${y+9}" y2="${y+9}" stroke="var(--text)"/><line x1="${a}" x2="${a}" y1="${y+4}" y2="${y+14}" stroke="var(--text)"/><line x1="${b}" x2="${b}" y1="${y+4}" y2="${y+14}" stroke="var(--text)"/>`; }
    s+=`<text x="${Math.max(x, m&&m.ci95&&m.ci95[1]!=null? left+(w-left-50)*m.ci95[1]:x)+6}" y="${y+14}">${pct(v)}</text>`; });
  $('recChart').innerHTML = svg(w, rs.length*28+10, s);
  const tmax=Math.max(1,...rs.map(r=>r.metrics.tokens_per_episode||0)); let c='';
  [0,0.5,1].forEach(t=>{ const y=190-170*t; c+=`<line x1="50" x2="520" y1="${y}" y2="${y}" stroke="var(--border)"/><text x="44" y="${y+4}" text-anchor="end">${t*100}%</text>`; });
  [0,0.5,1].forEach(t=>{ const x=50+440*t; c+=`<text x="${x}" y="206" text-anchor="middle">${num(tmax*t)}</text>`; });
  const placed=[];
  [...rs].sort((a,b)=>(a.metrics.tokens_per_episode||0)-(b.metrics.tokens_per_episode||0)).forEach(r=>{ const x=50+440*(r.metrics.tokens_per_episode||0)/tmax, y=190-170*(val(r.metrics.recovery_rate)||0);
    let ly=y-10; while(placed.some(p=>Math.abs(p[0]-x)<120&&Math.abs(p[1]-ly)<13)) ly-=13; placed.push([x,ly]);
    c+=`<circle cx="${x}" cy="${y}" r="6" fill="var(--accent)"><title>${esc(r.policy)}: ${pct(val(r.metrics.recovery_rate))}, ${num(r.metrics.tokens_per_episode)} tokens</title></circle><text x="${Math.min(x+8,430)}" y="${ly}">${esc(r.policy)}</text>`; });
  c+=`<text x="285" y="224" text-anchor="middle">tokens per failed episode</text>`;
  $('costChart').innerHTML = svg(560,230,c);
  const RC=[['recovery_rate','Recovery'],['conditional_recovery_rate','Conditional'],['harmful_intervention_rate','Harm'],['false_intervention_rate','False interv.'],['tokens_per_episode','Tokens/ep'],['tokens_per_recovered','Tokens/recovered'],['attempts_per_episode','Attempts'],['budget_breaches','Breaches']];
  const allModes=[...new Set(rs.flatMap(r=>Object.keys(r.by_mode)))];
  $('recTable').innerHTML = `<table><tr><th>Policy</th><th class="n">Failed</th><th class="n">Controls</th>${RC.map(c=>`<th class="n">${c[1]}</th>`).join('')}${allModes.map(m=>`<th class="n">${esc(short(m))}</th>`).join('')}</tr>`+
    rs.map(r=>`<tr><td>${esc(r.policy)}</td><td class="n">${r.n_failed}</td><td class="n">${r.n_controls}</td>${RC.map(([k])=>`<td class="n">${k.startsWith('tokens')||k.startsWith('attempts')||k==='budget_breaches'?num(val(r.metrics[k])):pct(val(r.metrics[k]))}</td>`).join('')}${allModes.map(m=>`<td class="n">${r.by_mode[m]?pct(r.by_mode[m].recovery_rate):''}</td>`).join('')}</tr>`).join('')+'</table>';
} else { $('recChart').innerHTML = '<p class="muted">No recovery reports given. Run <code>python3 -m benchmark recover</code>.</p>'; }

// ---------- robustness
if (D.robustness) {
  const R = D.robustness, rows = Object.entries(R.by_variant);
  let h = `<p class="muted">${R.eligible_tasks} of ${R.tasks} tasks eligible · ${R.injected} activated faults. Robustness = share of injected runs that still succeed (MAS-FIRE RS; ToolMaze PRR for tool faults). Recovery cost = ToolMaze RC (0 optimal, 1 failed).</p>`;
  h += `<table><tr><th>Fault variant</th><th>Mode</th><th class="n">n</th><th class="n">Failed</th><th class="n">Robustness</th><th class="n">Recovery cost</th></tr>` +
    rows.map(([k,v])=>`<tr><td>${esc(k)}</td><td>${esc(short(v.mode))}</td><td class="n">${v.n}</td><td class="n">${v.failed}</td><td class="n">${pct(val(v.robustness_score))}${ci(v.robustness_score)}</td><td class="n">${num(v.recovery_cost)}</td></tr>`).join('') + '</table>';
  const cells = Object.entries(R.toolmaze.cells || {});
  if (cells.length) h += `<h3 style="margin-top:16px">ToolMaze cells</h3><table><tr><th>Cell</th><th class="n">n</th><th class="n">Success (TSR = PRR)</th><th class="n">Recovery cost</th></tr>` +
    cells.map(([k,v])=>`<tr><td>${esc(k)}</td><td class="n">${v.n}</td><td class="n">${pct(val(v.robustness_score))}</td><td class="n">${num(v.recovery_cost)}</td></tr>`).join('') + '</table>';
  const gaps = Object.entries(R.toolmaze.explicit_minus_implicit || {});
  if (gaps.length) h += `<p class="muted">Explicit − implicit success gap: ${gaps.map(([k,v])=>`${k} ${(100*v).toFixed(0)} points`).join(', ')}.</p>`;
  $('robust').innerHTML = h;
} else { $('robust').innerHTML = '<p class="muted">Pass generation episodes to include robustness (python3 -m benchmark evaluate does this).</p>'; }

// ---------- trace explorer
const opt = (v,l) => `<option value="${esc(v)}">${esc(l)}</option>`;
$('exSplit').innerHTML = splits.map(s=>opt(s,s)).join('');
if (D.attribution[0] && splits.includes(D.attribution[0].split)) $('exSplit').value = D.attribution[0].split;
$('exMode').innerHTML = opt('', 'all labels') + LABELS.filter(l=>present.includes(l)).map(l=>opt(l,short(l))).join('');
function exList(){ const s=$('exSplit').value, m=$('exMode').value;
  const rs=D.rows.filter(r=>r.split===s && D.traces[r.tid] && (!m || label(r)===m));
  $('exTrace').innerHTML = rs.map(r=>opt(r.tid, `${r.tid} · ${short(label(r))}${r.variant?' · '+r.variant:''}`)).join(''); drawTrace(); }
let selStep=null;
function drawTrace(){
  const tid=$('exTrace').value, r=D.rows.find(x=>x.tid===tid), steps=D.traces[tid];
  if(!r||!steps){$('strip').innerHTML='';$('exInfo').textContent='No trace selected.';$('stepDetail').innerHTML='';return;}
  const a=D.attribution[+$('sysSel').value], pr=a? a.rows.find(x=>x.trajectory_id===tid):null;
  $('exInfo').innerHTML = `Status <b>${esc(r.status)}</b>`+(r.status==='failed'?` · gold: step ${r.critical_step}, ${esc(r.critical_agent)}, ${esc(short(r.mode||'unlabeled'))}`:'')+(r.variant?` · injected: ${esc(r.variant)} (${esc(r.timing)})`:'')+
    (pr&&pr.pred_step!=null?` · ${esc(a.system)}: step ${pr.pred_step}, ${esc(pr.pred_agent)}, ${esc(short(pr.pred_mode))} <span class="${pr.step_exact?'ok':'no'}">${pr.step_exact?'✓':'✗'}</span>`:'');
  $('strip').innerHTML = steps.map(s=>{ const cls=[s.i===r.critical_step?'gold':'', s.i===r.fault_step?'fault':'', (r.acceptable||[]).includes(s.i)?'acc':'', s.i===selStep?'sel':''].join(' ');
    return `<button type="button" class="${cls}" data-i="${s.i}" title="${esc(s.role)}">${s.i}<br><span class="muted">${esc(s.role.replace('_agent',''))}</span>${pr&&pr.pred_step===s.i?'<span class="p"></span>':''}</button>`; }).join('');
  $('strip').querySelectorAll('button').forEach(b=>b.onclick=()=>{selStep=+b.dataset.i; const s=steps.find(x=>x.i===selStep);
    $('stepDetail').innerHTML=`<pre>[step ${s.i}] ${esc(s.role)}${s.name?' ('+esc(s.name)+')':''}\n${esc(s.text)}</pre>`; drawTraceMarks();});
}
function drawTraceMarks(){ $('strip').querySelectorAll('button').forEach(b=>b.classList.toggle('sel', +b.dataset.i===selStep)); }
$('exSplit').onchange=exList; $('exMode').onchange=exList; $('exTrace').onchange=()=>{selStep=null;$('stepDetail').innerHTML='';drawTrace();};

// ---------- per-trajectory table
$('tSplit').innerHTML = opt('','all splits')+splits.map(s=>opt(s,s)).join('');
$('tStatus').innerHTML = opt('','all statuses')+[...new Set(D.rows.map(r=>r.status))].map(s=>opt(s,s)).join('');
$('tMode').innerHTML = opt('','all modes')+MODES.map(m=>opt(m,short(m))).join('');
function drawTable(){
  const s=$('tSplit').value, st=$('tStatus').value, m=$('tMode').value, q=$('tSearch').value.toLowerCase();
  const a=D.attribution[+$('sysSel').value]; const pm=a?Object.fromEntries(a.rows.map(x=>[x.trajectory_id,x])):{};
  const rs=D.rows.filter(r=>(!s||r.split===s)&&(!st||r.status===st)&&(!m||r.mode===m)&&(!q||(r.tid+' '+(r.variant||'')).toLowerCase().includes(q))).slice(0,500);
  const mark=v=>v==null?'':(v?'<span class="ok">✓</span>':'<span class="no">✗</span>');
  $('traceTable').innerHTML=`<table><tr><th>Trajectory</th><th>Split</th><th>Status</th><th>Mode</th><th>Variant</th><th>Timing</th><th class="n">Steps</th><th class="n">Gold step</th><th>Agent</th>${a?`<th class="n">Pred step</th><th>Step</th><th>Agent</th><th>Mode</th>`:''}</tr>`+
    rs.map(r=>{const p=pm[r.tid];return `<tr><td>${esc(r.tid)}</td><td>${esc(r.split)}</td><td>${esc(r.status)}</td><td>${esc(short(r.mode||''))}</td><td>${esc(r.variant||'')}</td><td>${esc(r.timing||'')}</td><td class="n">${r.n_steps}</td><td class="n">${r.critical_step??''}</td><td>${esc(r.critical_agent||'')}</td>${a?`<td class="n">${p?.pred_step??''}</td><td>${mark(p?.step_exact)}</td><td>${mark(p?.agent_correct)}</td><td>${mark(p?.mode_correct)}</td>`:''}</tr>`;}).join('')+'</table>'+(rs.length===500?'<p class="muted">Showing the first 500 rows.</p>':'');
}
['tSplit','tStatus','tMode'].forEach(id=>$(id).onchange=drawTable); $('tSearch').oninput=drawTable;
exList(); drawSystem(); drawTable();
</script>
</body>
</html>
'''
