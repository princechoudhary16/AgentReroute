"""Stylesheet for the presentation route. Dark studio palette, system fonts, no network assets."""

CSS = r'''
:root{color-scheme:dark;--bg:#090e11;--surface:#10181d;--raised:#162128;--border:#253238;--line:#1a252b;--text:#edf4f3;--muted:#94a6ad;--dim:#7d8f96;--mint:#bcf783;--violet:#b9a4ff;--cyan:#7fdbed;--amber:#f5cd88;--red:#ff9b9b;--sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;--mono:ui-monospace,SFMono-Regular,Consolas,monospace;--rail:64px;--top:56px;--gutter:clamp(24px,6vw,96px)}
*{box-sizing:border-box}
html{scroll-behavior:auto;background:var(--bg);scrollbar-width:none}
html::-webkit-scrollbar{display:none}
body{margin:0;background:var(--bg);color:var(--text);font:16px/1.5 var(--sans);-webkit-font-smoothing:antialiased;overflow-x:hidden}
a{color:inherit;text-decoration:none}button{font:inherit;color:inherit;cursor:pointer}
h1,h2,h3,p{margin:0}svg{display:block}
*:focus-visible{outline:2px solid var(--mint);outline-offset:4px}
.mono{font-family:var(--mono)}
.ol{font:11px var(--mono);letter-spacing:2px;text-transform:uppercase;color:var(--muted)}
.ol.mint{color:var(--mint)}.ol.violet{color:var(--violet)}.ol.cyan{color:var(--cyan)}.ol.amber{color:var(--amber)}.ol.red{color:var(--red)}
.h1{font-size:clamp(52px,6.4vw,104px);line-height:.98;letter-spacing:-.05em;font-weight:600}
.h2{font-size:clamp(34px,3.4vw,56px);line-height:1.04;letter-spacing:-.04em;font-weight:600}
.h3{font-size:clamp(22px,1.7vw,28px);line-height:1.2;letter-spacing:-.02em;font-weight:600}
.h1 em,.h2 em{font-style:normal;color:var(--mint)}
.lead{font-size:clamp(15px,1.15vw,19px);line-height:1.65;color:var(--muted)}
.big{font:500 clamp(44px,4.2vw,72px)/1 var(--mono);letter-spacing:-.05em}
.big small{font-size:.45em;letter-spacing:0;color:var(--muted)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px}
.pill{display:inline-block;border:1px solid var(--border);border-radius:20px;padding:5px 10px;font:10px var(--mono);letter-spacing:1px;color:var(--muted);white-space:nowrap}
.src{font:10px var(--mono);letter-spacing:1px;color:var(--dim)}

/* chrome */
.chrome{position:fixed;inset:0;pointer-events:none;z-index:20}
.progress{position:absolute;left:0;top:0;height:2px;width:0;background:var(--mint);transition:width .1s linear}
.rail{position:absolute;left:0;top:0;bottom:0;width:var(--rail);border-right:1px solid var(--line);display:flex;flex-direction:column;align-items:center;gap:16px;padding-top:30px;pointer-events:auto;background:#090e11cc;backdrop-filter:blur(10px)}
.rail a{width:22px;height:22px;display:grid;place-items:center;border-radius:50%}
.rail a i{width:8px;height:8px;border-radius:50%;background:var(--border);transition:background .3s,box-shadow .3s,transform .3s}
.rail a[aria-current] i{background:var(--mint);box-shadow:0 0 16px #bcf78377;transform:scale(1.15)}
.rail a:hover i{background:var(--muted)}
.rail .brand{margin-top:auto;margin-bottom:22px;color:var(--mint);width:28px;height:28px}
.topbar{position:absolute;left:var(--rail);right:0;top:0;height:var(--top);display:flex;align-items:center;justify-content:space-between;gap:20px;padding:0 40px;border-bottom:1px solid var(--line);background:#090e11cc;backdrop-filter:blur(10px);pointer-events:auto}
.topbar .ol{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#timer{color:var(--text);font-variant-numeric:tabular-nums;min-width:4ch;display:inline-block}
#timer.warn{color:var(--amber)}#timer.over{color:var(--red)}
.topbar-right{display:flex;align-items:center;gap:22px;white-space:nowrap}
.exit{color:var(--dim);pointer-events:auto}
.exit:hover{color:var(--text)}
.handoff{position:absolute;left:var(--rail);right:0;top:50%;transform:translateY(-50%);height:min(260px,36vh);background:#0c1317f2;border-top:1px solid #1f3327;border-bottom:1px solid #1f3327;display:flex;align-items:center;justify-content:space-between;padding:0 clamp(40px,8vw,120px);opacity:0;visibility:hidden;transition:opacity .35s,visibility .35s}
.handoff.show{opacity:1;visibility:visible}
.handoff .from{display:block;font-size:clamp(14px,1.2vw,18px);color:var(--dim)}
.handoff .to{display:block;margin-top:12px;font-size:clamp(34px,4vw,64px);font-weight:600;letter-spacing:-.045em;line-height:1}
.handoff .to em{font-style:normal;color:var(--mint)}
.handoff .budget{font:clamp(28px,3vw,44px) var(--mono);letter-spacing:-.04em}
.handoff .budget small{display:block;font-size:10px;letter-spacing:2px;color:var(--muted);margin-top:6px}

/* beats */
.beat{min-height:100vh;position:relative}
.beat.scene{height:calc(var(--len,3) * 100vh)}
.stage{position:relative;min-height:100vh;padding:calc(var(--top) + 4vh) var(--gutter) 9vh calc(var(--rail) + var(--gutter));display:flex;flex-direction:column}
.scene .stage{position:sticky;top:0;height:100vh;min-height:0;overflow:hidden}
.foot{margin-top:auto;display:flex;justify-content:space-between;gap:20px;padding-top:16px}
.foot .src{visibility:hidden}
.notes .foot .src{visibility:visible}
.hint{font:10px var(--mono);letter-spacing:1px;color:var(--dim)}
.grid2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:clamp(24px,4vw,70px)}
.grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:clamp(14px,1.6vw,24px)}
.grid4{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:clamp(14px,1.6vw,24px)}
.stack{display:flex;flex-direction:column;gap:14px}
.row{display:flex;gap:clamp(20px,3vw,44px);align-items:baseline;flex-wrap:wrap}

/* reveal */
.rv{opacity:0;transform:translateY(16px);transition:opacity .45s cubic-bezier(.2,.65,.3,1),transform .45s cubic-bezier(.2,.65,.3,1);transition-delay:calc(var(--d,0) * 55ms)}
.in .rv{opacity:1;transform:none}
.bar{height:100%;border-radius:4px;transform-origin:left;transform:scaleX(0);transition:transform .9s cubic-bezier(.2,.65,.3,1) calc(var(--d,0) * 120ms + .2s)}
.in .bar{transform:scaleX(1)}

/* 01 */
.title-art{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.title-art path{stroke-dasharray:3000;stroke-dashoffset:3000;transition:stroke-dashoffset 2.6s cubic-bezier(.3,.5,.2,1) .2s}
.in .title-art path{stroke-dashoffset:0}
.step{width:56px;height:74px;border:1px solid var(--border);border-radius:8px;background:var(--surface);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:7px;flex:none;opacity:.3;transition:opacity .35s,border-color .35s,background .35s,box-shadow .35s}
.step b{font:400 11px var(--mono);color:var(--dim)}.step i{width:18px;height:4px;border-radius:2px;background:var(--border)}
.step.lit{opacity:1}
/* the activation ring */
.ring-wrap{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:clamp(24px,4vw,70px);align-items:center;margin-top:2vh;flex:1;min-height:0}
.ring{position:relative;width:min(100%,58vh);aspect-ratio:1;margin:auto}
.ring svg{position:absolute;inset:0;width:100%;height:100%;overflow:visible}
.ring .orbit{fill:none;stroke:#1a252b;stroke-width:.5;stroke-dasharray:1.5 2.5}
.ring line{stroke:#1f2d33;stroke-width:.7;transition:stroke .3s,stroke-width .3s}
.ring line.lit{stroke:#4e6068}
.ring line.now{stroke:var(--cyan);stroke-width:1.1}
.ring line.cause{stroke:var(--mint);stroke-width:1.4;stroke-dasharray:3 2}
.rnode{position:absolute;transform:translate(-50%,-50%);width:clamp(104px,10vw,132px);padding:10px 12px;border:1px solid var(--border);border-radius:12px;background:var(--surface);text-align:center;transition:border-color .3s,box-shadow .3s,background .3s}
.rnode b{display:block;font-size:13px;font-weight:600;letter-spacing:-.01em}
.rnode small{display:block;font-size:10px;line-height:1.35;color:var(--muted);margin-top:3px}
.rnode .t{position:absolute;top:-11px;right:-10px;font:10px var(--mono);letter-spacing:1px;color:var(--cyan);background:#101f24;border:1px solid #2f4f58;border-radius:12px;padding:2px 7px;opacity:0;transition:opacity .25s}
.rnode.now{border-color:var(--cyan);box-shadow:0 0 28px #7fdbed33}
.rnode.now .t{opacity:1}
.rnode.sym{border-color:var(--red);background:#2a1717;box-shadow:0 0 30px #ff9b9b40}
.rnode.cause{border-color:var(--mint);background:#15261a;box-shadow:0 0 30px #bcf78350}
.rtag{position:absolute;left:50%;transform:translate(-50%,6px);font:11px var(--mono);letter-spacing:1px;padding:7px 12px;border-radius:6px;white-space:nowrap;opacity:0;transition:opacity .4s,transform .4s}
.rtag.show{opacity:1;transform:translate(-50%,0)}
.rtag.sym{top:-54px;background:#2a1717;color:var(--red);border:1px solid #5a3a3a}
.rtag.cause{top:calc(50% + 34px);background:#15261a;color:var(--mint);border:1px solid #3a5a3a}
.acts{display:flex;flex-direction:column;gap:5px;align-self:center}
.act{display:grid;grid-template-columns:44px 104px minmax(0,1fr);gap:12px;align-items:center;padding:7px 12px;border:1px solid transparent;border-radius:8px;font-size:13px;opacity:.28;transition:opacity .3s,border-color .3s,background .3s}
.act b{font:400 11px var(--mono);letter-spacing:1px;color:var(--cyan)}
.act i{font-style:normal;font-weight:600;color:var(--text)}
.act span{color:var(--muted);line-height:1.35}
.act.lit{opacity:1}
.act.visited{opacity:.4}
.act.now{border-color:#2f4f58;background:#101f24}
.act.sym{border-color:#5a3a3a;background:#2a1717}.act.sym b{color:var(--red)}
.act.cause{border-color:#3a5a3a;background:#13201a}.act.cause b{color:var(--mint)}
.hl{position:relative;display:block}.hl span{display:block;transition:opacity .5s,transform .5s}
.hl .b{position:absolute;left:0;top:0;opacity:0;transform:translateY(10px)}
.hl.resolved .a{opacity:0;transform:translateY(-10px)}.hl.resolved .b{opacity:1;transform:none}
.q{display:flex;gap:22px;align-items:flex-start;padding:20px 0;border-top:1px solid var(--line)}
.q b{font:400 13px var(--mono);letter-spacing:1px;color:var(--violet);min-width:44px;padding-top:4px}
.q span{font-size:clamp(15px,1.25vw,19px);line-height:1.5}
.q small{display:block;font:10px var(--mono);letter-spacing:1px;color:var(--dim);margin-top:8px}

/* 02 */
.flow{display:flex;flex-direction:column;gap:8px}
.flow .box{border:1px solid var(--border);border-radius:12px;background:var(--surface);padding:16px 22px;text-align:center}
.flow .box strong{display:block;font-size:clamp(14px,1.1vw,18px);font-weight:600;margin-top:6px}
.flow .arrow{text-align:center;color:#4e6068;font-size:20px;line-height:1}
.flow .judge{border:1px solid #4a3f78;background:#1c1830;border-radius:14px;padding:22px;text-align:center}
.flow .judge strong{display:block;font-size:clamp(18px,1.6vw,26px);font-weight:600;letter-spacing:-.02em;margin:8px 0 6px}
.flow .judge p{font-size:13px;color:var(--muted)}
.stat b{display:block;font:500 clamp(44px,4vw,64px)/1 var(--mono);letter-spacing:-.05em}
.stat span{display:block;font-size:13px;color:var(--muted);margin-top:8px;line-height:1.5}
.lens{border-radius:12px;padding:16px;border:1px solid var(--border);background:var(--surface)}
.lens .ol{font-size:9px}.lens strong{display:block;font-size:clamp(14px,1.1vw,17px);font-weight:600;margin:10px 0 6px;line-height:1.25}
.lens small{font-size:12px;color:var(--muted);line-height:1.5}
.lens.a{border-color:#3a5a3a;background:#13201a}.lens.a .ol{color:var(--mint)}
.lens.b{border-color:#2f4f58;background:#101f24}.lens.b .ol{color:var(--cyan)}
.lens.c{border-color:#5a4d2f;background:#231e12}.lens.c .ol{color:var(--amber)}
.insight{display:flex;gap:18px;padding:18px 0;border-top:1px solid var(--line)}
.insight b{font:400 12px var(--mono);letter-spacing:1px;color:var(--mint);padding-top:4px}
.insight strong{display:block;font-size:clamp(16px,1.3vw,19px);font-weight:600}
.insight p{font-size:14px;line-height:1.6;color:var(--muted);margin-top:6px}
.gapviz{position:relative;height:300px}
.gapviz .line{position:absolute;top:120px;height:2px;width:40%;transition:transform .2s linear}
.gapviz .line.l{left:0;background:var(--violet);transform:translateX(-110%)}
.gapviz .line.r{right:0;background:var(--cyan);transform:translateX(110%)}
.gapviz .lab{position:absolute;font:11px var(--mono);letter-spacing:1px;color:var(--muted)}
.gapviz .mark{position:absolute;left:30%;width:40%;top:170px;text-align:center;opacity:0;transform:translateY(16px);transition:opacity .5s,transform .5s}
.gapviz .mark strong{display:block;font-size:clamp(20px,1.8vw,26px);font-weight:600;color:var(--mint);letter-spacing:-.02em}
.gapviz .mark p{font-size:13px;color:var(--muted);line-height:1.55;margin-top:6px}
.gapviz.closed .mark{opacity:1;transform:none}
.net-wrap{position:absolute;left:calc(var(--rail) + 44%);right:0;top:var(--top);bottom:0;overflow:hidden;background-image:radial-gradient(#1b262c 1px,transparent 1px);background-size:28px 28px}
.net{position:absolute;left:0;top:0;width:900px;height:844px;transform-origin:21% 48%;will-change:transform}
.node{position:absolute;width:190px;border:1px solid #2b3a42;border-radius:12px;background:var(--surface);padding:14px 16px;box-shadow:0 10px 30px #00000066;transition:opacity .4s,transform .4s}
.node .ol{font-size:9px}.node strong{display:block;font-size:17px;font-weight:600;margin-top:6px}.node small{display:block;font-size:11px;color:var(--muted);margin-top:4px}
.node.p{border-color:var(--mint);background:#13201a}
.node.spec{opacity:0;transform:translateX(30px)}
.net.show-specs .node.spec{opacity:1;transform:none}
.tools{display:flex;gap:4px;margin-top:10px;opacity:0;transition:opacity .4s}
.net.show-tools .tools{opacity:1}
.tools i{width:26px;height:18px;border-radius:4px;background:#1c2a31;border:1px solid #2b3a42}
.net .signal{opacity:0}
.net.show-signal .signal{opacity:1;animation:dash 3s linear infinite}
.net .signal-dot{opacity:0}
.net.show-signal .signal-dot{opacity:1;animation:travel 3s linear infinite}
@keyframes dash{to{stroke-dashoffset:-140}}
@keyframes travel{0%{offset-distance:0}100%{offset-distance:100%}}

/* 03 */
.lens-track{position:relative;height:200px;margin:3vh 0 0}
.lens-track .lens{position:absolute;top:0;width:min(300px,28vw);box-shadow:0 14px 40px #00000088;will-change:transform}
.lens-track .lens.b{top:36px}.lens-track .lens.c{top:72px}
.trace-mini{display:flex;gap:6px;position:relative}
.trace-mini .step{flex:1 1 0;min-width:0;width:auto;height:58px;border-radius:6px;font:10px var(--mono);color:var(--dim);align-items:flex-end;padding-bottom:6px;opacity:.22;gap:0}
.trace-mini .step.lit{opacity:1;border-color:#2f4f58;background:#101f24;color:#c9d4d6}
.edge{position:absolute;top:-14px;bottom:-14px;width:1px;background:var(--cyan);opacity:.8;transition:left .1s linear}
.edge span{position:absolute;left:10px;top:-2px;font:10px var(--mono);letter-spacing:1px;color:var(--cyan);white-space:nowrap}
.edge.flip span{left:auto;right:10px}
.vios{position:relative;height:60px;margin-top:14px}
.vio{position:absolute;top:0;display:flex;flex-direction:column;align-items:center;gap:6px;font:10px var(--mono);letter-spacing:.5px;white-space:nowrap;opacity:0;transform:translateY(-10px);transition:opacity .4s,transform .4s}
.vio i{width:10px;height:10px;border-radius:50%}
.vio.show{opacity:1;transform:none}
table.blame{border-collapse:collapse;width:100%;font:13px var(--mono)}
.blame th,.blame td{text-align:left;padding:11px 12px;border-bottom:1px solid var(--line);font-weight:400}
.blame th{color:var(--muted);font-size:10px;letter-spacing:1px;text-transform:uppercase}
.blame .hot{color:var(--amber)}.blame .max{color:var(--red);font-weight:600}
.tflow{display:flex}
.tflow div{flex:1;border:1px solid var(--border);background:var(--surface);padding:12px;font-size:12px;line-height:1.5;color:#c9d4d6}
.tflow div:first-child{border-radius:10px 0 0 10px}.tflow div:last-child{border-radius:0 10px 10px 0}
.tflow div+div{border-left:0}
.tflow .bad{border-color:#5a3a3a;background:#1f1414;color:var(--red)}
.tflow b{display:block;font:400 9px var(--mono);letter-spacing:1px;color:var(--dim);margin-bottom:6px}
.cands{position:relative;min-height:300px}
.cand{position:absolute;width:min(230px,22vw);border:1px solid var(--border);border-radius:10px;background:var(--surface);padding:12px 14px;font-size:12px;line-height:1.5;color:#c9d4d6;transition:opacity .4s,border-color .4s}
.cand b{display:block;font:400 10px var(--mono);letter-spacing:1px;color:var(--dim);margin-bottom:4px}
.cands.chosen .cand:not(.pick){opacity:.35}
.cands.chosen .cand.pick{border-color:var(--mint)}.cands.chosen .cand.pick b{color:var(--mint)}
.wires path{fill:none;stroke:#2b3a42;stroke-width:1.5;stroke-dasharray:1200;stroke-dashoffset:1200}
.wires path.pick{stroke:var(--mint);stroke-width:2}
.wires path.out{stroke:var(--violet);stroke-dasharray:4 6}
.selcard{position:absolute;right:0;top:0;width:min(440px,34vw);border:1px solid var(--mint);border-radius:16px;background:#13201a;padding:26px;box-shadow:0 0 60px #bcf78322;will-change:transform,opacity}
.selcard strong{display:block;font-size:clamp(20px,1.7vw,26px);font-weight:600;letter-spacing:-.02em;margin:10px 0 8px}
.selcard p{font-size:13px;line-height:1.6;color:var(--muted)}
.selcard .inputs{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}
.selcard .inputs span{border:1px solid #3a5a3a;border-radius:20px;padding:5px 10px;font:10px var(--mono);color:var(--mint)}
.selcard .nogold{margin-top:16px;padding-top:12px;border-top:1px solid #1f3327;font:11px var(--mono);color:var(--muted);letter-spacing:.5px}
.chart{padding:22px 24px}
.chart h3{font-size:17px;font-weight:600;margin:4px 0 0}
.brow{margin-top:16px}
.brow .lbl{display:flex;justify-content:space-between;align-items:baseline;font-size:12px;color:#c9d4d6;margin-bottom:6px;gap:12px}
.brow small{font:10px var(--mono);color:var(--dim);margin-left:8px}
.brow strong{font:500 18px var(--mono);letter-spacing:-.03em}
.track{height:18px;border-radius:4px;background:var(--raised);overflow:hidden}
.rx .bar{background:var(--violet)}.rx strong{color:var(--violet)}
.ar .bar{background:var(--mint)}.ar strong{color:var(--mint)}
.wr .bar{background:#6f8f5c}.wr strong{color:#9fbf8c}

/* 04 */
.live{position:absolute;inset:var(--top) 0 0 var(--rail);display:flex}
.live .frame{flex:1;position:relative;border-right:1px solid var(--line);background:var(--bg);overflow:auto;overscroll-behavior:contain}
.live iframe{display:block;width:100%;height:100%;border:0;background:var(--bg)}
.live .side{width:min(420px,32vw);padding:26px 28px;display:flex;flex-direction:column;gap:14px}
.ms{display:flex;gap:10px;align-items:center;font-size:12px;color:var(--muted);padding:9px 0;border-top:1px solid var(--line)}
.ms i{display:none}
.guide{margin-top:auto;border-top:1px solid var(--line);padding-top:14px}
.guide strong{display:block;font-size:15px;font-weight:600;margin:6px 0 4px}
.guide p{font-size:12px;line-height:1.6;color:var(--muted)}
.rung{display:flex;gap:12px;align-items:center;padding:9px 14px;border:1px solid var(--border);border-radius:8px;background:var(--surface);font-size:13px;color:#c9d4d6}
.rung b{font:400 10px var(--mono);color:var(--dim);min-width:16px}
.rung.on{border-color:var(--mint);background:#13201a;color:var(--text)}
.rung.esc{border-color:#5a4d2f;color:var(--amber)}
.ev{font:clamp(13px,1.1vw,16px) var(--mono);padding:14px 18px;border:1px solid #3a5a3a;border-radius:10px;background:#13201a;color:var(--mint)}
.opt{border:1px solid var(--border);border-radius:10px;background:var(--surface);padding:12px 16px;font-size:13px}
.opt strong{display:block;font-weight:600;margin-bottom:4px}.opt span{font:12px var(--mono);color:var(--muted)}
.opt.sel{border-color:var(--mint);background:#13201a}.opt.sel em{display:block;margin-top:5px;color:var(--mint);font-size:12px}
.opt.no{opacity:.5}
.vdocs{position:absolute;left:0;top:0;right:0;display:flex;flex-direction:column;gap:12px}
.doc{border:1px solid var(--border);border-radius:12px;background:var(--surface);padding:14px 18px;font:12px/1.7 var(--mono);color:#c9d4d6;transition:filter .4s,opacity .4s}
.doc b{display:block;font-weight:400;font-size:9px;letter-spacing:1px;color:var(--dim);margin-bottom:6px}
.doc.hid{color:#3a4a52;border-color:var(--line);background:#0c1317;filter:blur(1.5px)}
.doc.hid b{color:#3a4a52}
.doc.pending{opacity:.45}
.vview{position:relative;height:min(560px,62vh);overflow:hidden}
.vwin{position:absolute;left:-12px;right:-12px;top:-12px;height:min(230px,30vh);border:2px solid var(--cyan);border-radius:14px;box-shadow:0 0 0 2000px #090e1166,0 0 40px #7fdbed55;will-change:transform;pointer-events:none}
.vwin span{position:absolute;left:14px;bottom:-24px;font:10px var(--mono);letter-spacing:1px;color:var(--cyan)}
.stamp{display:inline-flex;align-items:center;gap:12px;opacity:0;transform:scale(1.2);transition:opacity .3s,transform .3s}
.stamp.show{opacity:1;transform:none}
.stamp b{border:1px solid var(--mint);color:var(--mint);border-radius:6px;padding:8px 14px;font:14px var(--mono);letter-spacing:2px}
.stamp span{font-size:12px;color:var(--muted)}

/* 05 */
.field{display:grid;grid-template-columns:repeat(40,12px);gap:4px;align-content:start}
.field i{width:12px;height:12px;border-radius:50%;background:#2b3a42;display:block;opacity:0;transform:scale(.4);transition:opacity .35s,transform .35s,background .6s}
.field i.on{opacity:1;transform:none}
.field.sorted i[data-l="0"]{background:var(--mint)}.field.sorted i[data-l="1"]{background:var(--cyan)}.field.sorted i[data-l="2"]{background:var(--amber)}
.field.sorted i[data-l="3"]{background:var(--violet)}.field.sorted i[data-l="4"]{background:var(--red)}.field.sorted i[data-l="5"]{background:#8fb36a}
.field.sorted i[data-l="6"]{background:#d4a5ff}.field.sorted i[data-l="c"]{background:#4e6068}.field.sorted i[data-l="t"],.field.sorted i[data-l="x"]{background:#2b3a42}
.field.test i{opacity:.18}.field.test i[data-t]{opacity:1;box-shadow:0 0 8px currentColor}
.field.test i[data-t="h"]{background:#4e6068}
.leg{display:flex;flex-wrap:wrap;gap:8px 16px;font:10px var(--mono);letter-spacing:.5px;color:var(--muted);opacity:0;transition:opacity .5s}
.leg.show{opacity:1}
.leg i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;vertical-align:middle}
.counter{font:500 clamp(40px,4vw,64px)/1 var(--mono);letter-spacing:-.05em;transition:opacity .3s}
.counter small{font-size:.4em;letter-spacing:0;color:var(--muted)}
.crow{display:grid;grid-template-columns:minmax(160px,.9fr) minmax(0,2.4fr) 90px;gap:14px;align-items:center;margin-top:12px}
.crow .n{font-size:13px;line-height:1.3}.crow .n small{display:block;font:9px var(--mono);letter-spacing:.5px;color:var(--dim);margin-top:2px}
.crow .tr{position:relative;height:22px}
.crow .cl{position:absolute;left:0;top:0;height:22px;border:1px solid var(--amber);border-radius:4px;background:#f5cd8811;transform-origin:left;transform:scaleX(0);transition:transform .8s cubic-bezier(.2,.65,.3,1) calc(var(--d,0) * 100ms)}
.crow .cf{position:absolute;left:0;top:4px;height:14px;border-radius:3px;background:var(--mint);transform-origin:left;transform:scaleX(0);transition:transform .8s cubic-bezier(.2,.65,.3,1) calc(var(--d,0) * 100ms + .6s)}
.crow .cf.rx{background:var(--violet)}
.in .crow .cl,.in .crow .cf{transform:scaleX(1)}
.crow .v{font:13px var(--mono);text-align:right}.crow .v i{font-style:normal;color:var(--amber)}.crow .v b{font-weight:500;color:var(--mint)}
.harmed{display:flex;align-items:baseline;gap:12px;margin-top:14px}
.runs{display:grid;grid-template-columns:repeat(5,22px);gap:6px;margin-top:14px}
.runs i{width:22px;height:22px;border-radius:5px;background:#1c2a31;border:1px solid #2b3a42;display:block}
.runs i.x{background:#2a1717;border-color:var(--red)}
.cost{display:flex;align-items:center;gap:10px;margin-top:10px;font-size:12px;color:var(--muted)}
.cost i{height:8px;border-radius:4px;background:var(--cyan);display:block}
.mini-live{position:relative;height:min(380px,42vh);border:1px solid var(--border);border-radius:14px;overflow:auto;overscroll-behavior:contain;background:var(--bg)}
.mini-live iframe{display:block;width:100%;height:100%;border:0}

/* 06 */
.ans{display:grid;grid-template-columns:64px minmax(0,1.1fr) minmax(0,1fr);gap:clamp(16px,2.5vw,30px);align-items:start;padding:clamp(16px,2.4vh,28px) 0;border-top:1px solid var(--line)}
.ans b{font:400 13px var(--mono);letter-spacing:1px;color:var(--violet);padding-top:6px}
.ans h3{font-size:clamp(17px,1.5vw,24px);font-weight:600;letter-spacing:-.02em;line-height:1.25}
.ans .v{display:block;font:12px var(--mono);letter-spacing:.5px;color:var(--mint);margin-top:10px}
.ans p{font-size:clamp(13px,1.05vw,15px);line-height:1.65;color:var(--muted)}
.lim{border-top:1px solid var(--line);padding-top:14px;font-size:13px;line-height:1.6;color:var(--muted)}
.lim b{display:block;font:400 9px var(--mono);letter-spacing:1px;color:var(--dim);margin-bottom:6px}

/* short viewports: a 768-line laptop mirrored to a projector leaves about 650 px for the stage */
@media(max-height:820px){
 .stage{padding-top:calc(var(--top) + 2vh);padding-bottom:5vh}
 .h1{font-size:clamp(40px,5.2vw,80px)}.h2{font-size:clamp(26px,2.8vw,44px)}.h3{font-size:clamp(18px,1.5vw,24px)}
 .lead{font-size:clamp(13px,1vw,16px);line-height:1.55}
 .big,.stat b,.counter{font-size:clamp(34px,3.2vw,52px)}
 .chart{padding:14px 18px}.brow{margin-top:10px}.track{height:14px}
 .ans{padding:10px 0}.ans h3{font-size:clamp(15px,1.3vw,20px)}
 .runs{grid-template-columns:repeat(5,16px);gap:5px;margin-top:10px}.runs i{width:16px;height:16px}
 .rung{padding:6px 12px;font-size:12px}.opt{padding:8px 14px;font-size:12px}
 .doc{padding:10px 14px;font-size:11px;line-height:1.6}.vview{height:min(440px,56vh)}
 .lens-track{height:170px}.trace-mini .step{height:46px}
 .field{grid-template-columns:repeat(40,10px)}.field i{width:10px;height:10px}
 .q{padding:12px 0}.q span{font-size:clamp(13px,1.1vw,16px)}
 .insight{padding:12px 0}.flow .box{padding:10px 16px}.flow .judge{padding:14px}
 .step{height:62px}
}
/* motion off */
.rm .rv,.rm .bar,.rm .crow .cl,.rm .crow .cf,.rm .title-art path{transition:none!important;transform:none!important;opacity:1!important;stroke-dashoffset:0!important}
.rm *{animation:none!important}
@media(max-width:1000px){.grid2,.grid3,.grid4{grid-template-columns:1fr}.live{flex-direction:column}.live .side{width:auto}.live .frame{min-height:60vh}.net-wrap{left:var(--rail);top:55%}.ans{grid-template-columns:44px 1fr}.ans p{grid-column:2}.field{grid-template-columns:repeat(24,10px)}}
@media(max-width:760px){:root{--rail:40px}.topbar{padding:0 14px}.rail{gap:10px}.handoff{padding:0 24px}.exit{right:14px}}
'''
