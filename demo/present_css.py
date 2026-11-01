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
/* the intro flower: a 33-step trace as petals around a hub. --S is the flower's diameter */
.ring-wrap{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:clamp(24px,4vw,70px);align-items:center;margin-top:2vh;flex:1;min-height:0}
.flower{--S:min(54vh,520px);position:relative;width:var(--S);height:var(--S);margin:auto}
.petal{--r0:calc(var(--S) * .27);--h:calc(var(--S) * .2);--w:calc(var(--S) * .044);position:absolute;left:calc(50% - var(--w) / 2);top:calc(50% - var(--h));width:var(--w);height:var(--h);border-radius:999px;border:1px solid var(--border);background:var(--surface);transform-origin:50% 100%;transform:rotate(var(--a)) translateY(calc(-1 * var(--r0)));opacity:.5;transition:background .25s,border-color .25s,box-shadow .25s,opacity .25s}
.petal span{position:absolute;left:50%;top:7px;width:24px;margin-left:-12px;text-align:center;font:9px var(--mono);color:var(--dim);transform:rotate(calc(-1 * var(--a)))}
.petal.lit{opacity:1;border-color:#2f4f58;background:#101f24}
.petal.lit span{color:#9fc6cf}
.petal.visited{opacity:.3}
.petal.now{opacity:1;border-color:var(--cyan);background:#1d4650;box-shadow:0 0 22px #7fdbed99}
.petal.now span{color:var(--text)}
.petal.sym{opacity:1;border-color:var(--red);background:#4a2020;box-shadow:0 0 26px #ff9b9b88}
.petal.sym span{color:var(--red)}
.petal.cause{opacity:1;border-color:var(--mint);background:#26421f;box-shadow:0 0 28px #bcf78399}
.petal.cause span{color:var(--mint)}
.hub{position:absolute;left:50%;top:50%;width:calc(var(--S) * .4);height:calc(var(--S) * .4);transform:translate(-50%,-50%);border-radius:50%;border:1px solid var(--line);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px;text-align:center}
.hub .ol{font-size:9px}
.hub b{font:500 calc(var(--S) * .11)/1 var(--mono);letter-spacing:-.04em;color:var(--cyan);transition:color .25s}
.hub span{font:10px var(--mono);letter-spacing:1px;text-transform:uppercase;color:var(--cyan);transition:color .25s}
.hub[data-state="sym"] b,.hub[data-state="sym"] span{color:var(--red)}
.hub[data-state="cause"] b,.hub[data-state="cause"] span{color:var(--mint)}
.fkeys{display:flex;flex-direction:column;gap:10px;align-self:center}
.fkey{display:grid;grid-template-columns:14px minmax(0,1fr);gap:14px;align-items:start;padding:14px 16px;border:1px solid var(--line);border-radius:12px;opacity:.4;transition:opacity .3s,border-color .3s,background .3s}
.fkey i{width:12px;height:12px;border-radius:50%;margin-top:4px}
.fkey b{display:block;font-size:15px;font-weight:600}
.fkey small{display:block;font-size:13px;line-height:1.5;color:var(--muted);margin-top:3px}
.fkey.on{opacity:1}
.fkey[data-k="run"] i{background:var(--cyan)}.fkey[data-k="run"].on{border-color:#2f4f58;background:#101f24}
.fkey[data-k="sym"] i{background:var(--red)}.fkey[data-k="sym"].on{border-color:#5a3a3a;background:#2a1717}
.fkey[data-k="cause"] i{background:var(--mint)}.fkey[data-k="cause"].on{border-color:#3a5a3a;background:#13201a}
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
.silent-network{margin:clamp(22px,4vh,44px) 0 0;padding:clamp(18px,2vw,28px);border:1px solid var(--border);border-radius:16px;background:var(--bg)}
.silent-network figcaption{display:flex;justify-content:space-between;gap:16px;align-items:center}.silent-legend{font:10px var(--mono);color:var(--muted)}.silent-legend i{display:inline-block;width:18px;border-top:2px dashed var(--cyan);vertical-align:middle;margin-right:7px}
.silent-network-scroll{overflow-x:auto}.silent-canvas{position:relative;min-width:680px;height:clamp(280px,42vh,405px);background-image:radial-gradient(#1b262c 1px,transparent 1px);background-size:28px 28px}
.silent-wires{position:absolute;inset:0;width:100%;height:100%;overflow:visible}.silent-wires .wire{fill:none;stroke:#2b3a42;stroke-width:2}.silent-wires .fault-signal{fill:none;stroke:var(--cyan);stroke-width:2;stroke-dasharray:8 12;animation:dash 3s linear infinite}
.silent-node{position:absolute;width:20%;transform:translateY(-50%);border:1px solid #2b3a42;border-radius:12px;background:var(--surface);padding:clamp(10px,1.2vw,17px);box-shadow:0 10px 30px #0006}.silent-node .ol{font-size:9px;letter-spacing:1px}.silent-node strong{display:block;font-size:clamp(14px,1.3vw,19px);font-weight:600;margin-top:6px;overflow-wrap:anywhere}.silent-node small{display:block;font-size:clamp(10px,.9vw,13px);line-height:1.4;color:var(--muted);margin-top:4px}
.silent-node.planner{left:2%;top:50%;border-color:var(--mint);background:#13201a}.silent-node.product{left:43%;top:18%;opacity:.6}.silent-node.order{left:43%;top:50%;width:23%;border-color:var(--cyan);background:#101f24}.silent-node.returns{left:43%;top:82%;opacity:.6}.silent-node.failure{left:76%;top:50%;width:22%;border-color:var(--red);background:#29191c}
.silent-insight{padding-top:16px;border-top:1px solid var(--border);color:var(--text);font-size:clamp(13px,1.1vw,17px)}.silent-insight span{color:var(--mint);font-size:20px;margin-right:9px}
.fault-lenses{margin-top:clamp(40px,9vh,100px)}.fault-lenses .lens{padding:clamp(22px,2.6vw,40px)}.fault-lenses .lens strong{font-size:clamp(20px,2vw,30px);margin:24px 0 14px}.fault-lenses .lens small{font-size:clamp(14px,1.1vw,18px)}
@media(max-width:1000px){.silent-network{margin-top:24px;padding:16px}.silent-canvas{height:310px}.fault-lenses{margin-top:24px}.fault-lenses .lens{padding:18px}.fault-lenses .lens strong{margin:10px 0 5px}}

.net-wrap{position:absolute;left:calc(var(--rail) + 44%);right:0;top:var(--top);bottom:0;overflow:hidden;background-image:radial-gradient(#1b262c 1px,transparent 1px);background-size:28px 28px}
.net{position:absolute;left:0;top:0;width:900px;height:844px;transform-origin:21% 48%;will-change:transform}
.node{position:absolute;width:190px;border:1px solid #2b3a42;border-radius:12px;background:var(--surface);padding:14px 16px;box-shadow:0 10px 30px #00000066;will-change:transform}
.node .ol{font-size:9px}.node strong{display:block;font-size:17px;font-weight:600;margin-top:6px}.node small{display:block;font-size:11px;color:var(--muted);margin-top:4px}
.node.p{border-color:var(--mint);background:#13201a}
.net path.wire{fill:none;stroke:#2b3a42;stroke-width:2;stroke-dasharray:1;stroke-dashoffset:1}
.tools{display:flex;gap:4px;margin-top:10px}
.tools i{width:26px;height:18px;border-radius:4px;background:#1c2a31;border:1px solid #2b3a42}
.net .signal{opacity:0}
.net.show-signal .signal{opacity:1;animation:dash 3s linear infinite}
.net .signal-dot{opacity:0}
.net.show-signal .signal-dot{opacity:1;animation:travel 3s linear infinite}
@keyframes dash{to{stroke-dashoffset:-140}}
@keyframes travel{0%{offset-distance:0}100%{offset-distance:100%}}

/* 03 · one trace, three views read side by side; a single cursor sweeps all three at once */
.views{--lab:clamp(210px,17vw,260px);position:relative;margin-top:6vh}
.vrow{display:grid;grid-template-columns:var(--lab) minmax(0,1fr);gap:24px;align-items:center;min-height:clamp(56px,8.5vh,76px);border-top:1px solid var(--line)}
.vrow.vhead{border-top:0;min-height:46px}
.vlab b{display:flex;align-items:center;gap:10px;font-size:14px;font-weight:600}
.vlab b i{width:8px;height:8px;border-radius:50%;flex:none}
.vlab small{display:block;font-size:11px;line-height:1.45;color:var(--muted);margin:4px 0 0 18px}
.vrow[data-v="a"]{--c:var(--mint)}.vrow[data-v="b"]{--c:var(--cyan)}.vrow[data-v="c"]{--c:var(--amber)}
.vlab b i{background:var(--c)}
.vtrack{position:relative;align-self:stretch}
.vtrack .lane{position:absolute;left:0;right:0;top:50%;height:1px;background:var(--line)}
.chips{display:grid;grid-template-columns:repeat(var(--n),minmax(0,1fr));gap:5px;align-items:center}
.chips span{height:28px;display:grid;place-items:center;border:1px solid var(--border);border-radius:6px;font:10px var(--mono);color:var(--dim);transition:border-color .2s,background .2s,color .2s}
.chips span.seen{border-color:#2f4f58;background:#101f24;color:#c9d4d6}
.vmark{position:absolute;top:50%;display:flex;flex-direction:column;align-items:center;gap:5px;font:10px var(--mono);color:var(--c);opacity:0;transform:translate(-50%,-6px) scale(.4);transition:opacity .3s,transform .3s}
.vmark i{width:12px;height:12px;border-radius:50%;background:var(--c);box-shadow:0 0 14px var(--c)}
.vmark.show{opacity:1;transform:translate(-50%,-6px)}
.voverlay{position:absolute;top:0;bottom:0;left:calc(var(--lab) + 24px);right:0;pointer-events:none}
.vfuture{position:absolute;top:0;bottom:0;right:0;left:calc(var(--edge) * 100%);background:#090e11c7;transition:left .18s ease-out}
.vcursor{position:absolute;top:-10px;bottom:-6px;left:calc(var(--edge) * 100%);width:1px;background:var(--cyan);box-shadow:0 0 12px #7fdbed;transition:left .18s ease-out}
.vcursor span{position:absolute;bottom:100%;left:0;transform:translateX(calc(var(--edge) * -100%));transition:transform .18s ease-out;margin-bottom:6px;white-space:nowrap;font:10px var(--mono);letter-spacing:1px;color:var(--cyan)}
.vnote{margin-top:3vh;font-size:13px;line-height:1.6;color:var(--muted);max-width:760px}
/* 03 · the judge: three inputs built from the three views → one call → one diagnosis, on one axis */
.jf{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(0,.9fr) minmax(0,1fr);gap:64px;align-items:center;margin-top:3vh;flex:1;min-height:0}
.jcol{position:relative;display:flex;flex-direction:column;gap:10px}
.jcol.ins:after{content:'';position:absolute;right:-22px;top:10%;bottom:10%;width:12px;border:1px solid #4e6068;border-left:0;border-radius:0 10px 10px 0;opacity:0;transition:opacity .4s}
.jf.joined .jcol.ins:after{opacity:1}
.jcol+.jcol:before{content:'';position:absolute;left:-42px;top:50%;width:34px;height:1px;background:#4e6068}
.jcol+.jcol:after{content:'';position:absolute;left:-12px;top:calc(50% - 3px);width:6px;height:6px;border-top:1px solid #4e6068;border-right:1px solid #4e6068;transform:rotate(45deg)}
.jf [data-from]{opacity:0;transform:translateY(8px);transition:opacity .4s,transform .4s}
.jf [data-from].on{opacity:1;transform:none}
.jcard{border:1px solid var(--border);border-radius:12px;background:var(--surface);padding:12px 14px}
.jcard .ol{font-size:9px;letter-spacing:1.2px;margin-bottom:9px}
.jf [data-v="a"]{--c:var(--mint)}.jf [data-v="b"]{--c:var(--cyan)}.jf [data-v="c"]{--c:var(--amber)}
.jtrace{display:grid;grid-template-columns:repeat(var(--n),minmax(0,1fr));gap:3px}
.jtrace span{height:14px;border-radius:3px;border:1px solid var(--border);background:#0c1317}
.jtrace span[data-v]{border-color:var(--c);background:color-mix(in srgb,var(--c) 22%,transparent)}
.jm{width:100%;border-collapse:collapse;font:12px var(--mono)}
.jm th,.jm td{padding:5px 8px;text-align:left;font-weight:400;border-top:1px solid var(--line)}
.jm thead th{border-top:0;font-size:10px;letter-spacing:.5px;color:var(--muted)}
.jm thead th i{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--c);margin-right:6px;vertical-align:middle}
.jm tbody th{font:13px var(--sans);color:var(--text)}
.jm td{color:var(--dim)}.jm td.hit{color:var(--c);font-weight:600}
.jc{list-style:none;margin:0;padding:0}
.jc li{display:grid;grid-template-columns:8px 56px minmax(0,1fr);gap:10px;align-items:baseline;padding:5px 0;border-top:1px solid var(--line);font-size:12px;line-height:1.4}
.jc li:first-child{border-top:0}
.jc li i{width:8px;height:8px;border-radius:50%;background:var(--c);align-self:center}
.jc li b{font:400 11px var(--mono);color:var(--c)}
.jc li span{color:var(--muted)}
.jjudge{border:1px solid var(--mint);border-radius:16px;background:#13201a;padding:22px;box-shadow:0 0 50px #bcf78322}
.jjudge strong{display:block;font-size:clamp(18px,1.5vw,24px);font-weight:600;line-height:1.25;margin:10px 0 8px}
.jjudge p{font-size:12px;line-height:1.6;color:var(--muted)}
.jjudge .nogold{margin-top:14px;padding-top:10px;border-top:1px solid #1f3327;font:10px var(--mono);letter-spacing:1px;text-transform:uppercase;color:var(--muted)}
.jout{border:1px solid var(--mint);border-radius:14px;background:var(--surface);padding:18px 20px}
.jout strong{display:block;font-size:clamp(18px,1.5vw,24px);font-weight:600;margin:8px 0 6px}
.jout p{font-size:13px;line-height:1.55;color:var(--text)}
.jout small{display:block;margin-top:10px;font-size:11px;line-height:1.5;color:var(--violet)}
.jnext{position:absolute;left:4px;top:calc(100% + 10px);white-space:nowrap;font:10px var(--mono);letter-spacing:1px;color:var(--dim)}
/* 03 · the architecture diagram builds in stages 0 … 4 */
.arch{margin-top:3vh;flex:1;display:flex;flex-direction:column;gap:18px;min-height:0}
.arch-row{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:30px;align-items:center;padding-top:22px}
.acol{position:relative;display:flex;flex-direction:column;gap:8px;transition:opacity .45s,transform .45s}
.acol+.acol:before{content:'';position:absolute;left:-26px;top:50%;width:22px;height:1px;background:#4e6068}
.acol+.acol:after{content:'';position:absolute;left:-8px;top:calc(50% - 3px);width:6px;height:6px;border-top:1px solid #4e6068;border-right:1px solid #4e6068;transform:rotate(45deg)}
.arch [data-from]{opacity:0;transform:translateX(14px)}
.arch[data-stage="0"] [data-from="0"],.arch[data-stage="1"] [data-from="0"],.arch[data-stage="2"] [data-from="0"],.arch[data-stage="3"] [data-from="0"],.arch[data-stage="4"] [data-from="0"],
.arch[data-stage="1"] [data-from="1"],.arch[data-stage="2"] [data-from="1"],.arch[data-stage="3"] [data-from="1"],.arch[data-stage="4"] [data-from="1"],
.arch[data-stage="2"] [data-from="2"],.arch[data-stage="3"] [data-from="2"],.arch[data-stage="4"] [data-from="2"],
.arch[data-stage="3"] [data-from="3"],.arch[data-stage="4"] [data-from="3"],
.arch[data-stage="4"] [data-from="4"]{opacity:1;transform:none}
.abox{border:1px solid var(--border);border-radius:10px;background:var(--surface);padding:12px 14px;transition:opacity .4s,border-color .4s}
.abox b{display:block;font-size:clamp(12px,1vw,15px);font-weight:600;line-height:1.25}
.abox small{display:block;font-size:11px;line-height:1.4;color:var(--muted);margin-top:4px}
.abox.violet{border-color:#4a3f78;background:#1c1830}
.abox.mint{border-color:var(--mint);background:#13201a}
.abox.la{border-color:#3a5a3a;background:#13201a}.abox.lb{border-color:#2f4f58;background:#101f24}.abox.lc{border-color:#5a4d2f;background:#231e12}
.alabel{position:absolute;left:0;bottom:calc(100% + 6px);white-space:nowrap;font:9px var(--mono);letter-spacing:1.2px;text-transform:uppercase;color:var(--dim)}
.alabel.violet{color:var(--violet)}.alabel.mint{color:var(--mint)}
.acol.judge{display:grid;align-items:center}
.acol.judge>div{grid-area:1/1;position:relative;display:flex;flex-direction:column;gap:8px;transition:opacity .45s,transform .45s}
.acol.judge .three{opacity:0;transform:translateY(10px)}
.arch[data-stage="1"] .judge .one,.arch[data-stage="2"] .judge .one,.arch[data-stage="3"] .judge .one,.arch[data-stage="4"] .judge .one{opacity:0;transform:translateY(-10px);pointer-events:none}
.arch[data-stage="1"] .judge .three,.arch[data-stage="2"] .judge .three,.arch[data-stage="3"] .judge .three,.arch[data-stage="4"] .judge .three{opacity:1;transform:none}
.acol.decide{gap:28px}
.acol.decide>div{position:relative}
.acol.decide .alabel{left:14px}
.acol.decide .llm{opacity:0;transform:translateY(10px);transition:opacity .45s,transform .45s}
.arch[data-stage="4"] .decide .llm{opacity:1;transform:none}
.arch[data-stage="4"] .decide .policy .abox{opacity:.4;border-style:dashed}
.rl{display:flex;align-items:center;gap:16px;border:1px solid var(--border);border-radius:12px;padding:12px 16px;background:#0c1317;position:relative;transition:opacity .45s,transform .45s}
.rl-chips{flex:1;min-width:0;display:flex;flex-wrap:wrap;gap:8px 10px;align-items:center}
.rl-chips span{border:1px solid #2b3a42;border-radius:8px;padding:7px 10px;font-size:12px;line-height:1.35;color:var(--muted);background:var(--surface)}
.rl-chips span b{display:block;font:400 9px var(--mono);letter-spacing:1px;color:var(--cyan);text-transform:uppercase;margin-bottom:2px}
.rl-chips i{font-style:normal;color:#4e6068}
.rl .frozen{flex:0 1 230px;line-height:1.5;font:10px var(--mono);letter-spacing:1px;color:var(--red);border:1px solid #5a3a3a;border-radius:6px;padding:5px 9px;background:#1f1414;opacity:0;transition:opacity .4s}
.arch[data-stage="4"] .rl{opacity:.55}
.arch[data-stage="4"] .rl .frozen{opacity:1}
.acaps{position:relative;min-height:46px}
.acap{position:absolute;left:0;top:0;margin:0;font-size:clamp(13px,1.05vw,15px);line-height:1.6;color:var(--text);opacity:0;transition:opacity .35s}
.arch[data-stage="0"] .acap[data-stage="0"],.arch[data-stage="1"] .acap[data-stage="1"],.arch[data-stage="2"] .acap[data-stage="2"],.arch[data-stage="3"] .acap[data-stage="3"],.arch[data-stage="4"] .acap[data-stage="4"]{opacity:1}
.rlrow{display:grid;grid-template-columns:110px minmax(0,1fr);gap:14px;padding:9px 0;border-top:1px solid var(--line);font-size:13px;line-height:1.5}
.rlrow b{font:400 11px var(--mono);letter-spacing:1px;color:var(--cyan);text-transform:uppercase;padding-top:2px}
.rlrow span{color:var(--muted)}
table.blame{border-collapse:collapse;width:100%;font:13px var(--mono)}
.blame th,.blame td{text-align:left;padding:11px 12px;border-bottom:1px solid var(--line);font-weight:400}
.blame th{color:var(--muted);font-size:10px;letter-spacing:1px;text-transform:uppercase}
.blame .hot{color:var(--amber)}.blame .max{color:var(--red);font-weight:600}
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
.safeguard{margin-top:3vh;padding:14px 16px;border-left:2px solid var(--cyan);font-size:13px;line-height:1.6;color:var(--muted)}
.safeguard b{color:var(--text);font-weight:600}

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
 .vrow{min-height:58px}
 .rung{padding:6px 12px;font-size:12px}.opt{padding:8px 14px;font-size:12px}
 .field{grid-template-columns:repeat(40,10px)}.field i{width:10px;height:10px}
 .q{padding:12px 0}.q span{font-size:clamp(13px,1.1vw,16px)}
 .insight{padding:12px 0}.flow .box{padding:10px 16px}.flow .judge{padding:14px}
}
/* motion off */
.rm .rv,.rm .bar,.rm .crow .cl,.rm .crow .cf,.rm .title-art path{transition:none!important;transform:none!important;opacity:1!important;stroke-dashoffset:0!important}
.rm *{animation:none!important}
@media(max-width:1000px){.grid2,.grid3,.grid4{grid-template-columns:1fr}.live{flex-direction:column}.live .side{width:auto}.live .frame{min-height:60vh}.net-wrap{left:var(--rail);top:55%}.ans{grid-template-columns:44px 1fr}.ans p{grid-column:2}.field{grid-template-columns:repeat(24,10px)}}
@media(max-width:760px){:root{--rail:40px}.topbar{padding:0 14px}.rail{gap:10px}.handoff{padding:0 24px}.exit{right:14px}}
'''
