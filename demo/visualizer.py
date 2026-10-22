"""n8n-style Château de Versailles visualizer: one self-contained HTML page plus a stdlib HTTP + SSE server.

Static export embeds every scenario's frames as JSON; the live server embeds only scenario metadata and streams
frames from ``GET /api/stream?scenario=<id>`` as Server-Sent Events. The page has no external dependencies.
"""
import html
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from benchmark.env.workflow import AGENTS

from .scenarios import MILESTONES, SCENARIOS, run_scenario, scenario_meta

WORLD_W, WORLD_H = 1340, 760
NODE_W, NODE_H = 210, 96
BLOCK, GAP = 44, 14
NODES = {   # id: (x, y, title, kicker)
    'user': (40, 262, 'User Simulator', 'TRIGGER'),
    'planner': (360, 262, 'planner', 'AI AGENT · ROUTER'),
    'order_agent': (840, 40, 'order_agent', 'SPECIALIST I · PENDING'),
    'product_agent': (840, 280, 'product_agent', 'SPECIALIST II · CATALOGUE'),
    'returns_agent': (840, 520, 'returns_agent', 'SPECIALIST III · DELIVERED'),
}
MONOGRAMS = {
    'find_user_id_by_email': 'ID', 'find_user_id_by_name_zip': 'ZIP', 'get_user_details': 'USR',
    'get_order_details': 'ORD', 'get_product_details': 'PRD', 'list_all_product_types': 'LST',
    'transfer_to_human_agents': 'ESC', 'modify_pending_order_address': 'ADR', 'modify_pending_order_items': 'ITM',
    'modify_pending_order_payment': 'PAY', 'modify_user_address': 'UAD', 'cancel_pending_order': 'CXL',
    'exchange_delivered_order_items': 'EXC', 'return_delivered_order_items': 'RET', 'calculate': '∑',
}


# ------------------------------------------------------------------------------ static layout
def _ports(node):
    x, y = NODES[node][:2]
    return {'in': (x, y + NODE_H / 2), 'out': (x + NODE_W, y + NODE_H / 2), 'tool': (x + NODE_W / 2, y + NODE_H)}


def _bezier(a, b, vertical=False):
    (x1, y1), (x2, y2) = a, b
    if vertical:
        d = max(30, (y2 - y1) / 2)
        return f'M{x1:.0f},{y1:.0f} C{x1:.0f},{y1 + d:.0f} {x2:.0f},{y2 - d:.0f} {x2:.0f},{y2:.0f}'
    d = max(40, (x2 - x1) / 2)
    return f'M{x1:.0f},{y1:.0f} C{x1 + d:.0f},{y1:.0f} {x2 - d:.0f},{y2:.0f} {x2:.0f},{y2:.0f}'


def _tool_positions(agent):
    """Top-left corners of an agent's tool mini-blocks: a row under the agent (centred under the planner)."""
    tools = AGENTS[agent]['tools']
    x, y = NODES[agent][:2]
    width = len(tools) * BLOCK + (len(tools) - 1) * GAP
    left = x + NODE_W / 2 - width / 2 if agent == 'planner' else x
    top = y + NODE_H + (86 if agent == 'planner' else 52)
    return {t: (left + i * (BLOCK + GAP), top) for i, t in enumerate(tools)}


def render_canvas():
    """(svg wires, html nodes + mini-blocks) for the world layer."""
    wires = [f'<path id="wire-user-planner" class="wire" d="{_bezier(_ports("user")["out"], _ports("planner")["in"])}"/>']
    for spec in AGENTS['planner']['handoff']:
        wires.append(f'<path id="wire-planner-{spec}" class="wire" '
                     f'd="{_bezier(_ports("planner")["out"], _ports(spec)["in"])}"/>')
    nodes, blocks = [], []
    for node, (x, y, title, kicker) in NODES.items():
        tools = AGENTS.get(node, {}).get('tools', [])
        ports = '' if node == 'user' else '<span class="port port-in"></span>'
        ports += '<span class="port port-out"></span>' if node in ('user', 'planner') else ''
        ports += '<span class="port port-tool"></span>' if tools else ''
        sub = ('Simulated customer · scripted fixture' if node == 'user' else
               f'{len(tools)} tools' + (' · routes to 3 specialists' if node == 'planner' else ''))
        kind = 'node-trigger' if node == 'user' else 'node-agent'
        nodes.append(f'<div class="node {kind}" id="node-{node}" data-node="{node}" style="left:{x}px;top:{y}px;'
                     f'width:{NODE_W}px;height:{NODE_H}px">{ports}<div class="node-kicker">{html.escape(kicker)}</div>'
                     f'<div class="node-title">{html.escape(title)}</div><div class="node-sub">{html.escape(sub)}</div>'
                     '<div class="node-state"></div></div>')
        port = _ports(node)['tool']
        for tool, (bx, by) in (_tool_positions(node).items() if tools else ()):
            wires.append(f'<path id="tether-{node}-{tool}" class="wire tether" '
                         f'd="{_bezier(port, (bx + BLOCK / 2, by), vertical=True)}"/>')
            blocks.append(f'<div class="tool-miniblock" id="tb-{node}-{tool}" data-agent="{node}" data-tool="{tool}" '
                          f'style="left:{bx:.0f}px;top:{by:.0f}px" tabindex="0" aria-label="{tool} ({node})">'
                          f'<span class="mono-gram">{MONOGRAMS.get(tool, tool[:3].upper())}</span>'
                          f'<div class="tool-popper">{tool}</div></div>')
    token = ('<g id="flow-token" transform="translate(-100,-100)"><circle class="seal" r="13"/>'
             '<circle class="seal-ring" r="9"/><text class="seal-mark" text-anchor="middle" dy="4">⚜</text></g>')
    svg = (f'<svg id="wires" width="{WORLD_W}" height="{WORLD_H}" viewBox="0 0 {WORLD_W} {WORLD_H}" '
           f'aria-hidden="true">{"".join(wires)}{token}</svg>')
    return svg, ''.join(nodes) + ''.join(blocks)


def _buttons(meta):
    return ''.join(f'<button class="scenario-btn" data-sid="{m["id"]}" title="{html.escape(m["summary"])}">'
                   f'{html.escape(m["title"])}</button>' for m in meta)


def build_html(scenarios=None, live=False):
    """The page. ``scenarios`` (results of run_scenario) are embedded for offline playback; live pages stream."""
    meta = scenario_meta() if live else [{k: s[k] for k in ('id', 'title', 'summary', 'task_id')} for s in scenarios]
    data = {'live': live, 'scenarios': meta, 'frames': {} if live else {s['id']: s['frames'] for s in scenarios},
            'results': {} if live else {s['id']: {k: s[k] for k in ('decision', 'hypotheses', 'rl', 'operator')}
                                        for s in scenarios},
            'milestones': MILESTONES}
    svg, nodes = render_canvas()
    payload = json.dumps(data, ensure_ascii=False).replace('</', '<\\/')
    milestones = ''.join(f'<li class="milestone" id="ms-{i}"><span class="ms-dot"></span>{html.escape(m)}</li>'
                         for i, m in enumerate(['I. P1/P2/P3 Attribution Checks', 'II. DualLoopRLJudge Ranking',
                                                'III. Rollback & Directive Replay',
                                                'IV. Fresh-Context Verifier & θ Update'], 1))
    return (PAGE.replace('__SVG__', svg).replace('__NODES__', nodes).replace('__BUTTONS__', _buttons(meta))
            .replace('__MILESTONES__', milestones).replace('__MODE__', 'Live SSE' if live else 'Offline export')
            .replace('__WORLD_W__', str(WORLD_W)).replace('__WORLD_H__', str(WORLD_H)).replace('__DATA__', payload))


def build_demo_html(out_path, scenario_ids=None):
    """Run the curated scenarios and write the self-contained offline page; returns the path."""
    from .scenarios import all_scenarios
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(all_scenarios(scenario_ids)), encoding='utf-8')
    return out


# ------------------------------------------------------------------------------ live server
def make_server(host='127.0.0.1', port=8765, delay=0.9, comparison_service=None):
    """ThreadingHTTPServer serving the page, /api/scenarios and /api/stream (SSE). ``delay``: seconds per frame."""
    from .comparison import ComparisonService
    from .comparison_page import PAGE as COMPARISON_PAGE
    comparisons = comparison_service or ComparisonService()
    page = build_html(live=True).encode('utf-8')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, value):
            return self._send(code, json.dumps(value, allow_nan=False).encode('utf-8'), 'application/json')

        def do_POST(self):
            origin = self.headers.get('Origin')
            if origin:
                parsed = urlparse(origin)
                if parsed.scheme != 'http' or parsed.netloc != self.headers.get('Host') or parsed.path not in ('', '/'):
                    return self._json(403, {'error': 'Cross-origin comparison requests are not allowed.'})
            elif self.headers.get('Sec-Fetch-Site') == 'cross-site':
                return self._json(403, {'error': 'Cross-origin comparison requests are not allowed.'})
            if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
                return self._json(415, {'error': 'Send an application/json request.'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096:
                    raise ValueError('Request body must be between 1 and 4096 bytes.')
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError('Request must be a JSON object.')
                path = urlparse(self.path).path
                if path == '/api/comparisons':
                    if set(body) != {'scenario', 'model'}:
                        raise ValueError('Supply scenario and model only.')
                    return self._json(202, comparisons.start(body['scenario'], body['model']))
                parts = path.strip('/').split('/')
                if len(parts) == 4 and parts[:2] == ['api', 'comparisons'] and parts[3] == 'cancel':
                    if body:
                        raise ValueError('Cancel body must be empty.')
                    return self._json(200, comparisons.cancel(parts[2]))
                return self._json(404, {'error': 'Unknown comparison route.'})
            except KeyError:
                return self._json(404, {'error': 'Comparison not found; it may have expired or the server restarted.'})
            except (ValueError, TypeError, UnicodeError) as exc:
                return self._json(400, {'error': str(exc)})
            except RuntimeError as exc:
                return self._json(409, {'error': str(exc)})
            except OSError as exc:
                return self._json(503, {'error': 'Local Ollama is unavailable: ' + str(exc)})

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == '/compare':
                return self._send(200, COMPARISON_PAGE.encode('utf-8'), 'text/html; charset=utf-8')
            if url.path == '/api/models':
                try:
                    return self._json(200, {'models': comparisons.model_catalog()})
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    return self._json(503, {'error': 'Cannot read the local Ollama model catalog: ' + str(exc)})
            if url.path.startswith('/api/comparisons/'):
                job_id = url.path[len('/api/comparisons/'):]
                try:
                    cursor = int(parse_qs(url.query).get('after', ['0'])[0])
                    return self._json(200, comparisons.snapshot(job_id, cursor))
                except KeyError:
                    return self._json(404, {'error': 'Comparison not found; it may have expired or the server restarted.'})
                except ValueError as exc:
                    return self._json(400, {'error': str(exc)})
            if url.path in ('/', '/index.html'):
                return self._send(200, page, 'text/html; charset=utf-8')
            if url.path == '/api/scenarios':
                return self._send(200, json.dumps(scenario_meta()).encode('utf-8'), 'application/json')
            if url.path == '/api/stream':
                query = parse_qs(url.query)
                sid = query.get('scenario', [''])[0]
                if sid not in SCENARIOS:
                    return self._send(404, json.dumps({'error': f'unknown scenario {sid!r}'}).encode('utf-8'),
                                      'application/json')
                try:
                    speed = max(0.25, min(4.0, float(query.get('speed', ['1'])[0])))
                except ValueError:
                    speed = 1.0
                return self._stream(sid, delay / speed)
            return self._send(404, b'not found', 'text/plain')

        def _stream(self, sid, pause):
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'close')
            self.end_headers()
            result = run_scenario(sid)        # a fresh live run of the MAS and AgentReroute per request

            def event(name, obj):
                self.wfile.write(f'event: {name}\ndata: {json.dumps(obj, ensure_ascii=False)}\n\n'.encode('utf-8'))
                self.wfile.flush()
            try:
                event('meta', {k: result[k] for k in ('id', 'title', 'summary', 'decision', 'hypotheses', 'rl',
                                                      'operator')})
                for frame in result['frames']:
                    event('frame', frame)
                    time.sleep(pause)
                event('end', {'frames': len(result['frames'])})
            except (BrokenPipeError, ConnectionResetError):
                pass                           # viewer switched scenario or closed the tab
            self.close_connection = True

    class DemoServer(ThreadingHTTPServer):
        def server_close(self):
            comparisons.close()
            super().server_close()

    server = DemoServer((host, port), Handler)
    server.daemon_threads = True
    return server


def serve(port=8765, host='127.0.0.1', delay=0.9):
    server = make_server(host, port, delay)
    print(f'AgentReroute Versailles demo at http://{host}:{server.server_address[1]}/  (Ctrl+C to stop)')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


# ------------------------------------------------------------------------------ page template
PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Versailles MAS Visualizer</title>
<style>
:root{
  --vellum:#FAF6EE; --dot:#C8B898; --alabaster:#FFFDF9; --parchemin:#F3ECE0; --hairline:#D8C7A8;
  --royal:#1B365D; --royal-soft:#E6ECF5; --gold:#B68D40; --gold-soft:#F5EBD6;
  --carmine:#8B1E24; --carmine-soft:#F9E6E7; --emerald:#1E5631; --emerald-soft:#E5F0E8;
  --ink:#2A2418; --muted:#6E604A;
  --serif:Georgia,"Palatino Linotype","Book Antiqua",serif;
  --sans:system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;
  --mono:ui-monospace,Consolas,"Courier New",monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--vellum);color:var(--ink);font-family:var(--sans);font-size:14px}
button,select{font-family:var(--sans);font-size:13px;color:var(--ink)}
.page{max-width:1480px;margin:0 auto;padding:18px 16px 28px}
header.masthead{display:flex;flex-wrap:wrap;gap:12px 24px;align-items:flex-end;justify-content:space-between;
  border-bottom:1px solid var(--hairline);padding-bottom:12px;margin-bottom:14px}
.brand h1{font-family:var(--serif);font-weight:normal;color:var(--royal);font-size:26px;margin:0;letter-spacing:.2px}
.brand h1 .fleur{color:var(--gold)}
.brand p{margin:4px 0 0;color:var(--muted);font-family:var(--serif);font-style:italic}
.mode{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--gold);border:1px solid var(--gold);
  border-radius:999px;padding:3px 10px;background:var(--gold-soft)}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.ctl{background:var(--alabaster);border:1px solid var(--hairline);border-radius:8px;padding:7px 12px;cursor:pointer}
.ctl:hover{border-color:var(--gold)}
.ctl.primary{background:var(--royal);border-color:var(--royal);color:#fff;min-width:84px}
.ctl:focus-visible,.scenario-btn:focus-visible,.trace-row:focus-visible,.tool-miniblock:focus-visible{outline:2px solid var(--gold);outline-offset:2px}
.scenarios{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px}
.scenario-btn{font-family:var(--serif);background:var(--alabaster);border:1px solid var(--hairline);border-radius:8px;
  padding:8px 14px;cursor:pointer;color:var(--ink);font-size:14px}
.scenario-btn.on{background:var(--royal-soft);border-color:var(--royal);color:var(--royal)}
.summary{color:var(--muted);margin:0 0 14px;font-family:var(--serif);font-size:15px;min-height:1.4em}
.summary b{color:var(--royal);font-weight:normal}
/* ---------- chart card ---------- */
.chart{background:var(--alabaster);border:1px solid var(--hairline);border-radius:14px;overflow:hidden;
  box-shadow:0 1px 0 #fff inset,0 8px 24px -18px rgba(70,50,10,.35)}
.chart-head{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;border-bottom:1px solid var(--hairline);
  background:var(--parchemin)}
.chart-head h2{margin:0;font-family:var(--serif);font-weight:normal;font-size:17px;color:var(--royal)}
.zoombar{display:flex;gap:4px;align-items:center}
.zoombar button{background:var(--alabaster);border:1px solid var(--hairline);border-radius:6px;min-width:34px;height:30px;cursor:pointer}
.zoombar button:hover{border-color:var(--gold)}
#zoom-label{min-width:58px}
.canvas{position:relative;height:min(64vh,620px);min-height:380px;overflow:hidden;cursor:grab;
  background-color:var(--vellum);background-image:radial-gradient(circle,var(--dot) 1.1px,transparent 1.4px);
  background-size:22px 22px}
.canvas.dragging{cursor:grabbing}
#world{position:absolute;left:0;top:0;width:__WORLD_W__px;height:__WORLD_H__px;transform-origin:0 0}
#wires{position:absolute;left:0;top:0;overflow:visible;pointer-events:none}
.wire{fill:none;stroke:var(--hairline);stroke-width:2.2}
.wire.tether{stroke-dasharray:4 5;stroke-width:1.5}
.wire-active,.wire-gold,.wire-fault,.wire-recovered{stroke-width:3;stroke-dasharray:9 7;animation:march .9s linear infinite}
.wire-active{stroke:var(--royal)}
.wire-gold{stroke:var(--gold)}
.wire-fault{stroke:var(--carmine)}
.wire-recovered{stroke:var(--emerald)}
.wire.tether.wire-active,.wire.tether.wire-gold,.wire.tether.wire-fault,.wire.tether.wire-recovered{stroke-width:2.2}
@keyframes march{to{stroke-dashoffset:-32}}
#flow-token .seal{fill:var(--royal);stroke:var(--gold);stroke-width:2.5}
#flow-token .seal-ring{fill:none;stroke:var(--gold-soft);stroke-width:1}
#flow-token .seal-mark{fill:var(--gold-soft);font-size:12px;font-family:var(--serif)}
#flow-token.fault .seal{fill:var(--carmine)}
#flow-token.recovered .seal{fill:var(--emerald)}
.node{position:absolute;background:var(--alabaster);border:1.5px solid var(--hairline);border-radius:12px;
  padding:12px 16px;box-shadow:0 6px 16px -12px rgba(60,40,10,.45);transition:border-color .25s,box-shadow .25s,background .25s}
.node-trigger{border-radius:44px 12px 12px 44px;padding-left:26px}
.node-kicker{font-size:10px;letter-spacing:.14em;color:var(--gold);font-weight:600}
.node-title{font-family:var(--serif);font-size:19px;color:var(--royal);margin-top:3px}
.node-sub{font-size:11.5px;color:var(--muted);margin-top:3px}
.node-state{position:absolute;right:12px;top:-9px;font-size:9.5px;letter-spacing:.1em;font-weight:700}
.node-state::after{background:var(--alabaster);border:1px solid currentColor;border-radius:999px;padding:1px 7px}
.port{position:absolute;width:12px;height:12px;border-radius:50%;background:var(--alabaster);border:2px solid var(--gold)}
.port-in{left:-7px;top:calc(50% - 6px)}
.port-out{right:-7px;top:calc(50% - 6px)}
.port-tool{left:calc(50% - 6px);bottom:-7px;border-radius:2px;transform:rotate(45deg);background:var(--gold-soft)}
.node.active{border-color:var(--royal);background:var(--royal-soft);box-shadow:0 0 0 4px rgba(27,54,93,.12),0 8px 18px -12px rgba(27,54,93,.6)}
.node.active .node-state::after{content:"ACTIVE";color:var(--royal)}
.node.fault{border-color:var(--carmine);background:var(--carmine-soft);box-shadow:0 0 0 4px rgba(139,30,36,.13)}
.node.fault .node-state::after{content:"FAULT";color:var(--carmine)}
.node.blamed{border-color:var(--carmine);border-style:dashed;background:var(--carmine-soft);animation:pulse 1.4s ease-in-out infinite}
.node.blamed .node-state::after{content:"BLAMED";color:var(--carmine)}
.node.recovered{border-color:var(--emerald);background:var(--emerald-soft);box-shadow:0 0 0 4px rgba(30,86,49,.12)}
.node.recovered .node-state::after{content:"REPAIRED";color:var(--emerald)}
@keyframes pulse{50%{box-shadow:0 0 0 8px rgba(139,30,36,.10)}}
.tool-miniblock{position:absolute;width:44px;height:44px;border-radius:10px;background:var(--gold-soft);
  border:1.5px solid var(--gold);display:flex;align-items:center;justify-content:center;cursor:default;z-index:2;
  box-shadow:0 3px 8px -6px rgba(90,60,10,.6);transition:transform .2s,background .2s,border-color .2s}
.mono-gram{font-family:var(--mono);font-size:11.5px;font-weight:700;color:#7A5A1E}
.tool-miniblock.used{background:#fff}
.tool-miniblock.active{z-index:6;transform:translateY(-3px);background:var(--royal);border-color:var(--royal)}
.tool-miniblock.active .mono-gram{color:#fff}
.tool-miniblock.error{background:var(--carmine);border-color:var(--carmine)}
.tool-miniblock.error .mono-gram{color:#fff}
.tool-popper{position:absolute;bottom:calc(100% + 9px);left:50%;transform:translate(-50%,4px);white-space:nowrap;
  background:var(--alabaster);border:1px solid var(--gold);border-radius:7px;padding:5px 9px;font-family:var(--mono);
  font-size:11.5px;color:var(--royal);opacity:0;pointer-events:none;transition:opacity .2s,transform .2s;
  box-shadow:0 6px 14px -8px rgba(60,40,10,.5)}
.tool-popper::after{content:"";position:absolute;left:50%;top:100%;margin-left:-5px;border:5px solid transparent;border-top-color:var(--gold)}
.tool-miniblock:hover,.tool-miniblock:focus{z-index:7}
.tool-miniblock:hover .tool-popper,.tool-miniblock:focus .tool-popper,.tool-miniblock.active .tool-popper{opacity:1;transform:translate(-50%,0)}
.tool-miniblock.error .tool-popper{border-color:var(--carmine);color:var(--carmine);background:var(--carmine-soft)}
.tool-miniblock.error .tool-popper::after{border-top-color:var(--carmine)}
/* ---------- debugger progress bar ---------- */
.progress{border-top:1px solid var(--hairline);background:var(--parchemin);padding:12px 16px 14px}
.progress-top{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.progress-top .label{font-family:var(--serif);color:var(--royal);font-size:15px}
.badge{display:inline-block;font-size:10.5px;font-weight:700;letter-spacing:.1em;border-radius:999px;padding:3px 9px;
  border:1px solid currentColor;text-transform:uppercase}
.b-normal{color:var(--royal);background:var(--royal-soft)}
.b-fault{color:var(--carmine);background:var(--carmine-soft)}
.b-debug{color:#7A5A1E;background:var(--gold-soft)}
.b-recovered{color:var(--emerald);background:var(--emerald-soft)}
#pct{margin-left:auto;font-family:var(--serif);font-size:20px;color:var(--royal)}
.bar{height:10px;border-radius:999px;background:var(--alabaster);border:1px solid var(--hairline);margin:10px 0 10px;overflow:hidden}
#bar-fill{height:100%;width:0;background:linear-gradient(90deg,var(--royal),var(--gold));transition:width .6s ease}
.milestones{list-style:none;margin:0;padding:0;display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.milestone{font-size:12px;color:var(--muted);display:flex;gap:7px;align-items:center;font-family:var(--serif)}
.ms-dot{width:11px;height:11px;flex:none;border-radius:50%;border:1.5px solid var(--hairline);background:var(--alabaster)}
.milestone.done{color:var(--ink)}
.milestone.done .ms-dot{background:var(--gold);border-color:var(--gold)}
.milestone.current{color:var(--royal);font-weight:bold}
.milestone.current .ms-dot{background:var(--royal);border-color:var(--royal);box-shadow:0 0 0 3px var(--royal-soft)}
/* ---------- bottom dock ---------- */
.dock{display:grid;grid-template-columns:repeat(12,1fr);gap:14px;margin-top:14px}
.panel{grid-column:span 6;background:var(--alabaster);border:1px solid var(--hairline);border-radius:14px;display:flex;flex-direction:column;min-width:0}
.panel h3{margin:0;padding:10px 14px;border-bottom:1px solid var(--hairline);background:var(--parchemin);font-family:var(--serif);
  font-weight:normal;color:var(--royal);font-size:16px;border-radius:14px 14px 0 0}
.panel h3 small{color:var(--muted);font-style:italic;font-size:13px}
.panel-body{padding:12px 14px;min-width:0}
#step-title{font-family:var(--serif);font-size:18px;color:var(--ink);margin:0 0 8px}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}
.chip{font-size:11px;border:1px solid var(--hairline);border-radius:6px;padding:2px 8px;background:var(--parchemin);font-family:var(--mono)}
#step-narrative{font-family:var(--serif);font-size:15px;line-height:1.5;margin:0 0 10px;white-space:pre-wrap;word-break:break-word}
#step-fault{display:none;font-size:12.5px;color:var(--carmine);background:var(--carmine-soft);border:1px solid var(--carmine);
  border-radius:8px;padding:6px 10px;margin-bottom:10px}
pre#step-payload{margin:0;max-height:300px;overflow:auto;background:var(--parchemin);border:1px solid var(--hairline);
  border-radius:8px;padding:10px 12px;font-family:var(--mono);font-size:12px;line-height:1.45;white-space:pre-wrap;word-break:break-word}
#trace{list-style:none;margin:0;padding:6px;max-height:440px;overflow:auto}
.trace-row{width:100%;display:grid;grid-template-columns:38px 104px 1fr;gap:8px;align-items:center;text-align:left;
  border:1px solid transparent;border-radius:8px;background:none;padding:6px 8px;cursor:pointer}
.trace-row:hover{background:var(--parchemin)}
.trace-row.current{border-color:var(--royal);background:var(--royal-soft)}
.trace-row.future{opacity:.45}
.trace-row .seq{font-family:var(--mono);color:var(--muted);font-size:12px}
.trace-row .what{font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.trace-row .what b{font-family:var(--serif);font-weight:normal;color:var(--royal)}
.phase-sep{font-family:var(--serif);font-style:italic;color:var(--gold);font-size:12.5px;padding:8px 8px 2px;border-top:1px dashed var(--hairline);margin-top:4px}
footer{margin-top:16px;color:var(--muted);font-size:12px;font-family:var(--serif);font-style:italic}
@media (max-width:900px){.panel{grid-column:span 12}.milestones{grid-template-columns:repeat(2,1fr)}.brand h1{font-size:21px}}
</style>
</head>
<body>
<div class="page">
  <nav aria-label="Main navigation" style="display:flex;gap:22px;margin-bottom:18px;font-size:13px"><a href="/" aria-current="page" style="color:var(--royal)">Execution view</a><a href="/compare" style="color:var(--muted)">Compare systems →</a></nav>
  <header class="masthead">
    <div class="brand">
      <h1><span class="fleur">⚜</span> AgentReroute · Live MAS Control Propagation &amp; Recovery</h1>
      <p>Château de Versailles edition — a four-agent retail court, its faults, and their repair</p>
    </div>
    <div class="controls">
      <span class="mode" id="mode">__MODE__</span>
      <button class="ctl" id="btn-prev" aria-label="Previous step">◀ Prev</button>
      <button class="ctl primary" id="btn-play" aria-label="Play or pause">▶ Play</button>
      <button class="ctl" id="btn-next" aria-label="Next step">Next ▶</button>
      <select class="ctl" id="speed" aria-label="Playback speed">
        <option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="2">2x</option>
      </select>
    </div>
  </header>
  <nav class="scenarios" aria-label="Scenarios">__BUTTONS__</nav>
  <p class="summary" id="summary"></p>

  <section class="chart" aria-label="Workflow canvas">
    <div class="chart-head">
      <h2>Workflow Canvas <small style="color:var(--muted);font-style:italic">— drag to pan, wheel to zoom</small></h2>
      <div class="zoombar">
        <button id="zoom-out" aria-label="Zoom out">−</button>
        <button id="zoom-label" aria-label="Reset zoom to 100%">100%</button>
        <button id="zoom-in" aria-label="Zoom in">+</button>
        <button id="zoom-fit" aria-label="Fit view">Fit View</button>
      </div>
    </div>
    <div class="canvas" id="canvas">
      <div id="world">__SVG____NODES__</div>
    </div>
    <div class="progress" id="debugger">
      <div class="progress-top">
        <span class="label">AgentReroute Debugger &amp; Recovery</span>
        <span class="badge b-normal" id="dbg-badge">Idle</span>
        <span id="dbg-text" style="color:var(--muted);font-size:12.5px"></span>
        <span id="pct">0%</span>
      </div>
      <div class="bar"><div id="bar-fill"></div></div>
      <ol class="milestones">__MILESTONES__</ol>
    </div>
  </section>

  <section class="dock">
    <div class="panel" id="current-step">
      <h3>Current Step Inspection</h3>
      <div class="panel-body">
        <p id="step-title">—</p>
        <div class="chips"><span class="badge b-normal" id="step-status">NORMAL</span><span class="chip" id="step-actor">actor</span><span class="chip" id="step-phase">phase</span><span class="chip" id="step-index">step</span></div>
        <div id="step-fault"></div>
        <p id="step-narrative"></p>
        <pre id="step-payload">{}</pre>
      </div>
    </div>
    <div class="panel" id="live-trace">
      <h3>Live Trace Chronicle <small>— Gazette de Propagation</small></h3>
      <ol id="trace"></ol>
    </div>
  </section>
  <footer>Scripted agents are an offline fixture, not a model. AgentReroute (MPRLRecoveryPolicy) sees only the observable trace; injected-fault notes are shown to the viewer alone.</footer>
</div>
<script>
const DATA = __DATA__;
const WORLD = {w: __WORLD_W__, h: __WORLD_H__};
const $ = (id) => document.getElementById(id);
const S = {sid: null, frames: [], cur: -1, playing: false, speed: 1, timer: null, es: null, streaming: false,
           pan: {x: 0, y: 0}, scale: 1};
const PHASES = {initial_run: 'Original run', debugging: 'AgentReroute debugging', recovery_replay: 'Recovery replay',
                verified: 'Verification'};
const BADGE = {NORMAL: 'b-normal', FAULT: 'b-fault', DEBUG: 'b-debug', RECOVERED: 'b-recovered'};

/* ---------------- pan & zoom ---------------- */
const canvas = $('canvas'), world = $('world');
function applyView() {
  world.style.transform = `translate(${S.pan.x}px, ${S.pan.y}px) scale(${S.scale})`;
  canvas.style.backgroundSize = `${22 * S.scale}px ${22 * S.scale}px`;
  canvas.style.backgroundPosition = `${S.pan.x}px ${S.pan.y}px`;
  $('zoom-label').textContent = Math.round(S.scale * 100) + '%';
}
function zoomAt(next, cx, cy) {
  next = Math.min(1.65, Math.max(0.55, next));
  S.pan.x = cx - (cx - S.pan.x) * next / S.scale;
  S.pan.y = cy - (cy - S.pan.y) * next / S.scale;
  S.scale = next; applyView();
}
function fitView() {
  const r = canvas.getBoundingClientRect();
  S.scale = Math.min(1.65, Math.max(0.55, Math.min(r.width / (WORLD.w + 40), r.height / (WORLD.h + 40))));
  S.pan.x = (r.width - WORLD.w * S.scale) / 2; S.pan.y = (r.height - WORLD.h * S.scale) / 2; applyView();
}
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  const r = canvas.getBoundingClientRect();
  zoomAt(S.scale * (e.deltaY < 0 ? 1.1 : 1 / 1.1), e.clientX - r.left, e.clientY - r.top);
}, {passive: false});
let drag = null;
canvas.addEventListener('pointerdown', (e) => {
  if (e.target.closest('.tool-miniblock')) return;
  drag = {x: e.clientX - S.pan.x, y: e.clientY - S.pan.y}; canvas.classList.add('dragging');
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener('pointermove', (e) => { if (drag) { S.pan.x = e.clientX - drag.x; S.pan.y = e.clientY - drag.y; applyView(); } });
const endDrag = () => { drag = null; canvas.classList.remove('dragging'); };
canvas.addEventListener('pointerup', endDrag); canvas.addEventListener('pointercancel', endDrag);
const center = () => { const r = canvas.getBoundingClientRect(); return [r.width / 2, r.height / 2]; };
$('zoom-in').onclick = () => zoomAt(S.scale * 1.15, ...center());
$('zoom-out').onclick = () => zoomAt(S.scale / 1.15, ...center());
$('zoom-label').onclick = () => zoomAt(1, ...center());
$('zoom-fit').onclick = fitView;

/* ---------------- token animation ---------------- */
const token = $('flow-token');
let anim = null;
function pathFor(edge) {
  if (!edge) return null;
  const [a, b] = edge;
  if (b.startsWith('tool:')) return [document.getElementById(`tether-${a}-${b.slice(5)}`), false];
  if (a === 'user' || b === 'user') return [$('wire-user-planner'), a !== 'user'];
  if (a === 'planner') return [$(`wire-planner-${b}`), false];
  return [$(`wire-planner-${a}`), true];
}
function moveToken(path, reverse) {
  cancelAnimationFrame(anim);
  const len = path.getTotalLength(), t0 = performance.now(), dur = 700 / S.speed;
  const tick = (now) => {
    const k = Math.min(1, (now - t0) / dur), e = k < .5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
    const p = path.getPointAtLength(len * (reverse ? 1 - e : e));
    token.setAttribute('transform', `translate(${p.x},${p.y})`);
    if (k < 1) anim = requestAnimationFrame(tick);
  };
  anim = requestAnimationFrame(tick);
}
function parkToken(node) {
  const el = $(`node-${node}`);
  if (!el) return;
  cancelAnimationFrame(anim);
  token.setAttribute('transform', `translate(${el.offsetLeft + el.offsetWidth - 14},${el.offsetTop + 14})`);
}

/* ---------------- painting ---------------- */
function wireClass(f) {
  if (f.status === 'FAULT') return 'wire-fault';
  if (f.phase === 'recovery_replay' || f.status === 'RECOVERED') return 'wire-recovered';
  if (f.kind === 'tool' || f.kind === 'handoff') return 'wire-gold';
  return 'wire-active';
}
function paint(f) {
  document.querySelectorAll('.wire').forEach((w) => w.classList.remove('wire-active', 'wire-gold', 'wire-fault', 'wire-recovered'));
  document.querySelectorAll('.node').forEach((n) => n.classList.remove('active', 'fault', 'blamed', 'recovered'));
  document.querySelectorAll('.tool-miniblock').forEach((b) => b.classList.remove('active', 'error', 'used'));
  token.classList.remove('fault', 'recovered');
  if (f.status === 'FAULT') token.classList.add('fault');
  else if (f.phase === 'recovery_replay' || f.phase === 'verified') token.classList.add('recovered');
  for (const g of S.frames.slice(0, S.cur + 1)) {
    if (g.tool) { const b = document.getElementById(`tb-${g.tool.agent}-${g.tool.name}`); if (b) b.classList.add('used'); }
  }
  const node = (id) => document.getElementById(`node-${id}`);
  const route = pathFor(f.edge);
  if (route && route[0]) { route[0].classList.add(wireClass(f)); moveToken(route[0], route[1]); }
  if (f.kind === 'debug') {
    if (f.focus && node(f.focus)) node(f.focus).classList.add('blamed');
    parkToken(f.focus || 'planner');
  } else if (f.kind === 'verify') {
    const cls = f.verified ? 'recovered' : 'fault';
    ['user', 'planner'].forEach((n) => node(n).classList.add(cls));
    S.frames.forEach((g) => { if (g.phase === 'recovery_replay' && g.actor && node(g.actor)) node(g.actor).classList.add(cls); });
    if (!S.frames.some((g) => g.phase === 'recovery_replay'))
      S.frames.forEach((g) => { if (g.actor && node(g.actor)) node(g.actor).classList.add(cls); });
    parkToken('user');
  } else if (f.edge) {
    const cls = f.status === 'FAULT' ? 'fault' : f.phase === 'recovery_replay' ? 'recovered' : 'active';
    if (node(f.edge[0])) node(f.edge[0]).classList.add(cls);
    if (node(f.edge[1])) node(f.edge[1]).classList.add(f.handoff && f.handoff.fault ? 'fault' : cls === 'fault' ? 'active' : cls);
  }
  if (f.tool) {
    const b = document.getElementById(`tb-${f.tool.agent}-${f.tool.name}`);
    if (b) {
      b.classList.add('active'); if (f.tool.error || f.status === 'FAULT') b.classList.add('error');
      b.querySelector('.tool-popper').textContent = f.tool.popper_text;
    }
  }
  paintProgress(f); paintStep(f); paintTraceCursor();
}
function paintProgress(f) {
  $('bar-fill').style.width = f.debug_pct + '%';
  $('pct').textContent = f.debug_pct + '%';
  const badge = $('dbg-badge');
  let text = 'Monitoring · original run', cls = 'b-normal';
  if (f.phase === 'debugging') { text = 'Debugging'; cls = 'b-debug'; }
  if (f.phase === 'recovery_replay') { text = 'Replaying repair'; cls = 'b-recovered'; }
  if (f.phase === 'verified') {
    text = f.title.includes('Abstained') ? 'Verified clean · abstained' : f.verified ? 'Verified · θ updated' : 'Not verified';
    cls = f.verified ? 'b-recovered' : 'b-fault';
  }
  if (f.phase === 'initial_run' && f.status === 'FAULT') { text = 'Fault observed'; cls = 'b-fault'; }
  badge.textContent = text; badge.className = 'badge ' + cls;
  $('dbg-text').textContent = f.active_phase ? DATA.milestones[f.active_phase - 1] : '';
  for (let i = 1; i <= 4; i++) {
    const m = $(`ms-${i}`);
    m.classList.toggle('done', f.active_phase > i || (f.active_phase === i && i === 4));
    m.classList.toggle('current', f.active_phase === i && i < 4);
  }
}
function paintStep(f) {
  $('step-title').textContent = f.title;
  const st = $('step-status'); st.textContent = f.status; st.className = 'badge ' + BADGE[f.status];
  $('step-actor').textContent = f.actor || 'AgentReroute';
  $('step-phase').textContent = PHASES[f.phase] || f.phase;
  $('step-index').textContent = f.step_index ? `trace step ${f.step_index}` : `frame ${f.seq}`;
  $('step-narrative').textContent = f.narrative;
  const fault = $('step-fault');
  fault.style.display = f.fault ? 'block' : 'none';
  fault.textContent = f.fault ? `Injected fault: ${f.fault.operator} (${f.fault.mode}) — ${f.fault.note}` : '';
  $('step-payload').textContent = JSON.stringify(f.payload, null, 2);
}

/* ---------------- trace chronicle ---------------- */
function traceRow(f, i) {
  const li = document.createElement('li');
  if (i === 0 || S.frames[i - 1].phase !== f.phase) {
    const sep = document.createElement('div'); sep.className = 'phase-sep'; sep.textContent = PHASES[f.phase] || f.phase;
    li.appendChild(sep);
  }
  const b = document.createElement('button');
  b.className = 'trace-row'; b.dataset.i = i;
  const seq = document.createElement('span'); seq.className = 'seq'; seq.textContent = '#' + f.seq;
  const badge = document.createElement('span'); badge.className = 'badge ' + BADGE[f.status]; badge.textContent = f.status;
  const what = document.createElement('span'); what.className = 'what';
  const strong = document.createElement('b'); strong.textContent = f.actor || 'AgentReroute';
  what.append(strong, ' · ' + (f.tool ? f.tool.popper_text : f.title));
  b.append(seq, badge, what);
  b.onclick = () => { pause(); show(i); };
  li.appendChild(b);
  $('trace').appendChild(li);
}
function renderTrace() { $('trace').innerHTML = ''; S.frames.forEach(traceRow); }
function paintTraceCursor() {
  document.querySelectorAll('.trace-row').forEach((r) => {
    const i = +r.dataset.i;
    r.classList.toggle('current', i === S.cur); r.classList.toggle('future', i > S.cur);
    if (i === S.cur) {   // scroll the list only, never the page
      const list = $('trace'), top = r.offsetTop - list.offsetTop;
      if (top < list.scrollTop || top + r.offsetHeight > list.scrollTop + list.clientHeight)
        list.scrollTop = top - list.clientHeight / 2;
    }
  });
}

/* ---------------- playback ---------------- */
function show(i) {
  if (i < 0 || i >= S.frames.length) return;
  S.cur = i; paint(S.frames[i]);
}
function delay(f) { return (f && f.kind === 'debug' ? 2200 : 1400) / S.speed; }
function tick() {
  if (!S.playing) return;
  if (S.cur >= S.frames.length - 1) { if (!S.streaming) pause(); return; }
  show(S.cur + 1);
  S.timer = setTimeout(tick, delay(S.frames[S.cur]));
}
function play() {
  if (S.cur >= S.frames.length - 1 && !S.streaming) S.cur = -1;
  S.playing = true; $('btn-play').textContent = '❚❚ Pause';
  clearTimeout(S.timer); tick();
}
function pause() { S.playing = false; clearTimeout(S.timer); $('btn-play').textContent = '▶ Play'; }
$('btn-play').onclick = () => (S.playing ? pause() : play());
$('btn-prev').onclick = () => { pause(); show(S.cur - 1); };
$('btn-next').onclick = () => { pause(); show(S.cur + 1); };
$('speed').onchange = (e) => { S.speed = +e.target.value; };
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'SELECT') return;
  if (e.key === 'ArrowRight') $('btn-next').click();
  else if (e.key === 'ArrowLeft') $('btn-prev').click();
  else if (e.key === ' ' && e.target === document.body) { e.preventDefault(); $('btn-play').click(); }
});

function summarize(meta, result) {
  const s = $('summary'); s.textContent = '';
  const title = document.createElement('b'); title.textContent = meta.title + ' — ';
  s.append(title, meta.summary);
  if (result && result.decision) s.append(` Outcome: ${result.decision.status} (${result.decision.notes}).`);
}
function selectScenario(sid) {
  pause(); if (S.es) { S.es.close(); S.es = null; }
  S.sid = sid; S.cur = -1;
  document.querySelectorAll('.scenario-btn').forEach((b) => b.classList.toggle('on', b.dataset.sid === sid));
  const meta = DATA.scenarios.find((m) => m.id === sid);
  if (!DATA.live) {
    S.frames = DATA.frames[sid] || []; summarize(meta, DATA.results[sid]); renderTrace(); play(); return;
  }
  S.frames = []; renderTrace(); summarize(meta, null); S.streaming = true;
  S.es = new EventSource(`/api/stream?scenario=${encodeURIComponent(sid)}&speed=${S.speed}`);
  S.es.addEventListener('meta', (e) => summarize(meta, JSON.parse(e.data)));
  S.es.addEventListener('frame', (e) => {
    const f = JSON.parse(e.data); S.frames.push(f); traceRow(f, S.frames.length - 1);
    if (S.playing || S.cur === S.frames.length - 2) show(S.frames.length - 1); else paintTraceCursor();
  });
  const done = () => { S.streaming = false; if (S.es) S.es.close(); S.es = null; pause(); };
  S.es.addEventListener('end', done); S.es.onerror = done;
  S.playing = true; $('btn-play').textContent = '❚❚ Pause';
}
document.querySelectorAll('.scenario-btn').forEach((b) => (b.onclick = () => selectScenario(b.dataset.sid)));
fitView(); window.addEventListener('resize', fitView);
if (DATA.scenarios.length) selectScenario(DATA.scenarios[0].id);
</script>
</body>
</html>
'''
