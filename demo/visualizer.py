"""Interactive reliability studio: one self-contained HTML page plus a stdlib HTTP + SSE server.

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

from .studio_theme import BASE_CSS, navigation
from .scenarios import MILESTONES, SCENARIOS, run_scenario, scenario_meta

WORLD_W, WORLD_H = 1240, 560
NODE_W, NODE_H = 200, 82
BLOCK, GAP = 44, 10
NODES = {   # id: (x, y, title, kicker)
    'user': (30, 210, 'Customer', 'SIMULATED USER'),
    'planner': (330, 210, 'Planner', 'ORCHESTRATION'),
    'order_agent': (740, 15, 'Orders', 'ORDER OPERATIONS'),
    'product_agent': (740, 195, 'Catalog', 'PRODUCT DISCOVERY'),
    'returns_agent': (740, 375, 'Returns', 'RETURNS & EXCHANGES'),
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
    top = y + NODE_H + (58 if agent == 'planner' else 28)
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
        sub = ('Scripted customer' if node == 'user' else
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
             '<circle class="seal-ring" r="9"/><text class="seal-mark" text-anchor="middle" dy="4">↗</text></g>')
    svg = (f'<svg id="wires" width="{WORLD_W}" height="{WORLD_H}" viewBox="0 0 {WORLD_W} {WORLD_H}" '
           f'aria-hidden="true">{"".join(wires)}{token}</svg>')
    return svg, ''.join(nodes) + ''.join(blocks)


def _buttons(meta):
    names = dict(zip(SCENARIOS, ['Lost context', 'Wrong specialist', 'Unbacked claim', 'Healthy control']))
    return ''.join(f'<button class="scenario-btn" data-sid="{m["id"]}" title="{html.escape(m["summary"])}">'
                   f'<span class="case-number">0{i+1}</span><span>{names[m["id"]]}</span><span class="case-arrow">↗</span></button>'
                   for i, m in enumerate(meta))


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
                         for i, m in enumerate(['Identify constraints', 'Rank candidates', 'Replay correction', 'Verify & learn'], 1))
    return (PAGE.replace('__SVG__', svg).replace('__NODES__', nodes).replace('__BUTTONS__', _buttons(meta))
            .replace('__MILESTONES__', milestones).replace('__MODE__', 'Interactive simulation' if live else 'Offline simulation')
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
                return self._json(503, {'error': 'Model service is unavailable: ' + str(exc)})

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == '/compare':
                return self._send(200, COMPARISON_PAGE.encode('utf-8'), 'text/html; charset=utf-8')
            if url.path == '/api/models':
                try:
                    return self._json(200, comparisons.model_info())
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    return self._json(503, {'error': 'Cannot read model settings: ' + str(exc)})
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
    print(f'AgentReroute studio at http://{host}:{server.server_address[1]}/  (Ctrl+C to stop)')
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
<title>Live system · AgentReroute</title>
<style>__STUDIO_THEME__
:root{--vellum:var(--bg);--dot:#304149;--alabaster:var(--surface);--parchemin:#0d1519;--hairline:var(--border);--royal:var(--cyan);--royal-soft:#172b34;--gold:var(--violet);--gold-soft:#211e35;--carmine:var(--danger);--carmine-soft:#321e26;--emerald:var(--mint);--emerald-soft:#202d23;--ink:var(--text);--serif:var(--sans)}
.scenario-heading{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}.scenarios{display:grid;grid-template-columns:repeat(4,1fr);gap:9px}.scenario-btn{display:flex;align-items:center;gap:11px;text-align:left;border:1px solid var(--border);border-radius:10px;padding:15px;background:var(--surface);font-size:12px;transition:border-color .2s}.case-number{font:10px var(--mono);color:#667b83}.case-arrow{margin-left:auto;color:#63767e}.scenario-btn.on{border-color:#69894e;background:linear-gradient(110deg,#202c24,#111b1c);color:var(--mint)}.scenario-btn.on .case-number,.scenario-btn.on .case-arrow{color:var(--mint)}.scenario-btn:hover{border-color:#526b74}.summary{color:var(--muted);font-size:11px;line-height:1.7;margin:11px 0 21px}.summary b{color:#c4d0d3;font-weight:500}.summary-details{margin-bottom:18px}.summary-details summary{color:var(--muted);font-size:11px;padding:10px 0}.summary-details .summary{margin:0 0 10px}.workspace-grid{display:grid;grid-template-columns:minmax(0,1fr) 310px;gap:16px;align-items:stretch}.chart{background:var(--surface);border:1px solid var(--border);border-radius:14px;overflow:hidden;min-width:0}.chart-head{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:16px 18px;border-bottom:1px solid var(--border)}.chart-head h2{font-size:12px;font-weight:500;margin:0}.chart-head h2 span{display:inline-block;width:6px;height:6px;background:var(--mint);border-radius:50%;margin-right:8px}.zoombar{display:flex;gap:4px;align-items:center}.zoombar button{background:transparent;border:1px solid var(--border);border-radius:5px;min-width:27px;height:27px;font-size:10px}.zoombar button:hover{background:var(--raised)}#zoom-label{min-width:43px}.canvas{position:relative;height:410px;overflow:hidden;cursor:grab;background:radial-gradient(ellipse at 50% 50%,#15252a66,transparent 65%),radial-gradient(circle,#28383e 1px,transparent 1px),#0b1216;background-size:auto,22px 22px,auto;touch-action:none}.canvas.dragging{cursor:grabbing}#world{position:absolute;left:0;top:0;width:__WORLD_W__px;height:__WORLD_H__px;transform-origin:0 0}#wires{position:absolute;left:0;top:0;overflow:visible;pointer-events:none}.wire{fill:none;stroke:#344851;stroke-width:2.2}.wire.tether{stroke-dasharray:3 6;stroke-width:1.3}.wire-active,.wire-gold,.wire-fault,.wire-recovered{stroke-width:3;stroke-dasharray:9 7;animation:march .9s linear infinite}.wire-active{stroke:var(--cyan)}.wire-gold{stroke:var(--violet)}.wire-fault{stroke:var(--danger)}.wire-recovered{stroke:var(--mint)}@keyframes march{to{stroke-dashoffset:-32}}#flow-token .seal{fill:#21382a;stroke:var(--mint);stroke-width:2}#flow-token .seal-ring{fill:none;stroke:var(--mint);stroke-width:.6}#flow-token .seal-mark{fill:var(--mint);font-size:13px}#flow-token.fault .seal{fill:#54252e;stroke:var(--danger)}#flow-token.recovered .seal{fill:#21382a}.node{position:absolute;background:linear-gradient(145deg,#1c2c33,#101a20);border:1px solid #40565f;border-radius:13px;padding:15px 17px;box-shadow:0 10px 24px #0004;transition:border-color .2s,background .2s}.node-trigger{border-style:dashed}.node-kicker{font:9px var(--mono);letter-spacing:1px;color:var(--muted)}.node-title{font-size:21px;letter-spacing:-.6px;font-weight:550;color:var(--text);margin-top:5px}.node-sub{font-size:10px;color:var(--muted);margin-top:4px}.node-state{position:absolute;right:12px;top:-9px;font:9px var(--mono)}.node-state::after{background:#0d171c;border:1px solid currentColor;border-radius:5px;padding:3px 6px}.node.active{border-color:var(--cyan);box-shadow:0 0 24px #7fdbed18}.node.active .node-state::after{content:'ACTIVE';color:var(--cyan)}.node.fault,.node.blamed{border-color:var(--danger);background:#2c1c25;box-shadow:0 0 22px #ff9b9b15}.node.fault .node-state::after{content:'FAULT';color:var(--danger)}.node.blamed{border-style:dashed}.node.blamed .node-state::after{content:'IDENTIFIED';color:var(--danger)}.node.recovered{border-color:var(--mint);background:linear-gradient(145deg,#23372b,#12251f);box-shadow:0 0 22px #bcf78312}.node.recovered .node-state::after{content:'RECOVERED';color:var(--mint)}.port{position:absolute;width:9px;height:9px;border-radius:50%;background:#112028;border:2px solid #7c939b}.port-in{left:-5px;top:calc(50% - 5px)}.port-out{right:-5px;top:calc(50% - 5px)}.port-tool{left:calc(50% - 5px);bottom:-5px}.tool-miniblock{position:absolute;width:44px;height:44px;border-radius:9px;background:#151e2b;border:1px solid #4c4967;display:flex;align-items:center;justify-content:center;z-index:2;cursor:help}.mono-gram{font:10px var(--mono);color:#bdacd9}.tool-miniblock.used{background:#252a37}.tool-miniblock.active{z-index:6;transform:translateY(-3px);border-color:var(--violet);background:#312844}.tool-miniblock.active .mono-gram{color:#e6dfff}.tool-miniblock.error{border-color:var(--danger);background:#432330}.tool-popper{position:absolute;bottom:calc(100% + 9px);left:50%;transform:translateX(-50%);white-space:nowrap;background:#202b32;border:1px solid #665b83;border-radius:7px;padding:7px 10px;font:11px var(--mono);color:var(--text);opacity:0;pointer-events:none;box-shadow:0 8px 20px #0008}.tool-miniblock:hover,.tool-miniblock:focus{z-index:7}.tool-miniblock:hover .tool-popper,.tool-miniblock:focus .tool-popper,.tool-miniblock.active .tool-popper{opacity:1}.tool-miniblock.error .tool-popper{border-color:var(--danger)}.canvas-legend{display:flex;gap:18px;padding:0 18px 14px;background:#0b1216;font:9px var(--mono);color:var(--muted)}.canvas-legend span:before{content:'';display:inline-block;width:5px;height:5px;border-radius:50%;background:var(--cyan);margin-right:6px}.canvas-legend span:nth-child(2):before{background:var(--danger)}.canvas-legend span:nth-child(3):before{background:var(--mint)}.canvas-legend small{margin-left:auto;font-size:9px;color:#6b838b}.playback-bar{display:flex;align-items:center;gap:7px;padding:13px 16px;border-top:1px solid var(--border)}.playback-bar .ctl{padding:7px 10px;font-size:10px}.playback-bar input{min-width:40px;flex:1;width:100%;margin:0 10px}.frame-count{font:9px var(--mono);color:var(--muted);white-space:nowrap}.progress{padding:17px 18px;background:#101b20;border-top:1px solid var(--border)}.progress-top{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.progress-top .label{font-size:11px;color:var(--text)}.badge{display:inline-block;font:9px var(--mono);border:1px solid currentColor;border-radius:5px;padding:4px 7px;white-space:nowrap}.b-normal{color:var(--cyan);background:#152a33}.b-fault{color:var(--danger);background:#321e26}.b-debug{color:var(--violet);background:#272036}.b-recovered{color:var(--mint);background:#203023}#pct{margin-left:auto;font:12px var(--mono);color:var(--mint)}#dbg-text{display:none}.bar{height:3px;border-radius:5px;background:#273a40;margin:14px 0;overflow:hidden}#bar-fill{height:100%;width:0;background:linear-gradient(90deg,var(--cyan),var(--mint));transition:width .4s}.milestones{list-style:none;margin:0;padding:0;display:grid;grid-template-columns:repeat(4,1fr);gap:7px}.milestone{font-size:9px;color:var(--muted);display:flex;gap:5px;align-items:center;line-height:1.5}.ms-dot{width:5px;height:5px;flex:none;border-radius:50%;background:#48606a}.milestone.done,.milestone.current{color:var(--text)}.milestone.done .ms-dot{background:var(--mint)}.milestone.current .ms-dot{background:var(--cyan);box-shadow:0 0 10px #7fdbed55}.panel{background:var(--surface);border:1px solid var(--border);border-radius:14px;min-width:0;overflow:hidden}.panel h3{font:10px var(--mono);letter-spacing:1px;text-transform:uppercase;font-weight:400;padding:18px;margin:0;border-bottom:1px solid var(--border);color:var(--muted)}.panel-body{padding:20px 18px}#step-title{font-size:22px;line-height:1.35;letter-spacing:-.6px;font-weight:500;margin:0 0 17px}.chips{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:19px}.chip{font:9px var(--mono);color:var(--muted);background:var(--raised);padding:5px 7px;border-radius:5px}#step-narrative{font-size:12px;line-height:1.8;margin:0;color:#c1cfd2;white-space:pre-wrap;overflow-wrap:anywhere}#step-fault{display:none;font-size:11px;color:var(--danger);background:#2b1e25;border:1px solid #61343f;border-radius:8px;padding:10px;margin-bottom:15px;line-height:1.7}.payload-details{margin-top:22px;border-top:1px solid var(--border)}.payload-details summary{font-size:11px;color:var(--muted);padding:15px 0}.payload-details pre{max-height:320px;overflow:auto;margin:0;padding:12px;background:#0a1318;border:1px solid var(--border);border-radius:8px;font-size:10px}.inspector-footer{margin-top:auto;padding:16px 18px;border-top:1px solid var(--border);font-size:10px;color:var(--muted);line-height:1.7}#current-step{display:flex;flex-direction:column}.dock{margin-top:16px}.dock summary{padding:17px 19px;font-size:12px}.dock summary span{color:var(--muted);font-size:10px;margin-left:12px}#trace{list-style:none;margin:0;padding:0 12px 12px;max-height:390px;overflow:auto;position:relative}.trace-row{width:100%;display:grid;grid-template-columns:35px 95px minmax(0,1fr);gap:8px;align-items:center;text-align:left;border:1px solid transparent;border-radius:7px;background:none;padding:9px 8px}.trace-row:hover{background:var(--raised)}.trace-row.current{border-color:#376476;background:#172b34}.trace-row.future{opacity:.45}.trace-row .seq{font:10px var(--mono);color:var(--muted)}.trace-row .what{font-size:11px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted)}.trace-row .what b{font-weight:500;color:var(--text)}.phase-sep{font:9px var(--mono);letter-spacing:.7px;color:var(--violet);padding:14px 8px 8px;text-transform:uppercase}
@media(max-width:1150px){.workspace-grid{grid-template-columns:minmax(0,1fr) 270px}.canvas{height:390px}.milestones{grid-template-columns:repeat(2,1fr)}}
@media(max-width:850px){.workspace-grid{grid-template-columns:1fr}.scenarios{grid-template-columns:repeat(2,1fr)}.canvas{height:410px}#current-step .panel-body{padding:18px}.inspector-footer{display:none}.milestones{grid-template-columns:repeat(4,1fr)}}
@media(max-width:540px){.scenario-btn{font-size:11px;padding:12px;gap:8px}.canvas{height:300px}.canvas-legend{gap:12px;padding:0 12px 13px}.canvas-legend small{display:none}.chart-head{padding:13px}.playback-bar{padding:12px;gap:4px;flex-wrap:wrap}.playback-bar input{order:5;flex-basis:70%;margin:6px 0}.frame-count{margin-left:auto}.milestones{grid-template-columns:repeat(2,1fr)}.progress{padding:14px}.progress-top{gap:8px}.progress-top .label{font-size:10px}.trace-row{grid-template-columns:26px 80px minmax(0,1fr)}.summary-details .summary{font-size:10px}}
</style>
</head>
<body><main class="studio-shell">
__NAV__
<header class="studio-hero"><div><div class="eyebrow">Agent reliability, made visible</div><h1>Every failure.<br><em>A path forward.</em></h1><p>Follow the handoff. Find the fault. Watch a correction take shape.<br>An inside look at how AgentReroute gets an agent back on track.</p></div><div class="hero-aside"><div><strong>4 agents</strong>One connected system</div><i></i><div><strong>3 perspectives</strong>One recovery loop</div></div></header>
<div class="scenario-heading"><span class="eyebrow-label">Explore a scenario</span><span class="pill" id="mode">__MODE__</span></div>
<nav class="scenarios" aria-label="Scenarios">__BUTTONS__</nav>
<details class="summary-details"><summary>What happens in this scenario</summary><p class="summary" id="summary"></p></details>
<div class="workspace-grid">
<section class="chart" aria-label="Workflow canvas"><div class="chart-head"><h2><span></span>Agent network</h2><div class="zoombar"><button id="zoom-out" aria-label="Zoom out">−</button><button id="zoom-label" aria-label="Reset zoom to 100%">100%</button><button id="zoom-in" aria-label="Zoom in">+</button><button id="zoom-fit" aria-label="Fit view">Fit View</button></div></div>
<div class="canvas" id="canvas"><div id="world">__SVG____NODES__</div></div><div class="canvas-legend"><span>Active</span><span>Fault</span><span>Recovered</span><small>Drag to explore · Scroll to zoom</small></div>
<div class="playback-bar"><button class="ctl" id="btn-prev" aria-label="Previous step">←</button><button class="ctl primary" id="btn-play" aria-label="Play or pause">▶ Play</button><button class="ctl" id="btn-next" aria-label="Next step">→</button><input id="execution-cursor" type="range" min="0" max="0" value="0" aria-label="Execution frame"><span class="frame-count" id="frame-count">0 / 0</span><select class="ctl" id="speed" aria-label="Playback speed"><option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="2">2x</option></select></div>
<div class="progress" id="debugger"><div class="progress-top"><span class="label">Recovery loop</span><span class="badge b-normal" id="dbg-badge">Idle</span><span id="dbg-text"></span><span id="pct">0%</span></div><div class="bar"><div id="bar-fill"></div></div><ol class="milestones">__MILESTONES__</ol></div></section>
<aside class="panel" id="current-step"><h3>Inside this step</h3><div class="panel-body"><p id="step-title">Preparing the system</p><div class="chips"><span class="badge b-normal" id="step-status">NORMAL</span><span class="chip" id="step-actor">actor</span><span class="chip" id="step-phase">phase</span><span class="chip" id="step-index">step</span></div><div id="step-fault"></div><p id="step-narrative"></p><details class="payload-details"><summary>Inspect step data</summary><pre id="step-payload">{}</pre></details></div><div class="inspector-footer">Select a frame to inspect the decision, tool call or recovery action at that moment.</div></aside>
</div>
<section class="dock"><details class="panel" id="live-trace"><summary>Execution history <span>Every step, in order</span></summary><ol id="trace"></ol></details></section>
<footer class="studio-footer"><span>Scripted simulation · Real recovery policy · Fault labels are visible only to you.</span><span>Observe → Attribute → Recover → Verify</span></footer>
</main>
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
  canvas.style.backgroundSize = `auto, ${22 * S.scale}px ${22 * S.scale}px, auto`;
  canvas.style.backgroundPosition = `center, ${S.pan.x}px ${S.pan.y}px, center`;
  $('zoom-label').textContent = Math.round(S.scale * 100) + '%';
}
function zoomAt(next, cx, cy) {
  next = Math.min(1.65, Math.max(0.2, next));
  S.pan.x = cx - (cx - S.pan.x) * next / S.scale;
  S.pan.y = cy - (cy - S.pan.y) * next / S.scale;
  S.scale = next; applyView();
}
function fitView() {
  const r = canvas.getBoundingClientRect();
  S.scale = Math.min(1.65, Math.max(0.2, Math.min(r.width / (WORLD.w + 40), r.height / (WORLD.h + 40))));
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
  const len = path.getTotalLength(), t0 = performance.now(), dur = window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 1 : 700 / S.speed;
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
  $('execution-cursor').max = Math.max(0,S.frames.length-1);
  $('execution-cursor').value = i;
  $('frame-count').textContent = `${i+1} / ${S.frames.length}`;
}
function delay(f) { return (f && f.kind === 'debug' ? 2200 : 1400) / S.speed; }
function tick() {
  S.timer = null;
  if (!S.playing) return;
  if (S.cur >= S.frames.length - 1) { if (!S.streaming) pause(); return; }
  show(S.cur + 1);
  S.timer = setTimeout(tick, delay(S.frames[S.cur]));
}
function play() {
  if (S.cur >= S.frames.length - 1 && !S.streaming) S.cur = -1;
  S.playing = true; $('btn-play').textContent = '❚❚ Pause';
  clearTimeout(S.timer); S.timer = null; tick();
}
function pause() { S.playing = false; clearTimeout(S.timer); S.timer = null; $('btn-play').textContent = '▶ Play'; }
$('btn-play').onclick = () => (S.playing ? pause() : play());
$('btn-prev').onclick = () => { pause(); show(S.cur - 1); };
$('btn-next').onclick = () => { pause(); show(S.cur + 1); };
$('execution-cursor').oninput = e => { pause(); show(Number(e.target.value)); };
$('speed').onchange = (e) => { S.speed = +e.target.value; };
document.addEventListener('keydown', (e) => {
  if (['SELECT','INPUT','TEXTAREA','BUTTON','SUMMARY'].includes(e.target.tagName)) return;
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
    const following = S.playing && S.cur === S.frames.length - 1 && S.timer === null;
    const f = JSON.parse(e.data); S.frames.push(f); traceRow(f, S.frames.length - 1);
    $('execution-cursor').max = Math.max(0,S.frames.length-1);
    if (following) show(S.frames.length - 1); else { paintTraceCursor(); $('frame-count').textContent=`${S.cur+1} / ${S.frames.length}`; }
  });
  const done = () => {
    S.streaming = false; if (S.es) S.es.close(); S.es = null;
    if (S.cur >= S.frames.length - 1) pause();
    else if (S.playing && S.timer === null) tick();
  };
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

PAGE = PAGE.replace('__STUDIO_THEME__', BASE_CSS).replace('__NAV__', navigation('system'))
