import json
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from agentrx.rl_cross_branch import MPRLRecoveryPolicy
from benchmark.env.workflow import AGENTS
from demo import scenarios
from demo.visualizer import build_demo_html, build_html, make_server

ROOT = Path(__file__).resolve().parents[2]
IDS = ['handoff_drop_context', 'misroute_specialist', 'skip_write_claim', 'clean_control_flow']
OBSERVABLE_VIEW = {'first_user_message', 'policy', 'agents', 'tools'}


class GoldGuardedPolicy(MPRLRecoveryPolicy):
    """Fails the run if the policy is handed anything beyond the observable trace and task view."""
    seen = []

    def recover(self, env):
        dumped = json.dumps([env.trajectory_id, env.trace, env.task_view])
        assert set(env.task_view) == OBSERVABLE_VIEW, env.task_view.keys()
        assert not env.hypotheses, 'oracle hypotheses were supplied'
        assert '"faults"' not in dumped and '"actions"' not in dumped and '"outputs"' not in dumped
        for sid, spec in scenarios.SCENARIOS.items():
            assert sid not in dumped, 'scenario id leaked into what the policy sees'
            assert not spec['operator'] or spec['operator'][0].name not in dumped, 'fault operator leaked'
        type(self).seen.append(env.trajectory_id)
        return super().recover(env)


class ScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        GoldGuardedPolicy.seen = []
        with mock.patch.object(scenarios, 'MPRLRecoveryPolicy', GoldGuardedPolicy):
            cls.results = {s['id']: s for s in scenarios.all_scenarios()}

    def test_runs_all_four_through_the_guarded_policy(self):
        self.assertEqual(list(self.results), IDS)
        self.assertEqual(GoldGuardedPolicy.seen, [f'demo-case-{i}' for i in range(1, 5)])

    def test_frames_are_ordered_and_follow_the_phase_sequence(self):
        order = ['initial_run', 'debugging', 'recovery_replay', 'verified']
        for sid, result in self.results.items():
            frames = result['frames']
            self.assertEqual([f['seq'] for f in frames], list(range(1, len(frames) + 1)), sid)
            ranks = [order.index(f['phase']) for f in frames]
            self.assertEqual(ranks, sorted(ranks), sid)
            debug = [(f['debug_pct'], f['active_phase']) for f in frames if f['phase'] in ('debugging', 'verified')]
            self.assertEqual(debug, [(25, 1), (50, 2), (75, 3), (100, 4)], sid)
            self.assertEqual(frames[-1]['phase'], 'verified')
            self.assertTrue(frames[-1]['verified'], sid)

    def test_tool_frames_carry_popper_payloads(self):
        tools = [f['tool'] for r in self.results.values() for f in r['frames'] if f['kind'] == 'tool']
        self.assertTrue(tools)
        for t in tools:
            self.assertIn(t['name'], AGENTS[t['agent']]['tools'])
            self.assertTrue(t['popper_text'].startswith(t['name'] + '('))
            self.assertIsInstance(t['error'], bool)
        first = self.results['clean_control_flow']['frames'][1]['tool']
        self.assertEqual((first['agent'], first['popper_text']),
                         ('planner', "find_user_id_by_email('mia.garcia2723@example.com')"))

    def test_handoff_drop_context_blames_and_repairs_the_planner_handoff(self):
        r = self.results['handoff_drop_context']
        fault = next(f for f in r['frames'] if f['status'] == 'FAULT')
        self.assertEqual((fault['kind'], fault['handoff']['from'], fault['handoff']['to']),
                         ('handoff', 'planner', 'order_agent'))
        self.assertNotIn('order_id', fault['handoff']['context'])
        top = r['hypotheses'][0]
        self.assertEqual((top['agent'], top['step_index'], top['mode']), ('planner', fault['step_index'],
                                                                        'context_propagation'))
        self._assert_recovered(r, rollback=fault['step_index'], kind='guidance')
        repaired = next(f for f in r['frames'] if f['phase'] == 'recovery_replay')
        self.assertEqual(repaired['handoff']['context']['order_id'], '#W1000001')

    def test_misroute_is_rerouted_to_order_agent(self):
        r = self.results['misroute_specialist']
        fault = next(f for f in r['frames'] if f['status'] == 'FAULT')
        self.assertEqual(fault['handoff']['to'], 'product_agent')
        top = r['hypotheses'][0]
        self.assertEqual((top['agent'], top['mode']), ('planner', 'drift'))
        self._assert_recovered(r, rollback=fault['step_index'])
        repaired = next(f for f in r['frames'] if f['phase'] == 'recovery_replay')
        self.assertEqual(repaired['handoff']['to'], 'order_agent')

    def test_skip_write_forces_the_write_tool(self):
        r = self.results['skip_write_claim']
        fault = next(f for f in r['frames'] if f['status'] == 'FAULT')
        self.assertEqual((fault['kind'], fault['actor']), ('reply', 'returns_agent'))
        top = r['hypotheses'][0]
        self.assertEqual((top['agent'], top['step_index'], top['mode']),
                         ('returns_agent', fault['step_index'], 'instruction_adherence'))
        self._assert_recovered(r, rollback=fault['step_index'])
        writes = [f['tool']['name'] for f in r['frames'] if f['phase'] == 'recovery_replay' and f['tool']]
        self.assertIn('return_delivered_order_items', writes)

    def test_clean_run_abstains(self):
        r = self.results['clean_control_flow']
        self.assertEqual(r['decision']['status'], 'abstained')
        self.assertEqual((r['hypotheses'], r['rl']), ([], None))
        self.assertFalse(any(f['status'] == 'FAULT' for f in r['frames']))
        self.assertFalse(any(f['phase'] == 'recovery_replay' for f in r['frames']))
        self.assertEqual(r['frames'][-1]['title'], 'IV. Clean Run — Abstained')
        checks = next(f for f in r['frames'] if f['active_phase'] == 1)['payload']
        self.assertEqual(checks['violations'], 0)
        kinds = [f['kind'] for f in r['frames'] if f['phase'] == 'initial_run']
        self.assertEqual(kinds[:6], ['dialog', 'tool', 'reply', 'dialog', 'handoff', 'tool'])

    def _assert_recovered(self, r, rollback, kind=None):
        self.assertEqual(r['decision']['status'], 'intervened')
        self.assertEqual((r['rl']['verified'], r['rl']['reward']), (True, 1.0))
        m3 = next(f for f in r['frames'] if f['active_phase'] == 3 and f['phase'] == 'debugging')
        self.assertEqual(m3['payload']['rollback_step'], rollback)
        if kind:
            self.assertEqual(m3['payload']['directive_kind'], kind)
        replay = [f for f in r['frames'] if f['phase'] == 'recovery_replay']
        self.assertEqual(replay[0]['step_index'], rollback)
        self.assertTrue(replay[0]['directive_applied'])
        self.assertTrue(all(f['status'] == 'RECOVERED' for f in replay))
        m1 = next(f for f in r['frames'] if f['active_phase'] == 1)['payload']
        self.assertGreater(m1['violations'], 0)
        self.assertEqual(set(m1['checks_fired']), {'P1_INTENT_PLAN', 'P2_TOOL_EXECUTION', 'P3_HANDOFF_GROUNDING'})


class HtmlExportTests(unittest.TestCase):
    def test_build_demo_html_is_self_contained_and_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = build_demo_html(Path(tmp)/'sub'/'demo.html')
            page = path.read_text(encoding='utf-8')
        self.assertTrue(page.startswith('<!doctype html>'))
        for needle in ('id="zoom-in"', 'id="zoom-out"', 'id="zoom-fit"', 'Fit View', 'id="flow-token"',
                       'id="btn-play"', 'id="btn-prev"', 'id="btn-next"', '0.5x', '2x',
                       'id="current-step"', 'id="live-trace"', 'Execution history', 'id="execution-cursor"', 'id="bar-fill"',
                       'wire-active', 'wire-gold', 'wire-fault', 'wire-recovered'):
            self.assertIn(needle, page)
        for i, m in enumerate(['Identify constraints', 'Rank candidates', 'Replay correction', 'Verify &amp; learn'], 1):
            self.assertIn(f'id="ms-{i}"', page)
            self.assertIn(m, page)
        for agent in ('planner', 'order_agent', 'product_agent', 'returns_agent'):
            self.assertIn(f'id="node-{agent}"', page)
            for tool in AGENTS[agent]['tools']:
                self.assertIn(f'id="tb-{agent}-{tool}"', page)
                self.assertIn(f'id="tether-{agent}-{tool}"', page)
        blocks = sum(len(AGENTS[a]['tools']) for a in AGENTS)
        self.assertEqual(page.count('class="tool-miniblock"'), blocks)
        self.assertEqual(page.count('class="tool-popper"'), blocks)
        for sid in IDS:
            self.assertIn(f'data-sid="{sid}"', page)
        self.assertNotRegex(page, r'<(script|link)[^>]+(src|href)="https?://')   # no CDN

    def test_live_page_embeds_metadata_only(self):
        page = build_html(live=True)
        self.assertIn('"live": true', page)
        self.assertIn('"frames": {}', page)


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server(port=0, delay=0)
        cls.base = f'http://127.0.0.1:{cls.server.server_address[1]}'
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=30) as resp:
            return resp.status, resp.headers.get('Content-Type'), resp.read().decode('utf-8')

    def test_index_and_scenarios(self):
        status, ctype, body = self.get('/')
        self.assertEqual((status, ctype), (200, 'text/html; charset=utf-8'))
        self.assertIn('Interactive simulation', body)
        status, ctype, body = self.get('/api/scenarios')
        self.assertEqual((status, ctype), (200, 'application/json'))
        self.assertEqual([m['id'] for m in json.loads(body)], IDS)

    def test_stream_emits_sse_frames(self):
        status, ctype, body = self.get('/api/stream?scenario=handoff_drop_context')
        self.assertEqual((status, ctype), (200, 'text/event-stream'))
        events = []
        for block in body.strip().split('\n\n'):
            lines = dict(line.split(': ', 1) for line in block.splitlines())
            events.append((lines['event'], json.loads(lines['data'])))
        self.assertEqual(events[0][0], 'meta')
        self.assertEqual(events[-1], ('end', {'frames': len(events) - 2}))
        frames = [d for e, d in events if e == 'frame']
        self.assertEqual([f['seq'] for f in frames], list(range(1, len(frames) + 1)))
        self.assertEqual(frames[-1]['debug_pct'], 100)
        self.assertTrue(any(f['phase'] == 'recovery_replay' for f in frames))

    def test_unknown_scenario_is_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.get('/api/stream?scenario=nope')
        self.assertEqual(ctx.exception.code, 404)


class CliTests(unittest.TestCase):
    def test_python_m_demo_out_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'demo.html'
            proc = subprocess.run([sys.executable, '-m', 'demo', '--out', str(out)], cwd=ROOT,
                                  capture_output=True, text=True, timeout=300)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(out.exists() and out.stat().st_size > 10000)


if __name__ == '__main__':
    unittest.main()
