"""The presentation route: one page, eighteen beats, evidence that matches the research story."""
import re
import threading
import unittest
import urllib.request

from demo.present import BEATS_PER_CHAPTER, PAGE
from demo.present_data import CHAPTERS, CORPUS, RECOVERY
from demo.story_data import DOMAINS, FINAL_ROWS, summarize
from demo.visualizer import make_server


class PresentPageTests(unittest.TestCase):
    def test_eighteen_beats_in_six_chapters(self):
        self.assertEqual(sum(BEATS_PER_CHAPTER), 18)
        self.assertEqual(len(BEATS_PER_CHAPTER), len(CHAPTERS))
        ids = re.findall(r'<section class="beat(?: scene)?" id="(b\d-\d)"', PAGE)
        expected = [f'b{c}-{i}' for c, n in enumerate(BEATS_PER_CHAPTER) for i in range(1, n + 1)]
        self.assertEqual(ids, expected, 'Beats must appear in running order')

    def test_scenes_are_pinned_and_have_handlers(self):
        scenes = re.findall(r'data-scene="(\w+)"', PAGE)
        self.assertEqual(scenes, ['trace', 'gap', 'network', 'lenses', 'select', 'verifier', 'dots'])
        for scene in scenes:
            self.assertIn(f' {scene}(el,p)', PAGE, f'scene {scene} needs a scroll handler')

    def test_attribution_numbers_come_from_the_committed_evidence(self):
        for domain in DOMAINS:
            for variant in ('agentrx', 'seeds_select'):
                arm = summarize(FINAL_ROWS, domain, variant)
                self.assertIn(f'{arm["accuracy"]:.1f}%', PAGE)
                self.assertIn(f'{arm["hits"]}/{arm["n"]}', PAGE)
        self.assertNotIn('37.9%</h', PAGE, 'The written-checks figure is never a headline')

    def test_recovery_table_is_on_screen(self):
        for row in RECOVERY:
            self.assertIn(f'<i>{row["claimed"]}</i>', PAGE)
            self.assertIn(f'>{row["recovered"]}</b>', PAGE)
        self.assertEqual(PAGE.count("<i data-l="), CORPUS['generated'], 'One dot per generated episode')
        self.assertEqual(PAGE.count('data-t="f"></i>'), CORPUS['test_failed'])
        self.assertEqual(PAGE.count('data-t="h"></i>'), CORPUS['test_controls'])

    def test_self_contained_and_offline(self):
        self.assertNotRegex(PAGE, r'<(link|script)[^>]+(href|src)="https?://', 'No network assets on stage')
        self.assertEqual(PAGE.count('data-src="/widgets/system?scenario='), 2)
        self.assertNotIn('/widgets/compare', PAGE, 'The live comparison is not part of the ten minutes')
        for chapter in CHAPTERS:
            self.assertIn(chapter['speaker'], PAGE)


class PresentServerTests(unittest.TestCase):
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
            return resp.status, resp.read().decode('utf-8')

    def test_present_is_served_and_linked_from_home(self):
        status, body = self.get('/present')
        self.assertEqual(status, 200)
        self.assertIn('id="deck"', body)
        _, home = self.get('/')
        self.assertIn('href="/present"', home)
        self.assertNotIn('class="presentation"', home, 'The slide overlay is gone from the story page')

    def test_system_widget_preselects_a_requested_scenario(self):
        _, body = self.get('/widgets/system?scenario=clean_control_flow')
        self.assertIn("get('scenario')", body)


if __name__ == '__main__':
    unittest.main()
