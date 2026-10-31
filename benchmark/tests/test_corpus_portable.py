"""A built corpus must resolve its generation directories on a machine other than the one that built it.

The corpus is a research artifact: it gets copied between machines and handed to collaborators, and the
recovery harness reads the original episode files through paths the build recorded. Those paths used to
be absolute, so a corpus built anywhere but the exact checkout that would consume it failed at run time
with a FileNotFoundError naming a directory that never existed on that machine. Nothing at build time
warned about it.

The property under test is that the corpus and its generation directories can move together.
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from benchmark.corpus import generation_dir

from .helpers import ROOT, corpus


class CorpusPortabilityTest(unittest.TestCase):
    def setUp(self):
        self.corpus = corpus()
        self.index = json.loads((self.corpus / 'gold' / 'episodes.json').read_text())
        self.assertTrue(self.index, 'fixture corpus has no episodes to check')

    def test_generation_paths_are_stored_relative(self):
        for ref in self.index.values():
            stored = Path(ref['generation_dir'])
            self.assertFalse(stored.is_absolute(), f'{stored} is absolute, so the corpus cannot move')
            self.assertNotIn('\\', ref['generation_dir'], 'separators must be POSIX to cross platforms')

    def test_each_reference_resolves_to_a_readable_episode(self):
        for tid, ref in self.index.items():
            gen = generation_dir(ref, self.corpus)
            episode = gen / 'episodes' / f'{ref["episode_id"]}.json'
            self.assertTrue(episode.exists(), f'{tid} -> {episode} missing')
            self.assertTrue((gen / 'generation.json').exists(), f'{tid} -> no generation.json beside it')

    def test_the_corpus_resolves_after_being_copied_elsewhere(self):
        """The real portability check: move the tree, and the references must still land."""
        with tempfile.TemporaryDirectory() as d:
            copied_root = Path(d) / 'moved'
            shutil.copytree(ROOT, copied_root)
            moved = copied_root / self.corpus.relative_to(ROOT)
            index = json.loads((moved / 'gold' / 'episodes.json').read_text())
            for tid, ref in index.items():
                gen = generation_dir(ref, moved)
                self.assertTrue(gen.is_relative_to(copied_root),
                                f'{tid} escaped the copied tree to {gen}')
                self.assertTrue((gen / 'episodes' / f'{ref["episode_id"]}.json').exists(),
                                f'{tid} unreadable after the move')

    def test_an_absolute_path_from_an_older_corpus_still_works(self):
        """Corpora built before the change hold absolute paths; they must keep resolving."""
        ref = dict(next(iter(self.index.values())))
        absolute = generation_dir(ref, self.corpus).resolve()
        ref['generation_dir'] = str(absolute)
        self.assertEqual(generation_dir(ref, Path('/nonexistent/corpus')), absolute)

    def test_a_bare_string_resolves_like_an_index_row(self):
        ref = next(iter(self.index.values()))
        self.assertEqual(generation_dir(ref['generation_dir'], self.corpus),
                         generation_dir(ref, self.corpus))


if __name__ == '__main__':
    unittest.main()
