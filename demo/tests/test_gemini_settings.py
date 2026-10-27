"""Gemini UI configuration and jobs; all credentials and transports are fixtures."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentrx.io import write_json
from agentrx.providers import HTTPProvider
from demo import model_settings
from demo.comparison import ComparisonService
from demo.tests.test_comparison import TransportFixture, wait_done, write_fixture_corpus
from tests.test_providers import FakeOpener, completion


class GeminiSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env_file = self.root / '.env'
        self.corpus = write_fixture_corpus(self.root/'corpus')
        self.env_patch = patch.dict('os.environ', {}, clear=True)
        self.path_patch = patch.object(model_settings, 'ENV_FILE', self.env_file)
        self.env_patch.start()
        self.path_patch.start()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.env_patch.stop)
        self.addCleanup(self.path_patch.stop)

    def test_missing_key_blocks_creation_before_any_job_or_output_exists(self):
        service = ComparisonService(self.root / 'runs', corpus=self.corpus)
        self.addCleanup(service.close)
        info = service.model_info()
        self.assertEqual(info['provider'], 'gemini')
        self.assertFalse(info['configured'])
        self.assertIn('GEMINI_API_KEY', info['message'])
        with self.assertRaisesRegex(ValueError, 'GEMINI_API_KEY'):
            service.start('tau:fixture-1', info['default_model'])
        self.assertEqual(service.jobs, {})
        self.assertFalse((self.root / 'runs').exists())

    def test_key_added_after_service_creation_is_available_without_restart(self):
        service = ComparisonService(self.root / 'runs', corpus=self.corpus)
        self.addCleanup(service.close)
        self.assertFalse(service.model_info()['configured'])
        self.env_file.write_text('# local key\nGEMINI_API_KEY="first-fixture-key"\n')
        info = service.model_info()
        self.assertTrue(info['configured'])
        self.assertEqual(info['models'], ['gemini-3.1-flash-lite'])
        self.assertEqual(info['default_model'], 'gemini-3.1-flash-lite')
        first = service.provider_factory(info['default_model'])
        self.env_file.write_text('export GEMINI_API_KEY=second-fixture-key # updated\n')
        second = service.provider_factory(info['default_model'])
        self.assertEqual(first.api_key, 'first-fixture-key')
        self.assertEqual(second.api_key, 'second-fixture-key')
        self.assertNotIn('fixture-key', json.dumps(info))
        self.assertIsInstance(second, HTTPProvider)
        self.assertEqual(second.identity['provider'], 'gemini')
        self.assertEqual(second.settings, {'temperature': 0, 'max_tokens': 16000})
        self.assertEqual(second.tpm_limit, 200000)
        self.assertEqual(second.url,
                         'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions')

    def test_process_environment_wins_and_explicit_blank_disables_dotenv_key(self):
        self.env_file.write_text('GEMINI_API_KEY=file-fixture-key\n')
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'process-fixture-key'}):
            provider = model_settings.gemini_provider('gemini-3.1-flash-lite')
            self.assertEqual(provider.api_key, 'process-fixture-key')
        with patch.dict('os.environ', {'GEMINI_API_KEY': ''}):
            self.assertFalse(model_settings.model_info()['configured'])

    def test_dotenv_values_are_not_shell_code_or_endpoint_configuration(self):
        marker = self.root / 'must-not-exist'
        self.env_file.write_text(f'GEMINI_API_KEY="$(touch {marker})"\n'
                                 'GEMINI_BASE_URL=http://attacker.example\n')
        provider = model_settings.gemini_provider('gemini-3.1-flash-lite')
        self.assertEqual(provider.api_key, f'$(touch {marker})')
        self.assertFalse(marker.exists())
        self.assertTrue(provider.url.startswith('https://generativelanguage.googleapis.com/'))

    def test_unsupported_model_is_rejected_before_starting_a_job(self):
        self.env_file.write_text('GEMINI_API_KEY=fixture-key\n')
        service = ComparisonService(self.root / 'runs', corpus=self.corpus)
        self.addCleanup(service.close)
        with self.assertRaisesRegex(ValueError, 'available|supported'):
            service.start('tau:fixture-1', 'unapproved-paid-model')
        self.assertEqual(service.jobs, {})

    def test_invalid_environment_credential_is_not_published_in_setup_errors(self):
        service = ComparisonService(self.root / 'runs', corpus=self.corpus)
        self.addCleanup(service.close)
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'fixture-private\ninvalid-header'}):
            info = service.model_info()
            self.assertFalse(info['configured'])
            self.assertNotIn('fixture-private', json.dumps(info))
            with self.assertRaises(ValueError) as raised:
                service.start('tau:fixture-1', info['default_model'])
            self.assertNotIn('fixture-private', str(raised.exception))

    def test_valid_default_job_uses_http_attribution_and_persists_no_key(self):
        self.env_file.write_text('GEMINI_API_KEY=private-fixture-credential\n')
        # Constant valid JSON drives the real pipelines; only HTTP is replaced.
        reply = {'constraints': [], 'step_index': 4, 'agent': 'assistant',
                 'category': 'MisinterpretationOfToolOutput', 'rationale': 'Missing order identifier.',
                 'evidence_refs': [{'step_index': 4, 'substep_index': 1}],
                 'action': 'reply', 'content': 'I cannot complete this request.'}
        opener = FakeOpener([completion(json.dumps(reply)) for _ in range(100)])
        service = ComparisonService(self.root / 'runs', corpus=self.corpus)
        self.addCleanup(service.close)
        with patch('agentrx.providers.request.build_opener', return_value=opener):
            job = service.start('tau:fixture-1', 'gemini-3.1-flash-lite')
            saved = wait_done(service, job['id'])
        self.assertEqual(saved['status'], 'completed')
        self.assertEqual(saved['provider'], 'gemini')
        self.assertEqual(set(saved['results']), {'agentrx', 'agentreroute'})
        self.assertTrue(any(e.get('stage') == 'mp_rl_select' for e in saved['events']))
        self.assertFalse(any(e.get('stage', '').startswith('replay_agent/') for e in saved['events']))
        for result in saved['results'].values():
            self.assertNotEqual(result['status'], 'error', result.get('error'))
            self.assertEqual(result['provider']['provider'], 'gemini')
            self.assertGreater(result['metrics']['completed_calls'], 0)
        for path in (self.root / 'runs').rglob('*.json'):
            self.assertNotIn('private-fixture-credential', path.read_text(), str(path))

    def test_injected_test_transport_retains_offline_catalog_behavior(self):
        service = ComparisonService(self.root / 'runs', provider_factory=lambda _: TransportFixture(),
                                    model_catalog=lambda: ['fixture-model'])
        self.addCleanup(service.close)
        info = service.model_info()
        self.assertEqual(info['models'], ['fixture-model'])
        self.assertEqual(info['provider'], 'test')
        self.assertTrue(info['configured'])

    def test_saved_ollama_run_keeps_its_original_provider_identity(self):
        job_id = 'a' * 32
        write_json(self.root / 'runs' / job_id / 'comparison.json',
                   {'id': job_id, 'scenario': 'tau:fixture-1', 'model': 'qwen3:8b',
                    'status': 'completed', 'events': [],
                    'results': {'agentrx': {'provider': {'provider': 'ollama', 'model': 'qwen3:8b'}}}})
        service = ComparisonService(self.root / 'runs', corpus=self.corpus)
        self.addCleanup(service.close)
        saved = service.snapshot(job_id)
        self.assertEqual(saved['provider'], 'ollama')
        self.assertEqual(saved['results']['agentrx']['provider']['model'], 'qwen3:8b')

    def test_demo_pacing_is_configurable_and_not_the_research_project_quota(self):
        self.env_file.write_text('GEMINI_API_KEY=fixture-key\n')
        with patch.dict('os.environ', {'AGENTRX_DEMO_TPM_LIMIT': '50000'}):
            self.assertEqual(model_settings.gemini_provider(model_settings.DEFAULT_MODEL).tpm_limit, 50000)
        for value in ('0', '-1', 'invalid'):
            with patch.dict('os.environ', {'AGENTRX_DEMO_TPM_LIMIT': value}):
                with self.assertRaisesRegex(ValueError, 'positive integer'):
                    model_settings.gemini_provider(model_settings.DEFAULT_MODEL)
