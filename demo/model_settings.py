"""Gemini settings for the demo only; credentials stay in the server process."""
import os
import shlex
from pathlib import Path

from agentrx.providers import HTTPProvider


ENV_FILE = Path(__file__).resolve().parents[1] / '.env'
DEFAULT_MODEL = 'gemini-3.1-flash-lite'
MODELS = (DEFAULT_MODEL,)
BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/openai'
MISSING_KEY = 'Add GEMINI_API_KEY to the repository .env or process environment, then refresh this page.'


def _validate_key(key):
    if any(ord(c) < 32 or ord(c) > 126 for c in key):
        raise ValueError('GEMINI_API_KEY must be a single-line ASCII key. Check your local configuration.')
    return key


def _gemini_key():
    """Read one fixed file on demand, without expansion, execution, or environment mutation."""
    if 'GEMINI_API_KEY' in os.environ:
        return _validate_key(os.environ['GEMINI_API_KEY'].strip())
    try:
        with ENV_FILE.open(encoding='utf-8') as source:
            content = source.read(65_537)
    except FileNotFoundError:
        return ''
    except (OSError, UnicodeError):
        raise ValueError('Could not read the repository .env; check its permissions and UTF-8 encoding.') from None
    if len(content) > 65_536:
        raise ValueError('The repository .env is too large; keep it below 64 KiB.')
    key = ''
    for line in content.splitlines():
        name, separator, value = line.strip().removeprefix('export ').partition('=')
        if not separator or name.strip() != 'GEMINI_API_KEY':
            continue
        try:
            parts = shlex.split(value, comments=True, posix=True)
        except ValueError:
            raise ValueError('GEMINI_API_KEY has an unclosed quote in the repository .env.') from None
        if len(parts) > 1:
            raise ValueError('GEMINI_API_KEY must be one value in the repository .env.')
        key = parts[0].strip() if parts else ''
    return _validate_key(key)


def model_info():
    """Public configuration metadata; checking it makes no model or catalog requests."""
    try:
        configured = bool(_gemini_key())
        message = ('Key configured. Free-tier eligibility and quota depend on your Google project.'
                   if configured else MISSING_KEY)
    except ValueError as exc:
        configured, message = False, str(exc)
    return {'models': list(MODELS), 'provider': 'gemini', 'configured': configured,
            'default_model': DEFAULT_MODEL, 'message': message}


def gemini_models():
    return list(MODELS)


def gemini_provider(model):
    if model not in MODELS:
        raise ValueError('Select an available Gemini model.')
    key = _gemini_key()
    if not key:
        raise ValueError(MISSING_KEY)
    try:
        tpm_limit = int(os.environ.get('AGENTRX_DEMO_TPM_LIMIT', '200000'))
        if tpm_limit <= 0:
            raise ValueError
    except ValueError:
        raise ValueError('AGENTRX_DEMO_TPM_LIMIT must be a positive integer.') from None
    return HTTPProvider(model, BASE_URL, key, provider='gemini',
                        settings={'temperature': 0, 'max_tokens': 16000}, timeout=300, retries=1,
                        tpm_limit=tpm_limit)
