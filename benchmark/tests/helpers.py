"""Shared fixtures: scripted generation into a temporary directory (built once per process)."""
import io
import tempfile
from functools import lru_cache
from pathlib import Path

from benchmark.corpus import build
from benchmark.env.scripted import scripted_llm
from benchmark.env.tasks import load_fixture
from benchmark.env.workflow import Workflow
from benchmark.generate import generate

_TMP = tempfile.TemporaryDirectory(prefix='benchmark-tests-')
ROOT = Path(_TMP.name)


@lru_cache(maxsize=1)
def generated():
    out = ROOT / 'gen'
    generate({'backend': 'scripted', 'model': 'scripted', 'seeds': [0]}, out, log=io.StringIO())
    return out


@lru_cache(maxsize=1)
def corpus():
    out = ROOT / 'corpus'
    build([generated()], out)
    return out


def run_task(task_id, faults=(), directive=None, max_steps=60):
    domain = load_fixture()
    task = domain.task(task_id)
    return domain, task, Workflow(domain, task, scripted_llm(domain, task), 'scripted', faults=faults,
                                  directive=directive, max_steps=max_steps).run()
