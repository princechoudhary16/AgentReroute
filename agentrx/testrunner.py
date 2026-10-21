"""JUnit-style unittest runner with live progress, colour, and a failure summary."""
import json
import os
import shutil
import sys
import time
import unittest
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path

from .io import write_json
from .metrics import emit

GREEN, RED, YELLOW, DIM, BOLD, RESET = '\033[32m', '\033[31m', '\033[33m', '\033[2m', '\033[1m', '\033[0m'


def _enable_ansi():
    if os.name == 'nt':
        os.system('')  # Switches the Windows console into VT (ANSI) mode.


def use_color(stream):
    return stream.isatty() and 'NO_COLOR' not in os.environ


class ProgressResult(unittest.TestResult):
    def __init__(self, stream, total, color):
        super().__init__()
        self.stream, self.total, self.color = stream, total, color
        self.done = 0
        self.started = {}
        self.outcomes = []  # (test_id, module, status, seconds, detail)

    def _paint(self, text, code):
        return f'{code}{text}{RESET}' if self.color else text

    def _render(self):
        failed = len(self.failures) + len(self.errors)
        passed = self.done - failed - len(self.skipped)
        width = max(10, min(40, shutil.get_terminal_size((80, 20)).columns - 50))
        filled = int(width * self.done / max(self.total, 1))
        bar = '█' * filled + '░' * (width - filled)
        bar = self._paint(bar, RED if failed else GREEN)
        line = (f'\r{bar} {self.done}/{self.total}  '
                f'{self._paint(f"✓ {passed}", GREEN)}  '
                f'{self._paint(f"✗ {failed}", RED if failed else DIM)}  '
                f'{self._paint(f"○ {len(self.skipped)}", YELLOW if self.skipped else DIM)}')
        if self.stream.isatty():
            self.stream.write(line)
            self.stream.flush()

    def startTest(self, test):
        super().startTest(test)
        self.started[test.id()] = time.perf_counter()

    def _record(self, test, status, detail=''):
        elapsed = time.perf_counter() - self.started.pop(test.id(), time.perf_counter())
        module = test.id().rsplit('.', 2)[0]
        self.outcomes.append((test.id(), module, status, elapsed, detail))
        self.done += 1
        self._render()

    def addSuccess(self, test):
        super().addSuccess(test)
        self._record(test, 'pass')

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._record(test, 'fail', self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        self._record(test, 'error', self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._record(test, 'skip', reason)

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self._record(test, 'pass')

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self._record(test, 'fail', 'Unexpected success')


def _flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _flatten(item)
        else:
            yield item


def run_tests(start_dir='tests', top_level='.', pattern='test*.py', name_filter=None,
              failfast=False, json_path=None, stream=None):
    """Run discovered tests; returns (exit_code, summary_dict)."""
    stream = stream or sys.stdout
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')
    _enable_ansi()
    color = use_color(stream)
    sys.path.insert(0, str(Path(top_level).resolve()))
    suite = unittest.defaultTestLoader.discover(start_dir, pattern=pattern, top_level_dir=top_level)
    tests = [t for t in _flatten(suite) if not name_filter or name_filter.lower() in t.id().lower()]
    result = ProgressResult(stream, len(tests), color)
    paint = result._paint
    started = time.perf_counter()
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        for test in tests:
            test(result)
            if failfast and not result.wasSuccessful():
                break
    duration = time.perf_counter() - started
    if stream.isatty():
        stream.write('\n')

    by_module = defaultdict(lambda: defaultdict(int))
    for _, module, status, _, _ in result.outcomes:
        by_module[module][status] += 1
    stream.write('\n')
    for module in sorted(by_module):
        counts = by_module[module]
        bad = counts['fail'] + counts['error']
        mark = paint('✗', RED) if bad else paint('✓', GREEN)
        total = sum(counts.values())
        detail = paint(f'{bad} failed', RED) if bad else paint('all passed', GREEN)
        stream.write(f'  {mark} {module:<40} {total - bad - counts["skip"]:>3}/{total:<3} {detail}\n')

    failed = [o for o in result.outcomes if o[2] in ('fail', 'error')]
    if failed:
        stream.write(f'\n{paint("FAILED TESTS", RED + BOLD)}\n')
        for test_id, _, status, _, detail in failed:
            stream.write(f'  {paint("✗", RED)} {test_id}  [{status.upper()}]\n')
            for line in detail.rstrip().splitlines()[-6:]:
                stream.write(f'      {paint(line, DIM)}\n')
    passed = sum(o[2] == 'pass' for o in result.outcomes)
    skipped = sum(o[2] == 'skip' for o in result.outcomes)
    if not result.outcomes or failed:
        banner = f'  {len(failed)} FAILED / {passed} PASSED' + (f' / {skipped} SKIPPED' if skipped else '')
        banner += f'  ({len(result.outcomes)} run, {duration:.1f}s)  '
        banner = paint(banner, RED + BOLD)
    else:
        banner = paint(f'  ALL {passed} PASSED' + (f' ({skipped} skipped)' if skipped else '')
                       + f'  ({duration:.1f}s)  ', GREEN + BOLD)
    stream.write(f'\n{banner}\n')

    summary = {'total': len(result.outcomes), 'passed': passed, 'failed': len(failed),
               'skipped': skipped, 'duration_s': round(duration, 3),
               'failed_tests': [o[0] for o in failed],
               'modules': {m: dict(c) for m, c in by_module.items()}}
    if json_path:
        write_json(json_path, summary)
    emit('tests', **{k: v for k, v in summary.items() if k != 'modules'})
    return (0 if not failed and result.outcomes else 1), summary
