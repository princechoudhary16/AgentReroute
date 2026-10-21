"""`agentrx check`: one-screen health summary over tests, smoke, audit, and prevention."""
import json
import subprocess
import sys
import time
from pathlib import Path

from .metrics import emit
from .testrunner import GREEN, RED, YELLOW, DIM, BOLD, RESET, use_color, _enable_ansi

PASS, FAIL, WARN = 'pass', 'fail', 'warn'


def _run(argv):
    started = time.perf_counter()
    proc = subprocess.run([sys.executable, *argv], text=True, capture_output=True)
    return proc, time.perf_counter() - started


def _json_tail(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _last_line(text):
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ''


def check_tests():
    proc, secs = _run(['-m', 'agentrx', 'test', '--json', 'runs/metrics/last_tests.json', '--quiet'])
    data = _json_tail(Path('runs/metrics/last_tests.json').read_text(encoding='utf-8')
                      if Path('runs/metrics/last_tests.json').exists() else '')
    detail = f"{data.get('passed', '?')}/{data.get('total', '?')} passed"
    if proc.returncode:
        failed = ', '.join(data.get('failed_tests', [])[:3]) or _last_line(proc.stderr)
        return FAIL, detail, f'Run `python -m agentrx test` for details. Failing: {failed}', secs
    return PASS, detail, '', secs


def check_smoke():
    proc, secs = _run(['-m', 'agentrx', 'smoke'])
    data = _json_tail(proc.stdout)
    if proc.returncode:
        return FAIL, _last_line(proc.stderr) or 'smoke failed', 'Run `python -m agentrx smoke` for the raw error.', secs
    return PASS, f"{data.get('result_rows', '?')} result row(s), offline fixture", '', secs


def check_audit():
    if not Path('data/benchmark').exists():
        return WARN, 'benchmark corpus not prepared', 'Run `python -m agentrx prepare --fetch` (needs git + network).', 0.0
    proc, secs = _run(['-m', 'agentrx', 'audit'])
    data = _json_tail(proc.stdout)
    if proc.returncode:
        return FAIL, _last_line(proc.stderr) or 'audit failed', 'Corpus hashes or gold references are inconsistent; re-run prepare.', secs
    return PASS, f"{data.get('validated_pairs', '?')} trace/gold pairs validated", '', secs


def check_prevention(command):
    def check():
        proc, secs = _run(['-m', 'tool_prevention', command])
        if proc.returncode:
            return FAIL, _last_line(proc.stderr) or f'{command} failed', f'Run `python -m tool_prevention {command}`.', secs
        return PASS, 'ok', '', secs
    return check


def run_checks(quick=False, stream=None):
    stream = stream or sys.stdout
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')
    _enable_ansi()
    color = use_color(stream)

    def paint(text, code):
        return f'{code}{text}{RESET}' if color else text

    steps = [('Pipeline smoke', check_smoke), ('Benchmark audit', check_audit),
             ('Prevention smoke', check_prevention('smoke')), ('Prevention evaluate', check_prevention('evaluate'))]
    if not quick:
        steps.insert(0, ('Unit tests', check_tests))
    icons = {PASS: paint('✔', GREEN), FAIL: paint('✘', RED), WARN: paint('⚠', YELLOW)}
    stream.write(f'{paint("AgentReroute health check", BOLD)}\n\n')
    results = []
    for name, step in steps:
        status, detail, hint, secs = step()
        results.append({'check': name, 'status': status, 'detail': detail, 'seconds': round(secs, 2)})
        stream.write(f'  {icons[status]} {name:<22} {detail}  {paint(f"{secs:.1f}s", DIM)}\n')
        if hint:
            stream.write(f'      {paint("→ " + hint, YELLOW if status == WARN else RED)}\n')
        emit('check', check=name, status=status, detail=detail, seconds=round(secs, 2))
        stream.flush()
    failed = sum(r['status'] == FAIL for r in results)
    warned = sum(r['status'] == WARN for r in results)
    if failed:
        banner = paint(f'  {failed} CHECK(S) FAILED  ', RED + BOLD)
    elif warned:
        banner = paint(f'  HEALTHY WITH {warned} WARNING(S)  ', YELLOW + BOLD)
    else:
        banner = paint('  ALL CHECKS PASSED  ', GREEN + BOLD)
    stream.write(f'\n{banner}\n')
    emit('check_summary', failed=failed, warned=warned, total=len(results))
    return (1 if failed else 0), results
