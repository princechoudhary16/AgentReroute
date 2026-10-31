"""Readiness probe for an Ollama endpoint, against the exact native path this repo uses.

Run it ON the GPU box (default) or against any reachable base URL:

    python3 ollama_probe.py                                   # http://127.0.0.1:11434
    python3 ollama_probe.py --base-url http://10.0.0.5:11434 --model qwen2.5:32b

It deliberately speaks the native /api/chat endpoint with num_ctx, truncate=False and shift=False,
the way agentrx.providers.OllamaProvider does, and NOT the OpenAI-compatible /v1 endpoint, which
cannot set num_ctx and lets Ollama silently truncate anything longer than its 4096 default.

Checks, in order of how badly each one bites:

1. no silent truncation      prompt_eval_count must match the prompt we actually sent
2. decode throughput         tokens/s, which decides whether a 40-task corpus is hours or days
3. determinism               same request twice with the same seed must return identical bytes
4. schema-constrained JSON   Stage 1 measured this as the fix for malformed actions (0/14 vs 4/23)
5. weights digest            pins a generated corpus to exact weights in the settings lock

Standard library only, no third-party imports.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

ACTION_SCHEMA = {
    'type': 'object',
    'properties': {
        'thought': {'type': 'string'},
        'action': {'type': 'string', 'enum': ['tool', 'handoff', 'reply']},
        'tool': {'type': 'string'},
        'arguments': {'type': 'object'},
        'content': {'type': 'string'},
    },
    'required': ['action'],
}


def post(base, path, payload, timeout=900):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=body,
                                 headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def get(base, path, timeout=30):
    with urllib.request.urlopen(base + path, timeout=timeout) as response:
        return json.loads(response.read())


def chat(base, model, messages, num_ctx, seed=0, fmt=None, num_predict=128, timeout=900):
    payload = {'model': model, 'messages': messages, 'stream': False, 'keep_alive': '10m',
               'truncate': False, 'shift': False,
               'options': {'num_ctx': num_ctx, 'temperature': 0.0, 'seed': seed,
                           'num_predict': num_predict}}
    if fmt is not None:
        payload['format'] = fmt
    start = time.monotonic()
    raw = post(base, '/api/chat', payload, timeout)
    return raw, time.monotonic() - start


def filler(approx_tokens):
    """Prose of roughly the requested token count (~4 chars per token)."""
    unit = ('The returns agent reviewed order #W0000042 and confirmed the item identifiers before '
            'calling the exchange tool with the declared arguments. ')
    return unit * max(1, (approx_tokens * 4) // len(unit))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base-url', default='http://127.0.0.1:11434')
    ap.add_argument('--model', default='qwen2.5:32b')
    ap.add_argument('--num-ctx', type=int, default=32768)
    ap.add_argument('--long-prompt-tokens', type=int, default=10000)
    args = ap.parse_args()
    base = args.base_url.rstrip('/')
    if base.endswith('/v1'):
        print('REFUSING: /v1 cannot set num_ctx and truncates silently. Drop the /v1 suffix.')
        return 2

    print(f'endpoint {base}   model {args.model}   num_ctx {args.num_ctx}\n')

    try:
        tags = get(base, '/api/tags')
    except (urllib.error.URLError, OSError) as exc:
        print(f'UNREACHABLE: {exc}')
        print('If this is the GPU box itself, is `ollama serve` running? If it is remote, Ollama binds')
        print('127.0.0.1 by default, so it needs OLLAMA_HOST=0.0.0.0:11434 plus a firewall opening.')
        return 2
    installed = {m.get('name'): m.get('digest', '')[:12] for m in tags.get('models', [])}
    print('installed models:')
    for name, digest in sorted(installed.items()):
        print(f'  {name:34} digest {digest}')
    if args.model not in installed and f'{args.model}:latest' not in installed:
        print(f'\nMISSING: {args.model} is not pulled. Run: ollama pull {args.model}')
        return 2

    info = post(base, '/api/show', {'model': args.model})
    model_info = info.get('model_info') or {}
    ctx_key = next((k for k in model_info if k.endswith('context_length')), None)
    trained_ctx = model_info.get(ctx_key) if ctx_key else None
    print(f'\ntrained context length: {trained_ctx}')
    print(f'modelfile parameters:   {(info.get("parameters") or "none").strip()[:200]}')
    if trained_ctx and args.num_ctx > trained_ctx:
        print(f'WARNING: num_ctx {args.num_ctx} exceeds the trained context {trained_ctx}')

    print('\n--- 1. truncation (the one that silently poisons a corpus) ---')
    sent = filler(args.long_prompt_tokens)
    try:
        raw, seconds = chat(base, args.model, [{'role': 'user', 'content': sent + '\n\nReply with one short sentence.'}],
                            args.num_ctx, num_predict=32)
        counted = raw.get('prompt_eval_count', 0)
        ratio = counted / max(1, args.long_prompt_tokens)
        prefill = counted / max(1e-6, raw.get('prompt_eval_duration', 0) / 1e9)
        print(f'  sent ~{args.long_prompt_tokens} tokens, server counted {counted} ({ratio:.0%})')
        print(f'  prefill {prefill:,.0f} tok/s over {seconds:.1f}s')
        if ratio < 0.8:
            print('  FAIL: the prompt was truncated. num_ctx is not being applied; do not generate a corpus.')
        else:
            print('  PASS: the whole prompt reached the model.')
    except urllib.error.HTTPError as exc:
        detail = exc.read(500).decode('utf-8', 'replace')
        print(f'  HTTP {exc.code}: {detail[:300]}')
        print('  A context-size error here is the CORRECT loud failure; raise num_ctx or shorten the trace.')

    print('\n--- 2. decode throughput ---')
    prompt = [{'role': 'user', 'content': 'Count from 1 to 60, one number per line.'}]
    raw, seconds = chat(base, args.model, prompt, args.num_ctx, num_predict=256)
    out = raw.get('eval_count', 0)
    decode = out / max(1e-6, raw.get('eval_duration', 0) / 1e9)
    print(f'  {out} output tokens in {seconds:.1f}s -> {decode:,.1f} tok/s decode')
    calls, per_call = 11000, 150 / max(1e-6, decode)
    print(f'  corpus-v2 estimate: ~{calls:,} calls x ~150 output tokens = {calls * per_call / 3600:.1f} h decode')
    print('  (sequential; task-level parallelism in our loop divides this almost linearly)')

    print('\n--- 3. determinism at temperature 0 with a fixed seed ---')
    a, _ = chat(base, args.model, prompt, args.num_ctx, seed=0, num_predict=64)
    b, _ = chat(base, args.model, prompt, args.num_ctx, seed=0, num_predict=64)
    same = a['message']['content'] == b['message']['content']
    print(f'  identical across two calls: {same}')
    if not same:
        print('  Without this, a regenerated corpus is not comparable (BRANCH-STATUS warning 3).')

    print('\n--- 4. schema-constrained JSON ---')
    action = [{'role': 'system', 'content': 'You are a retail order agent. Reply with one JSON action.'},
              {'role': 'user', 'content': 'Cancel order #W0000042 because it is no longer needed.'}]
    raw, _ = chat(base, args.model, action, args.num_ctx, fmt=ACTION_SCHEMA, num_predict=256)
    text = raw.get('message', {}).get('content', '')
    try:
        parsed = json.loads(text)
        ok = isinstance(parsed, dict) and parsed.get('action') in ('tool', 'handoff', 'reply')
        print(f'  parsed: {ok}   action={parsed.get("action")!r}   keys={sorted(parsed)}')
    except json.JSONDecodeError as exc:
        print(f'  FAIL: schema mode returned unparseable JSON ({exc}): {text[:160]!r}')

    print('\n--- 5. settings-lock identity ---')
    print(f'  weights digest {installed.get(args.model) or installed.get(args.model + ":latest")}')
    print('  This is what pins a corpus to exact weights, which the Gemini actor cannot provide.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
