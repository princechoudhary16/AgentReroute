"""Isolate which field of the benchmark actor's /api/chat payload Ollama rejects.

Sends the exact payload benchmark.env.llm.OllamaClient builds, then variants with one thing removed at
a time, and prints the HTTP status and the server's own error body for each. The first OK line names
the culprit by elimination.

    python3 ollama_payload_probe.py --model qwen2.5:32b

Standard library only.
"""
import argparse
import copy
import json
import sys
import urllib.error
import urllib.request

ACTION_SCHEMA = {
    'type': 'object',
    'properties': {'thought': {'type': 'string'},
                   'action': {'type': 'string', 'enum': ['tool', 'handoff', 'reply']},
                   'tool': {'type': 'string'}, 'arguments': {'type': 'object'},
                   'to': {'type': 'string'}, 'message': {'type': 'string'},
                   'context': {'type': 'object'}, 'content': {'type': 'string'}},
    'required': ['action'],
}


def send(base, payload, timeout=120):
    req = urllib.request.Request(base + '/api/chat', data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read())
        return 200, repr((data.get('message') or {}).get('content', ''))[:70]
    except urllib.error.HTTPError as exc:
        detail = ''
        try:
            detail = exc.read(4096).decode('utf-8', errors='replace')
        except Exception:
            pass
        return exc.code, detail.strip()[:300]
    except Exception as exc:
        return 0, f'{type(exc).__name__}: {exc}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base-url', default='http://127.0.0.1:11434')
    ap.add_argument('--model', default='qwen2.5:32b')
    ap.add_argument('--num-ctx', type=int, default=16384)
    args = ap.parse_args()
    base = args.base_url.rstrip('/').removesuffix('/v1')

    # Exactly what OllamaClient.complete() builds for the first call of a run (the user simulator,
    # json_mode False, so no format key).
    full = {'model': args.model,
            'messages': [{'role': 'system', 'content': 'You are a user talking to an agent.'},
                         {'role': 'user', 'content': 'Hi! How can I help you today?'}],
            'stream': False, 'truncate': False, 'shift': False,
            'options': {'temperature': 0.0, 'seed': 0, 'num_ctx': args.num_ctx}}

    def without(*keys):
        payload = copy.deepcopy(full)
        for key in keys:
            payload.pop(key, None)
        return payload

    def plus(**fields):
        payload = copy.deepcopy(full)
        payload.update(fields)
        return payload

    variants = [
        ('as OllamaClient sends it (user turn)', full),
        ('without truncate and shift', without('truncate', 'shift')),
        ('without shift only', without('shift')),
        ('without truncate only', without('truncate')),
        ('minimal: model + messages + stream', {'model': args.model, 'messages': full['messages'],
                                                'stream': False}),
        ('with keep_alive, as the working probes sent it', plus(keep_alive='10m')),
        ('agent turn, format=json', plus(format='json')),
        ('agent turn, format=<action schema>', plus(format=ACTION_SCHEMA)),
        ('no-shift + format=<action schema>', {**without('shift'), 'format': ACTION_SCHEMA}),
    ]

    print(f'endpoint {base}   model {args.model}   num_ctx {args.num_ctx}\n')
    width = max(len(name) for name, _ in variants)
    for name, payload in variants:
        code, detail = send(base, payload)
        print(f'{"OK  " if code == 200 else f"{code} "} {name:{width}}  {detail}')
    print('\nThe first OK line, compared against the failing ones above it, names the rejected field.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
