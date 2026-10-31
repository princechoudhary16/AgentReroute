"""Measure, rather than assume, where the KV cache ceiling is for a loaded Ollama model.

Derives bytes-per-token from the model's OWN metadata, then loads the model at several num_ctx values
and reads Ollama's reported VRAM residency after each, so the ceiling comes from two independent
sources that have to agree. Also records decode speed at each size, which is where partial CPU offload
shows up as a throughput cliff.

    python3 ollama_vram_probe.py --model qwen2.5:32b
    python3 ollama_vram_probe.py --model qwen2.5:32b --sizes 8192,16384,32768,65536,98304

Each distinct num_ctx forces a reload, which for a 32B is 20-60s, so budget a few minutes.
Run `nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv` on the box alongside
this, to confirm nothing else holds VRAM.

Standard library only.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

GIB = 1024 ** 3


def post(base, path, payload, timeout=900):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(base, path, timeout=60):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.loads(r.read())


def kv_bytes_per_token(model_info):
    """KV cache bytes per token from the model's own metadata: 2 (K and V) x layers x kv_heads x head_dim x 2 bytes."""
    def pick(suffix):
        key = next((k for k in model_info if k.endswith(suffix)), None)
        return model_info.get(key) if key else None

    layers = pick('block_count')
    kv_heads = pick('attention.head_count_kv')
    heads = pick('attention.head_count')
    embedding = pick('embedding_length')
    head_dim = pick('attention.key_length')
    if head_dim is None and embedding and heads:
        head_dim = embedding // heads
    if not all(isinstance(v, int) for v in (layers, kv_heads, head_dim)):
        return None, {'layers': layers, 'kv_heads': kv_heads, 'head_dim': head_dim}
    return 2 * layers * kv_heads * head_dim * 2, {'layers': layers, 'kv_heads': kv_heads, 'head_dim': head_dim}


def resident(base, model):
    for entry in (get(base, '/api/ps').get('models') or []):
        if entry.get('name') == model or entry.get('model') == model:
            return entry
    return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base-url', default='http://127.0.0.1:11434')
    ap.add_argument('--model', default='qwen2.5:32b')
    ap.add_argument('--sizes', default='8192,16384,32768,65536')
    ap.add_argument('--vram-gib', type=float, default=45.0, help='card capacity, for the headroom estimate')
    args = ap.parse_args()
    base = args.base_url.rstrip('/').removesuffix('/v1')
    sizes = [int(s) for s in args.sizes.split(',') if s.strip()]

    info = post(base, '/api/show', {'model': args.model})
    per_token, parts = kv_bytes_per_token(info.get('model_info') or {})
    print(f'model {args.model}   card {args.vram_gib:.0f} GiB\n')
    print(f'metadata: layers={parts["layers"]} kv_heads={parts["kv_heads"]} head_dim={parts["head_dim"]}')
    if per_token:
        print(f'predicted KV cache: {per_token / 1024:.0f} KiB per token (fp16 K and V)\n')
    else:
        print('could not derive KV size from metadata; relying on measured residency only\n')

    hdr = (f'{"num_ctx":>8} {"total":>9} {"in VRAM":>9} {"on CPU":>8} {"residency":>10} '
           f'{"tok/s":>7} {"predicted KV":>13}')
    print(hdr); print('-' * len(hdr))
    rows = []
    for num_ctx in sizes:
        try:
            raw = post(base, '/api/chat', {
                'model': args.model, 'messages': [{'role': 'user', 'content': 'Say ok.'}],
                'stream': False, 'keep_alive': '5m', 'truncate': False, 'shift': False,
                'options': {'num_ctx': num_ctx, 'temperature': 0.0, 'seed': 0, 'num_predict': 48}})
        except urllib.error.HTTPError as exc:
            detail = exc.read(400).decode('utf-8', 'replace')
            print(f'{num_ctx:>8}  LOAD FAILED  HTTP {exc.code}: {detail[:120]}')
            rows.append((num_ctx, None, None, None))
            continue
        # Discard a warm-up generation before trusting the timing. Every row above is the first
        # generation after a reload, and an idle GPU ramps its clocks over the first few seconds, so
        # the earlier rows read slow for a reason that has nothing to do with num_ctx.
        try:
            post(base, '/api/chat', {'model': args.model, 'messages': [{'role': 'user', 'content': 'hi'}],
                                     'stream': False, 'keep_alive': '5m',
                                     'options': {'num_ctx': num_ctx, 'num_predict': 64}})
            raw = post(base, '/api/chat', {
                'model': args.model, 'messages': [{'role': 'user', 'content': 'Count 1 to 40.'}],
                'stream': False, 'keep_alive': '5m', 'truncate': False, 'shift': False,
                'options': {'num_ctx': num_ctx, 'temperature': 0.0, 'seed': 0, 'num_predict': 96}})
        except urllib.error.HTTPError:
            pass
        time.sleep(1)
        entry = resident(base, args.model)
        total = entry.get('size') or 0
        vram = entry.get('size_vram') or 0
        cpu = max(0, total - vram)
        share = (vram / total) if total else 0.0
        decode = (raw.get('eval_count', 0) / max(1e-6, raw.get('eval_duration', 0) / 1e9))
        pred = (per_token * num_ctx / GIB) if per_token else 0.0
        print(f'{num_ctx:>8} {total/GIB:8.1f}G {vram/GIB:8.1f}G {cpu/GIB:7.1f}G {share:9.0%} '
              f'{decode:7.1f} {pred:12.1f}G')
        rows.append((num_ctx, total, vram, decode))

    # A num_ctx above the model's trained context is silently capped, so that row reports the same
    # allocation as the largest real one. Including it in the regression averages across a span the
    # server never allocated and understates bytes/token. Keep only rows whose total actually moved.
    resident_rows = [r for r in rows if r[1] and r[2] and r[1] == r[2]]
    good, seen_totals = [], set()
    for row in resident_rows:
        if row[1] in seen_totals:
            print(f'  ignoring num_ctx {row[0]:,}: same allocation as a smaller size, so it was capped')
            continue
        seen_totals.add(row[1])
        good.append(row)
    offloaded = [r for r in rows if r[1] and r[2] and r[2] < r[1]]
    print()
    if len(good) >= 2:
        (c0, t0, _, _), (c1, t1, _, _) = good[0], good[-1]
        if c1 != c0:
            slope = (t1 - t0) / (c1 - c0)
            weights = t0 - slope * c0
            print(f'MEASURED slope {slope / 1024:.0f} KiB per token, weights+buffers {weights / GIB:.1f} GiB')
            if per_token:
                print(f'  against {per_token / 1024:.0f} KiB predicted from metadata '
                      f'({slope / per_token:.2f}x; above 1.0 means per-slot or compute buffers on top)')
            budget = args.vram_gib * GIB - weights
            print(f'  implied ceiling at 100% residency: ~{int(budget / slope):,} tokens of context')
    if offloaded:
        first = offloaded[0]
        print(f'OFFLOAD BEGINS at num_ctx {first[0]:,}: {(first[1]-first[2])/GIB:.1f} GiB on the host, '
              f'decode {first[3]:.1f} tok/s')
        print('  Partial offload is the usual cause of a 32B running at roughly half its bandwidth ceiling.')
    elif good:
        print(f'FULLY RESIDENT at every size tested, up to num_ctx {good[-1][0]:,}')
    print('\nNote: with OLLAMA_NUM_PARALLEL=N, Ollama allocates KV per slot, so total KV is N x num_ctx.')
    print('Parallel slots and context length compete for the same VRAM.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
