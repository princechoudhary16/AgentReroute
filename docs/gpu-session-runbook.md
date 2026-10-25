# GPU session runbook: generation on a Slurm-allocated Ollama box

Real values (login node address, username, job script) live in the author's own notes, not here.
Placeholders below: `<USER>`, `<LOGIN_NODE>`, `<COMPUTE_NODE_IP>`, `<JOB_ID>`.

The important property: the compute node changes every session, but the **tunnel always presents the
server at `http://localhost:11434`**, which is what `benchmark`'s defaults already target
(`DEFAULT_HOST`), and which `agentrx.providers.OllamaProvider` permits over plain HTTP because
`127.0.0.1`, `localhost` and `::1` are exempt from its remote-HTTPS rule. So no command in this repo
changes between sessions. Only the tunnel command does.

## Start of every session

```bash
# 1. VPN on, then from local PowerShell
ssh <USER>@<LOGIN_NODE>

# 2. On the login node: allocate and serve
sbatch start_ollama.sh
squeue -u <USER>
cat ollama_<JOB_ID>.log | grep "NODE IP"
```

```powershell
# 3. From local PowerShell, in a window you will not touch again for the rest of the session
ssh -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 11434:<COMPUTE_NODE_IP>:11434 <USER>@<LOGIN_NODE>
```

`-N` opens no remote shell, so there is nothing to accidentally `exit` from, and the keepalives stop an
idle VPN or firewall dropping it silently. **This tunnel is a single point of failure for every running
generate process.** If it dies they all fail with a transport error. They are resumable, so the cost is
restart time, not data.

```powershell
# 4. Verify, from the repo root
python3 -m benchmark check-ollama --model qwen2.5:32b
python3 ollama_vram_probe.py --model qwen2.5:32b --sizes 16384
```

Expect `34.3G` total at 100% residency with `OLLAMA_NUM_PARALLEL=4`: that is 18.5 GiB of weights plus
four slots of 4.0 GiB. A figure near 22.5G means only one slot was allocated and the four generate
processes will queue serially instead of running side by side.

## Before resuming a generation from a previous session

`generate` refuses to resume a directory whose settings lock disagrees, and `build-corpus` refuses to
merge directories that disagree with each other. The lock includes `model_digest`.

```powershell
python3 -c "import json;print(json.dumps(json.load(open('benchmark/runs/gen-v2-0/generation.json'))['identity'], indent=1))"
```

Compare `model_digest` against the digest `check-ollama` prints. If they match, resume by re-running the
identical generate command; completed episodes are skipped from `index.jsonl` and the response cache
makes a half-finished episode nearly free.

If they differ, the model on the box is not the one that made the corpus and there is no clean merge.
Decide deliberately: re-pull the matching version, or treat the remaining tasks as a separate corpus.

**Check once, in `start_ollama.sh`: where does `OLLAMA_MODELS` point?** A path in the home directory is
shared across compute nodes, so the digest stays stable and resume works every session. Node-local
scratch means each job re-pulls, and the digest can move under you.

## Generation

Four windows, one per slice. Every slice must pass identical `--model`, `--num-ctx`, `--seeds` and
`--backend`, or the merge will be refused.

```powershell
python3 -m benchmark generate --data-dir benchmark/data/tau_retail --backend ollama `
  --model qwen2.5:32b --num-ctx 16384 --seeds 0 --max-tasks 40 --task-slice 0/4 `
  --out benchmark/runs/gen-v2-0
```

Monitor from any window, mid-run, as often as you like:

```powershell
python3 generation_gate.py benchmark/runs/gen-v2-0 benchmark/runs/gen-v2-1 benchmark/runs/gen-v2-2 benchmark/runs/gen-v2-3
```

Merge when all four finish:

```powershell
python3 -m benchmark build-corpus --gen benchmark/runs/gen-v2-0 benchmark/runs/gen-v2-1 `
  benchmark/runs/gen-v2-2 benchmark/runs/gen-v2-3 --out benchmark/runs/corpus-v2
```

## Failure modes and what each one means

| symptom | cause | action |
|---|---|---|
| `Ollama unreachable` / transport failure | tunnel dropped, or the Slurm job hit its walltime | re-establish, then re-run the same generate command |
| `Ollama context window (16384) reached` | a prompt outgrew the window | raise `--num-ctx` in a **new** output directory; it is part of the settings lock |
| `was generated with different settings` | the lock disagrees, usually `model_digest` | compare identities before deciding; do not work around it |
| `generation directories disagree on [...]` | slices were run with different flags | rebuild the odd slice, or build separate corpora |
| gate prints `FAIL` | unrecovered actor replies at or above 2% | stop; decoding failures are thinning the corpus rather than the injected faults |
| many `natural_failure` statuses | the model cannot complete clean runs | an eligibility problem, not a decoding one; the corpus shrinks but stays valid |

## Why the pipeline is not run on the cluster

The tunnel puts the model one hop away while the repo, its caches and its outputs stay on local disk,
which is where the corpus has to end up anyway. Running `generate` locally also means a Slurm walltime
expiry kills only the model, not the bookkeeping: `index.jsonl` and `llm_cache/` survive on local disk
and the next session resumes against them.
