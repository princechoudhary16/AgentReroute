# AgentReroute

A causal recovery framework for multi-agent workflows.

This branch implements **AgentRx-style fault attribution**: analyze a saved agent
run, check its actions, and identify the first unrecovered failure, responsible
agent, and supporting evidence.

## Packages

| Package | Role |
|---|---|
| `agentrx` | Fault attribution: first unrecovered failure, responsible agent, evidence (AgentRx reconstruction) |
| `benchmark` | Fault-injected multi-agent runs, gold labels, scoring ([README](benchmark/README.md)) |
| `repair_strategies` | Turns an attributed fault into a typed, verifiable repair plan ([README](repair_strategies/README.md)) |
| `recovery_orchestrator` | Decides whether, where and how to intervene, with a verifier gate and budget ([README](recovery_orchestrator/README.md)) |
| `tool_prevention` | Deterministic pre-execution tool enforcement on a retail backend |
| `tool_middleware` | Generic, conservative automatic tool prerequisites |
| `demo` | Interactive recovery studio and Gemini comparison (`python3 -m demo --serve`) |

Tests: `python -m agentrx test`, plus
`python -m unittest discover -s <package>/tests -t .` for `demo`, `repair_strategies`,
`recovery_orchestrator` and `benchmark`.

## Quick start

Requires Python 3.10+. No third-party runtime dependencies or installation needed.

```bash
python3 -m agentrx smoke
python3 -m agentrx test     # live progress, green/red summary
python3 -m agentrx check    # one-screen health summary (tests, smoke, audit, prevention)
```

The demo runs offline with prerecorded model responses. It verifies the pipeline,
not model accuracy. Reports are saved under `runs/smoke/experiment/`.

## Compare AgentRx and AgentReroute

```bash
python3 -m demo --serve --port 8765
```

Add `GEMINI_API_KEY=your-key` to the ignored root `.env` file, then open
[the studio](http://localhost:8765/). Scroll through the recovery story, then use
the embedded **Live system** and **Compare engines** tools. The comparison runs
both algorithms through Gemini Flash-Lite when you press **Run comparison**.
Dedicated views remain at `/system` and `/compare`. Refresh settings after saving
the key. Free-tier access depends on your Google project.
See [the guide](docs/comparison-ui.md).


## Benchmark data

```bash
python3 -m agentrx prepare --fetch
python3 -m agentrx audit
```

The pinned authors' release provides **73 labeled traces**: 29 retail and
44 Magentic-One. The full paper benchmark has 170; additional retail, Flash, and
RelWork data is still missing. Downloaded data and run outputs are excluded from Git.

## Run experiments

### 1. Local models with Ollama

Run locally with [Ollama](https://ollama.com/). Once the model is installed, inference needs no API key or internet access:

1. Start the Ollama server:
   ```bash
   ollama serve
   ```
2. Pull the default coding model:
   ```bash
   ollama pull qwen2.5-coder:7b
   ```
3. Run the experiment with `--model ollama`:
   ```bash
   python3 -m agentrx run --model ollama --out runs/ollama
   ```

To use an already installed model, keep the Ollama config and override its model name:

```bash
python3 -m agentrx run --config configs/dev_ollama.json --model YOUR_INSTALLED_MODEL --out runs/local
```

Live local smoke and single-trace pilots have been verified; diagnostic accuracy
is not established. See [verification](docs/verification.md#ollama-integration-verification--2026-10-18).

### 2. Cloud models (Gemini, OpenAI, or custom endpoints)

If you cannot run Ollama locally or want to use a cloud model, set your API key in the environment:

- **Google Gemini**:
  ```bash
  export GEMINI_API_KEY="your-gemini-api-key"
  python3 -m agentrx run --config configs/dev_gemini.json --out runs/gemini
  ```
- **OpenAI**:
  ```bash
  export OPENAI_API_KEY="your-openai-api-key"
  python3 -m agentrx run --model openai --out runs/openai
  ```

Test your setup without spending tokens or making network calls with `--preflight`:
```bash
python3 -m agentrx run --config configs/dev_gemini.json --preflight --out runs/gemini
```

### 3. Adding any OpenAI-compatible model

You can easily route to any OpenAI-compatible provider (e.g. Groq, Together AI, DeepSeek, vLLM, LiteLLM) by providing its endpoint and API key in a config file (or in `configs/dev_multi.json`):

```json
{
  "name": "custom-run",
  "client": {
    "provider": "openai-compatible",
    "base_url": "https://api.groq.com/openai/v1",
    "model": "llama-3.3-70b-versatile",
    "api_key_env": "GROQ_API_KEY"
  },
  "repetitions": 1,
  "domains": ["tau"],
  "limit_per_domain": 1,
  "variants": [
    {"name": "baseline", "evidence": false, "checklist": false},
    {"name": "agentrx", "evidence": true, "checklist": true, "generation": "stepwise"}
  ]
}
```

Then run:
```bash
python3 -m agentrx run --config configs/my_config.json --out runs/custom
```

Runs save diagnoses, evidence, and JSON/CSV/Markdown accuracy reports. Repeat the
same command to resume; changed code, data, or model settings require a new output
directory. The test suite covers diagnosis, providers, and tool prevention.


## Ranked recovery handoff (Track B)

```bash
python3 -m agentrx smoke-track-b
python3 -m agentrx run --config configs/track_b_ollama.json --model YOUR_INSTALLED_MODEL --out runs/track-b
```

Adds a configurable divergence gate and up to three ranked fault candidates in
`recovery_handoff.json`. Confidence scores are uncalibrated model estimates.
See [payload, thresholds and recovery integration](docs/track-b.md).


## Generic tool middleware

Use an LLM to propose prerequisite checks from an existing tool catalog, then
wrap its dispatcher. Before a guarded call, the middleware automatically runs
explicitly designated read-only validators. Original APIs, arguments, and successful
return values stay unchanged.

```bash
python3 -m tool_middleware smoke
python3 -m tool_middleware evaluate
```

See [discovery, integration, and limits](docs/prevention.md). The earlier retail-specific `tool_prevention` package is separate.

## Documentation

- [Usage, configuration, and implementation details](docs/usage.md)
- [Verification results and limitations](docs/verification.md)
- [Benchmark design](docs/benchmark-design.md)
- [Repair strategies design](docs/repair-strategies-design.md)
- [Tool prevention and middleware](docs/prevention.md)
- [Track B recovery handoff](docs/track-b.md)
- [Source attribution and licenses](THIRD_PARTY_NOTICES.md)

Independent reconstruction of [AgentRx](https://arxiv.org/abs/2602.02475), with
attributed material from [Microsoft's implementation](https://github.com/microsoft/AgentRx).
