# AgentReroute

A causal recovery framework for multi-agent workflows.

This branch implements **AgentRx-style fault attribution**: analyze a saved agent
run, check its actions, and identify the first unrecovered failure, responsible
agent, and supporting evidence.

## Quick start

Requires Python 3.10+. No third-party runtime dependencies or installation needed.

```bash
python3 -m agentrx smoke
python3 -m unittest discover -v
```

The demo runs offline with prerecorded model responses. It verifies the pipeline,
not model accuracy. Reports are saved under `runs/smoke/experiment/`.

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
3. (Optional) Verify connectivity with the test script:
   ```bash
   python ask_ollama.py
   ```
4. Run the experiment with `--model ollama`:
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
  python3 -m agentrx run --model gemini --out runs/gemini
  ```
- **OpenAI**:
  ```bash
  export OPENAI_API_KEY="your-openai-api-key"
  python3 -m agentrx run --model openai --out runs/openai
  ```

Test your setup without spending tokens or making network calls with `--preflight`:
```bash
python3 -m agentrx run --model gemini --preflight --out runs/gemini
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


## Prevent tool errors

The prevention layer rewires retail tools with approval, ownership, state, and
idempotency checks. It includes an executable local backend and a dispatcher hook
for the project’s future live orchestrator backend.

```bash
python3 -m tool_prevention smoke
python3 -m tool_prevention evaluate
```

See [prevention setup, guarantees, and limits](docs/prevention.md).

## Documentation

- [Usage, configuration, and implementation details](docs/usage.md)
- [Verification results and limitations](docs/verification.md)
- [Design](docs/superpowers/specs/2026-10-14-agentrx-reproduction-design.md)
- [Source attribution and licenses](THIRD_PARTY_NOTICES.md)

Independent reconstruction of [AgentRx](https://arxiv.org/abs/2602.02475), with
attributed material from [Microsoft's implementation](https://github.com/microsoft/AgentRx).
