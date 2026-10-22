# Live algorithm comparison

Run from the repository checkout:

```bash
python3 -m demo --serve --port 8765
```

Open [Compare systems](http://localhost:8765/compare), choose an installed Ollama
model and a scenario, then press **Run comparison**. Ollama must be running with
an already installed model. No API key, model download or trained checkpoint is
required by this page.

The two panels show actual `diagnose` / `diagnose_mp_rl` calls, generated checks,
attribution, repair attempts and measured model usage. Outcomes, stages and key
numbers stay visible. Expand **Checks & model requests** for all requests; selecting
one opens its response in the event inspector. Full metrics and the shared input
trace are expandable below the panels. Validation warnings remain visible even
when details are closed. Use the
shared timeline to pause, step backward or forward, and return to live events.
Historical playback does not start additional inference or reveal future results
as current. Completed runs can be revisited through the `?run=<id>` URL, including
after restarting the server; refreshing an active page resumes observation.

## What is being compared

Both systems receive an identical **recorded scripted retail trace** and use the
same selected model/settings. AgentRx uses evidence, the checklist, one-shot
dynamic constraint generation and the original joint judge. AgentReroute uses the
three perspectives, LLM-generated constraints plus built-in seeds, and a fresh
policy with the initial weights. It does not load a trained checkpoint.

Both feed their findings into the **same repair engine and retail environment**.
The original prefix and user simulator are scripted to keep the case repeatable;
agent turns after rollback call Ollama and invoke the actual local retail tools.
Neither side is fed injected-fault labels or gold actions. Final retail-state and
required-output scores are computed after the repair policy returns; observable
repair acceptance supplies MP policy feedback. A failed repair remains failed on
the page, even when attribution found the injected step.

Systems execute sequentially to avoid competing for the local GPU. Timings include
model loading and queueing. Token counts come from native Ollama telemetry;
unavailable measurements stay blank. Known usage from failed requests is included;
if any finished request has unknown usage, its token totals stay blank. One run is an observation, not benchmark
accuracy, and a policy-selection score is not a calibrated failure probability.
Initial weights and benchmark-tuned seed checks limit generalization claims.

Ollama generation can take several minutes. **Stop** prevents further inference
after the current request finishes (up to its 180-second transport timeout).
There is a 64-request cap per system and a separate three-attempt recovery budget.
If model generation fails, its errors remain in the log; partial seed fallback is
not presented as an error-free model result. Rejected model rules and exhausted
validation retries show a partial-result warning; valid empty rule lists are
labeled separately as seed-only checks. The server permits one active
comparison, retains eight jobs in memory, and saves completed runs on disk.

Artifacts are under `runs/ui-comparisons/<id>/`: shared input, per-system request
and diagnosis logs, measured results, MP feedback state and `comparison.json`.
The browser cannot choose arbitrary endpoints, files or checkpoint paths. The
comparison is a local development tool using the bundled retail fixture.

## Verification

```bash
python3 -m unittest demo.tests.test_comparison -v
python3 -m unittest discover -q
```

Optional browser QA requires Playwright and Chromium (development tools only):

```bash
node demo/tests/browser_comparison.cjs
```

Use `PLAYWRIGHT_MODULE` for an installed module path and `CHROMIUM_PATH` for a
local Chrome executable when needed. The script starts its own isolated fixture
server on a free port, exercises the actual page and real pipelines with only
the external model transport replaced, and shuts it down. It checks historical
counters, stage completion, future-result isolation, model/scenario restoration,
trace inspection, cancellation, invalid-synthesis warnings, mobile layout and
browser errors. Screenshots go
to `runs/ui-comparison-qa/`. This fixture never replaces the real 8765 server.

A real local Qwen3-VL 8B pilot on the unbacked-write scenario completed on both
sides. Both selected step 9 / Instruction/PlanAdherenceFailure. AgentRx made 17
model requests and took about 206s; AgentReroute made 21 requests and took about
230s. Neither achieved the required final retail state, and neither replay was
accepted. The page preserves that failed outcome. These are single-run findings
with initial policy weights, not a trained-model or benchmark performance claim.
