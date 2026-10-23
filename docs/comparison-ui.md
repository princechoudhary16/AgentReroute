# Live algorithm comparison

Run from the repository checkout:

```bash
python3 -m demo --serve --port 8765
```

Add the Gemini key to the repository's ignored `.env` file:

```dotenv
GEMINI_API_KEY=your-key
```

Open [the studio](http://localhost:8765/). The homepage tells the recovery story
as you scroll: core ideas, an animated walkthrough, then both working tools inline.
Each tool has a short explanation that follows the currently selected step.
The walkthrough expands the planning team into intake, context and routing agents,
then follows a mistaken handoff through its outcome and correction. A visual
AgentRx/AgentReroute explanation and paired charts introduce the live comparison.
The closing section illustrates targeted replay and compares its role with tool
validation and retries.

**The presentation charts and 30%/20% impact targets are placeholders**, as labeled
on the page. They are not benchmark findings or outputs of the live comparison.
Their editable values are in `demo/story_data.py`; replace them, along with the
sample labels and evaluation context, when measured results are available.

**Live system** is a scripted retail simulation with the real recovery policy.
**Compare engines** runs AgentRx and AgentReroute through Gemini for diagnosis and
post-rollback agent turns. Choose a scenario and press **Run comparison**.
Scrolling and opening widgets do not start model inference.
Saving the key and refreshing settings is
enough; no server restart is needed. The server reads the key, never the browser.
An exported `GEMINI_API_KEY` takes precedence over `.env` (including an empty value).
The command-line benchmark runner still expects an exported key.

The default is `gemini-3.5-flash-lite`, with `gemini-3.8-flash` also available. Both
use Google's official OpenAI-compatible endpoint, low reasoning effort and a
4,096-token output cap. Google lists a
[free tier](https://ai.google.dev/gemini-api/docs/pricing) for both models. Eligibility,
limits and charges depend on the project's billing settings; the app cannot select
an account tier or guarantee zero charges. It makes no billing changes and has no
fallback to a paid-only model. A configured key is not proof of valid access or quota.

The two panels show actual `diagnose` / `diagnose_mp_rl` calls, generated checks,
attribution, repair attempts and measured model usage. Outcomes, stages and key
numbers stay visible. Expand **Inspect evidence** for all requests; selecting
one opens its response in the event inspector. Full metrics and the shared input
trace are expandable below the panels. Validation warnings remain visible even
when details are closed. Use the
shared timeline to pause, step backward or forward, and return to live events.
Historical playback does not start additional inference or reveal future results
as current. Completed runs can be revisited through the `?run=<id>` URL, including
after restarting the server; refreshing an active page resumes observation.

Use `/system` and `/compare` for dedicated full views. Inline saved comparisons
also have portable links: `/?run=<id>#compare-engines`. The page respects reduced
motion settings and offers a motion toggle. Chapter links let you skip directly
to either tool without scrolling through the explanation.

## What is being compared

Both systems receive an identical **recorded scripted retail trace** and use the
same selected model/settings. AgentRx uses evidence, the checklist, one-shot
dynamic constraint generation and the original joint judge. AgentReroute uses the
three perspectives, LLM-generated constraints plus built-in seeds, and a fresh
policy with the initial weights. It does not load a trained checkpoint.

Both feed their findings into the **same repair engine and retail environment**.
The original prefix and user simulator are scripted to keep the case repeatable;
agent turns after rollback call Gemini and invoke the actual local retail tools.
Neither side is fed injected-fault labels or gold actions. Final retail-state and
required-output scores are computed after the repair policy returns; observable
repair acceptance supplies MP policy feedback. A failed repair remains failed on
the page, even when attribution found the injected step.

Systems execute sequentially with identical model settings. Timings include
network latency and provider queueing. Token counts come from provider telemetry;
unavailable measurements stay blank. Known usage from failed requests is included;
if any finished request has unknown usage, its token totals stay blank. One run is an observation, not benchmark
accuracy, and a policy-selection score is not a calibrated failure probability.
Initial weights and benchmark-tuned seed checks limit generalization claims.

Model generation can take several minutes. **Stop** prevents further inference
after the current request finishes (each HTTP attempt has a 180-second timeout; one bounded retry may follow).
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
node demo/tests/browser_studio.cjs
node demo/tests/browser_story.cjs
```

Use `PLAYWRIGHT_MODULE` for an installed module path and `CHROMIUM_PATH` for a
local Chrome executable when needed. The script starts its own isolated fixture
server on a free port, exercises the actual page and real pipelines with only
the external model transport replaced, and shuts it down. It checks historical
counters, stage completion, future-result isolation, model/scenario restoration,
trace inspection, cancellation, invalid-synthesis warnings, mobile layout and
browser errors. Screenshots go
to `runs/ui-comparison-qa/`, `runs/ui-studio-qa/` and `runs/ui-story-qa/`.
The story checks also cover deferred widgets, current-step explanations, saved
run links, touch scrolling, preserved map zoom, and iframe sizing.
This fixture never replaces the real 8765 server.

Historical Ollama runs keep their original provider identity and remain viewable.
A real local Qwen3-VL 8B pilot on the unbacked-write scenario completed on both
sides. Both selected step 9 / Instruction/PlanAdherenceFailure. AgentRx made 17
model requests and took about 206s; AgentReroute made 21 requests and took about
230s. Neither achieved the required final retail state, and neither replay was
accepted. The page preserves that failed outcome. These are single-run findings
with initial policy weights, not a trained-model or benchmark performance claim.

A real Gemini Flash-Lite run on 2026-10-23 (`00d7dd20d564442da1a767accbad7b62`)
completed the misrouted-specialist scenario in about 25 seconds. Both systems
identified planner step 6, replayed an accepted repair, and met the final task.
AgentRx used 7 requests (8.72s elapsed); AgentReroute used 10 (16.61s), generated
5 valid model constraints, and had no rejected constraints or failed requests.
The actual saved results were checked in the browser on desktop and mobile.
This is a single controlled case, not a benchmark accuracy comparison.
