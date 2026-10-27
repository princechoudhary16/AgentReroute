# Research studio

```bash
python3 -m demo --serve --port 8765
```

Open http://localhost:8765/. The scrolling page explains the current fault-attribution
method, its evidence and limitations, followed by two working widgets. Dedicated
views are `/system` and `/compare`.

## What changed with the research

The final method combines perspective-scoped checks with an LLM that can choose
any step in the trace. Fixed checks use one selector call; written checks add
constraint synthesis. The initial THETA0 weights remain frozen. Training the
ranker and the repair cascade are reported negative results, not current benefits.

The previous synthetic-retail charts and 30.9%/21.2% recovery efficiency figures
are no longer headline results. Current charts derive from `results/final-evidence.csv`
and `results/selector-evidence.csv`, pinned to the final research revision. They
show exact step accuracy, Wilson intervals, paired comparisons and candidate recall.
The headline uses 29 tau and 44 magentic published traces, not corpus-v2. Contaminated
synthesis arms are excluded; magentic written-check performance remains unmeasured.
The CSV retains statuses and fallback counts, disclosed in the evidence details.
No improvement over AgentRx reaches statistical significance in these samples.

## Compare engines

Prepare the published corpus if needed:

```bash
python3 -m agentrx prepare --fetch
```

The server reads `data/benchmark/cases.json`. If absent, the widget explains how to
prepare it rather than substituting demonstration cases.

Set `GEMINI_API_KEY` in your process environment or in the ignored repository `.env`:

```dotenv
GEMINI_API_KEY=your-key
```

Refresh settings. The server keeps the credential out of HTML, request logs and
exported results. An environment variable takes precedence over `.env`.

Choose a published trace and **fixed checks** or **written checks**, then run the
comparison. Both arms receive the same observable input, policy and tools and use
Gemini 3.1 Flash Lite at temperature zero. AgentRx retains the existing constraint,
check and joint-judge path. AgentReroute uses the current trace selector, with no
repair cascade and no weight updates. The selected prediction is distinct from
the evidence shortlist: the selector may choose a step outside it.

Gold labels are held separately and used only after attribution for scoring. The
widget reports exact step, agent and category matches for that case. It cannot
infer task recovery or general benchmark accuracy from one diagnosis. Failed or
fallback calls are visible, and degraded comparisons are not labeled clean successes.

Requests are paced, including across the two arms. Set `AGENTRX_DEMO_TPM_LIMIT` to a positive token-per-minute limit if your project needs a different budget (default 200000). A quota or billing error can
still stop a run; the app does not guarantee free-tier access. Inspect the actual
request events and provider errors. Tests use fixture providers and never spend
model quota.

Pause, scrub or replay saved events to inspect earlier decisions. Playback makes
no inference calls. Saved links use `/compare?run=<id>` or
`/?run=<id>#compare-engines`. Historical runs retain their original model and
method context; older recovery demonstrations are not relabeled as current
published-trace results.

## Retail illustration

The four-agent retail workflow remains useful for explaining handoffs, context
loss and recovery. It is a scripted simulator using real fixed checks and repair
code, with initial policy weights frozen and no API calls. Its visible injected
fault labels are for the viewer only. Acceptance means the scripted replay passed
the repair checks; it is not evidence of benchmark accuracy or a learned policy gain.

## Verification

Run the unit suite with `python3 -m unittest discover -q`. Browser tests under
`demo/tests/browser_*.cjs` use a fixture server. Set `PLAYWRIGHT_MODULE` and
`CHROMIUM_PATH` if Playwright/Chromium are supplied outside the default environment.
They cover the embedded views, playback, current scoring, provider fallback states,
mobile layout, reduced motion and absence of inference while browsing.
