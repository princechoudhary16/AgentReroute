# Replace presentation placeholders with benchmark results

User supplied mp_rl/docs/evaluation-and-hld.md as the actual source. The updated remote branch e96aed0 also includes raw summaries. Use those numeric artifacts, preserve their model and scope, and disclose discrepancies with the report. No inference or checkpoint loading is required.

- [x] Pin the two raw summaries under demo/resources/benchmark_35case and derive chart data from them; preserve full precision for calculations.
- [x] Replace sample charts with model selection plus quality/efficiency views, full metrics and source links. Keep the hosted recovery tie and the local diagnosis slowdown visible. Explain 35 failures, 17 controls, seven retail modes and trained policy versus the initial-weight live demo.
- [x] Replace illustrative impact percentages with hosted relative recovery-token ratio reduction and diagnosis-time reduction. No claims about API bills or end-to-end task completion.
- [x] Verify numeric rendering, bar geometry, model switching, mobile, reduced motion and existing live tools; review, restart server and commit locally.

Source discrepancies: report's local MP tokens/recovery=8187 and budget breaches=0; saved local JSON=14295.8 and 3. Saved ratio values also allow a more precise hosted reduction (30.9%, rather than 31.1% from rounded report values). Prefer pinned raw summaries and document this choice.


Verification: 510 unit tests passed (one skipped), plus the story, studio and comparison browser suites. Production-page checks covered hosted metrics and the expanded mobile source table without overflow, browser errors or inference requests. An independent review verified byte-for-byte source provenance, numerical calculations, bar scaling and both views across 320, 390, 820 and 1440 pixel widths; no actionable findings. The refreshed local server serves the new results on port 8765. No cloud inference or benchmark rerun was performed.
