# Live comparison UI implementation plan

> Use superpowers:executing-plans inline; fresh whole-change review at the end. User explicitly requested implementation and actual Ollama execution.

**Goal:** Show AgentRx and AgentReroute live side by side on identical controlled input.
**Architecture:** Bounded background comparison jobs instrument real pipelines and shared repair/replay. A plain HTML page consumes sequenced JSON events and provides playback and observed metrics.
**Tech stack:** Existing Python stdlib server, native Ollama, vanilla HTML/CSS/JS, Playwright for QA.
**Spec:** docs/superpowers/specs/2026-10-22-comparison-ui-design.md

## Constraints

- Real Ollama inference, scripted shared input/user simulator explicitly labeled.
- No trained-checkpoint or calibrated-accuracy claims; no gold in algorithm inputs/rewards.
- Same model/settings and repair backend, sequential runs, honest missing/error metrics.
- Preserve propagation view and existing APIs; retain bounded jobs, cancel safely, restrict POST origin/input.
- Local feature branch, preserve working checkout and running UI; no unrequested push.

## Review focus

- Actual model calls cannot silently be replaced by seeds while UI claims full inference.
- Broken/abstaining/canceled systems cannot inherit the other side's success.
- Browser refresh and playback do not duplicate inference or show future outcomes as current.
- Hostile POST/model/scenario/cursor input cannot invoke other endpoints or touch arbitrary paths.
- Verification and latency reflect their actual source; no gold leakage or made-up performance.

### Task 1: Comparison execution and job service

Files: demo/comparison.py, demo/tests/test_comparison.py.
- [x] Write RED tests for same-input real pipelines with transport fixture, telemetry, error isolation, cancellation/budgets and job bounds.
- [x] Implement controlled scenario inputs, actual diagnosis/shared replay, measured provider and bounded job lifecycle.
- [x] Run focused and full tests; commit and ledger.

### Task 2: Comparison routes and interactive page

Files: demo/comparison_page.py, demo/visualizer.py, demo/tests/test_comparison.py, README.md.
- [x] Write RED tests for page/models/job/cancel routes, strict input and origin checks.
- [x] Add page navigation, two pipeline panels, shared trace and event playback, measured metrics/errors, model selection and responsive layout.
- [x] Verify interactive behavior with browser automation; run suite; commit and ledger.

### Task 3: Live local verification and review

- [x] Run real installed-model comparison and verify request artifacts, recovery/error handling and UI states.
- [x] Fresh whole-change review; important fixes reproduced RED→GREEN; final suite.
- [x] Restart 8765 on finished code, open comparison page and leave UI running; report limitations and local branch.
