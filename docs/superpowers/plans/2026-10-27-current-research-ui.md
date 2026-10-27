# Current research UI integration implementation plan

> For agentic workers: use superpowers:subagent-driven-development to implement scoped tasks; the parent integrates and verifies.

Goal: restore the UI with claims and actual execution aligned to mp_rl-gemini-actor.
Architecture: retain stdlib server and rendered HTML widgets; derive presentation data from committed evidence CSV; adapt comparison service to current published-trace attribution. Keep baseline and core algorithms unchanged.
Tech stack: Python stdlib, HTML/CSS/JS, unittest, Playwright.
Spec: docs/superpowers/specs/2026-10-27-current-research-ui.md

## Constraints and review focus
- No gold in decision inputs, no mutation of theta, no regression of baseline complete().
- Failed synthesis and selector fallbacks are visible; incomplete runs cannot count as clean success.
- Original paper subset (73 cases) is distinct from internal development corpora.
- Missing corpus, missing key, legacy saved runs, provider errors and small screens remain usable.
- No model API calls during development or verification. No secret contents in output or commits.

## Tasks
- [x] Integrate UI files preserving newer providers and research code; run baseline tests and inspect merge conflicts.
- [x] Presentation: derive headline and ablation data from CSV with tested denominators/intervals/taint status; rewrite story, graphics, impact and source links.
- [x] Execution: published-case catalog, blind normalized inputs, frozen trace selector, attribution-only outcome, current model settings, explicit degraded status, updated unit/fixture coverage.
- [x] Widgets: current algorithm stages, dataset/variant controls, scored selected predictions, replay compatibility and self-explaining copy. Label scripted retail as illustration and remove learning claims.
- [x] Verify numerical claims, both widget paths, missing dependencies, legacy runs, mobile and reduced motion; run full tests and browser checks; review integration; commit integration and serve locally.

## Decisions
User requested integration into this branch: work in place, preserve both histories via merge. Scope is adapted current UI, not an arena or fresh evaluation campaign. Current .env credential convenience may remain server-side for backwards compatibility; no key in config or output.

## Verification

606 Python tests passed (one skipped). Fixture-only Playwright checks passed for the comparison, retail explorer, and scroll story, including saved runs and mobile layout. The server catalog contains 29 tau and 44 magentic cases. No live Gemini calls were made.
