# AgentRx Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Deliver a runnable, audited reconstruction of AgentRx and its benchmark experiments.

**Architecture:** A dependency-free Python core separates observable trajectory data from labels. Injectable JSON model providers drive generation, semantic checking, and judging; a resumable experiment runner writes auditable artifacts and metrics.

**Tech Stack:** Python >=3.10, standard-library unittest, urllib HTTPS clients, setuptools packaging.

**Spec:** docs/superpowers/specs/2026-10-14-agentrx-reproduction-design.md

## Global Constraints

- Never invent examples or silently substitute newly generated trajectories for the original benchmark.
- Accuracy uses the fixed intended corpus denominator; abstentions and failed jobs cannot improve accuracy by dropping out.
- Repetitions cannot reuse stochastic generation or judging outputs from another repetition.
- Do not label fixture testing as successful reproduction of the paper's results.
- Keep the core pipeline usable without Azure-specific infrastructure.

## Review Focus

- Annotation aliases and mixed numeric/string IDs must join without selecting the wrong root.
- Sparse original step indices must never become Python list offsets.
- Invalid checker/model outputs must remain errors, rather than useful-looking predictions.
- Resume must invalidate on changes to model settings, prompts, corpus, or repetition.
- Missing predictions and data must not inflate accuracy or look like full-corpus replication.

---

### Task 1: Data contracts and audited corpus

**Files:** `agentrx/{__init__,data,taxonomy,io}.py`, `tests/test_data.py`, `pyproject.toml`, `.gitignore`, `THIRD_PARTY_NOTICES.md`.

**Interfaces:** `normalize(raw, domain, trajectory_id) -> dict`; `prepare_corpus(source, output) -> dict` writes `manifest.json` and `cases.json`; cases have `trajectory`, `gold`, `policy`, `tools` kept separately.

- [x] Write failing tests for label isolation, explicit indices, root lookup by ID, aliases, and mismatched gold steps.
- [x] Run `python3 -m unittest tests.test_data -v`; confirm missing implementation fails.
- [x] Implement contracts, pinned-source import, provenance/hash inventory, taxonomy, and license notices.
- [x] Run data tests and audit all released pairs; record actual usable/excluded counts.
- [x] Commit data-contract implementation.

### Task 2: Guarded Python and semantic checking

**Files:** `agentrx/{constraints,worker}.py`, `tests/test_constraints.py`.

**Interfaces:** `validate_constraint(dict) -> dict`; `guard_matches(constraint, step) -> bool`; `check_constraint(constraint, trajectory, offset, provider, context) -> dict` returns SKIP/SAT/VIOL/ERROR.

- [x] Write failing tests for compound event guards, ranges/regex, sparse indices, Python booleans/timeouts, rejected code, semantic UNCLEAR, malformed rubric, and prefix-only evidence.
- [x] Run `python3 -m unittest tests.test_constraints -v`; confirm failure.
- [x] Implement schema aliases, bounded isolated checks, rubric validation, and evidence records.
- [x] Run the checker tests and commit.

### Task 3: Models, prompts, and diagnostic pipeline

**Files:** `agentrx/{providers,prompts,pipeline}.py`, `tests/test_pipeline.py`, `tests/test_providers.py`, `.env.example`.

**Interfaces:** `provider.complete(stage, payload) -> dict`; `diagnose(trajectory, policy, tools, config, provider, artifacts) -> dict`; `CachedProvider` namespaces requests by experiment identity and records raw responses/usage.

- [x] Write failing tests for prefix synthesis, no future checking, recovered first failure, joint/two-stage judging, invalid predictions, request identity, independent repetitions, and HTTP errors/retries with a local server.
- [x] Run pipeline/provider tests; confirm missing behavior fails.
- [x] Implement paper taxonomy/checklist prompts, global/dynamic generation, both judging protocols, OpenAI-compatible/Azure clients, environment configuration, and resumable request artifacts.
- [x] Run pipeline/provider/checker/data tests and commit.

### Task 4: Repeated experiments and truthful reports

**Files:** `agentrx/{evaluation,experiments}.py`, `configs/*.json`, `references/paper_results.json`, `tests/test_evaluation.py`, `tests/test_experiments.py`.

**Interfaces:** `score(cases, predictions) -> dict`; `run_experiment(corpus_dir, config, output, provider_factory) -> dict`; `evaluate_run(corpus_dir, run_dir) -> dict`.

- [x] Write failing tests for missing/abstained/invalid predictions, temporal category ordering/ties, failed-agent aliases, mean/sample SD, fixed denominators, resume invalidation, and per-domain blocked reports.
- [x] Run experiment/evaluation tests; confirm failure.
- [x] Implement all six Table 4 configurations, both generation modes, repeated independent runs, JSON/CSV/Markdown tables and confusion matrices; add GPT-5/o3/DeepSeek configurations and external baseline/MAST import interfaces.
- [x] Run all tests and commit.

### Task 5: CLI, documentation, smoke test, and review

**Files:** `agentrx/{cli,__main__}.py`, `README.md`, `examples/*`, `tests/test_cli.py`.

**Interfaces:** `python -m agentrx {prepare,audit,diagnose,run,evaluate,import-predictions,stress}`.

- [x] Write failing subprocess tests for offline CLI smoke and missing-credential exit behavior.
- [x] Run CLI tests; confirm failure.
- [x] Implement commands and usage documentation, fake smoke example, benchmark limitations, and reconstruction deviations.
- [x] Run full suite, build/install package locally, audit 73 released pairs, execute smoke test and offline repeated experiment, and verify live-run preflight without credentials.
- [x] Request one independent whole-project code review and fix material findings with regression tests.
- [x] Record verification evidence and commit final implementation; report external blockers accurately.
