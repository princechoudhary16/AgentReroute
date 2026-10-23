# Studio Redesign Implementation Plan

**Goal:** A cohesive two-page agent reliability studio with Gemini inference.
**Architecture:** Shared embedded theme/navigation, existing independent page logic, fixed-endpoint Gemini backend with server-only .env reading.
**Tech Stack:** Python stdlib, HTML/CSS/JS, existing provider clients, Playwright QA.
**Spec:** docs/superpowers/specs/2026-10-23-studio-redesign.md

- [x] Gemini backend: complete_messages support, model_info contract, safe runtime key loading, defaults and focused regression tests. Delegated independently; exclude UI files.
- [x] Shared visual identity and live-system layout: keep graph/trace functionality; add scrubber and focused inspector. Browser verification plus existing demo/export tests.
- [x] Comparison redesign: coherent cards, visual algorithm progression, meaningful empty states, Gemini setup, recorded provider identity, preserved disclosures and playback. Existing browser QA and new setup/archival checks.
- [x] Integrate, review critical behavior, verify desktop/mobile and full suite, attempt bounded live Gemini smoke test if a key is present, restart server and document setup.

Review focus: no secret leaks; no fake successful connections or repairs; historical cursor cannot reveal future outcomes; simulator must remain labeled; page must fit mobile and retain keyboard navigation; Gemini 429s must remain errors; old Ollama records remain readable.

Verification completed: 510 unit tests (509 passed, one skipped), both Playwright
suites, desktop/mobile visual inspection, missing-key-to-configured refresh,
archived Ollama playback, and a real Gemini Flash-Lite comparison with accepted
repairs on both sides. Review identified streamed events skipping buffered replay;
a controlled-stream browser regression reproduced it and now passes. Keys were
absent from all 36 JSON artifacts in the Gemini run, and the HTTP server rejects
requests for .env. The larger Flash pilot was canceled after its in-flight request
finished; Flash-Lite is the default following the measured latency difference.
