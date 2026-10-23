# Guided Reliability Story Implementation Plan

**Goal:** One scroll-led explanation with working system and comparison widgets.
**Architecture:** New story page plus a shared widget bridge; reuse existing applications and server APIs. Native SVG/CSS scroll animation and accessible chapter navigation.
**Tech Stack:** Python stdlib, HTML/CSS/JavaScript, Playwright verification.
**Spec:** docs/superpowers/specs/2026-10-23-scroll-story-design.md

- [x] Add browser regression covering chapter navigation, deferred widget mounting, no implicit inference, live explanation updates, historic scrub explanations, resizing, reduced motion and mobile.
- [x] Implement story presentation, route diagram and chapter progression in demo/story_page.py; bridge/resizing in demo/widget_bridge.py; serve home, /system and /widgets/* from visualizer.py. Preserve /compare and exports.
- [x] Feed current system/comparison state into concise guides, preserve widget mode when starting runs, and update dedicated navigation.
- [x] Verify existing browser suites and full unit suite; inspect desktop/mobile and motion states; review changes, document routes and restart local server. No push.

Review focus: scroll must not start cloud calls; no iframe resize loop or trapped wheel scrolling; no future-result leakage in guides; errors cannot look like successful recovery; reduced motion and keyboard users can access every chapter.

## Verification

- Full unit suite: 510 tests, one skipped, no failures.
- All three browser suites passed: story, studio and comparison.
- Reproduced and fixed restored-session share links, disclosure-driven zoom resets, and tablet swipe trapping. Healthy-control narration now explains abstention and the absence of a repair target.
- Checked desktop/mobile layouts, reduced motion, current-event narration and iframe sizing. Scrolling and saved-run restoration issued no model-start requests.
- Refreshed the local server on port 8765; retained dedicated views, archives and offline exports.
