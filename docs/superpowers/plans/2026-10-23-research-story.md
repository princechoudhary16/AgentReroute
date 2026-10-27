# Research story implementation plan

**Goal:** Explain a multi-agent routing failure, the multi-perspective contribution, placeholder benchmark comparisons and practical implications.
**Architecture:** Keep the existing Python/HTML application and live widgets. Add reusable narrative sections and a small explicit dataset transcribed from the repository's evaluation report. No external dependencies or inference.
**Spec:** User's three review comments; extends docs/superpowers/specs/2026-10-23-scroll-story-design.md.

## Constraints and decisions

- Keep the current branch and all live controls, archives, accessibility and reduced-motion behavior.
- Replace only the mechanism diagram: intake agent → context agent → routing agent → wrong specialist → task remains incomplete. Correction targets the routing agent; earlier work stays intact.
- Insert an explanation of AgentRx's joint judge versus three scoped perspectives and recovery feedback before section 03, followed by paired bar charts.
- User clarified that actual figures will arrive later. Use clearly labeled illustrative data, stored in one editable Python module. Never present it as measured or as output from the live Gemini widget.
- Preview paired bars for fault-location accuracy, task recovery, relative cost and completion time. Switch between Quality and Efficiency. Show sample-data labels and do not invent dataset sizes or model names.
- Final impact cards use the user’s illustrative 30% cost and 20% time examples, explicitly labeled example targets, not measured savings. Keep all values in demo/story_data.py.
- Other-method comparison is qualitative, with appropriate uses and limitations; no invented benchmark bars for methods without measurements.

## Tasks

- [x] Add behavioral browser coverage for the multi-agent diagram's four states, dataset switching, exact chart values and impact/source labels. Observe failure before implementation.
- [x] Implement demo/story_sections.py (diagram, novelty, charts and impact), demo/story_data.py (sample values and derived percentages), integrate with story_page.py. Add keyboard-accessible diagram state buttons and model switches. Keep native scrolling and no implicit inference.
- [x] Inspect desktop/mobile screenshots; run story and existing browser suites plus unit tests. Review numeric claims and source provenance. Refresh the local server, document changes and commit locally.

## Review focus

- Root cause versus downstream symptom is visually obvious; correcting the router changes the specialist and final result.
- Placeholder figures retain clear sample labels in every context, and impact values agree with the editable data.
- Charts remain readable on mobile and without motion; data switching issues no network/model request.
- Scroll selection and manual diagram buttons cooperate; no keyboard or touch traps.
- Existing widget functionality and portable links remain intact.

## Completion evidence

- User clarified that real figures will arrive later; all comparison and impact numbers are explicitly illustrative and live in demo/story_data.py.
- Story browser suite passed, including paused-chart geometry, all four team states, model-free chart switching, existing widgets, 320px navigation, short-screen controls, mobile and reduced motion.
- Existing studio and comparison browser suites passed. Full unit suite: 510 tests, one skipped, no failures.
- Reviewer independently confirmed fixes for paused chart bars, short-screen controls and narrow navigation; no remaining findings.
- Local site refreshed on port 8765. Browser check confirmed the new sections, no page errors and zero model-start requests. Desktop/mobile screenshots inspected.
