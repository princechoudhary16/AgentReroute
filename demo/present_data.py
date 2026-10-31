"""Evidence and running order for the presentation route (/present).

Attribution numbers are not kept here: the page reads them from story_data, which validates the
committed CSVs against the pinned revision. What lives here is everything the CSVs do not hold:

* the running order (chapter, speaker, time budget, hand-off line), from the rehearsal script;
* the recovery results, Table 2 of the final report (no per-episode recovery CSV is committed yet);
* the corpus composition, from benchmark/runs/corpus-v2/manifest.json at revision b563572,
  copied so the page never depends on the generated run data being checked out.
"""

# Time budgets are the run of show in script v3 (150 words a minute).
CHAPTERS = (
    {'id': 'intro', 'title': 'Introduction', 'speaker': 'William', 'seconds': 71,
     'handoff': 'Badr will start with what already exists.'},
    {'id': 'existing', 'title': 'Existing work and the gap', 'speaker': 'Badr', 'seconds': 82,
     'handoff': 'Nischit will explain how we attribute a failure.'},
    {'id': 'attribution', 'title': 'Fault attribution', 'speaker': 'Nischit', 'seconds': 106,
     'handoff': 'Prince will show what we do with it.'},
    {'id': 'repair', 'title': 'Repair and orchestrator', 'speaker': 'Prince', 'seconds': 189,
     'handoff': 'Ananya will show what this did across a hundred failures.'},
    {'id': 'benchmark', 'title': 'Benchmark and results', 'speaker': 'Ananya', 'seconds': 83,
     'handoff': 'William will close.'},
    {'id': 'closing', 'title': 'Closing', 'speaker': 'William', 'seconds': 54, 'handoff': ''},
)

# Final report, Table 2. 100 failed test episodes; 25 healthy controls. Cost is tokens relative to the
# original run. ``harmed`` is the number of healthy controls the policy broke.
RECOVERY = (
    {'id': 'fixed', 'label': 'Fixed rules', 'note': 'no verifier',
     'claimed': 83, 'recovered': 51, 'harmed': 7, 'cost': 0.53},
    {'id': 'repair', 'label': 'Repair policy', 'note': 'acceptance checks only',
     'claimed': 68, 'recovered': 46, 'harmed': 3, 'cost': 0.65},
    {'id': 'standin', 'label': 'Orchestrator + stand-in check', 'note': '',
     'claimed': 60, 'recovered': 38, 'harmed': 0, 'cost': 0.36},
    {'id': 'verified', 'label': 'Orchestrator + LLM verifier', 'note': 'our attribution',
     'claimed': 39, 'recovered': 26, 'harmed': 0, 'cost': 0.40},
    {'id': 'verified_agentrx', 'label': 'Orchestrator + LLM verifier', 'note': 'AgentRx attribution',
     'claimed': 31, 'recovered': 18, 'harmed': None, 'cost': 0.45},
)
RECOVERY_EPISODES = 100
CONTROLS = 25
HARM_P = 0.016            # fixed rules, 7 of 25 broken, against the orchestrator's 0
RQ2_P = 0.057             # 26 vs 18 recoveries, paired
REPAIR_CONFIRMED_CLAIMS = 36   # of the repair policy's 68 claims, the hidden reward confirmed 36
VERIFIER_CONFIRMED_CLAIMS = 26  # of the verified orchestrator's 39 claims
ORCHESTRATOR_REJECTED = 14  # episodes only fixed rules recovered: repair rejected by the verifier
ORCHESTRATOR_ABSTAINED = 8  # same set: abstained because the repair cost too much

TRAINING = {'before': 44, 'after': 36, 'lost': 8, 'gained': 0, 'p': 0.008, 'reward_gap': 17.6,
            'train_episodes': 138, 'test_episodes': 100}

STRATEGIES = 22
DISTANCE_DISCOUNT = 0.97

# benchmark/runs/corpus-v2/manifest.json at b563572. 733 generated episodes; 491 entered the splits,
# the rest were dropped (held-out variant outside test, unlabeled natural failure, not activated).
CORPUS = {
    'tasks': 115,
    'generated': 733,
    'labelled': 491,
    'dropped': 242,
    'splits': {'train': 233, 'calibration': 93, 'test': 165},
    'test_failed': 100,
    'test_clean': 13,        # clean runs in the test split; the 25 controls are drawn from these plus tolerated runs
    'test_controls': 25,
    'labels': (
        ('context_propagation', 'Lost context', 79),
        ('instruction_adherence', 'Ignored instruction', 78),
        ('drift', 'Misrouted', 53),
        ('cycles', 'Stuck in a loop', 17),
        ('tool_failure', 'Tool failure', 15),
        ('misinterpretation_of_tool_output', 'Misread tool output', 10),
        ('missing_details', 'Missing details', 4),
    ),
    'clean': 35,
    'tolerated': 200,
}
