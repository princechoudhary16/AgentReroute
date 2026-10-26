"""The standard evaluation suite on one corpus split, in one call.

1. Attribution baselines (random, last step, heuristic, and the LLM judge when a
   model is available), each scored against gold.
2. Recovery conditions on the split's replayable episodes, controls included:
   no recovery, full restart, capped retry, generic reflection, fixed rules with
   each attribution baseline's predictions, and fixed rules with gold attribution
   (the oracle upper bound).
3. Paired McNemar tests of every condition against full restart.
4. Agent robustness from the generation episodes.
5. ``report.md`` and ``dashboard.html``.
"""
import sys
from pathlib import Path

from .corpus import generation_dir, load_gold, load_public
from .io import read_json, write_json
from .scoring.attribution import score_attribution
from .scoring.baselines import heuristic_baseline, last_step_baseline, llm_judge_baseline, random_baseline
from .scoring.recovery import POLICIES, BudgetCaps, compare, run_recovery
from .scoring.report import markdown
from .scoring.robustness import markdown as robustness_markdown
from .scoring.robustness import robustness

RECOVERY_CONDITIONS = ('no_recovery', 'full_restart', 'retry_last_step', 'generic_reflection')


def _slug(name):
    return name.replace(':', '-').replace('+', '-').replace('/', '-')


def evaluate(corpus_dir, split, llm_factory, out_dir, judge_llm=None, judge_model=None, caps=None,
             note=None, title='AgentReroute benchmark', log=sys.stderr, min_confidence=0.0):
    from .dashboard import build_dashboard
    from .generate import load_episodes
    corpus_dir, out_dir = Path(corpus_dir), Path(out_dir)
    traces, gold = load_public(corpus_dir, split), load_gold(corpus_dir, split)

    def say(*parts):
        print(*parts, file=log, flush=True)

    submissions = {'random': random_baseline(traces), 'last_step': last_step_baseline(traces),
                   'heuristic': heuristic_baseline(traces)}
    if judge_llm is not None:
        submissions['llm_judge'] = llm_judge_baseline(traces, judge_llm, judge_model)
    attribution = []
    for key, sub in submissions.items():
        sub['split'] = split
        write_json(out_dir / 'submissions' / f'{key}.json', sub)
        report = score_attribution(sub, gold)
        report['split'] = split
        write_json(out_dir / f'attribution-{_slug(sub["system"])}.json', report)
        attribution.append(report)
        say(f'attribution {sub["system"]}: step {report["metrics"]["step_accuracy"]["value"]}')

    conditions = [(name, POLICIES[name](), {}) for name in RECOVERY_CONDITIONS]
    for key in ('heuristic', 'llm_judge'):
        if key in submissions:
            hyps = {r['trajectory_id']: r['hypotheses'] for r in submissions[key]['predictions']}
            conditions.append((f'fixed_rules+{key}', POLICIES['fixed_rules'](min_confidence=min_confidence),
                               {'hypotheses': hyps}))
    conditions.append(('fixed_rules:oracle', POLICIES['fixed_rules'](), {'oracle': True}))
    recovery = []
    for label, policy, kwargs in conditions:
        report = run_recovery(corpus_dir, split, policy, llm_factory, caps or BudgetCaps(), **kwargs)
        report['policy'] = label
        write_json(out_dir / f'recovery-{_slug(label)}.json', report)
        recovery.append(report)
        say(f'recovery {label}: {report["metrics"]["recovery_rate"]["value"]}')
    baseline = next(r for r in recovery if r['policy'] == 'full_restart')
    comparisons = [compare(r, baseline) for r in recovery if r is not baseline]
    write_json(out_dir / 'comparisons.json', comparisons)

    gen_dirs = sorted({generation_dir(v, corpus_dir)
                       for v in read_json(corpus_dir / 'gold' / 'episodes.json').values()})
    robust = robustness([e for d in gen_dirs for e in load_episodes(d)])
    write_json(out_dir / 'robustness.json', robust)

    text = markdown(attribution, recovery, title=title, note=note)
    text += '\n### Paired comparison with full restart (McNemar, failed episodes)\n\n'
    text += '| Condition | n | Only this recovered | Only restart recovered | p |\n|---|---:|---:|---:|---:|\n'
    text += ''.join(f'| {c["a"]} | {c["n_paired"]} | {c["only_a"]} | {c["only_b"]} | {c["p_value"]:.3f} |\n'
                    for c in comparisons)
    text += '\n' + robustness_markdown(robust)
    (out_dir / 'report.md').write_text(text, encoding='utf-8')
    build_dashboard(corpus_dir, attribution, recovery, out_dir / 'dashboard.html', title, robustness=robust)
    return {'attribution': attribution, 'recovery': recovery, 'comparisons': comparisons, 'robustness': robust}
