"""Reporting-side RL metrics that read gold labels.

Kept out of the training/decision path (``rl_cross_branch``, ``pipeline``, ``rl_policy``), which never
reads gold. This module only SCORES predictions a frozen, read-only copy of the policy has already made;
the gold labels are never fed back to a policy under training.
"""
from pathlib import Path

from benchmark.corpus import load_gold, load_public
from benchmark.scoring.recovery import _EpisodeRunner

from .io import read_json
from .rl_cross_branch import MPRLRecoveryPolicy
from .rl_policy import DualLoopRLJudge


def attribution_exact_step(corpus_dir, split, judge, use_llm_synthesis, provider, llm_factory, limit=None):
    """Fraction of failed episodes in ``split`` whose top predicted step equals the gold critical step.

    A read-only copy of the policy attributes each trace, so training state and the checkpoint are
    untouched. ``_EpisodeRunner`` supplies the real task view (policy + tool schemas) and is never called,
    so no replay runs and the reward ledger is never read.
    """
    gold = load_gold(corpus_dir, split)
    traces = load_public(corpus_dir, split, 'full')
    index = read_json(Path(corpus_dir) / 'gold' / 'episodes.json')
    failed = [t for t, g in sorted(gold.items()) if g['status'] == 'failed' and t in index]
    if limit:
        failed = failed[:limit]
    if not failed:
        return None
    eval_judge = DualLoopRLJudge.from_dict(judge.to_dict())
    eval_judge.read_only = True
    policy = MPRLRecoveryPolicy(provider, rl_judge=eval_judge, learn=False, use_llm_synthesis=use_llm_synthesis)
    hits = 0
    for tid in failed:
        ref = index[tid]
        gen_dir = Path(ref['generation_dir'])
        episode = read_json(gen_dir / 'episodes' / f'{ref["episode_id"]}.json')
        gen_config = read_json(gen_dir / 'generation.json')['config']
        runner = _EpisodeRunner(episode, gen_config, llm_factory)
        env = type('Env', (), {'trajectory_id': tid, 'trace': traces[tid],
                               'task_view': runner.task_view(traces[tid]), 'hypotheses': []})()
        hyps = policy.attribute(env)
        hits += bool(hyps) and hyps[0]['step_index'] == gold[tid]['critical']['step_index']
    return hits / len(failed)
