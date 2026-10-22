# Checkpoints and results

- `rl_state.json` - trained `DualLoopRLJudge` state (76 episodes). Frozen: **copy it before using it as `rl_state_path`**, since training/eval runs write back to that path.
- `train_val_summary.json` - train/val metrics for that checkpoint.
- `results/<benchmark>/` - per-benchmark summaries and per-episode attribution logs. Add new benchmarks as sibling folders.
