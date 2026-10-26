# AgentReroute, from scratch

Written to be read top to bottom by someone who has never seen this repo. Each section assumes
only the ones before it. Stop at any point and you will still have a coherent picture.

Last updated 2026-10-26.

---

## 1. The question

A multi-agent system runs: a planner takes a customer request, hands off to specialists, they
call tools, something goes wrong, the customer leaves unhappy.

**Which step caused it?**

Not "the run failed", but "step 19, the returns agent, misread the tool output". That is
*fault attribution*, and it is the whole project.

## 2. Why it is hard

To prove you can find the culprit step, you need failures where the culprit is already known.

Real failures do not arrive labelled. Nobody annotates production traces with "the mistake
happened at step 19". So you cannot simply collect failures and check your answers.

**The way around it: manufacture failures whose cause you chose.** That is the generation phase,
and it is why half this repo exists.

## 3. Manufacturing labelled failures

Start with a task from tau-bench retail, a customer-service benchmark. For example:

> You are Yusuf Rossi in 19122. You received order #W2378156 and want to exchange the mechanical
> keyboard for one with clicky switches.

Run the multi-agent system on it. Three outcomes matter:

**The clean run.** No interference. Did the system solve the task?

**Eligibility.** A task is only useful if its clean run *succeeded*. The reason is strict:

> 1. The system can do this task unaided
> 2. You broke exactly one thing, at one step
> 3. It then failed
>
> Therefore that thing caused the failure.

Remove leg 1 and the argument collapses. If the system already fails, sabotaging it and watching
it fail proves nothing. On our corpus only **35 of 115 tasks (30%)** passed their clean run. The
other 80 were discarded. That is not waste, it is the precondition for a valid label.

**Injection.** Into an eligible task's trajectory, insert one deliberate fault at one position:
drop the context on a handoff, misroute to the wrong specialist, return a stale tool result. Then
replay. The culprit step is now known *by construction*, because you picked it.

### What comes out

One recorded attempt is an **episode**. Our corpus has 733 of them:

| status | meaning | useful for training? |
|---|---|---|
| `failed` | the injected fault caused failure | **yes, 322 of these** |
| `tolerated` | fault injected, system recovered anyway | no causal link |
| `not_activated` | the fault's precondition never arose | nothing happened |
| `clean` | the successful baseline | reference |
| `natural_failure` | a clean run that failed by itself | cause unknown |

**322 gold-labelled episodes** is the actual dataset.

> **Checkpoint.** Generation does not train anything. It produces data. If that is clear, continue.

## 4. Finding the fault

Now the other half: given a trajectory and no labels, name the culprit step.

### Step 1, run rules over every step

Five hand-written Python predicates, grouped into three viewpoints:

- **P1_INTENT_PLAN** (2 rules): did the plan stay coherent?
- **P2_TOOL_EXECUTION** (1 rule): were tools invoked correctly?
- **P3_HANDOFF_GROUNDING** (2 rules): did information survive the handoffs?

One of them, simplified:

```python
def p2_seed_valid_tool_invocation(trajectory, current_step_index):
    for call in tool_calls_at(current_step_index):
        if not isinstance(call.get('arguments'), dict):
            return False          # malformed invocation
        if call['name'] not in catalog:
            return False          # tool that does not exist
```

Each rule runs against each step and returns SAT, VIOL or SKIP. **No model is involved.**

A rule only sees `prefix(trajectory, offset)`, the steps up to and including the one being
checked. It cannot see the future. Without that, a check could flag step 5 because of what
happened at step 40, which is hindsight, and it would make the problem look easy for the wrong
reason.

### Step 2, aggregate into blame

For each (viewpoint, agent): `blame = violations / (violations + satisfactions)`. How often does
this viewpoint find this agent in violation?

### Step 3, build candidates

Every agent-owned step with a violation becomes a **candidate**. On tau traces there are about
4.5 candidates per trace. A violation on a tool output is folded into the agent that called it.

### Step 4, rank them

Each candidate becomes 10 numbers:

```
f0..f2   blame under P1, P2, P3
f3..f5   violation density per viewpoint
f6       how much evidence came from executable checks
f7       position in the trace
f8       how many viewpoints flagged it
f9       peak density
```

Score each with `theta . h`, softmax, take the highest. That is the prediction.

`theta` is 10 weights. That is the entire learned model.

> **Checkpoint.** With rules alone, this pipeline makes **zero model calls**. Verified: 16,325
> checks across 80 traces, all of them Python. The LLM is optional and enters later.

## 5. Where the LLM fits

Turn on the `evidence` flag and a model *writes additional constraints* for the specific trace,
on top of the five hand-written ones. Some are executable; some are natural-language rubrics that
need a judge call per step.

Crucially: **the model never picks the final step.** It enriches the evidence. `theta` still
ranks. In both configurations the chain above is what produces the answer.

This distinction matters because almost every accuracy number in this project is quoted without
it. **Always ask whether `evidence` was on.**

## 6. Training

The 10 weights start at `THETA0 = (0.85, 0.95, 0.90, 1.15, 1.35, 1.25, 0.85, 0.45, 0.90, 1.20)`,
hand-set. Training adjusts them.

### The constraint that shapes everything

A project rule: **gold labels must never be read in any decision, attribution, repair or RL
update path.** The system learns from what it can observe, not from the answer key.

So training cannot do the obvious thing (compare the prediction to the truth and minimise error).
Instead, per episode:

1. Rank candidates, pick one
2. Attempt a repair at that step
3. An *observable verifier* judges whether the repair worked
4. Reward in [-1, 1]
5. `theta += lr * (R - baseline) * (h_chosen - sum_j p_j h_j)`

That is REINFORCE. The reward is "did a repair here succeed", used as a proxy for "was this the
right step".

> **Checkpoint.** The thing being trained is the 10-weight ranker. The multi-agent system itself
> is frozen; it is the subject under observation, not the learner.

### How this differs from image classification

| | ImageNet | here |
|---|---|---|
| classes | fixed 1000 | this trace's ~4.5 candidates, different every time |
| supervision | the true label | a noisy proxy reward |
| loss | cross-entropy | REINFORCE, no loss function |
| training set | 1.2M | 113 episodes |
| parameters | 25M | **10** |

Ten parameters for 113 examples is a sensible ratio. The model is not too small. The limits are
the 10 hand-designed features and the candidate pool.

## 7. Where the project stands

### The ceiling

`theta` reorders candidates. It cannot pick a step that was never a candidate. So measure how
often the gold step is in the list at all:

| | top-1 | recall | mean pool |
|---|---|---|---|
| external tau | 13.8% | **41.4%** | 4.5 |
| external magentic | 20.5% | **20.5%** | 1.2 |

Reading this:

- **tau**: the answer is present 41.4% of the time and `theta` finds it 13.8%. Random choice in
  that pool is ~9.2%. So there are 27.6 points available to a better ranker.
- **magentic**: top-1, acc@3 and recall are identical because when gold is present it is the
  *only* candidate. **`theta` cannot change magentic by a single point.** magentic needs better
  constraints, not better ranking. These are different problems and must not be reported together.

### Training made it worse

One epoch, 138 episodes, synthesis off:

```
corpus-v2 test split, n=100
  untrained THETA0   44/100 = 44.00%
  trained            36/100 = 36.00%      -8.00 points
  McNemar p = 0.0078
```

Not a trade: the trained policy won **zero** cases that THETA0 lost. Only losses.

### Why, and why it matters

The epoch summary:

```
acceptance_rate         0.565     what the policy optimises
harness_recovery_rate   0.389     what actually recovered
gap                    +0.176
```

The observable verifier drifted 17.6 points above real recovery. The policy optimised the proxy
and got worse at the task. The weights show how: it cut `blame P1` by 82% and `temporal position`
by 42%, features that help attribution but apparently do not help pass the acceptance check.

**This is reward hacking, observed end to end with the instrumentation to prove it.** A proxy
reward diverged from the true objective, a policy chased the proxy, and measured performance fell
with statistical significance on held-out data. For a project about AI behaviour and safety, that
is a stronger result than a few points of accuracy.

### On the baseline comparison

The published agentrx figures, 31.03% tau and 18.18% magentic, were produced with
`gemini-3.1-flash-lite`. Run on a local `qwen2.5:32b`, agentrx returned `Inconclusive` on all
three traces tried so far.

So the often-quoted "31.03% versus 13.79%" compares a frontier model against **no model at all**.
It is not a like-for-like comparison and should not be presented as one. The matched-model run is
still incomplete.

## 8. Running it

```bash
# tests first: 343 should pass
python3 -m unittest discover -s tests -t .
python3 -m unittest discover -s benchmark/tests -t .

# attribution, no GPU needed, seeds only
python3 -m agentrx run --config configs/<cfg>.json --out runs/<name>
python3 watch_run.py runs/<name> --follow     # live accuracy while it runs

# training, resumable, prints a line per episode
python3 -m benchmark train --corpus benchmark/runs/corpus-v2 --out runs/train-v2 \
  --checkpoint runs/train-v2/rl_state_v2.json --backend ollama --model qwen2.5:32b \
  --num-ctx 32768 --epochs 1 --val-split calibration
```

The training line reads:

```
[ep  14/138]  11.6m  49.8s/ep  eta 102.9m | R +0.304 | accept 0.50 | |dtheta| 0.1736 | tr-0ecc...
```

`accept` is the proxy reward rate, `|dtheta|` is how far the weights have moved. If `accept`
climbs well above the gold recovery rate, that is the reward-hacking alarm and you should stop.

### Things that will bite you

- **GPU access.** The 32B model runs on a remote node reached by an SSH tunnel. A local Ollama can
  win port 11434 and answer instead. If you see "model not found", the wrong server replied.
- **`num_ctx` must be 32768** for agentrx. Its one-shot prompt overflows 16384 on long traces.
  That also caps `OLLAMA_NUM_PARALLEL` at 2 on a 45 GiB card.
- **The corpus cannot be regenerated identically.** Ollama's batching makes concurrent requests
  non-deterministic, so `temperature 0` does not give reproducibility. The committed files are the
  only record.

## 9. What to do next

**Measured and settled:** the corpus, the recall ceiling, the training degradation, the reward gap.

**Open, in priority order:**

1. **Does synthesis raise recall?** It is the only lever that can lift tau past 41.4%. About 87
   model calls per trace, so roughly 3 hours for all 29 tau traces.
2. **agentrx on a matched model, properly.** Three traces is not a result.
3. **The reward proxy.** The +0.176 gap is the core problem. Fixing it means a better observable
   verifier, since using gold in the update path is forbidden by design.
4. **magentic needs constraint work.** Pool size 1.2 means ranking is irrelevant there.

> **Final checkpoint.** If you take one thing: every accuracy number in this project must be quoted
> with its `evidence` flag, its dataset, and its n. Most of the confusion so far came from
> comparing numbers that were not comparable.
