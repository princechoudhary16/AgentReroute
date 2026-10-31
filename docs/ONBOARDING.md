# AgentReroute, from scratch

For a person who has never seen this project. It builds up from the problem, explains each piece as
it becomes necessary, and is honest about the parts that did not work, because those are where most
of the learning is. No prior knowledge of the codebase is assumed.

## 1. The problem

An AI agent is given a job: cancel a customer's order, find a fact on the web, process a return. It
works through the job in steps, calling tools and reading their output. Sometimes it fails.

When it fails, you want to know **which step caused it**. That is harder than it sounds, because
the step where the failure becomes visible is almost never the step that caused it. An agent that
misreads an order status at step 21 may not visibly go wrong until it tries to cancel at step 33.
Every step in between inherits the mistake and looks broken too.

This is called **fault attribution**. The task: read the trace, name the step and the agent
responsible. You are scored on exact step match. Naming step 22 when the answer is 21 scores zero.

## 2. The method we started from

A published system called **AgentRx** does this in the most direct way available. It gives one
language model the entire trajectory and asks: which step is the decisive failure? One call, one
answer.

It scores 31.03% on tau (a retail customer-service benchmark) and 18.18% on magentic (web
browsing). Those numbers are the bar. The first thing we did was reproduce them, and we land on
31.0% and 18.2%, which is how we know our measurement apparatus is honest.

Why look for something better? A single verdict is opaque. It tells you *where*, never *which
obligation was broken*. Nothing downstream can act on it.

## 3. Our idea: three lenses instead of one verdict

Rather than one judge, use three reviewers, each responsible for a different kind of failure:

- **Intent and plan**: did the step follow the user's request and the domain policy, and did it
  make real progress?
- **Tool execution**: was the call well formed, were required arguments present, did it succeed?
- **Handoff and grounding**: was every claim traceable to something actually observed?

Each reviewer checks **constraints**: concrete obligations that can be settled against the trace.
"The order id passed to `cancel_pending_order` must appear in an earlier message." Those are
checkable in Python, deterministically, with no model call at all.

Crucially, a check only ever sees the steps up to the one being judged. Letting it see the future
would be cheating, and the code enforces it.

When a check fails, you get a **violation**: this step, this agent, this broken obligation, this
evidence. That structured record is the thing a single judge never produces.

## 4. From violations to an answer

Violations are evidence, not an answer. Several steps usually carry some.

**The blame matrix** aggregates them: for each lens and each agent, what fraction of that agent's
checks failed. **Candidates** are then built, one per (agent, step) with a violation, each
described by ten numbers: how dense its violations are, how many lenses agree, how early it sits.

Then something picks one. The original design learned ten weights by reinforcement learning,
scoring each candidate and taking the highest.

That is where the project got interesting, because it did not work.

## 5. What went wrong, three times

### Training the weights made it worse

On a held-out set of 100 traces, untrained weights scored 44% and trained weights scored 36%. A
paired statistical test gives p = 0.0078, with eight traces broken and none fixed, so this is not
bad luck.

The cause was the reward. We rewarded the policy when, after repairing the blamed step, the number
of violations went down. That sounds reasonable and is not: some steps are simply easier to repair
than others, regardless of whether they caused anything. The policy learned to find those. We
measured the gap between the reward and real recovery at 17.6 points.

**The lesson**: a reward you can satisfy without solving the task will be satisfied without solving
the task. We stopped training and kept the untrained weights.

### Repairing the top three changed nothing

If the ranking is unreliable, test it. Take the three most blamed steps, try repairing each, and
answer with whichever repair holds up.

Accuracy before: 13.8%. After: 13.8%. It moved six answers, and all six moved to another wrong
step. In the five traces where the correct step was sitting at rank two or three, it picked that
rank zero times.

The reason is the same shape as before. Ask a model to rewrite any step and it will produce one
that satisfies more constraints. "Violations went down" is not evidence that this step was the
fault.

### The diagnosis

Two failures in, we stopped guessing and split the results by a single question: **was the correct
step even among the candidates?**

| | correct step in the pool | not in the pool |
|---|---|---|
| tau | 70% correct | 16% correct |
| magentic | 89% correct | 9% correct |

The decision stage was never the problem. Give it the right option and it finds it most of the
time. It simply was not being given the right option: the correct step was in the pool only 41.4%
of the time on tau and 20.5% on magentic.

Everything we had tried improved **ranking**. The binding constraint was **recall**. Three ideas
aimed at the wrong stage of the pipeline.

## 6. Why recall was so low

Something structural, and easy to miss.

Our training corpus is multi-agent: a planner routes work to specialist agents. The published tau
benchmark is a **single assistant** calling tools. Two of our three lenses ask about routing and
handoff, and on a single-agent trace those questions have no content.

The numbers are stark. The routing check fires 268 times on our corpus and **zero** times on the
benchmark. The handoff check, 104 against 8.

So on published tau we effectively run on two checks, and **every candidate carries exactly one
violation, 130 out of 130**. Nothing is ever corroborated. The features our policy ranks on, like
"how many lenses agree", are constants. That explains the training failure too: the weights were
fitted where evidence is corroborated and applied where it never is.

## 7. What we built instead

Two changes, in order.

**Put a model in the decision path.** Replace the ten weights with one call that receives the
trace, the blame matrix, and the candidates with the exact obligations each one broke. It is told
explicitly that the checks are machine generated, that false positives are common, and that the
true fault often raised no violation at all. It may name any step, not only a candidate.

On tau: 13.8% to 34.5%. Across both benchmarks this beats the learned ranker on nine traces and
loses on one, p = 0.0215. It is the one statistically solid result in the project.

Letting it range over the whole trace matters: confined to the candidate pool it scores 24.1%,
free it scores 34.5%. The pool was a ceiling.

**Then attack recall.** Let each lens write constraints for the trace in front of it instead of
only using fixed ones. On tau, pool recall rises 41.4% to 48.3% and accuracy reaches **37.9%**,
past the 31.0% baseline. On magentic the same change nearly doubles recall but inflates the pool to
twelve candidates, and the extra options distract more than they help.

## 8. Where it ends up

| | AgentRx | ours |
|---|---|---|
| tau, 29 traces | 31.0% | **37.9%** |
| magentic, 44 traces | 18.2% | **25.0%** |

Ahead on both. Read the next sentence carefully, because it is the one a careless reader gets
wrong: **neither difference is statistically significant.** With 29 traces, one trace is 3.4
points, and a four-to-two win gives p = 0.69. The honest claim is parity-to-ahead at about a third
of the baseline's model calls, plus one solid internal ablation.

One cell is open. The magentic synthesis run was destroyed first by rate limiting and then by a
spend cap, so that number is unmeasured rather than bad.

## 9. Running it yourself

```
python3 -m agentrx.cli test                       # 212 tests, about 40 seconds
python3 analyze_run.py results/final-evidence.csv # every reported table, from committed evidence
```

Reproducing a run needs a `GEMINI_API_KEY` in your environment, never in a config file:

```
python3 -m agentrx.cli run --corpus data/benchmark --config configs/gemini_final.json --out runs/final --preflight
python3 -m agentrx.cli run --corpus data/benchmark --config configs/gemini_final.json --out runs/final
python3 export_evidence.py runs/final results/final-evidence.csv
```

`--preflight` checks the corpus, the config and your credentials without spending anything.

## 10. How to read the repository

| path | what it holds |
|---|---|
| `agentrx/perspectives.py` | the three lenses, the seed constraints, the blame matrix |
| `agentrx/selector.py` | the model call that decides, and its fallbacks |
| `agentrx/rl_policy.py` | the linear policy, the repair loop, the verifier |
| `agentrx/pipeline.py` | `diagnose_mp_rl`, which wires the stages together |
| `benchmark/` | corpus generation, the fault injector, scoring |
| `results/` | the committed evidence and what is and is not reportable |
| `configs/` | one config per reported result |
| `docs/AGENT-BRIEF.md` | the dense version, for an AI assistant working on the code |

## 11. A few terms

- **Trace / trajectory**: the full record of an agent run, step by step.
- **Step**: one turn, usually an agent message plus any tool call and its output.
- **Gold**: the ground-truth answer. Never visible to any decision the system makes.
- **Constraint / check**: an obligation evaluated at a step. A **violation** is a failed one.
- **Candidate**: a (step, agent) pair carrying at least one violation.
- **Pool recall**: how often the correct step is among the candidates. The binding constraint.
- **McNemar test**: compares two methods on the same traces by counting where they disagree. The
  right test here, because both methods see identical inputs.
