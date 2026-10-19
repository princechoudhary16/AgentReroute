# What happens during an Ollama AgentRx run

```bash
python3 -m agentrx run \
  --config configs/dev_ollama.json \
  --model "hf.co/noctrex/Huihui-Qwen3-VL-8B-Instruct-abliterated-GGUF:Q4_K_M" \
  --out runs/ollama-my-test
```

This command investigates an existing saved agent run. It does not launch the retail
agent, replay its API actions, or activate the prevention middleware.

1. **Read the configuration.** Use the Ollama provider, a 16,384-token context window,
   temperature0, one repetition, and the first available retail case. The explicit
   model argument overrides the default model name while preserving those settings.
2. **Load the evidence.** Read `data/benchmark/`, verify its corpus hash, and select
   the saved conversation/tool history. Load the retail policy and descriptive tool
   definitions. The failure annotation is kept out of model prompts.
3. **Check and identify the local model.** Contact local Ollama, resolve the installed
   weights and server version, and write experiment metadata. Record config, corpus,
   code and prompt identities so incompatible runs cannot silently share results.
4. **Run the baseline.** Ask the model to inspect the full trajectory and choose the
   causally important unrecovered failure, responsible agent and failure category,
   without generated constraint evidence or the detailed diagnostic checklist.
5. **Generate global AgentRx checks.** Ask the same model for general checks grounded
   in the policy and tool descriptions. These checks concern the saved trace; they
   are not pre-execution middleware rules.
6. **Generate dynamic checks step by step.** At each recorded step, supply only the
   observed prefix through that step and ask for relevant additional checks. This
   is the config's `stepwise` mode. Global checks are also considered at each step.
7. **Evaluate applicable checks.** A trigger determines whether a check applies at
   this step. Python checks execute in a separate restricted process with a timeout;
   semantic checks ask the model for rubric judgments. Definite language failures
   become violations; unclear judgments are recorded without declaring a violation.
   Outcomes are SAT (no violation found), VIOL, SKIP (inapplicable), or ERROR.
8. **Make the AgentRx diagnosis.** Give the model the full trace, detected violations,
   checker-error summaries and taxonomy checklist. Ask it to identify the earliest
   unrecovered failure that explains the outcome. A mistake followed by successful
   recovery should not be selected merely because it occurred first. Validate that
   the predicted step, agent, category and evidence references are well formed.
9. **Score both diagnoses.** Now compare predictions with the independently stored
   benchmark annotation. Calculate step/category/agent agreement, step distance and
   related reported metrics. These comparisons evaluate the model; they do not fix
   its answer or feed the annotation back into diagnosis.
10. **Write reports and caches.** Save all job artifacts and cached model responses,
    then generate JSON summaries and CSV/Markdown tables under the output folder.

Key outputs:

- `experiment.json`: configuration, selected case and reproducibility identities.
- `jobs/baseline/...` and `jobs/agentrx/...`: trajectory, constraints, checks,
  `validation_log.json`, and `result.json` containing the diagnosis.
- `requests/`: cached model responses and usage/telemetry.
- `summary.json`, `predictions.csv`, `confusion_matrices.json`, and
  `table4.{md,csv}` / `table7.{md,csv}`: evaluation reports.

With N recorded steps, the configuration needs approximately N+3 model calls before
any applicable semantic checks: one baseline, one global synthesis, N dynamic
syntheses and one final AgentRx judgment. Prefixes grow as the run progresses, so
later calls can take longer. Context overflow produces an explicit error.

Rerunning with the same code, data, model weights and settings can resume completed
jobs and reuse cached responses. A changed identity requires a new output folder.
One trace and one repetition form an engineering pilot, not the full paper benchmark.
Previous local pilots completed but generated checks were all skipped, so pipeline
execution alone does not establish useful evidence or reliable diagnostic accuracy.
