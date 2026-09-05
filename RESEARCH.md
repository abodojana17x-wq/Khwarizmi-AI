# Khwarizmi AI: Evidence-Led Research & Frontier Comparison

**Document version:** 3.0

**Reviewed:** 2026-09-05
**Scope:** Research and planning, not a claim of implemented capability or a new model release.

## 1. Evidence rules

- **[SOURCE REPORT]:** A result or feature described by its publisher. Not independently reproduced here.
- **[CODE OBSERVATION]:** A property visible in the inspected repository, with a file reference.
- **[HISTORICAL MEASUREMENT]:** A recorded run tied to its original code revision and settings, not today's test result.
- **[HYPOTHESIS]:** A proposed benefit that requires a controlled experiment.
- **[DECISION]:** A project planning choice, not a measured benefit.

Every quantitative claim needs a source, date, model/checkpoint, task version, conditions and limitations. An unsupported test is not a pass, a failed measurement is not proof of a failed architecture, and a working tensor interface is not evidence of learned reasoning.

## 2. What is publicly reported about GPT-6?

The official OpenAI announcement and system card were accessed on **2026-09-05**. They identify **GPT-6 Astra**, with the announced API name `gpt-6-astra` [S1, S2]. This review did not call the model, test account access, purchase inference, or independently reproduce its scores. Availability is described as a staged rollout; verify access and the exact snapshot before any comparison.

| Item | Published information [SOURCE REPORT, S1] | Implication for Khwarizmi |
| --- | --- | --- |
| Software engineering | Terminal-Bench 4.0: **57.9%**; DeepSWE v1.1: **74.1%** | Compete on verified task completion, not the number of reasoning modules. These different task suites cannot be pooled into a Khwarizmi score. |
| Long-running coding sessions | Codex can preserve notes and search earlier context windows | Memory alone is not a unique advantage. Test freshness, provenance, restart recovery and local privacy. The described feature belongs to the model-plus-Codex system. |
| Reasoning and science | GPQA Diamond: **96.0%**; FrontierMath Tier 4 (v2): **97.6%** | A small-model claim of universal superiority requires extraordinary independent evidence. |
| Long context | MRCR v2 8-needle, 512K–1M: **96.3%** | Fixed-size recurrent state is a resource property, not evidence of comparable recall. |
| API price | Standard: **$10 / million input tokens**, **$50 / million output tokens**, separate cache rates | Price is a dated published quote, not a project spend authorization or a complete per-task cost. |

**Comparison caveat:** S1 says scores are the maximum at any effort and may come from a research environment or API rather than production ChatGPT. Different harnesses, budgets, tools, safeguards and task versions matter. Do not compare these numbers directly to local synthetic tests or use them as Khwarizmi acceptance thresholds.

**Safety lesson [SOURCE REPORT, S2]:** The system card reports improved boundary-following and prompt-injection resistance, but also decreased reasoning monitorability in adversarial tests. It does not support “perfectly safe” claims. Khwarizmi should enforce permissions outside the model and audit actions and artifacts, not trust the model's own confidence or narrative.

**Unknowns:** This review does not establish GPT-6's parameter count, detailed proprietary architecture, training compute or full training data. Do not invent these to design a supposed architectural countermeasure.

## 3. Project evidence and the actual gap

| Observation | Evidence | Consequence |
| --- | --- | --- |
| KSC, memory, MoE and adaptive-compute implementations exist | `khwarizmi/core/`, `memory/`, `experts/`, `reasoning/` | Useful research baseline; not a trained frontier competitor. |
| The neural agent's test encoder clamps character code points to the vocabulary limit | `khwarizmi/agent/agent_loop.py`, `encode_prompt_to_ids` | Distinct Arabic characters can collide for small vocabularies. Implement reversible encoding before claiming Arabic neural understanding. |
| The agent returns logits and diagnostics | `khwarizmi/agent/agent_loop.py`, `AgentResponseFrame` | Define a trained checkpoint, tokenizer and decoded generation path before calling this a usable assistant. |
| The current restricted Python executor is not an OS security boundary | `khwarizmi/coding/execution_sandbox.py`: same-process execution; non-allowlisted imports can fall through | Do not run untrusted generated programs until independent process/OS isolation is reviewed and tested. |
| Historic memory audit reports UPDATE absent from the main model path | `runs/reality_check/6ef7273377f3.json`, revision `e27dd766` | Reproduce on the current revision and test runtime integration before fixing or declaring it solved. |
| The same archived run has inconsistent summary counts and per-experiment statuses | `runs/reality_check/6ef7273377f3_summary.md` | Preserve raw history, but invalidate it as release certification; repair and rerun the evaluator. |
| Historical ARRC and MoE timings do not demonstrate a speed win | Same run: adaptive 16.828 ms vs fixed 6.1841 ms; sparse 1.1594 ms vs dense 0.2977 ms | Configuration-specific observations only. Compare trained models at matched quality, report uncertainty, and profile actual work. |

No comparable real-task GPT-6 result is established by the inspected artifacts. PyTorch is absent from the review environment; neural tests and benchmarks were **not rerun** during this documentation review. Historical test counts must not be presented as a current green test suite.

## 4. Corrected engineering principles

These replace the previous document's unsupported universal performance claims.

1. **Sequence length:** Standard dense self-attention has quadratic attention work in sequence length; autoregressive KV storage generally grows linearly with retained tokens. Optimized attention kernels, grouped-query attention, caching and windowing change practical memory and cost. A recurrent decoder's fixed-size state does not make training activations, input buffering, output logits or project storage constant-memory.
2. **Stability:** Bounded retention coefficients alone do not prove global stability of nonlinear blocks, unbounded inputs, learned projections, or all floating-point precisions. Validate state norms, gradients, NaN/Inf, long-sequence behavior and generation quality separately.
3. **Adaptive computation:** Extra inference compute may help specific trained models and tasks; it is not a general proof that inference scaling beats pretraining. Logical halting counts are not measured FLOPs or wall-clock savings.
4. **Sparse experts:** Top-k activation can reduce active expert computation, but all resident weights and routing overhead still count. Report both total and active parameters and separate matched-memory from matched-compute comparisons.
5. **Verification:** AST validity catches syntax/structure errors, not behavioral correctness. Hidden tests, property tests and independent specifications are required; model-generated tests alone can share the model's mistake [S3].
6. **Quantization:** GGUF is a file format, not automatic runtime support for arbitrary custom recurrent operators. KSC needs verified operator support or a dedicated backend before export promises [S4]. Quantization quality loss is an experiment, not assumed negligible.
7. **Memory:** Retrieval stores do not inherently outperform long contexts. Evaluate naive recent-context, text search, structured memory and learned memory under the same tasks and retrieval budgets.
8. **Confidence:** An uncalibrated neural head or heuristic score is not a probability of correctness. Measure calibration on held-out outcomes and evaluate abstention with coverage.

## 5. Minimal hypothesis queue

**[DECISION]** Keep existing components as controls, not as protected winners. Run at most two research experiments concurrently. Spend no scale-up budget before baseline data and evaluation are sound.

| ID | Hypothesis | Smallest controlled comparison | Promotion requirement (proposed, not achieved) |
| --- | --- | --- | --- |
| H1 | Evidence-backed project memory improves long-session outcomes | Same checkpoint and tasks: recent context vs text search vs provenance-tagged memory | Lower confidence bound of paired success improvement above zero; no stale-constraint or privacy regression. |
| H2 | Verification-guided repair improves code correctness | Same checkpoint: one attempt vs bounded test-and-repair; separately report a matched total-budget control | Better hidden-test success within the declared time cap; evaluator tests never exposed to the repair agent. |
| H3 | KSC gives a better quality/resource trade-off | Small KSC vs small dense Transformer, same tokenizer, data and training-token budget; additionally report training FLOPs and wall time | Comparable held-out quality with a measured memory or latency gain, across at least 3 training seeds. |
| H4 | ARRC spends compute where it helps | Trained checkpoint: fixed 1, fixed max and adaptive cycles | A measured quality/latency Pareto gain, not just varying halt probabilities. |
| H5 | Sparse MoE helps on the intended device | Dense vs MoE: one matched-total-parameter study and one matched-active-compute study | A quality/resource gain that survives real peak-RAM and p95 latency accounting. |

**Deferred:** Multi-agent swarms, automatic architecture invention, world models, continual online weight updates, multimodal expansion and a proliferation of specialist memories. Reopen one only when a measured failure justifies it. Specialist safety/alignment constraints still apply; deferring a domain does not remove safety review.

## 6. Source register

All sources below were read on **2026-09-05**. External content can change; freeze the benchmark versions and archive permitted source metadata before a formal experiment.

- **[S1] OpenAI — GPT-6 Astra: A new generation of intelligence.** https://openai.com/index/gpt-6-astra/ — first-party announcement, reported scores, harness caveats, API identifier and pricing. Not independent validation.
- **[S2] OpenAI — GPT-6 Astra System Card.** https://deploymentsafety.openai.com/gpt-6-astra — reviewed safety overview and relevant safety/alignment sections; not a claim to have reproduced the full system-card evaluation.
- **[S3] EvalPlus.** https://evalplus.github.io/ — project description of HumanEval+/MBPP+ expanded test suites and repository/code-efficiency evaluation. Use versioned artifacts and check licenses before ingestion.
- **[S4] ggml-org — llama.cpp.** https://github.com/ggml-org/llama.cpp — local inference, quantization and backend documentation. Does not establish compatibility of Khwarizmi's custom KSC implementation.

Execution priorities are in [ROADMAP.md](./ROADMAP.md); comparison rules are in [BENCHMARKS.md](./BENCHMARKS.md). Research references do not authorize online runtime dependencies, model downloads, teacher-data collection or paid API usage.
