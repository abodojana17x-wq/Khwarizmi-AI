# Khwarizmi AI — Risks & Failure Modes

## 1. Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | KSC underperforms baselines at our scale | Med | High | Mandatory ablations; redesign or fall back to Mamba-2/DeltaNet if proven better |
| R2 | Recurrent models forget long dependencies | Med | High | Surprise gate + LTM; validate retention benchmark |
| R3 | MoE training instability / collapse | Med | Med | Aux loss + expert dropout; remove if Phase 4 fails |
| R4 | Adaptive compute hurts hard tasks | Med | High | Confidence-calibrated halt; threshold sweep; fallback fixed |
| R5 | Quantization destroys reasoning quality | Med | High | Calibrate with imatrix; validate evals post-quant |
| R6 | Offline data scarcity (esp. Arabic/Egyptian reasoning) | High | Med | Synthetic + verification; quality>quantity |
| R7 | Scope creep / building everything at once | High | High | Phase gates; stop on benchmark failure |
| R8 | Tool layer latency dominates | Med | Med | Router-gated; only invoke when beneficial |
| R9 | Llama.cpp/GGUF incompatible with KSC | Med | Low | Custom C/CPU runtime; don't force compatibility |
| R10 | Hidden CoT leaks into responses | Low | Med | Synthesis pass separates internal state from output |
| R11 | Preserving old code for sentiment | Med | Med | Audit decisions enforce value-only survival |

## 2. Failure Modes (how the system can fail, and how we detect)

1. **Silent quality regression** — caught by per-phase benchmarks + 3-seed means.
2. **Memory overflow / forgetting project state** — caught by long-project consistency eval.
3. **Router thrashing** (flip-flopping paths) — log route distribution; add stability penalty.
4. **Verification skipped on buggy code** — measure verify-coverage vs correctness.
5. **Tool misuse** (invoking expensive tool needlessly) — track tool cost vs benefit reward.
6. **Training divergence** — NaN/Inf guards; fallback lr; abort + diagnose.
7. **Offline leak** (accidental network call) — CI lint forbids network imports at inference.
8. **Overfitting to benchmarks** — held-out sets; contamination checks in `DATA.md`.

## 3. Guardrails (non-negotiable)
- No network calls in the inference path (CI-enforced).
- Every phase has explicit failure + exit criteria (ROADMAP.md).
- Components removed/redesigned on evidence, not opinion.
- No large training until Phase 1–2 validate the math and the 50M prototype.
