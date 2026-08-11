# Khwarizmi AI — Training Strategy

> Optimize for limited resources. No frontier-scale compute assumed. Teacher-student
> distillation + high-quality synthetic data where useful.

## 1. Architecture validation (before scaling)
- Phase 1–2: prototype 50M; ablations vs baselines. No commitment to scale until KSC wins.

## 2. Pretraining
- **Objective:** next-token cross-entropy on mixed EN/AR/code/math corpus.
- **Optimizer:** AdamW / Sophia; cosine schedule; warmup. bf16.
- **Budget:** start small (tens of GB tokens for 50M; scale tokens with params). Colab-friendly.
- **Efficiency:** chunk-parallel KSC training (SSD/Gated-DeltaNet-style scan), gradient
  checkpointing, optional FSDP/DeepSpeed at larger tiers.

## 3. Instruction tuning
- High-quality EN/AR instruction pairs; small, curated, contamination-controlled.

## 4. Reasoning training
- Distillation from a strong teacher (offline-generated traces, *not* a live API) for
  decomposition + self-check + revision. Verify traces with local tools.

## 5. Coding training
- Code corpora + execution-verified examples (Python Brain + Safe Exec as verifier).

## 6. Project-management training
- Synthesize goal→plan→task-graph data; train the core to emit intent/goals that the
  Project Planner tool expands. (Planner itself stays symbolic/deterministic.)

## 7. Memory training
- Train the memory controller's surprise gate + read/write policy on recall/retention tasks.

## 8. Tool-use training
- Train the router + agent to emit tool-call decisions; reward when a tool actually helps
  (verified by local verification).

## 9. Verification training
- Train selective verification: emit "verify" only when it improves final correctness.

## 10. Distillation
- Teacher (larger KSC or a permissively-licensed local model used offline) → student (small
  KSC) via response/embedding distillation. No cloud APIs.

## 11. Quantization
- Post-train INT8 (sym) and INT4 (RTN/GPTQ-style or AWQ if feasible). Use imatrix-style
  calibration on a small offline set. Validate quality within tolerance before shipping.

## 12. Local deployment
- Export to the runtime defined in `DEPLOYMENT.md`; verify offline + RAM/latency budgets.

## Cost discipline
- Every training run is config-driven and resumable; logged in `experiments/`.
- Prefer many small ablations over one big run. Stop-and-fix on benchmark failure.
