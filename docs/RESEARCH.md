# Khwarizmi AI — Research Comparison

> Every claim is tagged: **[FACT]** established result · **[RF]** research finding from
> cited literature · **[HYP]** hypothesis (untested by us) · **[DD]** Khwarizmi design
> decision. We never present an untested hypothesis as fact.

## 1. Techniques reviewed

| Technique | Source | Class | Key idea |
|---|---|---|---|
| Softmax Attention | Vaswani et al. 2017 | Quadratic | Pairwise token mixing, unbounded KV cache |
| Mamba / Mamba-2 (SSD) | Gu & Dao 2023; Dao & Gu 2024 | SSM | Selective state space; structured state-space duality |
| Mamba-3 | Lahoti et al. 2026 (preprint) | SSM | Improved SSM sequence modeling |
| RWKV-4/5/6/7 | Peng et al. 2023+ | Linear RNN | Channel-wise decay, time mixing |
| xLSTM | Beck et al. NeurIPS 2024 | RNN | sLSTM/mLSTM, exponential gating, memory mixing |
| DeltaNet / Gated DeltaNet | Yang et al. ICLR 2025 | Linear attn | Delta-rule error-correcting memory |
| Gated DeltaNet-2 | 2026 preprint | Linear attn | Decoupled erase/write gates |
| Titans | Behrouz et al. 2025 (Google) | Hybrid memory | Neural long-term memory, test-time learning |
| Liquid Neural Nets | Hasani et al. | ODE/RNN | Continuous-time, adaptive dynamics |
| Sparse MoE | Shazeer 2017; Fedus 2021 (Switch) | Routing | Top-k expert FFN, load-balancing loss |
| Adaptive compute / Early-exit | 2024–2026 literature | Inference | Confidence-gated depth, learned halting |
| Quantization / GGUF | llama.cpp / ggml | Deployment | INT4–INT8, imatrix, CPU inference |

## 2. Per-technique analysis

### 2.1 Mamba / Mamba-2 (SSD)
- **[FACT]** Mamba-2 reformulates the selective SSM as structured state-space duality (SSD),
  a form of causal linear attention with **diagonal** `Aₜ`, enabling 2–8× faster training via
  tensor cores while staying competitive with Transformers (Dao & Gu 2024).
- **[RF]** Mamba-2 2.7B / 300B tokens beats Pythia-2.8B and Pythia-6.9B on downstream evals
  (marktechpost summary of the paper).
- **[RF]** Linear/recurrent models have a fixed-size state, so for `T > state size` they can
  lose information that attention retains (HackerNews discussion of Mamba-2; visarga).
- **[DD]** Khwarizmi borrows *selective decay* but uses a **matrix** associative memory + delta
  write, not a vector SSM state. We expect better recall at equal state size. **To validate.**

### 2.2 RWKV
- **[FACT]** RWKV is an RNN formulated as linear attention; O(T·d) time, O(d) space at
  inference; trained to 14B params (Peng et al. 2023).
- **[RF]** Linear attention limits precise recall of minutiae over very long contexts
  (funneling through one vector) — a known RWKV limitation.
- **[DD]** We adopt the *channel-wise decay* idea but combine it with a delta-rule matrix
  memory to overcome the single-vector funnel weakness.

### 2.3 xLSTM
- **[RF]** xLSTM[1:1] (mLSTM memory mixing) outperforms Mamba, RWKV-5/6 on associative
  recall at equal blocks; enhanced memory capacity (Beck et al. 2024).
- **[DD]** Confirms that **matrix-valued memory** helps recall — supports our KSC matrix state.
  We do not use LSTM gating directly; we use delta-rule + surprise gating.

### 2.4 DeltaNet / Gated DeltaNet / Gated DeltaNet-2
- **[FACT]** Delta rule `S += β k⊗(v − Sᵀk)` is error-correcting and improves retrieval over
  additive memory (DeltaNet; Gated DeltaNet, ICLR 2025).
- **[FACT]** Gated DeltaNet adds scalar decay `gₜ` and write rate `βₜ` (used in Qwen3.5 /
  Qwen3-Next linear attention). State = matrix `S ∈ R^{d_k×d_v}`, constant memory.
- **[FACT]** Gated DeltaNet-2 (2026) **decouples erase and write** with channel-wise gates
  `bₜ` (key-side erase) and `wₜ` (value-side write), reducing interference in compressed
  memory; reduces to KDA/Gated DeltaNet as special cases (arXiv 2605.22791).
- **[DD]** KSC **adopts** the delta-rule matrix memory and the decoupled erase/write idea, and
  **extends** it with a Titans-style surprise gate `σₜ`. This is our distinct contribution.

### 2.5 Titans (neural long-term memory)
- **[FACT]** Titans introduces a deep neural **long-term memory module** that learns to
  memorize/forget at *test time* via online gradient descent; an event "more memorable" if it
  violates expectations (surprise) (Behrouz et al. 2025, arXiv 2501.00663).
- **[FACT]** Three variants: Memory-as-Context, Memory-as-Gating, Memory-as-Layer. Scales to
  >2M token context; beats Transformers and recent recurrent models on long-context / BABILong.
- **[RF]** Deeper memory improves perplexity but adds compute (efficiency/expressiveness trade-off).
- **[DD]** Khwarizmi adopts the **surprise/importance gating** principle for *what to write*
  into both the recurrent state and the external long-term memory. We do **not** copy the
  test-time weight update; our write is a closed-form delta update (CPU-friendly).

### 2.6 Liquid Neural Networks
- **[RF]** Liquid nets use continuous-time ODE dynamics with adaptive time constants;
  strong in dynamical/control settings and interpretability, but not demonstrated as
  general-purpose language models at scale.
- **[DD]** **REJECT for the core language model.** The ODE formulation is expensive and not
  competitive for LLMing. We note it only as inspiration for *adaptive dynamics* (our
  data-dependent decay is a discrete analogue).

### 2.7 Sparse MoE
- **[FACT]** Switch Transformer routes each token to 1 expert; up to 7× pretraining speedup at
  equal compute; needs auxiliary load-balancing loss + expert dropout for stability (Fedus 2021).
- **[RF]** MoE increases *parameter* count at fixed *active* compute, improving quality-per-FLOP
  when routing is balanced.
- **[HYP]** MoE helps Khwarizmi's coding/reasoning/planning specialization.
- **[DD]** Include MoE **only as an ablation-gated option**. Remove if Phase 4 shows
  insufficient benefit or instability.

### 2.8 Adaptive compute / Early-exit / Learned halting
- **[FACT]** Early-exit LLMs add intermediate exit heads; terminate on confidence; Pareto
  trade-off between speed and accuracy (Early-Exit LLMs survey, 2026).
- **[FACT]** Training-free methods (DEER, CaR) prune low-utility reasoning and halt when
  confidence is high, cutting latency with accuracy preserved (DEER 2025; CaR EMNLP-Ind 2025).
- **[DD]** Khwarizmi builds adaptive compute **into the architecture** (early-exit heads +
  recurrent reasoning loops + confidence halt), not only as a post-hoc trick.

### 2.9 Quantization / GGUF / llama.cpp
- **[FACT]** GGUF + llama.cpp gives portable CPU/Apple-Silicon/GPU inference; Q4_K_M is the
  recommended quality/size balance; imatrix improves low-bit quality (llama.cpp docs).
- **[FACT]** 7B Q4_K_M needs ~6GB RAM minimum, ~16GB recommended; build flags
  `GGML_CUDA`/`GGML_METAL` enable acceleration.
- **[DD]** We target offline CPU/edge deployment. **We will not force llama.cpp compatibility
  if it damages KSC.** Plan: (a) PyTorch runtime for research; (b) a custom lightweight
  C/CPU runtime for edge; (c) export to ONNX and, *if feasible without harm*, GGUF. If KSC
  cannot be expressed in llama.cpp's operator set, we ship our own runtime. Documented in
  `DEPLOYMENT.md`.

### 2.10 Frontier systems (GPT-class / Claude-class / Kimi-family) — principles only
- **[RF]** Publicly, frontier reasoning models use large-scale pretraining + RLVR / test-time
  compute (budget forcing, long CoT) + verifier models (e.g., DeepSeek-R1 style, Kimi
  long-CoT). They rely on massive compute and (at inference) cloud scale.
- **[DD]** We **extract principles only**: (1) test-time compute improves hard reasoning; (2)
  a verifier improves reliability; (3) decomposition + self-check helps. We implement these
  *offline and small*: recurrent reasoning loops + local deterministic verifiers, not cloud
  scale. We do **not** copy their architectures or require their infrastructure.

## 3. Adoption matrix (Khwarizmi should adopt?)

| Technique | Adopt? | Role in Khwarizmi | Confidence |
|---|---|---|---|
| Selective decay (Mamba/RWKV) | **Yes** | KSC decay gate `λₜ` | High (RF) |
| Delta-rule matrix memory | **Yes** | KSC core state `S` | High (RF) |
| Decoupled erase/write (GDN-2) | **Yes** | KSC `εₜ` vs `βₜ` | High (RF) |
| Surprise gating (Titans) | **Yes** | KSC `σₜ` + LTM writes | Medium (HYP) |
| Local conv (GDN/Qwen) | **Yes** | short-range mixing | High (RF) |
| MoE | **Conditional** | optional expert FFN | Low–Med (HYP) |
| Early-exit / learned halting | **Yes** | adaptive compute | High (RF) |
| Test-time reasoning loops | **Yes** | neural reasoning | Med (HYP) |
| Liquid ODE dynamics | **No** | — | Med (RF) |
| Unbounded attention | **No** (core) | only tiny local window if needed | High (DD) |
| Cloud/distributed inference | **No** | offline hard requirement | — |

## 4. Open research questions (to resolve empirically)

1. Is the surprise gate `σₜ` worth its parameters vs. a fixed write rate? (ablation)
2. Does decoupled erase `εₜ` beat a tied scalar gate at our scale? (ablation)
3. At what model size does KSC surpass an equal-cost Transformer/Mamba on reasoning?
4. Does MoE help at 700M, or only at 5B+? (ablation)
5. How much does adaptive compute save on real workloads without hurting hard tasks?
6. What is the optimal long-term-memory write policy to avoid forgetting project state?

These are answered in Phases 2–11, not asserted here.
