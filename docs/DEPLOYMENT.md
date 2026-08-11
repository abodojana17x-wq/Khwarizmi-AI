# Khwarizmi AI — Deployment & Offline Requirements

> Hard requirement: 100% offline inference. No API, cloud, Wi-Fi, or remote inference.

## 1. Resource tiers (targets, NOT commitments)

| Tier | Params | Purpose | Target hardware |
|---|---|---|---|
| Prototype | 50M–150M | arch experiments, ablations, unit tests | Colab CPU/GPU |
| Small | 300M–700M | serious reasoning experiments | Colab / consumer GPU |
| Edge | 1B–3B | real local deployment, low-resource | CPU / consumer GPU / edge |
| Advanced | 5B–10B+ | **only if benchmarks justify** | consumer/server GPU |

Scale only on evidence (Phase 16).

## 2. Offline packaging
Everything required for inference ships locally:
- Model weights (checkpoint)
- Tokenizer (trained, local)
- Runtime (inference engine)
- Memory system (local store)
- Agent logic + Local Tools (Project Planner, Python Analysis, Symbolic Verify, Safe Exec)

## 3. Runtime options
1. **Research/Colab:** PyTorch (bf16) — fastest to iterate.
2. **Edge/CPU:** custom lightweight C/CPU runtime (no Python required at serve time), or
   ONNX Runtime. KSC's recurrence is matrix–vector ops → very CPU-friendly.
3. **GGUF / llama.cpp:** investigate compatibility. **We will NOT force it if it damages
   KSC.** If KSC's operators (delta-rule matrix state, decoupled gates) aren't expressible in
   llama.cpp's operator set, we ship our own runtime. (See RESEARCH.md §2.9.)

## 4. Quantization
- Train bf16 → quantize INT8 (sym) and INT4 (RTN / GPTQ-style / AWQ if feasible).
- Use imatrix-style calibration on a small offline set.
- Validate benchmark quality within tolerance before releasing a quantized build.
- Q4_K_M-class balance recommended where applicable.

## 5. Hardware optimization
- CPU inference: AVX2/AVX-512 / ARM NEON; thread count = physical cores.
- Consumer GPU: layer offload to VRAM where available; keep KV-free design.
- Edge/Android: investigate MNN/ONNX + int4; expose a minimal local API.
- Energy: measure RAPL/powermetrics where available; report in benchmarks.

## 6. Verification of offline-ness
- CI check: inference path imports no `socket`/network libs; no outbound calls.
- Smoke test runs with network disabled and still completes.
