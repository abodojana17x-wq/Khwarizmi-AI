# Khwarizmi AI: Fully Offline Hardware & Edge Deployment Blueprint
**Document Version:** 2.0 (Phase 0 Architecture Reset)  
**Date:** 2026-08-11  
**Status:** Implementation-Ready Deployment Specification  

---

## 1. Absolute Offline Operational Guarantee

A non-negotiable requirement of Khwarizmi AI is **100% Offline, Private, Local Inference**.

> **OFFLINE-FIRST MANDATE:**  
> The final deployment package must operate completely standalone without requiring:  
> * External API keys (OpenAI, Anthropic, Google, DeepSeek)  
> * Cloud computing or remote inference endpoints  
> * Wi-Fi, Ethernet, or cellular internet access  
> * Pretrained cloud embeddings or remote verification tools  
> 
> Everything required for inference—model weights, custom byte-fallback tokenizer, memory database, agent routing logic, and AST/DAG tools—must reside on local storage and execute inside local system memory.

---

## 2. Resource Targets & Model Scale Tiers

To ensure that model scaling is driven by empirical benchmark evidence rather than arbitrary parameter bloat, Khwarizmi AI defines four explicit hardware and model tiers:

```
+---------------------------------------------------------------------------------------------------------+
|                                  KHWARIZMI AI MODEL SCALE & RESOURCE TIERS                              |
+---------------------------------------------------------------------------------------------------------+
| [PROTOTYPE TIER]  50M - 150M Params   --> Purpose: Architecture ablation, fast Colab unit testing       |
|                                           Target RAM : < 300 MB | Min Tokens/sec (CPU) : 45 tok/s       |
|                                                                                                         |
| [SMALL TIER]      300M - 700M Params  --> Purpose: Serious architecture verification, reasoning research|
|                                           Target RAM : < 1.2 GB | Min Tokens/sec (CPU) : 30 tok/s       |
|                                                                                                         |
| [EDGE TIER]       1B - 3B Params      --> Purpose: Primary production release for laptops, Android, PCs |
|                                           Target RAM : < 2.5 GB | Min Tokens/sec (CPU) : 18 tok/s       |
|                                                                                                         |
| [ADVANCED TIER]   5B - 10B+ Params    --> Purpose: Only built if empirical scaling laws justify it      |
|                                           Target RAM : < 6.5 GB | Min Tokens/sec (GPU) : 25 tok/s       |
+---------------------------------------------------------------------------------------------------------+
```

### 2.1 Hardware Profile Specifications

| Specification | Low-Resource Edge (Android / PC) | Consumer Desktop / Laptop (CPU) | Consumer GPU Workstation |
| :--- | :--- | :--- | :--- |
| **Target Tier** | Small (700M) / Edge (2B) | Edge (2B–3B) | Edge (3B) / Advanced (7B) |
| **Target Quantization** | GGUF INT4 / INT5 | GGUF INT4 / INT8 | FP16 / INT8 |
| **Max Peak RAM / VRAM**| $\le 2.5\text{ GB}$ total RAM | $\le 4.0\text{ GB}$ total RAM | $\le 8.0\text{ GB}$ VRAM |
| **Minimum Processor** | ARM64 NEON (e.g., Snapdragon 8 Gen 2)| x86_64 CPU with AVX2 (Intel Core / Ryzen)| NVIDIA RTX 3060 / 4060 (8GB–12GB)|
| **Target Generation Speed**| $\ge 18\text{ tokens/sec}$ | $\ge 25\text{ tokens/sec}$ | $\ge 60\text{ tokens/sec}$ |
| **Max TTFT (1024 tokens)**| $\le 500\text{ ms}$ | $\le 300\text{ ms}$ | $\le 80\text{ ms}$ |
| **Offline Tools Executable**| `rafig/python_brain` (CPU AST) | All tools + DAG planner | All tools + DAG planner |

---

## 3. Hardware Optimization Strategies

### 3.1 SIMD Vectorization (x86_64 AVX2 / AVX-512 / ARM NEON)
Since KSC recurrent state updates involve matrix-vector multiplications ($q_t^T S_t$ and $k_t \otimes v_t^T$) rather than quadratic attention softmax over large KV tables, they are exceptionally well-suited for SIMD register vectorization:
* **AVX2 / NEON Micro-Kernels:** State matrix rows are tiled into 256-bit AVX2 or 128-bit NEON registers, allowing 8 consecutive FP32 multiply-add operations per CPU cycle without memory bus stalls.
* **Cache-Resident State:** Because each KSC head state matrix is compact ($d_k \times d_n = 64 \times 16$ FP32 floats = 4 KB), the entire recurrent state remains pinned in CPU L1/L2 cache during token decoding.

### 3.2 Low-RAM Memory-Mapped Runtime (`mmap`)
To deploy on devices with strictly limited RAM (<4 GB):
* Model weights are serialized in memory-mapped format (`.gguf` or `.ksc_map`).
* The operating system pages weight tensors into physical RAM on-demand, allowing a 2B parameter 4-bit model ($1.1\text{ GB}$ disk footprint) to run comfortably on a system with only $2\text{ GB}$ of free RAM without out-of-memory crashes.

### 3.3 Consumer GPU Offloading (VRAM Economy)
For workstations equipped with consumer GPUs (8 GB–16 GB VRAM):
* Models up to 7B parameters run entirely in FP16 or INT8 within VRAM.
* For MoE models (Phase 4), expert weights can be partitioned: active Top-2 experts are pinned in GPU VRAM, while inactive experts reside in host system RAM via pinned PCIe transfers.

---

## 4. GGUF & `llama.cpp` Compatibility Analysis

A critical research question is whether Khwarizmi AI should integrate with the industry-standard `llama.cpp` runtime and GGUF file format.

```
+---------------------------------------------------------------------------------------------------------+
|                                GGUF & LLAMA.CPP COMPATIBILITY EVALUATION                                |
+---------------------------------------------------------------------------------------------------------+
| 1. QUANTIZATION FORMAT COMPATIBILITY  --> FULLY COMPATIBLE (Q4_K_M, Q5_K_M, Q8_0 weights supported)     |
| 2. TOKENIZER COMPATIBILITY            --> FULLY COMPATIBLE (Byte-Fallback BPE/Unigram maps to GGUF)     |
| 3. CUSTOM OPERATOR SUPPORT (KSC)      --> REQUIRES CUSTOM GGML KERNEL EXTENSION                         |
|                                           (Standard llama.cpp lacks recurrent KSC matrix state ops)     |
| 4. ARCHITECTURAL DECISION             --> ADOPT GGUF DATA FORMAT + EXTEND GGML RUNTIME                  |
|                                           (Do not compromise KSC architecture just to avoid 500 lines   |
|                                            of C++ GGML operator code)                                   |
+---------------------------------------------------------------------------------------------------------+
```

### 4.1 Custom GGML Operator Extension for KSC
Standard `llama.cpp` natively supports standard Transformers, Mamba (SSM), and RWKV, but does not natively contain the Khwarizmi State Cell (KSC) recurrent operator.

**Design Decision:** We will **not** downgrade Khwarizmi to a standard attention transformer merely to use stock `llama.cpp`. Instead, in **Phase 12**, we will:
1. Export model weights and custom BPE tokenizer tables into standard `GGUF` format.
2. Provide a clean, standalone C++/GGML extension header (`ggml-khwarizmi.h` / `.cpp`) implementing the SIMD-optimized KSC recurrent state update micro-kernel.
3. Support pure-Python offline execution via NumPy/PyTorch SIMD bindings as a universal fallback for environments where compiling C++ extensions is restricted.

---

## 5. Deployment Verification Checklist

Before releasing any tier package, the engineering team must run:
```bash
# Verify offline 4-bit Small Tier deployment on local CPU
python -m khwarizmi.runtime.engine \
    --model-path=models/khwarizmi_small_4bit.gguf \
    --offline-mode=True \
    --test-prompt="Create a Python function to sort a list of dictionaries by key and explain structural complexity."
```
* **Success Gate:** Model must respond within $250\text{ ms}$ TTFT, generate $\ge 30\text{ tok/s}$, correctly invoke `Python Brain` AST check, and consume $\le 1.2\text{ GB}$ peak RAM.
