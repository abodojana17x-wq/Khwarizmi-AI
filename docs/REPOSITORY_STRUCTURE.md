# Khwarizmi AI — Future Repository Structure

> Clean separation of concerns. No unnecessary folders. Modular interfaces.

```
Khwarizmi-AI/
├── README.md                      # points to docs/BLUEPRINT.md
├── BLUEPRINT.md                   # Phase 0 consolidated deliverable (16 items)
├── main.py                        # offline agent CLI entry (refactored)
├── requirements.txt               # runtime/tool layer (stdlib + optional)
├── requirements-dev.txt           # training stack (torch, tokenizers, ...)
├── docs/                         # all planning docs (this set)
│   ├── ARCHITECTURE.md  RESEARCH.md  AUDIT.md  MEMORY.md
│   ├── ROADMAP.md  EXPERIMENTS.md  BENCHMARKS.md
│   ├── TRAINING.md  DATA.md  REPOSITORY_STRUCTURE.md
│   ├── RISKS.md  DEPLOYMENT.md  CONTRIBUTING.md
├── khwarizmi/
│   ├── config.py                 # KEEP (refactored settings)
│   ├── common/paths.py           # KEEP
│   ├── neural/                   # THE CLEAN NEURAL CORE
│   │   ├── ksc.py                # Khwarizmi State Cell (Phase 1)
│   │   ├── router.py             # Cognitive Router
│   │   ├── experts.py            # Sparse MoE (ablation-gated)
│   │   ├── adaptive.py           # early-exit / halting / loops
│   │   ├── memory_controller.py  # Dual Memory controller
│   │   ├── reasoning.py          # Neural Reasoning controller
│   │   ├── tokenizer/            # REPLACE char tokenizer (BPE/Unigram)
│   │   └── model.py              # full core assembly
│   ├── agent/                    # OFFLINE AGENT LAYER
│   │   ├── agent.py              # orchestration, session, safety
│   │   └── langid.py             # MOVE language-ID (router feature)
│   ├── tools/                    # LOCAL DETERMINISTIC TOOLS (router-gated)
│   │   ├── project_planner/      # EXTERNAL TOOL (old reasoning/*)
│   │   ├── symbolic_verify/      # EXTERNAL TOOL (constraints/causal)
│   │   └── python_analysis/      # KEEP (old python_brain/*)
│   ├── training/                 # Phase 8 infra
│   ├── data/                     # Phase 9 pipeline
│   ├── eval/                     # Phase 11 benchmarks
│   ├── runtime/                  # inference runtime (Phase 12/15)
│   └── deploy/                   # packaging, quantization, edge (Phase 15)
├── experiments/                  # logs, ablations, results (offline)
└── tests/
    ├── neural/                   # new: KSC math, router, memory, ...
    ├── tools/                    # kept: planner + python_brain regression
    └── agent/                    # integration
```

## Notes
- The neural core (`khwarizmi/neural`) imports **nothing** from the tool layer at inference.
- Tools are invoked only via the agent layer using a `LocalTool` protocol (`can_help`,
  `run`). The old `rafig/` package is migrated into `khwarizmi/tools/` + `khwarizmi/agent/`
  + `khwarizmi/common/`; `rafig/` is archived (not deleted) for reference during migration.
- Old `requirements.txt` (empty) is replaced by the split above.
