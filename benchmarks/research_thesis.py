"""Isolated, CPU-only falsification experiments; never modifies the protected core.

Run from the repository root:
    OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.research_thesis --output RUN.json

Contract violations are research results, not runner failures. A crashed experiment
is an ERROR and makes this command exit nonzero. No pretrained models, APIs, network,
or untrusted code execution. Toy task success is NOT assistant/scientific capability.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import random
import resource
import subprocess
import time
import traceback

import numpy as np
import torch
from torch import nn

from khwarizmi.config import get_tiny_test_config
from khwarizmi.core.ksc_cell import KhwarizmiStateCell
from khwarizmi.core.model import KhwarizmiModel
from khwarizmi.core.prototype import KhwarizmiKSCPrototype
from khwarizmi.reasoning.latent_reasoner import LatentReasoner
from khwarizmi.reasoning.neural_reasoning_core import ReasoningLosses
from khwarizmi.reasoning.reasoning_substrate import (
    ContradictionDetector, Evidence, EvidenceStore, Hypothesis, SourceMetadata,
    VerificationEngine, VerificationResult, VerificationStatus,
)
from khwarizmi.integration.agentic_loop import ProcessRewardModel

ROOT = Path(__file__).resolve().parents[1]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def small_config():
    cfg = get_tiny_test_config()
    cfg.d_model = 32
    cfg.n_heads = 4
    cfg.d_expansion = 8
    cfg.d_ff = 64
    cfg.dropout = 0.0
    return cfg


def contracts(seeds: list[int]) -> dict:
    """Counterexamples to causal, compute-accounting and evidence contracts."""
    rows = []
    for seed in seeds:
        seed_all(seed)
        original = torch.tensor([[1, 2, 3, 4, 5, 6]])
        changed = torch.tensor([[1, 2, 3, 9, 10, 11]])
        for layers in (1, 2):
            for cls in (KhwarizmiKSCPrototype, KhwarizmiModel):
                cfg = small_config()
                cfg.n_layers = layers
                cfg.enable_moe = False
                cfg.enable_adaptive_compute = False
                cfg.enable_reasoning_core = False
                seed_all(seed)
                model = cls(cfg).eval()
                with torch.no_grad():
                    a, b = model(original), model(changed)
                delta = (a.logits[:, :3] - b.logits[:, :3]).abs().max().item()
                rows.append({"seed": seed, "model": cls.__name__, "layers": layers,
                             "max_prefix_logit_delta": delta,
                             "causal_at_tolerance_1e-6": delta <= 1e-6})
        cfg = small_config()
        reasoner = LatentReasoner(cfg).eval()
        calls = []
        hook = reasoner.adaptive_compute.reasoning_cell.register_forward_hook(
            lambda module, args, output: calls.append(args[0].numel()))
        latent = torch.randn(1, 5, cfg.d_model)
        with torch.no_grad():
            out, _, ponder, diagnostics = reasoner.reason(
                latent, pathway_id=torch.tensor([0]), force_cycles=3)
        hook.remove()
        rows.append({"seed": seed, "probe": "FAST_execution",
                     "full_sequence_cell_calls": len(calls),
                     "reported_ponder_loss": ponder.item(),
                     "returned_representation_unchanged": torch.equal(out, latent),
                     "reported_mean_cycles": diagnostics["mean_cycles"]})
        model = KhwarizmiModel(cfg).eval()
        with torch.no_grad():
            a, b = model(original), model(changed)
        rows.append({"seed": seed, "probe": "fresh_state_router",
                     "routing_probs_identical": torch.equal(a.routing_probs, b.routing_probs),
                     "full_model_prefix_delta":
                     (a.logits[:, :3] - b.logits[:, :3]).abs().max().item()})

    cfg = small_config()
    confidence = torch.tensor([0.2, 0.8], requires_grad=True)
    loss = ReasoningLosses(cfg).consistency_loss(confidence)
    loss.backward()
    store = EvidenceStore(max_size=1)
    for i in range(3):
        store.add(Evidence(str(i), SourceMetadata(str(i), "test"),
                           verification_status=VerificationStatus.VERIFIED))
    e1 = Evidence("x = 1", SourceMetadata("a", "test"))
    e2 = Evidence("x = 2", SourceMetadata("b", "test"))
    # Harmless import-only test: no file access or sandboxed execution.
    from khwarizmi.coding.execution_sandbox import _restrict_imports
    allowed_pathlib = _restrict_imports()("pathlib").__name__ == "pathlib"
    from khwarizmi.training.ponder_loss_strict import AdaptiveComputeTrainer
    model = KhwarizmiModel(cfg)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer_error = None
    try:
        AdaptiveComputeTrainer(model, optimizer, None)
    except (AttributeError, TypeError) as exc:
        trainer_error = f"{type(exc).__name__}: {exc}"
    return {"status": "MEASURED", "config": cfg.to_dict(), "rows": rows,
            "confidence_loss_gradient": confidence.grad.tolist(),
            "confidence_target_uses_correctness_labels": False,
            "evidence_capacity_requested": 1, "evidence_size_after_verified_inserts": store.size(),
            "empty_detector_reports_conflict": ContradictionDetector().detect(e1, e2),
            "unknown_verifier_accepts_failure": ProcessRewardModel.verify("unknown", {"success": False})[0],
            "non_allowlisted_pathlib_import_succeeds": allowed_pathlib,
            "default_training_wrapper_error": trainer_error,
            "scope": "Counterexamples, not a security penetration test or exhaustive correctness proof."}


def state_decay(seeds: list[int]) -> dict:
    """Isolate retention of a perturbation under identical subsequent cell inputs.

    This tests the recurrent state channel only, not trained semantic recall.
    The exact contraction bound requires identical future input-dependent gates.
    """
    rows = []
    for seed in seeds:
        seed_all(seed)
        cfg = small_config()
        cell = KhwarizmiStateCell(cfg).eval()
        left = cell.init_state(1)
        right = left + 1.0
        denominator = (right - left).abs().max().item()
        x = torch.randn(1, cfg.d_model)
        with torch.no_grad():
            for step in range(1, 513):
                _, left, _ = cell.step_forward(x, left)
                _, right, _ = cell.step_forward(x, right)
                if step in (1, 16, 128, 512):
                    rows.append({"seed": seed, "steps": step,
                                 "observed_linf_ratio": (right-left).abs().max().item()/denominator,
                                 "theoretical_upper_bound_exact_arithmetic": cfg.gamma_max**step})
    return {"status": "MEASURED", "rows": rows,
            "scope": "Fixed identical future input; no assertion of a semantic memory capacity limit."}


class RecallProbe(nn.Module):
    """Learned readout of the real KSC cell; no access to answer labels as inputs."""
    def __init__(self):
        super().__init__()
        self.cfg = small_config()
        self.embedding = nn.Embedding(11, self.cfg.d_model)
        self.cell = KhwarizmiStateCell(self.cfg)
        self.head = nn.Linear(self.cfg.d_model, 4)

    def forward(self, ids):
        h, _, _ = self.cell(self.embedding(ids))
        return self.head(h[:, -1])


def recall_batch(generator: torch.Generator, count: int, gap: int):
    # Three typed writes; duplicate keys intentionally overwrite previous values.
    # PAD=0, WRITE=1, QUERY=2, keys=3..6, values=7..10.
    keys = torch.randint(0, 4, (count, 3), generator=generator)
    values = torch.randint(0, 4, (count, 3), generator=generator)
    query_position = torch.randint(0, 3, (count,), generator=generator)
    queries = keys[torch.arange(count), query_position]
    labels = torch.zeros(count, dtype=torch.long)
    ids = torch.zeros(count, 11 + gap, dtype=torch.long)
    for j in range(3):
        ids[:, j*3] = 1
        ids[:, j*3+1] = keys[:, j]+3
        ids[:, j*3+2] = values[:, j]+7
        labels = torch.where(keys[:, j] == queries, values[:, j], labels)
    ids[:, -2] = 2
    ids[:, -1] = queries+3
    return ids, labels


def exact_typed_lookup(ids: torch.Tensor) -> torch.Tensor:
    # An explicit upper control with a HAND-SUPPLIED parser, not neural extraction.
    answers = []
    for tokens in ids.tolist():
        table = {}
        for j in (0, 3, 6):
            table[tokens[j+1]] = tokens[j+2]-7
        answers.append(table[tokens[-1]])
    return torch.tensor(answers)


def delayed_recall(seeds: list[int], steps: int) -> dict:
    rows = []
    for seed in seeds:
        seed_all(seed)
        model = RecallProbe()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        train_gen = torch.Generator().manual_seed(seed+100)
        start = time.perf_counter()
        losses = []
        for step in range(steps):
            gap = step % 9  # training gaps 0..8, deterministic balanced curriculum
            ids, labels = recall_batch(train_gen, 32, gap)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(ids), labels)
            loss.backward()
            optimizer.step()
            if step in (0, steps-1):
                losses.append(loss.item())
        training_seconds = time.perf_counter()-start
        model.eval()
        for gap in (0, 8, 32, 128):
            test_gen = torch.Generator().manual_seed(seed+10000+gap)
            ids, labels = recall_batch(test_gen, 256, gap)
            with torch.no_grad():
                start = time.perf_counter()
                predictions = model(ids).argmax(-1)
                neural_seconds = time.perf_counter()-start
            start = time.perf_counter()
            exact = exact_typed_lookup(ids)
            lookup_seconds = time.perf_counter()-start
            rows.append({"seed": seed, "gap": gap, "test_examples": 256,
                         "correct": (predictions == labels).sum().item(),
                         "typed_lookup_correct": (exact == labels).sum().item(),
                         "parameters": sum(p.numel() for p in model.parameters()),
                         "training_steps": steps, "training_seconds": training_seconds,
                         "first_and_last_training_loss": losses,
                         "neural_batch_seconds": neural_seconds,
                         "typed_lookup_batch_seconds": lookup_seconds,
                         "cell_state_bytes_per_example": model.cell.init_state(1).numel()*4})
    return {"status": "MEASURED", "rows": rows, "chance_accuracy": 0.25,
            "training_examples_per_seed": steps*32,
            "scope": "Small trained KSC+readout; no MoE/ARRC. Typed lookup is an information-access upper control, NOT a matched learned competitor. Timings are single batches, NOT deployment benchmarks."}


def evaluate_polynomial(h, x):
    a, b, c = h
    return a*x*x + b*x + c


def discovery(seeds: list[int]) -> dict:
    """Version-space active experiment design using the existing verification API.

    All candidate laws and noiseless inputs are supplied by the experimenter.
    There is no learned hypothesis generation, causal discovery, or LLM here.
    """
    candidates = list(itertools.product(range(-2, 3), repeat=3))
    grid = list(range(-2, 3))
    rows, traces = [], []
    for seed in seeds:
        for mode in ("random", "information_gain"):
            rng = random.Random(seed)
            start = time.perf_counter()
            for truth in candidates:
                live = list(candidates)
                remaining = list(grid)
                trace = []
                for step in range(3):
                    if len(live) == 1:
                        break
                    if mode == "random":
                        x = rng.choice(remaining)
                    else:
                        def entropy(x):
                            counts = Counter(evaluate_polynomial(h, x) for h in live)
                            return -sum((v/len(live))*math.log2(v/len(live)) for v in counts.values())
                        x = max(remaining, key=lambda x: (entropy(x), -x))
                    observed = evaluate_polynomial(truth, x)
                    evidence = Evidence({"x": x, "y": observed},
                                        SourceMetadata(f"{truth}:{step}", "toy_simulator"))
                    engine = VerificationEngine()
                    def check(hypothesis):
                        law = tuple(map(int, hypothesis.claim.split(',')))
                        verdict = (VerificationStatus.VERIFIED if evaluate_polynomial(law, x) == observed
                                   else VerificationStatus.REFUTED)
                        return VerificationResult(verdict, "Agreement with ONE observation only")
                    engine.register_verifier("observation", check)
                    survivors = []
                    for law in live:
                        hypothesis = Hypothesis(','.join(map(str, law)),
                                                supporting_evidence=[evidence.evidence_id])
                        verdict = engine.verify(hypothesis, "observation")
                        if verdict.status == VerificationStatus.ERROR:
                            raise RuntimeError(verdict.error)
                        if verdict.status == VerificationStatus.VERIFIED:
                            survivors.append(law)
                    trace.append({"x": x, "y": observed, "before": len(live),
                                  "after": len(survivors), "evidence": evidence.to_dict()})
                    live = survivors
                    remaining.remove(x)
                rows.append({"seed": seed, "mode": mode, "truth": truth,
                             "queries": len(trace), "identified": live == [truth]})
                if truth == (2, -1, 1) and seed == seeds[0]:
                    traces.append({"mode": mode, "steps": trace})
            rows.append({"seed": seed, "mode": mode, "wall_seconds_all_125_laws": time.perf_counter()-start})
    # Falsify the closed-world assumption: a cubic has the same values as a
    # quadratic on the three discovery inputs, but fails independent holdout.
    observations = [(x, x**3) for x in (-1, 0, 1)]
    survivors = [h for h in candidates if all(evaluate_polynomial(h, x) == y for x, y in observations)]
    holdout_rejects = all(evaluate_polynomial(h, 2) != 8 for h in survivors)
    assert len(candidates) == 125
    assert all(r['identified'] for r in rows if 'identified' in r)
    assert survivors == [(0, 1, 0)] and holdout_rejects
    return {"status": "MEASURED", "rows": rows, "example_traces": traces,
            "misspecification": {"true_law": "x^3", "discovery_observations": observations,
                                 "false_unique_quadratic": survivors,
                                 "independent_x2_rejects": holdout_rejects},
            "scope": "Finite exact polynomial identification only. Active acquisition uses supplied candidate predictions. One-observation agreement must never mean universal theorem validity."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    args = parser.parse_args()
    if args.steps < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('steps must be positive and seeds must be distinct')
    target = args.output.resolve()
    if not target.is_relative_to(ROOT) or not target.parent.is_dir():
        parser.error('output parent must exist within the repository')
    torch.set_num_threads(1)
    protected = sorted(p for directory in ('core', 'experts', 'reasoning', 'memory', 'routing')
                       for p in (ROOT/'khwarizmi'/directory).glob('*.py'))
    def hashes():
        return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    before = hashes()
    report = {"schema": 1, "git_commit": subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              "git_dirty": bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
              "platform": platform.platform(), "threads": 1, "cuda": torch.cuda.is_available(),
              "seeds": args.seeds, "steps": args.steps, "protected_hashes_before": before,
              "experiments": {}, "energy_joules": None, "hardware_flops": None}
    failures = 0
    for name, fn in [('contracts', lambda: contracts(args.seeds)),
                     ('state_decay', lambda: state_decay(args.seeds)),
                     ('delayed_recall', lambda: delayed_recall(args.seeds, args.steps)),
                     ('discovery', lambda: discovery(args.seeds))]:
        start = time.perf_counter()
        try:
            result = fn()
        except Exception:
            failures += 1
            result = {"status": "ERROR", "traceback": traceback.format_exc()}
        result['wall_seconds'] = time.perf_counter()-start
        report['experiments'][name] = result
        report['process_peak_rss_kib_linux'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['protected_hashes_unchanged'] = hashes() == before
        target.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(name, result['status'], round(result['wall_seconds'], 3), flush=True)
    return int(failures > 0 or not report['protected_hashes_unchanged'])


if __name__ == '__main__':
    raise SystemExit(main())
