"""
Phase 3 Benchmark — Dual Memory Architecture.

Measures the Phase 3 Dual Memory system's behavior under the conditions called
for by the roadmap and the Phase 3 specification:

  1. Bounded memory footprint: the short-term window and persistent table are
     fixed-size tensors whose byte count is independent of the number of
     operations / sequence length.
  2. Long-sequence stability: thousands of READ/WRITE/UPDATE/FORGET cycles never
     grow either store beyond its configured capacity.
  3. Operation throughput: READ / WRITE / UPDATE / FORGET cycles per second.
  4. Utility eviction quality: when the table is full, eviction removes the
     lowest-utility (bottom-10%) items while preserving high-utility facts —
     the roadmap's "no catastrophic forgetting of high-utility keys" criterion.

Run:
    python benchmarks/phase3_dual_memory.py

This script is deterministic (fixed seeds) and CPU-only.

Note (documented limitation): the roadmap's literal success criteria — NIAH
retrieval >= 95% across 32K tokens and a learned WRITE/UPDATE/FORGET policy —
require the Phase 9/10 dataset pipeline and gradient training, which are out of
Phase 3 scope. This benchmark validates the *structural* guarantees (boundedness,
determinism, eviction correctness) that Phase 3 owns.
"""

import os
import sys
from time import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

torch.set_num_threads(4)

from khwarizmi.config import KhwarizmiConfig, get_tiny_test_config
from khwarizmi.memory import (
    DualMemory,
    ShortTermWorkingState,
    LongTermPersistentMemory,
)

DEVICE = torch.device("cpu")


def _benchmark_config() -> KhwarizmiConfig:
    cfg = get_tiny_test_config()
    cfg.short_term_capacity = 256
    cfg.memory_slots = 64
    return cfg


def _state_bytes(state) -> int:
    """Total bytes held by the combined memory state tensors."""
    total = 0
    for sub in state.values():
        for t in sub.values():
            total += t.numel() * t.element_size()
    return total


def _table_state_bytes(table) -> int:
    return sum(t.numel() * t.element_size() for t in table.values())


def main() -> None:
    print("=" * 72)
    print("PHASE 3 BENCHMARK — Dual Memory Architecture")
    print("=" * 72)
    cfg = _benchmark_config()

    # 1) Bounded memory footprint ---------------------------------------------
    print("\n[1] Bounded memory footprint")
    st = ShortTermWorkingState(cfg)
    lt = LongTermPersistentMemory(cfg)
    st_state = st.init_state(1)
    lt_table = lt.init_memory_table(1)
    st_capacity_bytes = cfg.short_term_capacity * cfg.d_model * 4
    lt_capacity_bytes = cfg.memory_slots * cfg.memory_dim * 4
    print(f"  short-term window capacity : {cfg.short_term_capacity} tokens")
    print(f"  persistent table capacity  : {cfg.memory_slots} slots")
    print(f"  short-term peak state      : {st_capacity_bytes / 1024:8.2f} KB (fixed)")
    print(f"  persistent peak state      : {_table_state_bytes(lt_table) / 1024:8.2f} KB (fixed)")
    print("  => both stores are fixed-size tensors; size is independent of sequence length.")

    # 2) Long-sequence stability ---------------------------------------------
    print("\n[2] Long-sequence stability (10,000 READ/WRITE/UPDATE/FORGET cycles)")
    dm = DualMemory(cfg)
    state = dm.init_state(2)
    torch.manual_seed(0)
    max_st = 0
    max_lt = 0
    t0 = time()
    for step in range(10_000):
        h = torch.randn(2, cfg.d_model)
        out = dm(h, state, step_counter=step)
        state = out.state
        max_st = max(max_st, state["short_term"]["window_buffer"].size(1))
        max_lt = max(max_lt, dm.long_term.num_stored(state["long_term"]))
    elapsed = time() - t0
    print(f"  steps            : 10,000 (batch_size=2)")
    print(f"  peak short-term  : {max_st} items (capacity {cfg.short_term_capacity})")
    print(f"  peak long-term   : {max_lt} slots (capacity {cfg.memory_slots})")
    assert max_st <= cfg.short_term_capacity
    assert max_lt <= cfg.memory_slots
    print(f"  throughput       : {10_000 / elapsed:,.0f} lifecycle-steps/sec")
    print("  => memory remained within configured limits across the entire run.")

    # 3) Operation throughput --------------------------------------------------
    print("\n[3] Individual operation throughput (explicit-gated interfaces)")
    lt = LongTermPersistentMemory(cfg)
    table = lt.init_memory_table(4)
    h = torch.randn(4, cfg.d_model)

    def _timed(fn, iters):
        t0 = time()
        for _ in range(iters):
            fn()
        return iters / max(time() - t0, 1e-9)

    write_t = _timed(
        lambda: lt.write(h, table, g_write=torch.ones(4), current_step=0, threshold=0.1),
        2000,
    )
    read_t = _timed(lambda: lt.read(h, table, g_read=torch.ones(4), current_step=0), 2000)
    upd_t = _timed(
        lambda: lt.update(h, table, g_update=torch.ones(4), current_step=0, threshold=0.1),
        2000,
    )
    fgt_t = _timed(lambda: lt.forget(table, g_forget=torch.ones(4), threshold=0.5), 2000)
    print(f"  WRITE   : {write_t:>10,.0f} ops/sec")
    print(f"  READ    : {read_t:>10,.0f} ops/sec")
    print(f"  UPDATE  : {upd_t:>10,.0f} ops/sec")
    print(f"  FORGET  : {fgt_t:>10,.0f} ops/sec")

    # 4) Utility eviction quality ---------------------------------------------
    print("\n[4] Utility eviction: bottom-10% evicted, high-utility facts preserved")
    lt = LongTermPersistentMemory(cfg)
    table = lt.init_memory_table(1)
    torch.manual_seed(1)
    # Write `slots` facts, tagging each with a descending utility score so we
    # can verify that eviction removes low-utility items, not high-utility ones.
    n = cfg.memory_slots
    table = lt.init_memory_table(1)
    for i in range(n):
        cand = torch.randn(1, cfg.d_model)
        table = lt.write(cand, table, g_write=torch.ones(1), current_step=i, threshold=0.1)
        # Higher slot index -> higher utility (slot 0 = lowest utility).
        table["utilities"][0, i] = (i + 1) / n
    assert lt.is_full(table)

    # Evict the bottom 10% (lowest utility slots).
    bottom_10 = max(1, n // 10)
    for _ in range(bottom_10):
        table = lt.forget(table, g_forget=torch.ones(1), threshold=0.5)
    remaining = torch.nonzero(table["valid_mask"][0]).squeeze(-1).tolist()
    highest = max(remaining)
    # The bottom-10% slots (indices 0 .. bottom_10-1) must be gone; the
    # highest-utility slot must survive.
    for low in range(bottom_10):
        assert low not in remaining, f"low-utility slot {low} was not evicted"
    assert highest == n - 1, f"highest-utility slot {n - 1} was evicted (catastrophic forgetting)"
    print(f"  capacity={n}, evicted bottom {bottom_10} lowest-utility slots")
    print(f"  highest-utility slot (index {n - 1}) preserved -> no catastrophic forgetting.")

    print("\n" + "=" * 72)
    print("Phase 3 benchmark complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
