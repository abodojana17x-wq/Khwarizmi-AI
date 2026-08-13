"""
Predefined Architectural Tiers for Khwarizmi AI.

Defines standard configurations to scale the system cleanly:
- Tiny Test Tier: Lightweight CPU-testable unit test configuration (< 500k params)
- Prototype Tier: 50M-150M experimental tier for initial validation
- Small Tier: 300M-700M parameters for multi-lingual/coding tasks
- Edge Tier: 1B-3B parameters for low-RAM offline edge devices
"""

from .settings import KhwarizmiConfig


def get_tiny_test_config() -> KhwarizmiConfig:
    """
    Return a tiny CPU-runnable test configuration (<500K parameters).
    Designed specifically for Arena sandboxes and automated unit testing.
    """
    return KhwarizmiConfig(
        vocab_size=512,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_expansion=16,
        d_ff=128,
        num_experts=4,
        top_k_experts=2,
        moe_frequency=2,
        max_seq_len=128,
        gamma_min=0.85,
        gamma_max=0.999,
        max_recurrent_cycles=3,
        memory_dim=64,
        memory_slots=16,
        num_pathways=5,
        dropout=0.0,
        temperature=1.0,
        load_balance_alpha=0.01,
        ponder_cost_beta=0.01,
        verification_threshold=0.75,
        tier_name="TinyTest",
    )


def get_prototype_config() -> KhwarizmiConfig:
    """
    Return Prototype Tier configuration (approx. 50M-150M parameters).

    Retained for backwards compatibility with the full KhwarizmiModel tests
    (it is imported directly by ``tests/test_khwarizmi_model.py`` and must
    keep ``tier_name == "Prototype"``). The two purpose-built Phase 2
    prototype configurations are :func:`get_prototype_50m_config` and
    :func:`get_prototype_150m_config`.
    """
    return KhwarizmiConfig(
        vocab_size=32768,
        d_model=512,
        n_layers=12,
        n_heads=8,
        d_expansion=32,
        d_ff=2048,
        num_experts=8,
        top_k_experts=2,
        moe_frequency=4,
        max_seq_len=2048,
        gamma_min=0.85,
        gamma_max=0.999,
        max_recurrent_cycles=4,
        memory_dim=512,
        memory_slots=256,
        num_pathways=5,
        dropout=0.1,
        tier_name="Prototype",
    )


def get_prototype_50m_config() -> KhwarizmiConfig:
    """
    Return the Phase 2 *50M* Prototype Tier configuration.

    A minimal KSC-only language-modeling configuration (~50M parameters when
    instantiated through :class:`khwarizmi.core.prototype.KhwarizmiKSCPrototype`).
    ``max_seq_len`` is raised to 16384 so the sinusoidal positional buffer covers
    the Phase 2 latency/memory benchmarks at 4K and 16K context.
    """
    return KhwarizmiConfig(
        vocab_size=32768,
        d_model=512,
        n_layers=8,
        n_heads=8,
        d_expansion=32,
        d_ff=1024,
        num_experts=8,
        top_k_experts=2,
        moe_frequency=4,
        max_seq_len=16384,
        gamma_min=0.85,
        gamma_max=0.999,
        max_recurrent_cycles=4,
        memory_dim=512,
        memory_slots=256,
        num_pathways=5,
        dropout=0.1,
        tier_name="Prototype-50M",
    )


def get_prototype_150m_config() -> KhwarizmiConfig:
    """
    Return the Phase 2 *150M* Prototype Tier configuration.

    A minimal KSC-only language-modeling configuration (~150M parameters when
    instantiated through :class:`khwarizmi.core.prototype.KhwarizmiKSCPrototype`).
    ``max_seq_len`` is raised to 16384 so the sinusoidal positional buffer covers
    the Phase 2 latency/memory benchmarks at 4K and 16K context.
    """
    return KhwarizmiConfig(
        vocab_size=32768,
        d_model=768,
        n_layers=16,
        n_heads=12,
        d_expansion=48,
        d_ff=2048,
        num_experts=8,
        top_k_experts=2,
        moe_frequency=4,
        max_seq_len=16384,
        gamma_min=0.85,
        gamma_max=0.999,
        max_recurrent_cycles=4,
        memory_dim=768,
        memory_slots=384,
        num_pathways=5,
        dropout=0.1,
        tier_name="Prototype-150M",
    )


def get_small_config() -> KhwarizmiConfig:
    """
    Return Small Tier configuration (approx. 300M-700M parameters).
    """
    return KhwarizmiConfig(
        vocab_size=32768,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        d_expansion=64,
        d_ff=4096,
        num_experts=8,
        top_k_experts=2,
        moe_frequency=4,
        max_seq_len=4096,
        gamma_min=0.85,
        gamma_max=0.999,
        max_recurrent_cycles=5,
        memory_dim=1024,
        memory_slots=512,
        num_pathways=5,
        dropout=0.1,
        tier_name="Small",
    )


def get_edge_config() -> KhwarizmiConfig:
    """
    Return Edge Tier configuration (approx. 1B-3B parameters).
    """
    return KhwarizmiConfig(
        vocab_size=65536,
        d_model=2048,
        n_layers=32,
        n_heads=32,
        d_expansion=64,
        d_ff=8192,
        num_experts=8,
        top_k_experts=2,
        moe_frequency=4,
        max_seq_len=8192,
        gamma_min=0.85,
        gamma_max=0.999,
        max_recurrent_cycles=6,
        memory_dim=2048,
        memory_slots=1024,
        num_pathways=5,
        dropout=0.1,
        tier_name="Edge",
    )
