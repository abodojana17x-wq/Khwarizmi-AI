"""
Khwarizmi AI Core Configuration Package.
"""

from .settings import KhwarizmiConfig
from .tiers import (
    get_tiny_test_config,
    get_prototype_config,
    get_small_config,
    get_edge_config,
)

__all__ = [
    "KhwarizmiConfig",
    "get_tiny_test_config",
    "get_prototype_config",
    "get_small_config",
    "get_edge_config",
]
