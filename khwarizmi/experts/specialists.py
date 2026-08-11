"""
Khwarizmi Specialist Expert Definitions.

Defines standard specialized expert subnetworks for initial candidate specializations
as specified in Section 4.4 of the Khwarizmi AI Blueprint:
    (1) Multilingual Language / Arabic
    (2) Egyptian Arabic / Dialect
    (3) Python / Coding
    (4) Software Engineering / Architecture
    (5) Mathematical / Symbolic Reasoning
    (6) Project Planning / DAGs
    (7) Tool Use / Verification
    (8) General Fact Recall
"""

from typing import List
from ..config.settings import KhwarizmiConfig
from .moe_layer import ExpertLayer

SPECIALIZATION_NAMES = [
    "Multilingual_Arabic",
    "Egyptian_Arabic_Dialect",
    "Python_Coding",
    "Software_Engineering",
    "Mathematical_Reasoning",
    "Project_Planning_DAG",
    "Tool_Use_Verification",
    "General_Fact_Recall",
]


def create_standard_specialists(config: KhwarizmiConfig) -> List[ExpertLayer]:
    """
    Instantiate standard specialized experts up to config.num_experts.
    If config.num_experts > len(SPECIALIZATION_NAMES), appends General experts.

    Args:
        config: KhwarizmiConfig instance.

    Returns:
        List of ExpertLayer instances with assigned specialization names.
    """
    experts = []
    for i in range(config.num_experts):
        if i < len(SPECIALIZATION_NAMES):
            name = SPECIALIZATION_NAMES[i]
        else:
            name = f"General_Specialist_{i}"
        experts.append(ExpertLayer(config, specialization_name=name))
    return experts
