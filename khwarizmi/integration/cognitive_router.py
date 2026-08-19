"""
Khwarizmi Cognitive Router — Domain Task Routing.

This module routes incoming tasks to appropriate domains:
    - CODE: Programming, code analysis, debugging tasks
    - SCIENCE: Physics, math, unit verification problems
    - ART: Aesthetic evaluation, composition analysis
    - CREATIVITY: Ideation, SCAMPER, brainstorming
    - GENERAL: Conversational, lookup, simple queries

Features:
    - Lightweight deterministic feature extraction
    - Integration with reasoning-core confidence scores
    - Offline-first operation with minimal memory footprint
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import re


@dataclass(frozen=True)
class DomainResult:
    """Result of domain classification."""
    domain: str  # CODE, SCIENCE, ART, CREATIVITY, GENERAL
    confidence: float  # 0.0 to 1.0
    features: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    
    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.7
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "confidence": self.confidence,
            "features": self.features,
            "reasoning": self.reasoning,
        }


class CognitiveRouter:
    """
    Deterministic cognitive router for task domain classification.
    
    Uses lightweight feature extraction and keyword matching to route
    tasks to appropriate processing domains. Designed for offline-first
    operation with <4GB RAM constraints.
    
    Domains:
        - CODE: Python programming, code review, debugging
        - SCIENCE: Physics equations, math problems, unit verification
        - ART: Aesthetic scoring, composition analysis
        - CREATIVITY: Brainstorming, SCAMPER ideation
        - GENERAL: Everything else (conversational, lookup)
    """
    
    # Code keywords with weights (expanded)
    CODE_KEYWORDS = {
        "python": 0.8, "code": 0.9, "function": 0.7, "variable": 0.6,
        "loop": 0.6, "class": 0.7, "module": 0.6, "import": 0.5,
        "debug": 0.8, "error": 0.5, "exception": 0.6, "syntax": 0.7,
        "algorithm": 0.7, "data structure": 0.8, "list": 0.5, "dict": 0.5,
        "compile": 0.7, "execute": 0.6, "run": 0.5, "test": 0.6,
        "refactor": 0.7, "optimize": 0.6, "performance": 0.5,
        "api": 0.6, "interface": 0.5, "implement": 0.7, "write": 0.4,
        "def ": 0.8, "return": 0.5, "if ": 0.4, "for ": 0.5, "while": 0.5,
        "print": 0.4, "range": 0.5, "lambda": 0.6, "self": 0.5,
    }
    
    SCIENCE_KEYWORDS = {
        "physics": 0.9, "equation": 0.8, "formula": 0.8, "calculate": 0.6,
        "force": 0.7, "mass": 0.7, "acceleration": 0.7, "velocity": 0.7,
        "energy": 0.7, "momentum": 0.7, "gravity": 0.7, "newton": 0.6,
        "unit": 0.7, "dimension": 0.7, "si": 0.6, "meter": 0.5, "second": 0.5,
        "kilogram": 0.5, "joule": 0.6, "watt": 0.6, "pascal": 0.6,
        "math": 0.8, "algebra": 0.7, "calculus": 0.7, "derivative": 0.7,
        "integral": 0.7, "vector": 0.6, "matrix": 0.6, "tensor": 0.6,
        "solve": 0.5, "compute": 0.5, "result": 0.4, "value": 0.4,
        "constant": 0.6, "coefficient": 0.6, "variable": 0.5,
        "= m *": 0.8, "= m*": 0.8, "F =": 0.8, "E =": 0.7, "v =": 0.6,
    }
    
    ART_KEYWORDS = {
        "art": 0.9, "aesthetic": 0.9, "composition": 0.8, "color": 0.7,
        "design": 0.8, "visual": 0.7, "image": 0.6, "picture": 0.6,
        "palette": 0.7, "hue": 0.6, "saturation": 0.7, "contrast": 0.7,
        "balance": 0.6, "symmetry": 0.7, "harmony": 0.7, "focal": 0.6,
        "rule of thirds": 0.8, "golden ratio": 0.8, "layout": 0.6,
        "beautiful": 0.5, "ugly": 0.5, "pretty": 0.5, "score": 0.5,
        "rate": 0.5, "evaluate": 0.5, "critique": 0.7, "feedback": 0.5,
        "تصميم": 0.9, "ارسم": 0.9, "لون": 0.8, "الوان": 0.7, "رسم": 0.8,
        "فن": 0.9, "جمالي": 0.9, "تركيب": 0.7, "تناغم": 0.7,
    }
    
    CREATIVITY_KEYWORDS = {
        "creativity": 0.9, "creative": 0.9, "brainstorm": 0.9, "ideate": 0.8,
        "scamper": 0.9, "innovate": 0.7, "invent": 0.8, "imagine": 0.8,
        "idea": 0.8, "concept": 0.6, "notion": 0.6, "proposal": 0.6,
        "alternative": 0.7, "option": 0.5, "possibility": 0.6, "what if": 0.8,
        "combine": 0.6, "modify": 0.6, "adapt": 0.6, "eliminate": 0.6,
        "reverse": 0.7, "substitute": 0.6, "rearrange": 0.6,
        "novel": 0.6, "unique": 0.6, "original": 0.6, "fresh": 0.5,
        "features": 0.5, "app": 0.4, "design": 0.5, "logo": 0.7,
        "poem": 0.8, "melody": 0.7, "song": 0.6, "story": 0.5,
        "ابتكر": 0.9, "فكرة": 0.8, "ابدع": 0.9, "تصميم": 0.7,
        "ارسم": 0.8, "لون": 0.7, "قصيدة": 0.8, "افكار": 0.7,
        "إبداع": 0.9, "خيال": 0.8, "ابتكار": 0.9,
    }
    
    # Code pattern detection
    CODE_PATTERNS = [
        (r'\bdef\s+\w+\s*\(', 0.9),      # Function definition
        (r'\bclass\s+\w+', 0.9),         # Class definition
        (r'\bfor\s+\w+\s+in\s+', 0.8),   # For loop
        (r'\bwhile\s+', 0.7),            # While loop
        (r'\bif\s+.+:', 0.6),            # If statement
        (r'\bimport\s+\w+', 0.7),        # Import statement
        (r'\bfrom\s+\w+\s+import', 0.7), # From import
        (r'=\s*\[.*\]', 0.6),            # List assignment
        (r'=\s*\{.*\}', 0.6),            # Dict/set assignment
        (r'\.\w+\s*\(', 0.5),            # Method call
        (r'lambda\s+', 0.7),             # Lambda function
    ]
    
    # Equation pattern detection
    EQUATION_PATTERNS = [
        (r'[A-Za-z]+\s*=\s*[A-Za-z].*\*', 0.8),  # Variable = expression with multiplication
        (r'[A-Za-z]+\s*=\s*[0-9]', 0.7),          # Variable = number
        (r'\*\*', 0.5),                            # Exponentiation
        (r'\/', 0.5),                              # Division in potential equation
    ]
    
    def __init__(self):
        self._domain_history: List[DomainResult] = []
    
    def route(self, task: str, core_confidence: Optional[float] = None) -> DomainResult:
        """
        Route a task to the appropriate domain.
        
        Args:
            task: The input task description or prompt
            core_confidence: Optional confidence from reasoning core
            
        Returns:
            DomainResult with domain classification and metadata
        """
        task_lower = task.lower()
        
        # Extract features for each domain
        features = {
            "code_score": self._compute_domain_score(task_lower, self.CODE_KEYWORDS),
            "science_score": self._compute_domain_score(task_lower, self.SCIENCE_KEYWORDS),
            "art_score": self._compute_domain_score(task_lower, self.ART_KEYWORDS),
            "creativity_score": self._compute_domain_score(task_lower, self.CREATIVITY_KEYWORDS),
        }
        
        # Add pattern-based features
        features["code_pattern_score"] = self._detect_patterns(task, self.CODE_PATTERNS)
        features["equation_pattern_score"] = self._detect_patterns(task, self.EQUATION_PATTERNS)
        
        # Boost code score if code patterns detected (lower threshold for stronger detection)
        if features["code_pattern_score"] > 0.15:
            features["code_score"] = max(features["code_score"], features["code_pattern_score"])
        
        # Boost science score if equation patterns detected
        if features["equation_pattern_score"] > 0.3:
            features["science_score"] = max(features["science_score"], features["equation_pattern_score"])
        
        # Integrate core confidence if provided
        if core_confidence is not None:
            # Use core confidence to adjust all scores proportionally
            for key in features:
                features[key] = features[key] * 0.7 + core_confidence * 0.3
        
        # Determine winning domain
        domain_scores = {
            "CODE": features["code_score"],
            "SCIENCE": features["science_score"],
            "ART": features["art_score"],
            "CREATIVITY": features["creativity_score"],
        }
        
        # Check for clear winner with lower threshold - domain rules evaluated BEFORE GENERAL fallback
        max_score = max(domain_scores.values())
        winners = [d for d, s in domain_scores.items() if s == max_score]
        
        if max_score < 0.12:
            # No strong signal - default to GENERAL
            domain = "GENERAL"
            confidence = max(0.3, 1.0 - sum(domain_scores.values()))
            reasoning = "No strong domain signals detected; routing to GENERAL"
        elif len(winners) > 1:
            # Tie-breaker: prefer more specific domains via reasoning-core confidence priority
            priority = ["CODE", "SCIENCE", "ART", "CREATIVITY"]
            for p in priority:
                if p in winners:
                    domain = p
                    break
            confidence = max_score
            reasoning = f"Tie between {winners}; selected {domain} by priority"
        else:
            domain = winners[0]
            confidence = max_score
            reasoning = f"Strong {domain} signal detected (score={max_score:.2f})"
        
        result = DomainResult(
            domain=domain,
            confidence=min(0.99, confidence),
            features=features,
            reasoning=reasoning,
        )
        
        self._domain_history.append(result)
        return result
    
    def _compute_domain_score(self, text: str, keywords: Dict[str, float]) -> float:
        """Compute domain relevance score based on keyword matches."""
        if not text:
            return 0.0
        
        total_score = 0.0
        max_possible = 0.0
        match_count = 0
        
        for keyword, weight in keywords.items():
            max_possible += weight
            if keyword in text:
                total_score += weight
                match_count += 1
        
        if max_possible == 0:
            return 0.0
        
        # Normalize to [0, 1] range
        normalized = total_score / max_possible
        
        # Boost score if multiple keywords match
        if match_count >= 3:
            normalized = min(1.0, normalized * 2.0)
        elif match_count >= 2:
            normalized = min(1.0, normalized * 1.5)
        
        return min(1.0, normalized)
    
    def _detect_patterns(self, text: str, patterns: List[Tuple[str, float]]) -> float:
        """Detect regex patterns in text and return aggregate score."""
        if not text:
            return 0.0
        
        total_score = 0.0
        max_score = 0.0
        
        for pattern, weight in patterns:
            max_score += weight
            if re.search(pattern, text):
                total_score += weight
        
        if max_score == 0:
            return 0.0
        
        return min(1.0, total_score / max_score * 1.5)
    
    def get_domain_for_tool(self, tool_name: str) -> str:
        """Map a tool name to its primary domain."""
        tool_domains = {
            "ExecutionSandbox": "CODE",
            "DataFlowAnalyzer": "CODE",
            "ControlFlowGraph": "CODE",
            "TypeInference": "CODE",
            "UnitConsistencyVerifier": "SCIENCE",
            "AestheticScorer": "ART",
            "ScamperEngine": "CREATIVITY",
        }
        return tool_domains.get(tool_name, "GENERAL")
    
    def get_history(self) -> List[DomainResult]:
        """Get routing history."""
        return list(self._domain_history)
    
    def clear_history(self) -> None:
        """Clear routing history."""
        self._domain_history.clear()


# Singleton instance
_router_instance: Optional[CognitiveRouter] = None


def get_router() -> CognitiveRouter:
    """Get the global cognitive router singleton."""
    global _router_instance
    if _router_instance is None:
        _router_instance = CognitiveRouter()
    return _router_instance


def route_task(task: str, core_confidence: Optional[float] = None) -> DomainResult:
    """Convenience function to route a task via the global router."""
    return get_router().route(task, core_confidence)
