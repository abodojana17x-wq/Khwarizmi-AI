"""
Phase 7B: Sharia-Aware Cognitive Router

Extends the standard cognitive router with Islamic compliance checking:
- SHARIA_CHECK pathway for automatic haram detection
- Multi-madhab routing for fatwa delivery
- Adversarial attempt detection
- Low-latency rejection (<100ms) on CPU

This ensures haram queries are blocked BEFORE any processing occurs.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from enum import Enum


class PathwayType(Enum):
    """Cognitive pathways including Islamic compliance"""
    FAST = "fast"  # Simple queries, no reasoning needed
    CODING = "coding"  # Programming tasks
    REASONING = "reasoning"  # Math/logic problems
    PROJECT_PLAN = "project_plan"  # Long-horizon planning
    VERIFICATION = "verification"  # Self-check and validation
    SHARIA_CHECK = "sharia_check"  # Islamic compliance verification (NEW)
    FATWA = "fatwa"  # Complex Islamic jurisprudence (NEW)


class ShariaRouter(nn.Module):
    """
    Sharia-Aware Cognitive Router
    
    Routes queries through appropriate pathways while ensuring:
    1. Haram queries are detected and rejected immediately
    2. Halal queries proceed to appropriate processing
    3. Fatwa requests are routed to multi-madhab reasoning
    4. Adversarial evasion attempts are detected
    
    Architecture:
    - Lightweight classifier head (<1M parameters)
    - Two-stage filtering: keyword + neural classification
    - Confidence thresholding for uncertain cases
    - Madhab-aware routing for complex fiqh questions
    """
    
    def __init__(self, 
                 hidden_size: int = 768,
                 num_pathways: int = 7,
                 sharia_threshold: float = 0.95,
                 adversarial_detection: bool = True,
                 madhab_support: List[str] = None):
        """
        Args:
            hidden_size: Input embedding dimension
            num_pathways: Number of routing pathways (default 7 with Islamic additions)
            sharia_threshold: Confidence threshold for haram detection (≥0.95 required)
            adversarial_detection: Enable detection of evasion attempts
            madhab_support: List of supported madhabs ['hanafi', 'shafii', 'maliki', 'hanbali']
        """
        super().__init__()
        
        self.sharia_threshold = sharia_threshold
        self.adversarial_detection = adversarial_detection
        self.madhabs = madhab_support or ['hanafi', 'shafii', 'maliki', 'hanbali']
        
        # Keyword-based pre-filtering (ultra-fast, <1ms)
        self.haram_keywords = {
            'riba': ['ربا', 'فائدة', 'interest', 'usury'],
            'gambling': ['قمار', 'ميسر', 'bet', 'gamble', 'casino'],
            'zina': ['زنا', 'sex outside marriage', 'illicit relationship'],
            'alcohol': ['خمر', 'كحول', 'alcohol', 'wine', 'beer'],
            'magic': ['سحر', 'شعوذة', 'magic', 'sorcery', 'black magic'],
            'shirk': ['شرك', 'associating partners with Allah'],
        }
        
        # Neural router for nuanced detection
        self.router_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # Classification heads
        self.pathway_classifier = nn.Linear(hidden_size, num_pathways)
        self.haram_classifier = nn.Linear(hidden_size, 1)  # Binary: haram vs not
        self.adversarial_classifier = nn.Linear(hidden_size, 1) if adversarial_detection else None
        
        # Madhab-specific routing for fatwas
        self.madhab_routers = nn.ModuleDict({
            madhab: nn.Linear(hidden_size, len(self.madhabs))
            for madhab in self.madhabs
        })
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize router weights with Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, 
                embeddings: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                metadata: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        Route query through appropriate pathway with Sharia compliance check
        
        Args:
            embeddings: Input embeddings [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            metadata: Optional metadata (user madhab preference, context, etc.)
            
        Returns:
            Dictionary containing:
            - 'pathway_probs': Probabilities for each pathway [batch_size, num_pathways]
            - 'is_haram': Binary prediction [batch_size]
            - 'haram_confidence': Confidence scores [batch_size]
            - 'selected_pathway': Index of selected pathway [batch_size]
            - 'madhab_routing': Optional madhab-specific routing for fatwas
            - 'blocked': Boolean mask of blocked queries [batch_size]
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device
        
        # Use mean pooling over sequence for classification
        if attention_mask is not None:
            pooled = (embeddings * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled = embeddings.mean(dim=1)
        
        # Get contextualized representation
        hidden = self.router_network(pooled)
        
        # ===== STAGE 1: Ultra-fast keyword filtering (<1ms) =====
        keyword_haram_scores = self._keyword_filter(metadata.get('text', '') if metadata else '')
        keyword_haram = torch.tensor(keyword_haram_scores, dtype=torch.float, device=device)
        
        # ===== STAGE 2: Neural classification =====
        # Haram detection
        haram_logits = self.haram_classifier(hidden).squeeze(-1)
        haram_probs = torch.sigmoid(haram_logits)
        
        # Combine keyword and neural signals (OR operation for safety)
        combined_haram_probs = torch.maximum(haram_probs, keyword_haram)
        is_haram = (combined_haram_probs > self.sharia_threshold).float()
        
        # Adversarial detection (if enabled)
        if self.adversarial_classifier is not None:
            adversarial_logits = self.adversarial_classifier(hidden).squeeze(-1)
            is_adversarial = (torch.sigmoid(adversarial_logits) > 0.8).float()
        else:
            is_adversarial = torch.zeros_like(is_haram)
        
        # Pathway classification (only for non-haram queries)
        pathway_logits = self.pathway_classifier(hidden)
        pathway_probs = torch.softmax(pathway_logits, dim=-1)
        
        # Force SHARIA_CHECK pathway for high-confidence haram detections
        sharia_check_idx = list(PathwayType).index(PathwayType.SHARIA_CHECK)
        pathway_probs = pathway_probs * (1 - is_haram.unsqueeze(-1))
        pathway_probs[:, sharia_check_idx] = is_haram * 0.99  # Strong signal for haram
        
        # Select primary pathway
        selected_pathway = pathway_probs.argmax(dim=-1)
        
        # Block haram queries from further processing
        blocked = is_haram.bool()
        
        # Madhab-specific routing for fatwa queries
        fatwa_idx = list(PathwayType).index(PathwayType.FATWA)
        is_fatwa_query = (selected_pathway == fatwa_idx)
        
        madhab_routing = {}
        if is_fatwa_query.any():
            for madhab in self.madhabs:
                madhab_routing[madhab] = self.madhab_routers[madhab](hidden[is_fatwa_query])
        
        return {
            'pathway_probs': pathway_probs,
            'is_haram': is_haram,
            'haram_confidence': combined_haram_probs,
            'selected_pathway': selected_pathway,
            'blocked': blocked,
            'is_adversarial': is_adversarial,
            'madhab_routing': madhab_routing if madhab_routing else None,
            'keyword_triggered': keyword_haram > 0.5,
        }
    
    def _keyword_filter(self, text: str) -> List[float]:
        """
        Ultra-fast keyword-based haram detection
        
        Args:
            text: Input query text
            
        Returns:
            List of binary scores (0 or 1) for each sample in batch
        """
        if not text:
            return [0.0]
        
        text_lower = text.lower()
        score = 0.0
        
        for category, keywords in self.haram_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    score = 1.0
                    break
            if score == 1.0:
                break
        
        return [score]
    
    def route_query(self, 
                   embeddings: torch.Tensor,
                   text: Optional[str] = None,
                   user_madhab: Optional[str] = None) -> Dict:
        """
        High-level API for routing a single query
        
        Args:
            embeddings: Query embeddings [1, seq_len, hidden_size]
            text: Raw query text for keyword filtering
            user_madhab: User's preferred madhab (optional)
            
        Returns:
            Routing decision dictionary
        """
        with torch.no_grad():
            result = self.forward(
                embeddings,
                metadata={'text': text, 'preferred_madhab': user_madhab}
            )
            
            is_blocked = result['blocked'].item()
            pathway_idx = result['selected_pathway'].item()
            pathway = list(PathwayType)[pathway_idx].value
            
            response = {
                'allowed': not is_blocked,
                'pathway': pathway,
                'confidence': result['haram_confidence'].item(),
                'requires_review': result['haram_confidence'].item() > 0.7 and not is_blocked,
            }
            
            if is_blocked:
                response['rejection_reason'] = 'sharia_violation'
                response['message'] = 'عذراً، لا يمكنني الإجابة على هذا السؤال لأنه يخالف الشريعة الإسلامية'
            
            return response


def create_sharia_router(hidden_size: int = 768,
                         pretrained_weights: Optional[str] = None) -> ShariaRouter:
    """
    Factory function to create a ShariaRouter
    
    Args:
        hidden_size: Embedding dimension
        pretrained_weights: Optional path to pretrained router weights
        
    Returns:
        Initialized ShariaRouter
    """
    router = ShariaRouter(hidden_size=hidden_size)
    
    if pretrained_weights is not None:
        router.load_state_dict(torch.load(pretrained_weights, map_location='cpu'))
        print(f"Loaded pretrained Sharia router weights from {pretrained_weights}")
    
    return router
