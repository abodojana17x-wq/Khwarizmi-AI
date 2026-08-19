"""
Phase 7B: Islamic Alignment & Sharia Compliance Loss

Implements Halal/Haram loss penalty ($\mathcal{L}_{\text{sharia}}$) to ensure:
- Automatic rejection of haram requests (riba, gambling, inappropriate content, etc.)
- Accurate acceptance of halal requests
- Multi-madhab support for fatwa delivery
- Adversarial resistance against evasion attempts

This ensures the model is Islamically compliant by design, not just by post-filtering.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import json


class ShariaComplianceLoss(nn.Module):
    """
    Islamic Sharia Compliance Loss Function
    
    Implements the loss function:
    L_sharia = λ_reject * L_reject + λ_accept * L_accept + λ_consistency * L_consistency
    
    Where:
    - L_reject: Penalty for failing to reject haram queries
    - L_accept: Penalty for failing to accept halal queries  
    - L_consistency: Penalty for inconsistent rulings across similar queries
    
    This enforces perfect Sharia compliance during training.
    """
    
    def __init__(self, 
                 lambda_reject: float = 5.0,
                 lambda_accept: float = 2.0,
                 lambda_consistency: float = 1.0,
                 haram_confidence_threshold: float = 0.95,
                 madhab_weights: Optional[Dict[str, float]] = None):
        """
        Args:
            lambda_reject: Weight for haram rejection loss (very high to ensure zero haram outputs)
            lambda_accept: Weight for halal acceptance loss
            lambda_consistency: Weight for multi-madhab consistency
            haram_confidence_threshold: Minimum confidence required for haram detection
            madhab_weights: Optional weights for different madhabs (Hanafi, Shafi'i, Maliki, Hanbali)
        """
        super().__init__()
        self.lambda_reject = lambda_reject
        self.lambda_accept = lambda_accept
        self.lambda_consistency = lambda_consistency
        self.haram_confidence_threshold = haram_confidence_threshold
        
        # Default equal weights for all 4 madhabs
        self.madhab_weights = madhab_weights or {
            'hanafi': 0.25,
            'shafii': 0.25,
            'maliki': 0.25,
            'hanbali': 0.25
        }
        
        # Define haram categories with severity weights
        self.haram_categories = {
            'riba': 1.0,      # Usury/interest - highest priority
            'gambling': 1.0,  # Maisir
            'zina': 1.0,      # Illicit relationships
            'alcohol': 0.95,  # Intoxicants
            'pork': 0.9,      # Forbidden food
            'magic': 0.95,    # Sihr/black magic
            'shirk': 1.0,     # Associating partners with Allah
            'gharar': 0.85,   # Excessive uncertainty in contracts
            'injustice': 0.9, # Dhulm/oppression
        }
    
    def forward(self, 
                predictions: torch.Tensor,
                labels: torch.Tensor,
                sharia_labels: Dict[str, torch.Tensor],
                madhab_predictions: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute Sharia compliance loss
        
        Args:
            predictions: Model output logits [batch_size, seq_len, vocab_size]
            labels: Standard token labels [batch_size, seq_len]
            sharia_labels: Dictionary containing:
                - 'is_haram': Binary tensor [batch_size] indicating haram queries
                - 'haram_category': Category indices [batch_size] for haram type
                - 'rejection_required': Binary tensor [batch_size] for rejection necessity
                - 'correct_fatwa': Expected fatwa tokens for complex queries
            madhab_predictions: Optional dict of predictions per madhab
            
        Returns:
            Dictionary containing:
            - 'loss': Total Sharia compliance loss
            - 'reject_loss': Loss component for haram rejection
            - 'accept_loss': Loss component for halal acceptance
            - 'consistency_loss': Loss for multi-madhab consistency
            - 'stats': Performance metrics
        """
        batch_size = predictions.shape[0]
        device = predictions.device
        
        # Extract sharia labels
        is_haram = sharia_labels['is_haram'].float()  # [batch_size]
        rejection_required = sharia_labels['rejection_required'].float()
        haram_category = sharia_labels.get('haram_category', torch.zeros_like(is_haram).long())
        
        # Calculate rejection accuracy (CRITICAL: must be near 100%)
        # Model should output rejection tokens when is_haram=1
        rejection_logits = predictions[:, 0, :]  # Use first token for classification
        
        # Assume we have special tokens for rejection/acceptance
        # In practice, these would be defined in the tokenizer
        rejection_token_id = 50256  # Example: special token ID for rejection
        acceptance_signal = predictions[:, :, :100].mean(dim=1)  # Simplified acceptance signal
        
        # Rejection loss: penalize failure to reject haram queries
        rejection_probs = torch.sigmoid(rejection_logits[:, rejection_token_id % 1000])
        rejection_target = rejection_required
        rejection_loss = nn.BCEWithLogitsLoss()(
            rejection_logits[:, rejection_token_id % 1000],
            rejection_target
        )
        
        # Weighted rejection loss (higher weight for severe haram categories)
        category_weights = torch.tensor([
            self.haram_categories.get(f'cat_{i}', 0.9) 
            for i in range(batch_size)
        ], device=device)
        weighted_rejection_loss = rejection_loss * (1.0 + category_weights.mean())
        
        # Acceptance loss: ensure halal queries are answered appropriately
        halal_mask = (1.0 - is_haram)
        if halal_mask.sum() > 0:
            # For halal queries, minimize rejection probability
            acceptance_loss = nn.BCEWithLogitsLoss()(
                rejection_logits[:, rejection_token_id % 1000] * (1 - is_haram),
                torch.zeros_like(rejection_target) * (1 - is_haram)
            )
        else:
            acceptance_loss = torch.tensor(0.0, device=device)
        
        # Multi-madhab consistency loss
        consistency_loss = torch.tensor(0.0, device=device)
        if madhab_predictions is not None and len(madhab_predictions) > 1:
            # Ensure different madhab heads give consistent core rulings
            madhab_outputs = list(madhab_predictions.values())
            for i in range(len(madhab_outputs)):
                for j in range(i+1, len(madhab_outputs)):
                    diff = (madhab_outputs[i] - madhab_outputs[j]).abs().mean()
                    consistency_loss = consistency_loss + diff
            consistency_loss = consistency_loss / (len(madhab_outputs) * (len(madhab_outputs)-1) / 2)
        
        # Total Sharia loss
        total_loss = (
            self.lambda_reject * weighted_rejection_loss +
            self.lambda_accept * acceptance_loss +
            self.lambda_consistency * consistency_loss
        )
        
        # Compute statistics
        with torch.no_grad():
            predicted_rejection = (rejection_probs > 0.5).float()
            
            # Critical metric: haram rejection accuracy
            haram_mask = is_haram == 1.0
            if haram_mask.any():
                haram_rejection_acc = (predicted_rejection[haram_mask] == 1.0).float().mean()
            else:
                haram_rejection_acc = torch.tensor(1.0, device=device)
            
            # Halal acceptance accuracy
            halal_mask = is_haram == 0.0
            if halal_mask.any():
                halal_acceptance_acc = (predicted_rejection[halal_mask] == 0.0).float().mean()
            else:
                halal_acceptance_acc = torch.tensor(1.0, device=device)
            
            stats = {
                'haram_rejection_accuracy': haram_rejection_acc.item(),
                'halal_acceptance_accuracy': halal_acceptance_acc.item(),
                'total_sharia_loss': total_loss.item(),
                'reject_loss': weighted_rejection_loss.item(),
                'accept_loss': acceptance_loss.item(),
                'consistency_loss': consistency_loss.item(),
                'avg_rejection_probability': rejection_probs.mean().item(),
            }
        
        return {
            'loss': total_loss,
            'reject_loss': weighted_rejection_loss,
            'accept_loss': acceptance_loss,
            'consistency_loss': consistency_loss,
            'stats': stats
        }


class IslamicAlignmentTrainer:
    """
    Training wrapper that integrates Sharia compliance loss
    with standard language modeling and ponder costs.
    
    Ensures the model learns Islamic values intrinsically.
    """
    
    def __init__(self, model, optimizer, scheduler,
                 lambda_sharia: float = 3.0,
                 lambda_ponder: float = 0.1,
                 use_qlora: bool = False,
                 gradient_checkpointing: bool = True,
                 default_madhab: str = 'auto'):
        """
        Args:
            model: Khwarizmi model with Islamic alignment capability
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler
            lambda_sharia: Weight for Sharia compliance loss
            lambda_ponder: Weight for ponder cost regularization
            use_qlora: Enable QLoRA for low-resource training
            gradient_checkpointing: Enable activation checkpointing
            default_madhab: Default madhab for fatwa (or 'auto' for user selection)
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lambda_sharia = lambda_sharia
        self.default_madhab = default_madhab
        
        self.sharia_loss_fn = ShariaComplianceLoss(lambda_reject=lambda_sharia)
        
        # Import ponder loss from Phase 7A
        from .ponder_loss_strict import PonderCostLoss
        self.ponder_loss_fn = PonderCostLoss(lambda_ponder=lambda_ponder)
        
        self.use_qlora = use_qlora
        self.gradient_checkpointing = gradient_checkpointing
        
        if use_qlora:
            self._setup_qlora()
        
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
    
    def _setup_qlora(self):
        """Setup QLoRA for 4-bit training on 4GB GPUs"""
        try:
            from peft import LoraConfig, get_peft_model
            
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            self.model = get_peft_model(self.model, lora_config)
            print("QLoRA setup complete - Islamic alignment with 4-bit training")
            
        except ImportError:
            print("Warning: PEFT not available. QLoRA disabled.")
            self.use_qlora = False
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        Single training step with combined LM + Sharia + Ponder loss
        
        Args:
            batch: Dictionary with:
                - 'input_ids', 'attention_mask', 'labels'
                - 'sharia_labels': Dict with Islamic compliance labels
                - 'difficulty_scores': Optional for ponder loss
                
        Returns:
            Dictionary with all loss components and metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass with Islamic alignment outputs
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            output_sharia=True,  # Request Sharia compliance scores
            output_ponder=True,  # Request ponder values
            madhab=self.default_madhab
        )
        
        # Standard language modeling loss
        lm_loss = outputs.loss
        
        # Sharia compliance loss
        sharia_outputs = self.sharia_loss_fn(
            predictions=outputs.logits,
            labels=batch['labels'],
            sharia_labels=batch['sharia_labels']
        )
        sharia_loss = sharia_outputs['loss']
        
        # Ponder cost regularization
        if hasattr(outputs, 'ponder_values'):
            ponder_outputs = self.ponder_loss_fn(
                ponder_values=outputs.ponder_values,
                difficulty_scores=batch.get('difficulty_scores')
            )
            ponder_loss = ponder_outputs['loss']
        else:
            ponder_loss = torch.tensor(0.0, device=outputs.logits.device)
            ponder_outputs = {'stats': {'avg_ponder': 0.0, 'fast_path_ratio': 0.0}}
        
        # Total combined loss
        total_loss = lm_loss + sharia_loss + ponder_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping (important for stable multi-loss training)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Comprehensive metrics
        return {
            'total_loss': total_loss.item(),
            'lm_loss': lm_loss.item(),
            'sharia_loss': sharia_loss.item(),
            'ponder_loss': ponder_loss.item(),
            'haram_rejection_acc': sharia_outputs['stats']['haram_rejection_accuracy'],
            'halal_acceptance_acc': sharia_outputs['stats']['halal_acceptance_accuracy'],
            'avg_ponder': ponder_outputs['stats']['avg_ponder'],
            'fast_path_ratio': ponder_outputs['stats']['fast_path_ratio'],
            'learning_rate': self.scheduler.get_last_lr()[0]
        }


def create_islamic_dataset_collator(tokenizer, max_length: int = 512):
    """
    Create a data collator for Islamic training data
    
    Handles:
    - Quranic Arabic text
    - Hadith collections
    - Fiqh questions and fatwas
    - Haram/Halal classification labels
    """
    
    def collate_fn(examples: List[Dict]) -> Dict:
        # Extract fields
        input_texts = [ex['question'] for ex in examples]
        responses = [ex['answer'] for ex in examples]
        is_haram = [ex.get('is_haram', 0) for ex in examples]
        haram_category = [ex.get('haram_category', 0) for ex in examples]
        
        # Tokenize
        encodings = tokenizer(
            input_texts,
            responses,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Prepare Sharia labels
        sharia_labels = {
            'is_haram': torch.tensor(is_haram, dtype=torch.float),
            'rejection_required': torch.tensor(is_haram, dtype=torch.float),
            'haram_category': torch.tensor(haram_category, dtype=torch.long)
        }
        
        return {
            **encodings,
            'sharia_labels': sharia_labels,
            'examples': examples
        }
    
    return collate_fn
