"""
Khwarizmi Subword and Positional Embeddings Module.

Provides token embeddings and positional encodings for sequence input
representations into the Khwarizmi core architecture.
"""

import math
import torch
import torch.nn as nn
from typing import Optional

from ..config.settings import KhwarizmiConfig


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for sequence stabilization.
    Produces deterministic positional vectors without introducing extra parameters.
    """

    def __init__(self, d_model: int, max_seq_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_len, d_model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add sinusoidal positional encodings to token embeddings.
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Positively encoded tensor of same shape.
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(dtype=x.dtype)


class KhwarizmiEmbeddings(nn.Module):
    """
    Input representation pipeline for Khwarizmi AI.
    Combines token embeddings, sinusoidal positional encodings, dropout, and normalization.
    """

    def __init__(self, config: KhwarizmiConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )
        self.positional_encoding = SinusoidalPositionalEncoding(
            config.d_model,
            max_seq_len=config.max_seq_len,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Transform token indices into normalized feature embeddings.

        Args:
            input_ids: LongTensor of shape (batch_size, seq_len) with token indices.

        Returns:
            Tensor of shape (batch_size, seq_len, d_model).
        """
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be 2-dimensional (batch_size, seq_len), got shape {input_ids.shape}"
            )
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x
