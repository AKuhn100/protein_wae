"""
Encoder model for protein sequences.
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn

from .layers import PositionalEncoding, MultiLayerConv


class ProteinEncoder(nn.Module):
    """Transformer-based encoder for protein sequences."""
    
    def __init__(self,
                 d_lat: int,
                 d_emb: int,
                 d_model: int,
                 vocab_size: int,
                 max_len: int,
                 pad_idx: int,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 dropout: float = 0.1):
        """
        Initialize protein encoder.
        
        Args:
            d_lat: Latent dimension
            d_emb: Embedding dimension
            d_model: Model/hidden dimension
            vocab_size: Size of vocabulary
            max_len: Maximum sequence length
            pad_idx: Padding token index
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.max_len = max_len
        self.pad_idx = pad_idx
        
        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, d_emb, padding_idx=pad_idx)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_emb, max_len)
        
        # Convolutional feature extraction
        self.conv = MultiLayerConv(d_emb, d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projections
        self.deterministic_proj = nn.Linear(d_model, d_lat)
        self.mu = nn.Linear(d_model, d_lat)
        self.logvar = nn.Linear(d_model, d_lat)
    
    def forward(self, 
                x_one_hot: torch.Tensor,
                lengths: torch.Tensor,
                deterministic: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Encode protein sequences.
        
        Args:
            x_one_hot: One-hot encoded sequences (batch_size, vocab_size, seq_len)
            lengths: Actual lengths of sequences (batch_size,)
            deterministic: If True, return deterministic encoding; else return distribution parameters
            
        Returns:
            If deterministic:
                - z: Latent encoding (batch_size, d_lat)
                - None
                - None
            Else:
                - mu: Mean of latent distribution (batch_size, d_lat)
                - logvar: Log variance of latent distribution (batch_size, d_lat)
                - h_pooled: Pooled hidden states (batch_size, d_model)
        """
        # Convert one-hot to token IDs
        token_ids = x_one_hot.argmax(dim=1)  # (batch_size, seq_len)
        
        # Get embeddings
        h = self.token_emb(token_ids)  # (batch_size, seq_len, d_emb)
        h = self.pos_encoder(h)
        
        # Create padding mask
        padding_mask = (token_ids == self.pad_idx)
        
        # Apply convolutional layers
        h = self.conv(h)  # (batch_size, seq_len, d_model)
        
        # Apply transformer
        h = self.transformer(h, src_key_padding_mask=padding_mask)
        
        # Masked mean pooling
        mask_expanded = padding_mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
        h_masked = h.masked_fill(mask_expanded, 0)
        h_sum = h_masked.sum(dim=1)  # (batch_size, d_model)
        h_pooled = h_sum / lengths.float().unsqueeze(-1).clamp(min=1)
        
        if deterministic:
            z = self.deterministic_proj(h_pooled)
            return z, None, None
        else:
            mu = self.mu(h_pooled)
            logvar = self.logvar(h_pooled)
            return mu, logvar, h_pooled
    
    def sample_latent(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Sample from the latent distribution using reparameterization trick.
        
        Args:
            mu: Mean of distribution
            logvar: Log variance of distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std