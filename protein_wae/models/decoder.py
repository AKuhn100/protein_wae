"""
Decoder models for protein sequence generation.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import PositionalEncoding


class CausalAutoregressiveDecoder(nn.Module):
    """Causal autoregressive decoder with improved latent conditioning."""
    
    def __init__(self,
                 d_lat: int,
                 d_model: int,
                 vocab_size: int,
                 max_len: int,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 pad_idx: int = 0,
                 dropout: float = 0.1):
        """
        Initialize causal autoregressive decoder.
        
        Args:
            d_lat: Latent dimension
            d_model: Model dimension
            vocab_size: Size of vocabulary
            max_len: Maximum sequence length
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            pad_idx: Padding token index
            dropout: Dropout probability
        """
        super().__init__()
        
        self.max_len = max_len
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Token and position embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Latent conditioning layers
        # Option 1: Additive conditioning (simple and effective)
        self.latent_projection = nn.Sequential(
            nn.Linear(d_lat, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Option 2: FiLM-style modulation (commented out, but available)
        # self.latent_to_scale = nn.Linear(d_lat, d_model)
        # self.latent_to_bias = nn.Linear(d_lat, d_model)
        
        # Use transformer encoder layers (we don't need cross-attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, vocab_size)
        )
        
        # Create causal mask
        self.register_buffer("causal_mask", self._generate_causal_mask(max_len))
    
    def _generate_causal_mask(self, sz: int) -> torch.Tensor:
        """Generate a causal mask for autoregressive decoding."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, 
                z: torch.Tensor,
                decoder_input: torch.Tensor,
                target_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            z: Latent code (batch_size, d_lat)
            decoder_input: Input token IDs (batch_size, seq_len)
            target_lengths: Actual lengths of target sequences (batch_size,)
            
        Returns:
            Logits for next token prediction (batch_size, seq_len, vocab_size)
        """
        batch_size = z.size(0)
        seq_len = decoder_input.size(1)
        
        # Get token embeddings
        h = self.token_emb(decoder_input)  # (B, L, D)
        h = self.pos_encoder(h)
        
        # Apply latent conditioning
        # Method 1: Additive conditioning (broadcast latent to all positions)
        latent_emb = self.latent_projection(z).unsqueeze(1)  # (B, 1, D)
        h = h + latent_emb  # (B, L, D)
        
        # Method 2: FiLM-style modulation (alternative approach)
        # scale = self.latent_to_scale(z).unsqueeze(1).sigmoid() * 2  # (B, 1, D)
        # bias = self.latent_to_bias(z).unsqueeze(1)  # (B, 1, D)
        # h = h * scale + bias
        
        # Create masks
        causal_mask = self.causal_mask[:seq_len, :seq_len].to(dtype=h.dtype)
        if target_lengths is not None:
            padding_mask = torch.arange(seq_len, device=decoder_input.device).unsqueeze(0) >= target_lengths.unsqueeze(1)
        else:
            padding_mask = (decoder_input == self.pad_idx)
        
        # Apply transformer with causal masking
        h = self.transformer(
            src=h,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )
        
        # Project to vocabulary
        logits = self.output_projection(h)
        
        return logits
    
    def generate(self,
                 z: torch.Tensor,
                 max_length: int,
                 bos_idx: int,
                 eos_idx: int,
                 temperature: float = 1.0) -> torch.Tensor:
        """
        Generate sequences autoregressively.
        
        Args:
            z: Latent codes (batch_size, d_lat)
            max_length: Maximum length to generate
            bos_idx: Beginning of sequence token index
            eos_idx: End of sequence token index
            temperature: Sampling temperature
            
        Returns:
            Generated token IDs (batch_size, seq_len)
        """
        batch_size = z.size(0)
        device = z.device
        
        # Start with BOS token
        generated = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length - 1):
            # Forward pass
            logits = self.forward(z, generated)
            
            # Get next token probabilities
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, 1)
            
            # Update finished sequences
            finished = finished | (next_token.squeeze(-1) == eos_idx)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences are finished
            if finished.all():
                break
        
        return generated


class PermutationDecoder(nn.Module):
    """Permutation language model decoder with improved latent conditioning."""
    
    def __init__(self,
                 d_lat: int,
                 d_model: int,
                 vocab_size: int,
                 max_len: int,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 pad_idx: int = 0,
                 mask_idx: int = 4,
                 dropout: float = 0.1):
        """
        Initialize permutation decoder.
        
        Args:
            d_lat: Latent dimension
            d_model: Model dimension
            vocab_size: Size of vocabulary
            max_len: Maximum sequence length
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            pad_idx: Padding token index
            mask_idx: Mask token index
            dropout: Dropout probability
        """
        super().__init__()
        
        self.max_len = max_len
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        
        # Token and position embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Latent conditioning (same as causal decoder for consistency)
        self.latent_projection = nn.Sequential(
            nn.Linear(d_lat, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Transformer encoder (bidirectional for PLM)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
    
    def forward(self,
                z: torch.Tensor,
                decoder_input: torch.Tensor,
                target_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            z: Latent code (batch_size, d_lat)
            decoder_input: Input token IDs (batch_size, seq_len)
            target_lengths: Actual lengths of target sequences (batch_size,)
            
        Returns:
            Logits for token prediction (batch_size, seq_len, vocab_size)
        """
        # Get token embeddings
        h = self.token_emb(decoder_input)  # (B, L, D)
        h = self.pos_encoder(h)
        
        # Apply latent conditioning (additive)
        latent_emb = self.latent_projection(z).unsqueeze(1)  # (B, 1, D)
        h = h + latent_emb  # Broadcast to all positions
        
        # Create padding mask if needed
        if target_lengths is not None:
            seq_len = decoder_input.size(1)
            padding_mask = torch.arange(seq_len, device=decoder_input.device).unsqueeze(0) >= target_lengths.unsqueeze(1)
        else:
            padding_mask = (decoder_input == self.pad_idx)
        
        # Apply transformer (no causal mask for PLM)
        h = self.transformer(src=h, src_key_padding_mask=padding_mask)
        
        # Project to vocabulary
        logits = self.output_projection(h)
        
        return logits
    
    def generate(self,
                 z: torch.Tensor,
                 lengths: torch.Tensor,
                 temperature: float = 1.0,
                 num_iterations: Optional[int] = None) -> torch.Tensor:
        """
        Generate sequences using iterative refinement.
        
        Args:
            z: Latent codes (batch_size, d_lat)
            lengths: Lengths of sequences to generate (batch_size,)
            temperature: Sampling temperature
            num_iterations: Number of refinement iterations (default: max_len)
            
        Returns:
            Generated token IDs (batch_size, max_len)
        """
        batch_size = z.size(0)
        device = z.device
        
        if num_iterations is None:
            num_iterations = self.max_len
        
        # Start with all mask tokens
        seq = torch.full((batch_size, self.max_len), self.mask_idx, dtype=torch.long, device=device)
        
        # Generate random permutation for each sequence
        perm = torch.stack([torch.randperm(self.max_len, device=device) for _ in range(batch_size)])
        
        for i in range(num_iterations):
            # Positions to predict in this iteration
            pos_to_predict = perm[:, i]
            
            # Forward pass
            logits = self.forward(z, seq, lengths)
            
            # Get predictions for current positions
            batch_indices = torch.arange(batch_size, device=device)
            position_logits = logits[batch_indices, pos_to_predict] / temperature
            
            # Sample tokens
            probs = F.softmax(position_logits, dim=-1)
            pred_tokens = torch.multinomial(probs, 1).squeeze(-1)
            
            # Update sequence
            seq[batch_indices, pos_to_predict] = pred_tokens
            
            # Stop if we've filled all positions up to max length
            if (i + 1) >= lengths.max():
                break
        
        # Apply padding mask
        padding_mask = torch.arange(self.max_len, device=device)[None, :] >= lengths[:, None]
        seq.masked_fill_(padding_mask, self.pad_idx)
        
        return seq