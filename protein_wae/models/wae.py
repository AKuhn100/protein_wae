"""
Wasserstein Autoencoder model combining encoder and decoder.
"""

from typing import Dict, Tuple, Optional, Union

import torch
import torch.nn as nn

from .encoder import ProteinEncoder
from .decoder import CausalAutoregressiveDecoder, PermutationDecoder


class WassersteinAutoencoder(nn.Module):
    """Wasserstein Autoencoder for protein sequences."""
    
    def __init__(self,
                 d_lat: int,
                 enc_d_emb: int,
                 enc_d_model: int,
                 dec_d_model: int,
                 max_len: int,
                 vocab_size: int,
                 pad_idx: int,
                 bos_idx: int,
                 eos_idx: int,
                 mask_idx: Optional[int] = None,
                 decoder_type: str = "causal",
                 n_heads: int = 8,
                 n_layers: int = 4,
                 dropout: float = 0.1):
        """
        Initialize Wasserstein Autoencoder.
        
        Args:
            d_lat: Latent dimension
            enc_d_emb: Encoder embedding dimension
            enc_d_model: Encoder model dimension
            dec_d_model: Decoder model dimension
            max_len: Maximum sequence length
            vocab_size: Size of vocabulary
            pad_idx: Padding token index
            bos_idx: Beginning of sequence token index
            eos_idx: End of sequence token index
            mask_idx: Mask token index (required for PLM decoder)
            decoder_type: Type of decoder ("causal" or "plm")
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.decoder_type = decoder_type
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        
        # Initialize encoder
        self.encoder = ProteinEncoder(
            d_lat=d_lat,
            d_emb=enc_d_emb,
            d_model=enc_d_model,
            vocab_size=vocab_size,
            max_len=max_len,
            pad_idx=pad_idx,
            n_heads=n_heads,
            n_layers=6,  # Encoder typically has more layers
            dropout=dropout
        )
        
        # Initialize decoder based on type
        if decoder_type == "causal":
            self.decoder = CausalAutoregressiveDecoder(
                d_lat=d_lat,
                d_model=dec_d_model,
                vocab_size=vocab_size,
                max_len=max_len,
                n_heads=n_heads,
                n_layers=n_layers,
                pad_idx=pad_idx,
                dropout=dropout
            )
        elif decoder_type == "plm":
            if mask_idx is None:
                raise ValueError("mask_idx must be provided for PLM decoder")
            self.decoder = PermutationDecoder(
                d_lat=d_lat,
                d_model=dec_d_model,
                vocab_size=vocab_size,
                max_len=max_len,
                n_heads=n_heads,
                n_layers=n_layers,
                pad_idx=pad_idx,
                mask_idx=mask_idx,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")
    
    def forward(self,
                batch: Dict[str, torch.Tensor],
                deterministic: bool = False,
                noise_scale: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through WAE.
        
        Args:
            batch: Dictionary containing:
                - encoder_input: One-hot encoded sequences
                - decoder_input: Decoder input sequences
                - length: Sequence lengths
            deterministic: If True, use deterministic encoding
            noise_scale: Scale of noise to add to latent codes
            
        Returns:
            - logits: Decoder output logits
            - z: Latent codes
        """
        enc_input = batch["encoder_input"]
        dec_input = batch["decoder_input"]
        lengths = batch["length"]
        
        # Adjust lengths for encoder (doesn't see EOS in causal mode)
        enc_lengths = lengths - 1 if self.decoder_type == "causal" else lengths
        
        if deterministic:
            # Deterministic encoding
            z, _, _ = self.encoder(enc_input, enc_lengths, deterministic=True)
            
            # Add noise if specified
            if noise_scale > 0:
                z = z + torch.randn_like(z) * noise_scale
        else:
            # Stochastic encoding
            mu, logvar, _ = self.encoder(enc_input, enc_lengths, deterministic=False)
            z = self.encoder.sample_latent(mu, logvar)
        
        # Decode
        logits = self.decoder(z, dec_input, lengths)
        
        return logits, z
    
    def encode(self,
               x_one_hot: torch.Tensor,
               lengths: torch.Tensor,
               deterministic: bool = True) -> torch.Tensor:
        """
        Encode sequences to latent space.
        
        Args:
            x_one_hot: One-hot encoded sequences
            lengths: Sequence lengths
            deterministic: If True, return mean; else sample
            
        Returns:
            Latent codes
        """
        if deterministic:
            z, _, _ = self.encoder(x_one_hot, lengths, deterministic=True)
            return z
        else:
            mu, logvar, _ = self.encoder(x_one_hot, lengths, deterministic=False)
            return self.encoder.sample_latent(mu, logvar)
    
    def decode(self,
               z: torch.Tensor,
               lengths: Optional[torch.Tensor] = None,
               temperature: float = 1.0) -> torch.Tensor:
        """
        Decode latent codes to sequences.
        
        Args:
            z: Latent codes
            lengths: Sequence lengths (required for PLM)
            temperature: Sampling temperature
            
        Returns:
            Generated sequences
        """
        if self.decoder_type == "causal":
            max_length = self.decoder.max_len
            return self.decoder.generate(
                z, max_length, self.bos_idx, self.eos_idx, temperature
            )
        else:  # PLM
            if lengths is None:
                raise ValueError("Lengths required for PLM generation")
            return self.decoder.generate(z, lengths, temperature)
    
    def generate(self,
                 z: torch.Tensor,
                 lengths: Optional[torch.Tensor] = None,
                 temperature: float = 1.0) -> torch.Tensor:
        """
        Generate sequences from latent codes.
        
        Alias for decode() method for compatibility.
        """
        return self.decode(z, lengths, temperature)
    
    def get_latent_params(self,
                          batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get parameters of latent distribution.
        
        Args:
            batch: Input batch
            
        Returns:
            - mu: Mean of latent distribution
            - logvar: Log variance of latent distribution
        """
        enc_input = batch["encoder_input"]
        lengths = batch["length"]
        
        # Adjust lengths for encoder
        enc_lengths = lengths - 1 if self.decoder_type == "causal" else lengths
        
        mu, logvar, _ = self.encoder(enc_input, enc_lengths, deterministic=False)
        return mu, logvar