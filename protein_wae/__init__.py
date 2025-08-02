"""
Protein Wasserstein Autoencoder (WAE) Package

A PyTorch implementation of Wasserstein Autoencoders for protein sequence generation
with support for both causal autoregressive and permutation language model decoders.
"""

__version__ = "0.1.0"
__author__ = "Adam Kuhn"

from .tokenizer import Tokenizer
from .models import WassersteinAutoencoder, ProteinEncoder, CausalAutoregressiveDecoder, PermutationDecoder
from .data import ProteinDataset
from .training import Trainer
from .sampling import Sampler

__all__ = [
    "Tokenizer",
    "WassersteinAutoencoder",
    "ProteinEncoder", 
    "CausalAutoregressiveDecoder",
    "PermutationDecoder",
    "ProteinDataset",
    "Trainer",
    "Sampler"
]