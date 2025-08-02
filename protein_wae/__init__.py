"""
Protein Wasserstein Autoencoder (WAE) Package

A PyTorch implementation of Wasserstein Autoencoders for protein sequence generation
with support for both causal autoregressive and permutation language model decoders.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .tokenizer import Tokenizer
from .models import WassersteinAutoencoder, ProteinEncoder, CausalAutoregressiveDecoder, PermutationDecoder
from .data import ProteinDataset, create_data_loaders
from .training import Trainer
from .sampling import Sampler

__all__ = [
    "Tokenizer",
    "WassersteinAutoencoder",
    "ProteinEncoder", 
    "CausalAutoregressiveDecoder",
    "PermutationDecoder",
    "ProteinDataset",
    "create_data_loaders",
    "Trainer",
    "Sampler"
]