# Protein WAE (Wasserstein Autoencoder)

A PyTorch implementation of Wasserstein Autoencoders for protein sequence generation, supporting both causal autoregressive and permutation language model decoders.

## Features

- **Wasserstein Autoencoder architecture** with MMD (Maximum Mean Discrepancy) regularization
- **Two decoder types**:
  - Causal autoregressive decoder for sequential generation
  - Permutation language model (PLM) decoder for non-autoregressive generation
- **Progressive training strategy** with deterministic warm-up
- **Efficient protein tokenization** with support for gaps and special tokens
- **Modular design** for easy extension and experimentation

## Installation

### From source

```bash
git clone https://github.com/AKuhn100/protein-wae.git
cd protein-wae
pip install .
```

### Requirements

- Python >= 3.8
- PyTorch >= 1.12.0
- NumPy >= 1.21.0
- tqdm >= 4.62.0

## Quick Start

### Training

```bash
# Train with causal decoder
python -m protein_wae.scripts.train \
    --fasta data/proteins.fasta \
    --decoder causal \
    --epochs 100 \
    --checkpoint-dir checkpoints/causal

# Train with PLM decoder
python -m protein_wae.scripts.train \
    --fasta data/proteins.fasta \
    --decoder plm \
    --epochs 100 \
    --checkpoint-dir checkpoints/plm

# Resume training
python -m protein_wae.scripts.train \
    --config checkpoints/causal/config.json \
    --resume
```

### Sampling

```bash
# Generate sequences
python -m protein_wae.scripts.sample \
    --checkpoint checkpoints/causal/wae_causal_best.pt \
    --num-samples 10000 \
    --output generated_sequences.fasta \
    --temperature 0.8
```

## Usage as a Library

```python
from protein_wae import Tokenizer, WassersteinAutoencoder, Trainer, Sampler
from protein_wae.config import get_default_config
from protein_wae.data import create_data_loaders

# Setup configuration
config = get_default_config(decoder_type="causal")
config.data.fasta_path = "data/proteins.fasta"

# Initialize tokenizer
tokenizer = Tokenizer()

# Create data loaders
train_loader, val_loader = create_data_loaders(config, tokenizer)

# Initialize model
model = WassersteinAutoencoder(
    d_lat=config.model.latent_dim,
    enc_d_emb=config.model.encoder_embed_dim,
    enc_d_model=config.model.encoder_hidden_dim,
    dec_d_model=config.model.decoder_hidden_dim,
    max_len=config.data.max_seq_len + 2,  # +2 for BOS/EOS
    vocab_size=tokenizer.vocab_size,
    pad_idx=tokenizer.pad_idx,
    bos_idx=tokenizer.bos_idx,
    eos_idx=tokenizer.eos_idx,
    decoder_type="causal"
)

# Train model
trainer = Trainer(model, train_loader, val_loader, config, tokenizer)
trainer.train()

# Generate sequences
sampler = Sampler(model, tokenizer)
sequences = sampler.generate(num_samples=1000, temperature=0.8)
```

## Configuration

The package uses a hierarchical configuration system. You can create a configuration file:

```json
{
  "data": {
    "fasta_path": "data/proteins.fasta",
    "min_seq_len": 100,
    "max_seq_len": 500,
    "batch_size": 64
  },
  "model": {
    "decoder_type": "causal",
    "latent_dim": 512,
    "encoder_hidden_dim": 512,
    "decoder_hidden_dim": 512
  },
  "training": {
    "total_epochs": 100,
    "learning_rate": 1e-4,
    "deterministic_epochs": 10,
    "mmd_weight": 10.0
  }
}
```

## Architecture

### Encoder
- Transformer-based encoder with convolutional feature extraction
- Outputs distributional parameters (mean and log-variance) for the latent space
- Supports both deterministic and stochastic encoding

### Decoders

#### Causal Autoregressive Decoder
- Generates sequences token by token from left to right
- Uses causal masking for autoregressive generation
- Suitable for high-quality sequence generation

#### Permutation Language Model Decoder
- Generates all positions in parallel with random ordering
- Non-autoregressive for faster generation
- Suitable for large-scale sampling

### Training Strategy
1. **Deterministic phase**: Train as standard autoencoder
2. **MMD ramp-up phase**: Gradually introduce Wasserstein regularization
3. **Full training phase**: Complete WAE training with MMD penalty

## License

MIT License

## Citation

If you use this code in your research, please cite:

```bibtex
@software{protein_wae,
  author = {Adam Kuhn},
  title = {Protein WAE: Wasserstein Autoencoder for Protein Sequences},
  year = {2024},
  url = {https://github.com/AKuhn100/protein-wae}
}
```
