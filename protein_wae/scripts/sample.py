#!/usr/bin/env python3
"""
Sampling script for generating sequences from trained Protein WAE models.

Usage:
    python sample.py --checkpoint model.pt --num-samples 1000
    python sample.py --checkpoint model.pt --config config.json
    python sample.py --checkpoint model.pt -n 10000 --temperature 0.8
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from protein_wae import Tokenizer, WassersteinAutoencoder, Sampler
from protein_wae.config import Config, SamplingConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate sequences from trained Protein WAE models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    
    # Config file
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to configuration JSON file"
    )
    
    # Sampling arguments
    parser.add_argument(
        "-n", "--num-samples", type=int, default=1000,
        help="Number of sequences to generate"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="generated_samples.fasta",
        help="Output FASTA file path"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--sampling-std", type=float, default=1.0,
        help="Standard deviation for latent sampling"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--sequence-length", type=int, default=None,
        help="Fixed sequence length (for PLM models)"
    )
    parser.add_argument(
        "--prefix", type=str, default="sample",
        help="Prefix for sequence IDs"
    )
    
    # Model architecture (if not using config)
    parser.add_argument(
        "--decoder-type", type=str, choices=["causal", "plm"], default=None,
        help="Decoder type (if not using config)"
    )
    parser.add_argument(
        "--latent-dim", type=int, default=512,
        help="Latent dimension (if not using config)"
    )
    parser.add_argument(
        "--max-len", type=int, default=305,
        help="Maximum sequence length (if not using config)"
    )
    
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, config: Config = None) -> WassersteinAutoencoder:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration (will try to load from checkpoint if None)
        
    Returns:
        Loaded model
    """
    import torch
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Try to get config from checkpoint
    if config is None and 'config' in checkpoint:
        print("📋 Loading configuration from checkpoint...")
        config = Config.from_dict(checkpoint['config'])
    elif config is None:
        print("⚠️  No configuration found in checkpoint, using defaults")
        config = Config(
            data=None,  # Not needed for sampling
            model=None,  # Will be set from args
            training=None  # Not needed for sampling
        )
    
    # Initialize tokenizer
    tokenizer = Tokenizer()
    
    # Determine model architecture
    if hasattr(config.model, 'decoder_type'):
        decoder_type = config.model.decoder_type
        latent_dim = config.model.latent_dim
        max_len = config.data.max_seq_len if config.data else 305
    else:
        # Try to infer from state dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        decoder_type = "plm" if any('mask_idx' in k for k in state_dict.keys()) else "causal"
        
        # Get dimensions from state dict
        for k, v in state_dict.items():
            if 'deterministic_proj.weight' in k:
                latent_dim = v.shape[0]
                break
        else:
            latent_dim = 512  # Default
        
        max_len = 305  # Default
    
    # Adjust max_len for causal decoder
    model_max_len = max_len + 2 if decoder_type == "causal" else max_len
    
    print(f"\n🔧 Model Architecture:")
    print(f"   Decoder type: {decoder_type}")
    print(f"   Latent dimension: {latent_dim}")
    print(f"   Max sequence length: {max_len}")
    
    # Initialize model
    model = WassersteinAutoencoder(
        d_lat=latent_dim,
        enc_d_emb=256,  # Default values
        enc_d_model=512,
        dec_d_model=512,
        max_len=model_max_len,
        vocab_size=tokenizer.vocab_size,
        pad_idx=tokenizer.pad_idx,
        bos_idx=tokenizer.bos_idx,
        eos_idx=tokenizer.eos_idx,
        mask_idx=tokenizer.mask_idx if decoder_type == "plm" else None,
        decoder_type=decoder_type,
        n_heads=8,
        n_layers=4,
        dropout=0.1
    )
    
    return model, tokenizer, decoder_type


def main():
    """Main sampling function."""
    # Parse arguments
    args = parse_args()
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Load configuration if provided
    config = None
    if args.config:
        print(f"📋 Loading configuration from {args.config}")
        config = Config.load(args.config)
    
    # Load model
    print(f"\n🧠 Loading model from {args.checkpoint}")
    model, tokenizer, decoder_type = load_model_from_checkpoint(args.checkpoint, config)
    
    # Override decoder type if specified
    if args.decoder_type:
        decoder_type = args.decoder_type
    
    # Create sampler
    sampler = Sampler.from_checkpoint(
        checkpoint_path=args.checkpoint,
        model=model,
        tokenizer=tokenizer
    )
    
    # Print sampling configuration
    print(f"\n🎲 Sampling Configuration:")
    print(f"   Number of samples: {args.num_samples}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Sampling std: {args.sampling_std}")
    print(f"   Temperature: {args.temperature}")
    print(f"   Output path: {args.output}")
    
    # Set sequence length for PLM
    sequence_length = args.sequence_length
    if decoder_type == "plm" and sequence_length is None:
        sequence_length = args.max_len
        print(f"   Sequence length (PLM): {sequence_length}")
    
    # Generate and save sequences
    sampler.sample_and_save(
        num_samples=args.num_samples,
        output_path=args.output,
        batch_size=args.batch_size,
        sampling_std=args.sampling_std,
        temperature=args.temperature,
        sequence_length=sequence_length,
        prefix=args.prefix
    )
    
    print("\n✨ Generation complete!")


if __name__ == "__main__":
    main()