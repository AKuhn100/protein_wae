#!/usr/bin/env python3
"""
Training script for Protein WAE models.

Usage:
    python train.py --config config.json
    python train.py --fasta data.fasta --decoder causal
    python train.py --fasta data.fasta --decoder plm --epochs 100
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from protein_wae import (
    Tokenizer, WassersteinAutoencoder, ProteinDataset,
    Trainer, create_data_loaders
)
from protein_wae.config import Config, get_default_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Protein WAE models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to configuration JSON file"
    )
    
    # Data arguments
    parser.add_argument(
        "--fasta", type=str, default=None,
        help="Path to FASTA file (overrides config)"
    )
    parser.add_argument(
        "--min-len", type=int, default=None,
        help="Minimum sequence length"
    )
    parser.add_argument(
        "--max-len", type=int, default=None,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--val-fraction", type=float, default=None,
        help="Validation set fraction"
    )
    
    # Model arguments
    parser.add_argument(
        "--decoder", type=str, choices=["causal", "plm"], default=None,
        help="Decoder type"
    )
    parser.add_argument(
        "--latent-dim", type=int, default=None,
        help="Latent dimension"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Total epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Checkpoint directory"
    )
    
    # Other arguments
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed"
    )
    
    return parser.parse_args()


def update_config_from_args(config: Config, args) -> Config:
    """Update configuration with command line arguments."""
    # Data config
    if args.fasta is not None:
        config.data.fasta_path = args.fasta
    if args.min_len is not None:
        config.data.min_seq_len = args.min_len
    if args.max_len is not None:
        config.data.max_seq_len = args.max_len
    if args.val_fraction is not None:
        config.data.val_fraction = args.val_fraction
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.seed is not None:
        config.data.seed = args.seed
    
    # Model config
    if args.decoder is not None:
        config.model.decoder_type = args.decoder
    if args.latent_dim is not None:
        config.model.latent_dim = args.latent_dim
    
    # Training config
    if args.epochs is not None:
        config.training.total_epochs = args.epochs
        # Adjust phase epochs proportionally
        phase_fraction = args.epochs / 65  # Default total
        config.training.deterministic_epochs = int(10 * phase_fraction)
        config.training.mmd_ramp_epochs = int(5 * phase_fraction)
        config.training.mmd_full_epochs = args.epochs - config.training.deterministic_epochs - config.training.mmd_ramp_epochs
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.checkpoint_dir is not None:
        config.training.checkpoint_dir = args.checkpoint_dir
    
    return config


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load or create configuration
    if args.config:
        print(f"📋 Loading configuration from {args.config}")
        config = Config.load(args.config)
    else:
        print("📋 Using default configuration")
        decoder_type = args.decoder or "causal"
        config = get_default_config(decoder_type)
    
    # Update config with command line arguments
    config = update_config_from_args(config, args)
    
    # Validate configuration
    if not Path(config.data.fasta_path).exists():
        print(f"❌ Error: FASTA file not found: {config.data.fasta_path}")
        print("   Please specify a valid FASTA file using --fasta or in the config file")
        sys.exit(1)
    
    # Print configuration
    print("\n🔧 Configuration:")
    print(json.dumps(config.to_dict(), indent=2))
    
    # Save configuration
    config_save_path = Path(config.training.checkpoint_dir) / "config.json"
    config.save(config_save_path)
    print(f"\n💾 Configuration saved to {config_save_path}")
    
    # Initialize tokenizer
    print("\n🔤 Initializing tokenizer...")
    tokenizer = Tokenizer()
    print(f"   Vocabulary size: {tokenizer.vocab_size}")
    
    # Create data loaders
    print("\n📊 Creating data loaders...")
    train_loader, val_loader = create_data_loaders(config, tokenizer)
    
    # Initialize model
    print("\n🧠 Initializing model...")
    model = WassersteinAutoencoder(
        d_lat=config.model.latent_dim,
        enc_d_emb=config.model.encoder_embed_dim,
        enc_d_model=config.model.encoder_hidden_dim,
        dec_d_model=config.model.decoder_hidden_dim,
        max_len=config.data.max_seq_len + (2 if config.model.decoder_type == "causal" else 0),
        vocab_size=tokenizer.vocab_size,
        pad_idx=tokenizer.pad_idx,
        bos_idx=tokenizer.bos_idx,
        eos_idx=tokenizer.eos_idx,
        mask_idx=tokenizer.mask_idx if config.model.decoder_type == "plm" else None,
        decoder_type=config.model.decoder_type,
        n_heads=config.model.decoder_num_heads,
        n_layers=config.model.decoder_num_layers,
        dropout=config.model.dropout
    )
    
    # Initialize trainer
    print("\n🏋️ Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        tokenizer=tokenizer
    )
    
    # Start training
    print("\n🚀 Starting training...")
    trainer.train(resume=args.resume)
    
    print("\n✨ Training complete!")


if __name__ == "__main__":
    main()