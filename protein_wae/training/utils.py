"""
Training utilities and helper functions.
"""

from typing import Tuple, Dict, List, Optional
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def setup_device() -> Tuple[torch.device, int, bool]:
    """
    Setup and detect available devices.
    
    Returns:
        - device: PyTorch device
        - num_gpus: Number of available GPUs
        - use_amp: Whether to use automatic mixed precision
    """
    print("\n🔍 Device Detection:")
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"   ✅ CUDA available with {num_gpus} GPU(s)")
        device = torch.device("cuda")
        use_amp = True
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("   ✅ Apple Silicon GPU (MPS) available")
        device = torch.device("mps")
        use_amp = False  # AMP not fully supported on MPS yet
        num_gpus = 1
    else:
        print("   ℹ️ No GPU available, using CPU")
        device = torch.device("cpu")
        use_amp = False
        num_gpus = 0
        
    return device, num_gpus, use_amp


def get_training_phase(epoch: int, config) -> Tuple[str, float, float]:
    """
    Determine training phase and parameters based on epoch.
    
    Args:
        epoch: Current epoch (1-indexed)
        config: Training configuration
        
    Returns:
        - phase: Training phase name
        - mmd_weight: MMD penalty weight
        - noise_scale: Noise scale for latent codes
    """
    if epoch <= config.training.deterministic_epochs:
        return "deterministic", 0.0, 0.0
    elif epoch <= config.training.deterministic_epochs + config.training.mmd_ramp_epochs:
        # Ramp up MMD weight
        progress = (epoch - config.training.deterministic_epochs) / config.training.mmd_ramp_epochs
        mmd_weight = config.training.mmd_weight * progress
        return "mmd", mmd_weight, 0.0
    else:
        return "mmd", config.training.mmd_weight, 0.0


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters in model and its components.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    # Access the actual model if it's wrapped in DataParallel
    actual_model = model.module if hasattr(model, 'module') else model

    total = sum(p.numel() for p in actual_model.parameters())
    encoder = sum(p.numel() for p in actual_model.encoder.parameters())
    decoder = sum(p.numel() for p in actual_model.decoder.parameters())
    
    return {
        "total": total,
        "encoder": encoder,
        "decoder": decoder
    }


def analyze_latent_space(model: nn.Module,
                        val_loader: DataLoader,
                        tokenizer,
                        device: torch.device,
                        epoch: int,
                        config,
                        num_examples: int = 3) -> float:
    """
    Analyze latent representations and generation quality.
    
    Args:
        model: WAE model
        val_loader: Validation data loader
        tokenizer: Tokenizer instance
        device: PyTorch device
        epoch: Current epoch
        config: Configuration
        num_examples: Number of examples to show
        
    Returns:
        Mean reconstruction accuracy
    """
    print(f"\n🔬 Latent Analysis - Epoch {epoch}")
    
    model.eval()
    
    phase, _, _ = get_training_phase(epoch, config)
    deterministic = (phase == "deterministic")
    
    latents = []
    accuracies = []
    examples = []
    gap_patterns = []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 5:  # Analyze first 5 batches
                break
            
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            logits, z = model(batch, deterministic=deterministic)
            predicted_tokens = torch.argmax(logits, dim=-1)
            
            latents.append(z.cpu().numpy())
            
            # Test generation for first batch
            if i == 0 and config.model.decoder_type == "causal":
                generated = model.decoder.generate(
                    z[:num_examples],
                    max_length=batch["decoder_input"].size(1),
                    bos_idx=tokenizer.bos_idx,
                    eos_idx=tokenizer.eos_idx,
                    temperature=1.0
                )
            
            # Analyze sequences
            for j in range(min(num_examples, z.size(0))):
                length = batch["length"][j].item()
                pred_seq = predicted_tokens[j, :length].cpu().tolist()
                target_seq = batch["target_sequence"][j, :length].cpu().tolist()
                
                # Calculate accuracy
                correct = sum(1 for p, t in zip(pred_seq, target_seq) if p == t)
                accuracy = correct / length if length > 0 else 0.0
                accuracies.append(accuracy)
                
                # Count gaps
                pred_gaps = sum(1 for p in pred_seq if p == tokenizer.gap_idx)
                target_gaps = sum(1 for t in target_seq if t == tokenizer.gap_idx)
                gap_patterns.append((target_gaps, pred_gaps))
                
                if len(examples) < num_examples:
                    pred_str = tokenizer.decode(pred_seq)
                    target_str = tokenizer.decode(target_seq)
                    
                    # Include autoregressive generation for causal decoder
                    if i == 0 and j < num_examples and config.model.decoder_type == "causal":
                        gen_str = tokenizer.decode(generated[j].cpu().tolist())
                        examples.append((target_str, pred_str, gen_str, accuracy))
                    else:
                        examples.append((target_str, pred_str, "", accuracy))
    
    # Analyze latent statistics
    latents = np.concatenate(latents, axis=0)
    
    print(f"   📊 Latent Statistics (mode: {phase}):")
    print(f"      Mean: {latents.mean(axis=0)[:3]}")
    print(f"      Std:  {latents.std(axis=0)[:3]}")
    print(f"      Range: [{latents.min():.3f}, {latents.max():.3f}]")
    
    # Pairwise distances
    if len(latents) >= 50:
        dists = np.linalg.norm(latents[:25] - latents[25:50], axis=1)
        print(f"      Pairwise distances: {dists.mean():.3f} ± {dists.std():.3f}")
    
    # Reconstruction accuracy
    mean_acc = np.mean(accuracies) if accuracies else 0.0
    print(f"   🎯 Reconstruction accuracy: {mean_acc:.3f}")
    
    # Gap analysis
    if gap_patterns:
        target_gaps_mean = np.mean([g[0] for g in gap_patterns])
        pred_gaps_mean = np.mean([g[1] for g in gap_patterns])
        print(f"   🔷 Gap patterns: Target {target_gaps_mean:.1f} → Predicted {pred_gaps_mean:.1f}")
    
    # Show examples
    print("   📝 Examples:")
    for i, ex in enumerate(examples):
        if len(ex) == 4:
            orig, recon, gen, acc = ex
            print(f"   {i+1}. Accuracy: {acc:.3f}")
            print(f"      Original: {orig[:50]}...")
            print(f"      Teacher:  {recon[:50]}...")
            if gen:
                print(f"      Autogen:  {gen[:50]}...")
        else:
            orig, recon, _, acc = ex
            print(f"   {i+1}. {acc:.3f}: {orig[:30]}... → {recon[:30]}...")
    
    model.train()
    return mean_acc


def save_checkpoint(model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   best_val_loss: float,
                   config,
                   path: str):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        best_val_loss: Best validation loss so far
        config: Configuration
        path: Path to save checkpoint
    """
    # Handle DataParallel
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'config': config.to_dict()
    }
    
    torch.save(checkpoint, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer = None) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        path: Path to checkpoint
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(path, map_location='cpu')
    
    # Handle potential DataParallel mismatch
    state_dict = checkpoint['model_state_dict']
    
    # Remove 'module.' prefix if present
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint