"""
Loss functions for WAE training.
"""

from typing import Tuple

import torch
import torch.nn.functional as F


def compute_mmd_penalty(z_real: torch.Tensor, 
                       z_fake: torch.Tensor, 
                       kernel_bandwidth: float = 1.0) -> torch.Tensor:
    """
    Compute Maximum Mean Discrepancy (MMD) between real and fake samples
    using RBF kernel. This replaces the KL divergence in WAE.
    
    Args:
        z_real: Samples from the prior distribution
        z_fake: Samples from the encoder (latent codes)
        kernel_bandwidth: Bandwidth parameter for RBF kernel
        
    Returns:
        MMD squared distance
    """
    def rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
        """Compute RBF kernel matrix between x and y."""
        xx = torch.sum(x**2, dim=1, keepdim=True)
        yy = torch.sum(y**2, dim=1, keepdim=True)
        xy = torch.mm(x, y.t())
        
        sq_dist = xx - 2*xy + yy.t()
        return torch.exp(-sq_dist / (2 * sigma**2))
    
    # Compute kernel matrices
    k_xx = rbf_kernel(z_real, z_real, kernel_bandwidth)
    k_yy = rbf_kernel(z_fake, z_fake, kernel_bandwidth)
    k_xy = rbf_kernel(z_real, z_fake, kernel_bandwidth)
    
    # Compute MMD squared
    mmd_sq = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    
    return mmd_sq


def compute_reconstruction_loss(logits: torch.Tensor,
                               targets: torch.Tensor,
                               lengths: torch.Tensor,
                               pad_idx: int) -> torch.Tensor:
    """
    Compute masked reconstruction loss.
    
    Args:
        logits: Model predictions (batch_size, seq_len, vocab_size)
        targets: Target sequences (batch_size, seq_len)
        lengths: Actual sequence lengths (batch_size,)
        pad_idx: Padding token index
        
    Returns:
        Average reconstruction loss per sequence
    """
    batch_size = logits.size(0)
    seq_len = logits.size(1)
    
    # Create length mask
    length_mask = torch.arange(seq_len, device=logits.device).unsqueeze(0) < lengths.unsqueeze(1)
    
    # Compute cross entropy loss
    ce_loss = F.cross_entropy(
        logits.transpose(1, 2),  # (B, C, L)
        targets,
        ignore_index=pad_idx,
        reduction='none'
    )  # (B, L)
    
    # Apply length mask
    ce_loss = ce_loss * length_mask.float()
    
    # Average over sequence length
    ce_per_seq = ce_loss.sum(dim=1) / lengths.float()
    
    # Average over batch
    return ce_per_seq.mean()


def compute_loss(logits: torch.Tensor,
                targets: torch.Tensor,
                lengths: torch.Tensor,
                z_encoded: torch.Tensor,
                pad_idx: int,
                phase: str = "deterministic",
                mmd_weight: float = 0.0,
                kernel_bandwidth: float = 1.0) -> Tuple[torch.Tensor, float, float]:
    """
    Compute total loss for different training phases.
    
    Args:
        logits: Model predictions
        targets: Target sequences
        lengths: Sequence lengths
        z_encoded: Latent codes from encoder
        pad_idx: Padding token index
        phase: Training phase ("deterministic", "mmd")
        mmd_weight: Weight for MMD penalty
        kernel_bandwidth: Bandwidth for MMD kernel
        
    Returns:
        - total_loss: Total loss (tensor for backprop)
        - recon_loss: Reconstruction loss value
        - reg_loss: Regularization loss value
    """
    # Reconstruction loss
    recon_loss = compute_reconstruction_loss(logits, targets, lengths, pad_idx)
    
    # Regularization loss (depends on phase)
    if phase == "deterministic":
        # Small L2 penalty for stability
        reg_loss = torch.mean(z_encoded**2) * 0.001
        reg_weight = 1.0
    elif phase == "mmd":
        # Sample from prior
        z_prior = torch.randn_like(z_encoded)
        reg_loss = compute_mmd_penalty(z_prior, z_encoded, kernel_bandwidth)
        reg_weight = mmd_weight
    else:
        reg_loss = torch.tensor(0.0, device=logits.device)
        reg_weight = 0.0
    
    # Total loss
    total_loss = recon_loss + reg_weight * reg_loss
    
    return total_loss, recon_loss.item(), reg_loss.item()


def compute_plm_loss(logits: torch.Tensor,
                     targets: torch.Tensor,
                     masked_positions: torch.Tensor,
                     pad_idx: int) -> torch.Tensor:
    """
    Compute loss for Permutation Language Model decoder.
    
    Args:
        logits: Model predictions (batch_size, seq_len, vocab_size)
        targets: Target sequences (batch_size, seq_len)
        masked_positions: Boolean mask of positions to predict (batch_size, seq_len)
        pad_idx: Padding token index
        
    Returns:
        Average loss over masked positions
    """
    # Flatten for loss computation
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)
    mask_flat = masked_positions.view(-1)
    
    # Compute loss only on masked positions
    ce_loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=pad_idx,
        reduction='none'
    )
    
    # Apply mask
    masked_loss = ce_loss * mask_flat.float()
    
    # Average over masked positions
    num_masked = mask_flat.sum().clamp(min=1)
    return masked_loss.sum() / num_masked