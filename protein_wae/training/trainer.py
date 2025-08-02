"""
Main trainer class for WAE models.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from .losses import compute_loss
from .utils import (
    setup_device, get_training_phase, count_parameters,
    analyze_latent_space, save_checkpoint, load_checkpoint
)


class Trainer:
    """Trainer for Wasserstein Autoencoder models."""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config,
                 tokenizer,
                 device: Optional[torch.device] = None,
                 num_gpus: Optional[int] = None,
                 use_amp: Optional[bool] = None):
        """
        Initialize trainer.
        
        Args:
            model: WAE model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration object
            tokenizer: Tokenizer instance
            device: PyTorch device (auto-detected if None)
            num_gpus: Number of GPUs (auto-detected if None)
            use_amp: Whether to use AMP (auto-detected if None)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.tokenizer = tokenizer
        
        # Setup device if not provided
        if device is None:
            device, num_gpus, use_amp = setup_device()
        self.device = device
        self.num_gpus = num_gpus if num_gpus is not None else 0
        self.use_amp = use_amp if use_amp is not None else False
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup multi-GPU if available
        if self.num_gpus > 1:
            print(f"   Using {self.num_gpus} GPUs with DataParallel")
            self.model = nn.DataParallel(self.model)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate
        )
        
        # Setup scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.training.scheduler_factor,
            patience=config.training.scheduler_patience,
            verbose=True
        )
        
        # Setup AMP scaler
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        
        # Setup paths
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        decoder_suffix = "_plm" if config.model.decoder_type == "plm" else "_causal"
        self.best_path = self.checkpoint_dir / f"wae{decoder_suffix}_best.pt"
        self.last_path = self.checkpoint_dir / f"wae{decoder_suffix}_last.pt"
        self.log_path = self.checkpoint_dir / f"training_log{decoder_suffix}.json"
        
        # Training state
        self.history = []
        self.best_val_loss = float('inf')
        self.start_epoch = 1
        
        # Count parameters
        params = count_parameters(self.model)
        print(f"\n🧠 Model Parameters:")
        print(f"   Total: {params['total']:,}")
        print(f"   Encoder: {params['encoder']:,}")
        print(f"   Decoder ({config.model.decoder_type}): {params['decoder']:,}")
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average reconstruction and regularization losses
        """
        self.model.train()
        
        phase, mmd_weight, noise_scale = get_training_phase(epoch, self.config)
        deterministic = (phase == "deterministic")
        
        total_recon_loss = 0.0
        total_reg_loss = 0.0
        num_samples = 0
        
        desc = f"Train E{epoch} {phase}"
        if phase == "mmd":
            desc += f" MMD={mmd_weight:.1f}"
        
        loop = tqdm(self.train_loader, leave=False, desc=desc)
        
        for batch in loop:
            # Move batch to device
            batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            batch_size = batch["encoder_input"].size(0)
            
            # Forward pass
            if self.use_amp:
                with torch.autocast(device_type='cuda'):
                    logits, z = self.model(batch, deterministic=deterministic, noise_scale=noise_scale)
                    loss, recon_loss, reg_loss = compute_loss(
                        logits, batch["target_sequence"], batch["length"],
                        z, self.tokenizer.pad_idx, phase, mmd_weight,
                        self.config.training.mmd_kernel_bandwidth
                    )
            else:
                logits, z = self.model(batch, deterministic=deterministic, noise_scale=noise_scale)
                loss, recon_loss, reg_loss = compute_loss(
                    logits, batch["target_sequence"], batch["length"],
                    z, self.tokenizer.pad_idx, phase, mmd_weight,
                    self.config.training.mmd_kernel_bandwidth
                )
            
            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip)
                self.optimizer.step()
            
            # Update statistics
            total_recon_loss += recon_loss * batch_size
            total_reg_loss += reg_loss * batch_size
            num_samples += batch_size
            
            # Update progress bar
            loop.set_postfix(recon=f"{recon_loss:.3f}", reg=f"{reg_loss:.3f}")
        
        return total_recon_loss / num_samples, total_reg_loss / num_samples
    
    @torch.no_grad()
    def evaluate(self, epoch: int) -> Tuple[float, float]:
        """
        Evaluate on validation set.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average reconstruction and regularization losses
        """
        self.model.eval()
        
        phase, mmd_weight, noise_scale = get_training_phase(epoch, self.config)
        deterministic = (phase == "deterministic")
        
        total_recon_loss = 0.0
        total_reg_loss = 0.0
        num_samples = 0
        
        desc = f"Val E{epoch} {phase}"
        if phase == "mmd":
            desc += f" MMD={mmd_weight:.1f}"
        
        loop = tqdm(self.val_loader, leave=False, desc=desc)
        
        for batch in loop:
            batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            batch_size = batch["encoder_input"].size(0)
            
            # Forward pass
            logits, z = self.model(batch, deterministic=deterministic, noise_scale=noise_scale)
            loss, recon_loss, reg_loss = compute_loss(
                logits, batch["target_sequence"], batch["length"],
                z, self.tokenizer.pad_idx, phase, mmd_weight,
                self.config.training.mmd_kernel_bandwidth
            )
            
            # Update statistics
            total_recon_loss += recon_loss * batch_size
            total_reg_loss += reg_loss * batch_size
            num_samples += batch_size
            
            loop.set_postfix(recon=f"{recon_loss:.3f}", reg=f"{reg_loss:.3f}")
        
        return total_recon_loss / num_samples, total_reg_loss / num_samples
    
    def train(self, resume: bool = False):
        """
        Main training loop.
        
        Args:
            resume: Whether to resume from checkpoint
        """
        # Resume if requested
        if resume and self.last_path.exists():
            print(f"\n📂 Resuming from {self.last_path}")
            checkpoint = load_checkpoint(self.last_path, self.model, self.optimizer)
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint['best_val_loss']
            print(f"   Resumed from epoch {checkpoint['epoch']}, best val loss: {self.best_val_loss:.4f}")
        
        # Print training plan
        print(f"\n🔄 Starting Progressive WAE Training...")
        print(f"   Phase 1: Deterministic ({self.config.training.deterministic_epochs} epochs)")
        print(f"   Phase 2: MMD Ramp-up ({self.config.training.mmd_ramp_epochs} epochs)")
        print(f"   Phase 3: Full MMD ({self.config.training.mmd_full_epochs} epochs)")
        print(f"   Total: {self.config.training.total_epochs} epochs")
        
        # Training loop
        for epoch in range(self.start_epoch, self.config.training.total_epochs + 1):
            # Train
            train_recon, train_reg = self.train_epoch(epoch)
            train_loss = train_recon + train_reg
            
            # Evaluate
            val_recon, val_reg = self.evaluate(epoch)
            val_loss = val_recon + val_reg
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Get phase info
            phase, mmd_weight, _ = get_training_phase(epoch, self.config)
            phase_desc = f"{phase}"
            if phase == "mmd":
                phase_desc += f"(λ={mmd_weight:.1f})"
            
            # Print results
            print(f"Ep {epoch:03d}/{self.config.training.total_epochs} | {phase_desc:15} | "
                  f"Train: {train_loss:.4f} (R:{train_recon:.4f}+Reg:{train_reg:.4f}) | "
                  f"Val: {val_loss:.4f} (R:{val_recon:.4f}+Reg:{val_reg:.4f})")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if self.config.training.save_best:
                    save_checkpoint(
                        self.model, self.optimizer, epoch,
                        self.best_val_loss, self.config, self.best_path
                    )
                print(f"   🏆 New best: {self.best_val_loss:.4f}")
            
            # Periodic analysis
            phase_transitions = [
                1,
                self.config.training.deterministic_epochs,
                self.config.training.deterministic_epochs + self.config.training.mmd_ramp_epochs
            ]
            if epoch in phase_transitions or epoch % self.config.training.analysis_interval == 0:
                analyze_latent_space(
                    self.model, self.val_loader, self.tokenizer,
                    self.device, epoch, self.config
                )
            
            # Log history
            self.history.append({
                'epoch': epoch,
                'phase': phase,
                'mmd_weight': mmd_weight,
                'train_recon': train_recon,
                'train_reg': train_reg,
                'val_recon': val_recon,
                'val_reg': val_reg,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
            
            # Save history
            with open(self.log_path, 'w') as f:
                json.dump({
                    'history': self.history,
                    'config': self.config.to_dict(),
                    'best_val_loss': self.best_val_loss
                }, f, indent=2)
            
            # Save last checkpoint
            if self.config.training.save_last:
                save_checkpoint(
                    self.model, self.optimizer, epoch,
                    self.best_val_loss, self.config, self.last_path
                )
        
        print(f"\n✅ Training complete! Best val loss: {self.best_val_loss:.4f}")
        print(f"   Models saved to {self.checkpoint_dir}")
        print(f"   Best checkpoint: {self.best_path}")
        print(f"   Last checkpoint: {self.last_path}")