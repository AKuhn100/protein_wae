"""
Sampler for generating sequences from trained WAE models.
"""

from pathlib import Path
from typing import List, Optional, Union
from collections import OrderedDict

import torch
import torch.nn as nn
from tqdm.auto import tqdm


class Sampler:
    """Sampler for generating protein sequences from trained WAE models."""
    
    def __init__(self,
                 model: nn.Module,
                 tokenizer,
                 device: Optional[torch.device] = None):
        """
        Initialize sampler.
        
        Args:
            model: Trained WAE model
            tokenizer: Tokenizer instance
            device: PyTorch device (auto-detected if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # Setup device
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print("✅ Using GPU for generation.")
            else:
                device = torch.device("cpu")
                print("ℹ️ Using CPU for generation.")
        self.device = device
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
    
    @classmethod
    def from_checkpoint(cls,
                       checkpoint_path: str,
                       model: nn.Module,
                       tokenizer,
                       device: Optional[torch.device] = None) -> "Sampler":
        """
        Create sampler from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            model: Model instance (architecture must match checkpoint)
            tokenizer: Tokenizer instance
            device: PyTorch device
            
        Returns:
            Initialized sampler
        """
        print(f"💾 Loading model weights from {checkpoint_path}...")
        
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle checkpoint format
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            print("\n⚠️  Warning: Could not load state_dict directly.")
            print("    Attempting to handle DataParallel mismatch...")
            
            # Try to handle DataParallel prefix
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            # Remove 'module.' prefix if present
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict)
            print("    ✅ Successfully loaded weights after modification.")
        
        return cls(model, tokenizer, device)
    
    @torch.no_grad()
    def sample_latent(self,
                     num_samples: int,
                     sampling_std: float = 1.0) -> torch.Tensor:
        """
        Sample latent codes from prior distribution.
        
        Args:
            num_samples: Number of samples
            sampling_std: Standard deviation for sampling
            
        Returns:
            Sampled latent codes
        """
        latent_dim = self.model.encoder.deterministic_proj.out_features
        z = torch.randn(num_samples, latent_dim, device=self.device) * sampling_std
        return z
    
    @torch.no_grad()
    def generate_batch(self,
                      z: torch.Tensor,
                      lengths: Optional[torch.Tensor] = None,
                      temperature: float = 1.0) -> List[str]:
        """
        Generate sequences from latent codes.
        
        Args:
            z: Latent codes
            lengths: Sequence lengths (required for PLM)
            temperature: Sampling temperature
            
        Returns:
            List of generated sequences
        """
        # Generate token sequences
        if hasattr(self.model, 'decoder_type'):
            decoder_type = self.model.decoder_type
        else:
            # Infer from decoder class
            decoder_type = "plm" if hasattr(self.model.decoder, 'mask_idx') else "causal"
        
        if decoder_type == "causal":
            max_length = self.model.decoder.max_len
            generated_tokens = self.model.decoder.generate(
                z, max_length, self.tokenizer.bos_idx,
                self.tokenizer.eos_idx, temperature
            )
        else:  # PLM
            if lengths is None:
                # Default to maximum length
                lengths = torch.full((z.size(0),), self.model.decoder.max_len,
                                   device=self.device, dtype=torch.long)
            generated_tokens = self.model.decoder.generate(z, lengths, temperature)
        
        # Decode to sequences
        sequences = []
        for tokens in generated_tokens:
            seq = self.tokenizer.decode(tokens.cpu().tolist(), remove_special_tokens=True)
            sequences.append(seq)
        
        return sequences
    
    def generate(self,
                num_samples: int,
                batch_size: int = 128,
                sampling_std: float = 1.0,
                temperature: float = 1.0,
                sequence_length: Optional[int] = None,
                show_progress: bool = True) -> List[str]:
        """
        Generate multiple sequences.
        
        Args:
            num_samples: Total number of sequences to generate
            batch_size: Batch size for generation
            sampling_std: Standard deviation for latent sampling
            temperature: Sampling temperature
            sequence_length: Fixed sequence length (for PLM)
            show_progress: Whether to show progress bar
            
        Returns:
            List of generated sequences
        """
        print(f"\n🔄 Generating {num_samples} sequences...")
        print(f"   Batch size: {batch_size}")
        print(f"   Sampling std: {sampling_std}")
        print(f"   Temperature: {temperature}")
        
        all_sequences = []
        
        # Create progress bar
        if show_progress:
            pbar = tqdm(total=num_samples, desc="Generating")
        
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            
            # Sample latent codes
            z = self.sample_latent(current_batch_size, sampling_std)
            
            # Set lengths if needed
            if sequence_length is not None:
                lengths = torch.full((current_batch_size,), sequence_length,
                                   device=self.device, dtype=torch.long)
            else:
                lengths = None
            
            # Generate sequences
            sequences = self.generate_batch(z, lengths, temperature)
            all_sequences.extend(sequences)
            
            if show_progress:
                pbar.update(current_batch_size)
        
        if show_progress:
            pbar.close()
        
        return all_sequences
    
    def save_fasta(self,
                  sequences: List[str],
                  output_path: str,
                  prefix: str = "sample"):
        """
        Save sequences to FASTA file.
        
        Args:
            sequences: List of sequences
            output_path: Output file path
            prefix: Prefix for sequence IDs
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📝 Writing {len(sequences)} sequences to {output_path}...")
        
        with open(output_path, 'w') as f:
            for i, seq in enumerate(sequences):
                f.write(f">{prefix}_{i+1}\n")
                f.write(f"{seq}\n")
        
        print("✅ Done!")
    
    def sample_and_save(self,
                       num_samples: int,
                       output_path: str,
                       batch_size: int = 128,
                       sampling_std: float = 1.0,
                       temperature: float = 1.0,
                       sequence_length: Optional[int] = None,
                       prefix: str = "sample"):
        """
        Generate sequences and save directly to FASTA file.
        
        Convenience method combining generation and saving.
        """
        sequences = self.generate(
            num_samples=num_samples,
            batch_size=batch_size,
            sampling_std=sampling_std,
            temperature=temperature,
            sequence_length=sequence_length
        )
        
        self.save_fasta(sequences, output_path, prefix)
        
        # Print some statistics
        lengths = [len(seq) for seq in sequences]
        print(f"\n📊 Generation Statistics:")
        print(f"   Total sequences: {len(sequences)}")
        print(f"   Length range: {min(lengths)}-{max(lengths)}")
        print(f"   Average length: {sum(lengths)/len(lengths):.1f}")
        
        # Count gaps
        gap_counts = [seq.count('-') for seq in sequences]
        print(f"   Average gaps: {sum(gap_counts)/len(gap_counts):.1f}")