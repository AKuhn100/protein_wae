"""
Configuration classes for Protein WAE models and training.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    fasta_path: str
    min_seq_len: int = 305
    max_seq_len: int = 305
    val_fraction: float = 0.1
    batch_size: int = 64
    num_workers: int = 4
    seed: int = 42


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    # Latent space
    latent_dim: int = 512
    
    # Encoder
    encoder_embed_dim: int = 256
    encoder_hidden_dim: int = 512
    encoder_num_layers: int = 6
    encoder_num_heads: int = 8
    
    # Decoder
    decoder_type: Literal["causal", "plm"] = "causal"
    decoder_hidden_dim: int = 512
    decoder_num_heads: int = 8
    decoder_num_layers: int = 4
    
    # Dropout
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    # General
    total_epochs: int = 65
    learning_rate: float = 1e-4
    gradient_clip: float = 1.0
    
    # Progressive training phases
    deterministic_epochs: int = 10
    mmd_ramp_epochs: int = 5
    mmd_full_epochs: int = 50
    
    # WAE specific
    mmd_weight: float = 10.0
    mmd_kernel_bandwidth: float = 1.0
    
    # Optimization
    use_amp: bool = True
    scheduler_patience: int = 15
    scheduler_factor: float = 0.5
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_best: bool = True
    save_last: bool = True
    log_interval: int = 100
    analysis_interval: int = 20


@dataclass
class SamplingConfig:
    """Configuration for sequence sampling."""
    num_samples: int = 10000
    batch_size: int = 128
    sampling_std: float = 1.0
    temperature: float = 1.0
    output_path: str = "generated_samples.fasta"


@dataclass
class Config:
    """Complete configuration combining all sub-configs."""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    sampling: Optional[SamplingConfig] = None
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create config from dictionary."""
        return cls(
            data=DataConfig(**config_dict.get("data", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            sampling=SamplingConfig(**config_dict.get("sampling", {})) if "sampling" in config_dict else None
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        config_dict = {
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__
        }
        if self.sampling:
            config_dict["sampling"] = self.sampling.__dict__
        return config_dict
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        import json
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "Config":
        """Load configuration from JSON file."""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Default configurations for different model types
def get_default_config(decoder_type: str = "causal") -> Config:
    """Get default configuration for a specific decoder type."""
    return Config(
        data=DataConfig(
            fasta_path="/path/to/your/data.fasta",  # User must set this
            min_seq_len=305,
            max_seq_len=305
        ),
        model=ModelConfig(
            decoder_type=decoder_type
        ),
        training=TrainingConfig(),
        sampling=SamplingConfig()
    )