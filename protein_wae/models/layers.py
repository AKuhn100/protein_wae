"""
Common layers and modules used across models.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 512):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create div_term for sin/cos frequencies
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1)]


class ConvBlock(nn.Module):
    """Convolutional block for feature extraction."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dropout: float = 0.1):
        """
        Initialize convolutional block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            dropout: Dropout probability
        """
        super().__init__()
        
        padding = (kernel_size - 1) // 2
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            padding=padding
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through conv block."""
        x = self.conv(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class MultiLayerConv(nn.Module):
    """Multi-layer convolutional feature extractor."""
    
    def __init__(self, d_emb: int, d_model: int, dropout: float = 0.1):
        """
        Initialize multi-layer convolution.
        
        Args:
            d_emb: Embedding dimension
            d_model: Model dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.layers = nn.Sequential(
            ConvBlock(d_emb, d_model // 2, kernel_size=7, dropout=dropout),
            ConvBlock(d_model // 2, d_model, kernel_size=5, dropout=dropout),
            ConvBlock(d_model, d_model, kernel_size=3, dropout=0.0)  # No dropout on last layer
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through convolutional layers.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_emb)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Permute for Conv1d: (B, L, D) -> (B, D, L)
        x = x.permute(0, 2, 1)
        x = self.layers(x)
        # Permute back: (B, D, L) -> (B, L, D)
        x = x.permute(0, 2, 1)
        return x


class LatentProjection(nn.Module):
    """Project between latent space and model dimensions."""
    
    def __init__(self, d_lat: int, d_model: int, num_layers: int = 2, 
                 dropout: float = 0.1):
        """
        Initialize latent projection.
        
        Args:
            d_lat: Latent dimension
            d_model: Model dimension
            num_layers: Number of projection layers
            dropout: Dropout probability
        """
        super().__init__()
        
        if num_layers == 1:
            self.projection = nn.Linear(d_lat, d_model)
        else:
            layers = []
            hidden_dim = d_model * 2
            
            # First layer
            layers.extend([
                nn.Linear(d_lat, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            
            # Additional layers
            for _ in range(num_layers - 2):
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
            
            # Final layer
            layers.append(nn.Linear(hidden_dim, d_model))
            
            self.projection = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Project latent code to model dimension."""
        return self.projection(z)