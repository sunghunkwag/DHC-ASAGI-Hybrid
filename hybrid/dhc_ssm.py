"""Simplified DHC-SSM implementation for hybrid system.
Based on DHC-SSM-Enhanced architecture.

Fixed temporal processing with GRU for proper sequence handling.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class SpatialEncoder(nn.Module):
    """CNN-based spatial feature extraction."""
    
    def __init__(self, input_channels: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.output_dim = hidden_dim * 4
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial features from input images.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
        
        Returns:
            Spatial features of shape (batch, hidden_dim * 4)
        """
        return self.conv_layers(x).squeeze(-1).squeeze(-1)


class StrategicReasoner(nn.Module):
    """Simplified causal reasoning module."""
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.reasoner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply strategic reasoning to temporal features.
        
        Args:
            x: Temporal features of shape (batch, hidden_dim)
        
        Returns:
            Strategic features of shape (batch, hidden_dim)
        """
        return self.reasoner(x)


class DHCSSMModel(nn.Module):
    """Simplified DHC-SSM model for hybrid integration.
    
    Architecture:
    1. Spatial Encoder (CNN)
    2. Temporal Processor (GRU)
    3. Strategic Reasoner
    4. Classification Head
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        hidden_dim = getattr(config, 'hidden_dim', 64)
        input_channels = getattr(config, 'input_channels', 3)
        output_dim = getattr(config, 'output_dim', 10)
        
        # Components
        self.spatial_encoder = SpatialEncoder(
            input_channels=input_channels,
            hidden_dim=hidden_dim
        )
        
        # GRU for temporal processing
        self.temporal_ssm = nn.GRU(
            input_size=hidden_dim * 4,
            hidden_size=hidden_dim * 4,
            num_layers=1,
            batch_first=True
        )
        
        self.strategic_reasoner = StrategicReasoner(
            hidden_dim=hidden_dim * 4
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Proper weight initialization for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """Forward pass through DHC-SSM.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            return_features: If True, return intermediate features
            
        Returns:
            logits or (logits, features_dict)
        """
        # Spatial encoding
        spatial_features = self.spatial_encoder(x)
        
        # Temporal processing with GRU
        spatial_features_seq = spatial_features.unsqueeze(1)
        temporal_output, _ = self.temporal_ssm(spatial_features_seq)
        temporal_features = temporal_output.squeeze(1)
        
        # Strategic reasoning
        strategic_features = self.strategic_reasoner(temporal_features)
        
        # Classification
        logits = self.classifier(strategic_features)
        
        if return_features:
            features = {
                'spatial': spatial_features,
                'temporal': temporal_features,
                'strategic': strategic_features,
                'logits': logits
            }
            return logits, features
        
        return logits
    
    def get_features(self, x: torch.Tensor, layer: str = 'all') -> Dict[str, torch.Tensor]:
        """Extract features from specific layer."""
        _, features = self.forward(x, return_features=True)
        
        if layer == 'all':
            return features
        elif layer in features:
            return {layer: features[layer]}
        else:
            raise ValueError(f"Unknown layer: {layer}")
    
    @property
    def num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
