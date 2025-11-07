"""Hybrid Meta-Controller integrating DHC-SSM and ASAGI.

Supports four integration modes:
1. Independent: Systems operate separately
2. Feature Sharing: Share intermediate representations
3. Goal-Driven: ASAGI sets goals for DHC-SSM
4. Deep Integration: Full collaborative processing
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import logging

from .dhc_ssm import DHCSSMModel
from .asagi import ASAGISystem
from .config import HybridConfig

logger = logging.getLogger(__name__)


class HybridInterface(nn.Module):
    """Interface for communication between DHC-SSM and ASAGI."""
    
    def __init__(self, communication_dim: int = 256):
        super().__init__()
        self.communication_dim = communication_dim
        
        dhc_out_dim = 64 * 4  # Default DHC-SSM output
        asagi_feat_dim = 256  # Default ASAGI feature dim
        
        self.dhc_to_asagi = nn.Linear(dhc_out_dim, communication_dim)
        self.asagi_to_dhc = nn.Linear(asagi_feat_dim, dhc_out_dim)
        
    def exchange_features(self, source: str, features: torch.Tensor) -> torch.Tensor:
        """Exchange features between systems."""
        if source == "dhc":
            return self.dhc_to_asagi(features)
        elif source == "asagi":
            return self.asagi_to_dhc(features)
        else:
            raise ValueError(f"Unknown source: {source}")


class IntegrationModeSelector(nn.Module):
    """Selects optimal integration mode dynamically."""
    
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        
        self.mode_predictor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 4)  # 4 modes
        )
        
        self.mode_names = [
            "independent",
            "feature_sharing",
            "goal_driven",
            "deep_integration"
        ]
        
    def select(self, state_features: torch.Tensor, default_mode: str = "feature_sharing") -> str:
        """Select integration mode based on state."""
        try:
            state_avg = state_features.mean(dim=0, keepdim=True)
            mode_scores = self.mode_predictor(state_avg)
            mode_idx = mode_scores.argmax().item()
            return self.mode_names[mode_idx]
        except Exception as e:
            logger.warning(f"Mode selection failed, using default: {e}")
            return default_mode


class HybridMetaController(nn.Module):
    """Main controller orchestrating DHC-SSM and ASAGI interaction."""
    
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        
        # Initialize subsystems
        self.dhc_ssm = DHCSSMModel(config.dhc_ssm)
        self.asagi = ASAGISystem(config.asagi)
        
        # Interface
        self.interface = HybridInterface(config.integration.communication_dim)
        
        # Mode selector
        if config.integration.enable_mode_switching:
            self.mode_selector = IntegrationModeSelector(
                feature_dim=self.dhc_ssm.spatial_encoder.output_dim
            )
        else:
            self.mode_selector = None
        
        self.current_mode = config.integration.default_mode
        
        # Fusion layers
        dhc_out_dim = self.dhc_ssm.spatial_encoder.output_dim
        asagi_feat_dim = config.asagi.feature_dim
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(dhc_out_dim + asagi_feat_dim, dhc_out_dim),
            nn.LayerNorm(dhc_out_dim),
            nn.GELU()
        )
        
        logger.info(f"Initialized HybridMetaController with mode: {self.current_mode}")
        logger.info(f"DHC-SSM parameters: {self.dhc_ssm.num_parameters:,}")
        logger.info(f"ASAGI parameters: {self.asagi.num_parameters:,}")
    
    def forward(self, observations: torch.Tensor, mode: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through hybrid system."""
        # Select mode
        if mode is None:
            if self.mode_selector:
                with torch.no_grad():
                    spatial_features = self.dhc_ssm.spatial_encoder(observations)
                self.current_mode = self.mode_selector.select(
                    spatial_features, self.config.integration.default_mode
                )
            mode = self.current_mode
        
        # Execute based on mode
        if mode == "independent":
            return self._independent_forward(observations)
        elif mode == "feature_sharing":
            return self._feature_sharing_forward(observations)
        elif mode == "goal_driven":
            return self._goal_driven_forward(observations)
        elif mode == "deep_integration":
            return self._deep_integration_forward(observations)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _independent_forward(self, observations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Mode 1: Independent operation."""
        dhc_logits, dhc_features = self.dhc_ssm(observations, return_features=True)
        
        asagi_input_features = dhc_features['spatial'].detach()
        asagi_output = self.asagi(asagi_input_features)
        
        return {
            'logits': dhc_logits,
            'dhc_features': dhc_features,
            'asagi_output': asagi_output,
            'mode': 'independent'
        }
    
    def _feature_sharing_forward(self, observations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Mode 2: Feature sharing."""
        dhc_logits, dhc_features = self.dhc_ssm(observations, return_features=True)
        
        shared_features = self.interface.exchange_features('dhc', dhc_features['spatial'])
        asagi_output = self.asagi(shared_features)
        
        asagi_features_dhc = self.interface.exchange_features('asagi', asagi_output['processed_features'])
        enhanced_features = dhc_features['temporal'] + asagi_features_dhc
        
        return {
            'logits': dhc_logits,
            'dhc_features': dhc_features,
            'asagi_output': asagi_output,
            'enhanced_features': enhanced_features,
            'mode': 'feature_sharing'
        }
    
    def _goal_driven_forward(self, observations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Mode 3: Goal-driven collaboration."""
        dhc_logits, dhc_features = self.dhc_ssm(observations, return_features=True)
        
        shared_features = self.interface.exchange_features('dhc', dhc_features['spatial'])
        asagi_output = self.asagi(shared_features)
        
        goals = asagi_output['goals']
        
        return {
            'logits': dhc_logits,
            'dhc_features': dhc_features,
            'asagi_output': asagi_output,
            'goals': goals,
            'mode': 'goal_driven'
        }
    
    def _deep_integration_forward(self, observations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Mode 4: Deep integration with full collaboration."""
        spatial_features = self.dhc_ssm.spatial_encoder(observations)
        
        shared_features = self.interface.exchange_features('dhc', spatial_features)
        asagi_output = self.asagi(shared_features)
        
        fused_features = self.feature_fusion(
            torch.cat([spatial_features, asagi_output['processed_features']], dim=-1)
        )
        
        temporal_features = self.dhc_ssm.temporal_ssm(fused_features.unsqueeze(1))[0].squeeze(1)
        strategic_features = self.dhc_ssm.strategic_reasoner(temporal_features)
        logits = self.dhc_ssm.classifier(strategic_features)
        
        causal_output = {}
        if self.asagi.causal_reasoner is not None:
            causal_output = self.asagi.causal_reasoner(fused_features)
        
        return {
            'logits': logits,
            'fused_features': fused_features,
            'temporal_features': temporal_features,
            'strategic_features': strategic_features,
            'asagi_output': asagi_output,
            'causal_output': causal_output,
            'mode': 'deep_integration'
        }
    
    def set_mode(self, mode: str):
        """Set integration mode."""
        assert mode in ["independent", "feature_sharing", "goal_driven", "deep_integration"]
        self.current_mode = mode
        logger.info(f"Integration mode set to: {mode}")
    
    def get_state(self) -> Dict:
        """Get current system state."""
        return {
            'mode': self.current_mode,
            'dhc_ssm_params': self.dhc_ssm.num_parameters,
            'asagi_params': self.asagi.num_parameters,
            'total_params': self.num_parameters
        }
    
    @property
    def num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def clone(self):
        """Create a deep copy of the controller."""
        import copy
        return copy.deepcopy(self)
