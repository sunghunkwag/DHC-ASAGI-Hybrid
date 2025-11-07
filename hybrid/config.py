"""Configuration classes for the DHC-ASAGI Hybrid System."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DHCSSMConfig:
    """Configuration for DHC-SSM component."""
    input_channels: int = 3
    hidden_dim: int = 64
    ssm_state_dim: int = 64
    output_dim: int = 10
    use_attention: bool = True
    use_mixed_precision: bool = False


@dataclass
class ASAGIConfig:
    """Configuration for ASAGI component."""
    feature_dim: int = 256
    decision_dim: int = 128
    num_objectives: int = 4
    enable_meta_cognition: bool = True
    enable_causal_reasoning: bool = True
    causal_num_variables: int = 8
    causal_hidden_dim: int = 128
    causal_num_layers: int = 2


@dataclass
class IntegrationConfig:
    """Configuration for system integration."""
    default_mode: str = "feature_sharing"  # independent, feature_sharing, goal_driven, deep_integration
    enable_mode_switching: bool = True
    communication_dim: int = 256


@dataclass
class HybridConfig:
    """Main configuration for the DHC-ASAGI Hybrid System."""
    dhc_ssm: DHCSSMConfig = field(default_factory=DHCSSMConfig)
    asagi: ASAGIConfig = field(default_factory=ASAGIConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100
    
    # Device
    device: str = "cpu"  # "cpu" or "cuda"
    
    # Logging
    log_level: str = "INFO"
    save_frequency: int = 100
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.integration.default_mode in [
            "independent", "feature_sharing", "goal_driven", "deep_integration"
        ], f"Invalid integration mode: {self.integration.default_mode}"
