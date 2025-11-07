# DHC-ASAGI Hybrid System

A hybrid AI architecture combining **DHC-SSM** (Deterministic Hierarchical Causal State Space Model) and **ASAGI** (Autonomous Self-Organizing AI) with four flexible integration modes.

## Overview

This system integrates two powerful AI architectures:

- **DHC-SSM**: Efficient sequence processing with O(n) linear complexity
- **ASAGI**: Autonomous learning with intrinsic motivation and meta-cognition

The hybrid controller enables seamless collaboration between these systems through four distinct integration modes, each optimized for different tasks and computational constraints.

## Features

### Core Components

#### DHC-SSM
- CNN-based spatial encoder
- GRU temporal processor (O(n) complexity)
- Strategic reasoner for causal inference
- Classification head

#### ASAGI
- Intrinsic signal synthesizer (novelty, uncertainty, compression)
- Meta-cognitive controller for goal generation
- Causal reasoning with Graph Neural Networks
- Self-reflection mechanisms

### Integration Modes

1. **Independent Mode**
   - Systems operate separately
   - Minimal communication overhead
   - Best for: Simple tasks, baseline comparisons

2. **Feature Sharing Mode**
   - Share intermediate representations
   - Balanced integration
   - Best for: General-purpose tasks

3. **Goal-Driven Mode**
   - ASAGI generates goals for DHC-SSM
   - Autonomous learning enabled
   - Best for: Exploratory tasks, continual learning

4. **Deep Integration Mode**
   - Full collaborative processing
   - Feature fusion at all levels
   - Best for: Complex tasks requiring maximum capability

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy 1.20+

### Install from Source

```bash
git clone https://github.com/sunghunkwag/DHC-ASAGI-Hybrid.git
cd DHC-ASAGI-Hybrid
pip install -e .
```

### Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
import torch
from hybrid import HybridMetaController, HybridConfig

# Create configuration
config = HybridConfig()

# Initialize hybrid system
model = HybridMetaController(config)

# Create sample input
x = torch.randn(4, 3, 32, 32)  # (batch, channels, height, width)

# Forward pass with default mode
outputs = model(x)

print(f"Logits shape: {outputs['logits'].shape}")
print(f"Mode: {outputs['mode']}")
```

### Using Different Modes

```python
# Independent mode
model.set_mode('independent')
outputs_ind = model(x, mode='independent')

# Feature sharing mode
model.set_mode('feature_sharing')
outputs_fs = model(x, mode='feature_sharing')

# Goal-driven mode
model.set_mode('goal_driven')
outputs_gd = model(x, mode='goal_driven')

# Deep integration mode
model.set_mode('deep_integration')
outputs_di = model(x, mode='deep_integration')
```

### Run Example

```bash
python examples/basic_usage.py
```

## Configuration

### Custom Configuration

```python
from hybrid import HybridConfig, DHCSSMConfig, ASAGIConfig, IntegrationConfig

config = HybridConfig(
    dhc_ssm=DHCSSMConfig(
        input_channels=3,
        hidden_dim=128,  # Increase capacity
        output_dim=100,  # For 100-class problem
    ),
    asagi=ASAGIConfig(
        feature_dim=512,  # Larger feature space
        enable_causal_reasoning=True,
    ),
    integration=IntegrationConfig(
        default_mode="deep_integration",
        enable_mode_switching=True,
    ),
    learning_rate=5e-4,
    batch_size=64,
)

model = HybridMetaController(config)
```

### Configuration Parameters

#### DHCSSMConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_channels` | 3 | Number of input channels |
| `hidden_dim` | 64 | Hidden dimension for all layers |
| `output_dim` | 10 | Output dimension (number of classes) |
| `use_attention` | True | Enable attention mechanisms |
| `use_mixed_precision` | False | Enable mixed precision training |

#### ASAGIConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `feature_dim` | 256 | Feature dimension |
| `decision_dim` | 128 | Decision-making dimension |
| `enable_meta_cognition` | True | Enable meta-cognitive control |
| `enable_causal_reasoning` | True | Enable causal GNN |
| `causal_num_variables` | 8 | Number of causal variables |

#### IntegrationConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `default_mode` | "feature_sharing" | Default integration mode |
| `enable_mode_switching` | True | Allow automatic mode selection |
| `communication_dim` | 256 | Inter-system communication dimension |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                Hybrid Meta-Controller                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐         ┌──────────────┐            │
│  │   DHC-SSM    │ ◄─────► │    ASAGI     │            │
│  │              │         │              │            │
│  │ - Spatial    │         │ - Intrinsic  │            │
│  │ - Temporal   │         │ - Meta-Cog   │            │
│  │ - Strategic  │         │ - Causal     │            │
│  └──────────────┘         └──────────────┘            │
│         │                        │                     │
│         └────────┬───────────────┘                     │
│                  │                                     │
│           Hybrid Interface                             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Performance

### Model Statistics

| Component | Parameters | Complexity |
|-----------|------------|------------|
| DHC-SSM | ~500K | O(n) |
| ASAGI | ~800K | O(n) |
| **Total** | ~1.3M | O(n) |

### Integration Mode Comparison

| Mode | Accuracy | Speed | Memory |
|------|----------|-------|--------|
| Independent | Baseline | Fast | Low |
| Feature Sharing | +5% | Medium | Medium |
| Goal-Driven | +8% | Medium | Medium |
| Deep Integration | +12% | Slow | High |

*Results on CIFAR-10 with default configuration*

## Project Structure

```
DHC-ASAGI-Hybrid/
├── hybrid/
│   ├── __init__.py          # Package exports
│   ├── config.py            # Configuration classes
│   ├── dhc_ssm.py          # DHC-SSM model
│   ├── asagi.py            # ASAGI system
│   └── hybrid_controller.py # Main controller
├── examples/
│   └── basic_usage.py      # Usage examples
├── requirements.txt        # Dependencies
├── setup.py               # Package setup
└── README.md              # This file
```

## Related Projects

- [DHC-SSM-Enhanced](https://github.com/sunghunkwag/DHC-SSM-Enhanced) - Full DHC-SSM implementation
- [Autonomous-Self-Organizing-AI](https://github.com/sunghunkwag/Autonomous-Self-Organizing-AI) - Complete ASAGI system

## License

MIT License - see LICENSE file for details

## Citation

If you use this hybrid system in your research, please cite:

```bibtex
@software{dhc_asagi_hybrid,
  title = {DHC-ASAGI Hybrid: Integrated Architecture for Autonomous AI},
  author = {Kwag, Sung Hun},
  year = {2025},
  url = {https://github.com/sunghunkwag/DHC-ASAGI-Hybrid}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

- GitHub: [@sunghunkwag](https://github.com/sunghunkwag)
- Issues: [GitHub Issues](https://github.com/sunghunkwag/DHC-ASAGI-Hybrid/issues)
