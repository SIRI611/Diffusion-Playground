# Diffusion Playground

A comprehensive framework for implementing and experimenting with various diffusion models, from DDPM and DDIM to Consistency Models and Flow-based approaches.

## Features

- **DDPM (Denoising Diffusion Probabilistic Models)**: Complete implementation with linear and cosine noise schedules
- **DDIM (Denoising Diffusion Implicit Models)**: Deterministic sampling with configurable eta parameter
- **Consistency Models**: Fast sampling with consistency training
- **Flow-based Models**: Normalizing Flows and Continuous Normalizing Flows (CNFs)
- **Enhanced UNet**: Modern architecture with time embeddings and self-attention
- **Training Scripts**: Ready-to-use training scripts for all model types
- **Sampling Tools**: Comprehensive sampling and visualization utilities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SIRI611/Diffusion-Playground.git
cd Diffusion-Playground
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### Training a DDPM Model

```bash
# Train on MNIST
python scripts/train_ddpm.py --dataset mnist --num_epochs 100 --batch_size 128

# Train on CIFAR-10
python scripts/train_ddpm.py --dataset cifar10 --num_epochs 200 --batch_size 64
```

### Training a DDIM Model

```bash
# Train DDIM on MNIST
python scripts/train_ddim.py --dataset mnist --num_epochs 100 --batch_size 128
```

### Training a Consistency Model

```bash
# Train Consistency Model on MNIST
python scripts/train_consistency.py --dataset mnist --num_epochs 100 --batch_size 128
```

### Training Flow-based Models

```bash
# Train Normalizing Flow
python scripts/train_flow.py --dataset mnist --flow_type normalizing --num_epochs 100

# Train Continuous Normalizing Flow
python scripts/train_flow.py --dataset mnist --flow_type continuous --num_epochs 100
```

### Sampling from Trained Models

```bash
# Sample from DDPM
python scripts/sample.py --model_type ddpm --checkpoint checkpoints/ddpm_final.pth --num_samples 16

# Sample from DDIM
python scripts/sample.py --model_type ddim --checkpoint checkpoints/ddim_final.pth --num_samples 16 --num_steps 50

# Sample from Consistency Model
python scripts/sample.py --model_type consistency --checkpoint checkpoints/consistency_final.pth --num_samples 16

# Sample from Flow Model
python scripts/sample.py --model_type flow --checkpoint checkpoints/flow_final.pth --num_samples 16
```

## Model Implementations

### DDPM (Denoising Diffusion Probabilistic Models)

The DDPM implementation includes:
- Linear and cosine noise schedules
- Forward and reverse diffusion processes
- Both stochastic (DDPM) and deterministic (DDIM) sampling
- Configurable timesteps and beta schedules

```python
from diffusion.policy.ddpm import DDPM
from diffusion.models.unet import SimpleUNet

# Create model
model = SimpleUNet(in_channels=1, out_channels=1, hidden_channels=64)
ddpm = DDPM(model=model, timesteps=1000, schedule='linear')

# Training
loss = ddpm.compute_loss(data)

# Sampling
samples = ddpm.sample((batch_size, 1, 28, 28))
```

### DDIM (Denoising Diffusion Implicit Models)

DDIM provides deterministic sampling with faster generation:
- Deterministic sampling with configurable eta parameter
- Faster sampling with fewer steps
- Interpolation capabilities

```python
from diffusion.policy.ddim import DDIM

# Create DDIM
ddim = DDIM(model=model, timesteps=1000, schedule='linear')

# Fast sampling
samples = ddim.sample((batch_size, 1, 28, 28), num_steps=50, eta=0.0)
```

### Consistency Models

Consistency Models enable fast sampling:
- Single-step or few-step sampling
- Consistency training loss
- Distillation from diffusion models

```python
from diffusion.policy.consistency_model import ConsistencyModel

# Create Consistency Model
consistency_model = ConsistencyModel(model=model, sigma_min=0.002, sigma_max=80.0)

# Fast sampling
samples = consistency_model.sample((batch_size, 1, 28, 28), num_steps=2)
```

### Flow-based Models

Both Normalizing Flows and Continuous Normalizing Flows are implemented:
- Normalizing Flows with coupling layers
- Continuous Normalizing Flows with neural ODEs
- Exact likelihood computation

```python
from diffusion.policy.flow_models import NormalizingFlow, ContinuousNormalizingFlow, FlowBasedModel

# Normalizing Flow
flow_model = NormalizingFlow(input_dim=784, hidden_dim=64, num_layers=8)
model = FlowBasedModel(flow_model)

# Continuous Normalizing Flow
cnf_model = ContinuousNormalizingFlow(input_dim=784, hidden_dim=64)
model = FlowBasedModel(cnf_model)

# Sampling
samples = model.sample(num_samples=16)
```

## Architecture Details

### Enhanced UNet

The UNet implementation includes:
- Time embeddings with sinusoidal position encoding
- Self-attention layers at multiple resolutions
- Residual blocks with skip connections
- Group normalization and SiLU activations

```python
from diffusion.models.unet import UNet

# Full UNet with attention
model = UNet(
    in_channels=3,
    out_channels=3,
    model_channels=128,
    num_res_blocks=2,
    attention_resolutions=(16, 8),
    dropout=0.1,
    channel_mult=(1, 2, 4, 8),
    num_heads=8
)

# Simple UNet for basic experiments
model = SimpleUNet(in_channels=1, out_channels=1, hidden_channels=64)
```

## Examples

Check the `examples/` directory for complete examples:
- `ddpm_example.py`: Complete DDPM training and sampling example
- More examples coming soon for other model types

## Configuration

The framework supports configuration through YAML files in the `config/` directory:
- Model configurations
- Training parameters
- Dataset settings
- Sampling parameters

## Evaluation and Metrics

The framework includes utilities for:
- Sample quality evaluation
- FID (Fréchet Inception Distance) computation
- IS (Inception Score) calculation
- Interpolation visualization

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - Ho et al.
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) - Song et al.
- [Consistency Models](https://arxiv.org/abs/2303.01469) - Song et al.
- [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366) - Chen et al.
- [Normalizing Flows for Probabilistic Modeling and Inference](https://arxiv.org/abs/1912.02762) - Papamakarios et al.
