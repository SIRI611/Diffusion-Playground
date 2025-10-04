#!/usr/bin/env python3
"""
Comprehensive demonstration of all diffusion models in the playground.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from diffusion.policy.ddpm import DDPM
from diffusion.policy.ddim import DDIM
from diffusion.policy.consistency_model import ConsistencyModel
from diffusion.policy.flow_models import NormalizingFlow, ContinuousNormalizingFlow, FlowBasedModel
from diffusion.models.unet import UNet, SimpleUNet

def create_models(device="cuda"):
    """Create all model instances."""
    
    # Create UNet models
    unet_simple = SimpleUNet(in_channels=1, out_channels=1, hidden_channels=64)
    unet_full = UNet(in_channels=1, out_channels=1, model_channels=128)
    
    # Create flow models
    flow_model = NormalizingFlow(input_dim=784, hidden_dim=64, num_layers=8, device=device)
    cnf_model = ContinuousNormalizingFlow(input_dim=784, hidden_dim=64, device=device)
    
    # Create diffusion models
    ddpm = DDPM(model=unet_simple, timesteps=1000, device=device)
    ddim = DDIM(model=unet_simple, timesteps=1000, device=device)
    consistency = ConsistencyModel(model=unet_simple, device=device)
    
    # Create flow-based model wrappers
    flow_based = FlowBasedModel(flow_model, device=device)
    cnf_based = FlowBasedModel(cnf_model, device=device)
    
    return {
        'ddpm': ddpm,
        'ddim': ddim,
        'consistency': consistency,
        'flow': flow_based,
        'cnf': cnf_based
    }

def load_mnist_data(batch_size=128):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def demonstrate_sampling(models, device="cuda"):
    """Demonstrate sampling from all models."""
    
    print("Generating samples from all models...")
    
    samples = {}
    
    # DDPM sampling
    print("  - DDPM sampling...")
    with torch.no_grad():
        samples['ddpm'] = models['ddpm'].sample((4, 1, 28, 28), num_steps=1000)
    
    # DDIM sampling
    print("  - DDIM sampling...")
    with torch.no_grad():
        samples['ddim'] = models['ddim'].sample((4, 1, 28, 28), num_steps=50, eta=0.0)
    
    # Consistency Model sampling
    print("  - Consistency Model sampling...")
    with torch.no_grad():
        samples['consistency'] = models['consistency'].sample((4, 1, 28, 28), num_steps=2)
    
    # Flow-based sampling
    print("  - Normalizing Flow sampling...")
    with torch.no_grad():
        flow_samples = models['flow'].sample(4)
        samples['flow'] = flow_samples.view(4, 1, 28, 28)
    
    # CNF sampling
    print("  - Continuous Normalizing Flow sampling...")
    with torch.no_grad():
        cnf_samples = models['cnf'].sample(4)
        samples['cnf'] = cnf_samples.view(4, 1, 28, 28)
    
    return samples

def visualize_samples(samples, save_path="all_models_comparison.png"):
    """Visualize samples from all models."""
    
    # Create subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    model_names = ['ddpm', 'ddim', 'consistency', 'flow', 'cnf']
    
    for i, model_name in enumerate(model_names):
        if model_name in samples:
            # Denormalize samples
            sample = samples[model_name]
            sample = (sample + 1) / 2  # Convert from [-1, 1] to [0, 1]
            sample = torch.clamp(sample, 0, 1)
            
            # Create grid
            grid = torchvision.utils.make_grid(sample, nrow=2, padding=2)
            
            # Plot
            axes[i].imshow(grid.cpu().numpy().transpose(1, 2, 0))
            axes[i].set_title(f'{model_name.upper()}', fontsize=14)
            axes[i].axis('off')
    
    # Remove empty subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Comparison saved to {save_path}")

def demonstrate_interpolation(models, device="cuda"):
    """Demonstrate interpolation capabilities."""
    
    print("Generating interpolations...")
    
    # Generate two random samples for interpolation
    with torch.no_grad():
        # For diffusion models
        x1_diff = models['ddpm'].sample((1, 1, 28, 28), num_steps=1000)
        x2_diff = models['ddpm'].sample((1, 1, 28, 28), num_steps=1000)
        
        # For flow models
        x1_flow = models['flow'].sample(1).view(1, 1, 28, 28)
        x2_flow = models['flow'].sample(1).view(1, 1, 28, 28)
    
    # Create interpolations
    interpolations = []
    for i in range(11):
        alpha = i / 10
        
        # Linear interpolation
        interp_diff = (1 - alpha) * x1_diff + alpha * x2_diff
        interp_flow = (1 - alpha) * x1_flow + alpha * x2_flow
        
        interpolations.append(torch.cat([interp_diff, interp_flow], dim=0))
    
    # Visualize interpolation
    interp_tensor = torch.cat(interpolations, dim=0)
    
    # Denormalize
    interp_tensor = (interp_tensor + 1) / 2
    interp_tensor = torch.clamp(interp_tensor, 0, 1)
    
    # Create grid
    grid = torchvision.utils.make_grid(interp_tensor, nrow=2, padding=2)
    
    # Plot
    plt.figure(figsize=(20, 10))
    plt.imshow(grid.cpu().numpy().transpose(1, 2, 0))
    plt.title('Interpolation: DDPM (top) vs Normalizing Flow (bottom)', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('interpolation_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Interpolation saved to interpolation_comparison.png")

def main():
    """Main demonstration function."""
    
    print("=== Diffusion Playground Demonstration ===\n")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create models
    print("Creating models...")
    models = create_models(device)
    print(f"Created {len(models)} model types\n")
    
    # Demonstrate sampling
    samples = demonstrate_sampling(models, device)
    print()
    
    # Visualize samples
    visualize_samples(samples)
    print()
    
    # Demonstrate interpolation
    demonstrate_interpolation(models, device)
    print()
    
    print("=== Demonstration Complete ===")
    print("\nModel Summary:")
    print("- DDPM: Denoising Diffusion Probabilistic Models")
    print("- DDIM: Denoising Diffusion Implicit Models")
    print("- Consistency: Consistency Models")
    print("- Flow: Normalizing Flows")
    print("- CNF: Continuous Normalizing Flows")
    
    print("\nKey Features:")
    print("- Multiple sampling strategies")
    print("- Interpolation capabilities")
    print("- Configurable noise schedules")
    print("- Modern UNet architectures")
    print("- Flow-based generative models")

if __name__ == "__main__":
    main()
