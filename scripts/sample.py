#!/usr/bin/env python3
"""
Sampling script for all diffusion models.
"""

import torch
import torch.nn as nn
import torchvision
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from diffusion.policy.ddpm import DDPM
from diffusion.policy.ddim import DDIM
from diffusion.policy.consistency_model import ConsistencyModel
from diffusion.policy.flow_models import NormalizingFlow, ContinuousNormalizingFlow, FlowBasedModel
from diffusion.models.unet import UNet, SimpleUNet

def load_model(model_type, checkpoint_path, dataset="mnist", device="cuda"):
    """Load a trained model from checkpoint."""
    
    if dataset == "mnist":
        if model_type in ["ddpm", "ddim"]:
            model = SimpleUNet(in_channels=1, out_channels=1, hidden_channels=128)
        else:
            model = SimpleUNet(in_channels=1, out_channels=1, hidden_channels=128)
        data_shape = (1, 28, 28)
    else:  # cifar10
        if model_type in ["ddpm", "ddim"]:
            model = UNet(in_channels=3, out_channels=3, model_channels=128)
        else:
            model = UNet(in_channels=3, out_channels=3, model_channels=128)
        data_shape = (3, 32, 32)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, data_shape

def sample_ddpm(model, data_shape, num_samples=16, num_steps=1000, device="cuda"):
    """Sample from DDPM model."""
    ddpm = DDPM(model=model, timesteps=1000, device=device)
    
    with torch.no_grad():
        samples = ddpm.sample((num_samples, *data_shape), num_steps=num_steps)
    
    return samples

def sample_ddim(model, data_shape, num_samples=16, num_steps=50, eta=0.0, device="cuda"):
    """Sample from DDIM model."""
    ddim = DDIM(model=model, timesteps=1000, device=device)
    
    with torch.no_grad():
        samples = ddim.sample((num_samples, *data_shape), num_steps=num_steps, eta=eta)
    
    return samples

def sample_consistency(model, data_shape, num_samples=16, num_steps=2, device="cuda"):
    """Sample from Consistency Model."""
    consistency_model = ConsistencyModel(model=model, device=device)
    
    with torch.no_grad():
        samples = consistency_model.sample((num_samples, *data_shape), num_steps=num_steps)
    
    return samples

def sample_flow(model, data_shape, num_samples=16, device="cuda"):
    """Sample from Flow-based model."""
    flow_model = FlowBasedModel(model, device=device)
    
    with torch.no_grad():
        samples = flow_model.sample(num_samples)
        
        # Reshape samples back to image format
        if data_shape[0] == 1:  # MNIST
            samples = samples.view(num_samples, 1, 28, 28)
        elif data_shape[0] == 3:  # CIFAR-10
            samples = samples.view(num_samples, 3, 32, 32)
    
    return samples

def save_samples(samples, filename, title="Generated Samples"):
    """Save generated samples as images."""
    # Denormalize samples
    samples = (samples + 1) / 2  # Convert from [-1, 1] to [0, 1]
    samples = torch.clamp(samples, 0, 1)
    
    # Create grid
    grid = torchvision.utils.make_grid(samples, nrow=4, padding=2)
    
    # Convert to numpy
    grid_np = grid.cpu().numpy().transpose(1, 2, 0)
    
    # Save
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_np)
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def interpolate_samples(model, model_type, data_shape, device="cuda"):
    """Generate interpolation between two samples."""
    
    if model_type in ["ddpm", "ddim"]:
        # Generate two samples
        if model_type == "ddpm":
            ddpm = DDPM(model=model, timesteps=1000, device=device)
            with torch.no_grad():
                x1 = ddpm.sample((1, *data_shape), num_steps=1000)
                x2 = ddpm.sample((1, *data_shape), num_steps=1000)
        else:  # ddim
            ddim = DDIM(model=model, timesteps=1000, device=device)
            with torch.no_grad():
                x1 = ddim.sample((1, *data_shape), num_steps=50)
                x2 = ddim.sample((1, *data_shape), num_steps=50)
        
        # Simple linear interpolation
        interpolations = []
        for i in range(11):
            alpha = i / 10
            x_interp = (1 - alpha) * x1 + alpha * x2
            interpolations.append(x_interp)
        
        return torch.cat(interpolations, dim=0)
    
    elif model_type == "consistency":
        consistency_model = ConsistencyModel(model=model, device=device)
        with torch.no_grad():
            x1 = consistency_model.sample((1, *data_shape), num_steps=2)
            x2 = consistency_model.sample((1, *data_shape), num_steps=2)
        
        # Simple linear interpolation
        interpolations = []
        for i in range(11):
            alpha = i / 10
            x_interp = (1 - alpha) * x1 + alpha * x2
            interpolations.append(x_interp)
        
        return torch.cat(interpolations, dim=0)
    
    elif model_type == "flow":
        flow_model = FlowBasedModel(model, device=device)
        with torch.no_grad():
            x1 = flow_model.sample(1)
            x2 = flow_model.sample(1)
            
            # Reshape
            if data_shape[0] == 1:  # MNIST
                x1 = x1.view(1, 1, 28, 28)
                x2 = x2.view(1, 1, 28, 28)
            elif data_shape[0] == 3:  # CIFAR-10
                x1 = x1.view(1, 3, 32, 32)
                x2 = x2.view(1, 3, 32, 32)
        
        # Simple linear interpolation
        interpolations = []
        for i in range(11):
            alpha = i / 10
            x_interp = (1 - alpha) * x1 + alpha * x2
            interpolations.append(x_interp)
        
        return torch.cat(interpolations, dim=0)

def main():
    parser = argparse.ArgumentParser(description='Sample from diffusion models')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['ddpm', 'ddim', 'consistency', 'flow'], 
                       help='Type of model to sample from')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='mnist', 
                       choices=['mnist', 'cifar10'], help='Dataset type')
    parser.add_argument('--num_samples', type=int, default=16, 
                       help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='./samples', 
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--interpolate', action='store_true', 
                       help='Generate interpolation samples')
    
    # Model-specific arguments
    parser.add_argument('--num_steps', type=int, default=1000, 
                       help='Number of sampling steps')
    parser.add_argument('--eta', type=float, default=0.0, 
                       help='DDIM eta parameter')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading {args.model_type} model from {args.checkpoint}")
    model, data_shape = load_model(args.model_type, args.checkpoint, args.dataset, device)
    print(f"Data shape: {data_shape}")
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    
    if args.model_type == "ddpm":
        samples = sample_ddpm(model, data_shape, args.num_samples, args.num_steps, device)
    elif args.model_type == "ddim":
        samples = sample_ddim(model, data_shape, args.num_samples, args.num_steps, args.eta, device)
    elif args.model_type == "consistency":
        samples = sample_consistency(model, data_shape, args.num_samples, 2, device)
    elif args.model_type == "flow":
        samples = sample_flow(model, data_shape, args.num_samples, device)
    
    # Save samples
    output_path = os.path.join(args.output_dir, f'{args.model_type}_samples.png')
    save_samples(samples, output_path, f'{args.model_type.upper()} Generated Samples')
    print(f"Samples saved to {output_path}")
    
    # Generate interpolation if requested
    if args.interpolate:
        print("Generating interpolation samples...")
        interp_samples = interpolate_samples(model, args.model_type, data_shape, device)
        interp_path = os.path.join(args.output_dir, f'{args.model_type}_interpolation.png')
        save_samples(interp_samples, interp_path, f'{args.model_type.upper()} Interpolation')
        print(f"Interpolation saved to {interp_path}")

if __name__ == "__main__":
    main()