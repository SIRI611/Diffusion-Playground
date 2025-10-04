#!/usr/bin/env python3
"""
Training script for DDIM (Denoising Diffusion Implicit Models).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from diffusion.policy.ddim import DDIM
from diffusion.models.unet import UNet, SimpleUNet

def get_data_loaders(dataset_name="mnist", batch_size=128, num_workers=4):
    """Get data loaders for training."""
    
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])
        
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        
    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, test_loader

def train_ddim(model, train_loader, test_loader, num_epochs=100, lr=1e-4, 
               device="cuda", save_dir="./checkpoints", use_wandb=False):
    """Train DDIM model."""
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize wandb if requested
    if use_wandb:
        wandb.init(project="ddim-training", config={
            "num_epochs": num_epochs,
            "learning_rate": lr,
            "device": device
        })
    
    # Optimizer
    optimizer = optim.Adam(model.model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    model.model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(device)
            
            # Compute loss
            loss = model.compute_loss(data)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Log to wandb
            if use_wandb:
                wandb.log({"train_loss": loss.item()})
        
        # Update learning rate
        scheduler.step()
        
        # Average loss
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, os.path.join(save_dir, f'ddim_epoch_{epoch+1}.pth'))
        
        # Generate samples
        if (epoch + 1) % 20 == 0:
            model.model.eval()
            with torch.no_grad():
                # Generate samples using DDIM sampling
                samples = model.sample((16, data.shape[1], data.shape[2], data.shape[3]), 
                                     num_steps=50, eta=0.0)
                
                # Save samples
                save_samples(samples, os.path.join(save_dir, f'samples_epoch_{epoch+1}.png'))
                
                # Log to wandb
                if use_wandb:
                    wandb.log({"samples": wandb.Image(samples[0])})
            
            model.model.train()
    
    # Save final model
    torch.save(model.model.state_dict(), os.path.join(save_dir, 'ddim_final.pth'))
    
    if use_wandb:
        wandb.finish()

def save_samples(samples, filename):
    """Save generated samples as images."""
    # Denormalize samples
    samples = (samples + 1) / 2  # Convert from [-1, 1] to [0, 1]
    samples = torch.clamp(samples, 0, 1)
    
    # Create grid
    grid = torchvision.utils.make_grid(samples, nrow=4, padding=2)
    
    # Convert to numpy
    grid_np = grid.cpu().numpy().transpose(1, 2, 0)
    
    # Save
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train DDIM')
    parser.add_argument('--dataset', type=str, default='mnist', 
                       choices=['mnist', 'cifar10'], help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Save directory')
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb logging')
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of timesteps')
    parser.add_argument('--schedule', type=str, default='linear', 
                       choices=['linear', 'cosine'], help='Noise schedule')
    parser.add_argument('--model_channels', type=int, default=128, help='Model channels')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(args.dataset, args.batch_size)
    
    # Get data shape
    sample_batch = next(iter(train_loader))
    data_shape = sample_batch[0].shape[1:]  # Remove batch dimension
    print(f"Data shape: {data_shape}")
    
    # Create model
    if args.dataset == 'mnist':
        model = SimpleUNet(in_channels=1, out_channels=1, hidden_channels=args.model_channels)
    else:  # cifar10
        model = UNet(in_channels=3, out_channels=3, model_channels=args.model_channels)
    
    # Create DDIM
    ddim = DDIM(
        model=model,
        timesteps=args.timesteps,
        schedule=args.schedule,
        device=device
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    train_ddim(
        ddim, train_loader, test_loader,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=device,
        save_dir=args.save_dir,
        use_wandb=args.use_wandb
    )

if __name__ == "__main__":
    main()
