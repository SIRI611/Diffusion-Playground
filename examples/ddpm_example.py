#!/usr/bin/env python3
"""
Example script demonstrating DDPM training and sampling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from diffusion.policy.ddpm import DDPM
from diffusion.models.unet import SimpleUNet

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    print(f'Dataset size: {len(train_dataset)}')
    print(f'Number of batches: {len(train_loader)}')
    
    # Create UNet model
    model = SimpleUNet(in_channels=1, out_channels=1, hidden_channels=64)
    
    # Create DDPM
    ddpm = DDPM(
        model=model,
        timesteps=1000,
        schedule='linear',
        device=device
    )
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Timesteps: {ddpm.timesteps}')
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # Training loop
    num_epochs = 50
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(device)
            
            # Compute loss
            loss = ddpm.compute_loss(data)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Update learning rate
        scheduler.step()
        
        # Average loss
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
    
    # Generate samples
    model.eval()
    with torch.no_grad():
        # DDPM sampling
        samples_ddpm = ddpm.sample((16, 1, 28, 28), num_steps=1000)
        
        # DDIM sampling (faster)
        samples_ddim = ddpm.ddim_sample((16, 1, 28, 28), num_steps=50, eta=0.0)
    
    print(f'Generated {samples_ddpm.shape[0]} samples')
    
    # Visualize results
    def plot_samples(samples, title, nrow=4):
        # Denormalize samples
        samples = (samples + 1) / 2  # Convert from [-1, 1] to [0, 1]
        samples = torch.clamp(samples, 0, 1)
        
        # Create grid
        grid = torchvision.utils.make_grid(samples, nrow=nrow, padding=2)
        
        # Plot
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.cpu().numpy().transpose(1, 2, 0))
        plt.title(title, fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # Plot DDPM samples
    plot_samples(samples_ddpm, 'DDPM Generated Samples')
    
    # Plot DDIM samples
    plot_samples(samples_ddim, 'DDIM Generated Samples')
    
    # Save trained model
    torch.save(model.state_dict(), 'ddpm_mnist.pth')
    print('Model saved to ddpm_mnist.pth')

if __name__ == "__main__":
    main()
