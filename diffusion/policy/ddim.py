import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from typing import Optional, Tuple, Union, List

class DDIM:
    """
    Denoising Diffusion Implicit Models (DDIM) implementation.
    
    Based on: "Denoising Diffusion Implicit Models" by Song et al.
    """
    
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02, 
                 device="cpu", schedule="linear"):
        self.model = model.to(device)
        self.timesteps = timesteps
        self.device = device
        
        # Beta schedule (same as DDPM)
        if schedule == "linear":
            self.beta = torch.linspace(beta_start, beta_end, timesteps).to(device)
        elif schedule == "cosine":
            self.beta = self._cosine_beta_schedule(timesteps, beta_start, beta_end).to(device)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
            
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        
        # Precompute coefficients
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine noise schedule as proposed in Improved DDPM."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def forward_diffusion(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply forward diffusion process (same as DDPM)."""
        noise = torch.randn_like(x0)
        
        # Reshape for broadcasting
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        
        xt = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
        return xt, noise

    def reverse_diffusion(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise using the model."""
        return self.model(x, t)

    def compute_loss(self, x0: torch.Tensor) -> torch.Tensor:
        """Compute DDIM loss (same as DDPM)."""
        batch_size = x0.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
        
        xt, noise = self.forward_diffusion(x0, t)
        predicted_noise = self.reverse_diffusion(xt, t)
        
        return F.mse_loss(noise, predicted_noise)

    def sample(self, shape: Tuple[int, ...], num_steps: int = 50, eta: float = 0.0) -> torch.Tensor:
        """
        Sample using DDIM deterministic sampling.
        
        Args:
            shape: Shape of the samples to generate
            num_steps: Number of denoising steps (fewer than timesteps for speed)
            eta: Controls stochasticity (0 = deterministic, 1 = stochastic)
        """
        # Create DDIM timestep schedule
        step_size = self.timesteps // num_steps
        timesteps = torch.arange(0, self.timesteps, step_size, device=self.device)
        
        # Start from pure noise
        x = torch.randn(shape, device=self.device)
        
        for i in reversed(range(len(timesteps))):
            t = torch.full((shape[0],), timesteps[i], device=self.device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.reverse_diffusion(x, t)
            
            # DDIM update
            alpha_cumprod_t = self.alpha_cumprod[t].view(-1, 1, 1, 1)
            
            if i > 0:
                alpha_cumprod_prev = self.alpha_cumprod[timesteps[i-1]].view(-1, 1, 1, 1)
                
                # Predict x0
                pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
                
                # DDIM direction
                dir_xt = torch.sqrt(1 - alpha_cumprod_prev - eta**2 * (1 - alpha_cumprod_prev)) * predicted_noise
                
                # Noise term
                noise = torch.randn_like(x) if eta > 0 else 0
                noise_term = eta * torch.sqrt(1 - alpha_cumprod_prev) * noise
                
                # Update x
                x = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt + noise_term
            else:
                # Final step
                pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
                x = pred_x0
                
        return x

    def encode(self, x0: torch.Tensor, num_steps: int = 50) -> torch.Tensor:
        """
        Encode real data to latent space using DDIM.
        
        Args:
            x0: Real data samples
            num_steps: Number of encoding steps
        """
        # Create DDIM timestep schedule
        step_size = self.timesteps // num_steps
        timesteps = torch.arange(0, self.timesteps, step_size, device=self.device)
        
        x = x0.clone()
        
        for i in range(len(timesteps)):
            t = torch.full((x.shape[0],), timesteps[i], device=self.device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.reverse_diffusion(x, t)
            
            # DDIM encoding step
            alpha_cumprod_t = self.alpha_cumprod[t].view(-1, 1, 1, 1)
            
            if i < len(timesteps) - 1:
                alpha_cumprod_next = self.alpha_cumprod[timesteps[i+1]].view(-1, 1, 1, 1)
                
                # Predict x0
                pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
                
                # DDIM encoding direction
                dir_xt = torch.sqrt(1 - alpha_cumprod_next) * predicted_noise
                
                # Update x
                x = torch.sqrt(alpha_cumprod_next) * pred_x0 + dir_xt
            else:
                # Final encoding step
                pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
                x = pred_x0
                
        return x

    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, num_steps: int = 50, 
                   interpolation_steps: int = 10) -> List[torch.Tensor]:
        """
        Interpolate between two samples in latent space.
        
        Args:
            x1, x2: Two samples to interpolate between
            num_steps: Number of DDIM steps for encoding/decoding
            interpolation_steps: Number of interpolation points
        """
        # Encode both samples
        z1 = self.encode(x1, num_steps)
        z2 = self.encode(x2, num_steps)
        
        # Interpolate in latent space
        interpolations = []
        for i in range(interpolation_steps + 1):
            alpha = i / interpolation_steps
            z_interp = (1 - alpha) * z1 + alpha * z2
            
            # Decode back to data space
            x_interp = self.sample(z_interp.shape, num_steps)
            interpolations.append(x_interp)
            
        return interpolations
