import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union, List

class ConsistencyModel(nn.Module):
    """
    Consistency Model implementation.
    
    Based on: "Consistency Models" by Song et al.
    """
    
    def __init__(self, model, sigma_min=0.002, sigma_max=80.0, device="cpu"):
        super(ConsistencyModel, self).__init__()
        self.model = model.to(device)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.device = device
        
        # Precompute noise schedule
        self._setup_noise_schedule()

    def _setup_noise_schedule(self):
        """Setup noise schedule for consistency training."""
        # Use log-normal distribution for noise levels
        self.log_sigmas = torch.linspace(
            np.log(self.sigma_max), np.log(self.sigma_min), 1000, device=self.device
        )
        self.sigmas = torch.exp(self.log_sigmas)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Forward pass of the consistency model."""
        return self.model(x, sigma)

    def add_noise(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Add noise to clean data."""
        noise = torch.randn_like(x) * sigma.view(-1, 1, 1, 1)
        return x + noise

    def compute_loss(self, x0: torch.Tensor) -> torch.Tensor:
        """Compute consistency loss."""
        batch_size = x0.shape[0]
        
        # Sample noise levels
        log_sigma = torch.rand(batch_size, device=self.device) * (self.log_sigmas[-1] - self.log_sigmas[0]) + self.log_sigmas[0]
        sigma = torch.exp(log_sigma)
        
        # Add noise to clean data
        x_noisy = self.add_noise(x0, sigma)
        
        # Predict clean data
        x_pred = self.forward(x_noisy, sigma)
        
        # Consistency loss
        loss = F.mse_loss(x_pred, x0)
        
        return loss

    def sample(self, shape: Tuple[int, ...], num_steps: int = 2) -> torch.Tensor:
        """
        Sample from the consistency model.
        
        Args:
            shape: Shape of the samples to generate
            num_steps: Number of sampling steps (typically 1-2)
        """
        # Start from pure noise
        x = torch.randn(shape, device=self.device) * self.sigma_max
        
        # Consistency sampling
        for i in range(num_steps):
            if i == num_steps - 1:
                # Final step
                sigma = torch.full((shape[0],), self.sigma_min, device=self.device)
            else:
                # Intermediate step
                sigma = torch.full((shape[0],), self.sigma_max, device=self.device)
            
            # Predict clean data
            x = self.forward(x, sigma)
            
        return x

    def consistency_training_loss(self, x0: torch.Tensor, x1: torch.Tensor, 
                                sigma0: torch.Tensor, sigma1: torch.Tensor) -> torch.Tensor:
        """
        Compute consistency training loss between two noise levels.
        
        Args:
            x0: Clean data
            x1: Data at noise level sigma1
            sigma0: First noise level
            sigma1: Second noise level
        """
        # Add noise to clean data
        x0_noisy = self.add_noise(x0, sigma0)
        x1_noisy = self.add_noise(x1, sigma1)
        
        # Predict clean data from both noise levels
        x0_pred = self.forward(x0_noisy, sigma0)
        x1_pred = self.forward(x1_noisy, sigma1)
        
        # Consistency loss: predictions should be close
        loss = F.mse_loss(x0_pred, x1_pred)
        
        return loss

    def distillation_loss(self, teacher_model, x0: torch.Tensor) -> torch.Tensor:
        """
        Compute distillation loss for consistency model training.
        
        Args:
            teacher_model: Pre-trained diffusion model
            x0: Clean data
        """
        batch_size = x0.shape[0]
        
        # Sample noise levels
        log_sigma = torch.rand(batch_size, device=self.device) * (self.log_sigmas[-1] - self.log_sigmas[0]) + self.log_sigmas[0]
        sigma = torch.exp(log_sigma)
        
        # Add noise
        x_noisy = self.add_noise(x0, sigma)
        
        # Teacher prediction (from diffusion model)
        teacher_pred = teacher_model.reverse_diffusion(x_noisy, sigma)
        
        # Student prediction (consistency model)
        student_pred = self.forward(x_noisy, sigma)
        
        # Distillation loss
        loss = F.mse_loss(student_pred, teacher_pred)
        
        return loss

    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, 
                   num_steps: int = 10) -> List[torch.Tensor]:
        """
        Interpolate between two samples.
        
        Args:
            x1, x2: Two samples to interpolate between
            num_steps: Number of interpolation steps
        """
        interpolations = []
        
        for i in range(num_steps + 1):
            alpha = i / num_steps
            x_interp = (1 - alpha) * x1 + alpha * x2
            interpolations.append(x_interp)
            
        return interpolations

    def denoise(self, x_noisy: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Denoise a noisy sample."""
        return self.forward(x_noisy, sigma)
