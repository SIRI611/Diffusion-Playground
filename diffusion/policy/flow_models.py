import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union, List
import math

class NormalizingFlow(nn.Module):
    """
    Normalizing Flow implementation.
    
    Based on: "Normalizing Flows for Probabilistic Modeling and Inference" by Papamakarios et al.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 8, device="cpu"):
        super(NormalizingFlow, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        
        # Build the flow
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(CouplingLayer(input_dim, hidden_dim))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: transform x to z and compute log determinant."""
        log_det = torch.zeros(x.shape[0], device=x.device)
        z = x
        
        for layer in self.layers:
            z, log_det_layer = layer(z)
            log_det += log_det_layer
            
        return z, log_det
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse pass: transform z back to x."""
        x = z
        
        for layer in reversed(self.layers):
            x = layer.inverse(x)
            
        return x
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log probability of x."""
        z, log_det = self.forward(x)
        
        # Log probability of z under standard Gaussian
        log_prob_z = -0.5 * (z**2).sum(dim=1) - 0.5 * self.input_dim * math.log(2 * math.pi)
        
        return log_prob_z + log_det
    
    def sample(self, num_samples: int) -> torch.Tensor:
        """Sample from the flow."""
        # Sample from base distribution (standard Gaussian)
        z = torch.randn(num_samples, self.input_dim, device=self.device)
        
        # Transform to data space
        x = self.inverse(z)
        
        return x

class CouplingLayer(nn.Module):
    """Affine coupling layer for normalizing flows."""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super(CouplingLayer, self).__init__()
        self.input_dim = input_dim
        self.split_dim = input_dim // 2
        
        # Network for computing scale and shift
        self.network = nn.Sequential(
            nn.Linear(self.split_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, (input_dim - self.split_dim) * 2)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through coupling layer."""
        x1, x2 = x[:, :self.split_dim], x[:, self.split_dim:]
        
        # Compute scale and shift
        params = self.network(x1)
        log_scale, shift = params.chunk(2, dim=1)
        
        # Apply transformation
        x2_transformed = x2 * torch.exp(log_scale) + shift
        
        # Combine
        z = torch.cat([x1, x2_transformed], dim=1)
        
        # Log determinant
        log_det = log_scale.sum(dim=1)
        
        return z, log_det
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse pass through coupling layer."""
        z1, z2 = z[:, :self.split_dim], z[:, self.split_dim:]
        
        # Compute scale and shift
        params = self.network(z1)
        log_scale, shift = params.chunk(2, dim=1)
        
        # Apply inverse transformation
        x2 = (z2 - shift) * torch.exp(-log_scale)
        
        # Combine
        x = torch.cat([z1, x2], dim=1)
        
        return x

class ContinuousNormalizingFlow(nn.Module):
    """
    Continuous Normalizing Flow (CNF) implementation.
    
    Based on: "Neural Ordinary Differential Equations" by Chen et al.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, device="cpu"):
        super(ContinuousNormalizingFlow, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Neural ODE network
        self.network = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for time
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass: compute dx/dt."""
        # Concatenate time
        t_expanded = t.view(-1, 1)
        xt = torch.cat([x, t_expanded], dim=1)
        
        return self.network(xt)
    
    def integrate(self, x0: torch.Tensor, t0: float = 0.0, t1: float = 1.0, 
                  num_steps: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate the ODE from t0 to t1.
        
        Args:
            x0: Initial state
            t0: Start time
            t1: End time
            num_steps: Number of integration steps
        """
        dt = (t1 - t0) / num_steps
        x = x0.clone()
        log_det = torch.zeros(x.shape[0], device=x.device)
        
        for i in range(num_steps):
            t = t0 + i * dt
            t_tensor = torch.full((x.shape[0],), t, device=x.device)
            
            # Compute dx/dt
            dx_dt = self.forward(x, t_tensor)
            
            # Euler step
            x = x + dt * dx_dt
            
            # Approximate log determinant (trace of Jacobian)
            # This is a simplified version - in practice, you'd use more sophisticated methods
            with torch.enable_grad():
                x_grad = x.requires_grad_(True)
                dx_dt_grad = self.forward(x_grad, t_tensor)
                trace = torch.autograd.grad(dx_dt_grad.sum(), x_grad, create_graph=True)[0]
                log_det += dt * trace.sum(dim=1)
        
        return x, log_det
    
    def log_prob(self, x: torch.Tensor, num_steps: int = 100) -> torch.Tensor:
        """Compute log probability of x."""
        # Integrate from data space to latent space
        z, log_det = self.integrate(x, t0=0.0, t1=1.0, num_steps=num_steps)
        
        # Log probability of z under standard Gaussian
        log_prob_z = -0.5 * (z**2).sum(dim=1) - 0.5 * self.input_dim * math.log(2 * math.pi)
        
        return log_prob_z + log_det
    
    def sample(self, num_samples: int, num_steps: int = 100) -> torch.Tensor:
        """Sample from the CNF."""
        # Sample from base distribution (standard Gaussian)
        z = torch.randn(num_samples, self.input_dim, device=self.device)
        
        # Integrate from latent space to data space
        x, _ = self.integrate(z, t0=1.0, t1=0.0, num_steps=num_steps)
        
        return x

class FlowBasedModel:
    """
    Wrapper class for flow-based models with training utilities.
    """
    
    def __init__(self, flow_model, device="cpu"):
        self.flow_model = flow_model.to(device)
        self.device = device
    
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute negative log likelihood loss."""
        log_prob = self.flow_model.log_prob(x)
        return -log_prob.mean()
    
    def sample(self, num_samples: int) -> torch.Tensor:
        """Sample from the model."""
        return self.flow_model.sample(num_samples)
    
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, 
                   num_steps: int = 10) -> List[torch.Tensor]:
        """Interpolate between two samples in latent space."""
        # Transform to latent space
        z1, _ = self.flow_model.forward(x1)
        z2, _ = self.flow_model.forward(x2)
        
        # Interpolate in latent space
        interpolations = []
        for i in range(num_steps + 1):
            alpha = i / num_steps
            z_interp = (1 - alpha) * z1 + alpha * z2
            
            # Transform back to data space
            x_interp = self.flow_model.inverse(z_interp)
            interpolations.append(x_interp)
            
        return interpolations
