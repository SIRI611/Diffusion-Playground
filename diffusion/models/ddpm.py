import torch
import torch.nn.functional as F
from torch import nn

class DDPM:
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.model = model.to(device)
        self.timesteps = timesteps
        self.device = device
        
        # Beta schedule
        self.beta = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)

    def forward_diffusion(self, x0, t):
        noise = torch.randn_like(x0).to(self.device)
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t]).unsqueeze(-1).unsqueeze(-1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod[t]).unsqueeze(-1).unsqueeze(-1)
        xt = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise
        return xt, noise

    def reverse_diffusion(self, x, t):
        return self.model(x, t)
