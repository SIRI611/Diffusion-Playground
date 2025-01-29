import torch
import torch.nn.functional as F

class DDPMSampler:
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.model = model.to(device)
        self.timesteps = timesteps
        self.device = device
        
        # Define beta schedule
        self.beta = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alpha_cumprod[:-1]])

        # Precompute constants
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod)
        self.posterior_variance = self.beta * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)

    def sample(self, shape):
        """Sample new data by reversing the diffusion process."""
        x_t = torch.randn(shape, device=self.device)  # Start with pure noise
        for t in reversed(range(self.timesteps)):
            # Convert timestep to a tensor
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)

            # Predict noise using the model
            noise_pred = self.model(x_t, t_tensor)

            # Compute mean of the posterior
            mean = (
                self.sqrt_alpha_cumprod[t] * x_t
                - self.beta[t] * noise_pred / self.sqrt_one_minus_alpha_cumprod[t]
            )
            if t > 0:
                # Add noise for stochasticity
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(self.posterior_variance[t]) * noise
            else:
                # Last step (no noise)
                x_t = mean

        return x_t
