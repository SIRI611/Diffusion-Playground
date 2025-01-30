import torch
import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from diffusion.models.unet import UNet
from diffusion.policy.sgm import ScoreModel
from diffusion.policy.consistency_model import ConsistencyModel
from diffusion.policy.ddpm import DDPM  # Diffusion process helper


class Sampler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.sample.device
        self.model_type = cfg.sample.model_type
        self.model = self.load_model(cfg.sample.model_path)

    def load_model(self, model_path):
        """Load the appropriate model based on type."""
        if self.model_type == "diffusion":
            model = UNet(1, 1, 64).to(self.device)
        elif self.model_type == "score":
            model = ScoreModel(784, 128).to(self.device)  # 28x28 images
        elif self.model_type == "consistency":
            model = ConsistencyModel(784, 128).to(self.device)
        else:
            raise ValueError("Invalid model type. Choose from: diffusion, score, consistency.")

        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def sample(self):
        """Generate samples based on the selected model type."""
        if self.model_type == "diffusion":
            return self.sample_diffusion()
        elif self.model_type == "score":
            return self.sample_sgm()
        elif self.model_type == "consistency":
            return self.sample_consistency()
        else:
            raise ValueError("Invalid model type.")

    def sample_ddpm(self):
        """Sample using Diffusion Model (iterative denoising)."""
        diffusion = DDPM(self.model, timesteps=self.cfg.sample.timesteps, device=self.device)
        x_t = torch.randn((self.cfg.sample.batch_size, 1, 28, 28), device=self.device)  # Start with noise
        return diffusion.reverse_diffusion(x_t, self.cfg.sample.timesteps - 1)

    def sample_sgm(self):
        """Sample using Score-Based Model (SDE solver)."""
        x_t = torch.randn((self.cfg.sample.batch_size, 784), device=self.device)  # Flattened 28x28
        dt = 1.0 / self.cfg.sample.timesteps

        for t in reversed(range(self.cfg.sample.timesteps)):
            t_tensor = torch.full((self.cfg.sample.batch_size,), t, device=self.device, dtype=torch.float32) / self.cfg.sample.timesteps
            score = self.model(x_t, t_tensor)
            noise = torch.randn_like(x_t) * self.cfg.sample.sigma
            x_t = x_t + dt * score + noise  # Euler-Maruyama solver

        return x_t.view(-1, 1, 28, 28)  # Reshape for visualization

    def sample_consistency(self):
        """Sample using Consistency Model (one-step denoising)."""
        x_t = torch.randn((self.cfg.sample.batch_size, 784), device=self.device)
        t_tensor = torch.ones((self.cfg.sample.batch_size, 1), device=self.device)  # Single time step
        x_0 = self.model(x_t, t_tensor)
        return x_0.view(-1, 1, 28, 28)  # Reshape for visualization


def visualize_samples(samples):
    """Helper function to display generated samples."""
    samples = samples.cpu().detach().numpy()
    plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(samples[i, 0], cmap="gray")
        plt.axis("off")
    plt.show()


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(f"Sampling from {cfg.sample.model_type} model...")
    sampler = Sampler(cfg)
    samples = sampler.sample()
    visualize_samples(samples)


if __name__ == "__main__":
    main()
