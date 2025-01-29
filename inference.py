import torch
from diffusion.models.unet import UNet
from diffusion.policy.ddpm_sampler import DiffusionSampler

# Load the trained U-Net model
model = UNet(in_channels=1, out_channels=1, hidden_channels=64)
model.load_state_dict(torch.load("unet_model.pth"))
model.eval()

# Initialize the diffusion sampler
sampler = DiffusionSampler(model, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda")

# Generate samples
samples = sampler.sample((16, 1, 28, 28))  # Shape (batch_size, channels, height, width)

# Save or visualize the results
import matplotlib.pyplot as plt

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(samples[i, 0].cpu().detach().numpy(), cmap="gray")
    plt.axis("off")
plt.show()