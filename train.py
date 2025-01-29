import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import optim

from diffusion.utils.unet import UNet
from diffusion.models.ddpm import DDPM
import torch.nn.functional as F

@hydra.main(config_path="config", config_name="ddpm")
def main(cfg: DictConfig):
    # Device setup
    device = cfg.train.device

    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)
    
    # Model and diffusion
    model = UNet(cfg.model.in_channels, cfg.model.out_channels, cfg.model.hidden_channels).to(device)
    diffusion = DDPM(model, cfg.train.timesteps, cfg.train.beta_start, cfg.train.beta_end, device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)
    
    # Training loop
    for epoch in range(cfg.train.epochs):
        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)
            t = torch.randint(0, cfg.train.timesteps, (x.size(0),), device=device)
            
            # Forward diffusion
            xt, noise = diffusion.forward_diffusion(x, t)

            # Predict noise using U-Net
            pred_noise = model(xt)
            loss = F.mse_loss(pred_noise, noise)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{cfg.train.epochs}, Loss: {loss.item():.4f}")

 
if __name__ == "__main__":
    main()
