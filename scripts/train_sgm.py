import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import optim
from diffusion.policy.sgm import ScoreModel

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    device = cfg.train.device

    # Load dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)

    # Initialize score-based model
    model = ScoreModel(cfg.model.input_dim, cfg.model.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)

    for epoch in range(cfg.train.epochs):
        for x, _ in dataloader:
            x = x.view(x.size(0), -1).to(device)  # Flatten the input
            t = torch.rand(x.size(0), 1).to(device)  # Random time step

            # Predict score function
            score = model(x, t)

            # Compute loss
            loss = torch.mean((score + x) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{cfg.train.epochs}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()