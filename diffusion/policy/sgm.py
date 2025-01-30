import torch
import torch.nn as nn
import torch.nn.functional as F

class ScoreModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ScoreModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t):
        """Predicts the score function ∇ log p(x_t)"""
        return self.network(x)