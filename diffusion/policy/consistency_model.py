import torch
import torch.nn as nn
import torch.nn.functional as F

class ConsistencyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ConsistencyModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t):
        """Predicts a consistent mapping for sampling"""
        return self.network(x)
