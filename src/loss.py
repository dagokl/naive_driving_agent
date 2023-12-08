import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    def __init__(self, weights, device):
        super().__init__()
        self.weights = torch.tensor(weights, device=device)

    def forward(self, predicted, target):
        return torch.mean(self.weights * (predicted - target) ** 2)
