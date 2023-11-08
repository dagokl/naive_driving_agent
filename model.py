import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, image_x: int, image_y: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
        )

        self.regressor = nn.Sequential(
            nn.Linear(int(16 * image_x * image_y * 2 ** (-4)), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x
