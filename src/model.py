import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models import (
    EfficientNet,
    EfficientNet_B0_Weights,
    efficientnet_b0,
)


class DrivingModel(nn.Module):
    def __init__(self, out_size):
        super().__init__()

        self.feature_extractor: EfficientNet = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )
        self.feature_extractor.classifier = nn.Sequential(nn.Identity())

        self.regressor = nn.Sequential(
            nn.Linear(in_features=1280, out_features=out_size),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.regressor(x)
        return x
