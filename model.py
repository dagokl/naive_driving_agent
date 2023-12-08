import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models import (
    EfficientNet,
    EfficientNet_B0_Weights,
    efficientnet_b0,
)
from torchvision.models._api import WeightsEnum


# Fix to get around problem with downloading pretrained weights
def get_state_dict(self, *args, **kwargs):
    kwargs.pop('check_hash')
    return load_state_dict_from_url(self.url, *args, **kwargs)
WeightsEnum.get_state_dict = get_state_dict


class SimpleCNN(nn.Module):
    def __init__(self, image_x: int, image_y: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 16, 3, 1, 1),
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


class DrivingModel(nn.Module):
    def __init__(self, nav_instruction_encoding_len: int = 7):
        super().__init__()

        self.feature_extractor: EfficientNet = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )
        self.feature_extractor.classifier = nn.Sequential(nn.Identity())

        self.regressor = nn.Sequential(
            nn.Linear(in_features=nav_instruction_encoding_len + 1280, out_features=3),
        )

    def forward(self, images, nav_instructions):
        features = self.feature_extractor(images)
        regressor_input = torch.cat((nav_instructions, features), dim=1)
        return self.regressor(regressor_input)
