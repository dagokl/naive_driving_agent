import random
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    EfficientNet,
    EfficientNet_B0_Weights,
    efficientnet_b0,
)


class ResidualBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=size, out_features=size),
            nn.ReLU(),
            nn.Linear(in_features=size, out_features=size),
            nn.ReLU(),
        )

    def forward(self, x):
        residual = self.layers(x)
        return x + residual


class DrivingModel(nn.Module):
    def __init__(
        self,
        out_size,
        train_vision_bottleneck=False,
        vision_bottleneck_capacity=0,
        train_ignore_vision_prob=0.0,
        ignore_vision_feature_vector_zeros=False,
        ignore_vision_feature_vector_random=True,
        totally_ignore_vision=False,
        totally_ignore_tp=False,
    ):
        super().__init__()
        self.train_vision_bottleneck = train_vision_bottleneck
        self.vision_bottleneck_capacity = vision_bottleneck_capacity
        self.totally_ignore_vision = totally_ignore_vision
        self.totally_ignore_tp = totally_ignore_tp

        self.train_ignore_vision_prob = train_ignore_vision_prob
        self.ignore_vision_feature_vector_zeros = ignore_vision_feature_vector_zeros
        self.ignore_vision_feature_vector_random = ignore_vision_feature_vector_random

        self.feature_extractor: EfficientNet = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )
        self.feature_extractor.classifier = nn.Sequential(nn.Identity())

        self.img_feature_size = 1280

        nav_tps_size = 3 * 2
        nav_cmds_size = 3
        self.regressor = nn.Sequential(
            nn.Linear(
                in_features=self.img_feature_size + nav_tps_size + nav_cmds_size, out_features=128
            ),
            nn.ReLU(),
            ResidualBlock(128),
            nn.Linear(in_features=128, out_features=out_size),
        )

    def forward(self, img, nav_tps=None, nav_cmds=None, ignore_vision=False):
        device = next(self.regressor.parameters()).device
        batch_size = img.shape[0]
        if nav_tps is None or self.totally_ignore_tp:
            nav_tps = torch.zeros((batch_size, 3 * 2), device=device)

        if nav_cmds is None:
            nav_cmds = torch.zeros((batch_size, 3), device=device)

        ignore_vision = (
            self.training and random.random() < self.train_ignore_vision_prob
        ) or self.totally_ignore_vision
        if not ignore_vision:
            img_features = self.feature_extractor(img)
            img_features = F.normalize(img_features, dim=-1)
            if self.train_vision_bottleneck and self.training:
                signal_to_noise = 2 ** (2 * self.vision_bottleneck_capacity) - 1
                scale_factor = 1 / sqrt(signal_to_noise * img_features.shape[-1])
                noise = torch.randn(img_features.shape, device=device)
                img_features = img_features + scale_factor * noise
                img_features = F.normalize(img_features, dim=-1)
        else:
            if self.ignore_vision_feature_vector_zeros:
                img_features = torch.zeros((batch_size, self.img_feature_size)).to(device)
            elif self.ignore_vision_feature_vector_random:
                img_features = torch.randn((batch_size, self.img_feature_size), device=device)
            else:
                raise ValueError()
            img_features = F.normalize(img_features, dim=-1)

        reg_input = torch.cat((img_features, nav_tps, nav_cmds), dim=1)
        return self.regressor(reg_input)

    @staticmethod
    def load_from_config(config):
        path = config['eval.model_path']

        totally_ignore_vision = config['model.totally_ignore_vision']
        totally_ignore_tp = config['model.totally_ignore_tp']

        if config['model.predict.type'] == 'waypoints':
            out_size = 2 * config['model.predict.num_waypoints']
        elif config['model.predict.type'] == 'direct_controls':
            out_size = 3
        else:
            ValueError()
        model = DrivingModel(
            out_size,
            train_vision_bottleneck=False,
            totally_ignore_vision=totally_ignore_vision,
            totally_ignore_tp=totally_ignore_tp,
        )
        model.load_state_dict(torch.load(path))
        model.eval()

        return model
