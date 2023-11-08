import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class CarControlDataset(Dataset):
    def __init__(self, dataset_path: Path):
        super().__init__()
        self.image_paths = []
        car_controls = []
        for episode_path in dataset_path.iterdir():
            with open(episode_path / 'car_controls.json', 'r') as car_controls_file:
                episode_car_controls = json.load(car_controls_file)
                for controls in episode_car_controls:
                    self.image_paths.append(controls['image_path'])
                    car_controls.append(
                        [controls['steer'], controls['throttle'], controls['brake']]
                    )
        self.y = torch.Tensor(car_controls)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        return read_image(self.image_paths[index])[:3] / 255.0, self.y[index]
