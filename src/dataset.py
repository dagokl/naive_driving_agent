import json
from pathlib import Path

import numpy as np
import torch
from strenum import StrEnum
from torch.utils.data import Dataset
from torchvision.io import read_image


class DatasetSplit(StrEnum):
    TRAIN = 'train'
    TEST = 'test'


class CarControlDataset(Dataset):
    def __init__(self, dataset_path: Path, split: DatasetSplit | None = None):
        super().__init__()
        self.image_paths = []
        car_controls = []

        folder_prefix = split if split is not None else ''
        episode_paths = [
            episode_folder
            for episode_folder in dataset_path.iterdir()
            if episode_folder.name.startswith(folder_prefix)
        ]

        for episode_path in episode_paths:
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
