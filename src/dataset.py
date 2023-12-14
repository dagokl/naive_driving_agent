import json
from pathlib import Path

import numpy as np
import torch
from strenum import StrEnum
from torch.utils.data import Dataset
from torchvision.io import read_image


class DatasetSplit(StrEnum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class CarControlDataset(Dataset):
    def __init__(self, dataset_path: Path, split: DatasetSplit | None = None):
        super().__init__()
        self.image_paths = []
        car_controls = []

        split_folders = (
            [dataset_path / split]
            if split is not None
            else [dataset_path / s for s in DatasetSplit]
        )

        episode_paths = [
            episode_folder
            for split_folder in split_folders
            for episode_folder in split_folder.iterdir()
        ]

        for episode_path in episode_paths:
            with open(episode_path / 'car_controls.json', 'r') as car_controls_file:
                episode_car_controls = json.load(car_controls_file)
                for controls in episode_car_controls:
                    self.image_paths.append((dataset_path / controls['image_path']).as_posix())
                    car_controls.append(
                        [controls['steer'], controls['throttle'], controls['brake']]
                    )
        self.y = torch.Tensor(car_controls)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        return read_image(self.image_paths[index])[:3] / 255.0, self.y[index]
