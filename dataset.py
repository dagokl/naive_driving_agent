import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class CarControlDataset(Dataset):
    def __init__(self, dataset_path: Path, episodes: list[str] | None = None):
        super().__init__()
        self.image_paths = []
        car_controls = []

        if episodes is None:
            episode_paths = dataset_path.iterdir()
        else:
            episode_paths = [dataset_path / episode for episode in episodes]

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


def get_train_test_car_control_datasets(
    dataset_path: Path, num_train_episodes: int, num_test_episodes: int
):
    episodes = [path.name for path in dataset_path.iterdir()]

    train_episodes = episodes[:num_train_episodes]
    test_episodes = episodes[num_train_episodes : num_train_episodes + num_test_episodes]

    assert len(train_episodes) == num_train_episodes
    assert len(test_episodes) == num_test_episodes

    train_dataset = CarControlDataset(dataset_path, train_episodes)
    test_dataset = CarControlDataset(dataset_path, test_episodes)

    return train_dataset, test_dataset
