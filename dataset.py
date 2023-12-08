import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

NAV_INSTRUCTION_INDEX_MAP = {
    'left': 0,
    'right': 1,
    'straight': 2,
    'lanefollow': 3,
    'changelaneleft': 4,
    'changelaneright': 5,
}


def encode_nav_instruction(instruction: str, distance: float) -> list[float]:
    instruction_index = NAV_INSTRUCTION_INDEX_MAP.get(instruction.lower(), 3) 

    encoded = [0.0 for _ in range(7)]
    encoded[instruction_index] = 1.0
    encoded[-1] = distance

    return encoded


class CarControlDataset(Dataset):
    def __init__(self, dataset_path: Path, episodes: list[str] | None = None):
        super().__init__()
        self.image_paths = []
        encoded_actions = []
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

                    action_type, action_distance = controls['next_action'].values()
                    encoded_actions.append(encode_nav_instruction(action_type, action_distance))

                    car_controls.append(
                        [controls['steer'], controls['throttle'], controls['brake']]
                    )

        self.encoded_nav_instructions = torch.Tensor(encoded_actions)
        self.y = torch.Tensor(car_controls)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        return (
            read_image(self.image_paths[index])[:3] / 255.0,
            self.encoded_nav_instructions[index],
            self.y[index],
        )


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
