from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from strenum import StrEnum
from torch.utils.data import Dataset
from torchvision.io import read_image

from config import config
from episode import DatasetSplit, Episode


class DirectControlDataset(Dataset):
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
            episode = Episode.read_from_file(episode_path)
            for state_snapshot in episode.state_snapshots:
                self.image_paths.append((dataset_path / state_snapshot.image_path).as_posix())
                car_controls.append(
                    [state_snapshot.steer, state_snapshot.throttle, state_snapshot.brake]
                )
        self.y = torch.Tensor(car_controls)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        return read_image(self.image_paths[index])[:3] / 255.0, self.y[index]


class WaypointPredictionDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        num_waypoints: int,
        waypoint_sampling_interval: float,
        split: DatasetSplit | None = None,
    ):
        super().__init__()
        self.image_paths = []
        waypoints = []

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

        target_wps_flattened = []
        for episode_path in episode_paths:
            episode = Episode.read_from_file(episode_path)
            for state_snapshot in episode.snapshot_iterator(trim_end=5):
                self.image_paths.append((dataset_path / state_snapshot.image_path).as_posix())

                future_points = episode.trajectory.sample_future_equidistant_points(
                    t0=state_snapshot.timestamp,
                    num_points=num_waypoints,
                    sample_distance_interval=waypoint_sampling_interval,
                )
                target_wps_flattened.append(future_points.flatten())
        self.y = torch.Tensor(np.array(target_wps_flattened))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        return read_image(self.image_paths[index])[:3] / 255.0, self.y[index]
