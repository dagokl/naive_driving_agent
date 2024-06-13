from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from strenum import StrEnum
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2 as transforms

from config import config
from episode import DatasetSplit, Episode


class DirectControlDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        img_transform: transforms.Transform | None = None,
        excluded_towns: list[str] = [],
    ):
        super().__init__()
        self.img_transform = img_transform
        self.image_paths = []

        episode_paths = [episode_folder for episode_folder in dataset_path.iterdir()]

        tps_flattend = []
        commands_flattend = []
        car_controls = []
        for episode_path in episode_paths:
            episode = Episode.read_from_file(episode_path)


            if episode.town in excluded_towns:
                continue

            for state_snapshot in episode.state_snapshots:
                self.image_paths.append((dataset_path / state_snapshot.image_path).as_posix())

                tps_flattend.append(
                    state_snapshot.nav_prev_tp
                    + state_snapshot.nav_current_tp
                    + state_snapshot.nav_next_tp
                )
                commands_flattend.append(
                    (
                        state_snapshot.nav_prev_command,
                        state_snapshot.nav_current_command,
                        state_snapshot.nav_next_command,
                    )
                )

                car_controls.append(
                    [state_snapshot.steer, state_snapshot.throttle, state_snapshot.brake]
                )
        self.tps = torch.tensor(tps_flattend, dtype=torch.float32)
        self.commands = torch.tensor(commands_flattend, dtype=torch.float32)
        self.y = torch.Tensor(car_controls)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = read_image(self.image_paths[index])[:3] / 255.0
        if self.img_transform:
            img = self.img_transform(img)
        return img, self.tps[index], self.commands[index], self.y[index]


class WaypointSamplingMethod(Enum):
    EQUIDISTANT_SPATIAL_SAMPLING = 'equidistant_spatial_sampling'
    UNIFORM_TIME_SAMPLING = 'uniform_time_sampling'


class WaypointPredictionDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        num_waypoints: int,
        waypoint_sampling_interval: float,
        waypoint_sampling_method: WaypointSamplingMethod,
        img_transform: transforms.Transform | None = None,
        excluded_towns: list[str] = [],
    ):
        super().__init__()
        self.img_transform = img_transform
        self.image_paths = []

        episode_paths = [episode_folder for episode_folder in dataset_path.iterdir()]

        tps_flattend = []
        commands_flattend = []
        target_wps_flattened = []
        for episode_path in episode_paths:
            episode = Episode.read_from_file(episode_path)

            if episode.town in excluded_towns:
                continue

            sampling_func = {
                WaypointSamplingMethod.EQUIDISTANT_SPATIAL_SAMPLING: episode.trajectory.sample_future_equidistant_points,
                WaypointSamplingMethod.UNIFORM_TIME_SAMPLING: episode.trajectory.sample_future_equitemporal_points,
            }[waypoint_sampling_method]

            for state_snapshot in episode.snapshot_iterator(trim_end=5):
                self.image_paths.append((dataset_path / state_snapshot.image_path).as_posix())

                tps_flattend.append(
                    state_snapshot.nav_prev_tp
                    + state_snapshot.nav_current_tp
                    + state_snapshot.nav_next_tp
                )
                commands_flattend.append(
                    (
                        state_snapshot.nav_prev_command,
                        state_snapshot.nav_current_command,
                        state_snapshot.nav_next_command,
                    )
                )

                future_points = sampling_func(
                    t0=state_snapshot.timestamp,
                    num_points=num_waypoints,
                    sample_distance_interval=waypoint_sampling_interval,
                )
                target_wps_flattened.append(future_points[:, :2].flatten())
        self.tps = torch.tensor(tps_flattend, dtype=torch.float32)
        self.commands = torch.tensor(commands_flattend, dtype=torch.float32)
        self.y = torch.tensor(np.array(target_wps_flattened), dtype=torch.float32)

        # TODO: Troubleshot very small portion of nan can occur in tps e.g. 6 values total in 174894 nav vectors, fixed by regenerating dataset.
        if torch.isnan(self.tps).any():
            nan_count = torch.isnan(self.tps).sum()
            print(
                f'Warning: nan in dataset nav tps. {nan_count} of {self.tps.numel()} values. Will be replaced with zeros'
            )
            self.tps[torch.isnan(self.tps)] = 0.0

        assert not torch.isnan(self.tps).any(), f'nan in dataset nav tps'
        assert not torch.isnan(self.commands).any(), 'nan in dataset nav commands'
        assert not torch.isnan(self.y).any(), 'nan in dataset target values'

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = read_image(self.image_paths[index])[:3] / 255.0
        if self.img_transform:
            img = self.img_transform(img)
        return img, self.tps[index], self.commands[index], self.y[index]
