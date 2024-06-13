from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import carla
import matplotlib.pyplot as plt
import numpy as np
from strenum import StrEnum

from config import config


class DatasetSplit(StrEnum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


@dataclass
class Episode:
    state_snapshots: list[StateSnapshot] = field(default_factory=list)
    town: Optional[str] = None

    def __post_init__(self):
        self._trajectory = None

    def snapshot_iterator(
        self, trim_start: float = 0, trim_end: float = 0
    ) -> Iterator[StateSnapshot]:
        min_timestamp = min(snapshot.timestamp for snapshot in self.state_snapshots) + trim_start
        max_timestamp = max(snapshot.timestamp for snapshot in self.state_snapshots) - trim_end

        for snapshot in self.state_snapshots:
            if snapshot.timestamp < min_timestamp:
                continue
            if snapshot.timestamp > max_timestamp:
                return
            yield snapshot

    @property
    def trajectory(self) -> Trajectory:
        if self._trajectory is None:
            self._trajectory = Trajectory(self.state_snapshots)
        return self._trajectory

    @classmethod
    def read_from_file(cls, folder_path: Path) -> Episode:
        with open(folder_path / 'episode_data.json') as file:
            episode_data = json.load(file)
        state_snapshots = list(
            StateSnapshot(**kwargs) for kwargs in episode_data['state_snapshots']
        )
        return cls(state_snapshots, episode_data['town'])

    def write_to_file(self, folder_path: Path):
        json_string = json.dumps(
            {
                'state_snapshots': list(asdict(state) for state in self.state_snapshots),
                'town': self.town,
            }
        )

        folder_path.mkdir(parents=True, exist_ok=True)
        with open(folder_path / 'episode_data.json', 'w') as file:
            file.write(json_string)


@dataclass
class StateSnapshot:
    frame: int
    timestamp: float
    image_path: str
    steer: float
    throttle: float
    brake: float
    position: tuple[float, float, float]
    orientation: tuple[float, float, float]
    velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]
    nav_prev_tp: tuple[float, float]
    nav_prev_command: int
    nav_current_tp: tuple[float, float]
    nav_current_command: int
    nav_next_tp: tuple[float, float]
    nav_next_command: int


def get_neighboring_indices(value: float, series: np.ndarray):
    if value < series[0]:
        i0, i1 = 0, 1
    elif value > series[-1]:
        i0, i1 = len(series) - 2, len(series) - 1
    else:
        insert_index = np.searchsorted(series, value)
        i0, i1 = insert_index - 1, insert_index
    return i0, i1


def linear_interpolation(x: float, x_values: np.ndarray, y_values: np.ndarray):
    i0, i1 = get_neighboring_indices(x, x_values)
    x0, x1 = x_values[i0], x_values[i1]
    y0, y1 = y_values[i0], y_values[i1]

    if x0 == x1:
        return y0
    return y0 + (y1 - y0) * ((x - x0) / (x1 - x0))


def transform_point_to_carla_transform_frame(point: np.ndarray, transform: carla.Transform):
    inv_transform = np.array(transform.get_inverse_matrix())
    return (inv_transform @ np.append(point, 1))[:3]


class Trajectory:
    """Represent the trajectory followed by a vehicle in a Episode.
    Points and other attributes can be sampled along the trajectory, and will be interpolated
    linearly from from the closest two datapoints.
    """

    def __init__(self, state_snapshots: list[StateSnapshot]):
        n = len(state_snapshots)
        self.timestamps = np.zeros(n)
        self.positions = np.zeros((n, 3))
        self.velocities = np.zeros((n, 3))
        self.orientations = np.zeros((n, 3))
        for i, state in enumerate(state_snapshots):
            self.timestamps[i] = state.timestamp
            self.positions[i] = state.position
            self.velocities[i] = state.velocity
            self.orientations[i] = state.orientation

        self.distances = np.zeros(n)
        for i in range(1, n):
            self.distances[i] = (
                np.sqrt(np.sum((self.positions[i] - self.positions[i - 1]) ** 2))
                + self.distances[i - 1]
            )

    def interpolated_position(self, timestamp: float) -> np.ndarray:
        return linear_interpolation(timestamp, self.timestamps, self.positions)

    def interpolated_position_by_distance(self, distance: float) -> np.ndarray:
        return linear_interpolation(distance, self.distances, self.positions)

    def interpolated_orientation(self, timestamp: float) -> np.ndarray:
        return linear_interpolation(timestamp, self.timestamps, self.orientations)

    def interpolated_distance(self, timestamp: float) -> float:
        return linear_interpolation(timestamp, self.timestamps, self.distances)

    def _carla_transform_at_timestamp(self, timestamp: float) -> carla.Transform:
        p0 = self.interpolated_position(timestamp)
        r0 = self.interpolated_orientation(timestamp)
        return carla.Transform(carla.Location(*p0), carla.Rotation(*r0))

    def sample_future_equitemporal_points(
        self, t0: float, num_points: int, sample_time_interval: float
    ) -> np.ndarray:
        """Sample points from future points on the trajectory.
        All points are in the forward-right-up frame of the car a time t0. The first point is
        sample_time_interval seconds after t0, then each point is sampled every
        sample_time_interval seconds."""
        carla_transform_t0 = self._carla_transform_at_timestamp(t0)

        points = np.empty((num_points, 3))
        for i in range(num_points):
            pi = self.interpolated_position(t0 + (i + 1) * sample_time_interval)
            points[i] = transform_point_to_carla_transform_frame(pi, carla_transform_t0)
        return points

    def sample_future_equidistant_points(
        self, t0: float, num_points: int, sample_distance_interval: float
    ) -> np.ndarray:
        """Sample points from future points on the trajectory.
        All points are in the forward-right-up frame of the car a time t0. The first point is
        sample_distance_interval meters after t0, then each point is sampled every
        sample_distance_interval meters."""
        carla_transform_t0 = self._carla_transform_at_timestamp(t0)
        d0 = self.interpolated_distance(t0)

        points = np.empty((num_points, 3))
        for i in range(num_points):
            pi = self.interpolated_position_by_distance(d0 + (i + 1) * sample_distance_interval)
            points[i] = transform_point_to_carla_transform_frame(pi, carla_transform_t0)
        return points

    def plot(self, plot_velocities=False):
        x, y, z = self.positions[:, 0], self.positions[:, 1], self.positions[:, 2]
        plt.plot(x, y, marker='o', label='position')

        if plot_velocities:
            vx, vy, vz = self.velocities[:, 0], self.velocities[:, 1], self.velocities[:, 2]
            plt.quiver(x, y, vx, vy, color='r', label='velocity')

    def plot_future_points(
        self,
        t0: float,
        num_points: int,
        sample_interval: float,
        sampling_domain: str = 'time',
    ):
        f = {
            'time': self.sample_future_equitemporal_points,
            'space': self.sample_future_equidistant_points,
        }
        future_points = f[sampling_domain](t0, num_points, sample_interval)
        x, y, z = future_points[:, 0], future_points[:, 1], future_points[:, 2]
        plt.plot(x, y, marker='x', label=f'future points sampled in {sampling_domain} domain')
