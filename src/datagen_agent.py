from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import carla
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from carla_garage.team_code.autopilot import AutoPilot
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from strenum import StrEnum

from agent_utils import get_surronding_tps_and_commands, setup_route_plot
from config import config
from episode import DatasetSplit, Episode, StateSnapshot


def get_entry_point():
    return 'NaiveDataAgent'


@dataclass
class EpisodeGenerationState:
    episode: Episode
    start_timestamp: float | None = None
    start_frame: int = 0
    done: bool = False


plot_route_stuff = config['agent.plot']


class NaiveDataAgent(AutoPilot):
    def setup(self, path_to_conf_file, route_index):
        print(f'Call to setup(path_to_conf_file={path_to_conf_file}, route_index={route_index})')
        super().setup(path_to_conf_file, route_index)

        self.ignore_stop_signs = config['dataset.ignore_stop_signs']
        self.ignore_red_lights = config['dataset.ignore_red_lights']

        episodes_relative_save_path = os.getenv('ROUTES')
        self.dataset_path = Path(config['dataset.folder_path'])
        self.episode_path = self.dataset_path / episodes_relative_save_path / route_index
        self.image_folder = self.episode_path / 'images'
        self.image_folder.mkdir(parents=True, exist_ok=True)

        episode = Episode()
        self.episode_generation_state = EpisodeGenerationState(episode)

        self.plot_initilized = False

    def first_step_setup(self):
        self.episode_generation_state.episode.town = (
            CarlaDataProvider.get_world().get_map().name
        )

        if plot_route_stuff:
            setup_route_plot(self._command_planner.route)
        self.plot_initilized = True

    def sensors(self):
        result = super().sensors()
        result += [
            {
                'type': 'sensor.camera.rgb',
                'width': config['camera.resolution.width'],
                'height': config['camera.resolution.height'],
                'fov': config['camera.fov'],
                'x': config['camera.x'],
                'y': config['camera.y'],
                'z': config['camera.z'],
                'roll': config['camera.roll'],
                'pitch': config['camera.pitch'],
                'yaw': config['camera.yaw'],
                'id': 'rgb',
            },
            {
                'type': 'sensor.other.gnss',
                'noise_alt_bias': 0.0,
                'noise_alt_stddev': 0.0,
                'noise_lat_bias': 0.0,
                'noise_lat_stddev': 0.0,
                'noise_lon_bias': 0.0,
                'noise_lon_stdde': 0.0,
                'x': 0.0,
                'y': 0.0,
                'z': 0.0,
                'roll': 0.0,
                'pitch': 0.0,
                'yaw': 0.0,
                'sensor_tick': 0.05,
                'id': 'gps',
            },
        ]
        return result

    @torch.inference_mode()
    def run_step(self, input_data, timestamp, sensors=None, plant=False):
        if 'rgb' not in input_data or 'hd_map' not in input_data and not self.initialized:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            return control

        control = super().run_step(input_data, timestamp, plant=plant)

        if not self.plot_initilized:
            self.first_step_setup()

        frame, image = input_data['rgb']

        if self.episode_generation_state.start_timestamp is None:
            self.episode_generation_state.start_timestamp = timestamp
            self.episode_generation_state.start_frame = frame

        frame = frame - self.episode_generation_state.start_frame
        elapsed = timestamp - self.episode_generation_state.start_timestamp

        image_path = self.image_folder / f'{frame:06}.jpg'
        cv2.imwrite(image_path.as_posix(), image)

        ego_transform: carla.Transform = self._vehicle.get_transform()
        loc = ego_transform.location
        position = (loc.x, loc.y, loc.z)
        rot = ego_transform.rotation
        orientation = (rot.pitch, rot.yaw, rot.roll)

        vel = self._vehicle.get_velocity()
        velocity = (vel.x, vel.y, vel.z)

        ang_vel = self._vehicle.get_angular_velocity()
        angular_velocity = (ang_vel.x, ang_vel.y, ang_vel.z)

        _, imu_data = input_data['imu']
        imu_acc = imu_data[0:3]
        imu_gyro = imu_data[3:6]
        imu_compass = imu_data[6]

        _, gps_coordinates = input_data['gps']
        carla_pos = self._command_planner.convert_gps_to_carla(gps_coordinates[:2])

        prev_tp, prev_command, current_tp, current_command, next_tp, next_command = (
            get_surronding_tps_and_commands(self._command_planner.route, carla_pos, imu_compass)
        )

        state_snapshot = StateSnapshot(
            frame,
            elapsed,
            Path(*image_path.parts[3:]).as_posix(),
            control.steer,
            control.throttle,
            control.brake,
            position,
            orientation,
            velocity,
            angular_velocity,
            tuple(prev_tp),
            prev_command.value,
            tuple(current_tp),
            current_command.value,
            tuple(next_tp),
            next_command.value,
        )

        self.episode_generation_state.episode.state_snapshots.append(state_snapshot)

        return control

    def destroy(self, results):
        self.episode_generation_state.episode.write_to_file(self.episode_path)
        super().destroy(results)
