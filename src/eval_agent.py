import os
from pathlib import Path

import carla
import cv2
import numpy as np
import torch
from leaderboard.autoagents.autonomous_agent import AutonomousAgent
from nav_planner import RoutePlanner

from agent_utils import AgentPlotter, get_surronding_tps
from config import config
from driving_model import DrivingModel
from simple_controller import VehiclePIDController

np.set_printoptions(precision=2, suppress=True)


def get_entry_point():
    return 'EvalAgent'


device = torch.device('cuda:0')


class EvalAgent(AutonomousAgent):
    def setup(self, path_to_conf_file, route_index=None):
        self.save_input_images = config['eval.save_input_images']
        self.save_external_images = config['eval.save_external_images']

        self.result_dir = Path(os.getenv('RESULT_DIR'))
        self.episode_result_dir = self.result_dir / route_index
        self.image_dir = self.episode_result_dir / 'images'
        self.plot_dir = self.episode_result_dir / 'plots'

        self.model = DrivingModel.load_from_config(config)
        self.controller = VehiclePIDController()

        self.agent_plotter = AgentPlotter(
            config['eval.show_plots'], config['eval.save_plots'], self.plot_dir
        )

        self.first_frame = 0
        self.initlized = False

    def first_step_setup(self, frame):
        route_planner_min_distance = 7.5
        route_planner_max_distance = 50.0
        self.route_planner = RoutePlanner(route_planner_min_distance, route_planner_max_distance)
        self.route_planner.set_route(self._global_plan, True)

        self.agent_plotter.init_route(self.route_planner.route)

        self.first_frame = frame
        self.initlized = True

    def sensors(self):
        result = super().sensors()
        dt = 0.05
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
                'type': 'sensor.other.imu',
                'x': 0.0,
                'y': 0.0,
                'z': 0.0,
                'roll': 0.0,
                'pitch': 0.0,
                'yaw': 0.0,
                'sensor_tick': dt,
                'id': 'imu',
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
            {'type': 'sensor.speedometer', 'reading_frequency': dt, 'id': 'speed'},
        ]
        if self.save_external_images:
            for camera in config['eval.external_cameras']:
                result.append(
                    {
                        'type': 'sensor.camera.rgb',
                        'width': camera['resolution']['width'],
                        'height': camera['resolution']['height'],
                        'fov': camera['fov'],
                        'x': camera['x'],
                        'y': camera['y'],
                        'z': camera['z'],
                        'roll': camera['roll'],
                        'pitch': camera['pitch'],
                        'yaw': camera['yaw'],
                        'id': camera['name'],
                    }
                )
        return result

    @torch.inference_mode()
    def run_step(self, input_data, timestamp, sensors=None, plant=False):
        if 'rgb' not in input_data:
            control = carla.VehicleControl(brake=1.0)
            return control

        frame = input_data['rgb'][0]
        if not self.initlized:
            self.first_step_setup(frame)
        relative_frame = frame - self.first_frame

        _, raw_image = input_data['rgb']
        image_transposed = np.transpose(raw_image[:, :, :3], (2, 0, 1))
        image_exp = np.expand_dims(image_transposed, axis=0)
        image_scaled = image_exp / 255.0
        image_tensor = torch.from_numpy(image_scaled.astype(np.float32))

        current_speed = input_data['speed'][1]['speed']

        _, imu_data = input_data['imu']
        # imu_acc = imu_data[0:3]
        # imu_gyro = imu_data[3:6]
        imu_compass = imu_data[6]

        _, gps_coordinates = input_data['gps']
        carla_pos = self.route_planner.convert_gps_to_carla(gps_coordinates[:2])

        self.route_planner.run_step(carla_pos)
        # prev_tp, prev_command, current_tp, current_command, next_tp, next_command = (
        #     get_surronding_tps_and_commands(self.route_planner.route, carla_pos, imu_compass)
        # )
        ego_tps, local_tps = get_surronding_tps(self.route_planner.route, carla_pos, imu_compass)

        # tp_vector = np.concatenate((prev_tp, current_tp, next_tp))
        tp_vector = np.concatenate(ego_tps)
        tp_tensor = torch.tensor(tp_vector, dtype=torch.float32).unsqueeze(0)
        # command_vector = np.array((prev_command.value, current_command.value, next_command.value))
        # command_tensor = torch.tensor(command_vector, dtype=torch.float32).unsqueeze(0)

        output = self.model(image_tensor, tp_tensor, None)
        output_numpy = output[0].cpu().detach().numpy()

        waypoint_index_for_control = 1
        num_waypoints = config['model.predict.num_waypoints']

        target_vectors = output_numpy.reshape((num_waypoints, 2))

        target_vec = target_vectors[waypoint_index_for_control]
        target_speed = 30 / 3.6

        control = self.controller.run_step(target_vec, target_speed, current_speed)

        self.save_images(input_data, relative_frame)
        self.agent_plotter.draw_step(carla_pos, target_vectors, ego_tps, local_tps, relative_frame)

        return control

    def save_images(self, input_data, relative_frame):
        if self.save_input_images:
            _, img_raw = input_data['rgb']
            self.save_image(relative_frame, img_raw, 'input')

        if self.save_external_images:
            for camera in config['eval.external_cameras']:
                camera_name = camera['name']
                _, img_raw = input_data[camera_name]
                self.save_image(relative_frame, img_raw, camera_name)

    def save_image(self, frame: int, img_raw: np.ndarray, camera_name: str):
        save_path = self.image_dir / camera_name / f'{frame:06}.jpg'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path.as_posix(), img_raw)

    def destroy(self, results=None):
        super().destroy()
        pass
