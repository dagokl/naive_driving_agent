import datetime
import json
import random
import time
from pathlib import Path
from typing import Any

import carla
from tqdm import tqdm

from carla_utils import create_and_attach_camera, spawn_ego_vehicle
from config import config

dataset_path = Path(config.get('dataset.folder_path'))


def save_data(
    image: carla.Image,
    vehicle: carla.Actor,
    traffic_manager: carla.TrafficManager,
    path: Path,
    episode_state,
):
    if 'start_timestamp' not in episode_state:
        episode_state['start_timestamp'] = image.timestamp
        episode_state['start_frame'] = image.frame

    elapsed = image.timestamp - episode_state['start_timestamp']
    if elapsed > config.get('dataset.episode_length'):
        episode_state['done'] = True
        return

    frame = image.frame - episode_state['start_frame']
    image_path = path / f'images/{frame:06}.png'
    control: carla.VehicleControl = vehicle.get_control()

    next_action, next_action_wp = traffic_manager.get_next_action(vehicle)
    distance_to_next_action = vehicle.get_location().distance(next_action_wp.transform.location)

    episode_state['car_controls'].append(
        {
            'frame': frame,
            'image_path': image_path.as_posix(),
            'timestamp': elapsed,
            'steer': control.steer,
            'throttle': control.throttle,
            'brake': control.brake,
            'next_action': {'type': next_action, 'distance': distance_to_next_action},
        }
    )

    image.save_to_disk(image_path.as_posix())


def gather_episode(world: carla.World, client: carla.Client):
    start_datetime = datetime.datetime.now()
    episode_path = dataset_path / start_datetime.strftime('%Y_%m_%d_%H_%M_%S')
    episode_path.mkdir(parents=True, exist_ok=True)

    ego_vehicle = spawn_ego_vehicle(world)
    time.sleep(0.1)
    ego_vehicle.set_autopilot(True)

    traffic_manager = client.get_trafficmanager()
    if config.get('dataset.ignore_traffic_lights'):
        traffic_manager.ignore_lights_percentage(ego_vehicle, 100)

    traffic_manager.auto_lane_change(ego_vehicle, False)
    traffic_manager.ignore_signs_percentage(ego_vehicle, 100)
    traffic_manager.keep_right_rule_percentage(ego_vehicle, 0)

    time.sleep(5)

    camera = create_and_attach_camera(
        world,
        ego_vehicle,
        config.get('camera.resolution.width'),
        config.get('camera.resolution.height'),
    )
    episode_state: dict[str, Any] = {'done': False, 'car_controls': []}
    camera.listen(
        lambda image: save_data(image, ego_vehicle, traffic_manager, episode_path, episode_state)
    )

    start_time = time.time()
    while not episode_state['done']:
        time.sleep(0.5)

    with open(episode_path / 'car_controls.json', 'w') as file:
        file.write(json.dumps(episode_state['car_controls']))

    camera.destroy()
    ego_vehicle.destroy()


def main():
    client = carla.Client('localhost', 2000)
    world = client.load_world(config.get('dataset.town'))

    traffic_manager: carla.TrafficManager = client.get_trafficmanager()

    num_episodes = config.get('dataset.num_episodes')
    episode_length = config.get('dataset.episode_length')
    for _ in tqdm(range(num_episodes)):
        gather_episode(world, client)


if __name__ == '__main__':
    main()
