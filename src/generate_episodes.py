import datetime
import json
import random
import shutil
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import carla
from carla_utils import create_and_attach_camera, spawn_ego_vehicle
from config import config
from dataset import DatasetSplit
from strenum import StrEnum
from tqdm import tqdm

dataset_path = Path(config['dataset.folder_path'])


@dataclass
class EpisodeSettings:
    town: str
    split: DatasetSplit
    length: float = config['dataset.episode_length']


def episode_settings_from_file(file_path: Path) -> EpisodeSettings:
    with open(file_path, 'r') as file:
        json_data = json.loads(file.read())
        return EpisodeSettings(**json_data)


def save_data_callback(image: carla.Image, vehicle: carla.Actor, path: Path, episode_state):
    if 'start_timestamp' not in episode_state:
        episode_state['start_timestamp'] = image.timestamp
        episode_state['start_frame'] = image.frame

    elapsed = image.timestamp - episode_state['start_timestamp']
    if elapsed > config['dataset.episode_length']:
        episode_state['done'] = True
        return

    frame = image.frame - episode_state['start_frame']
    image_path = path / f'images/{frame:06}.png'
    control: carla.VehicleControl = vehicle.get_control()
    episode_state['car_controls'].append(
        {
            'frame': frame,
            'image_path': Path(*image_path.parts[2:]).as_posix(),
            'timestamp': elapsed,
            'steer': control.steer,
            'throttle': control.throttle,
            'brake': control.brake,
        }
    )

    image.save_to_disk(image_path.as_posix())


def create_episode_settings_list() -> list[EpisodeSettings]:
    episode_settings = []

    def uniform_town_counts(towns, n):
        town_counts = {}
        for i, town in enumerate(towns):
            town_counts[town] = n // len(towns) + (1 if i < n % len(towns) else 0)
        return town_counts

    num_train_episodes = config['dataset.num_train_episodes']
    train_towns = config['dataset.train_towns']
    train_town_counts = uniform_town_counts(train_towns, num_train_episodes)

    num_val_episodes = config['dataset.num_val_episodes']
    val_towns = config['dataset.val_towns']
    val_town_counts = uniform_town_counts(val_towns, num_val_episodes)

    num_test_episodes = config['dataset.num_test_episodes']
    test_towns = config['dataset.test_towns']
    test_town_counts = uniform_town_counts(test_towns, num_test_episodes)

    for _ in range(num_train_episodes):
        town = random.choice([town for town, count in train_town_counts.items() if count > 0])
        train_town_counts[town] -= 1
        episode_settings.append(EpisodeSettings(town=town, split=DatasetSplit.TRAIN))

    for _ in range(num_val_episodes):
        town = random.choice([town for town, count in val_town_counts.items() if count > 0])
        val_town_counts[town] -= 1
        episode_settings.append(EpisodeSettings(town=town, split=DatasetSplit.VAL))

    for _ in range(num_test_episodes):
        town = random.choice([town for town, count in test_town_counts.items() if count > 0])
        test_town_counts[town] -= 1
        episode_settings.append(EpisodeSettings(town=town, split=DatasetSplit.TEST))

    # Check if dataset is partially generated
    # If so remove already generated episodes from episode settings list to resume where it stopped
    if dataset_path.exists():
        episode_folders = [
            episode_folder
            for split in DatasetSplit
            if (dataset_path / split).exists() and (dataset_path / split).is_dir()
            for episode_folder in (dataset_path / split).iterdir()
            if episode_folder.is_dir()
        ]

        for episode_folder in episode_folders:
            if not episode_folder.is_dir():
                continue

            settings_file = episode_folder / 'settings.json'
            if settings_file.exists():
                settings = episode_settings_from_file(settings_file)
                found_match = False
                for i, settings_instance in enumerate(episode_settings):
                    if settings == settings_instance:
                        found_match = True
                        del episode_settings[i]
                        break
                if not found_match:
                    raise RuntimeError(
                        (
                            'Dataset folder was not empty and resuming failed because'
                            'already generated episode did not match current config.'
                            f'Settings of conflicting episode: {settings}'
                        )
                    )

            else:
                print(f'No settings.json found in {episode_folder}. Deleting incomplete episode.')
                shutil.rmtree(episode_folder)

    return episode_settings


def gather_episode(client: carla.Client, settings: EpisodeSettings):
    world = client.load_world(settings.town)

    start_datetime = datetime.datetime.now()
    episode_path = (
        dataset_path / f'{settings.split.value}/{start_datetime.strftime("%Y_%m_%d_%H_%M_%S")}'
    )
    episode_path.mkdir(parents=True, exist_ok=True)

    time.sleep(1)
    ego_vehicle = spawn_ego_vehicle(world)

    time.sleep(0.1)
    ego_vehicle.set_autopilot(True)

    traffic_manager = client.get_trafficmanager()
    if config['dataset.ignore_traffic_lights']:
        traffic_manager.ignore_lights_percentage(ego_vehicle, 100)

    traffic_manager.auto_lane_change(ego_vehicle, False)
    traffic_manager.ignore_signs_percentage(ego_vehicle, 100)
    traffic_manager.keep_right_rule_percentage(ego_vehicle, 0)

    time.sleep(5)
    camera = create_and_attach_camera(
        world,
        ego_vehicle,
        config['camera.resolution.width'],
        config['camera.resolution.height'],
    )
    episode_state: dict[str, Any] = {'done': False, 'car_controls': []}
    camera.listen(lambda image: save_data_callback(image, ego_vehicle, episode_path, episode_state))

    start_time = time.time()
    while not episode_state['done']:
        time.sleep(0.5)

    with open(episode_path / 'car_controls.json', 'w') as file:
        file.write(json.dumps(episode_state['car_controls']))

    with open(episode_path / 'settings.json', 'w') as file:
        file.write(json.dumps(asdict(settings)))

    camera.destroy()
    ego_vehicle.destroy()


def main():
    client = carla.Client('localhost', 2000)

    episode_settings_list = create_episode_settings_list()
    for episode_settings in tqdm(episode_settings_list):
        gather_episode(client, episode_settings)


if __name__ == '__main__':
    main()
