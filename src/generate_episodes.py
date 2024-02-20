import datetime
import json
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import carla
from tqdm import tqdm

from carla_utils import create_and_attach_camera, spawn_ego_vehicle
from config import config
from episode import DatasetSplit, Episode, EpisodeGenerationSettings, StateSnapshot

dataset_path = Path(config['dataset.folder_path'])


@dataclass
class EpisodeGenerationState:
    episode: Episode
    start_timestamp: float | None = None
    start_frame: int = 0
    done: bool = False


def episode_settings_from_file(file_path: Path) -> EpisodeGenerationSettings:
    with open(file_path, 'r') as file:
        json_data = json.loads(file.read())
        return EpisodeGenerationSettings(**json_data)


def save_data_callback(
    image: carla.Image, vehicle: carla.Actor, path: Path, generation_state: EpisodeGenerationState
):
    if generation_state.start_timestamp is None:
        generation_state.start_timestamp = image.timestamp
        generation_state.start_frame = image.frame

    elapsed = image.timestamp - generation_state.start_timestamp
    if elapsed > config['dataset.episode_length']:
        generation_state.done = True
        return

    frame = image.frame - generation_state.start_frame
    image_path = path / f'images/{frame:06}.png'
    control: carla.VehicleControl = vehicle.get_control()

    transform = vehicle.get_transform()
    location = transform.location
    position = (location.x, location.y, location.z)
    rotation = transform.rotation
    orientation = (rotation.pitch, rotation.yaw, rotation.roll)

    vel = vehicle.get_velocity()
    velocity = (vel.x, vel.y, vel.z)
    ang_vel = vehicle.get_angular_velocity()
    angular_velocity = (ang_vel.x, ang_vel.y, ang_vel.z)

    state_snapshot = StateSnapshot(
        frame,
        elapsed,
        Path(*image_path.parts[2:]).as_posix(),
        control.steer,
        control.throttle,
        control.brake,
        position,
        orientation,
        velocity,
        angular_velocity,
    )
    generation_state.episode.state_snapshots.append(state_snapshot)

    image.save_to_disk(image_path.as_posix())


def create_episode_generation_plan(
    delete_episodes_not_matching_config: bool = True
) -> list[EpisodeGenerationSettings]:
    episode_generation_plan = []

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
        episode_generation_plan.append(
            EpisodeGenerationSettings(town, DatasetSplit.TRAIN, config['dataset.episode_length'])
        )

    for _ in range(num_val_episodes):
        town = random.choice([town for town, count in val_town_counts.items() if count > 0])
        val_town_counts[town] -= 1
        episode_generation_plan.append(
            EpisodeGenerationSettings(town, DatasetSplit.VAL, config['dataset.episode_length'])
        )

    for _ in range(num_test_episodes):
        town = random.choice([town for town, count in test_town_counts.items() if count > 0])
        test_town_counts[town] -= 1
        episode_generation_plan.append(
            EpisodeGenerationSettings(town, DatasetSplit.TEST, config['dataset.episode_length'])
        )

    # Remove entries for already generated episodes to resume dataset generation.
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

            try:
                episode = Episode.read_from_file(episode_folder)
            except:
                print(f'Failed to load episode from {episode_folder}. Deleting incomplete episode.')
                shutil.rmtree(episode_folder)
                continue

            settings = episode.generation_settings
            found_match = False
            for i, settings_instance in enumerate(episode_generation_plan):
                if settings == settings_instance:
                    found_match = True
                    del episode_generation_plan[i]
                    break
            if not found_match:
                if delete_episodes_not_matching_config:
                    print(
                        (
                            f'settings.json found in {episode_folder} did not match current '
                            'config. Deleting episode.'
                        )
                    )
                    shutil.rmtree(episode_folder)
                else:
                    raise RuntimeError(
                        (
                            'Dataset folder was not empty and resuming failed because already '
                            'generated episode did not match current config. Settings of '
                            f'conflicting episode: {settings}'
                        )
                    )

    return episode_generation_plan


def gather_episode(client: carla.Client, settings: EpisodeGenerationSettings):
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
    episode = Episode(settings)
    generation_state = EpisodeGenerationState(episode)
    camera.listen(
        lambda image: save_data_callback(image, ego_vehicle, episode_path, generation_state)
    )

    start_time = time.time()
    while not generation_state.done:
        time.sleep(0.5)

    generation_state.episode.write_to_file(episode_path)

    # TODO: Make sure that camera and vehicle are also destroyed when gather_episode encounters and
    # exception
    camera.destroy()
    ego_vehicle.destroy()


def main():
    client = carla.Client('localhost', 2000)

    generation_plan = create_episode_generation_plan()
    for episode_settings in tqdm(generation_plan):
        gather_episode(client, episode_settings)


if __name__ == '__main__':
    main()
