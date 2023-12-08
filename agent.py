import time

import carla
import numpy as np
import torch
import torch.nn as nn

from carla_utils import create_and_attach_camera, spawn_ego_vehicle
from config import config
from model import DrivingModel


def load_model(path):
    model = DrivingModel()
    model.load_state_dict(torch.load(path))
    model.eval()

    weights = model.regressor[-1].weight.data
    biases = model.regressor[-1].bias.data

    print('Weights contain negative values:', (weights < 0).any())
    print('Biases contain negative values:', (biases < 0).any())

    return model


def camera_callback(image: carla.Image, vehicle: carla.Actor, model: nn.Module):
    raw_image = np.reshape(
        np.copy(image.raw_data),
        (config.get('camera.resolution.height'), config.get('camera.resolution.width'), 4),
    )
    image_transposed = np.transpose(raw_image[:, :, :3], (2, 0, 1))
    image_exp = np.expand_dims(image_transposed, axis=0)
    image_scaled = image_exp / 255.0
    image_tensor = torch.from_numpy(image_scaled.astype(np.float32))

    output = model.forward(image_tensor)
    print(output)

    control = carla.VehicleControl()
    control.steer = float(output[0, 0])
    control.throttle = float(output[0, 1])
    vehicle.apply_control(control)


def main():
    client = carla.Client('localhost', 2000)
    world = client.load_world(config.get('agent.town'))

    model = load_model(config.get('agent.model_path'))

    ego_vehicle = spawn_ego_vehicle(world)

    time.sleep(1)

    spectator = world.get_spectator()
    spectator.set_transform(ego_vehicle.get_transform())

    camera = create_and_attach_camera(
        world,
        ego_vehicle,
        config.get('camera.resolution.width'),
        config.get('camera.resolution.height'),
    )

    camera.listen(lambda image: camera_callback(image, ego_vehicle, model))
    # traffic_manager = client.get_trafficmanager()
    # ego_vehicle.set_autopilot(True)
    try:
        while True:
            time.sleep(1)
            # next_action, next_action_wp = traffic_manager.get_next_action(ego_vehicle)

            # print()
            # print(f'{next_action = }')

            # distance_to_next_action = ego_vehicle.get_transform().location.distance(next_action_wp.transform.location)
            # print(f'{distance_to_next_action = }')

            # all_actions = traffic_manager.get_all_actions(ego_vehicle)
            # print(f'{all_actions = }')

            # for i, (action, wp) in enumerate(all_actions):
            #     print(i)
            #     print(f'{wp == next_action = }')
            #     print(f'{action = }')
            #     distance_from_ego = ego_vehicle.get_transform().location.distance(wp.transform.location)
            #     distance_to_next_action_wp = wp.transform.location.distance(next_action_wp.transform.location)
            #     print(f'{distance_from_ego = }')
            #     print(f'{distance_to_next_action_wp = }')

            # print()

    except KeyboardInterrupt:
        print('KeyboardInterupt')
    finally:
        camera.destroy()
        ego_vehicle.destroy()


if __name__ == '__main__':
    main()
