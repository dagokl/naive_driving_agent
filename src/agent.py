import time
from queue import Queue

import carla
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from carla_utils import create_and_attach_camera, spawn_ego_vehicle
from config import config
from controller import VehiclePIDController
from model import DrivingModel

image_queue: Queue[carla.Image] = Queue(2)


def load_model(path):
    if config['model.predict.type'] == 'waypoints':
        out_size = 3 * config['model.predict.num_waypoints']
    elif config['model.predict.type'] == 'direct_controls':
        out_size = 3
    else:
        ValueError()
    model = DrivingModel(out_size)
    model.load_state_dict(torch.load(path))
    model.eval()

    weights = model.regressor[-1].weight.data
    biases = model.regressor[-1].bias.data

    print('Weights contain negative values:', (weights < 0).any())
    print('Biases contain negative values:', (biases < 0).any())

    return model


def view_image(image: carla.Image):
    image_array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
    image_array = np.reshape(image_array, (image.height, image.width, 4))
    cv2.imshow('Camera', image_array)
    cv2.waitKey(1)


def transform_from_ego_frame_to_map_coordinates(ego_trans: carla.Transform, point: np.ndarray):
    fu_c = ego_trans.get_forward_vector().make_unit_vector()
    forward_unit_vec = np.array((fu_c.x, fu_c.y, fu_c.z))
    ru_c = ego_trans.get_right_vector().make_unit_vector()
    right_unit_vec = np.array((ru_c.x, ru_c.y, ru_c.z))

    ego_l = np.array((ego_trans.location.x, ego_trans.location.y, ego_trans.location.z))
    target_l = ego_l + point[0] * forward_unit_vec + point[1] * right_unit_vec
    return target_l


def camera_callback(
    image: carla.Image,
    vehicle: carla.Actor,
    controller: VehiclePIDController,
    model: nn.Module,
    world: carla.World,
):
    raw_image = np.reshape(
        np.copy(image.raw_data),
        (config['camera.resolution.height'], config['camera.resolution.width'], 4),
    )
    image_transposed = np.transpose(raw_image[:, :, :3], (2, 0, 1))
    image_exp = np.expand_dims(image_transposed, axis=0)
    image_scaled = image_exp / 255.0
    image_tensor = torch.from_numpy(image_scaled.astype(np.float32))

    output = model.forward(image_tensor)
    output_numpy = output[0].cpu().detach().numpy()
    print(output_numpy)

    if config['model.predict.type'] == 'waypoints':
        waypoint_index_for_control = 0
        visualize_waypoints = True
        num_waypoints = config['model.predict.num_waypoints']

        target_locations = []
        for i in range(0, 3 * num_waypoints, 3):
            point = transform_from_ego_frame_to_map_coordinates(
                vehicle.get_transform(), output_numpy[i : i + 3]
            )
            location = carla.Location(*point)
            target_locations.append(location)

            if visualize_waypoints:
                color = carla.Color(
                    255 * (i == 0),
                    255 * (i == 3),
                    255 * (i >= 6),
                )
                world.debug.draw_point(location, life_time=0.05, color=color)

        control = controller.run_step(20, target_locations[waypoint_index_for_control])
    elif config['model.predict.type'] == 'direct_controls':
        ignore_brake = True
        steer, throttle, brake = output_numpy
        control = carla.VehicleControl(
            float(throttle), float(steer), float(brake) if not ignore_brake else 0.0
        )
    else:
        ValueError()

    vehicle.apply_control(control)

    # image_queue.put(image)


def main():
    client = carla.Client('localhost', 2000)
    world: carla.World = client.load_world(config['agent.town'])

    model = load_model(config['agent.model_path'])

    ego_vehicle = spawn_ego_vehicle(world)

    time.sleep(1)

    map = world.get_map()

    spectator = world.get_spectator()
    spectator.set_transform(ego_vehicle.get_transform())

    controller = VehiclePIDController(
        ego_vehicle,
        {'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0, 'dt': 1 / 20},
        {'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0, 'dt': 1 / 20},
    )

    camera = create_and_attach_camera(
        world,
        ego_vehicle,
        config['camera.resolution.width'],
        config['camera.resolution.height'],
    )
    camera.listen(lambda image: camera_callback(image, ego_vehicle, controller, model, world))
    try:
        while True:
            time.sleep(0.1)
            # latest_image = None
            # while not image_queue.empty():
            #     latest_image = image_queue.get()

            # if latest_image:
            #     view_image(latest_image)
    except KeyboardInterrupt:
        print('KeyboardInterupt')
    finally:
        camera.destroy()
        ego_vehicle.destroy()
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
