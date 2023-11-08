import random

import carla
from strenum import StrEnum


class CameraBlueprint(StrEnum):
    RGB = 'sensor.camera.rgb'
    SEMANTIC_SEGMENTATION = 'sensor.camera.semantic_segmentation'


def create_and_attach_camera(
    world: carla.World,
    actor: carla.Actor,
    image_size_x: int,
    image_size_y: int,
    camera_type: CameraBlueprint = CameraBlueprint.RGB,
) -> carla.Actor:
    camera_transform = carla.Transform(carla.Location(x=1.0, z=1.5))
    camera_blueprint: carla.ActorBlueprint = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_blueprint.set_attribute('image_size_x', f'{image_size_x}')
    camera_blueprint.set_attribute('image_size_y', f'{image_size_y}')
    camera_blueprint.set_attribute('image_size_y', f'{image_size_y}')
    camera = world.spawn_actor(camera_blueprint, camera_transform, attach_to=actor)
    return camera


def spawn_ego_vehicle(world: carla.World) -> carla.Actor:
    vehicle_blueprint = world.get_blueprint_library().find('vehicle.lincoln.mkz_2020')
    vehicle_blueprint.set_attribute('role_name', 'hero')
    spawn_points = world.get_map().get_spawn_points()
    ego_vehicle = world.spawn_actor(vehicle_blueprint, random.choice(spawn_points))
    return ego_vehicle
