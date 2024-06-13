from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# TODO: Refactor the following two functions
def get_surronding_tps_and_commands(route, vehicle_position, vehicle_heading):
    last_route_element_reached = len(route) == 1
    targeting_last_route_element = len(route) == 2
    if last_route_element_reached:
        prev_tp, prev_command = route[0]
        current_tp, current_command = route[0]
        next_tp, next_command = route[0]
    elif targeting_last_route_element:
        prev_tp, prev_command = route[0]
        current_tp, current_command = route[1]
        next_tp, next_command = route[1]
    else:
        prev_tp, prev_command = route[0]
        current_tp, current_command = route[1]
        next_tp, next_command = route[2]

    prev_tp = transform_point_to_ego_frame(prev_tp, vehicle_position, vehicle_heading)
    current_tp = transform_point_to_ego_frame(current_tp, vehicle_position, vehicle_heading)
    next_tp = transform_point_to_ego_frame(next_tp, vehicle_position, vehicle_heading)


    return prev_tp, prev_command, current_tp, current_command, next_tp, next_command


def get_surronding_tps(route, vehicle_position, vehicle_heading):
    last_route_element_reached = len(route) == 1
    targeting_last_route_element = len(route) == 2
    if last_route_element_reached:
        prev_tp, prev_command = route[0]
        current_tp, current_command = route[0]
        next_tp, next_command = route[0]
    elif targeting_last_route_element:
        prev_tp, prev_command = route[0]
        current_tp, current_command = route[1]
        next_tp, next_command = route[1]
    else:
        prev_tp, prev_command = route[0]
        current_tp, current_command = route[1]
        next_tp, next_command = route[2]

    local_tps = (prev_tp, current_tp, next_tp)

    prev_tp = transform_point_to_ego_frame(prev_tp, vehicle_position, vehicle_heading)
    current_tp = transform_point_to_ego_frame(current_tp, vehicle_position, vehicle_heading)
    next_tp = transform_point_to_ego_frame(next_tp, vehicle_position, vehicle_heading)
    ego_tps = (prev_tp, current_tp, next_tp)

    return ego_tps, local_tps


def transform_point_to_ego_frame(point, vehicle_location, vehicle_heading):
    theta = -vehicle_heading + (np.pi / 2)
    ego_point = point - vehicle_location
    rot_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    return rot_matrix @ ego_point


class AgentPlotter:
    def __init__(self, show:bool=False, save:bool=False, save_path:Path=None, fig_size=(1.5*7, 1.5*9), resolution_height=1440):
        self._show = show
        self._save = save
        self._active = show or save
        self.save_path = save_path
        assert not (save and save_path is None)
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)

        if not self._active:
            return

        self.fig_size = fig_size
        self.dpi = resolution_height / fig_size[1]
        self.fig, self.axes = plt.subplots(2, 1, figsize=self.fig_size)
        self.axes[0].set_xlim(-15, 15)
        self.axes[0].set_ylim(-15, 15)
        self.axes[0].set_aspect('equal')

        self.axes[0].axhline(y=0, color='k', linestyle='-')
        self.axes[0].axvline(x=0, color='k', linestyle='-')
        self.ego_tp_scatter = self.axes[0].scatter([], [], label='tp input') 
        self.ego_pred_scatter = self.axes[0].scatter([], [], label='model output', c='red')
        self.axes[0].legend()

        self.route_scatter = self.axes[1].scatter([], [], label='route', c='blue')
        self.route_tp_scatter = self.axes[1].scatter([], [], label='tp input', c='green')
        self.route_vehicle_location_scatter = self.axes[1].scatter([], [], label='vehicle location', c='red')

        self.axes[1].set_aspect('equal')
        self.axes[1].invert_yaxis()
        self.axes[1].legend()

    def init_route(self, route):
        if not self._active:
            return

        tps = []
        x_values = []
        y_values = []
        for tp, command in route:
            tps.append(tp)
            x_values.append(tp[0])
            y_values.append(tp[1])
        self.axes[1].set_xlim(min(x_values) - 4, max(x_values) + 4)
        self.axes[1].set_ylim(max(y_values) + 4, min(y_values) - 4)
        self.route_scatter.set_offsets(tps)

    def draw_step(self, location, pred, tp_ego_frame, tp_local_frame, frame):
        if not self._active:
            return

        self.route_vehicle_location_scatter.set_offsets(location)
        self.route_tp_scatter.set_offsets(tp_local_frame)

        self.ego_tp_scatter.set_offsets(swap_axes(tp_ego_frame))
        self.ego_pred_scatter.set_offsets(swap_axes(pred))

        if self._show:
            plt.pause(0.0001)

        if self._save:
            plt.savefig((self.save_path / f'{frame:06}.png').as_posix(), dpi=self.dpi, transparent=False)


def swap_axes(points):
    return [(y, x) for x, y in points]
            
        
