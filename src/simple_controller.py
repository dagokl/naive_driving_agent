from collections import deque

import carla
import numpy as np


class VehiclePIDController:
    def __init__(
        self,
        lateral_k_p=1.0,
        lateral_k_i=0.0,
        lateral_k_d=0.0,
        lateral_buffer_size=10,
        longitudinal_k_p=1.0,
        longitudinal_k_i=0.0,
        longitudinal_k_d=0.0,
        longitudinal_buffer_size=10,
        dt=0.05,
    ):
        self.lateral_k_p = lateral_k_p
        self.lateral_k_i = lateral_k_i
        self.lateral_k_d = lateral_k_d
        self.longitudinal_k_p = longitudinal_k_p
        self.longitudinal_k_i = longitudinal_k_i
        self.longitudinal_k_d = longitudinal_k_d
        self.dt = dt

        self.angle_error_buffer = deque([0.0] * lateral_buffer_size, lateral_buffer_size)
        self.speed_error_buffer = deque([0.0] * longitudinal_buffer_size, longitudinal_buffer_size)

    def run_step(self, relative_target_vec, target_speed, current_speed):
        control = carla.VehicleControl()
        control.steer = self.lateral_control(relative_target_vec)
        control.throttle, control.brake = self.longitudinal_control(target_speed, current_speed)
        return control

    def lateral_control(self, relative_target_vec):
        target_forward, target_right = relative_target_vec
        vehicle_angle_error = np.arctan2(target_right, target_forward)

        self.angle_error_buffer.append(vehicle_angle_error)
        i_error = sum(self.angle_error_buffer) * self.dt
        d_error = (self.angle_error_buffer[-1] - self.angle_error_buffer[-2]) / self.dt

        return np.clip(
            self.lateral_k_p * vehicle_angle_error
            + self.lateral_k_i * i_error
            + self.lateral_k_d * d_error,
            -1.0,
            1.0,
        )

    def longitudinal_control(self, target_speed, current_speed):
        speed_error = target_speed - current_speed
        self.speed_error_buffer.append(speed_error)

        i_error = sum(self.speed_error_buffer) * self.dt
        d_error = (self.speed_error_buffer[-1] - self.speed_error_buffer[-2]) / self.dt

        control_signal = np.clip(
            self.longitudinal_k_p * speed_error
            + self.longitudinal_k_i * i_error
            + self.longitudinal_k_d * d_error,
            -1.0,
            1.0,
        )

        if control_signal > 0:
            throttle = control_signal
            brake = 0.0
        else:
            throttle = 0.0
            brake = abs(control_signal)

        return throttle, brake
