from yaw_controller import YawController
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle):
        # TODO: Implement
        self.yawcontroller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

        tau = .5
        ts = .02
        self.vel_lpf = LowPassFilter(tau, ts)

    def control(self, dbw_enabled, linear_velocity, angular_velocity, current_velocity):
        if not dbw_enabled:
          return 0., 0., 0.

        current_velocity = self.vel_lpf.filt(current_velocity)

        steer = self.yawcontroller.get_steering(linear_velocity, angular_velocity, current_velocity)
        return 1., 0., steer 
