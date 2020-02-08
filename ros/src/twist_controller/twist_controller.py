import rospy
from yaw_controller import YawController
from lowpass import LowPassFilter
from pid import PID

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit, 
                 wheel_radius, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle):
        # TODO: Implement
        self.yawcontroller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

        kp = .3
        ki = .1
        kd = 0.
        mn = 0
        mx = 0.2
        self.throttle_controller = PID(kp, ki, kd, mn, mx)

        tau = .5
        ts = .02
        self.vel_lpf = LowPassFilter(tau, ts)

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.last_time = rospy.get_time()

    def control(self, dbw_enabled, linear_velocity, angular_velocity, current_velocity):
        if not dbw_enabled:
          self.throttle_controller.reset()
          return 0., 0., 0.

        current_velocity = self.vel_lpf.filt(current_velocity)
        steer = self.yawcontroller.get_steering(linear_velocity, angular_velocity, current_velocity)
        velocity_error = linear_velocity - current_velocity
        self.last_velocity = current_velocity

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time
 
        throttle = self.throttle_controller.step(velocity_error, sample_time)
        brake = 0

        if linear_velocity == 0 and current_velocity < 0.1:
          throttle = 0
          brake = 400 
        elif throttle < .1 and velocity_error < 0:
          throttle = 0
          decel = max(velocity_error, self.decel_limit)
          brake = abs(decel) * self.vehicle_mass * self.wheel_radius

        return throttle, brake, steer 
