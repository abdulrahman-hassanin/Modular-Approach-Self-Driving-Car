""" This module contains PID controllers to perform lateral and longitudinal control """

# Impoer modules
from collections import deque
import math
import numpy as np
import carla
from agents.tools.misc import get_speed

# PID controller class
class VehiclePIDController():
    """
    VehiclePIDController is the combination of two PID controllers
    (lateral and longitudinal) to perform the
    low level control a vehicle from client side
    """

    # Initialization
    def __init__(self, vehicle, args_lateral, args_longitudinal, offset=0, max_throttle=0.75, max_brake=0.3, max_steering=0.8):
        """
        Constructor method

        Parameters:
        - vehicle: actor to apply to local planner logic onto
        - args_lateral: dictionary of arguments to set the lateral PID controller
        - args_longitudinal: dictionary of arguments to set the longitudinal
        using the following semantics:
            K_P -- Proportional term
            K_D -- Differential term
            K_I -- Integral term
        - offset: If different than zero, the vehicle will drive displaced from the center line
        Positive values imply a right offset while negative ones mean a left one. Numbers high enough
        to cause the vehicle to drive through other lanes might break the controller
        """

        # Store parameters
        self.max_brake = max_brake
        self.max_throt = max_throttle
        self.max_steer = max_steering
        self._vehicle = vehicle

        # Get world
        self._world = self._vehicle.get_world()

        # Get past vehicle steering
        self.past_steering = self._vehicle.get_control().steer

        # Get longitudinal and lateral control 
        self._lon_controller = PIDLongitudinalController(self._vehicle, **args_longitudinal)
        self._lat_controller = PIDLateralController(self._vehicle, offset, **args_lateral)

    # Run step
    def run_step(self, target_speed, waypoint):
        """
        Execute one step of control invoking both lateral and longitudinal
        PID controllers to reach a target waypoint at a given target_speed

        Parameters:
        - target_speed: desired vehicle speed
        - waypoint: target location encoded as a waypoint
        
        Return:
        - control
        """

        # Get acceleration and current steering
        acceleration = self._lon_controller.run_step(target_speed)
        current_steering = self._lat_controller.run_step(waypoint)

        # Check for acceleration
        control = carla.VehicleControl()
        if acceleration >= 0.0:
            control.throttle = min(acceleration, self.max_throt)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)

        # Check for steering, Steering regulation: changes cannot happen abruptly, can't steer too much.
        if current_steering > self.past_steering + 0.1:
            current_steering = self.past_steering + 0.1
        elif current_steering < self.past_steering - 0.1:
            current_steering = self.past_steering - 0.1

        if current_steering >= 0:
            steering = min(self.max_steer, current_steering)
        else:
            steering = max(-self.max_steer, current_steering)

        # Store values
        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering

        # Return control values
        return control

# Longitudinal controller class 
class PIDLongitudinalController():
    """
    PIDLongitudinalController implements longitudinal control using a PID
    """

    # Initialization
    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        Constructor method

        Parameters:
        - vehicle: actor to apply to local planner logic onto
        - K_P: Proportional term
        - K_D: Differential term
        - K_I: Integral term
        - dt: time differential in seconds
        """

        # Store parameters
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    # Run step
    def run_step(self, target_speed, debug=False):
        """
        Execute one step of longitudinal control to reach a given target speed

        Parameters:
        - target_speed: target speed in Km/h
        - debug: boolean for debugging
        
        Return:
        - throttle control
        """

        # Get current speed
        current_speed = get_speed(self._vehicle)

        # Check for debug
        if debug:
            print('Current speed = {}'.format(current_speed))

        # Return throttle control
        return self._pid_control(target_speed, current_speed)

    # Get vehicle throttle and brake values
    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle/brake of the vehicle based on the PID equations

        Parameters:
        - target_speed:  target speed in Km/h
        - current_speed: current speed of the vehicle in Km/h
        
        Return:
        - throttle/brake control
        """

        # Calculate speed error
        error = target_speed - current_speed
        self._error_buffer.append(error)

        # Check for error
        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0
 
        # Return throttle/brake control
        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)

# Longitudinal lateral class 
class PIDLateralController():
    """
    PIDLateralController implements lateral control using a PID
    """

    # Initialization
    def __init__(self, vehicle, offset=0, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        Constructor method

        Parameters:
        - vehicle: actor to apply to local planner logic onto
        - offset: distance to the center line. If might cause issues if the value
                  is large enough to make the vehicle invade other lanes.
        - K_P: Proportional term
        - K_D: Differential term
        - K_I: Integral term
        - dt: time differential in seconds
        """

        # Store parameters
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._offset = offset
        self._e_buffer = deque(maxlen=10)

    # Run step
    def run_step(self, waypoint):
        """
        Execute one step of lateral control to steer
        the vehicle towards a certain waypoin

        Parameters:
        - waypoint: target waypoint
        
        Return:
        - steering control in the range [-1, 1]
          where:
            -1 maximum steering to left
            +1 maximum steering to right
        """

        # Return steering control
        return self._pid_control(waypoint, self._vehicle.get_transform())

    # Get vehicle steering angle
    def _pid_control(self, waypoint, vehicle_transform):
        """
        Estimate the steering angle of the vehicle based on the PID equations

            :param waypoint: target waypoint
            :param vehicle_transform: current transform of the vehicle
            :return: steering control in the range [-1, 1]
        """
        
        # Get the ego's location and forward vector
        ego_loc = vehicle_transform.location
        v_vec = vehicle_transform.get_forward_vector()
        v_vec = np.array([v_vec.x, v_vec.y, 0.0])

        # Get the vector vehicle-target_wp
        if self._offset != 0:
            # Displace the wp to the side
            w_tran = waypoint.transform
            r_vec = w_tran.get_right_vector()
            w_loc = w_tran.location + carla.Location(x=self._offset*r_vec.x,
                                                         y=self._offset*r_vec.y)
        else:
            w_loc = waypoint.transform.location

        w_vec = np.array([w_loc.x - ego_loc.x,
                          w_loc.y - ego_loc.y,
                          0.0])

        _dot = math.acos(np.clip(np.dot(w_vec, v_vec) /
                                 (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))
        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0

        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * _dot) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)
