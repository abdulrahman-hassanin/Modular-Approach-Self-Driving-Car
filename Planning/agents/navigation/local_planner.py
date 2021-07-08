""" This module contains a local planner to perform low-level waypoint following based on PID controllers """

# Import modules
from enum import Enum
from collections import deque
import random
import os
import sys
import glob
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
from agents.navigation.controller import VehiclePIDController
from agents.tools.misc import draw_waypoints


# Road option class
class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


# Local planner class
class LocalPlanner(object):
    """
    LocalPlanner implements the basic behavior of following a trajectory of waypoints that is generated on-the-fly.
    The low-level motion of the vehicle is computed by using two PID controllers, one is used for the lateral control
    and the other for the longitudinal control (cruise speed)
    When multiple paths are available (intersections) this local planner makes a random choice
    """

    # Minimum distance to target waypoint as a percentage (e.g. within 90% of total distance)
    MIN_DISTANCE_PERCENTAGE = 0.9

    # Initialization
    def __init__(self, vehicle, opt_dict=None):
        """
        Contructor

        Parameters:
        - vehicle: actor to apply to local planner logic onto
        - opt_dict: dictionary of arguments with the following semantics:
                    dt -- time difference between physics control in seconds. This is typically fixed from server side
                          using the arguments -benchmark -fps=F . In this case dt = 1/F
                    target_speed -- desired cruise speed in Km/h
                    sampling_radius -- search radius for next waypoints in seconds: e.g. 0.5 seconds ahead
                    lateral_control_dict -- dictionary of arguments to setup the lateral PID controller
                                    {'K_P':, 'K_D':, 'K_I':, 'dt'}
                    longitudinal_control_dict -- dictionary of arguments to setup the longitudinal PID controller
                                        {'K_P':, 'K_D':, 'K_I':, 'dt'}
        """

        # Parameters initialization
        self._vehicle = vehicle
        self._map = self._vehicle.get_world().get_map()
        self._dt = None
        self._target_speed = None
        self._sampling_radius = None
        self._min_distance = None
        self._current_waypoint = None
        self._target_road_option = None
        self._next_waypoints = None
        self.target_waypoint = None
        self._vehicle_controller = None
        self._global_plan = None

        # Queue with tuples of (waypoint, RoadOption)
        self._waypoints_queue = deque(maxlen=20000)

        self._buffer_size = 5
        self._waypoint_buffer = deque(maxlen=self._buffer_size)

        # initializing controller
        self._init_controller(opt_dict)

    # # Delete
    # def __del__(self):
    #     if self._vehicle:
    #         self._vehicle.destroy()
    #         print("Destroying ego-vehicle!")

    # # Reset
    # def reset_vehicle(self):
    #     self._vehicle = None
    #     print("Resetting ego-vehicle!")

    # initialize controller
    def _init_controller(self, opt_dict):
        """
        Controller initialization

		Parameters:
        - opt_dict: dictionary of arguments
        """

        # Default params
        self._dt = 1.0 / 20.0
        self._target_speed = 20.0  # Km/h
        self._sampling_radius = self._target_speed * 1 / 3.6  # 1 seconds horizon
        self._min_distance = self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE
        self._max_brake = 0.3
        self._max_throt = 0.75
        self._max_steer = 0.8
        args_lateral_dict = {
            'K_P': 1.95,
            'K_D': 0.2,
            'K_I': 0.07,
            'dt': self._dt}
        args_longitudinal_dict = {
            'K_P': 1.0,
            'K_D': 0,
            'K_I': 0.05,
            'dt': self._dt}
        self._offset = 0

        # Parameters overload
        if opt_dict:
            if 'dt' in opt_dict:
                self._dt = opt_dict['dt']
            if 'target_speed' in opt_dict:
                self._target_speed = opt_dict['target_speed']
            if 'sampling_radius' in opt_dict:
                self._sampling_radius = self._target_speed * \
                                        opt_dict['sampling_radius'] / 3.6
            if 'lateral_control_dict' in opt_dict:
                args_lateral_dict = opt_dict['lateral_control_dict']
            if 'longitudinal_control_dict' in opt_dict:
                args_longitudinal_dict = opt_dict['longitudinal_control_dict']
            if 'max_throttle' in opt_dict:
                self._max_throt = opt_dict['max_throttle']
            if 'max_brake' in opt_dict:
                self._max_brake = opt_dict['max_brake']
            if 'max_steering' in opt_dict:
                self._max_steer = opt_dict['max_steering']
            if 'offset' in opt_dict:
                self._offset = opt_dict['offset']

        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._vehicle_controller = VehiclePIDController(self._vehicle,
                                                        args_lateral=args_lateral_dict,
                                                        args_longitudinal=args_longitudinal_dict,
                                                        offset=self._offset,
                                                        max_throttle=self._max_throt,
                                                        max_brake=self._max_brake,
                                                        max_steering=self._max_steer)

        self._global_plan = False

        # Compute initial waypoints
        self._waypoints_queue.append((self._current_waypoint.next(self._sampling_radius)[0], RoadOption.LANEFOLLOW))

        # Road option
        self._target_road_option = RoadOption.LANEFOLLOW
        
        # Fill waypoint trajectory queue
        self._compute_next_waypoints(k=200)

    # Set speed
    def set_speed(self, speed):
        """
        Request new target speed

		Parameters:
        - speed: new target speed in Km/h
        """
        self._target_speed = speed

    # Compute next watpoint
    def _compute_next_waypoints(self, k=1):
        """
        Add new waypoints to the trajectory queue

        :param k: how many waypoints to compute
        :return:
        """
        # check we do not overflow the queue
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)

        for _ in range(k):
            last_waypoint = self._waypoints_queue[-1][0]
            next_waypoints = list(last_waypoint.next(self._sampling_radius))

            if len(next_waypoints) == 0:
                break
            elif len(next_waypoints) == 1:
                # only one option available ==> lanefollowing
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            else:
                # random choice between the possible options
                road_options_list = _retrieve_options(
                    next_waypoints, last_waypoint)
                road_option = random.choice(road_options_list)
                next_waypoint = next_waypoints[road_options_list.index(
                    road_option)]

            self._waypoints_queue.append((next_waypoint, road_option))


    # Get incoming waypoints
    def get_incoming_waypoint_and_direction(self, steps=6):
        """
        Returns direction and waypoint at a distance ahead defined by the user.

        Parameters:
        - steps: number of steps to get the incoming waypoint

        Return:
        - Roadoption
        """

        # Check for waypoint
        if len(self._waypoints_queue) > steps:
            return self._waypoints_queue[steps]

        else:
            try:
                wpt, direction = self._waypoints_queue[-1]
                return wpt, direction
            except IndexError as i:
                print(i)
                return None, RoadOption.VOID

        # Return Roadoption
        return None, RoadOption.VOID


    # Set global plan
    def set_global_plan(self, current_plan):
        """
        Resets the waypoint queue and buffer to match the new plan
        Also sets the global_plan flag to avoid creating more waypoints

		Parameters:
        - current_plan: list of (carla.Waypoint, RoadOption)
        """

        # Reset the queue
        self._waypoints_queue.clear()
        for elem in current_plan:
            self._waypoints_queue.append(elem)
        self._target_road_option = RoadOption.LANEFOLLOW

        # and the buffer
        self._waypoint_buffer.clear()
        for _ in range(self._buffer_size):
            if self._waypoints_queue:
                self._waypoint_buffer.append(
                    self._waypoints_queue.popleft())
            else:
                break

        self._global_plan = True

    # Run strp
    def run_step(self, target_speed=None, debug=False):
        """
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        Parameters:
        - debug: boolean flag to activate waypoints debugging
        
        Return:
        - control to be applied
        """
        # if target_speed is not None:
        #     self.target_speed = target_speed
        # else:
        #     self.

        # Not enough waypoints in the horizon? => add more!
        if not self._global_plan and len(self._waypoints_queue) < int(self._waypoints_queue.maxlen * 0.5):
            self._compute_next_waypoints(k=100)
        if len(self._waypoints_queue) == 0 and len(self._waypoint_buffer) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False

            return control

        # Buffering the waypoints
        if not self._waypoint_buffer:
            for _ in range(self._buffer_size):
                if self._waypoints_queue:
                    self._waypoint_buffer.append(
                        self._waypoints_queue.popleft())
                else:
                    break

        # Current vehicle waypoint
        vehicle_transform = self._vehicle.get_transform()
        self._current_waypoint = self._map.get_waypoint(vehicle_transform.location)

        # Target waypoint
        self.target_waypoint, self._target_road_option = self._waypoint_buffer[0]

        # Move using PID controllers
        control = self._vehicle_controller.run_step(self._target_speed, self.target_waypoint)

        # Purge the queue of obsolete waypoints
        max_index = -1

        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            if waypoint.transform.location.distance(vehicle_transform.location) < self._min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoint_buffer.popleft()

        if debug:
            draw_waypoints(self._vehicle.get_world(), [self.target_waypoint], self._vehicle.get_location().z + 1.0)

        # Return options
        return control

    # Done
    def done(self):
        """
        Returns whether or not the planner has finished

        Return:
        - boolean
        """
        return len(self._waypoints_queue) == 0 and len(self._waypoint_buffer) == 0

# Retrieve options
def _retrieve_options(list_waypoints, current_waypoint):
    """
    Compute the type of connection between the current active waypoint and the multiple waypoints present in
    list_waypoints. The result is encoded as a list of RoadOption enums.

	Parameters:
    - list_waypoints: list with the possible target waypoints in case of multiple options
    - current_waypoint: current active waypoint
    
    Return:
    - list of RoadOption enums representing the type of connection from the active waypoint to each
             candidate in list_waypoints
    """

    # Initialize options
    options = []

    # Loop over waypoints list
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to
        # the beggining of an intersection, therefore the
        # variation in angle is small
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = _compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    # Return options
    return options

    
# Compute connection
def _compute_connection(current_waypoint, next_waypoint, threshold=35):
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint)

	Parameters:
    - current_waypoint: active waypoint
    - next_waypoint: target waypoint
    
    Return:
    - The type of topological connection encoded as a RoadOption enum:
            - RoadOption.STRAIGHT
            - RoadOption.LEFT
            - RoadOption.RIGHT
    """

    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    diff_angle = (n - c) % 180.0
    if diff_angle < threshold or diff_angle > (180 - threshold):
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT
