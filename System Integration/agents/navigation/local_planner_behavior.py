""" This module contains a local planner to perform
low-level waypoint following based on PID controllers. """

# Modules
from collections import deque
from enum import Enum
import carla
from agents.navigation.controller import VehiclePIDController
from agents.tools.misc import distance_vehicle, draw_waypoints

# Road option class
class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations
    when moving from a segment of lane to other.
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
    LocalPlanner implements the basic behavior of following a trajectory
    of waypoints that is generated on-the-fly.
    The low-level motion of the vehicle is computed by using two PID controllers,
    one is used for the lateral control
    and the other for the longitudinal control (cruise speed).

    When multiple paths are available (intersections)
    this local planner makes a random choice.
    """

    # FPS used for dt
    FPS = 20

    # Constructor
    def __init__(self, agent):
        """
        Parameters:
        - agent: agent that regulates the vehicle
        - vehicle: actor to apply to local planner logic onto
        """

        # Initialization
        self._vehicle = agent.vehicle
        self._map = agent.vehicle.get_world().get_map()
        self.route = None
        self._target_speed = None
        self.sampling_radius = None
        self._min_distance = None
        self._current_waypoint = None
        self.target_road_option = None
        self._next_waypoints = None
        self.target_waypoint = None
        self._vehicle_controller = None
        self._global_plan = None
        self._pid_controller = None
        self.waypoints_queue = deque(maxlen=20000)  # queue with tuples of (waypoint, RoadOption)
        self._buffer_size = 5
        self._waypoint_buffer = deque(maxlen=self._buffer_size)

        # Initializing controller
        self._init_controller()  

    # Initialize controller
    def _init_controller(self):
        """
        Controller initialization.

        dt -- time difference between physics control in seconds.
        This is can be fixed from server side
        using the arguments -benchmark -fps=F, since dt = 1/F

        target_speed -- desired cruise speed in km/h

        min_distance -- minimum distance to remove waypoint from queue

        lateral_dict -- dictionary of arguments to setup the lateral PID controller
                            {'K_P':, 'K_D':, 'K_I':, 'dt'}

        longitudinal_dict -- dictionary of arguments to setup the longitudinal PID controller
                            {'K_P':, 'K_D':, 'K_I':, 'dt'}
        """

        # Default parameters
        self.args_lat_hw_dict = {
            'K_P': 0.75,
            'K_D': 0.02,
            'K_I': 0.4,
            'dt': 1.0 / self.FPS}
        self.args_lat_city_dict = {
            'K_P': 0.58,
            'K_D': 0.02,
            'K_I': 0.5,
            'dt': 1.0 / self.FPS}
        self.args_long_hw_dict = {
            'K_P': 0.37,
            'K_D': 0.024,
            'K_I': 0.032,
            'dt': 1.0 / self.FPS}
        self.args_long_city_dict = {
            'K_P': 0.15,
            'K_D': 0.05,
            'K_I': 0.07,
            'dt': 1.0 / self.FPS}

        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())

        self._global_plan = False

        self._target_speed = self._vehicle.get_speed_limit()

        self._min_distance = 3

    # Set speed
    def set_speed(self, speed):
        """
        Request new target speed.

        Parameters:
        - speed: new target speed in km/h
        """
        self._target_speed = speed

    # Set global plan
    def set_global_plan(self, current_plan, clean=False):
        """
        Sets new global plan.

        Parameters:
        - current_plan: list of waypoints in the actual plan
        """

        # Append waypoints queue
        for elem in current_plan:
            self.waypoints_queue.append(elem)

        # Clear waypoint queue
        if clean:
            self._waypoint_buffer.clear()
            for _ in range(self._buffer_size):
                if self.waypoints_queue:
                    self._waypoint_buffer.append(self.waypoints_queue.popleft())
                else:
                    break

        self._global_plan = True
        self.route = current_plan

    # Get incoming waypoint and direction
    def get_incoming_waypoint_and_direction(self, steps=3):
        """
        Returns direction and waypoint at a distance ahead defined by the user.

        Parameters:
        - steps: number of steps to get the incoming waypoint.
        """
        if len(self.waypoints_queue) > steps:
            #print("Get Incoming from steps")
            return self.waypoints_queue[steps]

        else:
            try:
                wpt, direction = self.waypoints_queue[-1]
                return wpt, direction
            except IndexError as i:
                #print(i)
                return None, RoadOption.VOID
        return None, RoadOption.VOID

    # Run step
    def run_step(self, target_speed=None, debug=False):
        """
        Execute one step of local planning which involves
        running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        Parameters:
        - target_speed: desired speed
        - debug: boolean flag to activate waypoints debugging
        
        Return:
        - control
        """

        # Set target speed
        if target_speed is not None:
            self._target_speed = target_speed
        else:
            self._target_speed = self._vehicle.get_speed_limit()

        # At end of route, Stop
        if len(self.waypoints_queue) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False
            return control

        # Buffering the waypoints
        if not self._waypoint_buffer:
            for i in range(self._buffer_size):
                if self.waypoints_queue:
                    self._waypoint_buffer.append(self.waypoints_queue.popleft())
                else:
                    break

        # Current vehicle waypoint
        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())

        # Target waypoint
        self.target_waypoint, self.target_road_option = self._waypoint_buffer[0]

        if target_speed > 50:
            args_lat = self.args_lat_hw_dict
            args_long = self.args_long_hw_dict
        else:
            args_lat = self.args_lat_city_dict
            args_long = self.args_long_city_dict

        # Set PID controller
        self._pid_controller = VehiclePIDController(self._vehicle,
                                                    args_lateral=args_lat,
                                                    args_longitudinal=args_long)

        control = self._pid_controller.run_step(self._target_speed, self.target_waypoint)

        # Purge the queue of obsolete waypoints
        vehicle_transform = self._vehicle.get_transform()
        max_index = -1

        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            if distance_vehicle(
                    waypoint, vehicle_transform) < self._min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoint_buffer.popleft()

        if debug:
            draw_waypoints(self._vehicle.get_world(),
                           [self.target_waypoint], 1.0)
        return control
