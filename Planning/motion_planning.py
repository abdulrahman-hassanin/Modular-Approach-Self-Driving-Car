"""
========================= Planning module =========================
- Graduation project 2021 | Modular-approch self driving car system
- Faculty of engineering Helwan university
===================================================================
- Description:
              Apply planning modules to navigate vehicle road
===================================================================
- Authors:
          1. Abdulrahman Hasanin
          2. Assem Khaled
          3. Abdullah Nasser
          4. Omar Elsherif
===================================================================
"""

# Import modules
import math
import carla
import utils
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.local_planner import LocalPlanner
from agents.navigation.behavior_agent import BehaviorAgent

# Client setup
client, world, world_map = utils.client_setup("localhost", 2000, 20, "Town01")

# Glabal planner setup
sampling_resolution = 2
dao = GlobalRoutePlannerDAO(world_map, sampling_resolution)
grp = GlobalRoutePlanner(dao)
grp.setup()

# Global planning class
class GlobalPlanning:
    """
    Apply global planner to acheive best route from source to destiation
    """

    # Initialization
    def __init__(self):
        """Constructor"""
        pass

    # Create the global route
    def create_route(self, source, destination):
        """Create route from source to destination"""

        # Store paramters
        self.source      = source
        self.destination = destination

        # Create the route
        route = grp.trace_route(source, destination)

        # Return the route
        return route
        
    def draw_route(self, route):
        """Draw the route"""
        for wp in route:
            t =  wp[0].transform
            begin = t.location + carla.Location(z=0.5)
            angle = math.radians(t.rotation.yaw)
            end   = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
            world.debug.draw_arrow(begin, end, arrow_size=0.1, color=carla.Color(255,0,0), life_time=1000)

# Local planning class
class LocalPlanning():
    """
    Apply local planner within the route to follow a trajectory of waypoints,
    The low-level motion of the vehicle is computed by using two PID controllers to achieve the destination
    """

    # Initialization (Set a local planner within the route)
    def __init__(self):
        """Constructor"""
        pass

    # Setup PID
    def setup_PID(self, speed):
        """Setup PID controller parameters for local planner"""

        # Set PID parameters
        PID_parameters = {'dt':0.05,
                          'target_speed':speed,
                          'sampling_radius':0.5,
                          'lateral_control_dict': {'K_P':1.95, 'K_D':0.2, 'K_I':0.07, 'dt':0.05},
                          'longitudinal_control_dict': {'K_P':1.0, 'K_D':0, 'K_I':0.05, 'dt':0.05}
                        }

        # Return PID parameters
        return PID_parameters

    # Setup local planner
    def setup_LP(self, vehicle, PID_params):

        # Initialize parameters
        self.vehicle = vehicle
        self.PID_params = PID_params

        # Setup planner
        LP = LocalPlanner(vehicle, PID_params)

        return LP

# Behavior planning class
class BehaviorPlanning():
    """
    Apply behavior planner within the route that navigates scenes to reach a given
    target destination
    """

    # Initialization 
    def __init__(self, vehicle, source, destination):
        """Constructor"""

        # Initialize parameters
        self.vehicle     = vehicle
        self.source      = source
        self.destination = destination
 
        # Create behavior agent
        BA = BehaviorAgent(vehicle)
        BA.set_destination(source, destination)
        BA.run_step()