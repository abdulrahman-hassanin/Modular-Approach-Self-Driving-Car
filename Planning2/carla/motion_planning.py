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
from agents.tools.misc import draw_waypoints
from agents.navigation.behavior_agent import BehaviorAgent

# Motion planner class
class MotionPlanner:
    """
    Apply motion planner to create best route from source to destiation,
    and navigate the ego vehicle through this route using PID controller 
    """

    # Constructor
    def __init__(self, vehicle, world):
        
        # Initialization
        self.vehicle = vehicle
        self.world = world
        self.route = None
        self.behavior_planner = None
        
    # Create the global route
    def create_route(self, source, destination):
        """Create route from source to destination"""

        # Create the route using the global planner inside behavior agent
        route = BehaviorAgent(self.vehicle)._trace_route(source, destination)
        self.route = route
        
    # Draw route
    def draw_route(self):
        """Draw the route"""
        utils.draw_route(self.route, self.world)
            
    # Setup behavior agent
    def setup_behavior_planner(self):
        """ Setup bahavior agent that contain local planner """

        # Setup planner
        BA = BehaviorAgent(self.vehicle)
        BA._local_planner.set_global_plan(self.route)
        self.behavior_planner = BA

    # Run
    def run(self):
        self.behavior_planner.update_information()
        control = self.behavior_planner.run_step(debug=True)
        #self.behavior_planner.update_information()     
        self.vehicle.apply_control(control)

