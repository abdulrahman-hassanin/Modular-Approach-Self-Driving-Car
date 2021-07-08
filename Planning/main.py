"""
===================================================================
- Graduation project 2021 | Modular-approch self driving car system
- Faculty of engineering Helwan university
===================================================================
- Description:
			  Integrate all modules together to acheive full self
			  driving car system
===================================================================
- Authors:
          1. Abdulrahman Hasanin
          2. Assem Khaled
          3. Abdullah Nasser
          4. Omar Elsherif
===================================================================
"""

# Import modules
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
import utils
from motion_planning import GlobalPlanning
from motion_planning import LocalPlanning
#from motion_planning import BehaviorPlanning  
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.local_planner import LocalPlanner

# Specify CARLA distribution
utils.carla_setup()

# Client setup
client, world, world_map = utils.client_setup("localhost", 2000, 10, "Town01")

# Spawn points
spawn_points = world_map.get_spawn_points()

# Source and destination locations
source      = carla.Location(spawn_points[20].location)
destination = carla.Location(spawn_points[30].location)

# Source and destination waypoints
source_waypoint = world_map.get_waypoint(source)
dest_waypoint   = world_map.get_waypoint(destination)

# Main
def main():

    # Ego vehicle setup
    vehicle = utils.vehicle_setup(world, spawn_points[20])

    # Spectator setup at ego vehicle position
    utils.spectator_setup(world, vehicle)

    # Spawn some vehicles and pedestrians
    utils.spawn_actors(50, 50)

    # # Global planner setup
    GP = GlobalPlanning()
 
    # # Planner setup
    planner = LocalPlanning()

    # # Get PID parameters
    PID_params = planner.setup_PID(20)

    ######### TEST ########

    BA = BehaviorAgent(vehicle, PID_params)

    route = BA._trace_route(source_waypoint, dest_waypoint)
    #route = GP.create_route(source_waypoint, dest_waypoint)
    GP.draw_route(route)

       


    ########################

    # Apply control
    while True:
        #vehicle.apply_control(LP.run_step())
        control = BA.run_step()
        #print(f"Control : {control}")
        vehicle.apply_control(control)


# Run
if __name__ == "__main__":
     main()

