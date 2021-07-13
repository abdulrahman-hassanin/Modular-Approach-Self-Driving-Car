"""
============================= Main ================================
- Graduation project 2021 | Modular-approch self driving car system
- Faculty of engineering Helwan university
===================================================================
- Description:
              Integrate all modules together to acheive modular
              approch self driving car system
===================================================================
- Authors:
          1. Abdulrahman Hasanin
          2. Assem Khaled
          3. Abdullah Nasser
          4. Omar Elsherif
===================================================================
"""

# Import modules
import carla
import utils
import time
from motion_planning import MotionPlanner
from agents.navigation.behavior_agent import BehaviorAgent 
from agents.navigation.local_planner import LocalPlanner 

# Specify CARLA distribution
utils.carla_setup()

# Client setup
client, world, world_map = utils.client_setup(host="localhost", port=2000, timeout=5.0, town="Town02")

# Main
def main():
    """Integrate system modules together"""

    ######## 1 | Environment setup ########
    
    # 1.1 | Source and destination setup
    source, destination, source_waypoint, dest_waypoint = utils.source_dest_setup(world_map=world_map, s=20, d=50)

    # 1.2 | Ego vehicle setup
    vehicle = utils.vehicle_setup(world=world, vehicle_transform=carla.Transform(source, carla.Rotation(yaw=0)))

    # 1.3 | Spawn some vehicles and pedestrians
    utils.spawn_actors(no_of_vehicles=50, no_of_walkers=50)

    utils.spectator_setup(world=world, vehicle=vehicle)

    #######################################

    ###### 2 | Motion planning setup ######

    # 2.1 | Mothion planner setup
    planner = MotionPlanner(vehicle, world)

    # 2.2 | Create route (path) between source and destination
    planner.create_route(source=source_waypoint, destination=dest_waypoint)

    # 2.3 | Draw the route
    planner.draw_route()

    # 2.4 | Behavior planner setup
    planner.setup_behavior_planner()

    # 2.5 | Apply control
    while True:
    	
    	#print(f"WP : {planner.route[0]}")
    	#print(f"Length = {len(planner.behavior_planner._local_planner.waypoints_queue)}")
    	# Apply control signal
    	planner.run()

    	# Set spectator at vehicle location while moving
    	#utils.spectator_setup(world=world, vehicle=vehicle)

    #######################################

    

# Run
if __name__ == "__main__":
     main()


