"""
============================== Utils ==============================
- Graduation project 2021 | Modular-approch self driving car system
- Faculty of engineering Helwan university
===================================================================
- Description:
              A collection of general functions to be used alot
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
import math

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random

# CRALA distribution
def carla_setup():
    """Specify CARLA dist"""
    try:
        sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    except IndexError:
        pass
        
# Client setup
def client_setup(host, port, timeout, town):
    """Setup client , world and map"""
    
    # Setup
    # client     = carla.Client(host, port)
    # client.set_timeout(timeout)
    # client.get_server_version()
    # world      = client.load_world(town)
    # world.wait_for_tick(timeout)
    # world_map  = world.get_map()
    
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    world_map = world.get_map()
    # Return
    return client, world, world_map

# Source and destination setup
def source_dest_setup(world_map, s, d):
    """Setup the source and destination locations and waypoints of the route"""

    # Spawn points
    spawn_points = world_map.get_spawn_points()

    # Source and destination locations
    source      = carla.Location(spawn_points[s].location)
    destination = carla.Location(spawn_points[d].location)

    # Source and destination waypoints
    source_waypoint = world_map.get_waypoint(source)
    dest_waypoint   = world_map.get_waypoint(destination)

    return source, destination, source_waypoint, dest_waypoint

# Ego vehicle setup
def vehicle_setup(world, vehicle_transform):
    """Setup ego vehicle"""

    # Setup
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    vehicle = world.spawn_actor(vehicle_bp, vehicle_transform)
    
    # Return
    return vehicle

# Spectator setup
def spectator_setup(world, vehicle):
    """Spectator setup at ego vehicle position"""

    # Setup
    spectator = world.get_spectator()
    spectator_transform = vehicle.get_transform()
    spectator_transform.location += carla.Location(x=-5.0, y=0, z = 6.0)
    world_snapshot = world.wait_for_tick() 
    spectator.set_transform(spectator_transform)


    #col_bp = world.get_blueprint_library().find('sensor.other.collision')
    #col_transform = carla.Transform(carla.Location(x=-7, z=6))
    #ego_col = world.spawn_actor(col_bp, col_transform,attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
    #world_snapshot = world.wait_for_tick() 
    #spectator.set_transform(ego_col.get_transform())


# Spawn actors
def spawn_actors(no_of_vehicles, no_of_walkers):
    """Spawn vehicle and pedestrians to the map"""

    #os.system("C:\Windows\System32\cmd.exe /c spawn_actors.bat")
    os.putenv("VAR1", str(no_of_vehicles)) 
    os.putenv("VAR2", str(no_of_walkers)) 
    os.system("spawn_actors.bat")

# Draw route
def draw_route(route, world):
    """Draw the route"""
    length = len(route)
    for wp in route:
        t =  wp[0].transform
        begin = t.location + carla.Location(z=0.5)
        angle = math.radians(t.rotation.yaw)
        end   = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.01, color=carla.Color(0,0,0), life_time=10000)
        length = length-1
        if(length == 3):
            break
