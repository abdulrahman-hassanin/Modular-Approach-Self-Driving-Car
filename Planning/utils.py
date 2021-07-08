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
try:
    sys.path.append(glob.glob('./carla/dist/carla-*%d.%d-%s.egg' % (
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
        sys.path.append(glob.glob('./carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    except IndexError:
        pass
    import carla
        
# Client setup
def client_setup(host, port, timeout, town):
    """Setup client , world and map"""

    # Setup
    client     = carla.Client(host, port)
    client.set_timeout(timeout)
    world      = client.load_world(town)
    world_map  = world.get_map()

    # Return
    return client, world, world_map

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
    world_snapshot = world.wait_for_tick() 
    spectator.set_transform(vehicle.get_transform())

# Spawn actors
def spawn_actors(no_of_vehicles, no_of_walkers):
    """Spawn vehicle and pedestrians to the map"""

    #os.system("C:\Windows\System32\cmd.exe /c spawn_actors.bat")
    os.putenv("VAR1", str(no_of_vehicles)) 
    os.putenv("VAR2", str(no_of_walkers)) 
    os.system("spawn_actors.bat")
