import numpy as np
import pygame
import cv2
from imutils.video import FPS
from datetime import datetime

import carla_agent
import object_detection
import depth_estimation
import lane_detection
from motion_planning import MotionPlanner
from agents.navigation.behavior_agent import BehaviorAgent 
from agents.navigation.local_planner import LocalPlanner 
import utils2

import glob
import os
import sys
import time
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# Specify CARLA distribution
# utils2.carla_setup()

# Client setup
client, world, world_map = utils2.client_setup(host="localhost", port=2000, timeout=5.0, town="Town03")
 

cc = carla_agent.CarlaClient()
obj_detector = object_detection.ObjectDetection()
depth_estimator = depth_estimation.DepthEstimation()
lane_detector = lane_detection.LaneDetection()


def render(display, image_left, image_right):
    if image_left is not None and image_right is not None:
        img_left = np.frombuffer(image_left.raw_data, dtype=np.dtype("uint8"))
        img_left = np.reshape(img_left, (image_left.height, image_left.width, 4))
        img_left = img_left[:, :, :3]
        
        img_right = np.frombuffer(image_right.raw_data, dtype=np.dtype("uint8"))
        img_right = np.reshape(img_right, (image_right.height, image_right.width, 4))
        img_right = img_right[:, :, :3]

        m = 'point_cloud'
        
        if m == 'point_cloud':
            s = time.time()
            disp_left = depth_estimator.compute_disparity_PSMNet(img_left, img_right)
            cv2.imwrite('temp.png', disp_left)
            disp_left = cv2.imread('temp.png')
            disp_left = cv2.cvtColor(disp_left, cv2.COLOR_BGR2GRAY)
            cv2.imshow('disp map', disp_left)
            point_cloud = depth_estimator.calc_point_cloud(disp_left)
            points = point_cloud
            e = time.time()
            t = e - s
            fps = 1/(t)
            s = "Depth FPS : "+ str('%.2f'% fps)
            print(s)
        
        elif m == 'nearest_point':
            disp_left = depth_estimator.compute_left_disparity_map(img_left, img_right)
            depth_map = depth_estimator.calc_depth_map(disp_left)
            
            points = depth_map
            
        pred_bboxes = obj_detector.detect(img_left)
        array = obj_detector.draw_bbox(img_left, pred_bboxes, points, mode = m, show_label=False, Depth_by_bbox=False)
        vehicle_list, pedstrain_list = obj_detector.get_object_states_list(pred_bboxes, points)
        #array = depth_estimator.put_distance_txt(array, depth_map, pred_bboxes, obj_detector.classes, obj_detector.allowed_classes)

        array = array[:, :, ::-1]

        array = lane_detector.detect_draw_lane(img_left, array)

        surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        display.blit(surface, (0, 0))

        return vehicle_list, pedstrain_list 


if __name__ == '__main__':

    """Integrate system modules together"""

    ######## 1 | Environment setup ########
    
    # 1.1 | Source and destination setup
    source, destination, source_waypoint, dest_waypoint = utils2.source_dest_setup(world_map=world_map, s=30, d=70)

    # 1.2 | Ego vehicle setup
    cc.client , cc.world = client , world
    weather = carla.WeatherParameters( cloudiness=00.0, precipitation=00.0, sun_altitude_angle=0.0) 
    cc.world.set_weather(weather)
    vehicle = cc.setup_car(carla.Transform(source, carla.Rotation(yaw=90)))

    # 1.3 | Spawn some vehicles and pedestrians
    utils2.spawn_actors(no_of_vehicles=150, no_of_walkers=5)

    # 1.4 | Spectator setup
    utils2.spectator_setup(world=world, vehicle=vehicle)

    #######################################

    ###### 2 | Motion planning setup ######

    # 2.1 | Mothion planner setup
    planner = MotionPlanner(vehicle, world)

    # 2.4 | Behavior planner setup
    planner.setup_behavior_planner()

    # 2.2 | Create route (path) between source and destination
    planner.create_route(source=source_waypoint, destination=dest_waypoint)

    # 2.3 | Draw the route
    #planner.draw_route()

    # 2.5 | Setup global plan
    planner.setup_plan()
    
    try:
        pygame.init()
        clock = pygame.time.Clock()
        #cc.client = carla.Client('127.0.0.1', 2000)
        #cc.client.set_timeout(5.0)
        #cc.world = cc.client.get_world()

        #cc.setup_car()
        cc.setup_left_camera()
        cc.setup_right_camera()
        cc.display = pygame.display.set_mode((cc.img_width, cc.img_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame_clock = pygame.time.Clock()
        cc.set_synchronous_mode(True)
        cc.car.set_autopilot(False)
        while True:
            fps = FPS().start()
            cc.world.tick()
            cc.capture_left = True
            cc.capture_right = True
            pygame_clock.tick_busy_loop(30)
            # cc.render(cc.display, obj_detector, depth_estimator)
            
            vehicle_list, pedstrain_list  = render(cc.display, cc.image_left, cc.image_right)
            planner.run(vehicle_list, pedstrain_list)

            pygame.display.flip()
            pygame.event.pump()
            cv2.waitKey(1)
            fps.stop()
            print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    
    except Exception as e:
        print(e)
        
    finally:
        cc.set_synchronous_mode(False)
        cc.left_camera.destroy()
        cc.right_camera.destroy()
        cc.car.destroy()
        pygame.quit()
        cv2.destroyAllWindows()