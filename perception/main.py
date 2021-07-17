import numpy as np
import pygame
import cv2
from imutils.video import FPS
from datetime import datetime

import carla_agent
import object_detection
import depth_estimation

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

cc = carla_agent.CarlaClient()
obj_detector = object_detection.ObjectDetection()
depth_estimator = depth_estimation.DepthEstimation()

def render(display, image_left, image_right):
    if image_left is not None and image_right is not None:
        img_left = np.frombuffer(image_left.raw_data, dtype=np.dtype("uint8"))
        img_left = np.reshape(img_left, (image_left.height, image_left.width, 4))
        img_left = img_left[:, :, :3]
        
        img_right = np.frombuffer(image_right.raw_data, dtype=np.dtype("uint8"))
        img_right = np.reshape(img_right, (image_right.height, image_right.width, 4))
        img_right = img_right[:, :, :3]

        
        disp_left = depth_estimator.compute_left_disparity_map(img_left, img_right)
        # depth_map = depth_estimator.calc_depth_map(disp_left)
        point_cloud = depth_estimator.calc_point_cloud(disp_left)

        pred_bboxes = obj_detector.detect(img_left)
        array = obj_detector.draw_bbox(img_left, pred_bboxes, point_cloud, mode = 'point_cloud', show_label=False, Depth_by_bbox=False)
        #array = depth_estimator.put_distance_txt(array, depth_map, pred_bboxes, obj_detector.classes, obj_detector.allowed_classes)

        array = array[:, :, ::-1]
        surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        display.blit(surface, (0, 0))


if __name__ == '__main__':
    try:
        pygame.init()
        clock = pygame.time.Clock()
        cc.client = carla.Client('127.0.0.1', 2000)
        cc.client.set_timeout(5.0)
        cc.world = cc.client.get_world()

        cc.setup_car()
        cc.setup_left_camera()
        cc.setup_right_camera()
        cc.display = pygame.display.set_mode((cc.img_width, cc.img_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame_clock = pygame.time.Clock()
        cc.set_synchronous_mode(True)
        cc.car.set_autopilot(True)
        while True:
            fps = FPS().start()
            cc.world.tick()
            cc.capture_left = True
            cc.capture_right = True
            pygame_clock.tick_busy_loop(30)
            # cc.render(cc.display, obj_detector, depth_estimator)
            render(cc.display, cc.image_left, cc.image_right)
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