import glob
import os
import sys
import numpy as np

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random
import weakref

VIEW_WIDTH = 1920 // 2
VIEW_HEIGHT = 1080 // 2
VIEW_FOV = 90

class CarlaClient(object):    
    def __init__(self):
        self.client = None
        self.world = None
        self.left_camera = None
        self.right_camera = None
        self.car = None
        self.image_left = None
        self.image_right = None
        self.capture_left = True
        self.capture_right = True
        self.img_width = 1920//2
        self.img_height = 1080//2
        self.view_fov = 90
        self.focal_length = self.img_width / (2.0 * np.tan(self.view_fov * np.pi / 360.0))

        # Define K Projection matrix
        # k = [[Fx,  0, IMG_hight/2],
        #      [ 0, Fy, IMG_width/2],
        #      [ 0,  0,           1]]
        self.K = None
        self.object_detection = None
        self.depth_estimation = None
        
    def camera_bp(self):
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        return camera_bp
    
    def set_synchronous_mode(self, synchronous_mode):
        settings = self.world.get_settings()
        # settings = self.world.WorldSettings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)
        
    def setup_car(self):
        car_bp = self.world.get_blueprint_library().filter('model3')[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.spawn_actor(car_bp, location)
        
    def setup_left_camera(self):
        camera_transform = carla.Transform(carla.Location(x=2, y=-0.2, z=1.4))
        self.left_camera = self.world.spawn_actor(self.camera_bp(), camera_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.left_camera.listen(lambda image: weak_self().set_image_left(weak_self, image))

    def setup_right_camera(self):
        camera_transform = carla.Transform(carla.Location(x=2, y=0.2, z=1.4))
        self.right_camera = self.world.spawn_actor(self.camera_bp(), camera_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.right_camera.listen(lambda image: weak_self().set_image_right(weak_self, image))
         
    @staticmethod
    def set_image_left(weak_self, img):
        self = weak_self()
        if self.capture_left:
            self.image_left = img
            self.capture_left = False
            
    @staticmethod
    def set_image_right(weak_self, img):
        self = weak_self()
        if self.capture_right:
            self.image_right = img
            self.capture_right = False
