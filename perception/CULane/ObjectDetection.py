import cv2
import json
import torch
import agent
import numpy as np
from copy import deepcopy
from data_loader import Generator
import time
from parameters import Parameters
import util
import os
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from patsy import cr
import csaps
import random
import weakref
import pygame
from utils import pre_processing
from imutils.video import FPS
import colorsys

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

from tensorflow.python.saved_model import tag_constants
from PIL import Image
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

saved_model_loaded = tf.saved_model.load('../checkpoints/yolov4-416', tags=[tag_constants.SERVING])
YOLO = saved_model_loaded.signatures['serving_default']



VIEW_WIDTH = 1920 // 2
VIEW_HEIGHT = 1080 // 2
VIEW_FOV = 90

#######Lane detection######
p = Parameters()

###############################################################
##
## Training
## 
###############################################################

print('Testing')
    
#########################################################################
## Get dataset
#########################################################################
print("Get dataset")
loader = Generator()

##############################
## Get agent and model
##############################
print('Get agent')
if p.model_path == "":
    lane_agent = agent.Agent()
else:
    lane_agent = agent.Agent()
    lane_agent.load_weights(296, "tensor(1.6947)")
	
##############################
## Check GPU
##############################
print('Setup GPU mode')
if torch.cuda.is_available():
    lane_agent.cuda()

##############################
## testing
##############################
print('Testing loop')
lane_agent.evaluate_mode()

class CarlaClient():    
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
        self.classes = self.read_class_names("../classes/coco.names") 
        self.allowed_classes = list(self.read_class_names("../classes/coco.names").values())

        # Define K Projection matrix
        # k = [[Fx,  0, IMG_hight/2],
        #      [ 0, Fy, IMG_width/2],
        #      [ 0,  0,           1]]
        self.K = None
        
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

    def read_class_names(self, class_file_name):
        names = {}
        with open(class_file_name, 'r') as data:
            for ID, name in enumerate(data):
                names[ID] = name.strip('\n')
        return names

    def build_projection_matrix(self):
        self.K = np.identity(3)
        self.K[0, 0] = self.focal_length
        self.K[1, 1] = self.focal_length
        self.K[0, 2] = self.img_height / 2.0
        self.K[1, 2] = self.img_width  / 2.0

    def compute_left_disparity_map(self, img_left, img_right):
                # Parameters
        num_disparities = 6*16
        block_size = 11
        
        min_disparity = 0
        window_size = 6
        
        # Stereo SGBM matcher
        left_matcher_SGBM = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 3 * window_size ** 2,
            P2=32 * 3 * window_size ** 2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # Compute the left disparity map
        disp_left = left_matcher_SGBM.compute(img_left, img_right).astype(np.float32)/16
        
        return disp_left

    def calc_depth_map(self, disp_left):
        f = self.focal_length
        b = 0.4     # Baseline

        # Replace all instances of 0 and -1 disparity with a small minimum value (to avoid div by 0 or negatives)
        disp_left[disp_left == 0] = 0.1
        disp_left[disp_left == -1] = 0.1

        # Initialize the depth map to match the size of the disparity map
        depth_map = np.ones(disp_left.shape, np.single)

        # Calculate the depths 
        depth_map[:] = f * b / disp_left[:]
        
        return depth_map

    def object_detection(self, img):
        image_data = cv2.resize(img, (416, 416))
        image_data = image_data / 255.

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

        batch_data = tf.constant(images_data)
        with tf.device("/GPU:0"):
            tf.debugging.set_log_device_placement(True)
            pred_bbox = YOLO(batch_data)

        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        iou =  0.45
        score = 0.25
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou,
            score_threshold=score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

        return pred_bbox

    def calculate_nearest_point(self, depth_map, c1, c2):
        x, y = c1
        x2, y2 = c2
        x = int(x)
        y = int(y)
        x2 = int(x2)
        y2 = int(y2)

        obstacle_depth = depth_map[y:y2, x:x2]
        closest_point_depth = obstacle_depth.min()

        return closest_point_depth

    def draw_bbox(self, image, bboxes, depth_map, show_label=True):
        img_traffic = image # Used in raffic lights classification
        colors_dict = {'Green':0, 'Yellow':0, 'Red':0, 'Unknown color':10}
        
        image = np.float32(image)
        num_classes = len(self.classes)
        image_h, image_w, _ = image.shape
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        out_boxes, out_scores, out_classes, num_boxes = bboxes
        for i in range(num_boxes[0]):
            if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
            coor = out_boxes[0][i]
            coor[0] = int(coor[0] * image_h)
            coor[2] = int(coor[2] * image_h)
            coor[1] = int(coor[1] * image_w)
            coor[3] = int(coor[3] * image_w)

            fontScale = 0.5
            score = out_scores[0][i]
            class_ind = int(out_classes[0][i])
            class_name = self.classes[class_ind]

            # check if class is in allowed classes
            if class_name not in self.allowed_classes:
                continue
            else:
                bbox_color = colors[class_ind]
                bbox_thick = int(0.6 * (image_h + image_w) / 600)
                c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
                cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
                
                depth = self.calculate_nearest_point(depth_map, c1, c2)
                cv2.putText(image, str("{:.2f}".format(depth))+' m', (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale, (0, 255, 0), bbox_thick , lineType=cv2.LINE_AA)


                ## Traffic lights classification
                
                if self.classes[class_ind] == 'traffic light':
                    
                    pic = img_traffic[int(coor[0]):int(coor[2]), int(coor[1]):int(coor[3]), :] # crop the traffic light
                    
                    hsv = cv2.cvtColor(pic, cv2.COLOR_RGB2HSV) # Convert to HSV
                    
                    colors_dict['green'] = cv2.inRange(hsv,(60,0,235), (80,255,255)).sum()
                    colors_dict['yellow'] = cv2.inRange(hsv,(91,0,233), (95,255,255)).sum()
                    colors_dict['red'] = cv2.inRange(hsv, (106,0,245), (120,255,255)).sum()
                    
                    traffic_color = max(colors_dict, key=colors_dict.get) # save the most appeared color in the range of hsv
                    
                if show_label:
                    if self.classes[class_ind] == 'traffic light':
                        bbox_mess = '%s, %s: %.2f' % (str(traffic_color), self.classes[class_ind], score) # add traffic light color to the label
                    else:
                        bbox_mess = '%s: %.2f' % (self.classes[class_ind], score)
                        
                    t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                    c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
                    
                    cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled

                    cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
                
                elif self.classes[class_ind] == 'traffic light':
                    bbox_mess = '%s' % (str(traffic_color)) # add traffic light color to the label
                    
                    t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                    c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
                    
                    cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled

                    cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
                    
        return image


    ## Lane detection functions
    
    def test(self, lane_agent, test_images, thresh = p.threshold_point, index= -1):

        result = lane_agent.predict_lanes_test(test_images)
        torch.cuda.synchronize()
        confidences, offsets, instances = result[index]
        
        num_batch = len(test_images)

        out_x = []
        out_y = []
        out_images = []
        
        for i in range(num_batch):
            # test on test data set
            image = deepcopy(test_images[i])
            image =  np.rollaxis(image, axis=2, start=0)
            image =  np.rollaxis(image, axis=2, start=0)*255.0
            image = image.astype(np.uint8).copy()

            confidence = confidences[i].view(p.grid_y, p.grid_x).cpu().data.numpy()

            offset = offsets[i].cpu().data.numpy()
            offset = np.rollaxis(offset, axis=2, start=0)
            offset = np.rollaxis(offset, axis=2, start=0)
            
            instance = instances[i].cpu().data.numpy()
            instance = np.rollaxis(instance, axis=2, start=0)
            instance = np.rollaxis(instance, axis=2, start=0)

            # generate point and cluster
            raw_x, raw_y = self.generate_result(confidence, offset, instance, thresh)

            # eliminate fewer points
            in_x, in_y = self.eliminate_fewer_points(raw_x, raw_y)
                    
            # sort points along y 
            in_x, in_y = util.sort_along_y(in_x, in_y)  
 
            image_zeros = np.zeros(image.shape, np.uint8)
            result_image = util.draw_points(in_x, in_y, deepcopy(image_zeros))
            result_image = cv2.resize(result_image, (VIEW_WIDTH, VIEW_HEIGHT))
            

            out_x.append(in_x)
            out_y.append(in_y)
            out_images.append(result_image)

        return out_x, out_y,  out_images


    ## eliminate result that has fewer points than threshold
    def eliminate_fewer_points(self, x, y):
        # eliminate fewer points
        out_x = []
        out_y = []
        for i, j in zip(x, y):
            if len(i)>5:
                out_x.append(i)
                out_y.append(j)     
        return out_x, out_y   

    ## generate raw output
    def generate_result(self, confidance, offsets,instance, thresh):

        mask = confidance > thresh

        grid = p.grid_location[mask]
        offset = offsets[mask]
        feature = instance[mask]

        lane_feature = []
        x = []
        y = []
        for i in range(len(grid)):
            if (np.sum(feature[i]**2))>=0:
                point_x = int((offset[i][0]+grid[i][0])*p.resize_ratio)
                point_y = int((offset[i][1]+grid[i][1])*p.resize_ratio)
                if point_x > p.x_size or point_x < 0 or point_y > p.y_size or point_y < 0:
                    continue
                if len(lane_feature) == 0:
                    lane_feature.append(feature[i])
                    x.append([point_x])
                    y.append([point_y])
                else:
                    flag = 0
                    index = 0
                    min_feature_index = -1
                    min_feature_dis = 10000
                    for feature_idx, j in enumerate(lane_feature):
                        dis = np.linalg.norm((feature[i] - j)**2)
                        if min_feature_dis > dis:
                            min_feature_dis = dis
                            min_feature_index = feature_idx
                    if min_feature_dis <= p.threshold_instance:
                        lane_feature[min_feature_index] = (lane_feature[min_feature_index]*len(x[min_feature_index]) + feature[i])/(len(x[min_feature_index])+1)
                        x[min_feature_index].append(point_x)
                        y[min_feature_index].append(point_y)
                    elif len(lane_feature) < 12:
                        lane_feature.append(feature[i])
                        x.append([point_x])
                        y.append([point_y])
                    
        return x, y

    def render(self, display):
        if self.image_left is not None and self.image_right is not None:
            img_left = np.frombuffer(self.image_left.raw_data, dtype=np.dtype("uint8"))
            img_left = np.reshape(img_left, (self.image_left.height, self.image_left.width, 4))
            img_left = img_left[:, :, :3]
            
            img_right = np.frombuffer(self.image_right.raw_data, dtype=np.dtype("uint8"))
            img_right = np.reshape(img_right, (self.image_right.height, self.image_right.width, 4))
            img_right = img_right[:, :, :3]
            
            disp_left = self.compute_left_disparity_map(img_left, img_right)
            depth_map = self.calc_depth_map(disp_left)

            pred_bboxes = self.object_detection(img_left)
            array = self.draw_bbox(img_left, pred_bboxes, depth_map, show_label=False)

            
            array = array[:, :, ::-1]
            
            
            ## Lane detection
            
            img = cv2.resize(array, None, fx=1, fy=1)
            height, width, channels = img.shape
            blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
            img = cv2.resize(img, (512,256))/255.0
            img = np.rollaxis(img, axis=2, start=0)
            _, _, ti = self.test(lane_agent, np.array([img]))
            

            mask_gray = cv2.cvtColor(ti[0], cv2.COLOR_RGB2GRAY)
            mask_gray = np.where(mask_gray>0 , 255, mask_gray)

            mask_inv = cv2.bitwise_not(mask_gray)

            image_masked = cv2.bitwise_and(array, array, mask=mask_inv)

            array = np.add(image_masked, ti[0])

            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))
            
if __name__ == '__main__':
    
    cc = CarlaClient()
    try:
        pygame.init()
        clock = pygame.time.Clock()
        cc.client = carla.Client('127.0.0.1', 2000)
        cc.client.set_timeout(5.0)
        cc.world = cc.client.get_world()

        cc.setup_car()
        cc.setup_left_camera()
        cc.setup_right_camera()
        cc.build_projection_matrix()
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
            cc.render(cc.display)
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
        