"""
========================= Lane Detection =========================
- Graduation project 2021 | Modular-approch self driving car system
- Faculty of engineering Helwan university
===================================================================
- Description:
              Apply object detection module to detecet enviroment
              lanes
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
import cv2
import json
import torch
import time
import carla
import csaps
import random
import weakref
import pygame
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from CULane import util
from CULane import agent
from CULane.data_loader import Generator
from CULane.parameters import Parameters
from CULane.utils import pre_processing
from imutils.video import FPS
from sklearn.linear_model import LinearRegression

p = Parameters()
# Lane detecton class
class LaneDetection():

    # Constructor
    def __init__(self):
        """ Initialization """
        self.VIEW_WIDTH  = 1920 // 2
        self.VIEW_HEIGHT = 1080 // 2
        self.VIEW_FOV    = 90
        self.p           = Parameters()
        self.loader      = Generator()
        self.lane_agent  = None
        self.load_agent()
    
    def load_agent(self):        
        # Get agent and model
        print('Get agent')
        if self.p.model_path == "":
            self.lane_agent = agent.Agent()
        else:
            self.lane_agent = agent.Agent()
            self.lane_agent.load_weights(296, "tensor(1.6947)")
            
        # Check GPU
        print('Setup GPU mode')
        if torch.cuda.is_available():
            self.lane_agent.cuda()

        # Testing
        print('Testing loop')
        self.lane_agent.evaluate_mode()
    
    # Detect
    def detect(self, test_images, thresh = p.threshold_point, index= -1):
        """ Function used to detect lane """

        result = self.lane_agent.predict_lanes_test(test_images)
        torch.cuda.synchronize()
        confidences, offsets, instances = result[index]
        
        num_batch = len(test_images)

        out_x = []
        out_y = []
        out_images = []
        
        for i in range(num_batch):
            # Test on test data set
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

            # Generate point and cluster
            raw_x, raw_y = self.generate_result(confidence, offset, instance, thresh)

            # Eliminate fewer points
            in_x, in_y = self.eliminate_fewer_points(raw_x, raw_y)
                    
            # Sort points along y 
            in_x, in_y = util.sort_along_y(in_x, in_y)  
 
            image_zeros = np.zeros(image.shape, np.uint8)
            result_image = util.draw_points(in_x, in_y, deepcopy(image_zeros))
            result_image = cv2.resize(result_image, (self.VIEW_WIDTH, self.VIEW_HEIGHT))
            
            out_x.append(in_x)
            out_y.append(in_y)
            out_images.append(result_image)

        return out_x, out_y,  out_images

    # Eliminate result that has fewer points than threshold
    def eliminate_fewer_points(self, x, y):
        """ Function used to eliminate fewer points """
        out_x = []
        out_y = []
        for i, j in zip(x, y):
            if len(i)>5:
                out_x.append(i)
                out_y.append(j)     
        return out_x, out_y   

    # Generate raw output
    def generate_result(self, confidance, offsets,instance, thresh):
        """ Function used to generate output """

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

    def detect_draw_lane(self, img_left, array):
        img = cv2.resize(img_left, None, fx=1, fy=1)
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
        img = cv2.resize(img, (512,256))/255.0
        img = np.rollaxis(img, axis=2, start=0)
        _, _, ti = self.detect(np.array([img]))

        mask_gray = cv2.cvtColor(ti[0], cv2.COLOR_RGB2GRAY)
        mask_gray = np.where(mask_gray>0 , 255, mask_gray)

        mask_inv = cv2.bitwise_not(mask_gray)

        image_masked = cv2.bitwise_and(array, array, mask=mask_inv)

        array = np.add(image_masked, ti[0])
        return array     