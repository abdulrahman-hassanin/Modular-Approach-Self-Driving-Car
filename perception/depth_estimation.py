import numpy as np
import cv2
import colorsys
import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from PSMNet_asset.models import *

class DepthEstimation():
    def __init__(self):
        self.left_img = None
        self.right_img = None
        self.img_width = 1920//2
        self.img_height = 1080//2
        self.view_fov = 90
        self.focal_length = self.img_width / (2.0 * np.tan(self.view_fov * np.pi / 360.0))
        self.base_line = 0.4
        self.Q = None
        self.disparity_left = None
        self.depth_map = None
        self.nearest_point = None
        self.PSMNet = self.load_PSMNet()
        self.build_projection_matrix()

    def load_PSMNet(self, cuda=True, model='stackhourglass', maxdisp=192, model_path='./PSMNet_asset/trained/pretrained_model_KITTI2015.tar'):
        if model == 'stackhourglass':
            self.PSMNet = stackhourglass(maxdisp)
        else:
            self.PSMNet = basic(maxdisp)

        self.PSMNet = nn.DataParallel(self.PSMNet, device_ids=[0])
        self.PSMNet.cuda()
        
        state_dict = torch.load(model_path)
        self.PSMNet.load_state_dict(state_dict['state_dict'])

    def PSMNet_pred(self, imgL, imgR, cuda=True):
        imgL, imgR = self.procesing_left_right_img(self, imgL, imgR)
        
        self.PSMNet.eval()

        if cuda:
           imgL = imgL.cuda()
           imgR = imgR.cuda()     

        with torch.no_grad():
            disp = self.PSMNet(imgL,imgR)

        disp = torch.squeeze(disp)
        pred_disp = disp.data.cpu().numpy()

        return pred_disp

    def compute_disparity_PSMNet(self, imgL_o, imgR_o):
        normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                            'std': [0.229, 0.224, 0.225]}
        infer_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(**normal_mean_var)])    

        imgL = infer_transform(imgL_o)
        imgR = infer_transform(imgR_o)

        # pad to width and hight to 16 times
        if imgL.shape[1] % 16 != 0:
            times = imgL.shape[1]//16       
            top_pad = (times+1)*16 -imgL.shape[1]
        else:
            top_pad = 0

        if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2]//16                       
            right_pad = (times+1)*16-imgL.shape[2]
        else:
            right_pad = 0    

        imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
        imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

        pred_disp = self.PSMNet_pred(imgL,imgR)
        
        if top_pad !=0 and right_pad != 0:
            img = pred_disp[top_pad:,:-right_pad]
        elif top_pad ==0 and right_pad != 0:
            img = pred_disp[:,:-right_pad]
        elif top_pad !=0 and right_pad == 0:
            img = pred_disp[top_pad:,:]
        else:
            img = pred_disp
        
        img = (img*256).astype('uint16')
        
        return img

    def build_projection_matrix(self):
        # Define K Projection matrix
        # k = [[Fx,  0, IMG_hight/2],
        #      [ 0, Fy, IMG_width/2],
        #      [ 0,  0,           1]]
        # self.K = np.identity(3)
        # self.K[0, 0] = self.focal_length
        # self.K[1, 1] = self.focal_length
        # self.K[0, 2] = self.img_height / 2.0
        # self.K[1, 2] = self.img_width  / 2.0
        
        self.Q = np.float32([[1, 0, 0, -self.img_width / 2],
                            [0, 1, 0, -self.img_height / 2],
                            [0, 0, 0, self.focal_length],
                            [0, 0, 1/self.base_line, 0]])

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

    def calculate_distance_from_point_cloud(self, point_cloud, c1, c2):
        x, y = c1
        x2, y2 = c2
        x = int(x)
        y = int(y)
        x2 = int(x2)
        y2 = int(y2)

        obstacle_depth = point_cloud[y:y2, x:x2, :]
        X, Y, Z = obstacle_depth[:, :, 0].mean(), obstacle_depth[:, :, 1].mean(), obstacle_depth[:, :, 2].mean()
        P = np.sqrt(X**2 + Y**2 + Z**2)
        return X, Y, Z, P

    def calc_point_cloud(self, disparity):
        points = cv2.reprojectImageTo3D(disparity, self.Q)
        points = np.reshape(points, (self.img_height, self.img_width, 3))
        return points

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
    
    def clac_distance_bbox_width(self, bbox_width):
        if bbox_width > 80:
            depth = ((-1 / 33) * bbox_width ) + 12
            return max(depth, 0.5)
        elif bbox_width > 40:
            depth = ((-0.125) * bbox_width ) + 20
            return depth 
        elif bbox_width > 10:
            depth = ((-0.6) * bbox_width ) + 36
            return depth
   
        return 0
