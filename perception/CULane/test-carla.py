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

VIEW_WIDTH = 1920 // 2
VIEW_HEIGHT = 1080 // 2
VIEW_FOV = 90

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

 
############################################################################
## evaluate on the test dataset
############################################################################
def evaluation(loader, lane_agent, thresh = p.threshold_point, index= -1, name = None):
    progressbar = tqdm(range(loader.size_test//4))
    for test_image, ratio_w, ratio_h, path, target_h, target_lanes in loader.Generate_Test():
        x, y, _ = test(lane_agent, test_image, thresh, index= index)
        x_ = []
        y_ = []
        for i, j in zip(x, y):
            temp_x, temp_y = util.convert_to_original_size(i, j, ratio_w, ratio_h)
            x_.append(temp_x)
            y_.append(temp_y)
        #x_, y_ = find_target(x_, y_, ratio_w, ratio_h)
        x_, y_ = fitting(x_, y_, ratio_w, ratio_h)

        #util.visualize_points_origin_size(x_[0], y_[0], test_image[0]*255, ratio_w, ratio_h)
        #print(target_lanes)
        #util.visualize_points_origin_size(target_lanes[0], target_h[0], test_image[0]*255, ratio_w, ratio_h)

        result_data = write_result(x_, y_, path)
        progressbar.update(1)
    progressbar.close()

############################################################################
## linear interpolation for fixed y value on the test dataset
############################################################################
def find_target(x, y, ratio_w, ratio_h):
    # find exact points on target_h
    out_x = []
    out_y = []
    x_size = p.x_size/ratio_w
    y_size = p.y_size/ratio_h
    for x_batch, y_batch in zip(x,y):
        predict_x_batch = []
        predict_y_batch = []
        for i, j in zip(x_batch, y_batch):
            min_y = min(j)
            max_y = max(j)
            temp_x = []
            temp_y = []
            for h in range(100, 590, 10):
                temp_y.append(h)
                if h < min_y:
                    temp_x.append(-2)
                elif min_y <= h and h <= max_y:
                    for k in range(len(j)-1):
                        if j[k] >= h and h >= j[k+1]:
                            #linear regression
                            if i[k] < i[k+1]:
                                temp_x.append(int(i[k+1] - float(abs(j[k+1] - h))*abs(i[k+1]-i[k])/abs(j[k+1]+0.0001 - j[k])))
                            else:
                                temp_x.append(int(i[k+1] + float(abs(j[k+1] - h))*abs(i[k+1]-i[k])/abs(j[k+1]+0.0001 - j[k])))
                            break
                else:
                    temp_x.append(-2)
            predict_x_batch.append(temp_x)
            predict_y_batch.append(temp_y)
        out_x.append(predict_x_batch)
        out_y.append(predict_y_batch)            
    
    return out_x, out_y

def fitting(x, y, ratio_w, ratio_h):
    out_x = []
    out_y = []
    x_size = p.x_size/ratio_w
    y_size = p.y_size/ratio_h

    for x_batch, y_batch in zip(x,y):
        predict_x_batch = []
        predict_y_batch = []
        for i, j in zip(x_batch, y_batch):
            min_y = min(j)
            max_y = max(j)
            temp_x = []
            temp_y = []

            jj = []
            pre = -100
            for temp in j[::-1]:
                if temp > pre:
                    jj.append(temp)
                    pre = temp
                else:
                    jj.append(pre+0.00001)
                    pre = pre+0.00001
            sp = csaps.CubicSmoothingSpline(jj, i[::-1], smooth=0.0001)

            last = 0
            last_second = 0
            last_y = 0
            last_second_y = 0
            for pts in range(62, -1, -1):
                h = 590 - pts*5 - 1
                temp_y.append(h)
                if h < min_y:
                    temp_x.append(-2)
                elif min_y <= h and h <= max_y:
                    temp_x.append( sp([h])[0] )
                    last = temp_x[-1]
                    last_y = temp_y[-1]
                    if len(temp_x)<2:
                        last_second = temp_x[-1]
                        last_second_y = temp_y[-1]
                    else:
                        last_second = temp_x[-2]
                        last_second_y = temp_y[-2]
                else:
                    if last < last_second:
                        l = int(last_second - float(-last_second_y + h)*abs(last_second-last)/abs(last_second_y+0.0001 - last_y))
                        if l > x_size or l < 0 :
                            temp_x.append(-2)
                        else:
                            temp_x.append(l)
                    else:
                        l = int(last_second + float(-last_second_y + h)*abs(last_second-last)/abs(last_second_y+0.0001 - last_y))
                        if l > x_size or l < 0 :
                            temp_x.append(-2)
                        else:
                            temp_x.append(l)
            predict_x_batch.append(temp_x[::-1])
            predict_y_batch.append(temp_y[::-1])
        out_x.append(predict_x_batch)
        out_y.append(predict_y_batch) 


    return out_x, out_y

############################################################################
## write result
############################################################################
def write_result(x, y, path):
    
    batch_size = len(path)
    save_path = "/home/kym/research/autonomous_car_vision/lane_detection/code/ITS/CuLane/output"
    for i in range(batch_size):
        path_detail = path[i].split("/")
        first_folder = path_detail[0]
        second_folder = path_detail[1]
        file_name = path_detail[2].split(".")[0]+".lines.txt"
        if not os.path.exists(save_path+"/"+first_folder):
            os.makedirs(save_path+"/"+first_folder)
        if not os.path.exists(save_path+"/"+first_folder+"/"+second_folder):
            os.makedirs(save_path+"/"+first_folder+"/"+second_folder)      
        with open(save_path+"/"+first_folder+"/"+second_folder+"/"+file_name, "w") as f:  
            for x_values, y_values in zip(x[i], y[i]):
                count = 0
                if np.sum(np.array(x_values)>=0) > 1 : ######################################################
                    for x_value, y_value in zip(x_values, y_values):
                        if x_value >= 0:
                            f.write(str(x_value) + " " + str(y_value) + " ")
                            count += 1
                    if count>1:
                        f.write("\n")


############################################################################
## save result by json form
############################################################################
def save_result(result_data, fname):
    with open(fname, 'w') as make_file:
        for i in result_data:
            json.dump(i, make_file, separators=(',', ': '))
            make_file.write("\n")

############################################################################
## test on the input test image
############################################################################
def test(lane_agent, test_images, thresh = p.threshold_point, index= -1):

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
        raw_x, raw_y = generate_result(confidence, offset, instance, thresh)

        # eliminate fewer points
        in_x, in_y = eliminate_fewer_points(raw_x, raw_y)
                
        # sort points along y 
        in_x, in_y = util.sort_along_y(in_x, in_y)  

        result_image = util.draw_points(in_x, in_y, deepcopy(image))

        out_x.append(in_x)
        out_y.append(in_y)
        out_images.append(result_image)

    return out_x, out_y,  out_images

############################################################################
## eliminate result that has fewer points than threshold
############################################################################
def eliminate_fewer_points(x, y):
    # eliminate fewer points
    out_x = []
    out_y = []
    for i, j in zip(x, y):
        if len(i)>5:
            out_x.append(i)
            out_y.append(j)     
    return out_x, out_y   

############################################################################
## generate raw output
############################################################################
def generate_result(confidance, offsets,instance, thresh):

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

class CarlaClient():

    def __init__(self):
        self.client = None
        self.world = None
        self.camera = None
        self.car = None
        self.image = None
        self.display = None
        self.capture = True

    def setup(self):
        try:
            pygame.init()
            self.client = carla.Client('127.0.0.1', 2000)
            self.client.set_timeout(5.0)
            self.world = self.client.get_world()
            self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
        except Exception as e:
            print(e)

    def setdown(self):
        self.set_synchronous_mode(False)
        self.camera.destory()
        self.car.destory()
        pygame.quit()
        cv2.destroyAllWindows()

        
    def camera_bp(self):
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))

        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    def setup_car(self):
        car_bp = self.world.get_blueprint_library().filter('model3')[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.spawn_actor(car_bp, location)

    def setup_camera(self):
        camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7))
        self.camera = self.world.spawn_actor(self.camera_bp(), camera_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

    @staticmethod
    def set_image(weak_self, img):
        self = weak_self()
        if self.capture:
            self.image = img
            self.capture = False

    def render(self, display):
        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            # remove info do not need
            #array = pre_processing(array)
            img = cv2.resize(array, None, fx=1, fy=1)
            height, width, channels = img.shape
            blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
            img = cv2.resize(img, (512,256))/255.0
            img = np.rollaxis(img, axis=2, start=0)
            _, _, ti = test(lane_agent, np.array([img]))
            cv2.imshow('frame',ti[0])
            #lane_agent.setInput(blob)
            #outputs = lane_agent.forward(output_layers)
            
            array = array[:, ::-1]
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))

    
if __name__ == "__main__":
    cc = CarlaClient()
    cc.setup()
    cc.setup_car()
    cc.setup_camera()
    pygame_clock = pygame.time.Clock()
    cc.car.set_autopilot(True)

    while True:
        fps = FPS().start()
        cc.world.tick()
        cc.capture = True
        pygame_clock.tick_busy_loop(30)
        cc.render(cc.display)
        #pygame.display.flip()
        pygame.event.pump()
        cv2.waitKey(1)
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        
        
