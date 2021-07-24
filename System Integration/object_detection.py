#import tensorflow as tf
#from tensorflow.python.saved_model import tag_constants
import numpy as np
import cv2
import colorsys
import random



import depth_estimation

depth_estimator = depth_estimation.DepthEstimation()
yolov = 'yolov5'
if yolov == 'yolov5':
    import torch
    import cv2
    import time
    import numpy as np
    import colorsys
    import random
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained =True)
    model.eval()

elif yolov == 'yolov4':
    import tensorflow as tf
    from tensorflow.python.saved_model import tag_constants

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    saved_loaded_model = tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING])
    YOLO_model = saved_loaded_model.signatures['serving_default']

class ObjectDetection(object):
    def __init__(self):
        self.weight_path = './checkpoints/yolov4-416'
        self.YOLO_model = self.load_yolo()
        self.img = None
        self.classes = self.read_class_names("./classes/coco.names")
        self.allowed_classes = list(self.read_class_names("./classes/coco_allowed.names").values())

    def load_yolo(self):
        pass
        # physical_devices = tf.config.experimental.list_physical_devices('GPU')
        # if len(physical_devices) > 0:
        #     tf.config.experimental.set_memory_growth(physical_devices[0], True)
        # self.YOLO_model = tf.saved_model.load(self.weight_path, tags=[tag_constants.SERVING])
        # self.YOLO_model = self.YOLO_model.signatures['serving_default']

    def read_class_names(self, class_file_name):
        names = {}
        with open(class_file_name, 'r') as data:
            for ID, name in enumerate(data):
                names[ID] = name.strip('\n')
        return names

    def detect(self, img):
    
        if yolov == 'yolov4':
            image_data = cv2.resize(img, (416, 416))
            image_data = image_data / 255.
    
            images_data = []
            for i in range(1):
                images_data.append(image_data)
            images_data = np.asarray(images_data).astype(np.float32)
    
            batch_data = tf.constant(images_data)
            with tf.device("/GPU:0"):
                tf.debugging.set_log_device_placement(True)
                pred_bbox = YOLO_model(batch_data)
    
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
        
        elif yolov == 'yolov5':
                    
            h,w,c = img.shape
            
            results = model(img, size=w)
            pred_bbox = results.pandas().xyxy[0]
            
        return pred_bbox


    def draw_bbox(self, image, bboxes, points, mode='point_cloud', show_label=False, Depth_by_bbox=True):
        
        if yolov == 'yolov4':
            img_traffic = image # Used in traffic lights classification
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
                    
                    # Using Disparity map to calculate depth
                    if not Depth_by_bbox or self.classes[class_ind] != 'car':
                        if mode == 'point_cloud':
                            x, y, z, p = depth_estimator.calculate_distance_from_point_cloud(points, c1, c2)
                            cv2.putText(image, str('%.2f'% p)+' m', (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale, (0, 0, 255), bbox_thick // 2, lineType=cv2.LINE_AA)
                        elif mode == 'nearest_point':
                            depth = depth_estimator.calculate_nearest_point(points, c1, c2)
                            cv2.putText(image, str('%.2f'% depth)+' m', (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale, (0, 0, 255), bbox_thick // 2, lineType=cv2.LINE_AA)
                    
                    # Using bounding box to calculate depth
                    if Depth_by_bbox and self.classes[class_ind] == 'car' :
                        dist = depth_estimator.clac_distance_bbox_width(int(coor[3]) - int(coor[1]))
                        cv2.putText(image, str('%.2f'% dist)+' m', (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                    
                    # Traffic lights classification
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
                        bbox_mess = '%s, %s' % (str(traffic_color), self.classes[class_ind]) # add traffic light color to the label
                        
                        t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                        c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
                        
                        cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled
            
                        cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
        
        elif yolov == 'yolov5':
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
    
            num_boxes = len(bboxes)
    
            for i in range(num_boxes):
                xmin, ymin, xmax, ymax, confidence, class_, name = bboxes.loc[i].tolist()
                if confidence < 0.25:
                    print("YES " + str(name))
                    continue
                coor = [0,0,0,0]
                coor[0] = int(ymin)
                coor[2] = int(ymax)
                coor[1] = int(xmin)
                coor[3] = int(xmax)
    
                fontScale = 0.65
                score = confidence
                class_ind = int(class_)
                class_name = self.classes[class_ind]
    
                # check if class is in allowed classes
                if class_name not in self.allowed_classes:
                    continue
                else:
                    bbox_color = colors[class_ind]
                    bbox_thick = int(0.6 * (image_h + image_w) / 600)
                    c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
                    cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
                    
                    # Using Disparity map to calculate depth
                    if not Depth_by_bbox or self.classes[class_ind] != 'car':
                        if mode == 'point_cloud':
                            x, y, z, p = depth_estimator.calculate_distance_from_point_cloud(points, c1, c2)
                            cv2.putText(image, str('%.2f'% p)+' m', (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale, (0, 0, 255), 2, lineType=cv2.LINE_AA)
                        elif mode == 'nearest_point':
                            depth = depth_estimator.calculate_nearest_point(points, c1, c2)
                            cv2.putText(image, str('%.2f'% depth)+' m', (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale, (0, 0, 255), bbox_thick // 2, lineType=cv2.LINE_AA)
                    
                    # Using bounding box to calculate depth
                    if self.classes[class_ind] == 'car' and Depth_by_bbox :
                        
                        dist = depth_estimator.clac_distance_bbox_width(int(coor[3]) - int(coor[1]))
                        
                        cv2.putText(image, str('%.2f'% dist)+' m', (c1[0], (c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)            
    
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
                        
                        cv2.rectangle(image, c1, ((c3[0]), (c3[1])), bbox_color, -1) #filled
    
                        cv2.putText(image, bbox_mess, (c1[0], (c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
                    
                    elif self.classes[class_ind] == 'traffic light':
                        if traffic_color != 'Unknown color':
                            bbox_mess = '%s, %s' % (str(traffic_color), self.classes[class_ind]) # add traffic light color to the label
                        else:
                            bbox_mess = '%s' % (self.classes[class_ind]) # add traffic light color to the label
                        
                        t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                        c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
                        
                        cv2.rectangle(image, c1, ((c3[0]), (c3[1])), bbox_color, -1) #filled
    
                        cv2.putText(image, bbox_mess, (c1[0], (c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

        return image

    def get_object_states_list(self, bboxes, points_cloud):
        vehicles_list = []
        pedstrains_list = []

        num_boxes = len(bboxes)
        for i in range(num_boxes):
            xmin, ymin, xmax, ymax, confidence, class_, name = bboxes.loc[i].tolist()
            if confidence < 0.25:
                print("YES " + str(name))
                continue
            coor = [0,0,0,0]
            coor[0] = int(ymin)
            coor[2] = int(ymax)
            coor[1] = int(xmin)
            coor[3] = int(xmax)
            
            c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
            class_ind = int(class_)
            class_name = self.classes[class_ind]

            if class_name not in self.allowed_classes:
                continue
            
            if self.classes[class_ind] == 'car':
                x, y, z, p = depth_estimator.calculate_distance_from_point_cloud(points_cloud, c1, c2)
                vehicles_list.append([x, y, z, p])

            elif self.classes[class_ind] == 'person':
                x, y, z, p = depth_estimator.calculate_distance_from_point_cloud(points_cloud, c1, c2)
                pedstrains_list.append([x, y, z, p])
        
        return vehicles_list, pedstrains_list