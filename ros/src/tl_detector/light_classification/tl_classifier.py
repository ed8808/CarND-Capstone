from styx_msgs.msg import TrafficLight

import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import datetime as dt

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

class TLClassifier(object):
  def __init__(self):
        PATH = "/home/student/udacity/CarND-Capstone/data/"

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_CKPT = PATH + "frozen_inference_graph.pb"

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)

            tf.import_graph_def(od_graph_def, name='')

            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

        self.sess = tf.Session(graph=self.detection_graph)


  def get_classification(self, img):
        # Bounding Box Detection.
        with self.detection_graph.as_default():
          img = cv2.resize(img, (80,60))/255.
          
          # Expand dimension since the model expects image to have shape [1, None, None, 3].
          img_expanded = np.expand_dims(img, axis=0)

          print("s: ", dt.datetime.now())

          (num_detections, score, boxes, classe) = self.sess.run(
             [self.num_d, self.d_scores, self.d_boxes, self.d_classes],
             feed_dict={self.image_tensor: img_expanded})

              #im_height, im_width = img.shape[:2]
              #box = np.squeeze(boxes)
              #ymin, xmin, ymax, xmax = int(box[i][0]*im_height), int(box[i][1]*im_width), int(box[i][2]*im_height), int(box[i][3]*im_width)
              #img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (50*classe[0][i], 0, 0), thickness=2)
              #cv2.putText(img, str(classe[0][i])+ " --> "+str(score[0][i]), 
              #(im_width/4, im_height*3/4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
	  
          print("e: ", dt.datetime.now())
          max_i = np.argmax(score[0])
          if score[0][max_i] > 0.5:
            print(classe[0][max_i]," ",score[0][max_i])
            if classe[0][max_i]==0: return TrafficLight.RED
            elif classe[0][max_i]==1: return TrafficLight.YELLOW
            elif classe[0][max_i]==2: return TrafficLight.GREEN
          
          return TrafficLight.UNKNOWN

