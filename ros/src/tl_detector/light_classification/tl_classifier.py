from styx_msgs.msg import TrafficLight

import numpy as np
import tensorflow as tf
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

class TLClassifier(object):
  def __init__(self):
        PATH = "/home/student/CarND-Capstone/data/"

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
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')

        self.sess = tf.Session(graph=self.detection_graph)

  def get_classification(self, img):
        # Bounding Box Detection.
        with self.detection_graph.as_default():
          # Expand dimension since the model expects image to have shape [1, None, None, 3].
          img_expanded = np.expand_dims(img, axis=0)  
          (boxes, scores, classes, num) = self.sess.run(
             [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
             feed_dict={self.image_tensor: img_expanded})
          if classes == 'traffic_light_green':
            return TrafficLight.GREEN
          elif classes =='traffic_light_red':
            return TrafficLight.RED
          elif classes =='traffic_light_yellow':
            return TrafficLight.YELLOW

          return TrafficLight.UNKNOWN
