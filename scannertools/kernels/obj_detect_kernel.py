# Mostly taken from: https://github.com/tensorflow/models/blob/master/object_detection/object_detection_tutorial.ipynb

import numpy as np
import tensorflow as tf
import cv2
import os
from scannerpy.stdlib import kernel, writers
import pickle

##################################################################################################
# Assume that DNN model is located in PATH_TO_GRAPH with filename 'frozen_inference_graph.pb'    #
# Example model can be downloaded from:                                                          #
# http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz #
##################################################################################################

GRAPH_PATH = os.path.join(
    os.path.expanduser('~/.scanner/resources'), 'ssd_mobilenet_v1_coco_2017_11_17',
    'frozen_inference_graph.pb')


class ObjDetectKernel(kernel.TensorFlowKernel):
    def build_graph(self):
        dnn = tf.Graph()
        with dnn.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(GRAPH_PATH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return dnn

    # Evaluate object detection DNN model on a frame
    # Return bounding box position, class and score
    def execute(self, cols):
        image = cols[0]
        image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        scores = self.graph.get_tensor_by_name('detection_scores:0')
        classes = self.graph.get_tensor_by_name('detection_classes:0')
        with self.graph.as_default():
            (boxes, scores, classes) = self.sess.run(
                [boxes, scores, classes], feed_dict={
                    image_tensor: np.expand_dims(image, axis=0)
                })

            bboxes = [
                self.protobufs.BoundingBox(
                    x1=box[1], y1=box[0], x2=box[3], y2=box[2], score=score, label=cls)
                for (box, score, cls) in zip(
                    boxes.reshape(100, 4), scores.reshape(100, 1), classes.reshape(100, 1))
            ]

            return [writers.bboxes(bboxes, self.protobufs)]


KERNEL = ObjDetectKernel
