from scannerpy.util import download_temp_file, temp_directory
from .tensorflow import TensorFlowKernel
from scannerpy import FrameType, DeviceType, protobufs
from scannerpy.types import BboxList
import os
import tarfile
import numpy as np
import scannerpy
from typing import Sequence

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

LABEL_URL = 'https://storage.googleapis.com/scanner-data/public/mscoco_label_map.pbtxt'

GRAPH_PATH = os.path.join(
    temp_directory(), 'ssd_mobilenet_v1_coco_2017_11_17',
    'frozen_inference_graph.pb')


@scannerpy.register_python_op(device_sets=[[DeviceType.CPU, 0], [DeviceType.GPU, 1]], batch=5)
class DetectObjects(TensorFlowKernel):
    def build_graph(self):
        import tensorflow as tf
        dnn = tf.Graph()
        with dnn.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(GRAPH_PATH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return dnn

    def fetch_resources(self):
        model_tar_path = download_temp_file(DOWNLOAD_BASE + MODEL_FILE)
        with tarfile.open(model_tar_path) as f:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, temp_directory())
        download_temp_file(LABEL_URL)

    # Evaluate object detection DNN model on a frame
    # Return bounding box position, class and score
    def execute(self, frame: Sequence[FrameType]) -> Sequence[BboxList]:
        image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        scores = self.graph.get_tensor_by_name('detection_scores:0')
        classes = self.graph.get_tensor_by_name('detection_classes:0')
        with self.graph.as_default():
            (boxes, scores, classes) = self.sess.run(
                [boxes, scores, classes], feed_dict={image_tensor: np.concatenate(np.expand_dims(frame, axis=0), axis=0)})

            bboxes = [
                [
                    protobufs.BoundingBox(
                        x1=box[1], y1=box[0], x2=box[3], y2=box[2], score=score, label=cls)
                    for (box, score, cls) in zip(
                        boxes[i, :, :].reshape(100, 4), scores[i, :].reshape(100, 1), classes[i, :].reshape(100, 1))
                ]
                for i in range(len(frame))
            ]

            return bboxes
