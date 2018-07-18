from .prelude import *
from . import bboxes
from scannerpy.stdlib.util import download_temp_file, temp_directory
from scannerpy.stdlib import writers
from scannerpy.stdlib.tensorflow import TensorFlowKernel
from scannerpy import FrameType
import os
import tarfile
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

LABEL_URL = 'https://storage.googleapis.com/scanner-data/public/mscoco_label_map.pbtxt'

GRAPH_PATH = os.path.join(
    os.path.expanduser('~/.scanner/resources'), 'ssd_mobilenet_v1_coco_2017_11_17',
    'frozen_inference_graph.pb')


@scannerpy.register_python_op()
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

    # Evaluate object detection DNN model on a frame
    # Return bounding box position, class and score
    def execute(self, frame: FrameType) -> bytes:
        image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        scores = self.graph.get_tensor_by_name('detection_scores:0')
        classes = self.graph.get_tensor_by_name('detection_classes:0')
        with self.graph.as_default():
            (boxes, scores, classes) = self.sess.run(
                [boxes, scores, classes], feed_dict={image_tensor: np.expand_dims(frame, axis=0)})

            bboxes = [
                self.protobufs.BoundingBox(
                    x1=box[1], y1=box[0], x2=box[3], y2=box[2], score=score, label=cls)
                for (box, score, cls) in zip(
                    boxes.reshape(100, 4), scores.reshape(100, 1), classes.reshape(100, 1))
            ]

            return writers.bboxes(bboxes, self.protobufs)


class ObjectDetectionPipeline(Pipeline):
    """
    Detects objects in a video.

    Uses the SSD-Mobilenet architecture from the TensorFlow `Object Detection API <https://github.com/tensorflow/models/tree/abd504235f3c2eed891571d62f0a424e54a2dabc/research/object_detection>`_.
    """

    job_suffix = 'objdet'
    parser_fn = lambda _: readers.bboxes
    run_opts = {'pipeline_instances_per_node': 1}

    def fetch_resources(self):
        try_import('tensorflow', __name__)

        if not os.path.isdir(os.path.join(temp_directory(), MODEL_NAME)):
            log.debug('Downloading model')
            model_tar_path = download_temp_file(DOWNLOAD_BASE + MODEL_FILE)
            with tarfile.open(model_tar_path) as f:
                f.extractall(temp_directory())
            download_temp_file(LABEL_URL)

    def build_pipeline(self):
        bboxes = self._db.ops.DetectObjects(frame=self._sources[
            'frame_sampled'].op if 'frame_sampled' in self._sources else self._sources['frame'].op)
        outputs = {'bboxes': bboxes}

        # if nms_threshold is not None:
        #     outputs['nmsed_bboxes'] = self._db.ops.BboxNMS(bboxes, threshold=nms_threshold)

        return outputs


detect_objects = ObjectDetectionPipeline.make_runner()
