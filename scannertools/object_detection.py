from .prelude import *
from . import bboxes
from scannerpy.stdlib.util import download_temp_file, temp_directory
from scannerpy.stdlib import tensorflow, writers
from scannerpy import FrameType
import os
import tarfile
import numpy as np
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

LABEL_URL = 'https://storage.googleapis.com/scanner-data/public/mscoco_label_map.pbtxt'

GRAPH_PATH = os.path.join(
    os.path.expanduser('~/.scanner/resources'), 'ssd_mobilenet_v1_coco_2017_11_17',
    'frozen_inference_graph.pb')


@scannerpy.register_python_op()
class ObjDetect(tensorflow.TensorFlowKernel):
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


@autobatch(uniforms=[0, 'nms_threshold'])
def detect_objects(db, videos, frames=None, nms_threshold=None):
    """
    detect_objects(db, videos, frames=None, nms_threshold=None)
    Detects objects in a video.

    Uses the SSD-Mobilenet architecture from the TensorFlow `Object Detection API <https://github.com/tensorflow/models/tree/abd504235f3c2eed891571d62f0a424e54a2dabc/research/object_detection>`_.

    Args:
        db (scannerpy.Database): Handle to Scanner database.
        videos (Video, autobatched): Videos to process.
        frames (List[int], autobatched, optional): Frame indices to process.
        nms_threshold (float, optional): Fraction of IoU to merge bounding boxes during NMS.

    Returns:
        List[List[BoundingBox]] (autobatched): List of bounding boxes for each frame.
    """

    try_import('tensorflow', __name__)

    if not os.path.isdir(os.path.join(temp_directory(), MODEL_NAME)):
        log.debug('Downloading model')
        model_tar_path = download_temp_file(DOWNLOAD_BASE + MODEL_FILE)
        with tarfile.open(model_tar_path) as f:
            f.extractall(temp_directory())
        download_temp_file(LABEL_URL)

    log.debug('Ingesting video')
    scanner_ingest(db, videos)

    frame = db.sources.FrameColumn()
    frame_sampled = frame.sample()
    bboxes = db.ops.ObjDetect(frame=frame_sampled)
    outputs = {'bboxes': bboxes}

    if nms_threshold is not None:
        outputs['nmsed_bboxes'] = db.ops.BboxNMS(bboxes, threshold=nms_threshold)

    output = db.sinks.Column(columns=outputs)

    jobs = [
        Job(
            op_args={
                frame: db.table(v.scanner_name()).column('frame'),
                frame_sampled: db.sampler.gather(f) if f is not None else db.sampler.all(),
                output: v.scanner_name() + '_objdet'
            }) for v, f in zip(videos, frames or [None for _ in range(len(videos))])
    ]

    log.debug('Running object detection Scanner job')
    output_tables = db.run(output, jobs, force=True)

    log.debug('Loading bounding boxes')
    all_bboxes = [
        list(
            output_table.column('nmsed_bboxes' if nms_threshold is not None else 'bboxes').load(
                readers.bboxes)) for output_table in output_tables
    ]

    return all_bboxes
