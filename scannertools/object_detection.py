from .prelude import *
from scannerpy.stdlib.util import download_temp_file, temp_directory
import os
import tarfile
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

LABEL_URL = 'https://storage.googleapis.com/scanner-data/public/mscoco_label_map.pbtxt'


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

    log.debug('Registering kernel')
    try:
        db.register_op('ObjDetect', [('frame', ColumnType.Video)], ['bboxes'])
        db.register_python_kernel('ObjDetect', DeviceType.CPU,
                                  SCRIPT_DIR + '/kernels/obj_detect_kernel.py')
    except ScannerException:
        pass

    try:
        db.register_op('BboxNMS', [], ['nmsed_bboxes'], variadic_inputs=True)
        db.register_python_kernel('BboxNMS', DeviceType.CPU,
                                  SCRIPT_DIR + '/kernels/bbox_nms_kernel.py')
    except ScannerException:
        pass

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
