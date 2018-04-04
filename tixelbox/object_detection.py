from prelude import *
import bbox
from scannerpy.stdlib.util import download_temp_file, temp_directory
import os
import tarfile
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

LABEL_URL = 'https://storage.googleapis.com/scanner-data/public/mscoco_label_map.pbtxt'


@autobatch(uniforms=['nms_threshold'])
def detect_objects(videos, frames=None, nms_threshold=None):
    if not os.path.isdir(os.path.join(temp_directory(), MODEL_NAME)):
        log.debug('Downloading model')
        model_tar_path = download_temp_file(DOWNLOAD_BASE + MODEL_FILE)
        with tarfile.open(model_tar_path) as f:
            f.extractall(temp_directory())
        download_temp_file(LABEL_URL)

    log.debug('Connecting to scanner')
    with get_scanner_db() as db:
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
            db.register_op('BboxNMS', ['bboxes'], ['bboxes'])
            db.register_python_kernel('BboxNMS', DeviceType.CPU,
                                      SCRIPT_DIR + '/kernels/bbox_nms_kernel.py')
        except ScannerException:
            pass

        frame = db.sources.FrameColumn()
        frame_sampled = frame.sample()
        bboxes = db.ops.ObjDetect(frame=frame_sampled)
        outputs = {'bboxes': bboxes}

        if nms_threshold is not None:
            outputs['nmsed_bboxes'] = db.ops.BboxNMS(bboxes=bboxes, threshold=nms_threshold)

        output = db.sinks.Column(columns=outputs)

        jobs = [
            Job(
                op_args={
                    frame: db.table(v.scanner_name()).column('frame'),
                    frame_sampled: db.sampler.gather(f) if f is not None else db.sampler.all(),
                    output: v.scanner_name() + '_objdet'
                }) for v, f in zip(videos, frames or [None for _ in range(len(videos))])
        ]

        log.debug('Running job')
        output_tables = db.run(output, jobs, force=True)

        all_bboxes = [[
            pickle.loads(box)
            for box in output_table.column('nmsed_bboxes'
                                           if nms_threshold is not None else 'bboxes').load()
        ] for output_table in output_tables]

    return all_bboxes


@autobatch()
def draw_bboxes(videos, bboxes, frames=None, path=None):
    log.debug('Connecting to scanner')
    with get_scanner_db() as db:
        log.debug('Ingesting video')
        scanner_ingest(db, videps)

        try:
            db.register_op('BboxDraw', [('frame', ColumnType.Video), 'bboxes'],
                           [('frame', ColumnType.Video)])
            db.register_python_kernel('BboxDraw', DeviceType.CPU,
                                      SCRIPT_DIR + '/kernels/bbox_draw_kernel.py')
        except ScannerException:
            pass

        for vid_bboxes in bboxes:
            db.new_table(
                video.scanner_name() + '_bboxes_draw', ['bboxes'],
                [[pickle.dumps(bb)] for bb in vid_bboxes],
                force=True)

        frame = db.sources.FrameColumn()
        frame_sampled = frame.sample()
        bboxes = db.sources.Column()
        frame_drawn = db.ops.BboxDraw(frame=frame_sampled, bboxes=bboxes)
        output = db.sinks.Column(columns={'frame': frame_drawn})

        log.debug('Running job')
        jobs = [
            Job(
                op_args={
                    frame: db.table(v.scanner_name()).column('frame'),
                    frame_sampled: db.sampler.gather(f) if f is not None else db.sampler.all(),
                    bboxes: db.table(v.scanner_name() + '_bboxes_draw').column('bboxes'),
                    output: 'tmp'
                }) for v, f in zip(videos, frames or [None for _ in range(len(videos))])
        ]
        db.run(output, jobs, force=True)

        if path is None:
            path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name

        db.table('tmp').column('frame').save_mp4(os.path.splitext(path)[0])
