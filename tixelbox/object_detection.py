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


def detect_objects(video, frames=None, cache=False, nms_threshold=None):
    if not os.path.isdir(os.path.join(temp_directory(), MODEL_NAME)):
        log.debug('Downloading model')
        model_tar_path = download_temp_file(DOWNLOAD_BASE + MODEL_FILE)
        with tarfile.open(model_tar_path) as f:
            f.extractall(temp_directory())
        download_temp_file(LABEL_URL)

    log.debug('Connecting to scanner')
    with get_scanner_db() as db:
        log.debug('Ingesting video')
        video.add_to_scanner(db)

        bbox_output_name = video.scanner_name() + '_objdet'
        if not db.has_table(bbox_output_name) or not cache:
            log.debug('Registering kernel')

            try:
                db.register_op('ObjDetect', [('frame', ColumnType.Video)], ['bboxes'])
                db.register_python_kernel('ObjDetect', DeviceType.CPU,
                                          SCRIPT_DIR + '/obj_detect_kernel.py')
            except ScannerException:
                pass

            try:
                db.register_op('BboxNMS', ['bboxes'], ['bboxes'])
                db.register_python_kernel('BboxNMS', DeviceType.CPU,
                                          SCRIPT_DIR + '/bbox_nms_kernel.py')
            except ScannerException:
                pass

            frame = db.sources.FrameColumn()
            frame_sampled = frame.sample()
            bboxes = db.ops.ObjDetect(frame=frame_sampled)
            outputs = {'bboxes': bboxes}

            if nms_threshold is not None:
                outputs['nmsed_bboxes'] = db.ops.BboxNMS(bboxes=bboxes, threshold=nms_threshold)

            output = db.sinks.Column(columns=outputs)

            log.debug('Running job')
            job = Job(
                op_args={
                    frame:
                    video.scanner_table(db).column('frame'),
                    frame_sampled:
                    db.sampler.gather(frames) if frames is not None else db.sampler.all(),
                    output:
                    bbox_output_name
                })
            db.run(BulkJob(output=output, jobs=[job]), force=True, pipeline_instances_per_node=1)

        output_table = db.table(bbox_output_name)
        all_bboxes = [
            pickle.loads(box)
            for box in output_table.column('nmsed_bboxes'
                                           if nms_threshold is not None else 'bboxes').load()
        ]

    return all_bboxes


def draw_bboxes(video, bboxes, frames=None, path=None):
    log.debug('Connecting to scanner')
    with get_scanner_db() as db:
        log.debug('Ingesting video')
        video.add_to_scanner(db)

        try:
            db.register_op('BboxDraw', [('frame', ColumnType.Video), 'bboxes'],
                           [('frame', ColumnType.Video)])
            db.register_python_kernel('BboxDraw', DeviceType.CPU,
                                      SCRIPT_DIR + '/bbox_draw_kernel.py')
        except ScannerException:
            pass

        bbox_output_name = video.scanner_name() + '_bboxes_draw'
        db.new_table(
            bbox_output_name, ['bboxes'], [[pickle.dumps(bb)] for bb in bboxes], force=True)

        frame = db.sources.FrameColumn()
        frame_sampled = frame.sample()
        bboxes = db.sources.Column()
        frame_drawn = db.ops.BboxDraw(frame=frame_sampled, bboxes=bboxes)
        output = db.sinks.Column(columns={'frame': frame_drawn})

        log.debug('Running job')
        job = Job(
            op_args={
                frame: video.scanner_table(db).column('frame'),
                frame_sampled: db.sampler.gather(frames)
                if frames is not None else db.sampler.all(),
                bboxes: db.table(bbox_output_name).column('bboxes'),
                output: 'tmp'
            })
        db.run(BulkJob(output=output, jobs=[job]), force=True, pipeline_instances_per_node=1)

        if path is None:
            path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name

        db.table('tmp').column('frame').save_mp4(os.path.splitext(path)[0])
