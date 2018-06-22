from .prelude import *
from . import bboxes
from scannerpy import ScannerException
from scannerpy.stdlib import NetDescriptor, readers
from scannerpy.stdlib.util import download_temp_file, temp_directory
from scannerpy.stdlib.bboxes import proto_to_np
import os
import tarfile
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@autobatch(uniforms=[0])
def detect_faces(db, videos, frames=None):
    """
    detect_faces(db, videos, frames=None)
    Detects faces in a video.

    Args:
        db (scannerpy.Database): Handle to Scanner database.
        videos (Video, autobatched): Videos to process.
        frames (List[int], autobatched, optional): Frame indices to process.

    Returns:
        List[List[BoundingBox]] (autobatched): List of bounding boxes for each frame.
    """

    prototxt_path = download_temp_file(
        'https://storage.googleapis.com/scanner-data/nets/caffe_facenet/facenet_deploy.prototxt')
    model_weights_path = download_temp_file(
        'https://storage.googleapis.com/scanner-data/nets/caffe_facenet/facenet_deploy.caffemodel')
    templates_path = download_temp_file(
        'https://storage.googleapis.com/scanner-data/nets/caffe_facenet/facenet_templates.bin')

    log.debug('Ingesting video')
    scanner_ingest(db, videos)

    descriptor = NetDescriptor(db)
    descriptor.model_path = prototxt_path
    descriptor.model_weights_path = model_weights_path
    descriptor.input_layer_names = ['data']
    descriptor.output_layer_names = ['score_final']
    descriptor.mean_colors = [119.29959869, 110.54627228, 101.8384321]

    facenet_args = db.protobufs.FacenetArgs()
    facenet_args.templates_path = templates_path
    facenet_args.threshold = 0.5
    caffe_args = facenet_args.caffe_args
    caffe_args.net_descriptor.CopyFrom(descriptor.as_proto())

    if db.has_gpu():
        device = DeviceType.GPU
        pipeline_instances = -1
    else:
        device = DeviceType.CPU
        pipeline_instances = 1

    input_frame_columns = [db.table(v.scanner_name()).column('frame') for v in videos]
    output_names = ['{}_face_bboxes'.format(v.scanner_name()) for v in videos]
    output_samplings = frames if frames is not None else [
        list(range(db.table(v.scanner_name()))) for v in videos
    ]

    outputs = []
    scales = [1.0, 0.5, 0.25, 0.125]
    batch_sizes = [int((2**i)) for i in range(len(scales))]
    for scale, batch in zip(scales, batch_sizes):
        facenet_args.scale = scale
        caffe_args.batch_size = batch

        frame = db.sources.FrameColumn()
        frame_info = db.ops.InfoFromFrame(frame=frame)
        facenet_input = db.ops.FacenetInput(frame=frame, args=facenet_args, device=device)
        facenet = db.ops.Facenet(facenet_input=facenet_input, args=facenet_args, device=device)
        facenet_output = db.ops.FacenetOutput(
            facenet_output=facenet, original_frame_info=frame_info, args=facenet_args)
        sampled_output = db.streams.Gather(facenet_output)
        output = db.sinks.Column(columns={'bboxes': sampled_output})

        jobs = []
        for output_name, frame_column, output_sampling in zip(output_names, input_frame_columns,
                                                              output_samplings):
            job = Job(
                op_args={
                    frame: frame_column,
                    sampled_output: output_sampling,
                    output: '{}_{}'.format(output_name, scale)
                })
            jobs.append(job)

        log.debug('Running face detection (scale {}) Scanner job'.format(scale))
        output = db.run(
            output,
            jobs,
            force=True,
            work_packet_size=batch * 4,
            io_packet_size=batch * 20,
            pipeline_instances_per_node=pipeline_instances)

        output = [db.table('{}_{}'.format(output_name, scale)) for output_name in output_names]
        outputs.append(output)

    bbox_inputs = [db.sources.Column() for _ in outputs]
    nmsed_bboxes = db.ops.BboxNMS(*bbox_inputs, threshold=0.1)
    output = db.sinks.Column(columns={'nmsed_bboxes': nmsed_bboxes})

    jobs = []
    for i in range(len(input_frame_columns)):
        op_args = {}
        for bi, cols in enumerate(outputs):
            op_args[bbox_inputs[bi]] = cols[i].column('bboxes')
        op_args[output] = output_names[i]
        jobs.append(Job(op_args=op_args))

    log.debug('Running bbox nms Scanner job')
    output_tables = db.run(output, jobs, force=True)

    output_tables = [db.table(n) for n in output_names]
    all_bboxes = [
        list(output_table.column('nmsed_bboxes').load(readers.bboxes))
        for output_table in output_tables
    ]

    for vid, vid_bb in zip(videos, all_bboxes):
        for frame_bb in vid_bb:
            for bb in frame_bb:
                bb.label = 1
                bb.x1 /= vid.width()
                bb.x2 /= vid.width()
                bb.y1 /= vid.height()
                bb.y2 /= vid.height()

    return all_bboxes
