from .prelude import *
import pickle
import scannerpy
from scannerpy.stdlib import readers, writers
from scannerpy.stdlib.util import default, temp_directory, download_temp_file
from scannerpy.stdlib.bboxes import proto_to_np
from scannertools import tf_vis_utils
import numpy as np
import os
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@scannerpy.register_python_op(name='FlowDraw')
def flow_draw(config, frame: FrameType, flow: FrameType) -> FrameType:
    flow_vis = np.repeat(np.expand_dims(np.average(flow, axis=2), 2), 3, axis=2)
    return np.hstack((frame, (np.clip(flow_vis / np.max(flow_vis), None, 1.0) * 255).astype(
        np.uint8)))


@autobatch(uniforms=[0])
def draw_flow_fields(db, videos, flow_fields, frames=None, path=None):
    """
    draw_flow_fields(db, videos, flow_fields, frames=None, path=None)
    Draws flow fields next to the original video.

    Args:
        db (scannerpy.Database): Handle to Scanner database.
        videos (Video, autobatched): Video to process.
        flow_fields (str, autobatched): Scanner table name containing flow fields.
        frames (List[int], autobatched, optional): Frame indices to process.
        path (str, optional): Video output path.

    Returns:
        str: Path to output video.
    """

    log.debug('Ingesting video')
    scanner_ingest(db, videos)

    frame = db.sources.FrameColumn()
    frame_sampled = frame.sample()
    flow = db.sources.FrameColumn()
    frame_drawn = db.ops.FlowDraw(frame=frame_sampled, flow=flow)
    output = db.sinks.FrameColumn(columns={'frame': frame_drawn})

    log.debug('Running draw flow Scanner job')
    jobs = [
        Job(
            op_args={
                frame: db.table(v.scanner_name()).column('frame'),
                frame_sampled: db.sampler.gather(f) if f is not None else db.sampler.all(),
                flow: db.table(flow_tab).column('flow'),
                output: v.scanner_name() + '_tmp_frame'
            })
        for v, f, flow_tab in zip(videos, frames or [None for _ in range(len(videos))], flow_fields)
    ]
    db.run(output, jobs, force=True)

    if path is None:
        path = [
            tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
            for _ in range(len(videos))
        ]

    log.debug('Saving output video')
    for (video, p) in zip(videos, path):
        db.table(v.scanner_name() + '_tmp_frame').column('frame').save_mp4(os.path.splitext(p)[0])

    return path


@scannerpy.register_python_op()
class BboxDraw(scannerpy.Kernel):
    def __init__(self, config):
        categories = tf_vis_utils.parse_labelmap(config.args['label_path'])
        self._category_index = tf_vis_utils.create_category_index(categories)
        self._config = config

    def execute(self, frame: FrameType, bboxes: bytes) -> FrameType:
        bboxes = readers.bboxes(bboxes, self._config.protobufs)
        bboxes = proto_to_np(bboxes)
        bboxes[:, [0, 1]] = bboxes[:, [1, 0]]
        bboxes[:, [2, 3]] = bboxes[:, [3, 2]]

        return tf_vis_utils.visualize_boxes_and_labels_on_image_array(
            frame,
            bboxes[:, :4],
            bboxes[:, 5].astype(np.int32),
            bboxes[:, 4],
            self._category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.5)


@autobatch(uniforms=[0])
def draw_bboxes(db, videos, bboxes, frames=None, path=None):
    """
    draw_bboxes(db, videos, bboxes, frames=None, path=None)
    Draws bounding boxes on a video.

    Args:
        db (scannerpy.Database): Handle to Scanner database.
        videos (Video, autobatched): Video to process.
        bboxes (List[List[BoundingBox]], autobatched): Bounding boxes to draw.
        frames (List[int], autobatched, optional): Frame indices to process.
        path (str, optional): Video output path.

    Returns:
        str: Path to output video.
    """

    label_path = download_temp_file(
        'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt'
    )

    log.debug('Ingesting video')
    scanner_ingest(db, videos)

    for (video, vid_bboxes) in zip(videos, bboxes):
        db.new_table(
            video.scanner_name() + '_bboxes_draw', ['bboxes'], [[bb] for bb in vid_bboxes],
            fns=[writers.bboxes],
            force=True)

    frame = db.sources.FrameColumn()
    frame_sampled = frame.sample()
    bboxes = db.sources.Column()
    frame_drawn = db.ops.BboxDraw(frame=frame_sampled, bboxes=bboxes, label_path=label_path)
    output = db.sinks.Column(columns={'frame': frame_drawn})

    log.debug('Running draw bboxes Scanner job')
    jobs = [
        Job(
            op_args={
                frame: db.table(v.scanner_name()).column('frame'),
                frame_sampled: db.sampler.gather(f) if f is not None else db.sampler.all(),
                bboxes: db.table(v.scanner_name() + '_bboxes_draw').column('bboxes'),
                output: v.scanner_name() + '_tmp'
            }) for v, f in zip(videos, frames or [None for _ in range(len(videos))])
    ]
    db.run(output, jobs, force=True)

    if path is None:
        path = [
            tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
            for _ in range(len(videos))
        ]

    log.debug('Saving output video')
    for (video, p) in zip(videos, path):
        db.table(video.scanner_name() + '_tmp').column('frame').save_mp4(os.path.splitext(p)[0])

    return path


@scannerpy.register_python_op(name='PoseDraw')
def pose_draw(config, frame: FrameType, frame_poses: bytes) -> FrameType:
    for all_pose in readers.poses(frame_poses, config.protobufs):
        pose = all_pose.pose_keypoints()
        for i in range(18):
            if pose[i, 2] < 0.35: continue
            x = int(pose[i, 0] * frame.shape[1])
            y = int(pose[i, 1] * frame.shape[0])
            cv2.circle(frame, (x, y), 8, (255, 0, 0), 3)
    return frame


@autobatch(uniforms=[0])
def draw_poses(db, videos, poses, frames=None, path=None):
    log.debug('Ingesting video')
    video.add_to_scanner(db)

    poses_output_name = video.scanner_name() + '_poses_draw'
    db.new_table(poses_output_name, ['poses'], [[p] for p in poses], fn=writers.poses, force=True)

    frame = db.sources.FrameColumn()
    frame_sampled = frame.sample()
    poses = db.sources.Column()
    frame_drawn = db.ops.PoseDraw(frame=frame_sampled, poses=poses)
    output = db.sinks.Column(columns={'frame': frame_drawn})

    log.debug('Running job')
    job = Job(
        op_args={
            frame: video.scanner_table(db).column('frame'),
            frame_sampled: db.sampler.gather(frames) if frames is not None else db.sampler.all(),
            poses: db.table(poses_output_name).column('poses'),
            output: 'tmp'
        })
    db.run(BulkJob(output=output, jobs=[job]), force=True)

    if path is None:
        path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name

    db.table('tmp').column('frame').save_mp4(os.path.splitext(path)[0])
