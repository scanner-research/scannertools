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


class DrawFlowsPipeline(VideoOutputPipeline):
    job_suffix = 'draw_flow'
    additional_sources = ['flows']

    def build_pipeline(self):
        return {
            'frame':
            self._db.ops.FlowDraw(
                frame=self._sources['frame_sampled'].op, flow=self._sources['flows'].op)
        }


draw_flows = DrawFlowsPipeline.make_runner()


@scannerpy.register_python_op()
class BboxDraw(scannerpy.Kernel):
    def __init__(self, config):
        categories = tf_vis_utils.parse_labelmap(config.args['label_path'])
        self._category_index = tf_vis_utils.create_category_index(categories)
        self._config = config

    def execute(self, frame: FrameType, bboxes: bytes) -> FrameType:
        bboxes = readers.bboxes(bboxes, self._config.protobufs)
        if len(bboxes) == 0:
            return frame

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


class DrawBboxesPipeline(VideoOutputPipeline):
    job_suffix = 'draw_bboxes'
    additional_sources = ['bboxes']

    def fetch_resources(self):
        self._label_path = download_temp_file(
            'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt'
        )

    def build_pipeline(self):
        return {
            'frame':
            self._db.ops.BboxDraw(
                frame=self._sources['frame_sampled'].op,
                bboxes=self._sources['bboxes'].op,
                label_path=self._label_path)
        }


draw_bboxes = DrawBboxesPipeline.make_runner()


@scannerpy.register_python_op(name='PoseDraw')
def pose_draw(config, frame: FrameType, poses: bytes) -> FrameType:
    for pose in readers.poses(poses, config.protobufs):
        pose.draw(frame)
    return frame


class DrawPosesPipeline(VideoOutputPipeline):
    job_suffix = 'draw_pose'
    additional_sources = ['poses']

    def build_pipeline(self):
        return {
            'frame':
            self._db.ops.PoseDraw(
                frame=self._sources['frame_sampled'].op, poses=self._sources['poses'].op)
        }


draw_poses = DrawPosesPipeline.make_runner()
