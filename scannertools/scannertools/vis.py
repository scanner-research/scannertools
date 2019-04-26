import scannerpy as sp
from scannerpy import FrameType
from scannerpy.types import BboxList
import numpy as np
import cv2


@sp.register_python_op(name='DrawFlow')
def draw_flow(config, frame: FrameType, flow: FrameType) -> FrameType:
    flow_vis = np.repeat(np.expand_dims(np.average(flow, axis=2), 2), 3, axis=2)
    return np.hstack((frame, (np.clip(flow_vis / np.max(flow_vis), None, 1.0) * 255).astype(
        np.uint8)))


@sp.register_python_op(name='DrawBboxes')
def draw_bboxes(config, frame: FrameType, bboxes: BboxList) -> FrameType:
    [h, w] = frame.shape[:2]
    for bbox in bboxes:
        cv2.rectangle(
            frame,
            (int(bbox.x1 * w), int(bbox.y1 * h)),
            (int(bbox.x2 * w), int(bbox.y2 * h)),
            (255, 0, 0))
    return frame
