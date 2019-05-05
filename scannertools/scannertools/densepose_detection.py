import os
import tarfile
import numpy as np
import sys
import torch
import cv2

# scanner
from scannerpy.util import download_temp_file, temp_directory
import scannerpy
from scannerpy import FrameType, DeviceType, protobufs
from scannerpy.types import BboxList
from scannerpy.kernel import Kernel
from typing import Sequence, Tuple, Any

# maskrcnn libs
from caffe2.python import workspace
from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

import pycocotools.mask as mask_util


c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


CONFIG_FILE = "/opt/DensePose/configs/DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.yaml"
MODEL_FILE = "https://dl.fbaipublicfiles.com/densepose/DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.pkl"

@scannerpy.register_python_op(device_sets=[[DeviceType.CPU, 0], [DeviceType.GPU, 1]], batch=1)
class DensePose(Kernel):
    def __init__(self, 
        config,
    ):
        # set cpu/gpu
        self.cpu_only = True
        visible_device_list = []
        for handle in config.devices:
            if int(handle.type) == DeviceType.GPU.value:
                visible_device_list.append(handle.id)
                self.cpu_only = False
        if not self.cpu_only:
            print('Using GPU: {}'.format(visible_device_list[0]))
            torch.cuda.set_device(visible_device_list[0])
        else:
            print('Using CPU')
        assert(self.cpu_only == False, "Densepose does not support CPU")

        # set densepose config
        self.cfg = cfg
        merge_cfg_from_file(CONFIG_FILE)
        self.cfg.NUM_GPUS = 1



    def fetch_resources(self):
        # load model weights (download model weights and cache it)
        self.model_weights = cache_url(MODEL_FILE, detectron_cfg.DOWNLOAD_CACHE)
        assert_and_infer_cfg(cache_urls=False)


    def setup_with_resources(self):
        # load model weights from cache
        self.model = infer_engine.initialize_model_from_cfg(self.model_weights)
    

    def convert_result_to_dict(cls_boxes, cls_segms, cls_keyps, cls_bodys):
        PERSON_CATEGORY = 1
        boxes = cls_boxes[PERSON_CATEGORY]
        segms = cls_segms[PERSON_CATEGORY]
        keyps = cls_keyps[PERSON_CATEGORY]
        bodys = cls_bodys[PERSON_CATEGORY]
        keyps = [kp[:2].transpose(1, 0) for kp in keyps]
        result = [[{'bbox': {'x1' : bbox[0], 'y1': bbox[1], 'x2' : bbox[2], 'y2' : bbox[3]},
                    'mask' : mask, 'keyp': keyp, 'body': body.transpose(1, 2, 0), 'score' : bbox[4]}
                    for (bbox, mask, keyp, body) in zip(boxes, segms, keyps, bodys) ]]
        return result
    
    # Evaluate densepose model on a frame
    # For each person, return bounding box, keypoints, segmentation mask, densepose and score
    def execute(self, frame: Sequence[FrameType]) -> Sequence[Any]:
        assert(len(frame) == 1, "Densepose only support batch_size=1")
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps, cls_bodys = infer_engine.im_detect_all(
                self.model, frame[0], None, None
                )
        result = self.convert_result_to_dict(cls_boxes, cls_segms, cls_keyps, cls_bodys)
       
        return result


##################################################################################################
# Visualization Functions                                                                               #
##################################################################################################
# Mostly taken from https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/demo/predictor.py

@scannerpy.register_python_op(name='TorchDrawBoxes')
def draw_boxes(config, frame: FrameType, bundled_data: bytes) -> FrameType:
    min_score_thresh = config.args['min_score_thresh']
    metadata = pickle.loads(bundled_data)
    visualize_labels(frame, metadata, min_score_thresh)
    return frame


CATEGORIES = [
        "__background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",]


def visualize_labels(image, metadata, min_score_thresh=0.5, blending_alpha=0.5):
    if len(metadata) == 0:
        return 
    scores = [obj['score'] for obj in metadata]
    boxes = [obj['bbox'] for obj in metadata] if 'bbox' in metadata[0] else [None] * len(metadata)
    masks = [obj['mask'] for obj in metadata] if 'mask' in metadata[0] else [None] * len(metadata)
    labels = [obj['label'] for obj in metadata]
    colors = compute_colors_for_labels(labels).tolist()
    labels = [CATEGORIES[int(i)] for i in labels]
    
    for score, box, mask, label, color in zip(scores, boxes, masks, labels, colors):
        if score < min_score_thresh:
            continue

        # draw bbox 
        if not box is None:  
            if 'x1' in box:
                top_left = (int(box['x1']), int(box['y1'])) 
                bottom_right = (int(box['x2']), int(box['y2']))
            else:
                top_left = (int(box[0]), int(box[1])) 
                bottom_right = (int(box[2]), int(box[3]))
            image = cv2.rectangle(image, top_left, bottom_right, tuple(color), 3)

        
        if not mask is None:
            # H, W =  mask.shape  
            # mask_large = cv2.resize(mask, (W * mask_shrink, H * mask_shrink))
            mask = mask_util.decode([mask])[..., 0]
            # overlay mask
            for c in range(3):
                image[:, :, c] = np.where(mask > 0,
                                          image[:, :, c] * (1 - blending_alpha) + color[c] * blending_alpha,
                                          image[:, :, c])
            # draw mask contour
            # thresh = (mask_large[..., None] > 0).astype(np.uint8) 
            # contours, hierarchy = cv2_util.findContours(
            #     thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            # )
            # image = cv2.drawContours(image, contours, -1, color, 3)

        # draw class name
        if not box is None:
            template = "{}: {:.2f}"
            if 'x1' in box:
                x, y = int(box['x1']), int(box['y1'])
            else:
                x, y = int(box[0]), int(box[1])
            s = template.format(label, score)
            cv2.putText(image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)


def compute_colors_for_labels(labels):
    """
    Simple function that adds fixed colors depending on the class
    """
    palette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = np.array(labels)[:, None] * palette
    colors = (colors % 255).astype("uint8")
    return colors
