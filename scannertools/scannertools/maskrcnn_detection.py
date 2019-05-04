import os
import tarfile
import numpy as np
import sys
import torch
import cv2
from torchvision import transforms as T

# scanner
from scannerpy.util import download_temp_file, temp_directory
import scannerpy
from scannerpy import FrameType, DeviceType, protobufs
from scannerpy.types import BboxList
from scannerpy.kernel import Kernel
from typing import Sequence, Tuple, Any

# maskrcnn libs
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.utils import cv2_util

import pycocotools.mask as mask_util


CONFIG_FILE = "/opt/maskrcnn-benchmark/configs/caffe2/e2e_mask_rcnn_X_101_32x8d_FPN_1x_caffe2.yaml"

@scannerpy.register_python_op(device_sets=[[DeviceType.CPU, 0], [DeviceType.GPU, 1]], batch=5)
class MaskRCNNDetectObjects(Kernel):
    def __init__(self, 
        config,
    ):
        self.confidence_threshold = 0.5
        self.min_image_size = 800
        # self.mask_shrink = 4
        
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

        # set maskrcnn config
        cfg.merge_from_file(CONFIG_FILE)
        if self.cpu_only:
            cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
        else:
            cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
        cfg.merge_from_list(["INPUT.TO_BGR255", False])
        self.cfg = cfg.clone()

        # build model and transform
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.transforms = self.build_transform()
        self.masker = Masker(threshold=0.5, padding=1)
        self.cpu_device = torch.device("cpu")


    def fetch_resources(self):
        # load model weights (download model weights and cache it)
        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(self.cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(self.cfg.MODEL.WEIGHT)


    def setup_with_resources(self):
        # load model weights from cache
        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(self.cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(self.cfg.MODEL.WEIGHT)
    
    
    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]] * 255)

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.min_image_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform


    # Evaluate object detection DNN model on a frame
    # Return bounding box position, class and score
    def execute(self, frame: Sequence[FrameType]) -> Sequence[Any]:
        # apply pre-processing to image
        frame_trans = [self.transforms(img) for img in frame]
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(frame_trans, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # reshape predictiona (a BoxList) into the original image size
        height, width = frame[0].shape[:-1]
        predictions = [pred.resize((width, height)) for pred in predictions]

        for pred in predictions:
            if pred.has_field("mask"):
                # if we have masks, paste the masks in the right position
                # in the image, as defined by the bounding boxes
                masks = pred.get_field("mask")
                # always single image is passed at a time
                masks = self.masker([masks], [pred])[0]
                pred.add_field("mask", masks)

        top_predictions = []
        # filter predictions by confidence threshold
        for pred in predictions:
            scores = pred.get_field("scores")
            keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
            pred = pred[keep]
            scores = pred.get_field("scores")
            _, idx = scores.sort(0, descending=True)
            top_predictions += [pred[idx]]

        # def resize_mask(mask):
        #     H, W = mask.shape
        #     mask_small = cv2.resize(mask, (W // self.mask_shrink, H // self.mask_shrink))
        #     mask_small = (mask_small > 0).astype(np.uint8) * 255
        #     return mask_small

        def encode_mask(mask):
            return mask_util.encode(np.asfortranarray(mask.transpose(1, 2, 0)))[0]

        result = [[{'bbox': {'x1' : float(bbox[0]), 'y1': float(bbox[1]), 'x2' : float(bbox[2]), 'y2' : float(bbox[3])},
                'mask' : encode_mask(mask.numpy()), 
                'label' : float(label), 'score' : float(score)}
                for (bbox, mask, label, score) in zip(pred.bbox, pred.get_field("mask"), pred.get_field("labels"), pred.get_field("scores")) ]
                for pred in top_predictions]
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
