import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# scanner
import scannerpy
from scannerpy import FrameType, DeviceType
from scannerpy.kernel import Kernel
from typing import Sequence, Any

# maskrcnn libs
from caffe2.python import workspace
from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
import detectron.core.test_engine as infer_engine
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

from detectron.utils.colormap import colormap
from detectron.utils.vis import vis_mask, vis_bbox, vis_keypoints

import pycocotools.mask as mask_util


c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


CONFIG_FILE = "/opt/DensePose/configs/DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.yaml"
MODEL_FILE = "https://dl.fbaipublicfiles.com/densepose/DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.pkl"

@scannerpy.register_python_op(device_sets=[[DeviceType.CPU, 0], [DeviceType.GPU, 1]], batch=2)
class DensePoseDetectPerson(Kernel):
    def __init__(self, 
        config,
        confidence_threshold=0.5
    ):
        self.confidence_threshold = confidence_threshold

        # set cpu/gpu
        self.cpu_only = True
        visible_device_list = []
        for handle in config.devices:
            if int(handle.type) == DeviceType.GPU.value:
                visible_device_list.append(handle.id)
                self.cpu_only = False
        if not self.cpu_only:
            print('Using GPU: {}'.format(visible_device_list[0]))
            # torch.cuda.set_device(visible_device_list[0])
            # os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(visible_device_list[0])
            self.gpu_id = visible_device_list[0]
        else:
            print('Using CPU')
        assert self.cpu_only == False, "Densepose does not support CPU"

        # set densepose config
        self.cfg = cfg
        merge_cfg_from_file(CONFIG_FILE)
        self.cfg.NUM_GPUS = 2


    def fetch_resources(self):
        # load model weights (download model weights and cache it)
        cache_url(MODEL_FILE, self.cfg.DOWNLOAD_CACHE)
        assert_and_infer_cfg(cache_urls=False)


    def setup_with_resources(self):
        # load model weights from cache
        model_weights = cache_url(MODEL_FILE, self.cfg.DOWNLOAD_CACHE)
        self.model = infer_engine.initialize_model_from_cfg(model_weights, gpu_id=self.gpu_id)
    

    def convert_result_to_dict(self, cls_boxes, cls_segms, cls_keyps, cls_bodys):
        PERSON_CATEGORY = 1
        boxes = cls_boxes[PERSON_CATEGORY] # N x 5 (x1, y1, x2, y2, score)
        if len(boxes) == 0:
            return []
        segms = cls_segms[PERSON_CATEGORY] # N x {'counts', 'size'}
        keyps = cls_keyps[PERSON_CATEGORY] # N x np.array(4 x 17) (x, y, logit, prob)
        bodys = cls_bodys[PERSON_CATEGORY] # N x np.array(m, n).dtype(uint8) value 0-24
        
        valid_inds = boxes[:, 4] > self.confidence_threshold
        # compact version
        # keyps = [kp[:2].transpose(1, 0) for kp in keyps]
        # result = [[{'bbox': {'x1' : bbox[0], 'y1': bbox[1], 'x2' : bbox[2], 'y2' : bbox[3]},
        #             'mask' : mask, 'keyp': keyp, 'body': body.transpose(1, 2, 0), 'score' : bbox[4]}
        #             for (bbox, mask, keyp, body) in zip(boxes, segms, keyps, bodys) ]]
        # original version
        result = [{'bbox': bbox, 'mask' : mask, 'keyp': keyp, 'body': body, 'score' : bbox[4]}
                    for i, (bbox, mask, keyp, body) in enumerate(zip(boxes, segms, keyps, bodys)) 
                    if valid_inds[i] ]
        return result
    
    # Evaluate densepose model on a frame
    # For each person, return bounding box, keypoints, segmentation mask, densepose and score
    def execute(self, frame: Sequence[FrameType]) -> Sequence[Any]:
        assert len(frame) == 1, "Densepose only support batch_size=1"
        # change image to BGR
        image = frame[0][..., ::-1]

        with c2_utils.NamedCudaScope(self.gpu_id):
            cls_boxes, cls_segms, cls_keyps, cls_bodys = infer_engine.im_detect_all(
                self.model, image, None, None
                )

        result = self.convert_result_to_dict(cls_boxes, cls_segms, cls_keyps, cls_bodys)
       
        return [result]


##################################################################################################
# Visualization Functions                                                                               #
##################################################################################################
# Mostly taken from https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/demo/predictor.py

@scannerpy.register_python_op(name='DrawDensePose')
def draw_densepose(config, frame: FrameType, bundled_data: Any) -> FrameType:
    min_score_thresh = config.args.get('min_score_thresh', 0.7)
    show_box = config.args.get('show_box', True)
    show_keypoint = config.args.get('show_keypoint', True)
    show_mask = config.args.get('show_mask', True)
    show_body = config.args.get('show_body', False)

    if show_body:
        result = visualize_uvbody(frame, bundled_data, min_score_thresh, overlay=True)
    else:
        result = visualize_one_image(frame, bundled_data, min_score_thresh, show_box, show_keypoint, show_mask)
    return result


def visualize_one_image(image, metadata, min_score_thresh=0.5, show_box=True, show_keypoint=True, show_mask=True, kp_thresh=2):
    if len(metadata) == 0:
        return image
    if 'bbox' in metadata[0]:
        boxes = np.array([obj['bbox'] for obj in metadata])
    else:
        return image
    if 'mask' in metadata[0]:
        segms = [obj['mask'] for obj in metadata]
        masks = mask_util.decode(segms)
        color_list = colormap()
        mask_color_id = 0
    else: 
        segms = masks = None
    if 'keyp' in metadata[0]:
        keyps = np.array([obj['keyp'] for obj in metadata])
    else:
        keyps = None

    result = image.copy()
    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < min_score_thresh:
            continue

        # show box (off by default)
        if show_box:
            result = vis_bbox(
                result, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))

        # show keypoints
        if show_keypoint and keyps is not None and len(keyps) > i:
            result = vis_keypoints(result, keyps[i], kp_thresh)

         # show mask
        if show_mask and segms is not None and len(segms) > i:
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1
            result = vis_mask(result, masks[..., i], color_mask)

    return result


def visualize_uvbody(image, metadata, min_score_thresh, overlay=False):
    if len(metadata) == 0 or not 'bbox' in metadata[0] or not 'body' in metadata[0]:
        return image if overlay else np.zeros_like(image)
    boxes = np.array([obj['bbox'] for obj in metadata])
    IUV_fields = [obj['body'] for obj in metadata]
    #
    All_Coords = np.zeros(image.shape)
    # All_inds = np.zeros([im.shape[0],im.shape[1]])
    K = 26
    ##
    inds = np.argsort(boxes[:,4])
    ##
    for i, ind in enumerate(inds):
        entry = boxes[ind,:]
        if entry[4] > min_score_thresh:
            entry=entry[0:4].astype(int)
            ####
            output = IUV_fields[ind]
            ####
            All_Coords_Old = All_Coords[ entry[1] : entry[1]+output.shape[1],entry[0]:entry[0]+output.shape[2],:]
            All_Coords_Old[All_Coords_Old==0]=output.transpose([1,2,0])[All_Coords_Old==0]
            All_Coords[ entry[1] : entry[1]+output.shape[1],entry[0]:entry[0]+output.shape[2],:]= All_Coords_Old
            ###
            # CurrentMask = (output[0,:,:]>0).astype(np.float32)
            # All_inds_old = All_inds[ entry[1] : entry[1]+output.shape[1],entry[0]:entry[0]+output.shape[2]]
            # All_inds_old[All_inds_old==0] = CurrentMask[All_inds_old==0]*i
            # All_inds[ entry[1] : entry[1]+output.shape[1],entry[0]:entry[0]+output.shape[2]] = All_inds_old
    #
    All_Coords[:,:,1:3] = 255. * All_Coords[:,:,1:3]
    All_Coords[All_Coords>255] = 255.
    All_Coords = All_Coords.astype(np.uint8)
    # All_inds = All_inds.astype(np.uint8)
    #
    if overlay:
        result = image.copy()
        result[All_Coords!=0] = All_Coords[All_Coords!=0]
        return result
    else:
        return All_Coords