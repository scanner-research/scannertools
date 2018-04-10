import PIL.ImageDraw as ImageDraw
import scannerpy
from scannerpy.stdlib import parsers
from scannerpy.stdlib.util import default, temp_directory
from scannerpy.stdlib.bboxes import proto_to_np
import numpy as np
import pickle
from tixelbox import vis_utils
import os
import cv2
import PIL.Image as Image


class BboxDrawKernel(scannerpy.Kernel):
    def __init__(self, config, protobufs):
        categories = vis_utils.parse_labelmap(
            os.path.join(temp_directory(), 'mscoco_label_map.pbtxt'))
        self._category_index = vis_utils.create_category_index(categories)
        self._protobufs = protobufs

    def execute(self, input_columns):
        [frame, bboxes] = input_columns
        bboxes = parsers.bboxes(bboxes, self._protobufs)

        if len(bboxes) == 0:
            return [frame]

        bboxes = proto_to_np(bboxes)
        bboxes[:, [0, 1]] = bboxes[:, [1, 0]]
        bboxes[:, [2, 3]] = bboxes[:, [3, 2]]

        return [
            vis_utils.visualize_boxes_and_labels_on_image_array(
                frame,
                bboxes[:, :4],
                bboxes[:, 5].astype(np.int32),
                bboxes[:, 4],
                self._category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.5)
        ]


KERNEL = BboxDrawKernel
