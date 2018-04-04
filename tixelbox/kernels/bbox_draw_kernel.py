import scannerpy
from scannerpy.stdlib.util import default, temp_directory
import numpy as np
import pickle
from tixelbox import vis_utils
import os


class BboxDrawKernel(scannerpy.Kernel):
    def __init__(self, config, protobufs):
        categories = vis_utils.parse_labelmap(
            os.path.join(temp_directory(), 'mscoco_label_map.pbtxt'))
        self._category_index = vis_utils.create_category_index(categories)

    def execute(self, input_columns):
        [frame, bboxes] = input_columns
        bboxes = pickle.loads(bboxes)
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
