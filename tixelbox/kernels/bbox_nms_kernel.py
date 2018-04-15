import scannerpy
import scannerpy.stdlib.readers as readers
import scannerpy.stdlib.writers as writers
import scannerpy.stdlib.bboxes as bboxes
from scannerpy.stdlib.util import default
import numpy as np
import pickle


class BboxNMSKernel(scannerpy.Kernel):
    def __init__(self, config, protobufs):
        self._protobufs = protobufs
        self._threshold = default(config.args, 'threshold', 0.3)

    def execute(self, input_columns):
        bboxes_list = []
        for c in input_columns:
            bboxes_list += readers.bboxes(c, self._protobufs)

        nmsed_bboxes = bboxes.nms(bboxes_list, self._threshold)
        return [writers.bboxes(nmsed_bboxes, self._protobufs)]


KERNEL = BboxNMSKernel
