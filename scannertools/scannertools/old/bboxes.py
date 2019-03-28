import scannerpy
import scannerpy.stdlib.readers as readers
import scannerpy.stdlib.writers as writers
import scannerpy.stdlib.bboxes as bboxes
from scannerpy.stdlib.util import default


@scannerpy.register_python_op()
class BboxNMS(scannerpy.Kernel):
    def __init__(self, config):
        self._threshold = default(config.args, 'threshold', 0.3)
        self._config = config

    def execute(self, *input_columns) -> bytes:
        bboxes_list = []
        for c in input_columns:
            bboxes_list += readers.bboxes(c, self._config.protobufs)

        nmsed_bboxes = bboxes.nms(bboxes_list, self._threshold)
        return writers.bboxes(nmsed_bboxes, self._config.protobufs)
