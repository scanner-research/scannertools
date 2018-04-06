import scannerpy
import pickle
import numpy as np


class FlowDrawKernel(scannerpy.Kernel):
    def execute(self, input_columns):
        [frame, flow] = input_columns
        flow_vis = np.repeat(np.expand_dims(np.average(flow, axis=2), 2), 3, axis=2)
        combined = np.hstack((frame, flow_vis))
        return [combined]


KERNEL = FlowDrawKernel
