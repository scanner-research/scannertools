from .prelude import *
import os
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class OpticalFlowPipeline(Pipeline):
    """
    Computes optical flow on a video.

    Unlike other functions, flow fields aren't materialized into memory as they're simply
    too large.
    """

    job_suffix = 'flow'
    parser_fn = lambda _: lambda x: x

    def build_pipeline(self):
        return {
            'flow':
            self._db.ops.OpticalFlow(frame=self._sources['frame_sampled'].op, device=self._device)
        }


compute_flow = OpticalFlowPipeline.make_runner()
