from .prelude import *
import numpy as np
import os

class HistogramPipeline(Pipeline):
    job_suffix = 'hist'
    parser_fn = lambda _: readers.histograms

    def build_pipeline(self, batch=1):
        return {
            'histogram':
            self._db.ops.Histogram(
                frame=self._sources['frame'].op,
                device=DeviceType.CPU if self._cpu_only else DeviceType.GPU,
                batch=batch)
        }

compute_histograms = HistogramPipeline.make_runner()

class HSVHistogramPipeline(Pipeline):
    job_suffix = 'hsv_hist'
    parser_fn = lambda _: readers.histograms
    run_opts = {'pipeline_instances_per_node': 1}

    def build_pipeline(self, batch=1):
        hsv_frames = self._db.ops.ConvertToHSV(frame=self._sources['frame'].op)

        return {
            'histogram':
            self._db.ops.Histogram(
                frame=hsv_frames,
                device=DeviceType.CPU if self._cpu_only else DeviceType.GPU,
                batch=batch)
        }

compute_hsv_histograms = HSVHistogramPipeline.make_runner()

def flow_hist_reader(buf, protobufs):
    if buf is None:
        return None
    return np.split(np.frombuffer(buf, dtype=np.dtype(np.int32)), 2)

class OpticalFlowHistogramPipeline(Pipeline):
    """
    Computes histograms of optical flow on a video.
    """
    job_suffix = 'flow_hist'
    parser_fn = lambda _: flow_hist_reader
    run_opts = { 'pipeline_instances_per_node': 1 }

    def fetch_resources(self):
        cwd = os.path.dirname(os.path.abspath(__file__))

        self._db.load_op(
            os.path.join(cwd, 'cpp_ops/build/libimgproc_op.so'),
            os.path.join(cwd, 'cpp_ops/build/imgproc_pb2.py'))

    def build_pipeline(self):
        flow = self._db.ops.OpticalFlow(
                frame=self._sources['frame_sampled'].op,
                device=DeviceType.GPU if self._db.has_gpu() else DeviceType.CPU)
        return {
            'flow_hist':
            self._db.ops.FlowHistogram(
                flow=flow,
                device=DeviceType.CPU)
        }

compute_flow_histograms = OpticalFlowHistogramPipeline.make_runner()

