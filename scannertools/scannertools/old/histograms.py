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
                frame=self._sources['frame'].op, device=self._device, batch=batch)
        }


compute_histograms = HistogramPipeline.make_runner()


class HSVHistogramPipeline(Pipeline):
    job_suffix = 'hsv_hist'
    parser_fn = lambda _: readers.histograms

    def fetch_resources(self):
        cwd = os.path.dirname(os.path.abspath(__file__))

        self._db.load_op(
            os.path.join(cwd, 'cpp_ops/build/libimgproc_op.so'),
            os.path.join(cwd, 'cpp_ops/build/imgproc_pb2.py'))

    def build_pipeline(self, batch=1):
        hsv_frames = self._db.ops.ConvertToHSVCPP(frame=self._sources['frame'].op)

        return {
            'histogram': self._db.ops.Histogram(frame=hsv_frames, device=self._device, batch=batch)
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

    def fetch_resources(self):
        cwd = os.path.dirname(os.path.abspath(__file__))

        self._db.load_op(
            os.path.join(cwd, 'cpp_ops/build/libimgproc_op.so'),
            os.path.join(cwd, 'cpp_ops/build/imgproc_pb2.py'))

    def build_pipeline(self):
        small_frames = self._db.ops.Resize(
                frame=self._sources['frame_sampled'].op,
                device=DeviceType.GPU if self._db.has_gpu() else DeviceType.CPU,
                width=426,
                height=240)

        flow = self._db.ops.OpticalFlow(
                frame=small_frames,
                device=DeviceType.GPU if self._db.has_gpu() else DeviceType.CPU)
        return {
            'flow_hist':
            self._db.ops.FlowHistogram(
                flow=flow,
                device=DeviceType.CPU)
        }


compute_flow_histograms = OpticalFlowHistogramPipeline.make_runner()
