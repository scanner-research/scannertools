from .prelude import *

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

