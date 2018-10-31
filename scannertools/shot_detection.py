from .prelude import *
from scipy.spatial import distance
import numpy as np
from typing import Sequence
import pickle

WINDOW_SIZE = 500
BOUNDARY_BATCH = 10000000

@scannerpy.register_python_op(name='ShotBoundaries', batch=BOUNDARY_BATCH)
def shot_boundaries(config, histograms: Sequence[bytes]) -> Sequence[bytes]:
    hists = [readers.histograms(byts, config.protobufs) for byts in histograms]

    # Compute the mean difference between each pair of adjacent frames
    diffs = np.array([
        np.mean([distance.chebyshev(hists[i - 1][j], hists[i][j]) for j in range(3)])
        for i in range(1, len(hists))
    ])
    diffs = np.insert(diffs, 0, 0)
    n = len(diffs)

    # Do simple outlier detection to find boundaries between shots
    boundaries = []
    for i in range(1, n):
        window = diffs[max(i - WINDOW_SIZE, 0):min(i + WINDOW_SIZE, n)]
        if diffs[i] - np.mean(window) > 3 * np.std(window):
            boundaries.append(i)

    return [pickle.dumps(boundaries)] + ['\0' for _ in range(len(histograms)-1)]

class ShotBoundaryPipeline(Pipeline):
    job_suffix = 'boundaries'
    base_sources = ['videos', 'histograms']
    run_opts = {'io_packet_size': BOUNDARY_BATCH, 'work_packet_size': BOUNDARY_BATCH, 'pipeline_instances_per_node': 1}

    def build_pipeline(self):
        return {
            'boundaries':
            self._db.ops.ShotBoundaries(
                histograms=self._sources['histograms'].op,
                device=DeviceType.CPU if self._cpu_only else DeviceType.GPU)
        }

    def parse_output(self):
        boundaries = super().parse_output()
        def load(b):
            return pickle.loads(next(b._column.load(rows=[0]))) \
                if b is not None else None
        return par_for(load, boundaries, workers=8)

compute_shot_boundaries = ShotBoundaryPipeline.make_runner()

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

def detect_shots(db, videos, **kwargs):
    hists = compute_histograms(db, videos=videos, **kwargs)
    boundaries = compute_shot_boundaries(db, videos=videos, histograms=hists, **kwargs)
    return boundaries
