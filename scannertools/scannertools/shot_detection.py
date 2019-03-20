from scipy.spatial import distance
import numpy as np
from typing import Sequence, Any
import scannerpy
from scannerpy.types import Histogram

WINDOW_SIZE = 500
BOUNDARY_BATCH = 10000000


@scannerpy.register_python_op(name='ShotBoundaries', batch=BOUNDARY_BATCH)
def shot_boundaries(config, histograms: Sequence[Histogram]) -> Sequence[Any]:
    # Compute the mean difference between each pair of adjacent frames
    diffs = np.array([
        np.mean([distance.chebyshev(histograms[i - 1][j], histograms[i][j]) for j in range(3)])
        for i in range(1, len(histograms))
    ])
    diffs = np.insert(diffs, 0, 0)
    n = len(diffs)

    # Do simple outlier detection to find boundaries between shots
    boundaries = []
    for i in range(1, n):
        window = diffs[max(i - WINDOW_SIZE, 0):min(i + WINDOW_SIZE, n)]
        if diffs[i] - np.mean(window) > 2.5 * np.std(window):
            boundaries.append(i)

    return [boundaries] + [None for _ in range(len(histograms) - 1)]
