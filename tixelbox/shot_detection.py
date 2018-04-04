from prelude import *
from scipy.spatial import distance
import numpy as np
import cv2

WINDOW_SIZE = 500


def compute_shot_boundaries(hists):
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
    return boundaries


def detect_shots(video, cache=False):
    log.debug('Connecting to scanner')
    with get_scanner_db() as db:
        log.debug('Ingesting video')
        video.add_to_scanner(db)

        hist_output_name = video.scanner_name() + '_hist'
        if not db.has_table(hist_output_name) or not cache:
            frame = db.sources.FrameColumn()
            histogram = db.ops.Histogram(frame=frame)
            output = db.sinks.Column(columns={'histogram': histogram})
            job = Job(op_args={
                frame: video.scanner_table(db).column('frame'),
                output: hist_output_name
            })
            db.run(BulkJob(output=output, jobs=[job]), force=True)

        hists = list(db.table(hist_output_name).column('histogram').load(parsers.histograms))

    return compute_shot_boundaries(hists)
