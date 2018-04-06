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


@autobatch()
def detect_shots(videos):
    log.debug('Connecting to scanner')
    with get_scanner_db() as db:
        log.debug('Ingesting video')
        scanner_ingest(db, videos)

        frame = db.sources.FrameColumn()
        histogram = db.ops.Histogram(
            frame=frame, device=DeviceType.GPU if db.has_gpu() else DeviceType.CPU)
        output = db.sinks.Column(columns={'histogram': histogram})
        jobs = [
            Job(
                op_args={
                    frame: db.table(video.scanner_name()).column('frame'),
                    output: video.scanner_name() + '_hist'
                }) for video in videos
        ]

        log.debug('Running scanner job')
        output_tables = db.run(output, jobs, force=True)

        log.debug('Loading histograms')
        all_hists = [list(t.column('histogram').load(parsers.histograms)) for t in output_tables]

    log.debug('Computing shot boundaries from histograms')
    return [compute_shot_boundaries(vid_hists) for vid_hists in all_hists]
