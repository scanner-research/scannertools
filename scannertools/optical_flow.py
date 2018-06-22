from .prelude import *
import os
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@autobatch(uniforms=[0])
def compute_flow(db, videos, frames=None):
    """
    compute_flow(db, videos, frames=None)
    Computes optical flow on a video.

    Unlike other functions, flow fields aren't materialized into memory as they're simply
    too large.

    Args:
        db (scannerpy.Database): Handle to Scanner database.
        videos (Video, autobatched): Videos to process.
        frames (List[int], autobatched, optional): Frame indices to process.

    Returns:
        str (autobatched): Scanner table name
    """

    log.debug('Ingesting video')
    scanner_ingest(db, videos)

    frame = db.sources.FrameColumn()
    frame_sampled = db.streams.Gather(frame)
    flow = db.ops.OpticalFlow(
        frame=frame_sampled, device=DeviceType.GPU if db.has_gpu() else DeviceType.CPU)
    output = db.sinks.Column(columns={'flow': flow})
    jobs = [
        Job(
            op_args={
                frame: db.table(video.scanner_name()).column('frame'),
                frame_sampled: f if f is not None else list(range(db.table(v.scanner_name()).num_rows())),
                output: video.scanner_name() + '_flow'
            }) for video, f in zip(videos, frames or [None for _ in range(len(videos))])
    ]

    log.debug('Running optical flow Scanner job')
    output_tables = db.run(output, jobs, force=True)

    return [t.name() for t in output_tables]
