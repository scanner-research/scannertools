from prelude import *
import os
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@autobatch()
def compute_flow(videos, frames=None):
    log.debug('Connecting to scanner')
    with get_scanner_db() as db:
        log.debug('Ingesting video')
        scanner_ingest(db, videos)

        frame = db.sources.FrameColumn()
        frame_sampled = frame.sample()
        flow = db.ops.OpticalFlow(
            frame=frame_sampled, device=DeviceType.GPU if db.has_gpu() else DeviceType.CPU)
        output = db.sinks.Column(columns={'flow': flow})
        jobs = [
            Job(
                op_args={
                    frame: db.table(video.scanner_name()).column('frame'),
                    frame_sampled: db.sampler.gather(f) if f is not None else db.sampler.all(),
                    output: video.scanner_name() + '_flow'
                }) for video, f in zip(videos, frames or [None for _ in range(len(videos))])
        ]

        log.debug('Running optical flow Scanner job')
        # output_tables = db.run(output, jobs, force=True)
        output_tables = [db.table(video.scanner_name() + '_flow') for video in videos]

        return [t.name() for t in output_tables]


@autobatch()
def draw_flow_fields(videos, flow_fields, frames=None, path=None):
    log.debug('Connecting to scanner')
    with get_scanner_db() as db:
        log.debug('Ingesting video')
        scanner_ingest(db, videos)

        try:
            db.register_op('FlowDraw', [('frame', ColumnType.Video), ('flow', ColumnType.Video)],
                           [('frame', ColumnType.Video)])
            db.register_python_kernel('FlowDraw', DeviceType.CPU,
                                      SCRIPT_DIR + '/kernels/flow_draw_kernel.py')
        except ScannerException:
            pass

        frame = db.sources.FrameColumn()
        frame_sampled = frame.sample()
        flow = db.sources.FrameColumn()
        frame_drawn = db.ops.FlowDraw(frame=frame_sampled, flow=flow)
        output = db.sinks.FrameColumn(columns={'frame': frame_drawn})

        log.debug('Running draw flow Scanner job')
        jobs = [
            Job(
                op_args={
                    frame: db.table(v.scanner_name()).column('frame'),
                    frame_sampled: db.sampler.gather(f) if f is not None else db.sampler.all(),
                    flow: db.table(flow_tab).column('flow'),
                    output: v.scanner_name() + '_tmp_frame'
                })
            for v, f, flow_tab in zip(videos, frames or [None
                                                         for _ in range(len(videos))], flow_fields)
        ]
        db.run(output, jobs, force=True)

        if path is None:
            path = [
                tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
                for _ in range(len(videos))
            ]

        log.debug('Saving output video')
        for (video, p) in zip(videos, path):
            db.table(v.scanner_name() + '_tmp_frame').column('frame').save_mp4(
                os.path.splitext(p)[0])

    return path
