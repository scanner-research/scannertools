from prelude import *
from scannerpy.stdlib import writers
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@autobatch(uniforms=[0])
def draw_flow_fields(db, videos, flow_fields, frames=None, path=None):
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
        for v, f, flow_tab in zip(videos, frames or [None for _ in range(len(videos))], flow_fields)
    ]
    db.run(output, jobs, force=True)

    if path is None:
        path = [
            tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
            for _ in range(len(videos))
        ]

    log.debug('Saving output video')
    for (video, p) in zip(videos, path):
        db.table(v.scanner_name() + '_tmp_frame').column('frame').save_mp4(os.path.splitext(p)[0])

    return path


@autobatch(uniforms=[0])
def draw_bboxes(db, videos, bboxes, frames=None, path=None):
    log.debug('Ingesting video')
    scanner_ingest(db, videos)

    try:
        db.register_op('BboxDraw', [('frame', ColumnType.Video), 'bboxes'],
                       [('frame', ColumnType.Video)])
        db.register_python_kernel('BboxDraw', DeviceType.CPU,
                                  SCRIPT_DIR + '/kernels/bbox_draw_kernel.py')
    except ScannerException:
        pass

    for (video, vid_bboxes) in zip(videos, bboxes):
        db.new_table(
            video.scanner_name() + '_bboxes_draw', ['bboxes'], [[bb] for bb in vid_bboxes],
            fns=[writers.bboxes],
            force=True)

    frame = db.sources.FrameColumn()
    frame_sampled = frame.sample()
    bboxes = db.sources.Column()
    frame_drawn = db.ops.BboxDraw(frame=frame_sampled, bboxes=bboxes)
    output = db.sinks.Column(columns={'frame': frame_drawn})

    log.debug('Running draw bboxes Scanner job')
    jobs = [
        Job(
            op_args={
                frame: db.table(v.scanner_name()).column('frame'),
                frame_sampled: db.sampler.gather(f) if f is not None else db.sampler.all(),
                bboxes: db.table(v.scanner_name() + '_bboxes_draw').column('bboxes'),
                output: v.scanner_name() + '_tmp'
            }) for v, f in zip(videos, frames or [None for _ in range(len(videos))])
    ]
    db.run(output, jobs, force=True)

    if path is None:
        path = [
            tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
            for _ in range(len(videos))
        ]

    log.debug('Saving output video')
    for (video, p) in zip(videos, path):
        db.table(v.scanner_name() + '_tmp').column('frame').save_mp4(os.path.splitext(p)[0])

    return path
