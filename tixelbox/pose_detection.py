from prelude import *
from scannerpy.stdlib.util import temp_directory, download_temp_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@autobatch(uniforms=[0, 'cache', 'models_path', 'batch'])
def detect_poses(db, video, frames=None, cache=False, models_path=None, batch=1):
    """
    detect_poses(video, frames=None, cache=False, models_path=None, batch=1)
    WIP, don't use
    """

    if models_path is None:
        models_path = os.path.join(temp_directory(), 'openpose')
        pose_fs_url = 'http://posefs1.perception.cs.cmu.edu/OpenPose/models/'
        # Pose prototxt
        download_temp_file('https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/'
                           'openpose/master/models/pose/coco/pose_deploy_linevec.prototxt',
                           'openpose/pose/coco/pose_deploy_linevec.prototxt')
        # Pose model weights
        download_temp_file(
            os.path.join(pose_fs_url, 'pose/coco/pose_iter_440000.caffemodel'),
            'openpose/pose/coco/pose_iter_440000.caffemodel')
        # Hands prototxt
        download_temp_file('https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/'
                           'openpose/master/models/hand/pose_deploy.prototxt',
                           'openpose/hand/pose_deploy.prototxt')
        # Hands model weights
        download_temp_file(
            os.path.join(pose_fs_url, 'hand/pose_iter_102000.caffemodel'),
            'openpose/hand/pose_iter_102000.caffemodel')
        # Face prototxt
        download_temp_file('https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/'
                           'openpose/master/models/face/pose_deploy.prototxt',
                           'openpose/face/pose_deploy.prototxt')
        # Face model weights
        download_temp_file(
            os.path.join(pose_fs_url, 'face/pose_iter_116000.caffemodel'),
            'openpose/face/pose_iter_116000.caffemodel')
        # Face haar cascades
        download_temp_file('https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/'
                           'openpose/master/models/face/haarcascade_frontalface_alt.xml',
                           'openpose/face/haarcascade_frontalface_alt.xml')

    log.debug('Ingesting video')
    video.add_to_scanner(db)

    pose_output_name = video.scanner_name() + '_pose'
    if not db.has_table(pose_output_name) or not cache:
        pose_args = db.protobufs.OpenPoseArgs()
        pose_args.model_directory = models_path
        pose_args.pose_num_scales = 3
        pose_args.pose_scale_gap = 0.33
        pose_args.hand_num_scales = 4
        pose_args.hand_scale_gap = 0.4

        if db.has_gpu():
            device = DeviceType.GPU
            pipeline_instances = -1
        else:
            raise Exception("Pose detection pipeline currently requires a GPU to run.")
            device = DeviceType.CPU
            pipeline_instances = 1

        frame = db.sources.FrameColumn()
        poses_out = db.ops.OpenPose(frame=frame, device=device, args=pose_args, batch=batch)
        sampled_poses = poses_out.sample()
        output = db.sinks.Column(columns={'poses': sampled_poses})

        job = Job(
            op_args={
                frame: video.scanner_table(db).column('frame'),
                sampled_poses: db.sampler.gather(frames)
                if frames is not None else db.sampler.all(),
                output: pose_output_name
            })

        output = db.run(
            BulkJob(output=output, jobs=[job]),
            force=True,
            work_packet_size=8,
            pipeline_instances_per_node=pipeline_instances)

    return list(db.table(pose_output_name).column('poses').load(parsers.poses))


def draw_poses(db, video, poses, frames=None, path=None):
    log.debug('Ingesting video')
    video.add_to_scanner(db)

    try:
        db.register_op('PoseDraw', [('frame', ColumnType.Video), 'poses'],
                       [('frame', ColumnType.Video)])
        db.register_python_kernel('PoseDraw', DeviceType.CPU,
                                  SCRIPT_DIR + '/kernels/pose_draw_kernel.py')
    except ScannerException:
        pass

    poses_output_name = video.scanner_name() + '_poses_draw'
    db.new_table(poses_output_name, ['poses'], [[p] for p in poses], fn=writers.poses, force=True)

    frame = db.sources.FrameColumn()
    frame_sampled = frame.sample()
    poses = db.sources.Column()
    frame_drawn = db.ops.PoseDraw(frame=frame_sampled, poses=poses)
    output = db.sinks.Column(columns={'frame': frame_drawn})

    log.debug('Running job')
    job = Job(
        op_args={
            frame: video.scanner_table(db).column('frame'),
            frame_sampled: db.sampler.gather(frames) if frames is not None else db.sampler.all(),
            poses: db.table(poses_output_name).column('poses'),
            output: 'tmp'
        })
    db.run(BulkJob(output=output, jobs=[job]), force=True)

    if path is None:
        path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name

    db.table('tmp').column('frame').save_mp4(os.path.splitext(path)[0])
