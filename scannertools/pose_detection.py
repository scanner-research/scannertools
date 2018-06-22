from .prelude import *
from scannerpy.stdlib.util import temp_directory, download_temp_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@autobatch(uniforms=[0, 'cache', 'models_path', 'batch'])
def detect_poses(db, videos, frames=None, models_path=None, batch=1):
    """
    detect_poses(db, videos, frames=None, models_path=None, batch=1)
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

    log.debug('Ingesting videos')
    scanner_ingest(db, videos)

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

    frame = db.sources.FrameColumn()
    poses_out = db.ops.OpenPose(frame=frame, device=device, args=pose_args, batch=batch)
    sampled_poses = db.streams.Gather(poses_out)
    output = db.sinks.Column(columns={'poses': sampled_poses})

    jobs = [
        Job(
            op_args={
                frame: db.table(v.scanner_name()).column('frame'),
                sampled_poses: f if f is not None else list(range(db.table(v.scanner_name()).num_rows())),
                output: v.scanner_name() + '_pose'
            }) for v, f in zip(videos, frames or [None for _ in range(len(videos))])
    ]

    output_tables = db.run(
        output,
        jobs,
        force=True,
        work_packet_size=8,
        pipeline_instances_per_node=pipeline_instances)

    return [list(t.column('poses').load(readers.poses)) for t in output_tables]
