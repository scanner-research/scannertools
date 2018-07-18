from .prelude import *
from scannerpy.stdlib.util import temp_directory, download_temp_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class PoseDetectionPipeline(Pipeline):
    job_suffix = 'pose'
    parser_fn = lambda _: readers.poses

    def fetch_resources(self):
        self._models_path = os.path.join(temp_directory(), 'openpose')
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

    def build_pipeline(self):
        if not self._db.has_gpu():
            raise Exception("Pose detection requires GPU to run")

        pose_args = self._db.protobufs.OpenPoseArgs()
        pose_args.model_directory = self._models_path
        pose_args.pose_num_scales = 3
        pose_args.pose_scale_gap = 0.33
        pose_args.hand_num_scales = 4
        pose_args.hand_scale_gap = 0.4

        return {'poses': self._db.ops.OpenPose(
            frame=self._sources['frame_sampled'].op,
            device=DeviceType.GPU,
            args=pose_args,
            batch=5)}

detect_poses = PoseDetectionPipeline.make_runner()
