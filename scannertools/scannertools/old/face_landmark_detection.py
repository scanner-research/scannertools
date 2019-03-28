from .prelude import Pipeline
from scannerpy import FrameType, DeviceType
from scannerpy.stdlib import readers
import scannerpy
import pickle
import numpy as np

# NOTE: This pipeline will *NOT* work out of the box at the moment
# This pipeline relies on torch > 0.4, but other parts of scannertools rely
#   on torch = 0.3.1 (clothing detection in particular)
# Until then, you'll have to manually change the install version


@scannerpy.register_python_op(device_sets=[[DeviceType.CPU, 0], [DeviceType.GPU, 1]])
class DetectFaceLandmarks(scannerpy.Kernel):
    def __init__(self, config):
        import face_alignment

        self.config = config
        cpu_only = True
        for handle in config.devices:
            if handle.type == DeviceType.GPU.value:
                cpu_only = False
                break

        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D,
            device=('cpu' if cpu_only else 'cuda'),
            flip_input=False)

    def execute(self, frame: FrameType, bboxes: bytes) -> bytes:
        [h, w] = frame.shape[:2]
        bboxes = readers.bboxes(bboxes, self.config.protobufs)
        if len(bboxes) == 0:
            return pickle.dumps([])

        # This returns a numpy array of size (68, 2) for every bbox)
        predictions = self.fa.get_landmarks_from_image(
            frame,
            detected_faces=[(bbox.x1 * w, bbox.y1 * h, bbox.x2 * w, bbox.y2 * h)
                            for bbox in bboxes])

        predictions = [
            np.array([[width / w, height / h] for [width, height] in prediction])
            for prediction in predictions
        ]

        return pickle.dumps(predictions)


class FaceLandmarkDetectionPipeline(Pipeline):
    job_suffix = 'face_landmarks'
    parser_fn = lambda _: lambda buf, _: pickle.loads(buf)
    additional_sources = ['bboxes']
    run_opts = {'pipeline_instances_per_node': 1}

    def build_pipeline(self):
        return {
            'face_landmarks':
            self._db.ops.DetectFaceLandmarks(
                frame=self._sources['frame_sampled'].op,
                bboxes=self._sources['bboxes'].op,
                device=self._device)
        }


detect_face_landmarks = FaceLandmarkDetectionPipeline.make_runner()
