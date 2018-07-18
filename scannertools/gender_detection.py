from .prelude import Pipeline, try_import
from scannerpy.stdlib import readers
from scannerpy.stdlib.util import download_temp_file
from scannerpy import FrameType
import scannerpy
import cv2
import pickle

MODEL_FILE = 'https://storage.googleapis.com/esper/models/rude-carnie/21936.tar.gz'

@scannerpy.register_python_op()
class DetectGender(scannerpy.Kernel):
    def __init__(self, config):
        from carnie_helper import RudeCarnie
        self.config = config
        self.rc = RudeCarnie(model_dir=config.args['model_dir'])

    def execute(self, frame: FrameType, bboxes: bytes) -> bytes:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        [h, w] = frame.shape[:2]
        bboxes = readers.bboxes(bboxes, self.config.protobufs)
        frames = [frame[int(bbox.y1*h):int(bbox.y2*h), int(bbox.x1*w):int(bbox.x2*w)] for bbox in bboxes]
        genders = self.rc.get_gender_batch(frames)
        return pickle.dumps(genders)


class GenderDetectionPipeline(Pipeline):
    job_suffix = 'gender'
    parser_fn = lambda _: lambda buf, _: pickle.loads(buf)
    additional_sources = ['bboxes']
    run_opts = {'pipeline_instances_per_node': 1}

    def fetch_resources(self):
        try_import('carnie_helper', __name__)
        self._model_dir = download_temp_file(MODEL_FILE, untar=True) + '/21936'

    def build_pipeline(self):
        return {'genders': self._db.ops.DetectGender(
            frame=self._sources['frame_sampled'].op,
            bboxes=self._sources['bboxes'].op,
            model_dir=self._model_dir)}


detect_genders = GenderDetectionPipeline.make_runner()
