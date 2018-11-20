from .prelude import Pipeline
from scannerpy import FrameType
import scannerpy
import cv2
import pickle

@scannerpy.register_python_op(name='Sharpness')
def sharpness(config, frame: FrameType) -> bytes:
    return pickle.dumps(cv2.Laplacian(frame, cv2.CV_64F).var())

class SharpnessPipeline(Pipeline):
    job_suffix = 'sharpness'
    parser_fn = lambda _: lambda buf, _: pickle.loads(buf)

    def build_pipeline(self):
        return {
            'sharpness':
            self._db.ops.Sharpness(frame=self._sources['frame_sampled'].op)
        }

compute_sharpness = SharpnessPipeline.make_runner()
