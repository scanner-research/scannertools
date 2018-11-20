from .prelude import Pipeline
from scannerpy import FrameType
import scannerpy
import cv2
import pickle
import numpy as np

@scannerpy.register_python_op(name='Contrast')
def contrast(config, frame: FrameType) -> bytes:
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)

    (h, w, c) = frame.shape
    intensities = frame.reshape((h * w * c))[::3]

    # Calculate the average intensity
    average_intensity = np.mean(intensities)
    contrast = np.sqrt(np.mean((intensities - average_intensity) ** 2))
    return pickle.dumps(contrast)

class ContrastPipeline(Pipeline):
    job_suffix = 'contrast'
    parser_fn = lambda _: lambda buf, _: pickle.loads(buf)

    def build_pipeline(self):
        return {
            'contrast':
            self._db.ops.Contrast(
                frame=self._sources['frame_sampled'].op)
        }

compute_contrast = ContrastPipeline.make_runner()

