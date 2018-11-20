from .prelude import Pipeline
from scannerpy import FrameType
import scannerpy
import cv2
import pickle
import numpy as np

@scannerpy.register_python_op(name='Brightness')
def brightness(config, frame: FrameType) -> bytes:
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)

    # Calculate the mean value of the intensity channel
    brightness = np.mean(frame, axis=(0,1))[0]
    return pickle.dumps(brightness)

class BrightnessPipeline(Pipeline):
    job_suffix = 'brightness'
    parser_fn = lambda _: lambda buf, _: pickle.loads(buf)

    def build_pipeline(self):
        return {
            'brightness':
            self._db.ops.Brightness(
                frame=self._sources['frame_sampled'].op)
        }

compute_brightness = BrightnessPipeline.make_runner()
