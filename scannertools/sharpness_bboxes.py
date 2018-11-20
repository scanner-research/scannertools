from .prelude import Pipeline
from scannerpy.stdlib import readers
from scannerpy import FrameType
import scannerpy
import cv2
import pickle

@scannerpy.register_python_op(name='SharpnessBBox')
def sharpness(config, frame: FrameType, bboxes:bytes) -> bytes:
    bboxes = readers.bboxes(bboxes.self.config.protobufs)

    results = []
    for bbox in bboxes:
        img = frame[int(bbox.y1):int(bbox.y2), int(bbox.x1):int(bbox.x2), :]
        img = cv2.resize(img, (200, 200))
        results.append(cv2.Laplacian(img, cv2.CV_64F).var())

    return pickle.dumps(results)

class SharpnessBBoxPipeline(Pipeline):
    job_suffix = 'sharpness_bbox'
    parser_fn = lambda _: lambda buf, _: pickle.loads(buf)
    additional_sources = ['bboxes']

    def build_pipeline(self):
        return {
            'sharpness_bbox': self._db.ops.SharpnessBBox(
                frame=self._sources['frame_sampled'].op,
                bboxes=self._sources['bboxes'].op)
        }

compute_sharpness_bbox = SharpnessBBoxPipeline.make_runner()

