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
    run_opts = {'pipeline_instances_per_node': 1}

    def build_pipeline(self):
        return {
            'brightness':
            self._db.ops.Brightness(
                frame=self._sources['frame_sampled'].op)
        }

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
    run_opts = {'pipeline_instances_per_node': 1}

    def build_pipeline(self):
        return {
            'contrast':
            self._db.ops.Contrast(
                frame=self._sources['frame_sampled'].op)
        }

@scannerpy.register_python_op(name='Sharpness')
def sharpness(config, frame: FrameType) -> bytes:
    return pickle.dumps(cv2.Laplacian(frame, cv2.CV_64F).var())

class SharpnessPipeline(Pipeline):
    job_suffix = 'sharpness'
    parser_fn = lambda _: lambda buf, _: pickle.loads(buf)
    run_opts = {'pipeline_instances_per_node': 1}

    def build_pipeline(self):
        return {
            'sharpness':
            self._db.ops.Sharpness(frame=self._sources['frame_sampled'].op)
        }

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
    run_opts = {'pipeline_instances_per_node': 1}

    def build_pipeline(self):
        return {
            'sharpness_bbox': self._db.ops.SharpnessBBox(
                frame=self._sources['frame_sampled'].op,
                bboxes=self._sources['bboxes'].op)
        }

compute_brightness = BrightnessPipeline.make_runner()
compute_contrast = ContrastPipeline.make_runner()
compute_sharpness = SharpnessPipeline.make_runner()
compute_sharpness_bbox = SharpnessBBoxPipeline.make_runner()
