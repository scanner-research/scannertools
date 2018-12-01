from .prelude import Pipeline
from scannerpy import FrameType
import scannerpy
import cv2
import pickle
import numpy as np
import struct
import os

@scannerpy.register_python_op(name='Brightness')
def brightness(config, frame: FrameType) -> bytes:
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)

    # Calculate the mean value of the intensity channel
    brightness = np.mean(frame, axis=(0,1))[0]
    return pickle.dumps(brightness)

@scannerpy.register_python_op(name='Contrast')
def contrast(config, frame: FrameType) -> bytes:
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)

    (h, w, c) = frame.shape
    intensities = frame.reshape((h * w * c))[::3]

    # Calculate the average intensity
    average_intensity = np.mean(intensities)
    contrast = np.sqrt(np.mean((intensities - average_intensity) ** 2))
    return pickle.dumps(contrast)

@scannerpy.register_python_op(name='Sharpness')
def sharpness(config, frame: FrameType) -> bytes:
    sharpness = cv2.Laplacian(frame, cv2.CV_64F).var()
    return pickle.dumps(sharpness)

@scannerpy.register_python_op(name='ConvertToHSV')
def convert_to_hsv(config, frame: FrameType) -> FrameType:
    return cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

@scannerpy.register_python_op(name='SharpnessBBox')
def sharpness_bbox(config, frame: FrameType, bboxes:bytes) -> bytes:
    bboxes = readers.bboxes(bboxes.self.config.protobufs)

    results = []
    for bbox in bboxes:
        img = frame[int(bbox.y1):int(bbox.y2), int(bbox.x1):int(bbox.x2), :]
        img = cv2.resize(img, (200, 200))
        results.append(cv2.Laplacian(img, cv2.CV_64F).var())

    return pickle.dumps(results)

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

class BrightnessCPPPipeline(Pipeline):
    job_suffix = 'brightness_cpp'
    parser_fn = lambda _: lambda buf, _: struct.unpack('f', buf)[0]

    def build_pipeline(self):
        cwd = os.path.dirname(os.path.abspath(__file__))

        self._db.load_op(
            os.path.join(cwd, 'cpp_ops/build/libimgproc_op.so'),
            os.path.join(cwd, 'cpp_ops/build/imgproc_pb2.py'))

        return {
            'brightness':
            self._db.ops.BrightnessCPP(
                frame=self._sources['frame_sampled'].op)
        }

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

class ContrastCPPPipeline(Pipeline):
    job_suffix = 'contrast_cpp'
    parser_fn = lambda _: lambda buf, _: struct.unpack('f', buf)[0]

    def build_pipeline(self):
        cwd = os.path.dirname(os.path.abspath(__file__))

        self._db.load_op(
            os.path.join(cwd, 'cpp_ops/build/libimgproc_op.so'),
            os.path.join(cwd, 'cpp_ops/build/imgproc_pb2.py'))

        return {
            'contrast':
            self._db.ops.ContrastCPP(
                frame=self._sources['frame_sampled'].op)
        }

class SharpnessPipeline(Pipeline):
    job_suffix = 'sharpness'
    parser_fn = lambda _: lambda buf, _: pickle.loads(buf)
    run_opts = {'pipeline_instances_per_node': 1}

    def build_pipeline(self):
        return {
            'sharpness':
            self._db.ops.Sharpness(frame=self._sources['frame_sampled'].op)
        }

class SharpnessCPPPipeline(Pipeline):
    job_suffix = 'sharpness_cpp'
    parser_fn = lambda _: lambda buf, _: struct.unpack('f', buf)[0]

    def build_pipeline(self):
        cwd = os.path.dirname(os.path.abspath(__file__))

        self._db.load_op(
            os.path.join(cwd, 'cpp_ops/build/libimgproc_op.so'),
            os.path.join(cwd, 'cpp_ops/build/imgproc_pb2.py'))

        return {
            'sharpness':
            self._db.ops.SharpnessCPP(
                frame=self._sources['frame_sampled'].op)
        }

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

class SharpnessBBoxCPPPipeline(Pipeline):
    job_suffix = 'sharpness_bbox_cpp'
    parser_fn = lambda _: lambda buf, _: struct.unpack(
            '{}f'.format(int(len(buf) / 4)), buf)

    def build_pipeline(self):
        cwd = os.path.dirname(os.path.abspath(__file__))

        self._db.load_op(
            os.path.join(cwd, 'cpp_ops/build/libimgproc_op.so'),
            os.path.join(cwd, 'cpp_ops/build/imgproc_pb2.py'))

        return {
            'sharpness_bbox':
            self._db.ops.SharpnessBBoxCPP(
                frame=self._sources['frame_sampled'].op)
        }

compute_brightness = BrightnessPipeline.make_runner()
compute_brightness_cpp = BrightnessCPPPipeline.make_runner()
compute_contrast = ContrastPipeline.make_runner()
compute_contrast_cpp = ContrastCPPPipeline.make_runner()
compute_sharpness = SharpnessPipeline.make_runner()
compute_sharpness_cpp = SharpnessCPPPipeline.make_runner()
compute_sharpness_bbox = SharpnessBBoxPipeline.make_runner()
compute_sharpness_bbox_cpp = SharpnessBBoxCPPPipeline.make_runner()
