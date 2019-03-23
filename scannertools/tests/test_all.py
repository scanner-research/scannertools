from scannertools_infra.tests import sc, download_videos, needs_gpu
from scannertools.storage.audio_storage import AudioStream
from scannertools.storage.caption_storage import CaptionStream
from scannertools.storage.python_storage import PythonStream
from scannertools.storage.files_storage import FilesStream
import scannertools.imgproc
import scannertools.misc
import scannertools.storage
import scannertools.face_detection
import tempfile
import json
from scannerpy import CacheMode, DeviceType, protobufs, NamedStream, NamedVideoStream, PerfParams
import requests
import scannerpy
import struct
import pickle
import numpy as np


def test_audio(sc):
    (vid_path, _) = download_videos()
    audio = sc.io.Input([AudioStream(vid_path, 1.0)])
    ignored = sc.ops.DiscardFrame(ignore=audio)
    output = sc.io.Output(ignored, [NamedStream(sc, 'audio_test')])
    sc.run(output, PerfParams.estimate(), cache_mode=CacheMode.Overwrite)


def download_transcript():
    url = "https://storage.googleapis.com/scanner-data/test/transcript.cc1.srt"
    with tempfile.NamedTemporaryFile(delete=False, suffix='.cc1.srt') as f:
        resp = requests.get(url, stream=True)
        assert resp.ok
        for block in resp.iter_content(1024):
            f.write(block)
        return f.name


def run_op(sc, op):
    input = NamedVideoStream(sc, 'test1')
    frame = sc.io.Input([input])
    gather_frame = sc.streams.Gather(frame, [[0]])
    faces = op(frame=gather_frame)
    output = NamedStream(sc, 'output')
    output_op = sc.io.Output(faces, [output])
    sc.run(output_op, PerfParams.estimate(), cache_mode=CacheMode.Overwrite, show_progress=False, pipeline_instances_per_node=1)
    return list(output.load())


@scannerpy.register_python_op(name='DecodeCap')
def decode_cap(config, cap: bytes) -> bytes:
    cap = json.loads(cap.decode('utf-8'))
    return b' '


def test_captions(sc):
    caption_path = download_transcript()
    captions = sc.io.Input([CaptionStream(caption_path, window_size=10.0, max_time=3600)])
    ignored = sc.ops.DecodeCap(cap=captions)
    output = sc.io.Output(ignored, [NamedStream(sc, 'caption_test')])
    sc.run(output, PerfParams.estimate(), cache_mode=CacheMode.Overwrite, pipeline_instances_per_node=1)


def test_files_source(sc):
    # Write test files
    path_template = '/tmp/files_source_test_{:d}'
    num_elements = 4
    paths = []
    for i in range(num_elements):
        path = path_template.format(i)
        with open(path, 'wb') as f:
            # Write data
            f.write(struct.pack('=Q', i))
        paths.append(path)

    data = sc.io.Input([FilesStream(paths=paths)])
    pass_data = sc.ops.Pass(input=data)
    output = NamedStream(sc, 'test_files_source')
    output_op = sc.io.Output(pass_data, [output])
    sc.run(output_op, PerfParams.estimate(), cache_mode=CacheMode.Overwrite, show_progress=False)

    num_rows = 0
    for buf in output.load():
        (val, ) = struct.unpack('=Q', buf)
        assert val == num_rows
        num_rows += 1
    assert num_elements == num_rows


def test_files_sink(sc):
    # Write initial test files
    path_template = '/tmp/files_source_test_{:d}'
    num_elements = 4
    input_paths = []
    for i in range(num_elements):
        path = path_template.format(i)
        with open(path, 'wb') as f:
            # Write data
            f.write(struct.pack('=Q', i))
        input_paths.append(path)

    # Write output test files
    path_template = '/tmp/files_sink_test_{:d}'
    num_elements = 4
    output_paths = []
    for i in range(num_elements):
        path = path_template.format(i)
        output_paths.append(path)
    data = sc.io.Input([FilesStream(paths=input_paths)])
    pass_data = sc.ops.Pass(input=data)
    output = FilesStream(paths=output_paths)
    output_op = sc.io.Output(pass_data, [output])
    sc.run(output_op, PerfParams.estimate(), cache_mode=CacheMode.Overwrite, show_progress=False)

    # Read output test files
    for i, s in enumerate(output.load()):
        d, = struct.unpack('=Q', s)
        assert d == i



def test_python_source(sc):
    # Write test files
    py_data = [{'{:d}'.format(i): i} for i in range(4)]

    data = sc.io.Input([PythonStream(py_data)])
    pass_data = sc.ops.Pass(input=data)
    output = NamedStream(sc, 'test_python_source')
    output_op = sc.io.Output(pass_data, [output])
    sc.run(output_op, PerfParams.estimate(), cache_mode=CacheMode.Overwrite, show_progress=False)

    num_rows = 0
    for i, buf in enumerate(output.load()):
        d = pickle.loads(buf)
        assert d['{:d}'.format(i)] == i
        num_rows += 1
    assert num_rows == 4



class DeviceTestBench:
    def test_cpu(self, sc):
        self.run(sc, DeviceType.CPU)

    @needs_gpu()
    def test_gpu(self, sc):
        self.run(sc, DeviceType.GPU)


class TestHistogram(DeviceTestBench):
    def run(self, sc, device):
        input = NamedVideoStream(sc, 'test1')
        frame = sc.io.Input([input])
        hist = sc.ops.Histogram(frame=frame, device=device)
        output = NamedStream(sc, 'test_hist')
        output_op = sc.io.Output(hist, [output])

        sc.run(output_op, PerfParams.estimate(), cache_mode=CacheMode.Overwrite, show_progress=False)
        next(output.load())


class TestOpticalFlow(DeviceTestBench):
    def run(self, sc, device):
        input = NamedVideoStream(sc, 'test1')
        frame = sc.io.Input([input])
        flow = sc.ops.OpticalFlow(frame=frame, stencil=[-1, 0], device=device)
        flow_range = sc.streams.Range(flow, ranges=[{'start': 0, 'end': 50}])
        output = NamedStream(sc, 'test_flow')
        output_op = sc.io.Output(flow_range, [output])
        sc.run(output_op, PerfParams.estimate(), cache_mode=CacheMode.Overwrite, show_progress=False)
        assert output.len() == 50

        flow_array = next(output.load())
        assert flow_array.dtype == np.float32
        assert flow_array.shape[0] == 480
        assert flow_array.shape[1] == 640
        assert flow_array.shape[2] == 2


def test_blur(sc):
    input = NamedVideoStream(sc, 'test1')
    frame = sc.io.Input([input])
    range_frame = sc.streams.Range(frame, ranges=[{'start': 0, 'end': 30}])
    blurred_frame = sc.ops.Blur(frame=range_frame, kernel_size=3, sigma=0.1)
    output = NamedVideoStream(sc, 'test_blur')
    output_op = sc.io.Output(blurred_frame, [output])
    sc.run(output_op, PerfParams.estimate(), cache_mode=CacheMode.Overwrite, show_progress=False)

    frame_array = next(output.load())
    assert frame_array.dtype == np.uint8
    assert frame_array.shape[0] == 480
    assert frame_array.shape[1] == 640
    assert frame_array.shape[2] == 3


def test_face_detection(sc):
    output = run_op(sc, sc.ops.MTCNNDetectFaces)
    assert len(output[0]) == 1
    assert isinstance(output[0][0], protobufs.BoundingBox)


def test_face_embedding(sc):
    def make(frame):
        bboxes = sc.ops.MTCNNDetectFaces(frame=frame)
        return sc.ops.EmbedFaces(frame=frame, bboxes=bboxes)
    output = run_op(sc, make)
    assert len(output[0]) == 1


def test_gender(sc):
    def make(frame):
        bboxes = sc.ops.MTCNNDetectFaces(frame=frame)
        return sc.ops.DetectGender(frame=frame, bboxes=bboxes)
    output = run_op(sc, make)
    assert len(output[0]) == 1


def test_object_detection(sc):
    run_op(sc, sc.ops.DetectObjects)


def test_shot_detection(sc):
    input = NamedVideoStream(sc, 'test1')
    frame = sc.io.Input([input])
    range_frame = sc.streams.Range(frame, [{'start': 0, 'end': 1000}])
    hist = sc.ops.Histogram(frame=range_frame)
    boundaries = sc.ops.ShotBoundaries(histograms=hist)
    output = NamedStream(sc, 'output')
    output_op = sc.io.Output(boundaries, [output])
    sc.run(
        output_op, PerfParams.estimate(), cache_mode=CacheMode.Overwrite, show_progress=False, pipeline_instances_per_node=1,
        work_packet_size=1000, io_packet_size=1000)
    assert len(next(output.load(rows=[0]))) == 7
