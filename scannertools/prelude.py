from attr import attrs, attrib
import math
import numpy as np
import storehouse
# Note: it's critical to import Hwang before Scanner, or else protobuf will complain (fatally)
# that protobufs are being re-registered.
import hwang
import scannerpy
from scannerpy import ColumnType, DeviceType, Job, ScannerException, FrameType
from scannerpy.stdlib import readers, writers
import tempfile
import subprocess as sp
import os
from contextlib import contextmanager
import logging
import datetime
import importlib
from functools import wraps
from abc import ABC
import inspect
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
import multiprocessing as mp

STORAGE = None
LOCAL_STORAGE = None


def try_import(to_import, current_module):
    try:
        return importlib.import_module(to_import)
    except ImportError:
        raise Exception(
            'Module {} requires package `{}`, but you don\'t have it installed. Install it to use this module.'.
            format(current_module, to_import)) from None
    # Note: "raise from None" pattern means "don't show the old exception, just the new one"


log = logging.getLogger('scannertools')
log.setLevel(logging.DEBUG)
if not log.handlers:

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            level = record.levelname[0]
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')[2:]
            if len(record.args) > 0:
                record.msg = '({})'.format(', '.join(
                    [str(x) for x in [record.msg] + list(record.args)]))
                record.args = ()
            return '{level} {time} {filename}:{lineno:03d}] {msg}'.format(
                level=level, time=time, **record.__dict__)

    handler = logging.StreamHandler()
    handler.setFormatter(CustomFormatter())
    log.addHandler(handler)


def init_storage(bucket=None):
    global STORAGE
    global LOCAL_STORAGE
    if bucket is not None:
        storage_config = storehouse.StorageConfig.make_gcs_config(bucket)
        LOCAL_STORAGE = False
    else:
        storage_config = storehouse.StorageConfig.make_posix_config()
        LOCAL_STORAGE = True
    STORAGE = storehouse.StorageBackend.make_from_config(storage_config)


def get_storage():
    global STORAGE
    if STORAGE is None:
        init_storage()
    return STORAGE


def ffmpeg_fmt_time(t):
    return '{:02d}:{:02d}:{:02d}.{:03d}'.format(
        int(t / 3600), int(t / 60 % 60), int(t % 60), int(t * 1000 % 1000))


def ffmpeg_extract(input_path, output_ext=None, output_path=None, segment=None):
    if not LOCAL_STORAGE:
        raise Exception("Only works on locally stored files right now.")

    if output_path is None:
        assert output_ext is not None
        output_path = tempfile.NamedTemporaryFile(
            suffix='.{}'.format(output_ext), delete=False).name

    if segment is not None:
        (start, end) = segment
        start_str = '-ss {}'.format(ffmpeg_fmt_time(start))
        end_str = '-t {}'.format(ffmpeg_fmt_time(end - start))
    else:
        start_str = ''
        end_str = ''

    fnull = open(os.devnull, 'w')
    sp.check_call(
        'ffmpeg -y {} -i "{}" {} "{}"'.format(start_str, input_path, end_str, output_path),
        shell=True,
        stdout=fnull,
        stderr=fnull)

    return output_path


def imwrite(path, img):
    cv2 = try_import('cv2', 'scannertools')
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


@contextmanager
def sample_video(delete=True):
    import requests
    from .video import Video

    url = "https://storage.googleapis.com/scanner-data/public/sample-clip.mp4"

    if delete:
        f = tempfile.NamedTemporaryFile(suffix='.mp4')
    else:
        sample_path = '/tmp/sample_video.mp4'
        if os.path.isfile(sample_path):
            yield Video(sample_path)
            return

        f = open(sample_path, 'wb')

    with f as f:
        resp = requests.get(url, stream=True)
        assert resp.ok
        for block in resp.iter_content(1024):
            f.write(block)
        f.flush()
        yield Video(f.name)


def tile(imgs, rows=None, cols=None):
    # If neither rows/cols is specified, make a square
    if rows is None and cols is None:
        rows = int(math.sqrt(len(imgs)))

    if rows is None:
        rows = (len(imgs) + cols - 1) // cols
    else:
        cols = (len(imgs) + rows - 1) // rows

    # Pad missing frames with black
    diff = rows * cols - len(imgs)
    if diff != 0:
        imgs.extend([np.zeros(imgs[0].shape, dtype=imgs[0].dtype) for _ in range(diff)])

    return np.vstack([np.hstack(imgs[i * cols:(i + 1) * cols]) for i in range(rows)])


def scanner_ingest(db, videos):
    videos = [v for v in videos if not db.has_table(v.scanner_name())]
    if len(videos) > 0:
        # TODO(wcrichto): change this to inplace=True once
        # https://github.com/scanner-research/scanner/issues/162 is fixed.
        db.ingest_videos([(v.scanner_name(), v.path()) for v in videos])


class WithMany:
    def __init__(self, *args):
        self._args = args

    def __enter__(self):
        return tuple([obj.__enter__() for obj in self._args])

    def __exit__(self, *args, **kwargs):
        for obj in self._args:
            obj.__exit__(*args, **kwargs)


@attrs
class BoundOp:
    op = attrib()
    args = attrib()


class DataSource(ABC):
    def load(self):
        raise NotImplemented

    def scanner_source(self, db):
        raise NotImplemented

    def scanner_args(self, db):
        raise NotImplemented


class ScannerColumn(DataSource):
    def __init__(self, column, parser):
        self._column = column
        self._parser = parser

    def load(self):
        if self._parser is None:
            raise Exception("Pipeline used default Scanner output but did not provide a parser_fn")

        return self._column.load(self._parser)

    def scanner_source(self, db):
        return db.sources.Column()

    def scanner_args(self, db):
        return self._column


class Pipeline(ABC):
    job_suffix = None
    parser_fn = None
    base_sources = ['videos', 'frames']
    additional_sources = []
    run_opts = {}

    def __init__(self, db):
        self._db = db

    def _ingest(self, videos):
        videos = [v for v in videos if not self._db.has_table(v.scanner_name())]
        if len(videos) > 0:
            # TODO(wcrichto): change this to inplace=True once
            # https://github.com/scanner-research/scanner/issues/162 is fixed.
            self._db.ingest_videos([(v.scanner_name(), v.path()) for v in videos])

    def _build_jobs(self):
        source_keys = self._sources.keys()

        jobs = []
        for i in range(len(self._sink.args)):
            map_ = {}
            for k in source_keys:
                src = self._sources[k]
                map_[src.op] = src.args[i]
            map_[self._sink.op] = self._sink.args[i]
            jobs.append(Job(op_args=map_))

        return jobs

    def fetch_resources(self):
        pass

    def build_sources(self, videos=None, frames=None, **kwargs):
        sources = {}

        self._ingest(videos)

        frame = self._db.sources.FrameColumn()
        sources['frame'] = BoundOp(
            op=frame, args=[self._db.table(v.scanner_name()).column('frame') for v in videos])

        if frames is not None:
            frame_sampled = self._db.streams.Gather(frame)
            sources['frame_sampled'] = BoundOp(op=frame_sampled, args=frames)
        else:
            frame_sampled = self._db.streams.Stride(frame)
            sources['frame_sampled'] = BoundOp(
                op=frame_sampled, args=[1 for _ in range(len(videos))])

        for k, v in kwargs.items():
            source = v[0].scanner_source(self._db)
            sources[k] = BoundOp(op=source, args=[c.scanner_args(self._db) for c in v])

        return sources

    def build_sink(self):
        # TODO(wcrichto): use input table name, not index
        return BoundOp(
            op=self._db.sinks.Column(columns=self._output_ops),
            args=[
                '{}_{}'.format(i, self.job_suffix) for i in range(len(self._sources['frame'].args))
            ])

    def parse_output(self):
        if len(self._output_ops.keys()) == 1:
            col_name = list(self._output_ops.keys())[0]
            return [
                ScannerColumn(
                    self._db.table(t).column(col_name),
                    self.parser_fn() if self.parser_fn is not None else None)
                if self._db.table(t).committed() else None
                for t in self._sink.args
            ]
        else:
            raise Exception("Multiple outputs, no default can be returned")

    def build_pipeline(self):
        raise NotImplemented

    def execute(self, source_args={}, pipeline_args={}, sink_args={}, output_args={}, run_opts={}, no_execute=False):
        self.fetch_resources()

        self._sources = self.build_sources(**source_args)

        self._output_ops = self.build_pipeline(**pipeline_args)

        self._sink = self.build_sink(**sink_args)

        jobs = self._build_jobs()

        if not no_execute:
            self._db.run(self._sink.op, jobs, force=True, **{**self.run_opts, **run_opts})

        return self.parse_output(**output_args)

    @classmethod
    def make_runner(cls):
        def runner(db, run_opts={}, no_execute=False, **kwargs):
            pipeline = cls(db)

            def method_arg_names(f):
                return list(inspect.signature(f).parameters.keys())

            source_arg_names = pipeline.base_sources + pipeline.additional_sources + method_arg_names(pipeline.build_sources)
            pipeline_arg_names = method_arg_names(pipeline.build_pipeline)
            sink_arg_names = method_arg_names(pipeline.build_sink)
            output_arg_names = method_arg_names(pipeline.parse_output)

            source_args = {}
            pipeline_args = {}
            sink_args = {}
            output_args = {}

            for k, v in kwargs.items():
                found = False
                for (names, dct) in [(source_arg_names, source_args), \
                                     (pipeline_arg_names, pipeline_args), \
                                     (sink_arg_names, sink_args), \
                                     (output_arg_names, output_args)]:
                    if k in names:
                        dct[k] = v
                        found = True
                        break

                if not found:
                    raise Exception('Received unexpected arguments "{}"'.format(k))

            return pipeline.execute(
                source_args=source_args,
                pipeline_args=pipeline_args,
                sink_args=sink_args,
                output_args=output_args,
                run_opts=run_opts,
                no_execute=no_execute)

        return runner


class VideoOutputPipeline(Pipeline):
    def parse_output(self, paths=None):
        paths = paths if paths is not None else [
            tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
            for _ in range(len(self._sources['frame'].args))
        ]

        for (table, p) in zip(self._sink.args, paths):
            self._db.table(table).column('frame').save_mp4(os.path.splitext(p)[0])

        return paths

def par_for(f, l, process=False, workers=None, progress=True):
    Pool = ProcessPoolExecutor if process else ThreadPoolExecutor
    with Pool(max_workers=mp.cpu_count() if workers is None else workers) as executor:
        if progress:
            return list(tqdm(executor.map(f, l), total=len(l)))
        else:
            return list(executor.map(f, l))
