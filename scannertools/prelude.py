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

STORAGE = None
LOCAL_STORAGE = None


def try_import(to_import, current_module):
    try:
        return importlib.import_module(to_import)
    except ImportError:
        raise Exception(
            'Module {} requires package `{}`, but you don\'t have it installed. Install it to use this module.'.
            format(current_module, to_import))


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


def autobatch(uniforms=[]):
    def wrapper(fn):
        @wraps(fn)
        def newfn(*args, **kwargs):
            positions = [x for x in uniforms if isinstance(x, int)]
            keywords = [k for k in uniforms if isinstance(k, str)]

            if len(args) > 0:
                nonuniform = args[next(iter(set(range(len(args))) - set(positions)))]
            else:
                nonuniform = kwargs[next(iter(set(kwargs.keys()) - set(keywords)))]

            batched = isinstance(nonuniform, list)
            if not batched:
                args = [[x] if i not in positions and x is not None else x
                        for (i, x) in enumerate(args)]
                kwargs = {
                    k: [v] if k not in keywords and v is not None else v
                    for k, v in kwargs.items()
                }

            res = fn(*args, **kwargs)
            return res[0] if not batched else res

        return newfn

    return wrapper


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
