import numpy as np
import hwang
import scannerpy
import cv2
import storehouse
import tempfile
import subprocess as sp
import os
import requests
from contextlib import contextmanager
import logging
import datetime

STORAGE = None
LOCAL_STORAGE = None

log = logging.getLogger('tixelbox')
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


def get_scanner_db():
    return scannerpy.Database()


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


@contextmanager
def sample_video(delete=True):
    url = "https://storage.googleapis.com/scanner-data/test/short_video.mp4"

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


class Video:
    def __init__(self, video_path):
        self._path = video_path
        self._storage = get_storage()
        video_file = storehouse.RandomReadFile(self._storage, video_path.encode('ascii'))
        self._decoder = hwang.Decoder(video_file)
        self._fps = 29.97  # TODO: fetch this from video

    def frame(self, number=None, time=None):
        if time is not None:
            return self.frames(times=[time])[0]
        else:
            return self.frames(numbers=[number])[0]

    def frames(self, numbers=None, times=None):
        if times is not None:
            numbers = [int(n * self._fps) for n in times]

        return self._decoder.retrieve(numbers)

    def fps(self):
        return self._fps

    def audio(self):
        audio_path = ffmpeg_extract(input_path=self.path(), output_ext='.wav')
        return Audio(audio_path)

    def extract(self, path=None, ext='.mp4', segment=None):
        return ffmpeg_extract(
            input_path=self.path(), output_path=path, output_ext=ext, segment=segment)

    def path(self):
        return self._path

    def scanner_name(self):
        return self.path()

    def add_to_scanner(self, db):
        table = self.scanner_name()
        if not db.has_table(table):
            db.ingest_videos([(table, self.path())], inplace=True)

    def scanner_table(self, db):
        return db.table(self.scanner_name())


class Audio:
    def __init__(self, audio_path):
        self._path = audio_path

    def extract(self, path=None, ext='.wav', segment=None):
        return ffmpeg_extract(
            input_path=self.path(), output_path=path, output_ext=ext, segment=segment)

    def path(self):
        return self._path
