import numpy as np
import hwang
import scannerpy
import cv2
import storehouse
import tempfile
import subprocess as sp
import os

STORAGE = None
LOCAL_STORAGE = True


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


class Audio:
    def __init__(self, audio_path):
        self._path = audio_path

    def extract(self, path=None, ext='.wav', segment=None):
        return ffmpeg_extract(
            input_path=self.path(), output_path=path, output_ext=ext, segment=segment)

    def path(self):
        return self._path
