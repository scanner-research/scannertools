from .prelude import *

class AudioSource(DataSource):
    def __init__(self, video, frame_size=1.0, duration=None):
        self._video = video
        self._frame_size = frame_size
        self._duration = duration

    def scanner_source(self, db):
        return db.sources.Audio(frame_size=self._frame_size)

    def scanner_args(self, db):
        return {
            'path': self._video.path(),
            'duration': self._duration if self._duration is not None else 0.0
        }

class CaptionSource(DataSource):
    def __init__(self, captions, max_time, window_size=10.0):
        self._captions = captions
        self._window_size = window_size
        self._max_time = max_time

    def scanner_source(self, db):
        return db.sources.Captions(window_size=self._window_size)

    def scanner_args(self, db):
        return {'path': self._captions, 'max_time': self._max_time}
