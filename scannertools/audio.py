from .prelude import *

class AudioSource(DataSource):
    def __init__(self, video, frame_size=1.0):
        self._video = video
        self._frame_size = frame_size

    def scanner_source(self, db):
        return db.sources.Audio(frame_size=self._frame_size)

    def scanner_args(self, db):
        return {'path': self._video.path()}

