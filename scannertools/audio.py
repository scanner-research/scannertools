from .prelude import *


class Audio:
    """
    Reference to an audio file on disk.
    """

    def __init__(self, audio_path):
        self._path = audio_path

    def extract(self, path=None, ext='.wav', segment=None):
        return ffmpeg_extract(
            input_path=self.path(), output_path=path, output_ext=ext, segment=segment)

    def path(self):
        return self._path
