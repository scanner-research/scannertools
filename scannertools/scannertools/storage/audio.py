from scannerpy.storage import StorageBackend, StoredStream


class AudioStorage(StorageBackend):
    """StorageBackend for stream of elements from a compressed audio file.

    Currently input-only."""

    def source(self, sc, streams):
        return sc.sources.Audio(
            frame_size=[s._frame_size for s in streams],
            path=[s._path for s in streams])


class AudioStream(StoredStream):
    """Stream of elements from a compressed audio file."""

    def __init__(self, path, frame_size, storage=None):
        """
        Parameters
        ----------
        path: str
          Path on filesystem to audio file.

        frame_size: float
          Size (in seconds) of each element, e.g. a 2s frame size with a a 44.1 kHz generates
          stream elements of 88.2k samples per element.

        storage: AudioStorage
        """

        if storage is None:
            self._storage = AudioStorage()
        else:
            self._storage = storage

        self._path = path
        self._frame_size = frame_size

    def storage(self):
        return self._storage
