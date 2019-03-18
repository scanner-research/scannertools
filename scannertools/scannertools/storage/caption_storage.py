from scannerpy.storage import StorageBackend, StoredStream

class CaptionStorage(StorageBackend):
    """StorageBackend for caption streams."""

    def source(self, sc, streams):
        return sc.sources.Captions(
            window_size=[s._window_size for s in streams],
            path=[s._path for s in streams],
            max_time=[s._max_time for s in streams])


class CaptionStream(StoredStream):
    """Stream of captions out of a caption file.

    In order for the number of stream elements to be predictable (e.g. to zip a caption stream
    with an audio stream for transcript alignment), we represent caption streams as uniform time
    intervals. You provide a stream duration (e.g. 10 minutes) and a window size (e.g. 5 seconds)
    and the stream contains 10 minutes / 5 seconds number of elements, where each element contains
    all of the text for captions that overlap with that window.
    """

    def __init__(self, path, window_size, max_time, storage=None):
        """
        Parameters
        ----------
        path: str
          Path on the filesystem to the caption file.

        window_size: float
          Size of window in time (seconds) for each element. See class description.

        max_time: float
          Total time for the entire stream. See class description.

        storage: CaptionStorage
        """

        if storage is None:
            self._storage = CaptionStorage()
        else:
            self._storage = storage

        self._path = path
        self._max_time = max_time
        self._window_size = window_size

    def storage(self):
        return self._storage
