from scannerpy.storage import StorageBackend, StoredStream
import pickle
from typing import List, Any


class PythonStorage(StorageBackend):
    """StorageBackend for a stream of elements directly from the current Python process.

    Only supports input, not output.
    """

    def source(self, sc, streams):
        return sc.sources.Python(data=[pickle.dumps(stream._data) for stream in streams])


class PythonStream(StoredStream):
    """Stream of elements directly in the current Python process."""

    def __init__(self, data: List[Any]):
        """
        Parameters
        ----------
        data: List[Any]
          Arbitrary data to stream. Must be pickleable.
        """
        self._data = data

    def storage(self):
        return PythonStorage()
