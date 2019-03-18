from scannerpy.storage import StorageBackend, StoredStream
from typing import List
import os


class FilesStorage(StorageBackend):
    """Storage of streams where each element is its own file."""

    def __init__(self, storage_type: str = "posix", bucket: str = None, region: str = None, endpoint: str = None):
        """
        Parameters
        ----------
        storage_type: str
          Kind of filesystem the files are on. Either "posix" or "gcs" supported.

        bucket: str
          If filesystem is gcs, name of bucket

        region: str
          If filesytem is gcs, region name of bucket, e.g. us-west1

        endpoint: str
          If filesystem is gcs, URL of storage endpoint
        """

        self._storage_type = storage_type
        self._bucket = bucket
        self._region = region
        self._endpoint = endpoint

    def source(self, sc, streams):
        return sc.sources.Files(
            storage_type=self._storage_type,
            bucket=self._bucket,
            region=self._region,
            endpoint=self._endpoint,
            paths=[s._paths for s in streams])

    def sink(self, sc, op, streams):
        return sc.sinks.Files(
            input=op,
            storage_type=self._storage_type,
            bucket=self._bucket,
            region=self._region,
            endpoint=self._endpoint,
            paths=[s._paths for s in streams])

    def delete(self, sc, streams):
        # TODO
        pass


class FilesStream(StoredStream):
    """Stream where each element is a file."""

    def __init__(self, paths: List[str], storage: FilesStorage = None):
        """
        Parameters
        ----------
        paths: List[str]
          List of paths to the files in the stream.

        storage: FilesStorage
        """
        if storage is None:
            self._storage = FilesStorage()
        else:
            self._storage = storage

        self._paths = paths

    def load_bytes(self, rows=None):
        paths = self._paths
        if rows is not None:
            paths = [paths[i] for i in rows]

        for path in paths:
            yield open(path, 'rb').read()

    def storage(self):
        return self._storage

    def committed(self):
        # TODO
        return all(os.path.isfile(p) for p in self._paths)

    def exists(self):
        # TODO
        return any(os.path.isfile(p) for p in self._paths)

    def type(self):
        return None
