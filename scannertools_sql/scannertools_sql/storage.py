from scannerpy.storage import StorageBackend, StoredStream
from scannerpy.protobufs import protobufs


class SQLStorage(StorageBackend):
    """StorageBackend backend for streams from a SQL database.

    Currently only supports Postgres."""

    def __init__(self, config, job_table):
        """
        Parameters
        ----------
        config: protobufs.SQLConfig
          Database connection parameters

        job_table: str
          Name of table in the database to track completed jobs
        """

        self._config = config
        self._job_table = job_table

    def source(self, sc, streams):
        num_elements = [s._num_elements for s in streams] \
                       if streams[0]._num_elements is not None else None
        return sc.sources.SQL(
            query=streams[0]._query,
            config=self._config,
            enum_config=[self._config for _ in streams],
            enum_query=[s._query for s in streams],
            filter=[s._filter for s in streams],
            num_elements=num_elements)

    def sink(self, sc, op, streams):
        return sc.sinks.SQL(
            input=op,
            config=self._config,
            table=streams[0]._table,
            job_table=self._job_table,
            job_name=[s._job_name for s in streams],
            insert=streams[0]._insert)

    def delete(self, sc, streams):
        # TODO
        pass


class SQLInputStream(StoredStream):
    """Stream of elements from a SQL database used as input."""

    def __init__(self, query, filter, storage, num_elements=None):
        """
        Parameters
        ----------
        query: protobufs.SQLQuery
          Query that generates a table

        filter: str
          Filter on the query that picks the rows/elements only in this stream

        storage: SQLStorage

        num_elements: int
          Number of elements in this stream. Optional optimization to avoid Scanner having to count.
        """

        assert isinstance(storage, SQLStorage)
        self._query = query
        self._storage = storage
        self._filter = filter
        self._num_elements = num_elements

    def storage(self):
        return self._storage


class SQLOutputStream(StoredStream):
    """Stream of elements into a SQL database used as output."""

    def __init__(self, table, job_name, storage, insert=True):
        """
        Parameters
        ----------
        table: str
          Name of table to stream into.

        job_name: str
          Name of job to insert into the job table.

        storage: SQLStorage

        insert: bool
          Whether to insert new rows or update existing rows.
        """

        assert isinstance(storage, SQLStorage)
        self._storage = storage
        self._table = table
        self._job_name = job_name
        self._insert = insert

    def storage(self):
        return self._storage

    def exists(self):
        # TODO
        return False

    def committed(self):
        # TODO
        return False
