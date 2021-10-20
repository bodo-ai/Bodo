from urllib.parse import parse_qsl, urlparse

import snowflake.connector

import bodo
from bodo.utils import tracing


def get_connection_params(conn_str):
    """From Snowflake connection URL, return dictionary of connection
    parameters that can be passed directly to
    snowflake.connector.connect(**conn_params)
    """
    # Snowflake connection string URL format:
    # snowflake://<user_login_name>:<password>@<account_identifier>/<database_name>/<schema_name>?warehouse=<warehouse_name>&role=<role_name>
    # https://docs.snowflake.com/en/user-guide/sqlalchemy.html#additional-connection-parameters
    u = urlparse(conn_str)
    params = {}
    if u.username:
        params["user"] = u.username
    if u.password:
        params["password"] = u.password
    if u.hostname:
        params["account"] = u.hostname
    if u.port:
        params["port"] = u.port
    if u.path:
        # path contains "database_name/schema_name"
        path = u.path
        if path.startswith("/"):
            path = path[1:]
        database, schema = path.split("/")
        params["database"] = database
        if schema:
            params["schema"] = schema
    if u.query:
        # query contains warehouse_name and role_name
        for key, val in parse_qsl(u.query):
            params[key] = val
    return params


class SnowflakeDataset(object):
    """ Store dataset info in the way expected by Arrow reader in C++. """

    def __init__(self, batches, schema, conn):
        # pieces, _bodo_total_rows and _bodo_total_rows are the attributes
        # expected by ArrowDataframeReader, schema is for SnowflakeReader.
        # NOTE: getting this information from the batches is very cheap and
        # doesn't involve pulling data from Snowflake
        self.pieces = batches
        self._bodo_total_rows = 0
        for b in batches:
            b._bodo_num_rows = b.rowcount
            self._bodo_total_rows += b._bodo_num_rows
        self.schema = schema
        self.conn = conn  # SnowflakeConnection instance


def get_dataset(query, conn_str):
    """ Get snowflake dataset info required by Arrow reader in C++. """
    ev = tracing.Event("get_snowflake_dataset")

    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    # connect to Snowflake. This is the same connection that will be used
    # to read data.
    # We only trace on rank 0 (is_parallel=False) because we want 0 to start
    # executing the queries as soon as possible (don't sync event)
    ev_conn = tracing.Event("snowflake_connect", is_parallel=False)
    conn_params = get_connection_params(conn_str)
    conn = snowflake.connector.connect(**conn_params)
    ev_conn.finalize()

    if bodo.get_rank() == 0:
        cur = conn.cursor()

        # do a cheap query to get the Arrow schema (other ranks need the schema
        # before they can start reading, so we want to get the schema asap)
        # TODO is there a way to get Arrow schema without loading data?
        ev_get_schema = tracing.Event("get_schema", is_parallel=False)
        query_probe = f"select * from ({query}) x LIMIT {100}"
        schema = cur.execute(query_probe).fetch_arrow_all().schema
        ev_get_schema.finalize()

        # execute query
        ev_query = tracing.Event("execute_query", is_parallel=False)
        cur.execute(query)
        ev_query.finalize()

        # get the list of result batches (this doesn't load data).
        # Batch type is snowflake.connector.result_batch.ArrowResultBatch
        batches = cur.get_result_batches()
        comm.bcast((batches, schema))
    else:
        batches, schema = comm.bcast(None)

    ds = SnowflakeDataset(batches, schema, conn)
    ev.finalize()
    return ds
