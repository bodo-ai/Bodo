from urllib.parse import parse_qsl, urlparse

import pyarrow as pa
import snowflake.connector

import bodo
from bodo.utils import tracing

# Mapping of the Snowflake field types to the pyarrow types taken
# from the snowflake connector. These are not fully accurate and don't match
# the arrow types. However, they can be used when the returned data is empty.
# TODO: Improve this information to only require describe.
# https://github.com/snowflakedb/snowflake-connector-python/blob/a73a602f96678e4c761a63d89676fed182ef5093/src/snowflake/connector/result_batch.py#L671
# https://docs.snowflake.com/en/user-guide/python-connector-api.html#label-python-connector-type-codes
FIELD_TYPE_TO_PA_TYPE = [
    # Number/Int. TODO: handle signed/unsigned + bitwidth
    pa.int64(),
    # Float/Double. TODO: handle bitwidth
    pa.float64(),
    # String data
    pa.string(),
    # Date data. TODO: handle bitwidth
    pa.date64(),
    pa.timestamp("ns"),
    # Variant
    pa.string(),
    # Timestamp stored in utc TIMESTAMP_LTZ
    pa.timestamp("ns"),
    # Timestamp with a timezone, TIMESTAMP_TZ. TODO: handle timestamp
    pa.timestamp("ns"),
    # Timestamp without a timezone, TIMESTAMP_NTZ
    pa.timestamp("ns"),
    # Object
    pa.string(),
    # Array TODO: Fix ME???
    pa.string(),
    # Binary
    pa.binary(),
    # Time. Not supported in bodo. TODO: handle bitwidth
    pa.time64("ns"),
    # Boolean
    pa.bool_(),
]


def get_connection_params(conn_str):  # pragma: no cover
    """From Snowflake connection URL, return dictionary of connection
    parameters that can be passed directly to
    snowflake.connector.connect(**conn_params)
    """
    # Snowflake connection string URL format:
    # snowflake://<user_login_name>:<password>@<account_identifier>/<database_name>/<schema_name>?warehouse=<warehouse_name>&role=<role_name>
    # https://docs.snowflake.com/en/user-guide/sqlalchemy.html#additional-connection-parameters
    import json

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
            if key == "session_parameters":
                # Snowflake connector appends to session_parameters and
                # assumes it is a dictionary if provided. This is an existing
                # bug in SqlAlchemy/SnowflakeSqlAlchemy
                params[key] = json.loads(val)
    # pass Bodo identifier to Snowflake
    params["application"] = "bodo"
    return params


class SnowflakeDataset(object):
    """Store dataset info in the way expected by Arrow reader in C++."""

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
    """Get snowflake dataset info required by Arrow reader in C++."""
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
        arrow_data = cur.execute(query_probe).fetch_arrow_all()
        if arrow_data is None:
            # If we don't load any data we construct a schema from describe.
            described_query = cur.describe(query)
            # Construct the arrow schema from the describe info
            pa_fields = [
                pa.field(x.name, FIELD_TYPE_TO_PA_TYPE[x.type_code])
                for x in described_query
            ]
            schema = pa.schema(pa_fields)
        else:
            schema = arrow_data.schema

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
