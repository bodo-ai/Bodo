import os
import sys
import traceback
import warnings
from tempfile import TemporaryDirectory
from typing import Optional, Tuple
from urllib.parse import parse_qsl, urlparse
from uuid import uuid4

import pyarrow as pa
from mpi4py import MPI

import bodo
from bodo.io.helpers import (
    ExceptionPropagatingThread,
    update_env_vars,
    update_file_contents,
)
from bodo.utils import tracing
from bodo.utils.py_objs import install_py_obj_class
from bodo.utils.typing import BodoError, BodoWarning

# Imports for convenience.
try:
    from snowflake.connector import JSONResultBatch, SnowflakeConnection
except ImportError:
    JSONResultBatch = None
    SnowflakeConnection = None

# Whether to do a probe query to determine whether string columns should be
# dictionary-encoded. This doesn't effect the _bodo_read_as_dict argument.
SF_READ_AUTO_DICT_ENCODE_ENABLED = True

# A configurable variable by which we determine whether to dictionary-encode
# a string column.
# Encode if num of unique elem / num of total rows <= SF_READ_DICT_ENCODE_CRITERION
SF_READ_DICT_ENCODE_CRITERION = 0.5

# How long the dictionary encoding probe query should run for in the worst case.
# This is to guard against increasing compilation time prohibitively in case there are
# issues with Snowflake, the data, etc.
SF_READ_DICT_ENCODING_PROBE_TIMEOUT = 5

# Default behavior if the query to determine dictionary encoding times out.
# This is false by default since dict encoding is an optimization, and in cases where
# we cannot definitively determine if it should be used, we should not use it. The
# config flag is useful in cases where we (developers) want to test certain situations
# manually.
SF_READ_DICT_ENCODING_IF_TIMEOUT = False

# Maximum number of rows to read from Snowflake in the probe query
# This is calculated as # of string columns * # of rows
# The default 100M should take a negligible amount of time.
# This default value is based on empirical benchmarking to have
# a good balance between accuracy of the query and compilation
# time. Find more detailed analysis and results here:
# https://bodo.atlassian.net/wiki/spaces/B/pages/1134985217/Support+reading+dictionary+encoded+string+columns+from+Snowflake#Prediction-query-and-heuristic
SF_READ_DICT_ENCODING_PROBE_ROW_LIMIT = 100_000_000

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
    pa.date32(),
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

# SF_WRITE_COPY_INTO_ON_ERROR (str):
# Action to take when `COPY INTO` statements fail.
#  -  "continue": Continue to load the file if errors are found.
#  -  "skip_file": Skip a file when an error is found.
#  -  "skip_file_<num>": Skip a file when the number of error rows
#         found in the file is equal to or exceeds the specified number.
#  -  "skip_file_<num>%": Skip a file when the percentage of error rows
#         found in the file exceeds the specified percentage.
#  -  "abort_statement": Abort the load operation if any error is
#         found in a data file.
# Default follows documentation for Snowflake's COPY INTO command:
# (https://docs.snowflake.com/en/sql-reference/sql/copy-into-table.html#copy-options-copyoptions)
SF_WRITE_COPY_INTO_ON_ERROR = "abort_statement"

# SF_WRITE_OVERLAP_UPLOAD (bool):
# If True, attempt to overlap writing dataframe to Parquet files and
#     uploading Parquet files to Snowflake internal stage using Python
#     threads on each rank. This speeds up the upload process by
#     overlapping compute-heavy and I/O-heavy steps.
# If False, perform these steps in sequence without spawning threads.
SF_WRITE_OVERLAP_UPLOAD = True

# SF_WRITE_PARQUET_CHUNK_SIZE (int):
# Chunk size to use when writing dataframe to Parquet files, measured by
# the uncompressed memory usage of the dataframe to be compressed (in bytes).
# Decreasing the chunksize allows more load operations to run in parallel
# during `COPY_INTO`, while increasing the chunksize allows less processing
# overhead for each Parquet file. See Snowflake's File Sizing Best Practices
# and Limitations for guidance on how to choose this value:
# (https://docs.snowflake.com/en/user-guide/data-load-considerations-prepare.html#file-sizing-best-practices-and-limitations)
SF_WRITE_PARQUET_CHUNK_SIZE = int(256e6)

# SF_WRITE_PARQUET_COMPRESSION (str):
# The compression algorithm to use for Parquet files uploaded to Snowflake
# internal stage. Can be any compression algorithm supported by Pyarrow, but
# "snappy" and "gzip " should work best as they are specifically suited for parquet:
# "NONE", "SNAPPY", "GZIP", "BROTLI", "LZ4", "ZSTD". See this link for
# supported codecs: https://github.com/apache/parquet-format/blob/master/Compression.md
SF_WRITE_PARQUET_COMPRESSION = "snappy"

# SF_WRITE_UPLOAD_USING_PUT (bool):
# If True, `to_sql` saves the dataframe to Parquet files in a local
#     temporary directory, then uses Snowflake's PUT command to upload
#     to internal stage (https://docs.snowflake.com/en/sql-reference/sql/put.html)
#     This method supports all Snowflake stage types but may be slower.
# If False, `to_sql` directly uploads Parquet files to S3/ADLS/GCS using Bodo's
#     internal filesystem-write infrastructure. This method is faster but does
#     not support Azure and GCS-backed Snowflake accounts.
SF_WRITE_UPLOAD_USING_PUT = False


# Content to put in core-site.xml for Snowflake write. For ADLS backed stages,
# Snowflake gives us a SAS token for access to the stage. Unfortunately,
# SAS tokens cannot be directly provided in core-site, and instead require
# a SASTokenProvider implementation. This core-site specifies our own
# SASTokenProvider class (BodoSASTokenProvider) as the implementation.
# This implementation simply reads the token from a file
# (SF_AZURE_WRITE_SAS_TOKEN_FILE_LOCATION) and returns it. Note that
# this is meant to be a constant, not something that  users should
# need to modify.
SF_AZURE_WRITE_HDFS_CORE_SITE = """<configuration>
  <property>
    <name>fs.azure.account.auth.type</name>
    <value>SAS</value>
  </property>
  <property>
    <name>fs.azure.sas.token.provider.type</name>
    <value>org.bodo.azurefs.sas.BodoSASTokenProvider</value>
  </property>
  <property>
    <name>fs.abfs.impl</name>
    <value>org.apache.hadoop.fs.azurebfs.AzureBlobFileSystem</value>
  </property>
</configuration>
"""

# Temporary location to write the SAS token to. This will
# be read by BodoSASTokenProvider.
# Keep this in sync with the location in BodoSASTokenProvider.java
SF_AZURE_WRITE_SAS_TOKEN_FILE_LOCATION = os.path.join(
    bodo.HDFS_CORE_SITE_LOC_DIR.name, "sas_token.txt"
)


def snowflake_connect(conn_str, is_parallel=False):  # pragma: no cover
    """From Snowflake connection URL, connect to Snowflake.

    Args
        conn_str (str): Snowflake connection URL in the following format:
            snowflake://<user_login_name>:<password>@<account_identifier>/<database_name>/<schema_name>?warehouse=<warehouse_name>&role=<role_name>
            Required arguments include <user_login_name>, <password>, and
            <account_identifier>. Optional arguments include <database_name>,
            <schema_name>, <warehouse_name>, and <role_name>.
            Do not include the `snowflakecomputing.com` domain name as part of
            your account identifier. Snowflake automatically appends the domain
            name to your account identifier to create the required connection.
            (https://docs.snowflake.com/en/user-guide/sqlalchemy.html#connection-parameters)
        is_parallel (bool): True if this function being is called from all
            ranks, and False otherwise

    Returns
        conn (snowflake.connection): Snowflake connection
    """
    ev = tracing.Event("snowflake_connect", is_parallel=is_parallel)

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
        parts = path.split("/")
        if len(parts) == 2:
            database, schema = parts
        elif len(parts) == 1:  # pragma: no cover
            database = parts[0]
            schema = None
        else:  # pragma: no cover
            raise BodoError(
                f"Unexpected Snowflake connection string {conn_str}. Path is expected to contain database name and possibly schema"
            )
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
                import json

                params[key] = json.loads(val)
    # pass Bodo identifier to Snowflake
    params["application"] = "bodo"
    # Set a short login timeout so people don't have to wait the default
    # 60 seconds to find out they added the wrong credentials.
    params["login_timeout"] = 5

    try:
        import snowflake.connector
    except ImportError:
        raise BodoError(
            "Snowflake Python connector packages not found. "
            "Using 'to_sql' with Snowflake requires snowflake-connector-python. "
            "This can be installed by calling 'conda install -c conda-forge snowflake-connector-python' "
            "or 'pip install snowflake-connector-python'."
        )
    conn = snowflake.connector.connect(**params)

    ev.finalize()
    return conn


class SnowflakeDataset(object):
    """Store dataset info in the way expected by Arrow reader in C++."""

    def __init__(self, batches, schema, conn):
        # pieces, _bodo_total_rows and _bodo_total_rows are the attributes
        # expected by ArrowDataFrameReader, schema is for SnowflakeReader.
        # NOTE: getting this information from the batches is very cheap and
        # doesn't involve pulling data from Snowflake
        self.pieces = batches
        self._bodo_total_rows = 0
        for b in batches:
            b._bodo_num_rows = b.rowcount
            self._bodo_total_rows += b._bodo_num_rows
        self.schema = schema
        self.conn = conn  # SnowflakeConnection instance


class FakeArrowJSONResultBatch:
    """
    Results Batch used to return a JSONResult in arrow format while
    conforming to the same APIS as ArrowResultBatch
    """

    def __init__(self, json_batch: JSONResultBatch, schema: pa.Schema) -> None:
        self._json_batch = json_batch
        self._schema = schema

    @property
    def rowcount(self):
        return self._json_batch.rowcount

    def to_arrow(self, conn: Optional[SnowflakeConnection] = None) -> pa.Table:
        """
        Return the data in arrow format.

        Args:
            conn (Optional[snowflake.connection.SnowflakeConnection]): Connection
                that is accepted by ArrowResultBatch. We ignore this argument but
                conform to the same API.

        Returns:
            pa.Table: The data in arrow format.
        """
        # Iterate over the data to use the pa.Table.from_pylist
        # constructor
        pylist = []
        for row in self._json_batch.create_iter():
            pylist.append(
                {self._schema.names[i]: col_val for i, col_val in enumerate(row)}
            )
        table = pa.Table.from_pylist(pylist, schema=self._schema)
        return table


def get_dataset(
    query: str,
    conn_str: str,
    only_fetch_length: Optional[bool] = False,
    is_select_query: Optional[bool] = True,
) -> Tuple[SnowflakeDataset, int]:
    """Get snowflake dataset info required by Arrow reader in C++ and execute
    the Snowflake query

    Args:
        query (str): Query to execute inside Snowflake
        conn_str (str): Connection string Bodo will parse to connect to Snowflake.
        only_fetch_length (bool, optional): Is the query just used to fetch rather
            than return a table? If so we just run a COUNT(*) query and broadcast
            the length without any batches.. Defaults to False.
        is_select_query (bool, optional): Is this query a select?

    Raises:
        BodoError: Raises an error if Bodo returns the data in the wrong format.

    Returns:
        Tuple([SnowflakeDataset, int]): Returns a pair of values:
            - The SnowflakeDataset object that holds the information to access
              the actual data results.
            - The number of rows in the output.
    """
    assert not (
        only_fetch_length and not is_select_query
    ), "The only length optimization can only be run with select queries"

    # Snowflake import
    try:
        import snowflake.connector
    except ImportError:
        raise BodoError(
            "Snowflake Python connector packages not found. "
            "Fetching data from Snowflake requires snowflake-connector-python. "
            "This can be installed by calling 'conda install -c conda-forge snowflake-connector-python' "
            "or 'pip install snowflake-connector-python'."
        )

    ev = tracing.Event("get_snowflake_dataset")

    comm = MPI.COMM_WORLD

    # connect to Snowflake. This is the same connection that will be used
    # to read data.
    # We only trace on rank 0 (is_parallel=False) because we want 0 to start
    # executing the queries as soon as possible (don't sync event)
    conn = snowflake_connect(conn_str)
    # Number of rows loaded. This is only used if we are loading
    # 0 columns
    num_rows = -1
    batches = []
    schema = pa.schema([])

    if only_fetch_length and is_select_query:
        # If we are loading 0 columns, the query will just be a COUNT(*).
        # In this case we can skip computing the query
        if bodo.get_rank() == 0:
            cur = conn.cursor()
            ev_query = tracing.Event("execute_length_query", is_parallel=False)
            cur.execute(query)
            # We are just loading a single row of data so we can just load
            # all of the data.
            arrow_data = cur.fetch_arrow_all()
            num_rows = arrow_data[0][0].as_py()
            num_rows = comm.bcast(num_rows)
            ev_query.finalize()
        else:
            num_rows = comm.bcast(None)
    else:
        # We need to actually submit a Snowflake query
        if bodo.get_rank() == 0:
            cur = conn.cursor()
            # do a cheap query to get the Arrow schema (other ranks need the schema
            # before they can start reading, so we want to get the schema asap)
            # TODO is there a way to get Arrow schema without loading data?
            ev_get_schema = tracing.Event("get_schema", is_parallel=False)
            if is_select_query:
                query_probe = f"select * from ({query}) x LIMIT {100}"
                arrow_data = cur.execute(query_probe).fetch_arrow_all()
            else:
                arrow_data = None
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
            # Fetch the total number of rows that will be loaded globally
            num_rows = cur.rowcount
            # get the list of result batches (this doesn't load data).
            # Batch type is snowflake.connector.result_batch.ArrowResultBatch
            batches = cur.get_result_batches()
            if len(batches) > 0 and not isinstance(
                batches[0], snowflake.connector.result_batch.ArrowResultBatch
            ):
                if (
                    not is_select_query
                    and len(batches) == 1
                    and isinstance(
                        batches[0], snowflake.connector.result_batch.JSONResultBatch
                    )
                ):
                    # When executing a non-select query (e.g. DELETE), we may not obtain
                    # the result in Arrow format and instead get a JSONResultBatch. If so
                    # we convert the JSONResultBatch to a fake arrow that supports the same
                    # APIs.
                    #
                    # To be conservative against possible performance issues during development, we
                    # only allow a single batch. Every query that is currently supported only returns
                    # a single row.
                    batches = [FakeArrowJSONResultBatch(x, schema) for x in batches]
                else:
                    raise BodoError(
                        f"Batches returns from Snowflake don't match the expected format. Expected Arrow batches but got {type(batches[0])}"
                    )
            comm.bcast((num_rows, batches, schema))
        else:
            num_rows, batches, schema = comm.bcast(None)

    ds = SnowflakeDataset(batches, schema, conn)
    ev.finalize()
    return ds, num_rows


# --------------------------- snowflake_write helper functions ----------------------------
def create_internal_stage(cursor, is_temporary=False):
    """Create an internal stage within Snowflake. If `is_temporary=False`,
    the named stage must be dropped manualy in `drop_internal_stage()`

    Args
        cursor (snowflake.connector.cursor): Snowflake connection cursor
        is_temporary (bool): Whether the created stage is temporary.
            Named stages are suitable for data loads that could involve multiple users:
            https://docs.snowflake.com/en/user-guide/data-load-local-file-system-create-stage.html#named-stages.
            From experimentation, temporary stages are only accessible to the cursor
            that created them, and are not suitable for this operation which involves
            multiple simultaneous uploads from different connections.

    Returns
        stage_name (str): Name of created internal stage
    """
    ev = tracing.Event("create_internal_stage", is_parallel=False)

    # Snowflake import
    try:
        import snowflake.connector
    except ImportError:
        raise BodoError(
            "Snowflake Python connector packages not found. "
            "Using 'to_sql' with Snowflake requires snowflake-connector-python. "
            "This can be installed by calling 'conda install -c conda-forge snowflake-connector-python' "
            "or 'pip install snowflake-connector-python'."
        )

    stage_name = None  # forward declaration
    stage_name_err = None  # forward declaration

    # We will quickly generate a stage name that doesn't already exist within Snowflake.
    # An infinite loop here is extremely unlikely unless uuid4's are used up.
    while True:
        try:
            stage_name = f"bodo_io_snowflake_{uuid4()}"
            if is_temporary:
                create_stage_cmd = "CREATE TEMPORARY STAGE"
            else:
                create_stage_cmd = "CREATE STAGE"

            create_stage_sql = (
                f'{create_stage_cmd} "{stage_name}" '
                f"/* Python:bodo.io.snowflake.create_internal_stage() */ "
            )
            cursor.execute(create_stage_sql, _is_internal=True).fetchall()
            break

        except snowflake.connector.ProgrammingError as pe:
            if pe.msg.endswith("already exists."):
                continue
            stage_name_err = pe.msg
            break

    ev.finalize()

    if stage_name_err is not None:
        raise snowflake.connector.ProgrammingError(stage_name_err)
    return stage_name


def drop_internal_stage(cursor, stage_name):
    """Drop an internal stage within Snowflake.

    Args
        cursor (snowflake.connector.cursor): Snowflake connection cursor
        stage_name (str): Name of internal stage to drop
    """
    ev = tracing.Event("drop_internal_stage", is_parallel=False)

    drop_stage_sql = (
        f'DROP STAGE "{stage_name}" '
        f"/* Python:bodo.io.snowflake.drop_internal_stage() */ "
    )
    cursor.execute(drop_stage_sql, _is_internal=True)

    ev.finalize()


def do_upload_and_cleanup(
    cursor,
    chunk_idx,
    chunk_path,
    stage_name,
):
    """Upload the parquet file at the given file stream or path to Snowflake
    internal stage in a parallel thread, and perform needed cleanup.

    Args
        cursor (snowflake.connector.cursor): Snowflake connection cursor
        chunk_idx (int): Index of the current parquet chunk
        chunk_path (str): Path to the file to upload
        stage_name (str): Snowflake internal stage name to upload files to

    Returns
        upload_thread (threading.Thread): Return the upload thread responsible
            for uploading this chunk. If False, return None. Call
            `bodo.io.helpers.join_all_threads()` on a list of threads
            generated by this function to complete the upload.
    """

    def upload_cleanup_thread_func(
        chunk_idx,
        chunk_path,
        stage_name,
    ):
        ev_upload_parquet = tracing.Event(
            f"upload_parquet_file{chunk_idx}", is_parallel=False
        )
        upload_sql = (
            f"PUT 'file://{chunk_path}' @\"{stage_name}\" AUTO_COMPRESS=FALSE "
            f"/* Python:bodo.io.snowflake.do_upload_and_cleanup() */"
        )
        cursor.execute(upload_sql, _is_internal=True).fetchall()
        ev_upload_parquet.finalize()

        # Remove chunk file
        os.remove(chunk_path)

    if SF_WRITE_OVERLAP_UPLOAD:
        th = ExceptionPropagatingThread(
            target=upload_cleanup_thread_func,
            args=(chunk_idx, chunk_path, stage_name),
        )
        th.start()
    else:
        upload_cleanup_thread_func(chunk_idx, chunk_path, stage_name)
        th = None
    return th


def create_table_handle_exists(
    cursor,
    stage_name,
    location,
    df_columns,
    if_exists,
):
    """Automatically create a new table in Snowflake at the given location if
    it doesn't exist, following the schema of staged files.

    Args
        cursor (snowflake.connector.cursor): Snowflake connection cursor
        stage_name (str): Name of internal stage containing desired files
        location (str): Location to create a table
        df_columns (array-like): Array of dataframe column names
        if_exists (str): Action to take if table already exists:
            "fail": If table exists, raise a ValueError. Create if does not exist
            "replace": If table exists, drop it, recreate it, and insert data.
                Create if does not exist
            "append": If table exists, insert data. Create if does not exist

    Returns
        file_format_name (str): Name of created file format
    """
    ev = tracing.Event("create_table_if_not_exists", is_parallel=False)

    # Snowflake import
    try:
        import snowflake.connector
    except ImportError:
        raise BodoError(
            "Snowflake Python connector packages not found. "
            "Using 'to_sql' with Snowflake requires snowflake-connector-python. "
            "This can be installed by calling 'conda install -c conda-forge snowflake-connector-python' "
            "or 'pip install snowflake-connector-python'."
        )

    file_format_name = None  # forward declaration
    file_format_err = None  # forward declaration

    # Handle `if_exists`
    if if_exists == "fail":
        create_table_cmd = "CREATE TABLE"
    elif if_exists == "replace":
        create_table_cmd = "CREATE OR REPLACE TABLE"
    elif if_exists == "append":
        create_table_cmd = "CREATE TABLE IF NOT EXISTS"
    else:
        raise ValueError(f"'{if_exists}' is not valid for if_exists")

    # Create a new file format object that describes the data in staged files.
    # We will quickly generate a file format name that doesn't already exist within Snowflake.
    # An infinite loop here is extremely unlikely unless uuid4's are used up.
    ev_create_file_format = tracing.Event("create_file_format", is_parallel=False)
    while True:
        try:
            file_format_name = f"bodo_io_snowflake_write_{uuid4()}"
            file_format_sql = (
                f'CREATE FILE FORMAT "{file_format_name}" '
                f"TYPE=PARQUET COMPRESSION=AUTO "
                f"/* Python:bodo.io.snowflake.create_table_if_not_exists() */"
            )
            cursor.execute(file_format_sql, _is_internal=True)
            break
        except snowflake.connector.ProgrammingError as pe:
            if pe.msg.endswith("already exists."):
                continue
            file_format_err = pe.msg
            break
    ev_create_file_format.finalize()

    if file_format_err is not None:
        raise snowflake.connector.ProgrammingError(file_format_err)

    # Infer schema of staged files from file format object
    ev_infer_schema = tracing.Event("infer_schema", is_parallel=False)
    infer_schema_sql = (
        f"SELECT COLUMN_NAME, TYPE FROM table(infer_schema("
        f"location=>'@\"{stage_name}\"', file_format=>'\"{file_format_name}\"')) "
        f"/* Python:bodo.io.snowflake.create_table_if_not_exists() */"
    )
    column_type_mapping = dict(
        cursor.execute(infer_schema_sql, _is_internal=True).fetchall()
    )
    ev_infer_schema.finalize()

    # Infer schema can return the columns out of order depending on the
    # chunking we do when we upload, so we have to iterate through the
    # dataframe columns to make sure we create the table with its columns
    # in order.
    ev_create_table = tracing.Event("create_table", is_parallel=False)
    create_table_columns = ", ".join(
        [f'"{c}" {column_type_mapping[c]}' for c in df_columns]
    )
    create_table_sql = (
        f"{create_table_cmd} {location} ({create_table_columns}) "
        f"/* Python:bodo.io.snowflake.create_table_if_not_exists() */"
    )
    cursor.execute(create_table_sql, _is_internal=True)
    ev_create_table.finalize()

    # Drop temporary file format object
    ev_drop_file_format = tracing.Event("drop_file_format", is_parallel=False)
    drop_file_format_sql = (
        f'DROP FILE FORMAT IF EXISTS "{file_format_name}" '
        f"/* Python:bodo.io.snowflake.create_table_if_not_exists() */"
    )
    cursor.execute(drop_file_format_sql, _is_internal=True)
    ev_drop_file_format.finalize()

    ev.finalize()


def execute_copy_into(
    cursor,
    stage_name,
    location,
    df_columns,
):
    """Execute a COPY_INTO command from all files in stage to a table location.

    Args
        cursor (snowflake.connector.cursor): Snowflake connection cursor
        stage_name (str): Name of internal stage containing desired files
        location (str): Desired table location
        df_columns (array-like): Array of dataframe column names

    Returns (success, num_chunks, num_rows, output) where:
        success (bool): True if the function successfully wrote the data to the table
        num_chunks (int): Number of chunks of data that the function copied
        num_rows (int): Number of rows that the function inserted
        output (str): Output of the `COPY INTO <table>` command
    """
    ev = tracing.Event("execute_copy_into", is_parallel=False)

    columns = ",".join([f'"{c}"' for c in df_columns])

    # In Snowflake, all parquet data is stored in a single column, $1,
    # so we must select columns explicitly
    # See (https://docs.snowflake.com/en/user-guide/script-data-load-transform-parquet.html)
    parquet_columns = ",".join([f'$1:"{c}"' for c in df_columns])

    # Execute copy_into command with files from all ranks
    copy_into_sql = (
        f"COPY INTO {location} ({columns}) "
        f'FROM (SELECT {parquet_columns} FROM @"{stage_name}") '
        f"FILE_FORMAT=(TYPE=PARQUET COMPRESSION=AUTO) "
        f"PURGE=TRUE ON_ERROR={SF_WRITE_COPY_INTO_ON_ERROR} "
        f"/* Python:bodo.io.snowflake.execute_copy_into() */"
    )
    copy_results = cursor.execute(copy_into_sql, _is_internal=True).fetchall()

    # Compute diagnostic output
    nsuccess = sum(1 if e[1] == "LOADED" else 0 for e in copy_results)
    nchunks = len(copy_results)
    nrows = sum(int(e[3]) for e in copy_results)
    out = (nsuccess, nchunks, nrows, copy_results)

    ev.add_attribute("copy_into_nsuccess", nsuccess)
    ev.add_attribute("copy_into_nchunks", nchunks)
    ev.add_attribute("copy_into_nrows", nrows)

    if os.environ.get("BODO_SF_WRITE_DEBUG") is not None:
        print(f"[Snowflake Write] copy_into results: {repr(copy_results)}")
    ev.finalize()

    return out


# ------------------- Native Distributed Snowflake Write implementation -----------------
# Register opaque type for Snowflake Cursor so it can be shared between
# different sections of jitted code.
# Skip Python type registration if snowflake.connector is not installed,
# since this is an optional dependency.
try:
    import snowflake.connector

    snowflake_connector_cursor_python_type = snowflake.connector.cursor.SnowflakeCursor
except ImportError:
    snowflake_connector_cursor_python_type = None

SnowflakeConnectorCursorType = install_py_obj_class(
    types_name="snowflake_connector_cursor_type",
    python_type=snowflake_connector_cursor_python_type,
    module=sys.modules[__name__],
    class_name="SnowflakeConnectorCursorType",
    model_name="SnowflakeConnectorCursorModel",
)

# Register opaque type for TemporaryDirectory so it can be shared between
# different sections of jitted code
TemporaryDirectoryType = install_py_obj_class(
    types_name="temporary_directory_type",
    python_type=TemporaryDirectory,
    module=sys.modules[__name__],
    class_name="TemporaryDirectoryType",
    model_name="TemporaryDirectoryModel",
)


def get_snowflake_stage_info(cursor, stage_name, tmp_folder):
    """Get parquet path and credentials for a snowflake internal stage.
    This works by using `_execute_helper` to issue a dummy upload query

    Args
        cursor (snowflake.connector.cursor): Snowflake connection cursor
        stage_name (str): Stage name to query information about
        tmp_folder (TemporaryDirectory): A TemporaryDirectory() object
            representing a temporary directory on disk to store files
            prior to an upload

    Returns
        stage_info (Dict): Dictionary of snowflake stage info
    """
    ev = tracing.Event("get_snowflake_stage_info", is_parallel=False)

    # Create a unique filepath for dummy upload query with quotes/backslashes escaped
    query_path = os.path.join(tmp_folder.name, f"get_credentials_{uuid4()}.parquet")
    # To escape backslashes, we want to replace ( \ ) with ( \\ ), which can
    # be written as the string literals ( \\ ) and ( \\\\ ).
    # To escape quotes, we want to replace ( ' ) with ( \' ), which can
    # be written as the string literals ( ' ) and ( \\' ).
    query_path = query_path.replace("\\", "\\\\").replace("'", "\\'")

    # Run `_execute_helper` to get stage info dict from Snowflake
    upload_sql = (
        f"PUT 'file://{query_path}' @\"{stage_name}\" AUTO_COMPRESS=FALSE "
        f"/* Python:bodo.io.snowflake.get_snowflake_stage_info() */"
    )
    stage_info = cursor._execute_helper(upload_sql, is_internal=True)

    ev.finalize()
    return stage_info


def connect_and_get_upload_info(conn):
    """On rank 0, connect to Snowflake, create an internal stage, and issue
    an upload command to get parquet path and internal stage credentials.
    If the internal stage type is not supported or SF_WRITE_UPLOAD_USING_PUT
    is True, use the PUT implementation by connecting to Snowflake on all ranks.
    This function exists to be executed within objmode from `DataFrame.to_sql()`

    Args
        conn (str): Snowflake connection URL string

    Returns: (cursor, tmp_folder, stage_name, parquet_path, upload_using_snowflake_put, old_creds) where
        cursor (snowflake.connector.cursor): Snowflake connection cursor
        tmp_folder (TemporaryDirectory): A TemporaryDirectory() object
            representing a temporary directory on disk to store files
            prior to an upload
        stage_name (str): Name of created internal stage
        parquet_path (str): Parquet path of internal stage, either an S3/ADLS
            URI or a local path in the case of upload using PUT, with trailing slash
        upload_using_snowflake_put (bool): An updated boolean flag for whether
            we are using the PUT command in objmode to upload files. This is
            set to True if we don't support the stage type returned by Snowflake.
        old_creds (Dict(str, str)): Old environment variables that were
            overwritten to update credentials for uploading to stage
        azure_stage_direct_upload (boolean): Whether the stage is ADLS backed
            and we'll be writing parquet files to it directly using our existing
            hdfs and parquet infrastructure.
        old_core_site (str): In case azure_stage_direct_upload=True, we replace
            bodo.HDFS_CORE_SITE_LOC with a new core-site.xml. `old_core_site`
            contains the original contents of the file (or "__none__" if file
            didn't originally exist -- see bodo.io.helpers.update_file_contents
            for more details), so that it can be restored later during
            create_table_copy_into
        old_sas_token (str): In case azure_stage_direct_upload=True, we replace
            contents in SF_AZURE_WRITE_SAS_TOKEN_FILE_LOCATION (if any)
            with the SAS token for this upload. `old_sas_token`
            contains the original contents of the file (or "__none__" if file
            didn't originally exist -- see bodo.io.helpers.update_file_contents
            for more details), so that it can be restored later during
            create_table_copy_into
    """
    ev = tracing.Event("connect_and_get_upload_info")

    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()

    # Create a temporary directory on every rank
    tmp_folder = TemporaryDirectory()

    # On rank 0, create named internal stage and get stage info dict
    cursor = None  # Forward declaration
    stage_name = ""  # Forward declaration
    parquet_path = ""  # Forward declaration
    upload_creds = {}  # Forward declaration
    old_creds = {}  # Forward declaration
    old_core_site = ""  # Forward declaration
    sas_token = ""  # Forward declaration
    old_sas_token = ""  # Forward declaration

    err = None  # Forward declaration
    if my_rank == 0:
        try:
            # Connect to snowflake
            conn = snowflake_connect(conn)
            cursor = conn.cursor()
            # Avoid creating a temp stage at all in case of SF_WRITE_UPLOAD_USING_PUT
            is_temporary = not SF_WRITE_UPLOAD_USING_PUT
            stage_name = create_internal_stage(cursor, is_temporary=is_temporary)

            if SF_WRITE_UPLOAD_USING_PUT:
                # With this config option set, always upload using snowflake PUT.
                # An empty parquet path denotes fallback to objmode PUT below
                parquet_path = ""
            else:
                # Parse stage info dict for parquet path and credentials
                stage_info = get_snowflake_stage_info(cursor, stage_name, tmp_folder)
                upload_info = stage_info["data"]["uploadInfo"]

                location_type = upload_info.get("locationType", "UNKNOWN")
                fallback_to_put = False

                if location_type == "S3":
                    # Parquet path format: s3://<bucket_name>/<key_name>
                    # E.g. s3://sfc-va2-ds1-9-customer-stage/b9zr-s-v2st3620/stages/547e65a7-fa2c-491b-98c3-6e4313db7741/
                    # See https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-bucket-intro.html#accessing-a-bucket-using-S3-format
                    bucket_name, _, path = upload_info["location"].partition("/")
                    path = path.rstrip("/")

                    parquet_path = f"s3://{bucket_name}/{path}/"
                    upload_creds = {
                        "AWS_ACCESS_KEY_ID": upload_info["creds"]["AWS_KEY_ID"],
                        "AWS_SECRET_ACCESS_KEY": upload_info["creds"]["AWS_SECRET_KEY"],
                        "AWS_SESSION_TOKEN": upload_info["creds"]["AWS_TOKEN"],
                        "AWS_DEFAULT_REGION": upload_info["region"],
                    }
                elif location_type == "AZURE":

                    # We cannot upload directly to ADLS unless this package is installed,
                    # so check that first.
                    bodo_azurefs_sas_token_provider_installed = False
                    try:
                        import bodo_azurefs_sas_token_provider  # noqa

                        bodo_azurefs_sas_token_provider_installed = True
                    except ImportError:
                        pass

                    # Also check for environment variables such as HADOOP_HOME, ARROW_LIBHDFS_DIR
                    # and CLASSPATH, since if those are not set, the pq_write would fail anyway.
                    hadoop_env_vars_set = (
                        (len(os.environ.get("HADOOP_HOME", "")) > 0)
                        and (len(os.environ.get("ARROW_LIBHDFS_DIR", "")) > 0)
                        # CLASSPATH should be initialized by bodo_azurefs_sas_token_provider even if it
                        # didn't originally exist, but doesn't hurt to check.
                        and (len(os.environ.get("CLASSPATH", "")) > 0)
                    )

                    if (
                        bodo_azurefs_sas_token_provider_installed
                        and hadoop_env_vars_set
                    ):
                        # Upload path format: abfs[s]://<file_system>@<account_name>.dfs.core.windows.net/<path>/<file_name>
                        # E.g. abfs://stageszz05dc579c-e473-4aa2-b8a3-62a1ae425a11@qiavr8sfcb1stg.dfs.core.windows.net/<file_name>
                        # For URI syntax, see https://docs.microsoft.com/en-us/azure/storage/blobs/data-lake-storage-introduction-abfs-uri#uri-syntax
                        container_name, _, path = upload_info["location"].partition("/")
                        path = path.rstrip("/")

                        account_name = upload_info["storageAccount"]
                        sas_token = upload_info["creds"]["AZURE_SAS_TOKEN"].lstrip("?")

                        if len(path) == 0:
                            parquet_path = f"abfs://{container_name}@{account_name}.dfs.core.windows.net/"
                        else:
                            parquet_path = f"abfs://{container_name}@{account_name}.dfs.core.windows.net/{path}/"

                        if not ("BODO_PLATFORM_WORKSPACE_UUID" in os.environ):
                            # Since setting up Hadoop can be notoriously difficult, we warn the users that
                            # they may need to fall back to the PUT method manually in case of failure.
                            # Note that this shouldn't be an issue on our platform where Hadoop is set
                            # up correctly, and is meant for on-prem users. BODO_PLATFORM_WORKSPACE_UUID
                            # should be an environment variable on all platform clusters, which is why we
                            # use this as a heuristic for showing this warning.
                            warnings.warn(
                                BodoWarning(
                                    "Detected Azure Stage. Bodo will try to upload to the stage directly. "
                                    "If this fails, there might be issues with your Hadoop configuration "
                                    "and you may need to use the PUT method instead by setting\n"
                                    "import bodo\n"
                                    "bodo.io.snowflake.SF_WRITE_UPLOAD_USING_PUT = True\n"
                                    "before calling this function."
                                )
                            )

                    else:
                        # If the package doesn't exist or one of the required hadoop env vars is not set,
                        # fall back to PUT, but show a warning so users can install the package
                        # and/or set up hadoop for next time.
                        fallback_to_put = True
                        warning_msg = "Detected Azure Stage. "
                        if not bodo_azurefs_sas_token_provider_installed:
                            warning_msg += (
                                "Required package bodo_azurefs_sas_token_provider "
                                "is not installed. To use direct upload to stage in the future, install the package using: "
                                "'conda install bodo-azurefs-sas-token-provider -c bodo.ai -c conda-forge'.\n"
                            )
                        if not hadoop_env_vars_set:
                            warning_msg += (
                                "You need to download and set up Hadoop. For more information, "
                                "refer to our documentation: https://docs.bodo.ai/latest/file_io/?h=hdfs#HDFS.\n"
                            )
                        warning_msg += "Falling back to PUT command for upload for now."
                        warnings.warn(BodoWarning(warning_msg))

                else:
                    # Unsupported internal stage location. This code falls back to objmode upload
                    fallback_to_put = True
                    warnings.warn(
                        BodoWarning(
                            f"Direct upload to stage is not supported for internal stage "
                            f"type '{location_type}'. Falling back to PUT "
                            f"command for upload."
                        )
                    )

                if fallback_to_put:
                    # If falling back to PUT method, drop this stage and create a non-temporary
                    # stage instead.
                    drop_internal_stage(cursor, stage_name)
                    stage_name = create_internal_stage(cursor, is_temporary=False)

        except Exception as e:
            err = RuntimeError(str(e))
            if os.environ.get("BODO_SF_WRITE_DEBUG") is not None:
                print("".join(traceback.format_exception(None, e, e.__traceback__)))

    err = comm.bcast(err)
    if isinstance(err, Exception):
        raise err

    parquet_path = comm.bcast(parquet_path)
    azure_stage_direct_upload = parquet_path.startswith("abfs://")

    if parquet_path == "":
        # Falling back to PUT for upload. The internal stage type could be
        # unsupported, or the `upload_using_snowflake_put` flag could be set to True.
        upload_using_snowflake_put = True
        parquet_path = tmp_folder.name + "/"

        # Objmode PUT requires a Snowflake connection on all ranks, not just rank 0
        if my_rank != 0:
            # Since we already connected to Snowflake successfully on rank 0,
            # unlikely we'll have an exception here.
            conn = snowflake_connect(conn)
            cursor = conn.cursor()

    else:
        upload_using_snowflake_put = False

        # On all ranks, update environment variables with internal stage credentials
        upload_creds = comm.bcast(upload_creds)
        old_creds = update_env_vars(upload_creds)

        if azure_stage_direct_upload:
            # parquet_path will be start with abfs:// if this import worked on rank-0,
            # so it should be safe to do it here on all other ranks.
            # This adds the required jars to the CLASSPATH.
            import bodo_azurefs_sas_token_provider  # noqa

            # If writing directly to an ADLS stage, we need to initialize
            # the directory for core-site and the core-site itself.
            bodo.HDFS_CORE_SITE_LOC_DIR.initialize()
            # We want to get original the contents of the files
            # (core-site.xml and sas_token.txt) so that
            # we can restore them later.
            # This happens in `create_table_copy_into`.
            old_core_site = update_file_contents(
                bodo.HDFS_CORE_SITE_LOC, SF_AZURE_WRITE_HDFS_CORE_SITE
            )
            sas_token = comm.bcast(sas_token)
            old_sas_token = update_file_contents(
                SF_AZURE_WRITE_SAS_TOKEN_FILE_LOCATION, sas_token
            )

    stage_name = comm.bcast(stage_name)

    ev.finalize()
    return (
        cursor,
        tmp_folder,
        stage_name,
        parquet_path,
        upload_using_snowflake_put,
        old_creds,
        azure_stage_direct_upload,
        old_core_site,
        old_sas_token,
    )


def create_table_copy_into(
    cursor,
    stage_name,
    location,
    df_columns,
    if_exists,
    old_creds,
    tmp_folder,
    azure_stage_direct_upload,
    old_core_site,
    old_sas_token,
):
    """Auto-create a new table if needed, execute COPY_INTO, and clean up
    created internal stage, and restore old environment variables.
    This function exists to be executed within objmode from `DataFrame.to_sql()`

    Args
        cursor (snowflake.connector.cursor): Snowflake connection cursor
        stage_name (str): Name of internal stage containing files to copy_into
        location (str): Destination table location
        df_columns (array-like): Array of dataframe column names
        if_exists (str): Action to take if table already exists:
            "fail": If table exists, raise a ValueError. Create if does not exist
            "replace": If table exists, drop it, recreate it, and insert data.
                Create if does not exist
            "append": If table exists, insert data. Create if does not exist
        old_creds (Dict(str, str or None)): Old environment variables to restore.
            Previously overwritten to update credentials for uploading to stage
        tmp_folder (TemporaryDirectory): TemporaryDirectory object to clean up
        azure_stage_direct_upload (boolean): Whether the stage is ADLS backed
            and we wrote parquet files to it directly using our existing
            hdfs and parquet infrastructure.
        old_core_site (str): In case azure_stage_direct_upload=True, we replaced
            bodo.HDFS_CORE_SITE_LOC with a new core-site.xml in
            connect_and_get_upload_info. `old_core_site` contains the original
            contents of the file (or "__none__" if file didn't originally
            exist -- see bodo.io.helpers.update_file_contents
            for more details), which we'll restore in this function.
        old_sas_token (str): In case azure_stage_direct_upload=True, we replaced
            contents of SF_AZURE_WRITE_SAS_TOKEN_FILE_LOCATION with a new SAS
            token in connect_and_get_upload_info. `old_sas_token` contains the
            original contents of the file (or "__none__" if file didn't
            originally exist -- see bodo.io.helpers.update_file_contents
            for more details), which we'll restore in this function.
    """
    ev = tracing.Event("create_table_copy_into", is_parallel=False)

    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()

    # On rank 0, create a new table if needed, then execute COPY_INTO
    err = None  # Forward declaration
    if my_rank == 0:
        try:
            begin_transaction_sql = (
                "BEGIN /* Python:bodo.io.snowflake.create_table_copy_into() */"
            )
            cursor.execute(begin_transaction_sql)

            create_table_handle_exists(
                cursor,
                stage_name,
                location,
                df_columns,
                if_exists,
            )
            (nsuccess, nchunks, nrows, copy_into_result) = execute_copy_into(
                cursor,
                stage_name,
                location,
                df_columns,
            )

            if nsuccess != nchunks:
                raise BodoError(f"Snowflake write copy_into failed: {copy_into_result}")

            commit_transaction_sql = (
                "COMMIT /* Python:bodo.io.snowflake.create_table_copy_into() */"
            )
            cursor.execute(commit_transaction_sql)

            drop_internal_stage(cursor, stage_name)

            cursor.close()

        except Exception as e:
            err = RuntimeError(str(e))
            if os.environ.get("BODO_SF_WRITE_DEBUG") is not None:
                print("".join(traceback.format_exception(None, e, e.__traceback__)))

    err = comm.bcast(err)
    if isinstance(err, Exception):
        raise err

    # Put back the environment variables
    update_env_vars(old_creds)

    # Cleanup the folder that was created to store parquet chunks for upload
    tmp_folder.cleanup()

    # azure_stage_direct_upload will be true if direct upload to Azure was used.
    # If it was, restore the contents. It is highly unlikely
    # that there was actual contents, and in most cases the file
    # will just be removed.
    if azure_stage_direct_upload:
        update_file_contents(bodo.HDFS_CORE_SITE_LOC, old_core_site)
        update_file_contents(SF_AZURE_WRITE_SAS_TOKEN_FILE_LOCATION, old_sas_token)

    ev.finalize()
