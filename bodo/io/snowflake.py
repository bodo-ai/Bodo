import os
import sys
import warnings
from tempfile import TemporaryDirectory
from urllib.parse import parse_qsl, urlparse
from uuid import uuid4

import pyarrow as pa
from mpi4py import MPI

import bodo
from bodo.io.helpers import ExceptionPropagatingThread, update_env_vars
from bodo.utils import tracing
from bodo.utils.py_objs import install_py_obj_class
from bodo.utils.typing import BodoError, BodoWarning

# A configurable variable by which we determine whether to dictionary-encode
# a string column.
# Encode if num of unique elem / num of total rows <= DICT_ENCODE_CRITERION
DICT_ENCODE_CRITERION = 0.5
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

    comm = MPI.COMM_WORLD

    # connect to Snowflake. This is the same connection that will be used
    # to read data.
    # We only trace on rank 0 (is_parallel=False) because we want 0 to start
    # executing the queries as soon as possible (don't sync event)
    conn = snowflake_connect(conn_str)

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

                if upload_info["locationType"] == "S3":
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

                else:
                    # Unsupported internal stage location. This code falls back to objmode upload
                    warnings.warn(
                        BodoWarning(
                            f"Direct upload to stage is not supported for internal stage "
                            f"type '{upload_info['locationType']}'. Falling back to PUT "
                            f"command for upload."
                        )
                    )

                    drop_internal_stage(cursor, stage_name)
                    stage_name = create_internal_stage(cursor, is_temporary=False)

        except Exception as e:
            err = e

    err = comm.bcast(err)
    if isinstance(err, Exception):
        raise err

    parquet_path = comm.bcast(parquet_path)

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

    stage_name = comm.bcast(stage_name)

    ev.finalize()
    return (
        cursor,
        tmp_folder,
        stage_name,
        parquet_path,
        upload_using_snowflake_put,
        old_creds,
    )


def create_table_copy_into(
    cursor,
    stage_name,
    location,
    df_columns,
    if_exists,
    old_creds,
    tmp_folder,
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
            err = e

    err = comm.bcast(err)
    if isinstance(err, Exception):
        raise err

    update_env_vars(old_creds)

    tmp_folder.cleanup()

    ev.finalize()
