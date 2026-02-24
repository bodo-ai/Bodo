"""
Tests for writing to Snowflake using Python APIs
"""

import time
import uuid

import pandas as pd
import pytest
from mpi4py import MPI

import bodo
from bodo.tests.utils import (  # noqa
    get_snowflake_connection_string,
    pytest_perf_regression,
)

# Skip for all CI
pytestmark = pytest_perf_regression


@pytest.fixture(
    params=[
        pytest.param(True, id="verbose"),
        # pytest.param(False, id="quiet"),
    ]
)
def verbose(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(1, id="s3"),
        pytest.param(3, id="adls"),
    ]
)
def snowflake_user(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param("DEMO_WH", id="X-Small"),
        pytest.param("SMALL_WH", id="Small"),
        pytest.param("TEST_TPCDS", id="Medium"),
        pytest.param("LARGE_WH_2", id="Large"),
        pytest.param("TEST_WH", id="X-Large"),
    ]
)
def warehouse_name(request):
    return request.param


@pytest.fixture(
    params=[
        "TPCH_SF1",
        "TPCH_SF10",
        "TPCH_SF100",
    ]
)
def table_name(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(int(256e6), id="write_chunk_256e6"),
    ]
)
def sf_write_chunk_size(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(4096, id="read_chunk_4096"),
    ]
)
def sf_read_chunk_size(request):
    return request.param


@pytest.fixture(
    params=[
        # pytest.param(64, id="num_files_64"),
        pytest.param(128, id="num_files_128"),
        # pytest.param(256, id="num_files_256"),
    ]
)
def sf_write_num_files(request):
    return request.param


@pytest.fixture(
    params=[
        # pytest.param(True, id="with-put"),
        pytest.param(False, id="no-put")
    ]
)
def sf_write_use_put(request):
    return request.param


@pytest.mark.parametrize("it", [0, 1, 2])
def test_streaming_write(
    it,
    snowflake_user,
    warehouse_name,
    table_name,
    sf_write_chunk_size,
    sf_read_chunk_size,
    sf_write_num_files,
    sf_write_use_put,
    verbose,
):
    """
    Benchmark a simple use of streaming Snowflake writes by reading a table, writing
    the results, then reading again
    """
    import bodo.decorators  # isort:skip # noqa
    import bodo.io.snowflake
    from bodo.io.arrow_reader import read_arrow_next
    from bodo.io.snowflake_write import (
        snowflake_writer_append_table,
        snowflake_writer_init,
    )

    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "l_orderkey",
            "l_partkey",
            "l_suppkey",
            "l_linenumber",
            "l_quantity",
            "l_extendedprice",
            "l_discount",
            "l_tax",
            "l_returnflag",
            "l_linestatus",
            "l_shipdate",
            "l_commitdate",
            "l_receiptdate",
            "l_shipinstruct",
            "l_shipmode",
            "l_comment",
        )
    )

    conn_r = bodo.tests.utils.get_snowflake_connection_string(
        "SNOWFLAKE_SAMPLE_DATA", table_name, user=snowflake_user
    ).replace("DEMO_WH", warehouse_name)
    conn_w = bodo.tests.utils.get_snowflake_connection_string(
        "TEST_DB", "PUBLIC", user=snowflake_user
    ).replace("DEMO_WH", warehouse_name)
    if bodo.get_rank() == 0:
        cursor_w = bodo.io.snowflake.snowflake_connect(conn_w).cursor()

    table_r = "LINEITEM"
    table_w = None  # Forward declaration
    if bodo.get_rank() == 0:
        table_w = f'"LINEITEM_TEST_{uuid.uuid4()}"'
    table_w = MPI.COMM_WORLD.bcast(table_w)

    # To test multiple COPY INTO, temporarily reduce Parquet write chunk size
    # and the number of files included in each streaming COPY INTO
    old_use_put = bodo.io.snowflake.SF_WRITE_UPLOAD_USING_PUT
    old_chunk_size = bodo.io.snowflake.SF_WRITE_PARQUET_CHUNK_SIZE
    old_streaming_num_files = bodo.io.snowflake.SF_WRITE_STREAMING_NUM_FILES
    try:
        bodo.io.snowflake.SF_WRITE_UPLOAD_USING_PUT = sf_write_use_put
        bodo.io.snowflake.SF_WRITE_PARQUET_CHUNK_SIZE = sf_write_chunk_size
        bodo.io.snowflake.SF_WRITE_STREAMING_COPY_INTO_FILES = sf_write_num_files
        ctas_meta = bodo.utils.typing.CreateTableMetaType()

        @bodo.jit(cache=False, distributed=["table", "reader0", "writer"])
        def impl_write(conn_r, conn_w):
            t_start = time.time()

            reader0 = pd.read_sql(
                f"SELECT * FROM {table_r}", conn_r, _bodo_chunksize=sf_read_chunk_size
            )  # type: ignore
            writer = snowflake_writer_init(
                -1,
                conn_w,
                table_w,
                "PUBLIC",
                "replace",
                "",
            )

            all_is_last = False
            it = 1
            t_stream_loop_write = time.time()

            while not all_is_last:
                t_read_next = time.time()
                table, is_last = read_arrow_next(reader0, True)
                t_read_next = time.time() - t_read_next
                if verbose and t_read_next >= 5e-4:
                    print(
                        f"Rank {bodo.get_rank()}, it={it}: t_read_next={t_read_next:.3f}s"
                    )

                t_sync_is_last = time.time()
                all_is_last = bodo.libs.distributed_api.sync_is_last(is_last, it)
                t_sync_is_last = time.time() - t_sync_is_last
                if verbose and t_sync_is_last >= 5e-4:
                    print(
                        f"Rank {bodo.get_rank()}, it={it}: t_sync_is_last={t_sync_is_last:.3f}s"
                    )

                t_writer_append = time.time()
                snowflake_writer_append_table(
                    writer, table, col_meta, all_is_last, None, ctas_meta
                )
                t_writer_append = time.time() - t_writer_append
                if verbose and t_writer_append >= 5e-4:
                    print(
                        f"Rank {bodo.get_rank()}, it={it}: t_writer_append={t_writer_append:.3f}s"
                    )

                it += 1

            t_stream_loop_write = time.time() - t_stream_loop_write
            if verbose and t_stream_loop_write >= 5e-4:
                print(
                    f"Rank {bodo.get_rank()}, niters={it}: t_stream_loop_write={t_stream_loop_write:.3f}s"
                )

            t_end = time.time()
            return t_end - t_start

        write_time = impl_write(conn_r, conn_w)
        print(f"Streaming R/W time={write_time:.3f}s")

    finally:
        bodo.io.snowflake.SF_WRITE_UPLOAD_USING_PUT = old_use_put
        bodo.io.snowflake.SF_WRITE_PARQUET_CHUNK_SIZE = old_chunk_size
        bodo.io.snowflake.SF_WRITE_STREAMING_NUM_FILES = old_streaming_num_files

        if bodo.get_rank() == 0:
            cleanup_table_sql = (
                f"DROP TABLE IF EXISTS {table_w} "
                "/* tests.test_sql:test_streaming_write() */"
            )
            cursor_w.execute(cleanup_table_sql, _is_internal=True).fetchall()
            cursor_w.close()


@pytest.mark.parametrize("it", [0, 1, 2])
def test_nonstream_write(
    it,
    warehouse_name,
    table_name,
    sf_write_use_put,
    snowflake_user,
    verbose,
):
    """
    Test a simple use of non-streaming Snowflake writes by reading a table,
    writing the results, then reading again
    """

    conn_r = bodo.tests.utils.get_snowflake_connection_string(
        "SNOWFLAKE_SAMPLE_DATA", table_name, user=snowflake_user
    ).replace("DEMO_WH", warehouse_name)
    conn_w = bodo.tests.utils.get_snowflake_connection_string(
        "TEST_DB", "PUBLIC", user=snowflake_user
    ).replace("DEMO_WH", warehouse_name)
    if bodo.get_rank() == 0:
        cursor_w = bodo.io.snowflake.snowflake_connect(conn_w).cursor()

    table_r = "LINEITEM"
    table_w = None  # Forward declaration
    if bodo.get_rank() == 0:
        table_w = f'"LINEITEM_TEST_{uuid.uuid4()}"'
    table_w = MPI.COMM_WORLD.bcast(table_w)

    # To test multiple COPY INTO, temporarily reduce Parquet write chunk size
    # and the number of files included in each streaming COPY INTO
    old_use_put = bodo.io.snowflake.SF_WRITE_UPLOAD_USING_PUT
    try:
        bodo.io.snowflake.SF_WRITE_UPLOAD_USING_PUT = sf_write_use_put

        @bodo.jit(cache=False, distributed=["df"])
        def impl_write(conn_r, conn_w):
            t_start = time.time()

            t_read = time.time()
            df = pd.read_sql(f"SELECT * FROM {table_r}", conn_r)  # type: ignore
            t_read = time.time() - t_read
            if verbose and t_read >= 5e-4:
                print(f"Rank {bodo.get_rank()}: t_read={t_read:.3f}s")

            t_write = time.time()
            df.to_sql(
                name=table_w,
                con=conn_w,
                schema="PUBLIC",
                if_exists="replace",
                _bodo_create_table_type="",
            )
            t_write = time.time() - t_write
            if verbose and t_write >= 5e-4:
                print(f"Rank {bodo.get_rank()}: t_write={t_write:.3f}s")

            t_end = time.time()
            return df, t_end - t_start

        _, t_total = impl_write(conn_r, conn_w)
        print(f"Non-streaming R/W Time={t_total:.3f}s")

        bodo.barrier()

    finally:
        bodo.io.snowflake.SF_WRITE_UPLOAD_USING_PUT = old_use_put

        if bodo.get_rank() == 0:
            cleanup_table_sql = (
                f"DROP TABLE IF EXISTS {table_w} "
                f"/* tests.test_sql:test_nonstream_write() */"
            )
            cursor_w.execute(cleanup_table_sql, _is_internal=True).fetchall()
            cursor_w.close()
