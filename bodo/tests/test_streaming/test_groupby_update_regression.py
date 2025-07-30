"""
Tests for checking regression in streaming accumulated groupby.
This is meant to detect serious regressions in single-core
performance, but can be used for the multi-core case as well.
To check for regressions, run with your base branch, collect
the numbers from the RelNode timers, and compare them against
your branch. In particular, look for "Groupby build took ..."
which represents the build time.
The test uses the "bodopartner.us-east-1" Snowflake account
(re-uses the get_snowflake_connection_string infrastructure)
and therefore requires setting the following environment variables:
AGENT_NAME (to anything), SF_USERNAME, SF_PASSWORD, and
BODO_RUN_REGRESSION_TESTS (to anything).
After setting these, run using:
pytest -s -v test_stream_groupby_acc_regression.py
We recommend running this on a single-node cluster on the
platform to ensure reproducible results. A single
c5n.18xlarge instance was used during development, but
c5n.9xlarge should suffice for SF10. For best IO performance,
use an AWS workspace in the us-east-1 region, but us-east-2
should be fine as well.

In case the underlying streaming APIs change or there's a
meaningful change in the code structure, we might need
to re-create this test. To do so, we just need to generate
the Pandas code for the SQL query (mentioned in the docstring
of the 'impl' function) using BodoSQL and then apply the
changes marked with "# Codegen change:" in this file.
To generate the pandas code, we can use the BodoSQLWrapper
with the "--pandas_out_filename <output_file_location>"
command line argument. Alternatively, we can set up a
BodoSQLContext manually, and then use `bc.convert_to_pandas`
to generate the pandas code. For both of these, we
need to use the "bodopartner.us-east-1" Snowflake account
as mentioned above, and use the "SNOWFLAKE_SAMPLE_DATA"
database and "TPCH_SF10" schema.
"""

import time

import pandas as pd

import bodo
from bodo.tests.utils import (
    get_snowflake_connection_string,
    pytest_mark_snowflake,
    pytest_perf_regression,
)
from bodo.utils.typing import ColNamesMetaType, MetaType

# Skip for all CI
pytestmark = pytest_perf_regression

global_2 = MetaType((0, 1, 2))
global_3 = MetaType((1, 2))
global_1 = MetaType((0,))
global_4 = MetaType(("mean", "sum"))
global_5 = ColNamesMetaType(("l_suppkey", "EXPR$1", "EXPR$2"))


@bodo.jit
def impl(conn_str):  # Codegen change: add conn_str
    """
    Simple read from Snowflake followed by a group by.
    The equivalent SQL query is:
        select
            l_suppkey,
            avg(l_quantity),
            sum(l_extendedprice)
        from
            lineitem
            group by 1


    The query essentially reads one TPCH table and performs aggregations.
    We've chose columns that cover different
    data types such as integers and floats.
    To avoid planner based differences, we're using a
    slightly modified version of pandas code generated
    by BodoSQL directly.
    In particular, we've added timers around the build
    and concat steps.
    """
    # Codegen change: Add print and overall timer
    print("Started executing query...")
    t0 = time.time()
    __bodo_is_last_streaming_output_1 = False
    _iter_1 = 0
    _temp4 = 0.0
    _temp5 = time.time()
    state_1 = pd.read_sql(
        'SELECT "L_SUPPKEY", "L_QUANTITY", "L_EXTENDEDPRICE" FROM "LINEITEM"',
        conn_str,  # Codegen change: use conn_str
        _bodo_is_table_input=False,
        _bodo_chunksize=4096,
        _bodo_read_as_table=True,
    )
    # Codegen change: timer for build step
    build_time = 0.0
    _temp7 = time.time()
    _temp8 = _temp7 - _temp5
    _temp4 = _temp4 + _temp8
    _produce_output_1 = True
    _temp1 = 0.0
    _temp2 = time.time()
    state_2 = bodo.libs.streaming.groupby.init_groupby_state(
        -1, global_1, global_4, global_2, global_3
    )
    _temp12 = time.time()
    _temp13 = _temp12 - _temp2
    _temp1 = _temp1 + _temp13
    _temp14 = False
    while not (_temp14):
        _temp6 = time.time()
        (
            T1,
            __bodo_is_last_streaming_output_1,
        ) = bodo.io.arrow_reader.read_arrow_next(state_1, _produce_output_1)
        _temp9 = time.time()
        _temp10 = _temp9 - _temp6
        _temp4 = _temp4 + _temp10
        _temp3 = time.time()
        # Codegen change: Track build time
        t_build = time.time()
        _temp14 = bodo.libs.streaming.groupby.groupby_build_consume_batch(
            state_2, T1, __bodo_is_last_streaming_output_1, True
        )
        build_time += time.time() - t_build
        _temp15 = time.time()
        _temp16 = _temp15 - _temp3
        _temp1 = _temp1 + _temp16
        _iter_1 = _iter_1 + 1
    bodo.io.arrow_reader.arrow_reader_del(state_1)
    _temp11 = 'SELECT "L_SUPPKEY", "L_QUANTITY", "L_EXTENDEDPRICE" FROM "LINEITEM"'
    bodo.user_logging.log_message(
        "IO TIMING", f"""Execution time for reading table {_temp11}: {_temp4}"""
    )
    # Codegen change: print build timer
    bodo.user_logging.log_message("RELNODE_TIMING", f"Groupby build took {build_time}s")
    # Codegen change: timer for production and concat steps
    production_time = 0.0
    concat_time = 0.0
    _temp17 = False
    _iter_2 = 0
    _produce_output_2 = True
    __bodo_streaming_batches_table_builder_1 = (
        bodo.libs.table_builder.init_table_builder_state(2)
    )
    while not (_temp17):
        _temp3 = time.time()
        # Codegen change: track output production time
        t_production = time.time()
        (
            T2,
            _temp17,
        ) = bodo.libs.streaming.groupby.groupby_produce_output_batch(
            state_2, _produce_output_2
        )
        production_time += time.time() - t_production
        _temp18 = time.time()
        _temp19 = _temp18 - _temp3
        _temp1 = _temp1 + _temp19
        # Codegen change: track concat time
        t_concat = time.time()
        bodo.libs.table_builder.table_builder_append(
            __bodo_streaming_batches_table_builder_1, T2
        )
        concat_time += time.time() - t_concat
        _iter_2 = _iter_2 + 1
    bodo.libs.streaming.groupby.delete_groupby_state(state_2)
    _temp20 = "PandasAggregate(group=[{0}], EXPR$1=[AVG($1)], EXPR$2=[SUM($2)])"
    bodo.user_logging.log_message(
        "RELNODE_TIMING", f"""Execution time for RelNode {_temp20}: {_temp1}"""
    )
    # Codegen change: print ouput production and concat time
    bodo.user_logging.log_message(
        "RELNODE_TIMING", f"Groupby output production took {production_time}s"
    )
    t_concat = time.time()
    T3 = bodo.libs.table_builder.table_builder_finalize(
        __bodo_streaming_batches_table_builder_1
    )
    index_1 = bodo.hiframes.pd_index_ext.init_range_index(0, len(T3), 1, None)
    df1 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T3,), index_1, global_5)
    concat_time += time.time() - t_concat
    bodo.user_logging.log_message("RELNODE_TIMING", f"Concat Time: {concat_time}")
    # Codegen change: print overall execution time
    print(f"Finished executing the query. It took {time.time() - t0} seconds.")
    print("Output shape: ", df1.shape)
    return df1


# Codegen change: call the function
@pytest_mark_snowflake
def test_simple_groupby_update(verbose_mode_on):
    # Get snowflake connection string from environment variables
    # (SF_USERNAME and SF_PASSWORD).
    # Use SF10 since that has sufficient data and compute,
    # but we can change the schema to TPCH_SF1, TPCH_SF100, etc.
    # to test different data sizes.
    conn_str = get_snowflake_connection_string("SNOWFLAKE_SAMPLE_DATA", "TPCH_SF10")
    impl(conn_str)
