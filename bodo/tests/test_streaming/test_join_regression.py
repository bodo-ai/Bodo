"""
Tests for checking regression in streaming hash join.
This is meant to detect serious regressions in single-core
performance, but can be used for the multi-core case as well.
To check for regressions, run with your base branch, collect
the numbers from the RelNode timers, and compare them against
your branch. In particular, look for "Join build took ..."
and "Join probe took ..." which represent the build and
probe times respectively.
The test uses the "bodopartner.us-east-1" Snowflake account
(re-uses the get_snowflake_connection_string infrastructure)
and therefore requires setting the following environment variables:
AGENT_NAME (to anything), SF_USERNAME, SF_PASSWORD, and
BODO_RUN_REGRESSION_TESTS (to anything).
After setting these, run using:
pytest -s -v test_stream_join_regression.py
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
import pytest

import bodo
from bodo.tests.utils import (
    get_snowflake_connection_string,
    pytest_mark_snowflake,
    pytest_perf_regression,
)
from bodo.utils.typing import ColNamesMetaType, MetaType

# Skip for all CI
pytestmark = pytest_perf_regression

# Codegen change: turn verbose mode on
# bodo.set_verbose_level(2)

global_2 = MetaType((0,))
global_3 = ColNamesMetaType(("O_ORDERKEY", "O_CUSTKEY", "O_ORDERPRIORITY", "O_COMMENT"))
global_1 = MetaType((0, 1, 5, 8))
global_6 = MetaType((4, 5, 6, 7, 8, 2, 3, 1))
global_7 = ColNamesMetaType(
    (
        "o_orderpriority",
        "o_comment",
        "o_custkey",
        "l_orderkey",
        "l_extendedprice",
        "l_shipinstruct",
        "l_shipmode",
        "l_comment",
    )
)
global_4 = ColNamesMetaType(
    ("L_ORDERKEY", "L_EXTENDEDPRICE", "L_SHIPINSTRUCT", "L_SHIPMODE", "L_COMMENT")
)
global_5 = MetaType((0, 5, 13, 14, 15))

# Codegen change: Global variables for setting build-outer and probe-outer flags.
global_build_outer = False
global_probe_outer = False
build_interval_cols = bodo.utils.typing.MetaType(())


def impl(conn_str):  # Codegen change: add conn_str
    """
    Simple read from Snowflake followed by a hash join.
    The equivalent SQL query is:
        select
            o_orderpriority,
            o_comment,
            o_custkey,
            l_orderkey,
            l_extendedprice,
            l_shipinstruct,
            l_shipmode,
            l_comment
        from
            lineitem,
            orders
        where
            l_orderkey = o_orderkey
    The query essentially reads two TPCH tables and
    joins them. We've chose columns that cover different
    data types such as integers, floats, strings and
    dictionary-encoded strings.
    To avoid planner based differences, we're using a
    slightly modified version of pandas code generated
    by BodoSQL directly.
    In particular, we've added timers around the build
    and probe steps.
    """
    # Codegen change: Add print and overall timer
    print("Started executing query...")
    t0 = time.time()
    __bodo_is_last_streaming_output_1 = False
    _iter_1 = 0
    _temp4 = 0.0
    _temp5 = time.time()
    state_1 = pd.read_sql(
        "ORDERS",
        conn_str,  # Codegen change: use conn_str
        _bodo_is_table_input=True,
        _bodo_chunksize=4096,
        _bodo_read_as_table=True,
    )
    # Codegen change: timer for build step
    build_time = 0.0
    _temp7 = time.time()
    _temp8 = _temp7 - _temp5
    _temp4 = _temp4 + _temp8
    _temp12 = 0.0
    _temp13 = time.time()
    state_2 = bodo.libs.streaming.dict_encoding.init_dict_encoding_state()
    _temp15 = time.time()
    _temp16 = _temp15 - _temp13
    _temp12 = _temp12 + _temp16
    _temp1 = 0.0
    _temp2 = time.time()
    # Codegen change: Use global flags for build-outer and probe-outer
    state_3 = bodo.libs.streaming.join.init_join_state(
        -1,
        global_2,
        global_2,
        global_3,
        global_4,
        global_build_outer,
        global_probe_outer,
        False,
        build_interval_cols,
        -1,
    )
    _temp20 = time.time()
    _temp21 = _temp20 - _temp2
    _temp1 = _temp1 + _temp21
    _temp22 = False
    while not (_temp22):
        _temp6 = time.time()
        (
            T1,
            __bodo_is_last_streaming_output_1,
        ) = bodo.io.arrow_reader.read_arrow_next(state_1, True)
        _temp9 = time.time()
        _temp10 = _temp9 - _temp6
        _temp4 = _temp4 + _temp10
        _temp14 = time.time()
        T2 = bodo.hiframes.table.table_subset(T1, global_1, False)
        _temp17 = time.time()
        _temp18 = _temp17 - _temp14
        _temp12 = _temp12 + _temp18
        _temp3 = time.time()
        # Codegen change: Track build time
        t_build = time.time()
        _temp22, _ = bodo.libs.streaming.join.join_build_consume_batch(
            state_3, T2, __bodo_is_last_streaming_output_1
        )
        build_time += time.time() - t_build
        _temp23 = time.time()
        _temp24 = _temp23 - _temp3
        _temp1 = _temp1 + _temp24
        _iter_1 = _iter_1 + 1
    bodo.io.arrow_reader.arrow_reader_del(state_1)
    _temp11 = "ORDERS"
    bodo.user_logging.log_message(
        "IO TIMING", f"""Execution time for reading table {_temp11}: {_temp4}"""
    )
    bodo.libs.streaming.dict_encoding.delete_dict_encoding_state(state_2)
    _temp19 = "PandasProject(O_ORDERKEY=[$0], O_CUSTKEY=[$1], O_ORDERPRIORITY=[$5], O_COMMENT=[$8])"
    bodo.user_logging.log_message(
        "RELNODE_TIMING", f"""Execution time for RelNode {_temp19}: {_temp12}"""
    )
    # Codegen change: print build timer
    bodo.user_logging.log_message("RELNODE_TIMING", f"Join build took {build_time}s")
    print(
        "Join OperatorPool bytes_pinned: ",
        bodo.libs.streaming.join.get_op_pool_bytes_pinned(state_3),
    )
    print(
        "Join OperatorPool bytes_allocated: ",
        bodo.libs.streaming.join.get_op_pool_bytes_allocated(state_3),
    )
    __bodo_is_last_streaming_output_2 = False
    _iter_2 = 0
    _temp25 = 0.0
    _temp26 = time.time()
    state_4 = pd.read_sql(
        "LINEITEM",
        conn_str,  # Codegen change: use conn_str
        _bodo_is_table_input=True,
        _bodo_chunksize=4096,
        _bodo_read_as_table=True,
    )
    _temp28 = time.time()
    _temp29 = _temp28 - _temp26
    _temp25 = _temp25 + _temp29
    _temp33 = 0.0
    _temp34 = time.time()
    state_5 = bodo.libs.streaming.dict_encoding.init_dict_encoding_state()
    _temp36 = time.time()
    _temp37 = _temp36 - _temp34
    _temp33 = _temp33 + _temp37
    _temp41 = False
    _temp45 = 0.0
    _temp46 = time.time()
    state_6 = bodo.libs.streaming.dict_encoding.init_dict_encoding_state()
    _temp48 = time.time()
    _temp49 = _temp48 - _temp46
    _temp45 = _temp45 + _temp49
    # Codegen change: timer for probe step
    probe_time = 0.0
    __bodo_streaming_batches_list_1 = []
    input_request = True
    while not (_temp41):
        _temp27 = time.time()
        (
            T3,
            __bodo_is_last_streaming_output_2,
        ) = bodo.io.arrow_reader.read_arrow_next(state_4, input_request)
        _temp30 = time.time()
        _temp31 = _temp30 - _temp27
        _temp25 = _temp25 + _temp31
        _temp35 = time.time()
        T4 = bodo.hiframes.table.table_subset(T3, global_5, False)
        _temp38 = time.time()
        _temp39 = _temp38 - _temp35
        _temp33 = _temp33 + _temp39
        _temp3 = time.time()
        # Codegen change: track probe time
        t_probe = time.time()
        (T5, _temp41, input_request) = (
            bodo.libs.streaming.join.join_probe_consume_batch(
                state_3, T4, __bodo_is_last_streaming_output_2, True
            )
        )
        probe_time += time.time() - t_probe
        _temp42 = time.time()
        _temp43 = _temp42 - _temp3
        _temp1 = _temp1 + _temp43
        _temp47 = time.time()
        T6 = bodo.hiframes.table.table_subset(T5, global_6, False)
        _temp50 = time.time()
        _temp51 = _temp50 - _temp47
        _temp45 = _temp45 + _temp51
        __bodo_streaming_batches_list_1.append(T6)
        _iter_2 = _iter_2 + 1
    bodo.io.arrow_reader.arrow_reader_del(state_4)
    # Codegen change: print probe time
    bodo.user_logging.log_message("RELNODE_TIMING", f"Join probe took {probe_time}s")
    _temp32 = "LINEITEM"
    bodo.user_logging.log_message(
        "IO TIMING", f"""Execution time for reading table {_temp32}: {_temp25}"""
    )
    bodo.libs.streaming.dict_encoding.delete_dict_encoding_state(state_5)
    _temp40 = "PandasProject(L_ORDERKEY=[$0], L_EXTENDEDPRICE=[$5], L_SHIPINSTRUCT=[$13], L_SHIPMODE=[$14], L_COMMENT=[$15])"
    bodo.user_logging.log_message(
        "RELNODE_TIMING", f"""Execution time for RelNode {_temp40}: {_temp33}"""
    )
    print(
        "Join OperatorPool bytes_pinned: ",
        bodo.libs.streaming.join.get_op_pool_bytes_pinned(state_3),
    )
    print(
        "Join OperatorPool bytes_allocated: ",
        bodo.libs.streaming.join.get_op_pool_bytes_allocated(state_3),
    )
    bodo.libs.streaming.join.delete_join_state(state_3)
    _temp44 = "PandasJoin(condition=[=($4, $0)], joinType=[inner])"
    bodo.user_logging.log_message(
        "RELNODE_TIMING", f"""Execution time for RelNode {_temp44}: {_temp1}"""
    )
    bodo.libs.streaming.dict_encoding.delete_dict_encoding_state(state_6)
    _temp52 = "PandasProject(l_orderkey=[$4], l_extendedprice=[$5], l_shipinstruct=[$6], l_shipmode=[$7], l_comment=[$8], o_orderpriority=[$2], o_comment=[$3], o_custkey=[$1])"
    bodo.user_logging.log_message(
        "RELNODE_TIMING", f"""Execution time for RelNode {_temp52}: {_temp45}"""
    )
    # Codegen change: track concat time
    concat_time_start = time.time()
    T7 = bodo.utils.table_utils.concat_tables(__bodo_streaming_batches_list_1)
    index_1 = bodo.hiframes.pd_index_ext.init_range_index(0, len(T7), 1, None)
    df1 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T7,), index_1, global_7)
    bodo.user_logging.log_message(
        "RELNODE_TIMING", f"Concat Time: {time.time() - concat_time_start}"
    )
    # Codegen change: print overall execution time
    print(f"Finished executing the query. It took {time.time() - t0} seconds.")
    print("Output shape: ", df1.shape)
    return df1


# Codegen change: call the function
@pytest_mark_snowflake
@pytest.mark.parametrize(
    "build_outer,probe_outer",
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_simple_hash_join(build_outer, probe_outer):
    # Reset the global variables as per the parametrized inputs
    global global_build_outer, global_probe_outer
    global_build_outer = build_outer
    global_probe_outer = probe_outer
    # Get snowflake connection string from environment variables
    # (SF_USERNAME and SF_PASSWORD).
    # Use SF10 since that has sufficient data and compute,
    # but we can change the schema to TPCH_SF1, TPCH_SF100, etc.
    # to test different data sizes.
    conn_str = get_snowflake_connection_string("SNOWFLAKE_SAMPLE_DATA", "TPCH_SF10")
    # Force-recompile since the build-outer and probe-outer flags are globals.
    bodo.jit(impl)(conn_str)
