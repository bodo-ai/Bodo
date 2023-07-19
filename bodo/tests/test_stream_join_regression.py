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
AGENT_NAME (to anything), SF_USERNAME and SF_PASSWORD.
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
)
from bodo.utils.typing import ColNamesMetaType, MetaType

# Codegen change: turn verbose mode on
bodo.set_verbose_level(2)

global_2 = MetaType((0,))
global_3 = MetaType((0,))
global_1 = ColNamesMetaType(
    (
        "o_orderkey",
        "o_custkey",
        "o_orderstatus",
        "o_totalprice",
        "o_orderdate",
        "o_orderpriority",
        "o_clerk",
        "o_shippriority",
        "o_comment",
    )
)
global_11 = ColNamesMetaType(
    (
        "l_orderkey",
        "l_extendedprice",
        "l_shipinstruct",
        "l_shipmode",
        "l_comment",
        "o_orderpriority",
        "o_comment",
        "o_custkey",
    )
)
global_10 = MetaType((0, 1, 2, 3, 4, 5, 6, 7))
global_8 = MetaType((0, 1, 2, 3, 4))
global_9 = ColNamesMetaType(
    (
        "O_ORDERKEY",
        "O_CUSTKEY",
        "O_ORDERPRIORITY",
        "O_COMMENT",
        "L_ORDERKEY",
        "L_EXTENDEDPRICE",
        "L_SHIPINSTRUCT",
        "L_SHIPMODE",
        "L_COMMENT",
    )
)
global_6 = MetaType((0, 1, 2, 3))
global_7 = ColNamesMetaType(
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
global_4 = ColNamesMetaType(("O_ORDERKEY", "O_CUSTKEY", "O_ORDERPRIORITY", "O_COMMENT"))
global_5 = ColNamesMetaType(
    ("L_ORDERKEY", "L_EXTENDEDPRICE", "L_SHIPINSTRUCT", "L_SHIPMODE", "L_COMMENT")
)

# Codegen change: Global variables for setting build-outer and probe-outer flags.
global_build_outer = False
global_probe_outer = False


def impl(conn_str):  # Codegen change: add conn_str
    """
    Simple read from Snowflake followed by a hash join.
    The equivalent SQL query is:
        select
            l_orderkey,
            l_extendedprice,
            l_shipinstruct,
            l_shipmode,
            l_comment,
            o_orderpriority,
            o_comment,
            o_custkey
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
    print(f"Started executing query...")
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
    )
    # Codegen change: timer for build step
    build_time = 0.0
    _temp7 = time.time()
    _temp8 = _temp7 - _temp5
    _temp4 = _temp4 + _temp8
    _temp14 = 0.0
    _temp15 = time.time()
    state_2 = bodo.libs.stream_dict_encoding.init_dict_encoding_state()
    _temp17 = time.time()
    _temp18 = _temp17 - _temp15
    _temp14 = _temp14 + _temp18
    _temp1 = 0.0
    _temp2 = time.time()
    # Codegen change: Use global flags for build-outer and probe-outer
    state_3 = bodo.libs.stream_join.init_join_state(
        global_2, global_3, global_4, global_5, global_build_outer, global_probe_outer
    )
    _temp23 = time.time()
    _temp24 = _temp23 - _temp2
    _temp1 = _temp1 + _temp24
    while not (__bodo_is_last_streaming_output_1):
        _temp6 = time.time()
        (
            T9,
            __bodo_is_last_streaming_output_1,
        ) = bodo.io.arrow_reader.read_arrow_next(state_1)
        index_1 = bodo.hiframes.pd_index_ext.init_range_index(0, len(T9), 1, None)
        df10 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T9,), index_1, global_1)
        _temp11 = time.time()
        _temp12 = _temp11 - _temp6
        _temp4 = _temp4 + _temp12
        _temp16 = time.time()
        df19 = df10.loc[
            :, ["o_orderkey", "o_custkey", "o_orderpriority", "o_comment"]
        ].rename(
            columns={
                "o_orderkey": "O_ORDERKEY",
                "o_custkey": "O_CUSTKEY",
                "o_orderpriority": "O_ORDERPRIORITY",
                "o_comment": "O_COMMENT",
            },
            copy=False,
        )
        _temp20 = time.time()
        _temp21 = _temp20 - _temp16
        _temp14 = _temp14 + _temp21
        _temp3 = time.time()
        __bodo_is_last_streaming_output_1 = bodo.libs.distributed_api.sync_is_last(
            __bodo_is_last_streaming_output_1, _iter_1
        )
        T25 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df19), (), global_6, 4
        )
        # Codegen change: Track build time
        t_build = time.time()
        bodo.libs.stream_join.join_build_consume_batch(
            state_3, T25, __bodo_is_last_streaming_output_1
        )
        build_time += time.time() - t_build
        _temp26 = time.time()
        _temp27 = _temp26 - _temp3
        _temp1 = _temp1 + _temp27
        _iter_1 = _iter_1 + 1
    bodo.io.arrow_reader.arrow_reader_del(state_1)
    _temp13 = "ORDERS"
    bodo.user_logging.log_message(
        "IO TIMING", f"""Execution time for reading table {_temp13}: {_temp4}"""
    )
    bodo.libs.stream_dict_encoding.delete_dict_encoding_state(state_2)
    _temp22 = "PandasProject(O_ORDERKEY=[$0], O_CUSTKEY=[$1], O_ORDERPRIORITY=[$5], O_COMMENT=[$8])"
    bodo.user_logging.log_message(
        "RELNODE_TIMING", f"""Execution time for RelNode {_temp22}: {_temp14}"""
    )
    # Codegen change: print build timer
    bodo.user_logging.log_message("RELNODE_TIMING", f"Join build took {build_time}s")
    __bodo_is_last_streaming_output_2 = False
    _iter_2 = 0
    _temp28 = 0.0
    _temp29 = time.time()
    state_4 = pd.read_sql(
        "LINEITEM",
        conn_str,  # Codegen change: use conn_str
        _bodo_is_table_input=True,
        _bodo_chunksize=4096,
    )
    _temp31 = time.time()
    _temp32 = _temp31 - _temp29
    _temp28 = _temp28 + _temp32
    _temp38 = 0.0
    _temp39 = time.time()
    state_5 = bodo.libs.stream_dict_encoding.init_dict_encoding_state()
    _temp41 = time.time()
    _temp42 = _temp41 - _temp39
    _temp38 = _temp38 + _temp42
    _temp47 = False
    _temp54 = 0.0
    _temp55 = time.time()
    state_6 = bodo.libs.stream_dict_encoding.init_dict_encoding_state()
    _temp57 = time.time()
    _temp58 = _temp57 - _temp55
    _temp54 = _temp54 + _temp58
    # Codegen change: timer for probe step
    probe_time = 0.0
    __bodo_streaming_batches_list_1 = []
    while not (_temp47):
        _temp30 = time.time()
        (
            T33,
            __bodo_is_last_streaming_output_2,
        ) = bodo.io.arrow_reader.read_arrow_next(state_4)
        index_2 = bodo.hiframes.pd_index_ext.init_range_index(0, len(T33), 1, None)
        df34 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T33,), index_2, global_7)
        _temp35 = time.time()
        _temp36 = _temp35 - _temp30
        _temp28 = _temp28 + _temp36
        _temp40 = time.time()
        df43 = df34.loc[
            :,
            [
                "l_orderkey",
                "l_extendedprice",
                "l_shipinstruct",
                "l_shipmode",
                "l_comment",
            ],
        ].rename(
            columns={
                "l_orderkey": "L_ORDERKEY",
                "l_extendedprice": "L_EXTENDEDPRICE",
                "l_shipinstruct": "L_SHIPINSTRUCT",
                "l_shipmode": "L_SHIPMODE",
                "l_comment": "L_COMMENT",
            },
            copy=False,
        )
        _temp44 = time.time()
        _temp45 = _temp44 - _temp40
        _temp38 = _temp38 + _temp45
        _temp3 = time.time()
        __bodo_is_last_streaming_output_2 = bodo.libs.distributed_api.sync_is_last(
            __bodo_is_last_streaming_output_2, _iter_2
        )
        T48 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df43), (), global_8, 5
        )
        # Codegen change: track probe time
        t_probe = time.time()
        (
            T49,
            _temp47,
        ) = bodo.libs.stream_join.join_probe_consume_batch(
            state_3, T48, __bodo_is_last_streaming_output_2
        )
        probe_time += time.time() - t_probe
        index_3 = bodo.hiframes.pd_index_ext.init_range_index(0, len(T49), 1, None)
        df50 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T49,), index_3, global_9)
        _temp51 = time.time()
        _temp52 = _temp51 - _temp3
        _temp1 = _temp1 + _temp52
        _temp56 = time.time()
        df59 = df50.loc[
            :,
            [
                "L_ORDERKEY",
                "L_EXTENDEDPRICE",
                "L_SHIPINSTRUCT",
                "L_SHIPMODE",
                "L_COMMENT",
                "O_ORDERPRIORITY",
                "O_COMMENT",
                "O_CUSTKEY",
            ],
        ].rename(
            columns={
                "L_ORDERKEY": "l_orderkey",
                "L_EXTENDEDPRICE": "l_extendedprice",
                "L_SHIPINSTRUCT": "l_shipinstruct",
                "L_SHIPMODE": "l_shipmode",
                "L_COMMENT": "l_comment",
                "O_ORDERPRIORITY": "o_orderpriority",
                "O_COMMENT": "o_comment",
                "O_CUSTKEY": "o_custkey",
            },
            copy=False,
        )
        _temp60 = time.time()
        _temp61 = _temp60 - _temp56
        _temp54 = _temp54 + _temp61
        T63 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df59),
            (),
            global_10,
            8,
        )
        __bodo_streaming_batches_list_1.append(T63)
        _temp47 = bodo.libs.distributed_api.sync_is_last(_temp47, _iter_2)
        _iter_2 = _iter_2 + 1
    bodo.io.arrow_reader.arrow_reader_del(state_4)
    # Codegen change: print probe time
    bodo.user_logging.log_message("RELNODE_TIMING", f"Join probe took {probe_time}s")
    _temp37 = "LINEITEM"
    bodo.user_logging.log_message(
        "IO TIMING", f"""Execution time for reading table {_temp37}: {_temp28}"""
    )
    bodo.libs.stream_dict_encoding.delete_dict_encoding_state(state_5)
    _temp46 = "PandasProject(L_ORDERKEY=[$0], L_EXTENDEDPRICE=[$5], L_SHIPINSTRUCT=[$13], L_SHIPMODE=[$14], L_COMMENT=[$15])"
    bodo.user_logging.log_message(
        "RELNODE_TIMING", f"""Execution time for RelNode {_temp46}: {_temp38}"""
    )
    bodo.libs.stream_join.delete_join_state(state_3)
    _temp53 = "PandasJoin(condition=[=($4, $0)], joinType=[inner])"
    bodo.user_logging.log_message(
        "RELNODE_TIMING", f"""Execution time for RelNode {_temp53}: {_temp1}"""
    )
    bodo.libs.stream_dict_encoding.delete_dict_encoding_state(state_6)
    _temp62 = "PandasProject(l_orderkey=[$4], l_extendedprice=[$5], l_shipinstruct=[$6], l_shipmode=[$7], l_comment=[$8], o_orderpriority=[$2], o_comment=[$3], o_custkey=[$1])"
    bodo.user_logging.log_message(
        "RELNODE_TIMING", f"""Execution time for RelNode {_temp62}: {_temp54}"""
    )
    # Codegen change: track concat time
    concat_time_start = time.time()
    T64 = bodo.utils.table_utils.concat_tables(__bodo_streaming_batches_list_1)
    index_4 = bodo.hiframes.pd_index_ext.init_range_index(0, len(T64), 1, None)
    df65 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T64,), index_4, global_11)
    bodo.user_logging.log_message(
        "RELNODE_TIMING", f"Concat Time: {time.time() - concat_time_start}"
    )
    # Codegen change: print overall execution time
    print(f"Finished executing the query. It took {time.time() - t0} seconds.")
    print("Output shape: ", df65.shape)
    return df65


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
