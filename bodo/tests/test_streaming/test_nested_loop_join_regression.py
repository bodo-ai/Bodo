"""
Tests for checking regression in streaming nested loop join.
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
AGENT_NAME (to anything), SF_USERNAME and SF_PASSWORD, and
BODO_RUN_REGRESSION_TESTS (to anything).
After setting these, run using:
pytest -s -v test_stream_join_regression.py
We recommend running this on a single-node cluster on the
platform to ensure reproducible results. We only use the top
num_rows rows as the entire table is too big for a nested loop
join. For best IO performance, use an AWS workspace in the
us-east-1 region, but us-east-2 should be fine as well.

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
database and "TPCH_SF1" schema.

For spilling: please note that the following test cases don't
work for output spilling. That is because Numba tracks and
frees data buffers by reference counting. As we don't support
spilling for list and concat as of now, chunks of output data
will have a reference in our output list in Python, and thus
remain pinned in memory. Spilling for build and probe tables
do work though as we don't keep reference of them in our unit
tests.
If spilling is enabled, manually unpin the output chunks
simply by deleting them. For example, we can do
    while len(__bodo_streaming_batches_list_1) > 9:
        __bodo_streaming_batches_list_1.pop()
after appending the output table to the list. This ensures
that at most 9 chunks are pinned at the same time.
And another option is that we can replace
__bodo_streaming_batches_list_1.append(df56) with print(df56).
Printing df56 is to make sure critical codes not deleted due
to compiler optimization. The code is more intuitive while
printing may be more time-consuming.
"""

import time

import numpy as np
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

# Codegen change
# @param: number of rows
num_rows = 10000

# global for nested loop join
global_2 = MetaType(())
global_3 = MetaType(())
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
global_8 = MetaType((0, 1, 2, 3, 4, 5))
global_8_2 = MetaType((0, 1, 2, 3, 4))
global_9 = ColNamesMetaType(
    (
        "O_CUSTKEY",
        "O_ORDERDATE",
        "O_ORDERPRIORITY",
        "O_COMMENT",
        "L_ORDERKEY",
        "L_EXTENDEDPRICE",
        "L_SHIPDATE",
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
global_4 = ColNamesMetaType(
    ("O_CUSTKEY", "O_ORDERDATE", "O_ORDERPRIORITY", "O_COMMENT")
)
global_5 = ColNamesMetaType(
    (
        "L_ORDERKEY",
        "L_EXTENDEDPRICE",
        "L_SHIPDATE",
        "L_SHIPINSTRUCT",
        "L_SHIPMODE",
        "L_COMMENT",
    )
)

# addition global for pure cross join
global_8_2 = MetaType((0, 1, 2, 3, 4))
global_9_2 = ColNamesMetaType(
    (
        "o_custkey",
        "o_orderpriority",
        "o_comment",
        "l_orderkey",
        "l_extendedprice",
        "l_shipinstruct",
        "l_shipmode",
        "l_comment",
    )
)
global_6_2 = MetaType((0, 1, 2))
global_4_2 = ColNamesMetaType(("o_custkey", "o_orderpriority", "o_comment"))
global_5_2 = ColNamesMetaType(
    ("l_orderkey", "l_extendedprice", "l_shipinstruct", "l_shipmode", "l_comment")
)

# Codegen change: Global variables for setting build-outer and probe-outer flags.
global_build_outer = False
global_probe_outer = False
build_interval_cols = bodo.utils.typing.MetaType(())


def impl(conn_str):  # Codegen change: add conn_str
    """
    Simple read from Snowflake followed by a nested loop join.
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
            (select * from lineitem limit num_rows),
            (select * from orders limit num_rows)
        where
            l_shipdate < o_orderdate
    The query essentially reads two TPCH tables, get the
    top num_rows (10k by default) rows and joins them.
    We've chose columns that cover different data types
    such as integers, floats, strings and
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
    _temp4 = 0.0
    _temp5 = time.time()
    __bodo_streaming_reader_1 = pd.read_sql(
        f'SELECT * FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."ORDERS"\nFETCH NEXT {num_rows} ROWS ONLY',
        conn_str,  # Codegen change: use conn_str
        _bodo_is_table_input=False,
        _bodo_chunksize=4096,
    )
    # Codegen change: timer for build step
    build_time = 0.0
    _temp7 = time.time()
    _temp8 = _temp7 - _temp5
    _temp4 = _temp4 + _temp8
    _temp15 = 0.0
    _temp1 = 0.0
    _temp2 = time.time()
    # Codegen change: Use global flags for build-outer and probe-outer
    _temp22 = bodo.libs.streaming.join.init_join_state(
        global_2,
        global_3,
        global_4,
        global_5,
        global_build_outer,
        global_probe_outer,
        build_interval_cols,
        False,
        non_equi_condition="(left.`L_SHIPDATE` < right.`O_ORDERDATE`)",
    )
    _temp23 = time.time()
    _temp24 = _temp23 - _temp2
    _temp1 = _temp1 + _temp24
    while not (__bodo_is_last_streaming_output_1):
        _temp6 = time.time()
        (
            T9,
            __bodo_is_last_streaming_output_1,
        ) = bodo.io.arrow_reader.read_arrow_next(__bodo_streaming_reader_1, True)
        _temp10 = bodo.hiframes.pd_index_ext.init_range_index(0, len(T9), 1, None)
        df11 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T9,), _temp10, global_1)
        _temp12 = time.time()
        _temp13 = _temp12 - _temp6
        _temp4 = _temp4 + _temp13
        _temp17 = time.time()
        df18 = df11.loc[
            :, ["o_custkey", "o_orderdate", "o_orderpriority", "o_comment"]
        ].rename(
            columns={
                "o_custkey": "O_CUSTKEY",
                "o_orderdate": "O_ORDERDATE",
                "o_orderpriority": "O_ORDERPRIORITY",
                "o_comment": "O_COMMENT",
            },
            copy=False,
        )
        _temp19 = time.time()
        _temp20 = _temp19 - _temp17
        _temp15 = _temp15 + _temp20
        _temp3 = time.time()
        __bodo_is_last_streaming_output_1 = bodo.libs.distributed_api.dist_reduce(
            __bodo_is_last_streaming_output_1,
            np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
        )
        T25 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df18), (), global_6, 4
        )
        # Codegen change: Track build time
        t_build = time.time()
        bodo.libs.streaming.join.join_build_consume_batch(
            _temp22, T25, __bodo_is_last_streaming_output_1
        )
        build_time += time.time() - t_build
        _temp26 = time.time()
        _temp27 = _temp26 - _temp3
        _temp1 = _temp1 + _temp27
    _temp14 = f'SELECT * FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."ORDERS"\nFETCH NEXT {num_rows} ROWS ONLY'
    bodo.user_logging.log_message(
        "IO TIMING", f"""Execution time for reading table {_temp14}: {_temp4}"""
    )
    bodo.io.arrow_reader.arrow_reader_del(__bodo_streaming_reader_1)
    _temp21 = "PandasProject(O_CUSTKEY=[$1], O_ORDERDATE=[$4], O_ORDERPRIORITY=[$5], O_COMMENT=[$8])"
    bodo.user_logging.log_message(
        "RELNODE_TIMING", f"""Execution time for RelNode {_temp21}: {_temp15}"""
    )
    # Codegen change: print build timer
    bodo.user_logging.log_message("RELNODE_TIMING", f"Join build took {build_time}s")
    __bodo_is_last_streaming_output_2 = False
    _temp28 = 0.0
    _temp29 = time.time()
    __bodo_streaming_reader_2 = pd.read_sql(
        f'SELECT * FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."LINEITEM"\nFETCH NEXT {num_rows} ROWS ONLY',
        conn_str,  # Codegen change: use conn_str
        _bodo_is_table_input=False,
        _bodo_chunksize=4096,
    )
    _temp31 = time.time()
    _temp32 = _temp31 - _temp29
    _temp28 = _temp28 + _temp32
    _temp39 = 0.0
    _temp46 = False
    _temp53 = 0.0
    # Codegen change: timer for probe step
    probe_time = 0.0
    __bodo_streaming_batches_list_1 = []
    input_request1 = True
    while not (_temp46):
        _temp30 = time.time()
        (
            T33,
            __bodo_is_last_streaming_output_2,
        ) = bodo.io.arrow_reader.read_arrow_next(
            __bodo_streaming_reader_2, input_request1
        )
        _temp34 = bodo.hiframes.pd_index_ext.init_range_index(0, len(T33), 1, None)
        df35 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T33,), _temp34, global_7)
        _temp36 = time.time()
        _temp37 = _temp36 - _temp30
        _temp28 = _temp28 + _temp37
        _temp41 = time.time()
        df42 = df35.loc[
            :,
            [
                "l_orderkey",
                "l_extendedprice",
                "l_shipdate",
                "l_shipinstruct",
                "l_shipmode",
                "l_comment",
            ],
        ].rename(
            columns={
                "l_orderkey": "L_ORDERKEY",
                "l_extendedprice": "L_EXTENDEDPRICE",
                "l_shipdate": "L_SHIPDATE",
                "l_shipinstruct": "L_SHIPINSTRUCT",
                "l_shipmode": "L_SHIPMODE",
                "l_comment": "L_COMMENT",
            },
            copy=False,
        )
        _temp43 = time.time()
        _temp44 = _temp43 - _temp41
        _temp39 = _temp39 + _temp44
        _temp3 = time.time()
        __bodo_is_last_streaming_output_2 = bodo.libs.distributed_api.dist_reduce(
            __bodo_is_last_streaming_output_2,
            np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
        )
        T47 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df42), (), global_8, 6
        )
        # Codegen change: track probe time
        t_probe = time.time()
        (T48, _temp46, input_request1) = (
            bodo.libs.streaming.join.join_probe_consume_batch(
                _temp22, T47, __bodo_is_last_streaming_output_2, True
            )
        )
        probe_time += time.time() - t_probe
        index_1 = bodo.hiframes.pd_index_ext.init_range_index(0, len(T48), 1, None)
        df49 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T48,), index_1, global_9)
        _temp50 = time.time()
        _temp51 = _temp50 - _temp3
        _temp1 = _temp1 + _temp51
        _temp55 = time.time()
        df56 = df49.loc[
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
        _temp57 = time.time()
        _temp58 = _temp57 - _temp55
        _temp53 = _temp53 + _temp58
        __bodo_streaming_batches_list_1.append(df56)
        _temp46 = bodo.libs.distributed_api.dist_reduce(
            _temp46, np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value)
        )
    # Codegen change: print probe time
    bodo.user_logging.log_message("RELNODE_TIMING", f"Join probe took {probe_time}s")
    _temp38 = f'SELECT * FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."LINEITEM"\nFETCH NEXT {num_rows} ROWS ONLY'
    bodo.user_logging.log_message(
        "IO TIMING", f"""Execution time for reading table {_temp38}: {_temp28}"""
    )
    bodo.io.arrow_reader.arrow_reader_del(__bodo_streaming_reader_2)
    _temp45 = "PandasProject(L_ORDERKEY=[$0], L_EXTENDEDPRICE=[$5], L_SHIPDATE=[$10], L_SHIPINSTRUCT=[$13], L_SHIPMODE=[$14], L_COMMENT=[$15])"
    bodo.user_logging.log_message(
        "RELNODE_TIMING", f"""Execution time for RelNode {_temp45}: {_temp39}"""
    )
    bodo.libs.streaming.join.delete_join_state(_temp22)
    _temp52 = "PandasJoin(condition=[<($6, $1)], joinType=[inner])"
    bodo.user_logging.log_message(
        "RELNODE_TIMING", f"""Execution time for RelNode {_temp52}: {_temp1}"""
    )
    _temp59 = "PandasProject(l_orderkey=[$4], l_extendedprice=[$5], l_shipinstruct=[$7], l_shipmode=[$8], l_comment=[$9], o_orderpriority=[$2], o_comment=[$3], o_custkey=[$0])"
    bodo.user_logging.log_message(
        "RELNODE_TIMING", f"""Execution time for RelNode {_temp59}: {_temp53}"""
    )
    # Codegen change: track concat time
    concat_time_start = time.time()
    df60 = pd.concat(__bodo_streaming_batches_list_1, ignore_index=True)
    bodo.user_logging.log_message(
        "RELNODE_TIMING", f"Concat Time: {time.time() - concat_time_start}"
    )
    # Codegen change: print overall execution time
    print(f"Finished executing the query. It took {time.time() - t0} seconds.")
    print("Output shape: ", df60.shape)
    return df60


def impl_wo_condition(conn_str):  # Codegen change: add conn_str
    """
    SQL query similar to the above, but without a condition (i.e. pure cross join):
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
            (select * from lineitem limit num_rows),
            (select * from orders limit num_rows)
    """
    # Codegen change: Add print and overall timer
    print("Started executing query...")
    t0 = time.time()
    __bodo_is_last_streaming_output_1 = False
    _temp4 = 0.0
    _temp5 = time.time()
    __bodo_streaming_reader_1 = pd.read_sql(
        f'SELECT * FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."ORDERS"\nFETCH NEXT {num_rows} ROWS ONLY',
        conn_str,  # Codegen change: use conn_str
        _bodo_is_table_input=False,
        _bodo_chunksize=4096,
    )
    # Codegen change: timer for build step
    build_time = 0.0
    _temp7 = time.time()
    _temp8 = _temp7 - _temp5
    _temp4 = _temp4 + _temp8
    _temp15 = 0.0
    _temp1 = 0.0
    _temp2 = time.time()
    # Codegen change: Use global flags for build-outer and probe-outer
    _temp22 = bodo.libs.streaming.join.init_join_state(
        global_2,
        global_3,
        global_4_2,
        global_5_2,
        global_build_outer,
        global_probe_outer,
        build_interval_cols,
        False,
    )
    _temp23 = time.time()
    _temp24 = _temp23 - _temp2
    _temp1 = _temp1 + _temp24
    while not (__bodo_is_last_streaming_output_1):
        _temp6 = time.time()
        (
            T9,
            __bodo_is_last_streaming_output_1,
        ) = bodo.io.arrow_reader.read_arrow_next(__bodo_streaming_reader_1, True)
        _temp10 = bodo.hiframes.pd_index_ext.init_range_index(0, len(T9), 1, None)
        df11 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T9,), _temp10, global_1)
        _temp12 = time.time()
        _temp13 = _temp12 - _temp6
        _temp4 = _temp4 + _temp13
        _temp17 = time.time()
        df18 = df11.loc[:, ["o_custkey", "o_orderpriority", "o_comment"]]
        _temp19 = time.time()
        _temp20 = _temp19 - _temp17
        _temp15 = _temp15 + _temp20
        _temp3 = time.time()
        __bodo_is_last_streaming_output_1 = bodo.libs.distributed_api.dist_reduce(
            __bodo_is_last_streaming_output_1,
            np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
        )
        T25 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df18),
            (),
            global_6_2,
            3,
        )
        # Codegen change: Track build time
        t_build = time.time()
        bodo.libs.streaming.join.join_build_consume_batch(
            _temp22, T25, __bodo_is_last_streaming_output_1
        )
        build_time += time.time() - t_build
        _temp26 = time.time()
        _temp27 = _temp26 - _temp3
        _temp1 = _temp1 + _temp27
    _temp14 = f'SELECT * FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."ORDERS"\nFETCH NEXT {num_rows} ROWS ONLY'
    bodo.user_logging.log_message(
        "IO TIMING", f"""Execution time for reading table {_temp14}: {_temp4}"""
    )
    bodo.io.arrow_reader.arrow_reader_del(__bodo_streaming_reader_1)
    _temp21 = "PandasProject(o_custkey=[$1], o_orderpriority=[$5], o_comment=[$8])"
    bodo.user_logging.log_message(
        "RELNODE_TIMING", f"""Execution time for RelNode {_temp21}: {_temp15}"""
    )
    # Codegen change: print build timer
    bodo.user_logging.log_message("RELNODE_TIMING", f"Join build took {build_time}s")
    __bodo_is_last_streaming_output_2 = False
    _temp28 = 0.0
    _temp29 = time.time()
    __bodo_streaming_reader_2 = pd.read_sql(
        f'SELECT * FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."LINEITEM"\nFETCH NEXT {num_rows} ROWS ONLY',
        conn_str,  # Codegen change: use conn_str
        _bodo_is_table_input=False,
        _bodo_chunksize=4096,
    )
    _temp31 = time.time()
    _temp32 = _temp31 - _temp29
    _temp28 = _temp28 + _temp32
    _temp39 = 0.0
    _temp46 = False
    _temp53 = 0.0
    # Codegen change: timer for probe step
    probe_time = 0.0
    __bodo_streaming_batches_list_1 = []
    input_request2 = True
    while not (_temp46):
        _temp30 = time.time()
        (
            T33,
            __bodo_is_last_streaming_output_2,
        ) = bodo.io.arrow_reader.read_arrow_next(
            __bodo_streaming_reader_2, input_request2
        )
        _temp34 = bodo.hiframes.pd_index_ext.init_range_index(0, len(T33), 1, None)
        df35 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T33,), _temp34, global_7)
        _temp36 = time.time()
        _temp37 = _temp36 - _temp30
        _temp28 = _temp28 + _temp37
        _temp41 = time.time()
        df42 = df35.loc[
            :,
            [
                "l_orderkey",
                "l_extendedprice",
                "l_shipinstruct",
                "l_shipmode",
                "l_comment",
            ],
        ]
        _temp43 = time.time()
        _temp44 = _temp43 - _temp41
        _temp39 = _temp39 + _temp44
        _temp3 = time.time()
        __bodo_is_last_streaming_output_2 = bodo.libs.distributed_api.dist_reduce(
            __bodo_is_last_streaming_output_2,
            np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
        )
        # Codegen change: track probe time
        t_probe = time.time()
        T47 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df42),
            (),
            global_8_2,
            5,
        )
        (T48, _temp46, input_request2) = (
            bodo.libs.streaming.join.join_probe_consume_batch(
                _temp22, T47, __bodo_is_last_streaming_output_2, True
            )
        )
        probe_time += time.time() - t_probe
        index_1 = bodo.hiframes.pd_index_ext.init_range_index(0, len(T48), 1, None)
        df49 = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (T48,), index_1, global_9_2
        )
        _temp50 = time.time()
        _temp51 = _temp50 - _temp3
        _temp1 = _temp1 + _temp51
        _temp55 = time.time()
        df56 = df49.loc[
            :,
            [
                "o_orderpriority",
                "o_comment",
                "o_custkey",
                "l_orderkey",
                "l_extendedprice",
                "l_shipinstruct",
                "l_shipmode",
                "l_comment",
            ],
        ]
        _temp57 = time.time()
        _temp58 = _temp57 - _temp55
        _temp53 = _temp53 + _temp58
        __bodo_streaming_batches_list_1.append(df56)
        _temp46 = bodo.libs.distributed_api.dist_reduce(
            _temp46, np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value)
        )
    # Codegen change: print probe time
    bodo.user_logging.log_message("RELNODE_TIMING", f"Join probe took {probe_time}s")
    _temp38 = f'SELECT * FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."LINEITEM"\nFETCH NEXT {num_rows} ROWS ONLY'
    bodo.user_logging.log_message(
        "IO TIMING", f"""Execution time for reading table {_temp38}: {_temp28}"""
    )
    bodo.io.arrow_reader.arrow_reader_del(__bodo_streaming_reader_2)
    _temp45 = "PandasProject(l_orderkey=[$0], l_extendedprice=[$5], l_shipinstruct=[$13], l_shipmode=[$14], l_comment=[$15])"
    bodo.user_logging.log_message(
        "RELNODE_TIMING", f"""Execution time for RelNode {_temp45}: {_temp39}"""
    )
    bodo.libs.streaming.join.delete_join_state(_temp22)
    _temp52 = "PandasJoin(condition=[true], joinType=[inner])"
    bodo.user_logging.log_message(
        "RELNODE_TIMING", f"""Execution time for RelNode {_temp52}: {_temp1}"""
    )
    _temp59 = "PandasProject(l_orderkey=[$3], l_extendedprice=[$4], l_shipinstruct=[$5], l_shipmode=[$6], l_comment=[$7], o_orderpriority=[$1], o_comment=[$2], o_custkey=[$0])"
    bodo.user_logging.log_message(
        "RELNODE_TIMING", f"""Execution time for RelNode {_temp59}: {_temp53}"""
    )
    # Codegen change: track concat time
    concat_time_start = time.time()
    df60 = pd.concat(__bodo_streaming_batches_list_1, ignore_index=True)
    bodo.user_logging.log_message(
        "RELNODE_TIMING", f"Concat Time: {time.time() - concat_time_start}"
    )
    # Codegen change: print overall execution time
    print(f"Finished executing the query. It took {time.time() - t0} seconds.")
    print("Output shape: ", df60.shape)
    return df60


def impl_unbalanced(conn_str):  # Codegen change: add conn_str
    """
    SQL query with huge build table:
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
            (select * from orders limit 100000000),
            (select * from lineitem limit 5)
        where
            l_shipdate < o_orderdate
    Note that this query can't be generated directly as
    the codegen is smart enough to select the smaller table
    for the build step. What we do is
    1. run codegen for this query:
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
            (select * from orders limit 5),
            (select * from lineitem limit 100000000)
        where
            l_shipdate < o_orderdate
    and 2. manually swap the limit for the two tables
    """
    # Codegen change: Add print and overall timer
    print("Started executing query...")
    t0 = time.time()
    __bodo_is_last_streaming_output_1 = False
    _temp4 = 0.0
    _temp5 = time.time()
    # Codegen change: Change number of rows for ORDERS
    __bodo_streaming_reader_1 = pd.read_sql(
        'SELECT * FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF100"."ORDERS"\nFETCH NEXT 100000000 ROWS ONLY',
        conn_str,  # Codegen change: use conn_str
        _bodo_is_table_input=False,
        _bodo_chunksize=4096,
    )
    # Codegen change: timer for build step
    build_time = 0.0
    _temp7 = time.time()
    _temp8 = _temp7 - _temp5
    _temp4 = _temp4 + _temp8
    _temp15 = 0.0
    _temp1 = 0.0
    _temp2 = time.time()
    # Codegen change: Use global flags for build-outer and probe-outer
    _temp22 = bodo.libs.streaming.join.init_join_state(
        global_2,
        global_3,
        global_4,
        global_5,
        global_build_outer,
        global_probe_outer,
        build_interval_cols,
        False,
        non_equi_condition="(left.`L_SHIPDATE` < right.`O_ORDERDATE`)",
    )
    _temp23 = time.time()
    _temp24 = _temp23 - _temp2
    _temp1 = _temp1 + _temp24
    while not (__bodo_is_last_streaming_output_1):
        _temp6 = time.time()
        (
            T9,
            __bodo_is_last_streaming_output_1,
        ) = bodo.io.arrow_reader.read_arrow_next(__bodo_streaming_reader_1, True)
        _temp10 = bodo.hiframes.pd_index_ext.init_range_index(0, len(T9), 1, None)
        df11 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T9,), _temp10, global_1)
        _temp12 = time.time()
        _temp13 = _temp12 - _temp6
        _temp4 = _temp4 + _temp13
        _temp17 = time.time()
        df18 = df11.loc[
            :, ["o_custkey", "o_orderdate", "o_orderpriority", "o_comment"]
        ].rename(
            columns={
                "o_custkey": "O_CUSTKEY",
                "o_orderdate": "O_ORDERDATE",
                "o_orderpriority": "O_ORDERPRIORITY",
                "o_comment": "O_COMMENT",
            },
            copy=False,
        )
        _temp19 = time.time()
        _temp20 = _temp19 - _temp17
        _temp15 = _temp15 + _temp20
        _temp3 = time.time()
        __bodo_is_last_streaming_output_1 = bodo.libs.distributed_api.dist_reduce(
            __bodo_is_last_streaming_output_1,
            np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
        )
        T25 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df18), (), global_6, 4
        )
        # Codegen change: Track build time
        t_build = time.time()
        bodo.libs.streaming.join.join_build_consume_batch(
            _temp22, T25, __bodo_is_last_streaming_output_1
        )
        build_time += time.time() - t_build
        _temp26 = time.time()
        _temp27 = _temp26 - _temp3
        _temp1 = _temp1 + _temp27
    # Codegen change: Change number of rows for ORDERS
    _temp14 = 'SELECT * FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF100"."ORDERS"\nFETCH NEXT 100000000 ROWS ONLY'
    bodo.user_logging.log_message(
        "IO TIMING", f"""Execution time for reading table {_temp14}: {_temp4}"""
    )
    bodo.io.arrow_reader.arrow_reader_del(__bodo_streaming_reader_1)
    _temp21 = "PandasProject(O_CUSTKEY=[$1], O_ORDERDATE=[$4], O_ORDERPRIORITY=[$5], O_COMMENT=[$8])"
    bodo.user_logging.log_message(
        "RELNODE_TIMING", f"""Execution time for RelNode {_temp21}: {_temp15}"""
    )
    # Codegen change: print build timer
    bodo.user_logging.log_message("RELNODE_TIMING", f"Join build took {build_time}s")
    __bodo_is_last_streaming_output_2 = False
    _temp28 = 0.0
    _temp29 = time.time()
    # Codegen change: Change number of rows for LINEITEM
    __bodo_streaming_reader_2 = pd.read_sql(
        'SELECT * FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF100"."LINEITEM"\nFETCH NEXT 5 ROWS ONLY',
        conn_str,  # Codegen change: use conn_str
        _bodo_is_table_input=False,
        _bodo_chunksize=4096,
    )
    _temp31 = time.time()
    _temp32 = _temp31 - _temp29
    _temp28 = _temp28 + _temp32
    _temp39 = 0.0
    _temp46 = False
    _temp53 = 0.0
    # Codegen change: timer for probe step
    probe_time = 0.0
    __bodo_streaming_batches_list_1 = []
    input_request3 = True
    while not (_temp46):
        _temp30 = time.time()
        (
            T33,
            __bodo_is_last_streaming_output_2,
        ) = bodo.io.arrow_reader.read_arrow_next(
            __bodo_streaming_reader_2, input_request3
        )
        _temp34 = bodo.hiframes.pd_index_ext.init_range_index(0, len(T33), 1, None)
        df35 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T33,), _temp34, global_7)
        _temp36 = time.time()
        _temp37 = _temp36 - _temp30
        _temp28 = _temp28 + _temp37
        _temp41 = time.time()
        df42 = df35.loc[
            :,
            [
                "l_orderkey",
                "l_extendedprice",
                "l_shipdate",
                "l_shipinstruct",
                "l_shipmode",
                "l_comment",
            ],
        ].rename(
            columns={
                "l_orderkey": "L_ORDERKEY",
                "l_extendedprice": "L_EXTENDEDPRICE",
                "l_shipdate": "L_SHIPDATE",
                "l_shipinstruct": "L_SHIPINSTRUCT",
                "l_shipmode": "L_SHIPMODE",
                "l_comment": "L_COMMENT",
            },
            copy=False,
        )
        _temp43 = time.time()
        _temp44 = _temp43 - _temp41
        _temp39 = _temp39 + _temp44
        _temp3 = time.time()
        __bodo_is_last_streaming_output_2 = bodo.libs.distributed_api.dist_reduce(
            __bodo_is_last_streaming_output_2,
            np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
        )
        # Codegen change: track probe time
        t_probe = time.time()
        T47 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df42), (), global_8, 6
        )
        (
            T48,
            _temp46,
            input_request3,
        ) = bodo.libs.streaming.join.join_probe_consume_batch(
            _temp22, T47, __bodo_is_last_streaming_output_2, True
        )
        probe_time += time.time() - t_probe
        index_1 = bodo.hiframes.pd_index_ext.init_range_index(0, len(T48), 1, None)
        df49 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T48,), index_1, global_9)
        _temp50 = time.time()
        _temp51 = _temp50 - _temp3
        _temp1 = _temp1 + _temp51
        _temp55 = time.time()
        df56 = df49.loc[
            :,
            [
                "O_ORDERPRIORITY",
                "O_COMMENT",
                "O_CUSTKEY",
                "L_ORDERKEY",
                "L_EXTENDEDPRICE",
                "L_SHIPINSTRUCT",
                "L_SHIPMODE",
                "L_COMMENT",
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
        _temp57 = time.time()
        _temp58 = _temp57 - _temp55
        _temp53 = _temp53 + _temp58
        __bodo_streaming_batches_list_1.append(df56)
        _temp46 = bodo.libs.distributed_api.dist_reduce(
            _temp46, np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value)
        )
    # Codegen change: print probe time
    bodo.user_logging.log_message("RELNODE_TIMING", f"Join probe took {probe_time}s")
    # Codegen change: Change number of rows for LINEITEM
    _temp38 = 'SELECT * FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF100"."LINEITEM"\nFETCH NEXT 5 ROWS ONLY'
    bodo.user_logging.log_message(
        "IO TIMING", f"""Execution time for reading table {_temp38}: {_temp28}"""
    )
    bodo.io.arrow_reader.arrow_reader_del(__bodo_streaming_reader_2)
    _temp45 = "PandasProject(L_ORDERKEY=[$0], L_EXTENDEDPRICE=[$5], L_SHIPDATE=[$10], L_SHIPINSTRUCT=[$13], L_SHIPMODE=[$14], L_COMMENT=[$15])"
    bodo.user_logging.log_message(
        "RELNODE_TIMING", f"""Execution time for RelNode {_temp45}: {_temp39}"""
    )
    bodo.libs.streaming.join.delete_join_state(_temp22)
    _temp52 = "PandasJoin(condition=[<($6, $1)], joinType=[inner])"
    bodo.user_logging.log_message(
        "RELNODE_TIMING", f"""Execution time for RelNode {_temp52}: {_temp1}"""
    )
    _temp59 = "PandasProject(l_orderkey=[$4], l_extendedprice=[$5], l_shipinstruct=[$7], l_shipmode=[$8], l_comment=[$9], o_orderpriority=[$2], o_comment=[$3], o_custkey=[$0])"
    bodo.user_logging.log_message(
        "RELNODE_TIMING", f"""Execution time for RelNode {_temp59}: {_temp53}"""
    )
    # Codegen change: track concat time
    concat_time_start = time.time()
    df60 = pd.concat(__bodo_streaming_batches_list_1, ignore_index=True)
    bodo.user_logging.log_message(
        "RELNODE_TIMING", f"Concat Time: {time.time() - concat_time_start}"
    )
    # Codegen change: print overall execution time
    print(f"Finished executing the query. It took {time.time() - t0} seconds.")
    print("Output shape: ", df60.shape)
    return df60


# Codegen change: call the function
@pytest_mark_snowflake
@pytest.mark.parametrize(
    "build_outer,probe_outer",
    [(False, False), (False, True), (True, False), (True, True)],
)
# nested loop join with condition
def test_nested_loop_join(build_outer, probe_outer):
    # Reset the global variables as per the parametrized inputs
    global global_build_outer, global_probe_outer
    global_build_outer = build_outer
    global_probe_outer = probe_outer
    # Get snowflake connection string from environment variables
    # (SF_USERNAME and SF_PASSWORD).
    # Change the number of rows per table to test different data sizes.
    conn_str = get_snowflake_connection_string("SNOWFLAKE_SAMPLE_DATA", "TPCH_SF1")
    # Force-recompile since the build-outer and probe-outer flags are globals.
    bodo.jit(impl)(conn_str)


# Codegen change: call the function
@pytest_mark_snowflake
@pytest.mark.parametrize(
    "build_outer,probe_outer",
    [(False, False), (False, True), (True, False), (True, True)],
)
# pure cross join without condition
def test_pure_nested_loop_join(build_outer, probe_outer):
    # Reset the global variables as per the parametrized inputs
    global global_build_outer, global_probe_outer
    global_build_outer = build_outer
    global_probe_outer = probe_outer
    # Get snowflake connection string from environment variables
    # (SF_USERNAME and SF_PASSWORD).
    # Change the number of rows per table to test different data sizes.
    conn_str = get_snowflake_connection_string("SNOWFLAKE_SAMPLE_DATA", "TPCH_SF1")
    # Force-recompile since the build-outer and probe-outer flags are globals.
    bodo.jit(impl_wo_condition)(conn_str)


# Codegen change: call the function
@pytest_mark_snowflake
@pytest.mark.parametrize(
    "build_outer,probe_outer",
    [(False, False), (False, True), (True, False), (True, True)],
)
# pure cross join without condition
def test_nested_loop_join_unbalanced(build_outer, probe_outer, verbose_mode_on):
    # Reset the global variables as per the parametrized inputs
    global global_build_outer, global_probe_outer
    global_build_outer = build_outer
    global_probe_outer = probe_outer
    # Get snowflake connection string from environment variables
    # (SF_USERNAME and SF_PASSWORD).
    # Change the number of rows per table to test different data sizes.
    conn_str = get_snowflake_connection_string("SNOWFLAKE_SAMPLE_DATA", "TPCH_SF100")
    # Force-recompile since the build-outer and probe-outer flags are globals.
    bodo.jit(impl_unbalanced)(conn_str)
