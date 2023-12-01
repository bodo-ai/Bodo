import math
import random
import string

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
import bodo.io.snowflake
import bodo.tests.utils
from bodo.io.arrow_reader import arrow_reader_del, read_arrow_next
from bodo.libs.memory import BufferPool
from bodo.libs.stream_join import (
    delete_join_state,
    get_partition_state,
    init_join_state,
    join_build_consume_batch,
    join_probe_consume_batch,
)
from bodo.tests.utils import (
    _get_dist_arg,
    _test_equal,
    check_func,
    pytest_mark_snowflake,
    set_broadcast_join,
    temp_env_override,
)


@pytest.fixture(params=[True, False])
def broadcast(request):
    return request.param


# Listing them separately since otherwise the traceback during errors is huge.
hash_join_basic_test_params = [
    # Equivalent query:
    # select  L_PARTKEY, L_COMMENT, L_ORDERKEY, P_PARTKEY, P_COMMENT, P_NAME, P_SIZE
    # from lineitem inner join part on P_PARTKEY = L_PARTKEY and P_COMMENT = L_COMMENT
    # where l_orderkey > {l_orderkey_start} and l_orderkey < {l_orderkey_end}
    # and p_size > {p_size_limit}
    (
        False,
        False,
        pd.DataFrame(
            {
                "l_partkey": [183728],
                "l_comment": ["bold deposi"],
                "l_orderkey": [685476],
                "p_partkey": [183728],
                "p_comment": ["bold deposi"],
                "p_name": ["yellow turquoise cornflower coral saddle"],
                "p_size": [37],
            }
        ),
    ),
    # Equivalent query:
    # select L_PARTKEY, L_COMMENT, L_ORDERKEY, P_PARTKEY, P_COMMENT, P_NAME, P_SIZE
    # from (
    #     select  L_PARTKEY, L_COMMENT, L_ORDERKEY from lineitem where l_orderkey > {l_orderkey_start} and l_orderkey < {l_orderkey_end}
    # ) lineitem_filtered
    # right outer join (
    #     select P_PARTKEY, P_COMMENT, P_NAME, P_SIZE from part where p_size = {p_size_limit} and p_partkey > {p_orderkey_start} and p_partkey < {p_orderkey_end}
    # ) part_filtered
    # on P_PARTKEY = L_PARTKEY and P_COMMENT = L_COMMENT
    (
        True,
        False,
        pd.DataFrame(
            {
                "l_partkey": [183728, pd.NA, pd.NA],
                "l_comment": [
                    "bold deposi",
                    pd.NA,
                    pd.NA,
                ],
                "l_orderkey": [685476, pd.NA, pd.NA],
                "p_partkey": [183728, 183820, 183846],
                "p_comment": [
                    "bold deposi",
                    "s. quickly unusua",
                    " foxes are",
                ],
                "p_name": [
                    "yellow turquoise cornflower coral saddle",
                    "tomato goldenrod black turquoise maroon",
                    "cream dim blush moccasin drab",
                ],
                "p_size": [37] * 3,
            }
        ),
    ),
    # Equivalent query:
    # select L_PARTKEY, L_COMMENT, L_ORDERKEY, P_PARTKEY, P_COMMENT, P_NAME, P_SIZE
    # from lineitem left outer join (
    #     select P_PARTKEY, P_COMMENT, P_NAME, P_SIZE from part where p_size > {p_size_limit}
    # ) on P_PARTKEY = L_PARTKEY and P_COMMENT = L_COMMENT
    # where l_orderkey > {l_orderkey_start} and l_orderkey < {l_orderkey_end}
    (
        False,
        True,
        pd.DataFrame(
            {
                "l_partkey": [45677, 85880, 174117, 183728, 106836, 191705, 171506],
                "l_comment": [
                    "requests wake permanently among the e",
                    "te above the silent platelets. furiously",
                    "lyly express accounts are blithely f",
                    "bold deposi",
                    "t, regular requests cajole ",
                    "ve the blithely even requests haggle",
                    "ding packages; ironic accounts ",
                ],
                "l_orderkey": [
                    685476,
                    685476,
                    685476,
                    685476,
                    685476,
                    685476,
                    685476,
                ],
                "p_partkey": [pd.NA, pd.NA, pd.NA, 183728, pd.NA, pd.NA, pd.NA],
                "p_comment": [
                    pd.NA,
                    pd.NA,
                    pd.NA,
                    "bold deposi",
                    pd.NA,
                    pd.NA,
                    pd.NA,
                ],
                "p_name": [
                    pd.NA,
                    pd.NA,
                    pd.NA,
                    "yellow turquoise cornflower coral saddle",
                    pd.NA,
                    pd.NA,
                    pd.NA,
                ],
                "p_size": [pd.NA, pd.NA, pd.NA, 37, pd.NA, pd.NA, pd.NA],
            }
        ),
    ),
    # Equivalent query:
    # select L_PARTKEY, L_COMMENT, L_ORDERKEY, P_PARTKEY, P_COMMENT, P_NAME, P_SIZE
    # from (
    #     select  L_PARTKEY, L_COMMENT, L_ORDERKEY from lineitem where l_orderkey > {l_orderkey_start} and l_orderkey < {l_orderkey_end}
    # ) lineitem_filtered
    # full outer join (
    #     select P_PARTKEY, P_COMMENT, P_NAME, P_SIZE from part where p_size = {p_size_limit} and p_partkey > {p_orderkey_start} and p_partkey < {p_orderkey_end}
    # ) part_filtered
    # on P_PARTKEY = L_PARTKEY and P_COMMENT = L_COMMENT
    (
        True,
        True,
        pd.DataFrame(
            {
                "l_partkey": [
                    45677,
                    85880,
                    174117,
                    183728,
                    106836,
                    191705,
                    171506,
                    pd.NA,
                    pd.NA,
                ],
                "l_comment": [
                    "requests wake permanently among the e",
                    "te above the silent platelets. furiously",
                    "lyly express accounts are blithely f",
                    "bold deposi",
                    "t, regular requests cajole ",
                    "ve the blithely even requests haggle",
                    "ding packages; ironic accounts ",
                    pd.NA,
                    pd.NA,
                ],
                "l_orderkey": [
                    685476,
                    685476,
                    685476,
                    685476,
                    685476,
                    685476,
                    685476,
                    pd.NA,
                    pd.NA,
                ],
                "p_partkey": [
                    pd.NA,
                    pd.NA,
                    pd.NA,
                    183728,
                    pd.NA,
                    pd.NA,
                    pd.NA,
                    183820,
                    183846,
                ],
                "p_comment": [
                    pd.NA,
                    pd.NA,
                    pd.NA,
                    "bold deposi",
                    pd.NA,
                    pd.NA,
                    pd.NA,
                    "s. quickly unusua",
                    " foxes are",
                ],
                "p_name": [
                    pd.NA,
                    pd.NA,
                    pd.NA,
                    "yellow turquoise cornflower coral saddle",
                    pd.NA,
                    pd.NA,
                    pd.NA,
                    "tomato goldenrod black turquoise maroon",
                    "cream dim blush moccasin drab",
                ],
                "p_size": [pd.NA, pd.NA, pd.NA, 37, pd.NA, pd.NA, pd.NA, 37, 37],
            }
        ),
    ),
]


@pytest_mark_snowflake
@pytest.mark.parametrize(
    "build_outer,probe_outer,expected_df",
    hash_join_basic_test_params,
)
def test_hash_join_basic(build_outer, probe_outer, expected_df, memory_leak_check):
    """
    Tests support for the basic inner/outer streaming hash join functionality.
    This also loads data from IO so we actually stream significant amounts of data.
    """
    conn_str = bodo.tests.utils.get_snowflake_connection_string(
        "SNOWFLAKE_SAMPLE_DATA", "TPCH_SF1"
    )
    l_orderkey_start = 685_475
    l_orderkey_end = 685_477
    p_orderkey_start = 183_700
    p_orderkey_end = 183_850
    p_size_limit = 37

    build_keys_inds = bodo.utils.typing.MetaType((0, 1))
    probe_keys_inds = bodo.utils.typing.MetaType((0, 1))
    build_col_meta = bodo.utils.typing.ColNamesMetaType(
        ("p_partkey", "p_comment", "p_name", "p_size")
    )
    probe_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "l_partkey",
            "l_comment",
            "l_orderkey",
        )
    )
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "l_partkey",
            "l_comment",
            "l_orderkey",
            "p_partkey",
            "p_comment",
            "p_name",
            "p_size",
        )
    )

    @bodo.jit
    def test_hash_join(conn):
        join_state = init_join_state(
            -1,
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            build_outer,
            probe_outer,
            16 * 1024 * 1024,
        )

        # read PART table and build join hash table
        reader1 = pd.read_sql(
            f"SELECT P_PARTKEY, P_COMMENT, P_NAME, P_SIZE FROM PART where P_SIZE = {p_size_limit} and p_partkey > {p_orderkey_start} and p_partkey < {p_orderkey_end}",
            conn,
            _bodo_chunksize=4000,
        )
        while True:
            table1, is_last1 = read_arrow_next(reader1, True)
            is_last1 = join_build_consume_batch(join_state, table1, is_last1)
            if is_last1:
                break

        partition_state = get_partition_state(join_state)

        # read LINEITEM table and probe
        reader2 = pd.read_sql(
            f"SELECT L_PARTKEY, L_COMMENT, L_ORDERKEY FROM LINEITEM where l_orderkey > {l_orderkey_start} and l_orderkey < {l_orderkey_end}",
            conn,
            _bodo_chunksize=4000,
        )
        __bodo_streaming_batches_table_builder_1 = (
            bodo.libs.table_builder.init_table_builder_state(-1)
        )
        while True:
            table2, is_last2 = read_arrow_next(reader2, True)
            out_table, is_last3, _ = join_probe_consume_batch(
                join_state, table2, is_last2, True
            )
            bodo.libs.table_builder.table_builder_append(
                __bodo_streaming_batches_table_builder_1, out_table
            )
            if is_last3:
                break
        delete_join_state(join_state)
        arrow_reader_del(reader1)
        arrow_reader_del(reader2)
        out_table = bodo.libs.table_builder.table_builder_finalize(
            __bodo_streaming_batches_table_builder_1
        )
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        return df, partition_state

    out_df, partition_state = test_hash_join(conn_str)
    assert partition_state == [(0, 0)]
    out_df = bodo.allgatherv(out_df)
    _test_equal(
        out_df,
        expected_df,
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )


# Listing them separately since otherwise the traceback during errors is huge.
nested_loop_join_test_params = [
    # Equivalent query:
    # select *
    # from (SELECT L_PARTKEY, L_COMMENT, L_ORDERKEY from lineitem where l_orderkey < 2 and l_partkey < 20000)
    # inner join (SELECT P_PARTKEY, P_COMMENT, P_NAME, P_SIZE from part where p_size > 49 and p_partkey > 199840) as part_filtered
    # on P_PARTKEY > L_PARTKEY + 197800
    (
        False,
        False,
        pd.DataFrame(
            {
                "l_partkey": [2132, 2132],
                "l_comment": ["lites. fluffily even de", "lites. fluffily even de"],
                "l_orderkey": [1, 1],
                "p_partkey": [199978, 199995],
                "p_comment": ["ess, i", "packa"],
                "p_name": [
                    "linen magenta saddle slate turquoise",
                    "blanched floral red maroon papaya",
                ],
                "p_size": [50, 50],
            }
        ),
    ),
    # Equivalent query:
    # select *
    # from (SELECT L_PARTKEY, L_COMMENT, L_ORDERKEY from lineitem where l_orderkey < 2 and l_partkey < 20000)
    # right join (SELECT P_PARTKEY, P_COMMENT, P_NAME, P_SIZE from part where p_size > 49 and p_partkey > 199840) as part_filtered
    # on P_PARTKEY > L_PARTKEY + 197800
    (
        True,
        False,
        pd.DataFrame(
            {
                "l_partkey": [2132, 2132, pd.NA, pd.NA, pd.NA],
                "l_comment": [
                    "lites. fluffily even de",
                    "lites. fluffily even de",
                    pd.NA,
                    pd.NA,
                    pd.NA,
                ],
                "l_orderkey": [1, 1, pd.NA, pd.NA, pd.NA],
                "p_partkey": [199978, 199995, 199843, 199847, 199898],
                "p_comment": [
                    "ess, i",
                    "packa",
                    "refully f",
                    " reques",
                    "around the",
                ],
                "p_name": [
                    "linen magenta saddle slate turquoise",
                    "blanched floral red maroon papaya",
                    "pale orchid deep linen chocolate",
                    "hot black red powder smoke",
                    "firebrick brown gainsboro orchid medium",
                ],
                "p_size": [50, 50, 50, 50, 50],
            }
        ),
    ),
    # Equivalent query:
    # select *
    # from (SELECT L_PARTKEY, L_COMMENT, L_ORDERKEY from lineitem where l_orderkey < 2 and l_partkey < 20000)
    # left join (SELECT P_PARTKEY, P_COMMENT, P_NAME, P_SIZE from part where p_size > 49 and p_partkey > 199840) as part_filtered
    # on P_PARTKEY > L_PARTKEY + 197800
    (
        False,
        True,
        pd.DataFrame(
            {
                "l_partkey": [2132, 2132, 15635],
                "l_comment": [
                    "lites. fluffily even de",
                    "lites. fluffily even de",
                    "arefully slyly ex",
                ],
                "l_orderkey": [1, 1, 1],
                "p_partkey": [199978, 199995, pd.NA],
                "p_comment": ["ess, i", "packa", pd.NA],
                "p_name": [
                    "linen magenta saddle slate turquoise",
                    "blanched floral red maroon papaya",
                    pd.NA,
                ],
                "p_size": [50, 50, pd.NA],
            }
        ),
    ),
    # Equivalent query:
    # select *
    # from (SELECT L_PARTKEY, L_COMMENT, L_ORDERKEY from lineitem where l_orderkey < 2 and l_partkey < 20000)
    # full outer join (SELECT P_PARTKEY, P_COMMENT, P_NAME, P_SIZE from part where p_size > 49 and p_partkey > 199840) as part_filtered
    # on P_PARTKEY > L_PARTKEY + 197800
    (
        True,
        True,
        pd.DataFrame(
            {
                "l_partkey": [2132, 2132, pd.NA, pd.NA, pd.NA, 15635],
                "l_comment": [
                    "lites. fluffily even de",
                    "lites. fluffily even de",
                    pd.NA,
                    pd.NA,
                    pd.NA,
                    "arefully slyly ex",
                ],
                "l_orderkey": [1, 1, pd.NA, pd.NA, pd.NA, 1],
                "p_partkey": [199978, 199995, 199843, 199847, 199898, pd.NA],
                "p_comment": [
                    "ess, i",
                    "packa",
                    "refully f",
                    " reques",
                    "around the",
                    pd.NA,
                ],
                "p_name": [
                    "linen magenta saddle slate turquoise",
                    "blanched floral red maroon papaya",
                    "pale orchid deep linen chocolate",
                    "hot black red powder smoke",
                    "firebrick brown gainsboro orchid medium",
                    pd.NA,
                ],
                "p_size": [50, 50, 50, 50, 50, pd.NA],
            }
        ),
    ),
]


@pytest_mark_snowflake
@pytest.mark.parametrize(
    "build_outer,probe_outer,expected_df",
    nested_loop_join_test_params,
)
@pytest.mark.parametrize("use_dict_encoding", [True, False])
def test_nested_loop_join(
    build_outer, probe_outer, expected_df, use_dict_encoding, memory_leak_check
):
    """
    Test streaming nested loop join
    """
    conn_str = bodo.tests.utils.get_snowflake_connection_string(
        "SNOWFLAKE_SAMPLE_DATA", "TPCH_SF1"
    )

    l_orderkey_end = 2
    l_partkey_end = 20_000
    p_size_limit = 49
    p_partkey_limit = 199_840

    build_keys_inds = bodo.utils.typing.MetaType(())
    probe_keys_inds = bodo.utils.typing.MetaType(())
    build_col_meta = bodo.utils.typing.ColNamesMetaType(
        ("p_partkey", "p_comment", "p_name", "p_size")
    )
    probe_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "l_partkey",
            "l_comment",
            "l_orderkey",
        )
    )
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "l_partkey",
            "l_comment",
            "l_orderkey",
            "p_partkey",
            "p_comment",
            "p_name",
            "p_size",
        )
    )
    non_equi_condition = "(right.p_partkey) > ((left.l_partkey) + 197800)"

    # select P_PARTKEY, P_COMMENT, P_NAME, P_SIZE, L_PARTKEY, L_COMMENT, L_ORDERKEY
    # from lineitem inner join part on P_PARTKEY > L_PARTKEY + 197800 where l_orderkey < 2 and p_size > 49

    @bodo.jit
    def test_nested_loop_join(conn):
        join_state = init_join_state(
            -1,
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            build_outer,
            probe_outer,
            non_equi_condition=non_equi_condition,
        )

        # read PART table
        reader1 = pd.read_sql(
            f"SELECT P_PARTKEY, P_COMMENT, P_NAME, P_SIZE FROM PART where P_SIZE > {p_size_limit} and P_PARTKEY > {p_partkey_limit}",
            conn,
            _bodo_chunksize=4000,
        )
        while True:
            table1, is_last1 = read_arrow_next(reader1, True)
            is_last1 = join_build_consume_batch(join_state, table1, is_last1)
            if is_last1:
                break

        # read LINEITEM table and probe
        reader2 = pd.read_sql(
            f"SELECT L_PARTKEY, L_COMMENT, L_ORDERKEY FROM LINEITEM where l_orderkey < {l_orderkey_end} and l_partkey < {l_partkey_end}",
            conn,
            _bodo_chunksize=4000,
        )
        _table_builder = bodo.libs.table_builder.init_table_builder_state(-1)
        while True:
            table2, is_last2 = read_arrow_next(reader2, True)
            out_table, is_last3, _ = join_probe_consume_batch(
                join_state, table2, is_last2, True
            )
            bodo.libs.table_builder.table_builder_append(_table_builder, out_table)
            if is_last3:
                break
        arrow_reader_del(reader1)
        arrow_reader_del(reader2)
        delete_join_state(join_state)
        out_table = bodo.libs.table_builder.table_builder_finalize(_table_builder)
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        return df

    # TODO[BSE-439]: Support dict-encoded strings
    saved_SF_READ_AUTO_DICT_ENCODE_ENABLED = (
        bodo.io.snowflake.SF_READ_AUTO_DICT_ENCODE_ENABLED
    )
    try:
        bodo.io.snowflake.SF_READ_AUTO_DICT_ENCODE_ENABLED = use_dict_encoding
        out_df = test_nested_loop_join(conn_str)
    finally:
        bodo.io.snowflake.SF_READ_AUTO_DICT_ENCODE_ENABLED = (
            saved_SF_READ_AUTO_DICT_ENCODE_ENABLED
        )
    out_df = bodo.allgatherv(out_df)
    _test_equal(
        out_df,
        expected_df,
        check_dtype=False,
        reset_index=True,
        sort_output=True,
    )


@pytest_mark_snowflake
@pytest.mark.parametrize("use_dict_encoding", [True, False])
def test_broadcast_nested_loop_join(use_dict_encoding, broadcast, memory_leak_check):
    """
    Test streaming broadcast nested loop join
    """
    conn_str = bodo.tests.utils.get_snowflake_connection_string(
        "SNOWFLAKE_SAMPLE_DATA", "TPCH_SF1"
    )

    l_orderkey_end = 2
    l_partkey_end = 20_000
    p_size_limit = 49
    p_partkey_limit = 199_840

    build_keys_inds = bodo.utils.typing.MetaType(())
    probe_keys_inds = bodo.utils.typing.MetaType(())
    build_col_meta = bodo.utils.typing.ColNamesMetaType(
        ("p_partkey", "p_comment", "p_name", "p_size")
    )
    probe_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "l_partkey",
            "l_comment",
            "l_orderkey",
        )
    )
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "l_partkey",
            "l_comment",
            "l_orderkey",
            "p_partkey",
            "p_comment",
            "p_name",
            "p_size",
        )
    )
    non_equi_condition = "(right.p_partkey) > ((left.l_partkey) + 197800)"

    # select L_PARTKEY, L_COMMENT, L_ORDERKEY, P_PARTKEY, P_COMMENT, P_NAME, P_SIZE
    # from lineitem inner join part on P_PARTKEY > L_PARTKEY + 197800 where l_orderkey < 2 and p_size > 49

    expected_df = pd.DataFrame(
        {
            "l_partkey": [2132, 2132],
            "l_comment": ["lites. fluffily even de", "lites. fluffily even de"],
            "l_orderkey": [1, 1],
            "p_partkey": [199978, 199995],
            "p_comment": ["ess, i", "packa"],
            "p_name": [
                "linen magenta saddle slate turquoise",
                "blanched floral red maroon papaya",
            ],
            "p_size": [50, 50],
        }
    )

    @bodo.jit
    def test_nested_loop_join(conn):
        join_state = init_join_state(
            -1,
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            False,
            False,
            non_equi_condition=non_equi_condition,
        )

        # read PART table
        reader1 = pd.read_sql(
            f"SELECT P_PARTKEY, P_COMMENT, P_NAME, P_SIZE FROM PART where P_SIZE > {p_size_limit} and P_PARTKEY > {p_partkey_limit}",
            conn,
            _bodo_chunksize=4000,
        )
        while True:
            table1, is_last1 = read_arrow_next(reader1, True)
            is_last1 = join_build_consume_batch(join_state, table1, is_last1)
            if is_last1:
                break

        # read LINEITEM table and probe
        reader2 = pd.read_sql(
            f"SELECT L_PARTKEY, L_COMMENT, L_ORDERKEY FROM LINEITEM where l_orderkey < {l_orderkey_end} and l_partkey < {l_partkey_end}",
            conn,
            _bodo_chunksize=4000,
        )
        _table_builder = bodo.libs.table_builder.init_table_builder_state(-1)
        while True:
            table2, is_last2 = read_arrow_next(reader2, True)
            out_table, is_last3, _ = join_probe_consume_batch(
                join_state, table2, is_last2, True
            )
            bodo.libs.table_builder.table_builder_append(_table_builder, out_table)
            if is_last3:
                break
        arrow_reader_del(reader1)
        arrow_reader_del(reader2)
        delete_join_state(join_state)
        out_table = bodo.libs.table_builder.table_builder_finalize(_table_builder)
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        out_df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        return out_df

    # TODO[BSE-439]: Support dict-encoded strings
    saved_SF_READ_AUTO_DICT_ENCODE_ENABLED = (
        bodo.io.snowflake.SF_READ_AUTO_DICT_ENCODE_ENABLED
    )
    try:
        bodo.io.snowflake.SF_READ_AUTO_DICT_ENCODE_ENABLED = use_dict_encoding
        with set_broadcast_join(broadcast):
            out_df = test_nested_loop_join(conn_str)
    finally:
        bodo.io.snowflake.SF_READ_AUTO_DICT_ENCODE_ENABLED = (
            saved_SF_READ_AUTO_DICT_ENCODE_ENABLED
        )
    out_df = bodo.allgatherv(out_df)
    _test_equal(
        out_df,
        expected_df,
        check_dtype=False,
        reset_index=True,
        sort_output=True,
    )


@pytest_mark_snowflake
def test_hash_join_reorder(memory_leak_check):
    """
    Test stream join where the keys have to be reordered to the front of
    the table.

    Note: The state still has to be initialized with the correct final order.
    """
    conn_str = bodo.tests.utils.get_snowflake_connection_string(
        "SNOWFLAKE_SAMPLE_DATA", "TPCH_SF1"
    )
    l_orderkey_start = 680_000
    l_orderkey_end = 690_000
    p_size_limit = 35
    # Equivalent query:
    # select P_PARTKEY, P_COMMENT, P_NAME, P_SIZE, L_PARTKEY, L_COMMENT, L_ORDERKEY
    # from lineitem inner join part on P_PARTKEY = L_PARTKEY and P_COMMENT = L_COMMENT
    # where l_orderkey > {l_orderkey_start} and l_orderkey < {l_orderkey_end}
    # and p_size > {p_size_limit}

    build_keys_inds = bodo.utils.typing.MetaType((3, 1))
    probe_keys_inds = bodo.utils.typing.MetaType((0, 2))
    build_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "p_name",
            "p_comment",
            "p_size",
            "p_partkey",
        )
    )
    probe_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "l_partkey",
            "l_orderkey",
            "l_comment",
        )
    )
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "l_partkey",
            "l_orderkey",
            "l_comment",
            "p_name",
            "p_comment",
            "p_size",
            "p_partkey",
        )
    )

    @bodo.jit
    def test_hash_join(conn):
        join_state = init_join_state(
            -1,
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            False,
            False,
        )

        # read PART table and build join hash table
        reader1 = pd.read_sql(
            f"SELECT P_NAME, P_COMMENT, P_SIZE, P_PARTKEY FROM PART where P_SIZE > {p_size_limit}",
            conn,
            _bodo_chunksize=4000,
        )
        while True:
            table1, is_last1 = read_arrow_next(reader1, True)
            is_last1 = join_build_consume_batch(join_state, table1, is_last1)
            if is_last1:
                break

        # read LINEITEM table and probe
        reader2 = pd.read_sql(
            f"SELECT L_PARTKEY, L_ORDERKEY, L_COMMENT FROM LINEITEM where l_orderkey > {l_orderkey_start} and l_orderkey < {l_orderkey_end}",
            conn,
            _bodo_chunksize=4000,
        )
        _table_builder = bodo.libs.table_builder.init_table_builder_state(-1)
        while True:
            table2, is_last2 = read_arrow_next(reader2, True)
            out_table, is_last3, _ = join_probe_consume_batch(
                join_state, table2, is_last2, True
            )
            bodo.libs.table_builder.table_builder_append(_table_builder, out_table)
            if is_last3:
                break
        delete_join_state(join_state)
        arrow_reader_del(reader1)
        arrow_reader_del(reader2)
        out_table = bodo.libs.table_builder.table_builder_finalize(_table_builder)
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        return df

    out_df = test_hash_join(conn_str)
    expected_df = pd.DataFrame(
        {
            "l_partkey": [183728],
            "l_orderkey": [685476],
            "l_comment": ["bold deposi"],
            "p_name": ["yellow turquoise cornflower coral saddle"],
            "p_comment": ["bold deposi"],
            "p_size": [37],
            "p_partkey": [183728],
        }
    )
    out_df = bodo.allgatherv(out_df)
    _test_equal(
        out_df,
        expected_df,
        check_dtype=False,
    )


# Note we mark this as slow because the volume of data in
# the output makes checking correctness slow.
@pytest.mark.slow
@pytest.mark.parametrize(
    "build_outer,probe_outer",
    [
        pytest.param(True, True, id="full-outer"),
        pytest.param(True, False, id="right"),
        pytest.param(True, False, id="left"),
    ],
)
def test_hash_join_non_nullable_outer(build_outer, probe_outer, memory_leak_check):
    """
    Test stream join where an outer join requires casting non-nullable
    types to nullable types.

    The codegen here is heavily influence by the BodoSQL generated code.
    """
    df1 = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5] * 25,
            "B": np.array([1, 2, 3, 4, 5] * 25, dtype=np.int32),
        }
    )
    df2 = pd.DataFrame(
        {
            "C": [2, 6] * 25,
            "D": np.array([2, 6] * 25, dtype=np.int8),
        }
    )
    build_keys_inds = bodo.utils.typing.MetaType((0,))
    probe_keys_inds = bodo.utils.typing.MetaType((0,))
    kept_cols = bodo.utils.typing.MetaType((0, 1))
    build_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
        )
    )
    probe_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
        )
    )
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
            "A",
            "B",
        )
    )

    def test_hash_join(df1, df2):
        join_state = init_join_state(
            -1,
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            build_outer,
            probe_outer,
        )
        _temp1 = 0
        is_last1 = False
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), kept_cols, 2
        )
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_temp1 * 4000), ((_temp1 + 1) * 4000))
            )
            is_last1 = (_temp1 * 4000) >= len(df1)
            _temp1 = _temp1 + 1
            is_last1 = join_build_consume_batch(join_state, T2, is_last1)

        _temp2 = 0
        is_last2 = False
        is_last3 = False
        T3 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2), (), kept_cols, 2
        )
        _table_builder = bodo.libs.table_builder.init_table_builder_state(-1)
        while not is_last3:
            T4 = bodo.hiframes.table.table_local_filter(
                T3, slice((_temp2 * 4000), ((_temp2 + 1) * 4000))
            )
            is_last2 = (_temp2 * 4000) >= len(df2)
            _temp2 = _temp2 + 1
            out_table, is_last3, _ = join_probe_consume_batch(
                join_state, T4, is_last2, True
            )
            bodo.libs.table_builder.table_builder_append(_table_builder, out_table)
        delete_join_state(join_state)
        out_table = bodo.libs.table_builder.table_builder_finalize(_table_builder)
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        return df

    # Generate expected output for each type of join
    right_outer = pd.DataFrame(
        {
            "C": pd.array([None] * 100, dtype="Int64"),
            "D": pd.array([None] * 100, dtype="Int8"),
            "A": [1, 3, 4, 5] * 25,
            "B": np.array([1, 3, 4, 5] * 25, dtype=np.int32),
        }
    )
    left_outer = pd.DataFrame(
        {
            "C": [6] * 25,
            "D": pd.array([6] * 25, dtype="int8"),
            "A": pd.array([None] * 25, dtype="Int64"),
            "B": pd.array([None] * 25, dtype="Int32"),
        }
    )
    inner = pd.DataFrame(
        {
            "C": [2] * 25 * 25,
            "D": np.array([2] * 25 * 25, dtype=np.int8),
            "A": [2] * 25 * 25,
            "B": np.array([2] * 25 * 25, dtype=np.int32),
        }
    )
    # Fuse the outputs and cast.
    if build_outer and probe_outer:
        left_outer = left_outer.astype({"A": "Int64", "B": "Int32"})
        right_outer = right_outer.astype({"C": "Int64", "D": "Int8"})
        inner = inner.astype({"A": "Int64", "B": "Int32", "C": "Int64", "D": "Int8"})
        expected_df = pd.concat([left_outer, right_outer, inner])
    elif probe_outer:
        inner = inner.astype({"C": "Int64", "D": "Int8"})
        expected_df = pd.concat([left_outer, inner])
    else:
        inner = inner.astype({"A": "Int64", "B": "Int32"})
        expected_df = pd.concat([right_outer, inner])
    check_func(
        test_hash_join,
        (df1, df2),
        py_output=expected_df,
        reset_index=True,
        sort_output=True,
    )


@pytest.mark.parametrize(
    "probe_outer",
    [
        # Note for checking types we only need to check 1 side of
        # an outer join.
        pytest.param(False, id="inner"),
        pytest.param(True, id="left", marks=pytest.mark.slow),
    ],
)
def test_hash_join_key_cast(probe_outer, memory_leak_check):
    df1 = pd.DataFrame(
        {
            "A": np.array(list(range(2500)) * 5, dtype=np.int16),
            "B": np.array(list(range(2500)) * 5, dtype=np.int32),
        }
    )
    df2 = pd.DataFrame(
        {
            "C": pd.array([2, -1] * 5, dtype="Int8"),
            "D": np.array([2, -1] * 5, dtype=np.int8),
        }
    )
    build_keys_inds = bodo.utils.typing.MetaType((0, 1))
    probe_keys_inds = bodo.utils.typing.MetaType((0, 1))
    kept_cols = bodo.utils.typing.MetaType((0, 1))
    build_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
        )
    )
    probe_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
        )
    )
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
            "A",
            "B",
        )
    )

    def test_hash_join(df1, df2):
        join_state = init_join_state(
            -1,
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            False,
            probe_outer,
        )
        _temp1 = 0
        is_last1 = False
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), kept_cols, 2
        )
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_temp1 * 4000), ((_temp1 + 1) * 4000))
            )
            is_last1 = (_temp1 * 4000) >= len(df1)
            _temp1 = _temp1 + 1
            is_last1 = join_build_consume_batch(join_state, T2, is_last1)

        _temp2 = 0
        is_last2 = False
        is_last3 = False
        T3 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2), (), kept_cols, 2
        )
        _table_builder = bodo.libs.table_builder.init_table_builder_state(-1)
        while not is_last3:
            T4 = bodo.hiframes.table.table_local_filter(
                T3, slice((_temp2 * 4000), ((_temp2 + 1) * 4000))
            )
            is_last2 = (_temp2 * 4000) >= len(df2)
            _temp2 = _temp2 + 1
            out_table, is_last3, _ = join_probe_consume_batch(
                join_state, T4, is_last2, True
            )
            bodo.libs.table_builder.table_builder_append(_table_builder, out_table)
        delete_join_state(join_state)
        out_table = bodo.libs.table_builder.table_builder_finalize(_table_builder)
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        return df

    # Generate expected output for each type of join
    left_outer = pd.DataFrame(
        {
            "C": pd.array([-1] * 5, dtype="Int16"),
            "D": pd.array([-1] * 5, dtype="int32"),
            "A": pd.array([None] * 5, dtype="Int16"),
            "B": pd.array([None] * 5, dtype="Int32"),
        }
    )
    inner = pd.DataFrame(
        {
            "C": pd.array([2] * 25, dtype="Int16"),
            "D": np.array([2] * 25, dtype=np.int32),
            "A": pd.array([2] * 25, dtype="Int16"),
            "B": np.array([2] * 25, dtype=np.int32),
        }
    )
    # Fuse the outputs and cast.
    if probe_outer:
        inner = inner.astype({"B": "Int32"})
        expected_df = pd.concat([left_outer, inner])
    else:
        expected_df = inner

    check_func(
        test_hash_join,
        (df1, df2),
        py_output=expected_df,
        reset_index=True,
        sort_output=True,
    )


@pytest.mark.parametrize(
    "build_outer,probe_outer",
    [
        pytest.param(False, False, id="inner"),
        pytest.param(True, True, id="full-outer"),
        pytest.param(True, False, id="right", marks=pytest.mark.slow),
        pytest.param(False, True, id="left", marks=pytest.mark.slow),
    ],
)
def test_non_equi_join_cond(build_outer, probe_outer, broadcast, memory_leak_check):
    """Test streaming hash join with a non-equality condition."""
    df1 = pd.DataFrame(
        {
            "A": np.array(list(range(2500)) * 5, dtype=np.int16),
            "B": np.array(list(range(2500)) * 5, dtype=np.int32),
        }
    )
    df2 = pd.DataFrame(
        {
            "C": pd.array([2, 4] * 5, dtype="Int8"),
            "D": np.array([4, 2] * 5, dtype=np.int8),
        }
    )
    build_keys_inds = bodo.utils.typing.MetaType((0,))
    probe_keys_inds = bodo.utils.typing.MetaType((0,))
    kept_cols = bodo.utils.typing.MetaType((0, 1))
    build_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
        )
    )
    probe_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
        )
    )
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
            "A",
            "B",
        )
    )
    non_equi_cond = "(right.`A` < left.`D`)"

    def test_hash_join(df1, df2):
        join_state = init_join_state(
            -1,
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            build_outer,
            probe_outer,
            non_equi_condition=non_equi_cond,
        )
        _temp1 = 0
        is_last1 = False
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), kept_cols, 2
        )
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_temp1 * 4000), ((_temp1 + 1) * 4000))
            )
            is_last1 = (_temp1 * 4000) >= len(df1)
            _temp1 = _temp1 + 1
            is_last1 = join_build_consume_batch(join_state, T2, is_last1)

        _temp2 = 0
        is_last2 = False
        is_last3 = False
        T3 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2), (), kept_cols, 2
        )
        _table_builder = bodo.libs.table_builder.init_table_builder_state(-1)
        while not is_last3:
            T4 = bodo.hiframes.table.table_local_filter(
                T3, slice((_temp2 * 4000), ((_temp2 + 1) * 4000))
            )
            is_last2 = (_temp2 * 4000) >= len(df2)
            _temp2 = _temp2 + 1
            out_table, is_last3, _ = join_probe_consume_batch(
                join_state, T4, is_last2, True
            )
            bodo.libs.table_builder.table_builder_append(_table_builder, out_table)
        delete_join_state(join_state)
        out_table = bodo.libs.table_builder.table_builder_finalize(_table_builder)
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        return df

    # Generate expected output for each type of join
    left_missed_keys = list(set(range(2500)) - {2}) * 5
    build_outer_df = pd.DataFrame(
        {
            "C": pd.array([None] * len(left_missed_keys), dtype="Int16"),
            "D": pd.array([None] * len(left_missed_keys), dtype="Int8"),
            "A": pd.array(left_missed_keys, dtype="Int16"),
            "B": pd.array(left_missed_keys, dtype="int32"),
        }
    )
    probe_outer_df = pd.DataFrame(
        {
            "C": pd.array([4] * 5, dtype="Int16"),
            "D": pd.array([2] * 5, dtype="int8"),
            "A": pd.array([None] * 5, dtype="Int16"),
            "B": pd.array([None] * 5, dtype="Int32"),
        }
    )
    inner = pd.DataFrame(
        {
            "C": pd.array([2] * 25, dtype="Int16"),
            "D": np.array([4] * 25, dtype=np.int8),
            "A": pd.array([2] * 25, dtype="Int16"),
            "B": np.array([2] * 25, dtype=np.int32),
        }
    )
    # Fuse the outputs and cast.
    if build_outer and probe_outer:
        build_outer_df = build_outer_df.astype({"B": "Int32"})
        probe_outer_df = probe_outer_df.astype({"D": "Int8"})
        inner = inner.astype({"B": "Int32", "D": "Int8"})
        expected_df = pd.concat([build_outer_df, probe_outer_df, inner])
    elif build_outer:
        inner = inner.astype({"D": "Int8"})
        expected_df = pd.concat([build_outer_df, inner])
    elif probe_outer:
        inner = inner.astype({"B": "Int32"})
        expected_df = pd.concat([probe_outer_df, inner])
    else:
        expected_df = inner
    with set_broadcast_join(broadcast):
        check_func(
            test_hash_join,
            (df1, df2),
            py_output=expected_df,
            reset_index=True,
            sort_output=True,
        )


def test_join_key_prune(memory_leak_check):
    """Test streaming hash join with pruning columns from
    the output."""
    df1 = pd.DataFrame(
        {
            "A": np.array(list(range(2500)) * 5, dtype=np.int16),
            "B": np.array(list(range(2500)) * 5, dtype=np.int32),
        }
    )
    df2 = pd.DataFrame(
        {
            "C": pd.array([2, 4] * 5, dtype="Int8"),
            "D": np.array([4, 2] * 5, dtype=np.int8),
        }
    )
    build_keys_inds = bodo.utils.typing.MetaType((0,))
    probe_keys_inds = bodo.utils.typing.MetaType((0,))
    kept_cols = bodo.utils.typing.MetaType((0, 1))
    build_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
        )
    )
    probe_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
        )
    )
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
            "A",
            "B",
        )
    )

    def test_hash_join(df1, df2):
        join_state = init_join_state(
            -1,
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            False,
            False,
        )
        _temp1 = 0
        is_last1 = False
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), kept_cols, 2
        )
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_temp1 * 4000), ((_temp1 + 1) * 4000))
            )
            is_last1 = (_temp1 * 4000) >= len(df1)
            _temp1 = _temp1 + 1
            is_last1 = join_build_consume_batch(join_state, T2, is_last1)

        _temp2 = 0
        is_last2 = False
        is_last3 = False
        T3 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2), (), kept_cols, 2
        )
        _table_builder = bodo.libs.table_builder.init_table_builder_state(-1)
        while not is_last3:
            T4 = bodo.hiframes.table.table_local_filter(
                T3, slice((_temp2 * 4000), ((_temp2 + 1) * 4000))
            )
            is_last2 = (_temp2 * 4000) >= len(df2)
            _temp2 = _temp2 + 1
            out_table, is_last3, _ = join_probe_consume_batch(
                join_state, T4, is_last2, True
            )
            bodo.libs.table_builder.table_builder_append(_table_builder, out_table)
        delete_join_state(join_state)
        out_table = bodo.libs.table_builder.table_builder_finalize(_table_builder)
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        return df[["B", "D"]]

    expected_df = pd.DataFrame(
        {
            "B": np.array([2, 4] * 25, dtype=np.int32),
            "D": np.array([4, 2] * 25, dtype=np.int8),
        }
    )
    # TODO: Figure out how to verify pruning in the testing.
    check_func(
        test_hash_join,
        (df1, df2),
        py_output=expected_df,
        reset_index=True,
        sort_output=True,
    )


def test_key_multicast(memory_leak_check):
    """
    Test stream join where a key is used in multiple comparisons. Since we use
    1 column for a column used in multiple comparisons, we need to make sure
    that both uses of the key share a common type.

    The codegen here is heavily influence by the BodoSQL generated code.
    """
    df1 = pd.DataFrame({"A": np.array([1, 2, 3, 4, 5] * 2500, dtype=np.int8)})
    df2 = pd.DataFrame(
        {
            "C": np.array([2, 6] * 2500, dtype=np.int64),
            "D": np.array([2, 6] * 2500, dtype=np.int32),
        }
    )
    build_keys_inds = bodo.utils.typing.MetaType((0, 0))
    probe_keys_inds = bodo.utils.typing.MetaType((0, 1))
    kept_cols1 = bodo.utils.typing.MetaType((0,))
    kept_cols2 = bodo.utils.typing.MetaType((0, 1))
    build_col_meta = bodo.utils.typing.ColNamesMetaType(("A",))
    probe_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
        )
    )
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
            "A",
        )
    )

    def test_hash_join(df1, df2):
        join_state = init_join_state(
            -1,
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            False,
            False,
        )
        _temp1 = 0
        is_last1 = False
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1),
            (),
            kept_cols1,
            2,
        )
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_temp1 * 4000), ((_temp1 + 1) * 4000))
            )
            is_last1 = (_temp1 * 4000) >= len(df1)
            _temp1 = _temp1 + 1
            is_last1 = join_build_consume_batch(join_state, T2, is_last1)

        _temp2 = 0
        is_last2 = False
        is_last3 = False
        T3 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2),
            (),
            kept_cols2,
            2,
        )
        _table_builder = bodo.libs.table_builder.init_table_builder_state(-1)
        while not is_last3:
            T4 = bodo.hiframes.table.table_local_filter(
                T3, slice((_temp2 * 4000), ((_temp2 + 1) * 4000))
            )
            is_last2 = (_temp2 * 4000) >= len(df2)
            _temp2 = _temp2 + 1
            out_table, is_last3, _ = join_probe_consume_batch(
                join_state, T4, is_last2, True
            )
            bodo.libs.table_builder.table_builder_append(_table_builder, out_table)
        delete_join_state(join_state)
        out_table = bodo.libs.table_builder.table_builder_finalize(_table_builder)
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        return df

    # Everything should be upcast to int64.
    expected_df = pd.DataFrame(
        {
            "C": [2] * 2500 * 2500,
            "D": [2] * 2500 * 2500,
            "A": [2] * 2500 * 2500,
        }
    )
    check_func(
        test_hash_join,
        (df1, df2),
        py_output=expected_df,
        reset_index=True,
        sort_output=True,
    )


@pytest.mark.parametrize(
    "build_outer,probe_outer,build_dist",
    [
        pytest.param(False, False, False, id="inner-probe_dist"),
        pytest.param(False, False, True, id="inner-build_dist", marks=pytest.mark.slow),
        pytest.param(
            True, True, False, id="full-outer-probe-dist", marks=pytest.mark.slow
        ),
        pytest.param(True, True, True, id="full-outer-build-dist"),
        pytest.param(True, False, False, id="right-probe_dist", marks=pytest.mark.slow),
        pytest.param(True, False, True, id="right-build_dist", marks=pytest.mark.slow),
        pytest.param(False, True, False, id="left-probe_dist", marks=pytest.mark.slow),
        pytest.param(False, True, True, id="left-build_dist", marks=pytest.mark.slow),
    ],
)
def test_only_one_distributed(
    build_outer, probe_outer, build_dist, broadcast, memory_leak_check
):
    """Test streaming hash join where only build or probe table is
    distributed but not both."""
    df1 = pd.DataFrame(
        {
            "A": pd.array(list(range(2500)) * 5, dtype="Int64"),
            "B": pd.array(list(range(2500)) * 5, dtype="Int64"),
        }
    )
    df2 = pd.DataFrame(
        {
            "C": pd.array([2, -1] * 5, dtype="Int64"),
            "D": pd.array([2, -1] * 5, dtype="Int64"),
        }
    )

    if build_dist:
        df1 = _get_dist_arg(df1)
    else:
        df2 = _get_dist_arg(df2)

    build_keys_inds = bodo.utils.typing.MetaType((0,))
    probe_keys_inds = bodo.utils.typing.MetaType((0,))
    kept_cols = bodo.utils.typing.MetaType((0, 1))
    build_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
        )
    )
    probe_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
        )
    )
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
            "A",
            "B",
        )
    )

    dist_df = "df1" if build_dist else "df2"

    @bodo.jit(distributed=[dist_df])
    def test_hash_join(df1, df2):
        join_state = init_join_state(
            -1,
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            build_outer,
            probe_outer,
        )
        _temp1 = 0
        is_last1 = False
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), kept_cols, 2
        )
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_temp1 * 4000), ((_temp1 + 1) * 4000))
            )
            is_last1 = (_temp1 * 4000) >= len(df1)
            _temp1 = _temp1 + 1
            is_last1 = join_build_consume_batch(join_state, T2, is_last1)

        _temp2 = 0
        is_last2 = False
        is_last3 = False
        T3 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2), (), kept_cols, 2
        )
        _table_builder = bodo.libs.table_builder.init_table_builder_state(-1)
        while not is_last3:
            T4 = bodo.hiframes.table.table_local_filter(
                T3, slice((_temp2 * 4000), ((_temp2 + 1) * 4000))
            )
            is_last2 = (_temp2 * 4000) >= len(df2)
            _temp2 = _temp2 + 1
            out_table, is_last3, _ = join_probe_consume_batch(
                join_state, T4, is_last2, True
            )
            bodo.libs.table_builder.table_builder_append(_table_builder, out_table)
        delete_join_state(join_state)
        out_table = bodo.libs.table_builder.table_builder_finalize(_table_builder)
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        return df

    # Generate expected output for each type of join
    left_missed_keys = list(set(range(2500)) - {2}) * 5
    build_outer_df = pd.DataFrame(
        {
            "C": pd.array([None] * len(left_missed_keys), dtype="Int64"),
            "D": pd.array([None] * len(left_missed_keys), dtype="Int64"),
            "A": pd.array(left_missed_keys, dtype="Int64"),
            "B": pd.array(left_missed_keys, dtype="Int64"),
        }
    )
    probe_outer_df = pd.DataFrame(
        {
            "C": pd.array([-1] * 5, dtype="Int64"),
            "D": pd.array([-1] * 5, dtype="Int64"),
            "A": pd.array([None] * 5, dtype="Int64"),
            "B": pd.array([None] * 5, dtype="Int32"),
        }
    )
    inner = pd.DataFrame(
        {
            "C": pd.array([2] * 25, dtype="Int64"),
            "D": pd.array([2] * 25, dtype="Int64"),
            "A": pd.array([2] * 25, dtype="Int64"),
            "B": pd.array([2] * 25, dtype="Int64"),
        }
    )
    # Fuse the outputs and cast.
    if build_outer and probe_outer:
        expected_df = pd.concat([build_outer_df, probe_outer_df, inner])
    elif build_outer:
        expected_df = pd.concat([build_outer_df, inner])
    elif probe_outer:
        expected_df = pd.concat([probe_outer_df, inner])
    else:
        expected_df = inner

    with set_broadcast_join(broadcast):
        bodo_output = test_hash_join(df1, df2)

    out_df = bodo.allgatherv(bodo_output)
    _test_equal(
        out_df,
        expected_df,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.skipif(
    bodo.get_size() > 1,
    reason="Only test with a single rank to simplify the checks",
)
def test_long_strings_chunked_table_builder(memory_leak_check):
    """
    Tests for the edge cases related to handling of long strings
    in ChunkedTableBuilder. The output buffers of streaming
    hash join use the ChunkedTableBuilder, so we use that to
    test the behavior of ChunkedTableBuilder indirectly.
    """

    def generate_random_string(length):
        # Get all the ASCII letters in lowercase
        letters = string.ascii_lowercase
        random_string = "".join(f"{random.choice(letters)}" for _ in range(length))
        return random_string

    # This is the minimum size of any of the allocated buffers in ChunkedTableBuilder
    smallest_alloc_size = BufferPool.default().get_smallest_size_class_size()
    # Should match CHUNKED_TABLE_DEFAULT_STRING_PREALLOCATION defined in
    # _chunked_table_builder.h
    string_prealloc = 32
    num_prealloc_strings_in_smallest_frame = smallest_alloc_size // string_prealloc
    # Should match DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES defined in
    # _stream_join.h
    max_resize_count = 2

    # Use a single int column as key. Keep all columns around.
    build_keys_inds = bodo.utils.typing.MetaType((0,))
    probe_keys_inds = bodo.utils.typing.MetaType((0,))
    kept_cols = bodo.utils.typing.MetaType((0, 1))
    build_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
        )
    )
    probe_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
        )
    )
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
            "A",
            "B",
        )
    )

    @bodo.jit(distributed=["df1", "df2"])
    def test_hash_join_impl(df1, df2, batch_size):
        join_state = init_join_state(
            -1,
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            False,
            False,
        )
        _temp1 = 0
        is_last1 = False
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1),
            (),
            kept_cols,
            2,
        )
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_temp1 * 4000), ((_temp1 + 1) * 4000))
            )
            is_last1 = (_temp1 * batch_size) >= len(df1)
            _temp1 = _temp1 + 1

            is_last1 = join_build_consume_batch(join_state, T2, is_last1)

        _temp2 = 0
        out_dfs = []
        is_last2 = False
        is_last3 = False
        T3 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2),
            (),
            kept_cols,
            2,
        )
        while not is_last3:
            T4 = bodo.hiframes.table.table_local_filter(
                T3, slice((_temp2 * 4000), ((_temp2 + 1) * 4000))
            )
            is_last2 = ((_temp2 + 1) * batch_size) >= len(df2)
            _temp2 = _temp2 + 1
            out_table, is_last3, _ = join_probe_consume_batch(
                join_state, T4, is_last2, True
            )
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(out_table), 1, None
            )
            df_final = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (out_table,), index_var, col_meta
            )
            out_dfs.append(df_final)
        delete_join_state(join_state)
        return out_dfs

    def test_helper(df1, df2, batch_size):
        """
        Helper function for the tests. Does the required setup
        and reset for the tests.

        Args:
            df1 (pd.DataFrame): Build Table
            df2 (pd.DataFrame): Probe Table
            batch_size (int): Batch size to use

        Returns:
            List[pd.DataFrame]: Output chunks
        """
        # Force string dtype
        saved_SF_READ_AUTO_DICT_ENCODE_ENABLED = (
            bodo.io.snowflake.SF_READ_AUTO_DICT_ENCODE_ENABLED
        )
        # Set batch size
        saved_bodosql_streaming_batch_size = bodo.bodosql_streaming_batch_size

        try:
            bodo.io.snowflake.SF_READ_AUTO_DICT_ENCODE_ENABLED = False
            bodo.bodosql_streaming_batch_size = batch_size
            # Use non-broadcast to simplify testing
            with temp_env_override({"BODO_BCAST_JOIN_THRESHOLD": "0"}):
                # Test is only run with one rank, so we don't need
                # to use _get_dist_arg.
                out_dfs = test_hash_join_impl(
                    df1, df2, bodo.bodosql_streaming_batch_size
                )
                return out_dfs
        finally:
            bodo.io.snowflake.SF_READ_AUTO_DICT_ENCODE_ENABLED = (
                saved_SF_READ_AUTO_DICT_ENCODE_ENABLED
            )
            bodo.bodosql_streaming_batch_size = saved_bodosql_streaming_batch_size

    # Test 1: Check that resizing works as expected. We will make it
    # so that each string is (string_prealloc) * 2**(max_resize_count)
    # bytes and verify that a single batch fits the entire output.
    def test1():
        str_len = string_prealloc * (2**max_resize_count)
        batch_size = num_prealloc_strings_in_smallest_frame
        build_df = pd.DataFrame(
            {
                "A": pd.array(np.arange(batch_size), dtype="Int64"),
                "B": pd.array(np.arange(batch_size), dtype="Int64"),
            }
        )
        probe_df = pd.DataFrame(
            {
                "C": pd.array(np.arange(batch_size), dtype="Int64"),
                "D": [generate_random_string(str_len)] * batch_size,
            }
        )
        out_dfs = test_helper(build_df, probe_df, batch_size)
        assert len(out_dfs) == bodo.stream_loop_sync_iters
        assert len(out_dfs[bodo.stream_loop_sync_iters - 1]) == batch_size
        for i in range(bodo.stream_loop_sync_iters - 1):
            assert len(out_dfs[i]) == 0
        assert (
            out_dfs[bodo.stream_loop_sync_iters - 1]["D"].str.len()
            == pd.Series([str_len] * batch_size)
        ).all()

    # Test 2: Check that if each string is > ((string_prealloc) * 2**(max_resize_count))
    # bytes, we get expected number of batches and the size of the batches is as expected.
    def test2():
        str_len = (string_prealloc * (2**max_resize_count)) + 10
        batch_size = num_prealloc_strings_in_smallest_frame
        max_buffer_size = max(
            smallest_alloc_size,
            (batch_size * string_prealloc * (2**max_resize_count)),
        )
        exp_out_chunks = math.ceil(str_len * batch_size / max_buffer_size)
        # Round to nearest bodo.stream_loop_sync_iters to account for the fact that is_last
        # can only be true every bodo.stream_loop_sync_iters batches.
        exp_out_chunks_rounded = (
            math.ceil(exp_out_chunks / bodo.stream_loop_sync_iters)
            * bodo.stream_loop_sync_iters
        )
        build_df = pd.DataFrame(
            {
                "A": pd.array(np.arange(batch_size), dtype="Int64"),
                "B": pd.array(np.arange(batch_size), dtype="Int64"),
            }
        )
        probe_df = pd.DataFrame(
            {
                "C": pd.array(np.arange(batch_size), dtype="Int64"),
                "D": [generate_random_string(str_len)] * batch_size,
            }
        )
        out_dfs = test_helper(build_df, probe_df, batch_size)
        assert len(out_dfs) == exp_out_chunks_rounded
        assert len(out_dfs[0]) == (max_buffer_size // str_len)
        assert (
            out_dfs[0]["D"].str.len()
            == pd.Series([str_len] * (max_buffer_size // str_len))
        ).all()
        for i in range(1, exp_out_chunks_rounded - 1):
            assert len(out_dfs[i]) == 0
        assert sum([len(df) for df in out_dfs]) == batch_size

    # Test 3: Check that if we create a string that's larger than
    # max buffer size, the batch size is 1 if it's the first one.
    def test3():
        batch_size = num_prealloc_strings_in_smallest_frame
        max_buffer_size = max(
            smallest_alloc_size,
            (batch_size * string_prealloc * (2**max_resize_count)),
        )
        build_df = pd.DataFrame(
            {
                "A": pd.array(np.arange(batch_size), dtype="Int64"),
                "B": pd.array(np.arange(batch_size), dtype="Int64"),
            }
        )
        probe_df = pd.DataFrame(
            {
                "C": pd.array(np.arange(batch_size), dtype="Int64"),
                "D": [generate_random_string(max_buffer_size + 10)]
                + ([generate_random_string(string_prealloc)] * (batch_size - 1)),
            }
        )
        out_dfs = test_helper(build_df, probe_df, batch_size)
        assert len(out_dfs) == bodo.stream_loop_sync_iters
        assert len(out_dfs[0]) == 1
        assert len(out_dfs[-1]) == (batch_size - 1)
        assert len(out_dfs[0]["D"][0]) == max_buffer_size + 10
        for i in range(1, bodo.stream_loop_sync_iters - 2):
            assert len(out_dfs[i]) == 0

    # Test 4: Fill up buffer so that it has resized one fewer than max allowed times.
    # Then add a string that's larger than max buffer size, and it should still
    # be returned in the first batch.
    def test4():
        batch_size = num_prealloc_strings_in_smallest_frame
        max_buffer_size = max(
            smallest_alloc_size,
            (batch_size * string_prealloc * (2**max_resize_count)),
        )

        # We need to allocate total of str_len_p1_sum to get the buffer to resize
        # required number of times.
        str_len_p1_sum = string_prealloc * (2 ** (max_resize_count - 1)) * batch_size
        # Let's have (batch_size - 2) strings of length str_len_p11
        str_len_p11 = (
            string_prealloc * (2 ** (max_resize_count - 1)) * batch_size
        ) // (batch_size - 2)
        # Let's have 1 string of length required to get to str_len_p1_sum
        str_len_p12 = str_len_p1_sum - (str_len_p11 * (batch_size - 2))
        # Last string will be of size:
        str_len_p2 = max_buffer_size * 2

        build_df = pd.DataFrame(
            {
                "A": pd.array(np.arange(batch_size), dtype="Int64"),
                "B": pd.array(np.arange(batch_size), dtype="Int64"),
            }
        )
        probe_df = pd.DataFrame(
            {
                "C": pd.array(np.arange(batch_size), dtype="Int64"),
                "D": ([generate_random_string(str_len_p11)] * (batch_size - 2))
                + [
                    generate_random_string(str_len_p12),
                    generate_random_string(str_len_p2),
                ],
            }
        )
        out_dfs = test_helper(build_df, probe_df, batch_size)
        assert len(out_dfs) == bodo.stream_loop_sync_iters
        assert len(out_dfs[-1]) == batch_size
        assert len(out_dfs[-1]["D"][batch_size - 1]) == str_len_p2
        for i in range(bodo.stream_loop_sync_iters - 2):
            assert len(out_dfs[i]) == 0

    # Test 5: Fill up the buffer so that it has resized max number of times.
    # Then, try adding a string that's larger than max allowed size and verify
    # that it's added to the second output chunk and not the first one.
    def test5():
        batch_size = num_prealloc_strings_in_smallest_frame
        max_buffer_size = max(
            smallest_alloc_size,
            (batch_size * string_prealloc * (2**max_resize_count)),
        )

        # We need to allocate total of str_len_p1_sum to get the buffer to resize
        # required number of times.
        str_len_p1_sum = string_prealloc * (2**max_resize_count) * batch_size
        # Let's have (batch_size - 2) strings of length str_len_p11
        str_len_p11 = (
            string_prealloc * (2 ** (max_resize_count - 1)) * batch_size
        ) // (batch_size - 2)
        # Let's have 1 string of length required to get to slightly less than str_len_p1_sum
        str_len_p12 = max(str_len_p1_sum - (str_len_p11 * (batch_size - 2)) - 1, 0)
        # Last string will be of size:
        str_len_p2 = max_buffer_size * 2

        build_df = pd.DataFrame(
            {
                "A": pd.array(np.arange(batch_size), dtype="Int64"),
                "B": pd.array(np.arange(batch_size), dtype="Int64"),
            }
        )
        probe_df = pd.DataFrame(
            {
                "C": pd.array(np.arange(batch_size), dtype="Int64"),
                "D": ([generate_random_string(str_len_p11)] * (batch_size - 2))
                + [
                    generate_random_string(str_len_p12),
                    generate_random_string(str_len_p2),
                ],
            }
        )
        out_dfs = test_helper(build_df, probe_df, batch_size)
        assert len(out_dfs) == bodo.stream_loop_sync_iters
        assert len(out_dfs[0]) == (batch_size - 1)
        assert len(out_dfs[-1]) == 1
        assert len(out_dfs[-1]["D"][0]) == str_len_p2
        for i in range(1, 98):
            assert len(out_dfs[i]) == 0

    # Run all tests:
    # Set sync iterations explicitly since used in tests
    saved_sync_iters = bodo.stream_loop_sync_iters
    try:
        bodo.stream_loop_sync_iters = bodo.default_stream_loop_sync_iters
        test1()
        test2()
        test3()
        test4()
        test5()
    finally:
        bodo.stream_loop_sync_iters = saved_sync_iters


@pytest.mark.slow
@pytest.mark.parametrize(
    "build_outer",
    [
        pytest.param(False, id="inner"),
        pytest.param(True, id="outer"),
    ],
)
def test_prune_na(build_outer, memory_leak_check):
    """Test that NA values in keys are properly
    removed. This is useful for verifying NA filter
    changes/optimizations.
    """
    df1 = pd.DataFrame(
        {
            "A": pd.array([2, None] * 6, dtype="Int64"),
            "B": pd.array([1, None, 2] * 4, dtype="Int64"),
        }
    )
    df2 = pd.DataFrame(
        {
            "C": pd.array([None, 2] * 15, dtype="Int64"),
            "D": pd.array([2, None, 2] * 10, dtype="Int64"),
        }
    )
    build_keys_inds = bodo.utils.typing.MetaType((0, 1))
    probe_keys_inds = bodo.utils.typing.MetaType((0, 1))
    kept_cols = bodo.utils.typing.MetaType((0, 1))
    build_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
        )
    )
    probe_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
        )
    )
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
            "A",
            "B",
        )
    )

    def test_hash_join(df1, df2):
        join_state = init_join_state(
            -1,
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            build_outer,
            False,
        )
        _temp1 = 0
        is_last1 = False
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), kept_cols, 2
        )
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_temp1 * 4000), ((_temp1 + 1) * 4000))
            )
            is_last1 = (_temp1 * 4000) >= len(df1)
            _temp1 = _temp1 + 1
            is_last1 = join_build_consume_batch(join_state, T2, is_last1)

        _temp2 = 0
        is_last2 = False
        is_last3 = False
        T3 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2), (), kept_cols, 2
        )
        _table_builder = bodo.libs.table_builder.init_table_builder_state(-1)
        while not is_last3:
            T4 = bodo.hiframes.table.table_local_filter(
                T3, slice((_temp2 * 4000), ((_temp2 + 1) * 4000))
            )
            is_last2 = (_temp2 * 4000) >= len(df2)
            _temp2 = _temp2 + 1
            out_table, is_last3, _ = join_probe_consume_batch(
                join_state, T4, is_last2, True
            )
            bodo.libs.table_builder.table_builder_append(_table_builder, out_table)
        delete_join_state(join_state)
        out_table = bodo.libs.table_builder.table_builder_finalize(_table_builder)
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        return df

    # Generate expected output for each type of join
    left_outer = pd.DataFrame(
        {
            "C": pd.array([None] * 10, dtype="Int64"),
            "D": pd.array([None] * 10, dtype="Int64"),
            "A": pd.array(
                [2, None, None, 2, None, 2, None, None, 2, None], dtype="Int64"
            ),
            "B": pd.array([1, None, 1, None, 2, 1, None, 1, None, 2], dtype="Int64"),
        }
    )
    inner = pd.DataFrame(
        {
            "C": pd.array([2] * 20, dtype="Int64"),
            "D": pd.array([2] * 20, dtype="Int64"),
            "A": pd.array([2] * 20, dtype="Int64"),
            "B": pd.array([2] * 20, dtype="Int64"),
        }
    )
    # Fuse the outputs and cast.
    if build_outer:
        expected_df = pd.concat([left_outer, inner])
    else:
        expected_df = inner

    check_func(
        test_hash_join,
        (df1, df2),
        py_output=expected_df,
        reset_index=True,
        sort_output=True,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "build_dist",
    [
        pytest.param(False, id="probe_dist"),
        pytest.param(True, id="build_dist"),
    ],
)
def test_outer_join_na_one_dist(build_dist, broadcast, memory_leak_check):
    """Test streaming hash join with build_outer_table=True
    where only build or probe table is distributed but not both. This is
    used to check NA filtering works as expected with outer join"""
    df1 = pd.DataFrame(
        {
            "A": pd.array([1, 2, None, 3, None] * 10000, dtype="Int64"),
            "B": pd.array([-1] * 50000, dtype="Int64"),
        }
    )
    df2 = pd.DataFrame(
        {
            "C": pd.array([2, -1, 3, 5, 8], dtype="Int64"),
            "D": pd.array([2, -1, 3, 5, 8], dtype="Int64"),
        }
    )

    if build_dist:
        df1 = _get_dist_arg(df1)
    else:
        df2 = _get_dist_arg(df2)

    build_keys_inds = bodo.utils.typing.MetaType((0,))
    probe_keys_inds = bodo.utils.typing.MetaType((0,))
    kept_cols = bodo.utils.typing.MetaType((0, 1))
    build_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
        )
    )
    probe_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
        )
    )
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
            "A",
            "B",
        )
    )

    dist_df = "df1" if build_dist else "df2"

    @bodo.jit(distributed=[dist_df])
    def test_hash_join(df1, df2):
        join_state = init_join_state(
            -1,
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            True,
            False,
        )
        _temp1 = 0
        is_last1 = False
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), kept_cols, 2
        )
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_temp1 * 4000), ((_temp1 + 1) * 4000))
            )
            is_last1 = (_temp1 * 4000) >= len(df1)
            _temp1 = _temp1 + 1
            is_last1 = join_build_consume_batch(join_state, T2, is_last1)

        _temp2 = 0
        is_last2 = False
        is_last3 = False
        T3 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2), (), kept_cols, 2
        )
        _table_builder = bodo.libs.table_builder.init_table_builder_state(-1)
        while not is_last3:
            T4 = bodo.hiframes.table.table_local_filter(
                T3, slice((_temp2 * 4000), ((_temp2 + 1) * 4000))
            )
            is_last2 = (_temp2 * 4000) >= len(df2)
            _temp2 = _temp2 + 1
            out_table, is_last3, _ = join_probe_consume_batch(
                join_state, T4, is_last2, True
            )
            bodo.libs.table_builder.table_builder_append(_table_builder, out_table)
        delete_join_state(join_state)
        out_table = bodo.libs.table_builder.table_builder_finalize(_table_builder)
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        return df

    # Generate expected output
    build_outer_df = pd.DataFrame(
        {
            "C": pd.array([None] * 30000, dtype="Int64"),
            "D": pd.array([None] * 30000, dtype="Int64"),
            "A": pd.array([1, None, None] * 10000, dtype="Int64"),
            "B": pd.array([-1] * 30000, dtype="Int64"),
        }
    )
    inner = pd.DataFrame(
        {
            "C": pd.array([2, 3] * 10000, dtype="Int64"),
            "D": pd.array([2, 3] * 10000, dtype="Int64"),
            "A": pd.array([2, 3] * 10000, dtype="Int64"),
            "B": pd.array([-1] * 20000, dtype="Int64"),
        }
    )
    # Fuse the outputs and cast.
    expected_df = pd.concat([build_outer_df, inner])

    with set_broadcast_join(broadcast):
        bodo_output = test_hash_join(df1, df2)

    out_df = bodo.allgatherv(bodo_output)
    _test_equal(
        out_df,
        expected_df,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.slow
@pytest.mark.parametrize("side", ["build", "probe"])
@pytest.mark.parametrize("insert_loc", ["middle", "last"])
def test_hash_join_empty_table(side, insert_loc, broadcast, memory_leak_check):
    """Test hash join with empty table inserted in the middle or at the end of all batches,
    on the build or probe side and with or without broadcast"""
    df1 = pd.DataFrame(
        {
            "A": np.array(list(range(2500)) * 5),
            "B": np.array(list(range(2500)) * 5),
        }
    )
    df2 = pd.DataFrame(
        {
            "C": pd.array(list(range(2499, 4999)) * 5),
            "D": np.array(list(range(2500, 5000)) * 5),
        }
    )
    build_keys_inds = bodo.utils.typing.MetaType((0,))
    probe_keys_inds = bodo.utils.typing.MetaType((0,))
    kept_cols = bodo.utils.typing.MetaType((0, 1))
    build_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
        )
    )
    probe_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
        )
    )
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
            "A",
            "B",
        )
    )

    def test_hash_join(df1, df2):
        join_state = init_join_state(
            -1,
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            False,
            False,
        )
        _temp1 = 0
        build_idx = 0
        insert_idx = 0
        is_last1 = False
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), kept_cols, 2
        )
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_temp1 * 4000), ((_temp1 + 1) * 4000))
            )
            is_last1 = (_temp1 * 4000) >= len(df1)

            if build_idx == 1 and insert_loc == "middle" and side == "build":
                T2 = bodo.hiframes.table.table_local_filter(
                    T1, slice((_temp1 * 4000), ((_temp1) * 4000))
                )
            else:
                _temp1 += 1

            if (
                is_last1
                and insert_loc == "last"
                and side == "build"
                and build_idx != insert_idx
            ):
                # Don't mark this batch as last so that an empty batch is inserted next iteration
                is_last1 = False
                insert_idx = build_idx + 1

            build_idx += 1

            is_last1 = join_build_consume_batch(join_state, T2, is_last1)

        _temp2 = 0
        is_last2 = False
        is_last3 = False
        probe_idx = 0
        insert_idx = 0
        T3 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2), (), kept_cols, 2
        )
        _table_builder = bodo.libs.table_builder.init_table_builder_state(-1)
        while not is_last3:
            T4 = bodo.hiframes.table.table_local_filter(
                T3, slice((_temp2 * 4000), ((_temp2 + 1) * 4000))
            )
            is_last2 = (_temp2 * 4000) >= len(df2)

            if probe_idx == 1 and insert_loc == "middle" and side == "probe":
                T4 = bodo.hiframes.table.table_local_filter(
                    T3, slice((_temp2 * 4000), ((_temp2) * 4000))
                )
            else:
                _temp2 += 1

            if (
                is_last2
                and insert_loc == "last"
                and side == "probe"
                and probe_idx != insert_idx
            ):
                # Don't mark this batch as last so that an empty batch is inserted next iteration
                is_last1 = False
                insert_idx = probe_idx + 1

            probe_idx += 1

            out_table, is_last3, _ = join_probe_consume_batch(
                join_state, T4, is_last2, True
            )
            bodo.libs.table_builder.table_builder_append(_table_builder, out_table)
        delete_join_state(join_state)
        out_table = bodo.libs.table_builder.table_builder_finalize(_table_builder)
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        return df

    expected_df = pd.DataFrame(
        {"C": [2499] * 25, "D": [2500] * 25, "A": [2499] * 25, "B": [2499] * 25}
    )

    with set_broadcast_join(broadcast):
        check_func(
            test_hash_join,
            (df1, df2),
            py_output=expected_df,
            reset_index=True,
            sort_output=True,
            check_dtype=False,
        )


@pytest.mark.slow
@pytest.mark.parametrize("side", ["build", "probe"])
@pytest.mark.parametrize("insert_loc", ["middle", "last"])
def test_nested_loop_join_empty_table(side, insert_loc, broadcast, memory_leak_check):
    """Test nested loop join with empty table inserted in the middle or at the end of all batches,
    on the build or probe side and with or without broadcast"""
    df1 = pd.DataFrame(
        {
            "A": np.array(list(range(2500)) * 5),
            "B": np.array(list(range(2500)) * 5),
        }
    )
    df2 = pd.DataFrame(
        {
            "C": pd.array(list(range(2499, 4999)) * 5),
            "D": np.array(list(range(2500, 5000)) * 5),
        }
    )
    build_keys_inds = bodo.utils.typing.MetaType(())
    probe_keys_inds = bodo.utils.typing.MetaType(())
    kept_cols = bodo.utils.typing.MetaType((0, 1))
    build_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
        )
    )
    probe_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
        )
    )
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
            "A",
            "B",
        )
    )
    non_equi_condition = "right.A >= left.C"

    def test_nested_loop_join(df1, df2):
        join_state = init_join_state(
            -1,
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            False,
            False,
            non_equi_condition=non_equi_condition,
        )
        _temp1 = 0
        build_idx = 0
        is_last1 = False
        insert_idx = 0
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), kept_cols, 2
        )
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_temp1 * 4000), ((_temp1 + 1) * 4000))
            )
            is_last1 = (_temp1 * 4000) >= len(df1)

            if build_idx == 1 and insert_loc == "middle" and side == "build":
                T2 = bodo.hiframes.table.table_local_filter(
                    T1, slice((_temp1 * 4000), ((_temp1) * 4000))
                )
            else:
                _temp1 += 1

            if (
                is_last1
                and insert_loc == "last"
                and side == "build"
                and build_idx != insert_idx
            ):
                # Don't mark this batch as last so that an empty batch is inserted next iteration
                is_last1 = False
                insert_idx = build_idx + 1

            build_idx += 1

            is_last1 = join_build_consume_batch(join_state, T2, is_last1)

        _temp2 = 0
        is_last2 = False
        is_last3 = False
        probe_idx = 0
        insert_idx = 0
        T3 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2), (), kept_cols, 2
        )
        _table_builder = bodo.libs.table_builder.init_table_builder_state(-1)
        while not is_last3:
            T4 = bodo.hiframes.table.table_local_filter(
                T3, slice((_temp2 * 4000), ((_temp2 + 1) * 4000))
            )
            is_last2 = (_temp2 * 4000) >= len(df2)

            if probe_idx == 1 and insert_loc == "middle" and side == "probe":
                T4 = bodo.hiframes.table.table_local_filter(
                    T3, slice((_temp2 * 4000), ((_temp2) * 4000))
                )
            else:
                _temp2 += 1

            if (
                is_last2
                and insert_loc == "last"
                and side == "probe"
                and probe_idx != insert_idx
            ):
                # Don't mark this batch as last so that an empty batch is inserted next iteration
                is_last1 = False
                insert_idx = probe_idx + 1

            probe_idx += 1

            out_table, is_last3, _ = join_probe_consume_batch(
                join_state, T4, is_last2, True
            )
            bodo.libs.table_builder.table_builder_append(_table_builder, out_table)
        delete_join_state(join_state)
        out_table = bodo.libs.table_builder.table_builder_finalize(_table_builder)
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        return df

    expected_df = pd.DataFrame(
        {"C": [2499] * 25, "D": [2500] * 25, "A": [2499] * 25, "B": [2499] * 25}
    )

    with set_broadcast_join(broadcast):
        check_func(
            test_nested_loop_join,
            (df1, df2),
            py_output=expected_df,
            reset_index=True,
            sort_output=True,
            check_dtype=False,
        )


@pytest.mark.slow
@pytest.mark.parametrize("df_size", [10, 1000], ids=["small", "large"])
def test_request_input(df_size, memory_leak_check):
    """
    Test that the request_input return value is true
    if the output buffer is smaller than two batches and false otherwise
    """
    df1 = pd.DataFrame(
        {
            "A": np.arange(df_size),
            "B": np.arange(df_size),
        }
    )
    df2 = pd.DataFrame(
        {
            "C": np.arange(df_size),
            "D": np.arange(df_size),
        }
    )
    build_keys_inds = bodo.utils.typing.MetaType(())
    probe_keys_inds = bodo.utils.typing.MetaType(())
    kept_cols = bodo.utils.typing.MetaType((0, 1))
    build_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
        )
    )
    probe_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
        )
    )
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
            "A",
            "B",
        )
    )

    @bodo.jit
    def test_request_input(df1, df2):
        join_state = init_join_state(
            -1,
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            True,
            True,
        )
        _temp1 = 0
        is_last1 = False
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), kept_cols, 2
        )
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_temp1 * 4000), ((_temp1 + 1) * 4000))
            )
            is_last1 = (_temp1 * 4000) >= len(df1)
            _temp1 += 1
            is_last1 = join_build_consume_batch(join_state, T2, is_last1)

        _temp2 = 0
        is_last2 = False
        is_last3 = False
        out_tables = []
        request_input = True
        always_requested_input = True
        T3 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2), (), kept_cols, 2
        )
        while not is_last3:
            is_last2 = (_temp2 * 4000) >= len(df2)
            T4 = bodo.hiframes.table.table_local_filter(
                T3, slice((_temp2 * 4000), ((_temp2 + 1) * 4000))
            )
            _temp2 += int(request_input)

            out_table, is_last3, request_input = join_probe_consume_batch(
                join_state, T4, is_last2, True
            )
            out_tables.append(out_table)
            always_requested_input &= request_input
        out_table = bodo.utils.table_utils.concat_tables(out_tables)
        delete_join_state(join_state)

        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        return always_requested_input, bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )

    always_requested_input, _ = test_request_input(df1, df2)
    if 2 * (df_size**2) > bodo.bodosql_streaming_batch_size:
        assert not always_requested_input
    else:
        assert always_requested_input


@pytest.mark.slow
def test_produce_output(memory_leak_check):
    """
    Test that output len is 0 if the produce_output flag is False
    """
    df1 = pd.DataFrame(
        {
            "A": np.arange(1000),
            "B": np.arange(1000),
        }
    )
    df2 = pd.DataFrame(
        {
            "C": np.arange(1000),
            "D": np.arange(1000),
        }
    )
    build_keys_inds = bodo.utils.typing.MetaType(())
    probe_keys_inds = bodo.utils.typing.MetaType(())
    kept_cols = bodo.utils.typing.MetaType((0, 1))
    build_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
        )
    )
    probe_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
        )
    )
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
            "A",
            "B",
        )
    )

    @bodo.jit
    def test_request_input(df1, df2):
        join_state = init_join_state(
            -1,
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            True,
            True,
        )
        _temp1 = 0
        is_last1 = False
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), kept_cols, 2
        )
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_temp1 * 4000), ((_temp1 + 1) * 4000))
            )
            is_last1 = (_temp1 * 4000) >= len(df1)
            _temp1 += 1
            is_last1 = join_build_consume_batch(join_state, T2, is_last1)

        _temp2 = 0
        is_last2 = False
        is_last3 = False
        out_tables = []
        output_when_not_request_input = False
        T3 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2), (), kept_cols, 2
        )
        while not is_last3:
            is_last2 = (_temp2 * 4000) >= len(df2)
            T4 = bodo.hiframes.table.table_local_filter(
                T3, slice((_temp2 * 4000), ((_temp2 + 1) * 4000))
            )
            _temp2 += 1

            produce_output = _temp2 != 1
            out_table, is_last3, _ = join_probe_consume_batch(
                join_state, T4, is_last2, produce_output
            )
            if 1 == _temp2 and len(out_table) != 0:
                output_when_not_request_input = True
            out_tables.append(out_table)
        out_table = bodo.utils.table_utils.concat_tables(out_tables)
        delete_join_state(join_state)

        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        return (
            bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (out_table,), index_var, col_meta
            ),
            output_when_not_request_input,
        )

    # Ensure that the output is empty if produce_output is False
    assert not (test_request_input(df1, df2)[1])


def test_hash_join_nested_array(memory_leak_check):
    probe_table = pd.DataFrame(
        {
            "A": np.array([1, 2, 3, 4] * 5),
            "B": np.array([[[1, 2], [3]], [[None], [4]], [[5], [6]], None] * 5, object),
            "C": pd.array([1, 2, 3, 4] * 5),
            "D": np.array(["1", "2", "3", "4"] * 5),
        }
    )
    build_table = pd.DataFrame(
        {
            "E": np.array([1, 2, 3, 4] * 5),
            "F": np.array([[[1, 2], [3]], [[None], [4]], [[5], [6]], None] * 5, object),
            "G": np.array([[[9, 8], [7]], [[None], [6]], [[5], [4]], None] * 5, object),
            "H": pd.array([5, 6, 7, 8] * 5),
        }
    )
    build_keys_inds = bodo.utils.typing.MetaType((0,))
    probe_keys_inds = bodo.utils.typing.MetaType((0,))
    kept_cols = bodo.utils.typing.MetaType((0, 1, 2, 3))
    build_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
            "C",
            "D",
        )
    )
    probe_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "E",
            "F",
            "G",
            "H",
        )
    )
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
        )
    )

    def test_hash_join(df1, df2):
        join_state = init_join_state(
            -1,
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            False,
            False,
        )
        _temp1 = 0
        is_last1 = False
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), kept_cols, 4
        )
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_temp1 * 4000), ((_temp1 + 1) * 4000))
            )
            is_last1 = (_temp1 * 4000) >= len(df1)
            _temp1 = _temp1 + 1
            is_last1 = join_build_consume_batch(join_state, T2, is_last1)

        _temp2 = 0
        is_last2 = False
        is_last3 = False
        T3 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2), (), kept_cols, 4
        )
        _table_builder = bodo.libs.table_builder.init_table_builder_state(-1)

        while not is_last3:
            T4 = bodo.hiframes.table.table_local_filter(
                T3, slice((_temp2 * 4000), ((_temp2 + 1) * 4000))
            )
            is_last2 = (_temp2 * 4000) >= len(df2)
            _temp2 = _temp2 + 1
            out_table, is_last3, _ = join_probe_consume_batch(
                join_state, T4, is_last2, True
            )
            bodo.libs.table_builder.table_builder_append(_table_builder, out_table)

        delete_join_state(join_state)
        out_table = bodo.libs.table_builder.table_builder_finalize(_table_builder)
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        return df

    # Generate expected output for each type of join
    expected_df = pd.DataFrame(
        {
            "A": np.array([1, 2, 3, 4] * 25),
            "B": pd.array(
                [[[1, 2], [3]], [[None], [4]], [[5], [6]], None] * 25, object
            ),
            "C": pd.array([1, 2, 3, 4] * 25),
            "D": np.array(["1", "2", "3", "4"] * 25),
            "E": np.array([1, 2, 3, 4] * 25),
            "F": pd.array(
                [[[1, 2], [3]], [[None], [4]], [[5], [6]], None] * 25, object
            ),
            "G": pd.array(
                [[[9, 8], [7]], [[None], [6]], [[5], [4]], None] * 25, object
            ),
            "H": pd.array([5, 6, 7, 8] * 25),
        }
    )

    check_func(
        test_hash_join,
        (build_table, probe_table),
        py_output=expected_df,
        reset_index=True,
        convert_columns_to_pandas=True,
        sort_output=True,
    )


def test_nested_loop_join_nested_array(memory_leak_check):
    probe_table = pd.DataFrame(
        {
            "A": np.array([1, 2, 3, 4] * 5),
            "B": np.array([[[1, 2], [3]], [[None], [4]], [[5], [6]], None] * 5, object),
        }
    )
    build_table = pd.DataFrame(
        {
            "C": np.array([2, 3, 4, 5] * 5),
            "D": np.array([[[9, 8], [7]], [[None], [6]], [[5], [4]], None] * 5, object),
        }
    )
    build_keys_inds = bodo.utils.typing.MetaType(())
    probe_keys_inds = bodo.utils.typing.MetaType(())
    kept_cols = bodo.utils.typing.MetaType((0, 1))
    build_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
        )
    )
    probe_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
        )
    )
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
            "C",
            "D",
        )
    )

    def test_nested_loop_join(df1, df2):
        join_state = init_join_state(
            -1,
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            False,
            False,
            non_equi_condition="right.A <= left.C",
        )
        _temp1 = 0
        is_last1 = False
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), kept_cols, 2
        )
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_temp1 * 4000), ((_temp1 + 1) * 4000))
            )
            is_last1 = (_temp1 * 4000) >= len(df1)
            _temp1 = _temp1 + 1
            is_last1 = join_build_consume_batch(join_state, T2, is_last1)

        _temp2 = 0
        is_last2 = False
        is_last3 = False
        T3 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2), (), kept_cols, 2
        )
        _table_builder = bodo.libs.table_builder.init_table_builder_state(-1)

        while not is_last3:
            T4 = bodo.hiframes.table.table_local_filter(
                T3, slice((_temp2 * 4000), ((_temp2 + 1) * 4000))
            )
            is_last2 = (_temp2 * 4000) >= len(df2)
            _temp2 = _temp2 + 1
            out_table, is_last3, _ = join_probe_consume_batch(
                join_state, T4, is_last2, True
            )
            bodo.libs.table_builder.table_builder_append(_table_builder, out_table)

        delete_join_state(join_state)
        out_table = bodo.libs.table_builder.table_builder_finalize(_table_builder)
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        return df

    # Generate expected output for each type of join
    expected_df = pd.DataFrame(
        {
            "A": np.array([2, 3, 4, 3, 4, 4] * 25),
            "B": pd.array(
                [[[None], [4]], [[5], [6]], None, [[5], [6]], None, None] * 25, object
            ),
            "C": np.array([2, 2, 2, 3, 3, 4] * 25),
            "D": pd.array(
                [
                    [[9, 8], [7]],
                    [[9, 8], [7]],
                    [[9, 8], [7]],
                    [[None], [6]],
                    [[None], [6]],
                    [[5], [4]],
                ]
                * 25,
                object,
            ),
        }
    )

    check_func(
        test_nested_loop_join,
        (build_table, probe_table),
        py_output=expected_df,
        reset_index=True,
        convert_columns_to_pandas=True,
        sort_output=True,
    )


def test_hash_join_struct_array(memory_leak_check):
    probe_table = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3, 4, 5, 6] * 5),
            "B": [
                {
                    "X": "AB",
                    "Y": [1.1, 2.2],
                    "Z": [[1], None, [3, None]],
                    "W": {"A": 1, "B": "A"},
                },
                {
                    "X": "C",
                    "Y": [1.1],
                    "Z": [[11], None],
                    "W": {"A": 1, "B": "ABC"},
                },
                None,
                {
                    "X": "D",
                    "Y": [4.0, 6.0],
                    "Z": [[1], None],
                    "W": {"A": 1, "B": ""},
                },
                {
                    "X": "VFD",
                    "Y": [1.2],
                    "Z": [[], [3, 1]],
                    "W": {"A": 1, "B": "AA"},
                },
                {
                    "X": "LMMM",
                    "Y": [9.0, 1.2, 3.1],
                    "Z": [[10, 11], [11, 0, -3, -5]],
                    "W": {"A": 1, "B": "DFG"},
                },
            ]
            * 5,
        }
    )
    build_table = pd.DataFrame(
        {
            "C": pd.array([1, 2, 3, 4, 5, 6] * 5),
            "D": [
                {
                    "X": "AB",
                    "Y": [1.1, 2.2],
                    "Z": [[1], None, [3, None]],
                    "W": {"A": 1, "B": "A"},
                },
                {
                    "X": "C",
                    "Y": [1.1],
                    "Z": [[11], None],
                    "W": {"A": 1, "B": "ABC"},
                },
                None,
                {
                    "X": "D",
                    "Y": [4.0, 6.0],
                    "Z": [[1], None],
                    "W": {"A": 1, "B": ""},
                },
                {
                    "X": "VFD",
                    "Y": [1.2],
                    "Z": [[], [3, 1]],
                    "W": {"A": 1, "B": "AA"},
                },
                {
                    "X": "LMMM",
                    "Y": [9.0, 1.2, 3.1],
                    "Z": [[10, 11], [11, 0, -3, -5]],
                    "W": {"A": 1, "B": "DFG"},
                },
            ]
            * 5,
        }
    )
    build_keys_inds = bodo.utils.typing.MetaType((0,))
    probe_keys_inds = bodo.utils.typing.MetaType((0,))
    kept_cols = bodo.utils.typing.MetaType((0, 1))
    build_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
        )
    )
    probe_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
        )
    )
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
            "C",
            "D",
        )
    )

    def test_hash_join(df1, df2):
        join_state = init_join_state(
            -1,
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            False,
            False,
        )
        _temp1 = 0
        is_last1 = False
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), kept_cols, 2
        )
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_temp1 * 4000), ((_temp1 + 1) * 4000))
            )
            is_last1 = (_temp1 * 4000) >= len(df1)
            _temp1 = _temp1 + 1
            is_last1 = join_build_consume_batch(join_state, T2, is_last1)

        _temp2 = 0
        is_last2 = False
        is_last3 = False
        T3 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2), (), kept_cols, 2
        )
        _table_builder = bodo.libs.table_builder.init_table_builder_state(-1)

        while not is_last3:
            T4 = bodo.hiframes.table.table_local_filter(
                T3, slice((_temp2 * 4000), ((_temp2 + 1) * 4000))
            )
            is_last2 = (_temp2 * 4000) >= len(df2)
            _temp2 = _temp2 + 1
            out_table, is_last3, _ = join_probe_consume_batch(
                join_state, T4, is_last2, True
            )
            bodo.libs.table_builder.table_builder_append(_table_builder, out_table)

        delete_join_state(join_state)
        out_table = bodo.libs.table_builder.table_builder_finalize(_table_builder)
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        return df

    # Generate expected output for each type of join
    expected_df = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3, 4, 5, 6] * 25),
            "B": [
                {
                    "X": "AB",
                    "Y": [1.1, 2.2],
                    "Z": [[1], None, [3, None]],
                    "W": {"A": 1, "B": "A"},
                },
                {
                    "X": "C",
                    "Y": [1.1],
                    "Z": [[11], None],
                    "W": {"A": 1, "B": "ABC"},
                },
                None,
                {
                    "X": "D",
                    "Y": [4.0, 6.0],
                    "Z": [[1], None],
                    "W": {"A": 1, "B": ""},
                },
                {
                    "X": "VFD",
                    "Y": [1.2],
                    "Z": [[], [3, 1]],
                    "W": {"A": 1, "B": "AA"},
                },
                {
                    "X": "LMMM",
                    "Y": [9.0, 1.2, 3.1],
                    "Z": [[10, 11], [11, 0, -3, -5]],
                    "W": {"A": 1, "B": "DFG"},
                },
            ]
            * 25,
            "C": pd.array([1, 2, 3, 4, 5, 6] * 25),
            "D": [
                {
                    "X": "AB",
                    "Y": [1.1, 2.2],
                    "Z": [[1], None, [3, None]],
                    "W": {"A": 1, "B": "A"},
                },
                {
                    "X": "C",
                    "Y": [1.1],
                    "Z": [[11], None],
                    "W": {"A": 1, "B": "ABC"},
                },
                None,
                {
                    "X": "D",
                    "Y": [4.0, 6.0],
                    "Z": [[1], None],
                    "W": {"A": 1, "B": ""},
                },
                {
                    "X": "VFD",
                    "Y": [1.2],
                    "Z": [[], [3, 1]],
                    "W": {"A": 1, "B": "AA"},
                },
                {
                    "X": "LMMM",
                    "Y": [9.0, 1.2, 3.1],
                    "Z": [[10, 11], [11, 0, -3, -5]],
                    "W": {"A": 1, "B": "DFG"},
                },
            ]
            * 25,
        }
    )

    check_func(
        test_hash_join,
        (build_table, probe_table),
        py_output=expected_df,
        reset_index=True,
        convert_columns_to_pandas=True,
        sort_output=True,
    )


def test_nested_loop_join_struct_array(memory_leak_check):
    probe_table = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3] * 5),
            "B": [
                {
                    "X": "D",
                    "Y": [4.0, 6.0],
                    "Z": [[1], None],
                    "W": {"A": 1, "B": ""},
                },
                {
                    "X": "VFD",
                    "Y": [1.2],
                    "Z": [[], [3, 1]],
                    "W": {"A": 1, "B": "AA"},
                },
                {
                    "X": "LMMM",
                    "Y": [9.0, 1.2, 3.1],
                    "Z": [[10, 11], [11, 0, -3, -5]],
                    "W": {"A": 1, "B": "DFG"},
                },
            ]
            * 5,
        }
    )
    build_table = pd.DataFrame(
        {
            "C": pd.array([1, 2, 3] * 5),
            "D": [
                {
                    "X": "AB",
                    "Y": [1.1, 2.2],
                    "Z": [[1], None, [3, None]],
                    "W": {"A": 1, "B": "A"},
                },
                {
                    "X": "C",
                    "Y": [1.1],
                    "Z": [[11], None],
                    "W": {"A": 1, "B": "ABC"},
                },
                {
                    "X": "D",
                    "Y": [4.0, 6.0],
                    "Z": [[1], None],
                    "W": {"A": 1, "B": ""},
                },
            ]
            * 5,
        }
    )
    build_keys_inds = bodo.utils.typing.MetaType(())
    probe_keys_inds = bodo.utils.typing.MetaType(())
    kept_cols = bodo.utils.typing.MetaType((0, 1))
    build_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
        )
    )
    probe_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
        )
    )
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
            "C",
            "D",
        )
    )

    def test_nested_loop_join(df1, df2):
        join_state = init_join_state(
            -1,
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            False,
            False,
            non_equi_condition="right.A > left.C",
        )
        _temp1 = 0
        is_last1 = False
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), kept_cols, 2
        )
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_temp1 * 4000), ((_temp1 + 1) * 4000))
            )
            is_last1 = (_temp1 * 4000) >= len(df1)
            _temp1 = _temp1 + 1
            is_last1 = join_build_consume_batch(join_state, T2, is_last1)

        _temp2 = 0
        is_last2 = False
        is_last3 = False
        T3 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2), (), kept_cols, 2
        )
        _table_builder = bodo.libs.table_builder.init_table_builder_state(-1)

        while not is_last3:
            T4 = bodo.hiframes.table.table_local_filter(
                T3, slice((_temp2 * 4000), ((_temp2 + 1) * 4000))
            )
            is_last2 = (_temp2 * 4000) >= len(df2)
            _temp2 = _temp2 + 1
            out_table, is_last3, _ = join_probe_consume_batch(
                join_state, T4, is_last2, True
            )
            bodo.libs.table_builder.table_builder_append(_table_builder, out_table)

        delete_join_state(join_state)
        out_table = bodo.libs.table_builder.table_builder_finalize(_table_builder)
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        return df

    # Generate expected output for each type of join
    expected_df = pd.DataFrame(
        {
            "A": pd.array([1, 1, 2] * 25),
            "B": [
                {
                    "X": "D",
                    "Y": [4.0, 6.0],
                    "Z": [[1], None],
                    "W": {"A": 1, "B": ""},
                },
                {
                    "X": "D",
                    "Y": [4.0, 6.0],
                    "Z": [[1], None],
                    "W": {"A": 1, "B": ""},
                },
                {
                    "X": "VFD",
                    "Y": [1.2],
                    "Z": [[], [3, 1]],
                    "W": {"A": 1, "B": "AA"},
                },
            ]
            * 25,
            "C": pd.array([2, 3, 3] * 25),
            "D": [
                {
                    "X": "C",
                    "Y": [1.1],
                    "Z": [[11], None],
                    "W": {"A": 1, "B": "ABC"},
                },
                {
                    "X": "D",
                    "Y": [4.0, 6.0],
                    "Z": [[1], None],
                    "W": {"A": 1, "B": ""},
                },
                {
                    "X": "D",
                    "Y": [4.0, 6.0],
                    "Z": [[1], None],
                    "W": {"A": 1, "B": ""},
                },
            ]
            * 25,
        }
    )

    check_func(
        test_nested_loop_join,
        (build_table, probe_table),
        py_output=expected_df,
        reset_index=True,
        convert_columns_to_pandas=True,
        sort_output=True,
    )


def test_hash_join_map_array(memory_leak_check):
    probe_table = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3, 4] * 5),
            "B": pd.Series(
                [
                    {1: 1.4, 2: 3.1},
                    None,
                    {7: -1.2},
                    {11: 3.4, 21: 3.1, 9: 8.1},
                ]
                * 5,
                dtype=pd.ArrowDtype(pa.map_(pa.int64(), pa.float64())),
            ),
        }
    )
    build_table = pd.DataFrame(
        {
            "C": pd.array([1, 2, 3, 4] * 5),
            "D": pd.Series(
                [
                    {4: 9.4, 6: 4.1},
                    {7: -1.2},
                    {},
                    {8: 3.3, 5: 6.3},
                ]
                * 5,
                dtype=pd.ArrowDtype(pa.map_(pa.int64(), pa.float64())),
            ),
        }
    )
    build_keys_inds = bodo.utils.typing.MetaType((0,))
    probe_keys_inds = bodo.utils.typing.MetaType((0,))
    kept_cols = bodo.utils.typing.MetaType((0, 1))
    build_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
        )
    )
    probe_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
        )
    )
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
            "C",
            "D",
        )
    )

    def test_hash_join(df1, df2):
        join_state = init_join_state(
            -1,
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            False,
            False,
        )
        _temp1 = 0
        is_last1 = False
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), kept_cols, 2
        )
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_temp1 * 4000), ((_temp1 + 1) * 4000))
            )
            is_last1 = (_temp1 * 4000) >= len(df1)
            _temp1 = _temp1 + 1
            is_last1 = join_build_consume_batch(join_state, T2, is_last1)

        _temp2 = 0
        is_last2 = False
        is_last3 = False
        T3 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2), (), kept_cols, 2
        )
        _table_builder = bodo.libs.table_builder.init_table_builder_state(-1)

        while not is_last3:
            T4 = bodo.hiframes.table.table_local_filter(
                T3, slice((_temp2 * 4000), ((_temp2 + 1) * 4000))
            )
            is_last2 = (_temp2 * 4000) >= len(df2)
            _temp2 = _temp2 + 1
            out_table, is_last3, _ = join_probe_consume_batch(
                join_state, T4, is_last2, True
            )
            bodo.libs.table_builder.table_builder_append(_table_builder, out_table)

        delete_join_state(join_state)
        out_table = bodo.libs.table_builder.table_builder_finalize(_table_builder)
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        return df

    # Generate expected output for each type of join
    expected_df = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3, 4] * 25),
            "B": pd.Series(
                [
                    {1: 1.4, 2: 3.1},
                    None,
                    {7: -1.2},
                    {11: 3.4, 21: 3.1, 9: 8.1},
                ]
                * 25,
                dtype=pd.ArrowDtype(pa.map_(pa.int64(), pa.float64())),
            ),
            "C": pd.array([1, 2, 3, 4] * 25),
            "D": pd.Series(
                [
                    {4: 9.4, 6: 4.1},
                    {7: -1.2},
                    {},
                    {8: 3.3, 5: 6.3},
                ]
                * 25,
                dtype=pd.ArrowDtype(pa.map_(pa.int64(), pa.float64())),
            ),
        }
    )

    check_func(
        test_hash_join,
        (build_table, probe_table),
        py_output=expected_df,
        reset_index=True,
        convert_columns_to_pandas=True,
        sort_output=True,
    )


def test_nested_loop_join_map_array(memory_leak_check):
    probe_table = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3] * 5),
            "B": pd.Series(
                [
                    {1: 1.4, 2: 3.1},
                    None,
                    {7: -1.2},
                ]
                * 5,
                dtype=pd.ArrowDtype(pa.map_(pa.int64(), pa.float64())),
            ),
        }
    )
    build_table = pd.DataFrame(
        {
            "C": pd.array([1, 2, 3] * 5),
            "D": pd.Series(
                [
                    {4: 9.4, 6: 4.1},
                    {7: -1.2},
                    {},
                ]
                * 5,
                dtype=pd.ArrowDtype(pa.map_(pa.int64(), pa.float64())),
            ),
        }
    )
    build_keys_inds = bodo.utils.typing.MetaType(())
    probe_keys_inds = bodo.utils.typing.MetaType(())
    kept_cols = bodo.utils.typing.MetaType((0, 1))
    build_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
        )
    )
    probe_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
        )
    )
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
            "C",
            "D",
        )
    )

    def test_nested_loop_join(df1, df2):
        join_state = init_join_state(
            -1,
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            False,
            False,
            non_equi_condition="right.A > left.C",
        )
        _temp1 = 0
        is_last1 = False
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), kept_cols, 2
        )
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_temp1 * 4000), ((_temp1 + 1) * 4000))
            )
            is_last1 = (_temp1 * 4000) >= len(df1)
            _temp1 = _temp1 + 1
            is_last1 = join_build_consume_batch(join_state, T2, is_last1)

        _temp2 = 0
        is_last2 = False
        is_last3 = False
        T3 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2), (), kept_cols, 2
        )
        _table_builder = bodo.libs.table_builder.init_table_builder_state(-1)

        while not is_last3:
            T4 = bodo.hiframes.table.table_local_filter(
                T3, slice((_temp2 * 4000), ((_temp2 + 1) * 4000))
            )
            is_last2 = (_temp2 * 4000) >= len(df2)
            _temp2 = _temp2 + 1
            out_table, is_last3, _ = join_probe_consume_batch(
                join_state, T4, is_last2, True
            )
            bodo.libs.table_builder.table_builder_append(_table_builder, out_table)

        delete_join_state(join_state)
        out_table = bodo.libs.table_builder.table_builder_finalize(_table_builder)
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        return df

    # Generate expected output for each type of join
    expected_df = pd.DataFrame(
        {
            "A": pd.array([1, 1, 2] * 25),
            "B": pd.Series(
                [
                    {1: 1.4, 2: 3.1},
                    {1: 1.4, 2: 3.1},
                    None,
                ]
                * 25,
                dtype=pd.ArrowDtype(pa.map_(pa.int64(), pa.float64())),
            ),
            "C": pd.array([2, 3, 3] * 25),
            "D": pd.Series(
                [
                    {7: -1.2},
                    {},
                    {},
                ]
                * 25,
                dtype=pd.ArrowDtype(pa.map_(pa.int64(), pa.float64())),
            ),
        }
    )

    check_func(
        test_nested_loop_join,
        (build_table, probe_table),
        py_output=expected_df,
        reset_index=True,
        convert_columns_to_pandas=True,
        sort_output=True,
    )


@pytest.mark.skip("TODO[BSE-2076]: Support tuple array in Arrow boxing/unboxing")
def test_hash_join_tuple_array(memory_leak_check):
    probe_table = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5, 6],
            "B": [
                (2, 3.1),
                None,
                (-1, 7.8),
                (3, 4.0),
                (-3, -1.2),
                (None, 9.0),
            ],
        }
    )
    build_table = pd.DataFrame(
        {
            "C": [1, 2, 3, 4, 5, 6],
            "D": [
                (-1.1, -1.1),
                (2.1, np.nan),
                (3.1, 4.1),
                None,
                (-3.1, -1.1),
                (5.1, 9.1),
            ],
        }
    )
    build_keys_inds = bodo.utils.typing.MetaType((0,))
    probe_keys_inds = bodo.utils.typing.MetaType((0,))
    kept_cols = bodo.utils.typing.MetaType((0, 1))
    build_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
        )
    )
    probe_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
        )
    )
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
            "C",
            "D",
        )
    )

    def test_hash_join(df1, df2):
        join_state = init_join_state(
            -1,
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            False,
            False,
        )
        _temp1 = 0
        is_last1 = False
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), kept_cols, 2
        )
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_temp1 * 4000), ((_temp1 + 1) * 4000))
            )
            is_last1 = (_temp1 * 4000) >= len(df1)
            _temp1 = _temp1 + 1
            is_last1 = join_build_consume_batch(join_state, T2, is_last1)

        _temp2 = 0
        is_last2 = False
        is_last3 = False
        T3 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2), (), kept_cols, 2
        )
        _table_builder = bodo.libs.table_builder.init_table_builder_state(-1)

        while not is_last3:
            T4 = bodo.hiframes.table.table_local_filter(
                T3, slice((_temp2 * 4000), ((_temp2 + 1) * 4000))
            )
            is_last2 = (_temp2 * 4000) >= len(df2)
            _temp2 = _temp2 + 1
            out_table, is_last3, _ = join_probe_consume_batch(
                join_state, T4, is_last2, True
            )
            bodo.libs.table_builder.table_builder_append(_table_builder, out_table)

        delete_join_state(join_state)
        out_table = bodo.libs.table_builder.table_builder_finalize(_table_builder)
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        return df

    # Generate expected output for each type of join
    expected_df = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3, 4, 5, 6]),
            "B": np.array(
                [
                    (2, 3.1),
                    None,
                    (-1, 7.8),
                    (3, 4.0),
                    (-3, -1.2),
                    (None, 9.0),
                ],
                dtype=object,
            ),
            "C": pd.array([1, 2, 3, 4, 5, 6]),
            "D": np.array(
                [
                    (-1.1, -1.1),
                    (2.1, None),
                    (3.1, 4.1),
                    None,
                    (-3.1, -1.1),
                    (5.1, 9.1),
                ],
                dtype=object,
            ),
        }
    )

    check_func(
        test_hash_join,
        (build_table, probe_table),
        py_output=expected_df,
        reset_index=True,
        convert_columns_to_pandas=True,
        sort_output=True,
    )


@pytest.mark.skip("TODO[BSE-2076]: Support tuple array in Arrow boxing/unboxing")
def test_nested_loop_join_tuple_array(memory_leak_check):
    probe_table = pd.DataFrame(
        {
            "A": [1, 2, 3] * 5,
            "B": [
                (2, 3.1),
                (-3, -1.1),
                (4, 9.0),
            ]
            * 5,
        }
    )
    build_table = pd.DataFrame(
        {
            "C": [1, 2, 3] * 5,
            "D": [
                (-3.1, -1.1),
                (-3.4, 2.2),
                (2.1, 1.5),
            ]
            * 5,
        }
    )
    build_keys_inds = bodo.utils.typing.MetaType(())
    probe_keys_inds = bodo.utils.typing.MetaType(())
    kept_cols = bodo.utils.typing.MetaType((0, 1))
    build_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
        )
    )
    probe_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
        )
    )
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
            "C",
            "D",
        )
    )

    def test_nested_loop_join(df1, df2):
        join_state = init_join_state(
            -1,
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            False,
            False,
            non_equi_condition="right.A > left.C",
        )
        _temp1 = 0
        is_last1 = False
        tt = bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1)
        T1 = bodo.hiframes.table.logical_table_to_table(tt, (), kept_cols, 2)
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_temp1 * 4000), ((_temp1 + 1) * 4000))
            )
            is_last1 = (_temp1 * 4000) >= len(df1)
            _temp1 = _temp1 + 1
            is_last1 = join_build_consume_batch(join_state, T2, is_last1)

        _temp2 = 0
        is_last2 = False
        is_last3 = False
        T3 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2), (), kept_cols, 2
        )
        _table_builder = bodo.libs.table_builder.init_table_builder_state(-1)

        while not is_last3:
            T4 = bodo.hiframes.table.table_local_filter(
                T3, slice((_temp2 * 4000), ((_temp2 + 1) * 4000))
            )
            is_last2 = (_temp2 * 4000) >= len(df2)
            _temp2 = _temp2 + 1
            out_table, is_last3, _ = join_probe_consume_batch(
                join_state, T4, is_last2, True
            )
            bodo.libs.table_builder.table_builder_append(_table_builder, out_table)

        delete_join_state(join_state)
        out_table = bodo.libs.table_builder.table_builder_finalize(_table_builder)
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        return df

    # Generate expected output for each type of join

    expected_df = pd.DataFrame(
        {
            "A": pd.array([1, 1, 2] * 25),
            "B": [
                (2, 3.1),
                (2, 3.1),
                (-3, -1.1),
            ]
            * 25,
            "C": pd.array([2, 3, 3] * 25),
            "D": [
                (-3.4, 2.2),
                (2.1, 1.5),
                (2.1, 1.5),
            ]
            * 25,
        }
    )

    check_func(
        test_nested_loop_join,
        (build_table, probe_table),
        py_output=expected_df,
        reset_index=True,
        convert_columns_to_pandas=True,
        sort_output=True,
    )


# Note we mark this as slow because the volume of data in
# the output makes checking correctness slow.
@pytest.mark.slow
def test_shuffle_batching(memory_leak_check):
    assert (
        bodo.bodosql_streaming_batch_size < 10000
    ), "If the batch size is more than 10k there will only be one batch and this won't test anything"
    build = pd.DataFrame({"A": np.arange(60000), "B": [1, 2, 3, 4, 5, 6] * 10000})
    probe = pd.DataFrame({"C": np.arange(60000), "D": [1, 2, 3, 4, 5, 6] * 10000})
    expected_df = pd.DataFrame(
        {
            "A": np.arange(60000),
            "B": [1, 2, 3, 4, 5, 6] * 10000,
            "C": np.arange(60000),
            "D": [1, 2, 3, 4, 5, 6] * 10000,
        }
    )
    prev_batch_size = bodo.bodosql_streaming_batch_size
    build_keys_inds = bodo.utils.typing.MetaType((0,))
    probe_keys_inds = bodo.utils.typing.MetaType((0,))
    kept_cols = bodo.utils.typing.MetaType((0, 1))
    build_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
        )
    )
    probe_col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "C",
            "D",
        )
    )
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
            "C",
            "D",
        )
    )

    def shuffle_batching(df1, df2):
        join_state = init_join_state(
            -1,
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            False,
            False,
        )
        _temp1 = 0
        is_last1 = False
        tt = bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1)
        T1 = bodo.hiframes.table.logical_table_to_table(tt, (), kept_cols, 2)
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_temp1 * 4000), ((_temp1 + 1) * 4000))
            )
            is_last1 = (_temp1 * 4000) >= len(df1)
            _temp1 = _temp1 + 1
            is_last1 = join_build_consume_batch(join_state, T2, is_last1)

        _temp2 = 0
        is_last2 = False
        is_last3 = False
        T3 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2),
            (),
            kept_cols,
            2,
        )
        _table_builder = bodo.libs.table_builder.init_table_builder_state(-1)

        while not is_last3:
            T4 = bodo.hiframes.table.table_local_filter(
                T3, slice((_temp2 * 4000), ((_temp2 + 1) * 4000))
            )
            is_last2 = (_temp2 * 4000) >= len(df2)
            _temp2 = _temp2 + 1
            out_table, is_last3, _ = join_probe_consume_batch(
                join_state, T4, is_last2, True
            )
            bodo.libs.table_builder.table_builder_append(_table_builder, out_table)

        delete_join_state(join_state)
        out_table = bodo.libs.table_builder.table_builder_finalize(_table_builder)
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        return df

    check_func(
        shuffle_batching,
        (build, probe),
        py_output=expected_df,
        reset_index=True,
        sort_output=True,
    )
