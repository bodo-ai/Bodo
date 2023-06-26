import math
import random
import string

import numpy as np
import pandas as pd
import pytest

import bodo
import bodo.io.snowflake
import bodo.tests.utils
from bodo.io.arrow_reader import read_arrow_next
from bodo.libs.memory import BufferPool
from bodo.libs.stream_join import (
    delete_join_state,
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
    # select P_PARTKEY, P_COMMENT, P_NAME, P_SIZE, L_PARTKEY, L_COMMENT, L_ORDERKEY
    # from lineitem inner join part on P_PARTKEY = L_PARTKEY and P_COMMENT = L_COMMENT
    # where l_orderkey > {l_orderkey_start} and l_orderkey < {l_orderkey_end}
    # and p_size > {p_size_limit}
    (
        False,
        False,
        pd.DataFrame(
            {
                "p_partkey": [183728],
                "p_comment": ["bold deposi"],
                "p_name": ["yellow turquoise cornflower coral saddle"],
                "p_size": [37],
                "l_partkey": [183728],
                "l_comment": ["bold deposi"],
                "l_orderkey": [685476],
            }
        ),
    ),
    # Equivalent query:
    # select P_PARTKEY, P_COMMENT, P_NAME, P_SIZE, L_PARTKEY, L_COMMENT, L_ORDERKEY
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
                "l_partkey": [183728, pd.NA, pd.NA],
                "l_comment": [
                    "bold deposi",
                    pd.NA,
                    pd.NA,
                ],
                "l_orderkey": [685476, pd.NA, pd.NA],
            }
        ),
    ),
    # Equivalent query:
    # select P_PARTKEY, P_COMMENT, P_NAME, P_SIZE, L_PARTKEY, L_COMMENT, L_ORDERKEY
    # from lineitem left outer join (
    #     select P_PARTKEY, P_COMMENT, P_NAME, P_SIZE from part where p_size > {p_size_limit}
    # ) on P_PARTKEY = L_PARTKEY and P_COMMENT = L_COMMENT
    # where l_orderkey > {l_orderkey_start} and l_orderkey < {l_orderkey_end}
    (
        False,
        True,
        pd.DataFrame(
            {
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
            }
        ),
    ),
    # Equivalent query:
    # select P_PARTKEY, P_COMMENT, P_NAME, P_SIZE, L_PARTKEY, L_COMMENT, L_ORDERKEY
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
            "p_partkey",
            "p_comment",
            "p_name",
            "p_size",
            "l_partkey",
            "l_comment",
            "l_orderkey",
        )
    )

    @bodo.jit
    def test_hash_join(conn):
        join_state = init_join_state(
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            build_outer,
            probe_outer,
        )

        # read PART table and build join hash table
        reader1 = pd.read_sql(
            f"SELECT P_PARTKEY, P_COMMENT, P_NAME, P_SIZE FROM PART where P_SIZE = {p_size_limit} and p_partkey > {p_orderkey_start} and p_partkey < {p_orderkey_end}",
            conn,
            _bodo_chunksize=4000,
        )
        while True:
            table1, is_last1 = read_arrow_next(reader1)
            is_last1 = bodo.libs.distributed_api.dist_reduce(
                is_last1,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            join_build_consume_batch(join_state, table1, is_last1)
            if is_last1:
                break

        # read LINEITEM table and probe
        reader2 = pd.read_sql(
            f"SELECT L_PARTKEY, L_COMMENT, L_ORDERKEY FROM LINEITEM where l_orderkey > {l_orderkey_start} and l_orderkey < {l_orderkey_end}",
            conn,
            _bodo_chunksize=4000,
        )
        out_dfs = []
        while True:
            table2, is_last2 = read_arrow_next(reader2)
            is_last2 = bodo.libs.distributed_api.dist_reduce(
                is_last2,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            out_table, is_last3 = join_probe_consume_batch(join_state, table2, is_last2)
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(out_table), 1, None
            )
            df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (out_table,), index_var, col_meta
            )
            out_dfs.append(df)
            is_last3 = bodo.libs.distributed_api.dist_reduce(
                is_last3,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            if is_last3:
                break
        delete_join_state(join_state)
        return pd.concat(out_dfs)

    out_df = test_hash_join(conn_str)
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
                "p_partkey": [199978, 199995],
                "p_comment": ["ess, i", "packa"],
                "p_name": [
                    "linen magenta saddle slate turquoise",
                    "blanched floral red maroon papaya",
                ],
                "p_size": [50, 50],
                "l_partkey": [2132, 2132],
                "l_comment": ["lites. fluffily even de", "lites. fluffily even de"],
                "l_orderkey": [1, 1],
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
                "l_partkey": [2132, 2132, pd.NA, pd.NA, pd.NA],
                "l_comment": [
                    "lites. fluffily even de",
                    "lites. fluffily even de",
                    pd.NA,
                    pd.NA,
                    pd.NA,
                ],
                "l_orderkey": [1, 1, pd.NA, pd.NA, pd.NA],
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
                "p_partkey": [199978, 199995, pd.NA],
                "p_comment": ["ess, i", "packa", pd.NA],
                "p_name": [
                    "linen magenta saddle slate turquoise",
                    "blanched floral red maroon papaya",
                    pd.NA,
                ],
                "p_size": [50, 50, pd.NA],
                "l_partkey": [2132, 2132, 15635],
                "l_comment": [
                    "lites. fluffily even de",
                    "lites. fluffily even de",
                    "arefully slyly ex",
                ],
                "l_orderkey": [1, 1, 1],
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
            "p_partkey",
            "p_comment",
            "p_name",
            "p_size",
            "l_partkey",
            "l_comment",
            "l_orderkey",
        )
    )
    non_equi_condition = "(left.p_partkey) > ((right.l_partkey) + 197800)"

    # select P_PARTKEY, P_COMMENT, P_NAME, P_SIZE, L_PARTKEY, L_COMMENT, L_ORDERKEY
    # from lineitem inner join part on P_PARTKEY > L_PARTKEY + 197800 where l_orderkey < 2 and p_size > 49

    @bodo.jit
    def test_nested_loop_join(conn):
        join_state = init_join_state(
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
            table1, is_last1 = read_arrow_next(reader1)
            is_last1 = bodo.libs.distributed_api.dist_reduce(
                is_last1,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            join_build_consume_batch(join_state, table1, is_last1)
            if is_last1:
                break

        # read LINEITEM table and probe
        reader2 = pd.read_sql(
            f"SELECT L_PARTKEY, L_COMMENT, L_ORDERKEY FROM LINEITEM where l_orderkey < {l_orderkey_end} and l_partkey < {l_partkey_end}",
            conn,
            _bodo_chunksize=4000,
        )
        out_dfs = []
        while True:
            table2, is_last2 = read_arrow_next(reader2)
            is_last2 = bodo.libs.distributed_api.dist_reduce(
                is_last2,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            out_table, is_last3 = join_probe_consume_batch(join_state, table2, is_last2)
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(out_table), 1, None
            )
            df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (out_table,), index_var, col_meta
            )
            out_dfs.append(df)
            is_last3 = bodo.libs.distributed_api.dist_reduce(
                is_last3,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            if is_last3:
                break
        delete_join_state(join_state)
        return pd.concat(out_dfs)

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
            "p_partkey",
            "p_comment",
            "p_name",
            "p_size",
            "l_partkey",
            "l_comment",
            "l_orderkey",
        )
    )
    non_equi_condition = "(left.p_partkey) > ((right.l_partkey) + 197800)"

    # select P_PARTKEY, P_COMMENT, P_NAME, P_SIZE, L_PARTKEY, L_COMMENT, L_ORDERKEY
    # from lineitem inner join part on P_PARTKEY > L_PARTKEY + 197800 where l_orderkey < 2 and p_size > 49

    expected_df = pd.DataFrame(
        {
            "p_partkey": [199978, 199995],
            "p_comment": ["ess, i", "packa"],
            "p_name": [
                "linen magenta saddle slate turquoise",
                "blanched floral red maroon papaya",
            ],
            "p_size": [50, 50],
            "l_partkey": [2132, 2132],
            "l_comment": ["lites. fluffily even de", "lites. fluffily even de"],
            "l_orderkey": [1, 1],
        }
    )

    @bodo.jit
    def test_nested_loop_join(conn):
        join_state = init_join_state(
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
            table1, is_last1 = read_arrow_next(reader1)
            is_last1 = bodo.libs.distributed_api.dist_reduce(
                is_last1,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            join_build_consume_batch(join_state, table1, is_last1)
            if is_last1:
                break

        # read LINEITEM table and probe
        reader2 = pd.read_sql(
            f"SELECT L_PARTKEY, L_COMMENT, L_ORDERKEY FROM LINEITEM where l_orderkey < {l_orderkey_end} and l_partkey < {l_partkey_end}",
            conn,
            _bodo_chunksize=4000,
        )
        out_dfs = []
        while True:
            table2, is_last2 = read_arrow_next(reader2)
            is_last2 = bodo.libs.distributed_api.dist_reduce(
                is_last2,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            out_table, is_last3 = join_probe_consume_batch(join_state, table2, is_last2)
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(out_table), 1, None
            )
            df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (out_table,), index_var, col_meta
            )
            out_dfs.append(df)
            is_last3 = bodo.libs.distributed_api.dist_reduce(
                is_last3,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            if is_last3:
                break
        delete_join_state(join_state)
        return pd.concat(out_dfs)

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
            "p_name",
            "p_comment",
            "p_size",
            "p_partkey",
            "l_partkey",
            "l_orderkey",
            "l_comment",
        )
    )

    @bodo.jit
    def test_hash_join(conn):
        join_state = init_join_state(
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
            table1, is_last1 = read_arrow_next(reader1)
            is_last1 = bodo.libs.distributed_api.dist_reduce(
                is_last1,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            join_build_consume_batch(join_state, table1, is_last1)
            if is_last1:
                break

        # read LINEITEM table and probe
        reader2 = pd.read_sql(
            f"SELECT L_PARTKEY, L_ORDERKEY, L_COMMENT FROM LINEITEM where l_orderkey > {l_orderkey_start} and l_orderkey < {l_orderkey_end}",
            conn,
            _bodo_chunksize=4000,
        )
        out_dfs = []
        while True:
            table2, is_last2 = read_arrow_next(reader2)
            is_last2 = bodo.libs.distributed_api.dist_reduce(
                is_last2,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            out_table, is_last3 = join_probe_consume_batch(join_state, table2, is_last2)
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(out_table), 1, None
            )
            df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (out_table,), index_var, col_meta
            )
            out_dfs.append(df)
            is_last3 = bodo.libs.distributed_api.dist_reduce(
                is_last3,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            if is_last3:
                break
        delete_join_state(join_state)
        return pd.concat(out_dfs)

    out_df = test_hash_join(conn_str)
    expected_df = pd.DataFrame(
        {
            "p_name": ["yellow turquoise cornflower coral saddle"],
            "p_comment": ["bold deposi"],
            "p_size": [37],
            "p_partkey": [183728],
            "l_partkey": [183728],
            "l_orderkey": [685476],
            "l_comment": ["bold deposi"],
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
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "build_outer,probe_outer",
    [
        pytest.param(True, True, id="full-outer"),
        pytest.param(True, False, id="left"),
        pytest.param(True, False, id="right"),
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
            "A": [1, 2, 3, 4, 5] * 2500,
            "B": np.array([1, 2, 3, 4, 5] * 2500, dtype=np.int32),
        }
    )
    df2 = pd.DataFrame(
        {
            "C": [2, 6] * 2500,
            "D": np.array([2, 6] * 2500, dtype=np.int8),
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
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            build_outer,
            probe_outer,
        )
        _temp1 = 0
        is_last1 = False
        while not is_last1:
            batch1 = df1.iloc[(_temp1 * 4000) : ((_temp1 + 1) * 4000)]
            is_last1 = (_temp1 * 4000) >= len(df1)
            _temp1 = _temp1 + 1
            is_last1 = bodo.libs.distributed_api.dist_reduce(
                is_last1,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            table1 = bodo.hiframes.table.logical_table_to_table(
                bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(batch1),
                (),
                kept_cols,
                2,
            )
            join_build_consume_batch(join_state, table1, is_last1)

        _temp2 = 0
        out_dfs = []
        is_last2 = False
        is_last3 = False
        while not is_last3:
            batch2 = df2.iloc[(_temp2 * 4000) : ((_temp2 + 1) * 4000)]
            is_last2 = (_temp2 * 4000) >= len(df2)
            _temp2 = _temp2 + 1
            is_last2 = bodo.libs.distributed_api.dist_reduce(
                is_last2,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            table2 = bodo.hiframes.table.logical_table_to_table(
                bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(batch2),
                (),
                kept_cols,
                2,
            )
            out_table, is_last3 = join_probe_consume_batch(join_state, table2, is_last2)
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(out_table), 1, None
            )
            df_final = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (out_table,), index_var, col_meta
            )
            out_dfs.append(df_final)
            is_last3 = bodo.libs.distributed_api.dist_reduce(
                is_last3,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
        delete_join_state(join_state)
        return pd.concat(out_dfs)

    # Generate expected output for each type of join
    left_outer = pd.DataFrame(
        {
            "A": [1, 3, 4, 5] * 2500,
            "B": np.array([1, 3, 4, 5] * 2500, dtype=np.int32),
            "C": pd.array([None] * 10000, dtype="Int64"),
            "D": pd.array([None] * 10000, dtype="Int8"),
        }
    )
    right_outer = pd.DataFrame(
        {
            "A": pd.array([None] * 2500, dtype="Int64"),
            "B": pd.array([None] * 2500, dtype="Int32"),
            "C": [6] * 2500,
            "D": pd.array([6] * 2500, dtype="int8"),
        }
    )
    inner = pd.DataFrame(
        {
            "A": [2] * 2500 * 2500,
            "B": np.array([2] * 2500 * 2500, dtype=np.int32),
            "C": [2] * 2500 * 2500,
            "D": np.array([2] * 2500 * 2500, dtype=np.int8),
        }
    )
    # Fuse the outputs and cast.
    if build_outer and probe_outer:
        left_outer = left_outer.astype({"A": "Int64", "B": "Int32"})
        right_outer = right_outer.astype({"C": "Int64", "D": "Int8"})
        inner = inner.astype({"A": "Int64", "B": "Int32", "C": "Int64", "D": "Int8"})
        expected_df = pd.concat([left_outer, right_outer, inner])
    elif build_outer:
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
        pytest.param(True, id="right", marks=pytest.mark.slow),
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
            "A",
            "B",
            "C",
            "D",
        )
    )

    def test_hash_join(df1, df2):
        join_state = init_join_state(
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            False,
            probe_outer,
        )
        _temp1 = 0
        is_last1 = False
        while not is_last1:
            batch1 = df1.iloc[(_temp1 * 4000) : ((_temp1 + 1) * 4000)]
            is_last1 = (_temp1 * 4000) >= len(df1)
            _temp1 = _temp1 + 1
            is_last1 = bodo.libs.distributed_api.dist_reduce(
                is_last1,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            table1 = bodo.hiframes.table.logical_table_to_table(
                bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(batch1),
                (),
                kept_cols,
                2,
            )
            join_build_consume_batch(join_state, table1, is_last1)

        _temp2 = 0
        out_dfs = []
        is_last2 = False
        is_last3 = False
        while not is_last3:
            batch2 = df2.iloc[(_temp2 * 4000) : ((_temp2 + 1) * 4000)]
            is_last2 = (_temp2 * 4000) >= len(df2)
            _temp2 = _temp2 + 1
            is_last2 = bodo.libs.distributed_api.dist_reduce(
                is_last2,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            table2 = bodo.hiframes.table.logical_table_to_table(
                bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(batch2),
                (),
                kept_cols,
                2,
            )
            out_table, is_last3 = join_probe_consume_batch(join_state, table2, is_last2)
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(out_table), 1, None
            )
            df_final = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (out_table,), index_var, col_meta
            )
            out_dfs.append(df_final)
            is_last3 = bodo.libs.distributed_api.dist_reduce(
                is_last3,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
        delete_join_state(join_state)
        return pd.concat(out_dfs)

    # Generate expected output for each type of join
    right_outer = pd.DataFrame(
        {
            "A": pd.array([None] * 5, dtype="Int16"),
            "B": pd.array([None] * 5, dtype="Int32"),
            "C": pd.array([-1] * 5, dtype="Int16"),
            "D": pd.array([-1] * 5, dtype="int32"),
        }
    )
    inner = pd.DataFrame(
        {
            "A": pd.array([2] * 25, dtype="Int16"),
            "B": np.array([2] * 25, dtype=np.int32),
            "C": pd.array([2] * 25, dtype="Int16"),
            "D": np.array([2] * 25, dtype=np.int32),
        }
    )
    # Fuse the outputs and cast.
    if probe_outer:
        inner = inner.astype({"B": "Int32"})
        expected_df = pd.concat([right_outer, inner])
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
        pytest.param(True, False, id="left", marks=pytest.mark.slow),
        pytest.param(False, True, id="right", marks=pytest.mark.slow),
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
            "A",
            "B",
            "C",
            "D",
        )
    )
    non_equi_cond = "(left.`A` < right.`D`)"

    def test_hash_join(df1, df2):
        join_state = init_join_state(
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
        while not is_last1:
            batch1 = df1.iloc[(_temp1 * 4000) : ((_temp1 + 1) * 4000)]
            is_last1 = (_temp1 * 4000) >= len(df1)
            _temp1 = _temp1 + 1
            is_last1 = bodo.libs.distributed_api.dist_reduce(
                is_last1,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            table1 = bodo.hiframes.table.logical_table_to_table(
                bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(batch1),
                (),
                kept_cols,
                2,
            )
            join_build_consume_batch(join_state, table1, is_last1)

        _temp2 = 0
        out_dfs = []
        is_last2 = False
        is_last3 = False
        while not is_last3:
            batch2 = df2.iloc[(_temp2 * 4000) : ((_temp2 + 1) * 4000)]
            is_last2 = (_temp2 * 4000) >= len(df2)
            _temp2 = _temp2 + 1
            is_last2 = bodo.libs.distributed_api.dist_reduce(
                is_last2,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            table2 = bodo.hiframes.table.logical_table_to_table(
                bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(batch2),
                (),
                kept_cols,
                2,
            )
            out_table, is_last3 = join_probe_consume_batch(join_state, table2, is_last2)
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(out_table), 1, None
            )
            df_final = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (out_table,), index_var, col_meta
            )
            out_dfs.append(df_final)
            is_last3 = bodo.libs.distributed_api.dist_reduce(
                is_last3,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
        delete_join_state(join_state)
        return pd.concat(out_dfs)

    # Generate expected output for each type of join
    left_missed_keys = list(set(range(2500)) - {2}) * 5
    build_outer_df = pd.DataFrame(
        {
            "A": pd.array(left_missed_keys, dtype="Int16"),
            "B": pd.array(left_missed_keys, dtype="int32"),
            "C": pd.array([None] * len(left_missed_keys), dtype="Int16"),
            "D": pd.array([None] * len(left_missed_keys), dtype="Int8"),
        }
    )
    probe_outer_df = pd.DataFrame(
        {
            "A": pd.array([None] * 5, dtype="Int16"),
            "B": pd.array([None] * 5, dtype="Int32"),
            "C": pd.array([4] * 5, dtype="Int16"),
            "D": pd.array([2] * 5, dtype="int8"),
        }
    )
    inner = pd.DataFrame(
        {
            "A": pd.array([2] * 25, dtype="Int16"),
            "B": np.array([2] * 25, dtype=np.int32),
            "C": pd.array([2] * 25, dtype="Int16"),
            "D": np.array([4] * 25, dtype=np.int8),
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
            "A",
            "B",
            "C",
            "D",
        )
    )

    def test_hash_join(df1, df2):
        join_state = init_join_state(
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            False,
            False,
        )
        _temp1 = 0
        is_last1 = False
        while not is_last1:
            batch1 = df1.iloc[(_temp1 * 4000) : ((_temp1 + 1) * 4000)]
            is_last1 = (_temp1 * 4000) >= len(df1)
            _temp1 = _temp1 + 1
            is_last1 = bodo.libs.distributed_api.dist_reduce(
                is_last1,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            table1 = bodo.hiframes.table.logical_table_to_table(
                bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(batch1),
                (),
                kept_cols,
                2,
            )
            join_build_consume_batch(join_state, table1, is_last1)

        _temp2 = 0
        out_dfs = []
        is_last2 = False
        is_last3 = False
        while not is_last3:
            batch2 = df2.iloc[(_temp2 * 4000) : ((_temp2 + 1) * 4000)]
            is_last2 = (_temp2 * 4000) >= len(df2)
            _temp2 = _temp2 + 1
            is_last2 = bodo.libs.distributed_api.dist_reduce(
                is_last2,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            table2 = bodo.hiframes.table.logical_table_to_table(
                bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(batch2),
                (),
                kept_cols,
                2,
            )
            out_table, is_last3 = join_probe_consume_batch(join_state, table2, is_last2)
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(out_table), 1, None
            )
            df_join = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (out_table,), index_var, col_meta
            )
            df_final = df_join[["B", "D"]]
            out_dfs.append(df_final)
            is_last3 = bodo.libs.distributed_api.dist_reduce(
                is_last3,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
        delete_join_state(join_state)
        return pd.concat(out_dfs)

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
            "A",
            "C",
            "D",
        )
    )

    def test_hash_join(df1, df2):
        join_state = init_join_state(
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            False,
            False,
        )
        _temp1 = 0
        is_last1 = False
        while not is_last1:
            batch1 = df1.iloc[(_temp1 * 4000) : ((_temp1 + 1) * 4000)]
            is_last1 = (_temp1 * 4000) >= len(df1)
            _temp1 = _temp1 + 1
            is_last1 = bodo.libs.distributed_api.dist_reduce(
                is_last1,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            table1 = bodo.hiframes.table.logical_table_to_table(
                bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(batch1),
                (),
                kept_cols1,
                2,
            )
            join_build_consume_batch(join_state, table1, is_last1)

        _temp2 = 0
        out_dfs = []
        is_last2 = False
        is_last3 = False
        while not is_last3:
            batch2 = df2.iloc[(_temp2 * 4000) : ((_temp2 + 1) * 4000)]
            is_last2 = (_temp2 * 4000) >= len(df2)
            _temp2 = _temp2 + 1
            is_last2 = bodo.libs.distributed_api.dist_reduce(
                is_last2,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            table2 = bodo.hiframes.table.logical_table_to_table(
                bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(batch2),
                (),
                kept_cols2,
                2,
            )
            out_table, is_last3 = join_probe_consume_batch(join_state, table2, is_last2)
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(out_table), 1, None
            )
            df_final = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (out_table,), index_var, col_meta
            )
            out_dfs.append(df_final)
            is_last3 = bodo.libs.distributed_api.dist_reduce(
                is_last3,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
        delete_join_state(join_state)
        return pd.concat(out_dfs)

    # Everything should be upcast to int64.
    expected_df = pd.DataFrame(
        {
            "A": [2] * 2500 * 2500,
            "C": [2] * 2500 * 2500,
            "D": [2] * 2500 * 2500,
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
        pytest.param(True, False, False, id="left-probe_dist", marks=pytest.mark.slow),
        pytest.param(True, False, True, id="left-build_dist", marks=pytest.mark.slow),
        pytest.param(False, True, False, id="right-probe_dist", marks=pytest.mark.slow),
        pytest.param(False, True, True, id="right-build_dist", marks=pytest.mark.slow),
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
            "A",
            "B",
            "C",
            "D",
        )
    )

    dist_df = "df1" if build_dist else "df2"

    @bodo.jit(distributed=[dist_df])
    def test_hash_join(df1, df2):
        join_state = init_join_state(
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            build_outer,
            probe_outer,
        )
        _temp1 = 0
        is_last1 = False
        while not is_last1:
            batch1 = df1.iloc[(_temp1 * 4000) : ((_temp1 + 1) * 4000)]
            is_last1 = (_temp1 * 4000) >= len(df1)
            _temp1 = _temp1 + 1
            is_last1 = bodo.libs.distributed_api.dist_reduce(
                is_last1,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            table1 = bodo.hiframes.table.logical_table_to_table(
                bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(batch1),
                (),
                kept_cols,
                2,
            )
            join_build_consume_batch(join_state, table1, is_last1)

        _temp2 = 0
        out_dfs = []
        is_last2 = False
        is_last3 = False
        while not is_last3:
            batch2 = df2.iloc[(_temp2 * 4000) : ((_temp2 + 1) * 4000)]
            is_last2 = (_temp2 * 4000) >= len(df2)
            _temp2 = _temp2 + 1
            is_last2 = bodo.libs.distributed_api.dist_reduce(
                is_last2,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            table2 = bodo.hiframes.table.logical_table_to_table(
                bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(batch2),
                (),
                kept_cols,
                2,
            )
            out_table, is_last3 = join_probe_consume_batch(join_state, table2, is_last2)
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(out_table), 1, None
            )
            df_final = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (out_table,), index_var, col_meta
            )
            out_dfs.append(df_final)
            is_last3 = bodo.libs.distributed_api.dist_reduce(
                is_last3,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
        delete_join_state(join_state)
        return pd.concat(out_dfs)

    # Generate expected output for each type of join
    left_missed_keys = list(set(range(2500)) - {2}) * 5
    build_outer_df = pd.DataFrame(
        {
            "A": pd.array(left_missed_keys, dtype="Int64"),
            "B": pd.array(left_missed_keys, dtype="Int64"),
            "C": pd.array([None] * len(left_missed_keys), dtype="Int64"),
            "D": pd.array([None] * len(left_missed_keys), dtype="Int64"),
        }
    )
    probe_outer_df = pd.DataFrame(
        {
            "A": pd.array([None] * 5, dtype="Int64"),
            "B": pd.array([None] * 5, dtype="Int32"),
            "C": pd.array([-1] * 5, dtype="Int64"),
            "D": pd.array([-1] * 5, dtype="Int64"),
        }
    )
    inner = pd.DataFrame(
        {
            "A": pd.array([2] * 25, dtype="Int64"),
            "B": pd.array([2] * 25, dtype="Int64"),
            "C": pd.array([2] * 25, dtype="Int64"),
            "D": pd.array([2] * 25, dtype="Int64"),
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
            "A",
            "B",
            "C",
            "D",
        )
    )

    @bodo.jit(distributed=["df1", "df2"])
    def test_hash_join_impl(df1, df2, batch_size):
        join_state = init_join_state(
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            False,
            False,
        )
        _temp1 = 0
        is_last1 = False
        while not is_last1:
            batch1 = df1.iloc[(_temp1 * batch_size) : ((_temp1 + 1) * batch_size)]
            is_last1 = (_temp1 * batch_size) >= len(df1)
            # We don't need the Allreduce call for is_last1 since the test only runs on 1 rank.
            _temp1 = _temp1 + 1
            table1 = bodo.hiframes.table.logical_table_to_table(
                bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(batch1),
                (),
                kept_cols,
                2,
            )
            join_build_consume_batch(join_state, table1, is_last1)

        _temp2 = 0
        out_dfs = []
        is_last2 = False
        is_last3 = False
        while not is_last3:
            batch2 = df2.iloc[(_temp2 * batch_size) : ((_temp2 + 1) * batch_size)]
            is_last2 = ((_temp2 + 1) * batch_size) >= len(df2)
            # We don't need the Allreduce call for is_last2 since the test only runs on 1 rank.
            _temp2 = _temp2 + 1
            table2 = bodo.hiframes.table.logical_table_to_table(
                bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(batch2),
                (),
                kept_cols,
                2,
            )
            out_table, is_last3 = join_probe_consume_batch(join_state, table2, is_last2)
            # We don't need the Allreduce call for is_last3 since the test only runs on 1 rank.
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
        assert len(out_dfs) == 1
        assert len(out_dfs[0]) == batch_size
        assert (out_dfs[0]["D"].str.len() == pd.Series([str_len] * batch_size)).all()

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
        assert len(out_dfs) == exp_out_chunks
        assert len(out_dfs[0]) == (max_buffer_size // str_len)
        assert (
            out_dfs[0]["D"].str.len()
            == pd.Series([str_len] * (max_buffer_size // str_len))
        ).all()
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
        assert len(out_dfs) == 2
        assert len(out_dfs[0]) == 1
        assert len(out_dfs[1]) == (batch_size - 1)
        assert len(out_dfs[0]["D"][0]) == max_buffer_size + 10

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
        assert len(out_dfs) == 1
        assert len(out_dfs[0]) == batch_size
        assert len(out_dfs[0]["D"][batch_size - 1]) == str_len_p2

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
        assert len(out_dfs) == 2
        assert len(out_dfs[0]) == (batch_size - 1)
        assert len(out_dfs[1]) == 1
        assert len(out_dfs[1]["D"][0]) == str_len_p2

    # Run all tests:
    test1()
    test2()
    test3()
    test4()
    test5()


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
            "A",
            "B",
            "C",
            "D",
        )
    )

    def test_hash_join(df1, df2):
        join_state = init_join_state(
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            build_outer,
            False,
        )
        _temp1 = 0
        is_last1 = False
        while not is_last1:
            batch1 = df1.iloc[(_temp1 * 4000) : ((_temp1 + 1) * 4000)]
            is_last1 = (_temp1 * 4000) >= len(df1)
            _temp1 = _temp1 + 1
            is_last1 = bodo.libs.distributed_api.dist_reduce(
                is_last1,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            table1 = bodo.hiframes.table.logical_table_to_table(
                bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(batch1),
                (),
                kept_cols,
                2,
            )
            join_build_consume_batch(join_state, table1, is_last1)

        _temp2 = 0
        out_dfs = []
        is_last2 = False
        is_last3 = False
        while not is_last3:
            batch2 = df2.iloc[(_temp2 * 4000) : ((_temp2 + 1) * 4000)]
            is_last2 = (_temp2 * 4000) >= len(df2)
            _temp2 = _temp2 + 1
            is_last2 = bodo.libs.distributed_api.dist_reduce(
                is_last2,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            table2 = bodo.hiframes.table.logical_table_to_table(
                bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(batch2),
                (),
                kept_cols,
                2,
            )
            out_table, is_last3 = join_probe_consume_batch(join_state, table2, is_last2)
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(out_table), 1, None
            )
            df_final = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (out_table,), index_var, col_meta
            )
            out_dfs.append(df_final)
            is_last3 = bodo.libs.distributed_api.dist_reduce(
                is_last3,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
        delete_join_state(join_state)
        return pd.concat(out_dfs)

    # Generate expected output for each type of join
    right_outer = pd.DataFrame(
        {
            "A": pd.array(
                [2, None, None, 2, None, 2, None, None, 2, None], dtype="Int64"
            ),
            "B": pd.array([1, None, 1, None, 2, 1, None, 1, None, 2], dtype="Int64"),
            "C": pd.array([None] * 10, dtype="Int64"),
            "D": pd.array([None] * 10, dtype="Int64"),
        }
    )
    inner = pd.DataFrame(
        {
            "A": pd.array([2] * 20, dtype="Int64"),
            "B": pd.array([2] * 20, dtype="Int64"),
            "C": pd.array([2] * 20, dtype="Int64"),
            "D": pd.array([2] * 20, dtype="Int64"),
        }
    )
    # Fuse the outputs and cast.
    if build_outer:
        expected_df = pd.concat([right_outer, inner])
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
            "A",
            "B",
            "C",
            "D",
        )
    )

    dist_df = "df1" if build_dist else "df2"

    @bodo.jit(distributed=[dist_df])
    def test_hash_join(df1, df2):
        join_state = init_join_state(
            build_keys_inds,
            probe_keys_inds,
            build_col_meta,
            probe_col_meta,
            True,
            False,
        )
        _temp1 = 0
        is_last1 = False
        while not is_last1:
            batch1 = df1.iloc[(_temp1 * 4000) : ((_temp1 + 1) * 4000)]
            is_last1 = (_temp1 * 4000) >= len(df1)
            _temp1 = _temp1 + 1
            is_last1 = bodo.libs.distributed_api.dist_reduce(
                is_last1,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            table1 = bodo.hiframes.table.logical_table_to_table(
                bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(batch1),
                (),
                kept_cols,
                2,
            )
            join_build_consume_batch(join_state, table1, is_last1)

        _temp2 = 0
        out_dfs = []
        is_last2 = False
        is_last3 = False
        while not is_last3:
            batch2 = df2.iloc[(_temp2 * 4000) : ((_temp2 + 1) * 4000)]
            is_last2 = (_temp2 * 4000) >= len(df2)
            _temp2 = _temp2 + 1
            is_last2 = bodo.libs.distributed_api.dist_reduce(
                is_last2,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            table2 = bodo.hiframes.table.logical_table_to_table(
                bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(batch2),
                (),
                kept_cols,
                2,
            )
            out_table, is_last3 = join_probe_consume_batch(join_state, table2, is_last2)
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(out_table), 1, None
            )
            df_final = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (out_table,), index_var, col_meta
            )
            out_dfs.append(df_final)
            is_last3 = bodo.libs.distributed_api.dist_reduce(
                is_last3,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
        delete_join_state(join_state)
        return pd.concat(out_dfs)

    # Generate expected output
    build_outer_df = pd.DataFrame(
        {
            "A": pd.array([1, None, None] * 10000, dtype="Int64"),
            "B": pd.array([-1] * 30000, dtype="Int64"),
            "C": pd.array([None] * 30000, dtype="Int64"),
            "D": pd.array([None] * 30000, dtype="Int64"),
        }
    )
    inner = pd.DataFrame(
        {
            "A": pd.array([2, 3] * 10000, dtype="Int64"),
            "B": pd.array([-1] * 20000, dtype="Int64"),
            "C": pd.array([2, 3] * 10000, dtype="Int64"),
            "D": pd.array([2, 3] * 10000, dtype="Int64"),
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
