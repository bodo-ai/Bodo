import numpy as np
import pandas as pd
import pytest

import bodo
import bodo.io.snowflake
import bodo.tests.utils
from bodo.io.arrow_reader import read_arrow_next
from bodo.libs.stream_join import (
    delete_join_state,
    init_join_state,
    join_build_consume_batch,
    join_probe_consume_batch,
)
from bodo.tests.utils import _test_equal, pytest_mark_snowflake


@pytest_mark_snowflake
@pytest.mark.parametrize(
    "build_outer,probe_outer,expected_df",
    [
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
    ],
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

    # TODO[BSE-439]: Support dict-encoded strings
    saved_SF_READ_AUTO_DICT_ENCODE_ENABLED = (
        bodo.io.snowflake.SF_READ_AUTO_DICT_ENCODE_ENABLED
    )
    try:
        bodo.io.snowflake.SF_READ_AUTO_DICT_ENCODE_ENABLED = False
        out_df = test_hash_join(conn_str)
    finally:
        bodo.io.snowflake.SF_READ_AUTO_DICT_ENCODE_ENABLED = (
            saved_SF_READ_AUTO_DICT_ENCODE_ENABLED
        )
    out_df = bodo.allgatherv(out_df)
    _test_equal(
        out_df,
        expected_df,
        check_dtype=False,
        sort_output=True,
        reset_index=True,
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

    # TODO[BSE-439]: Support dict-encoded strings
    saved_SF_READ_AUTO_DICT_ENCODE_ENABLED = (
        bodo.io.snowflake.SF_READ_AUTO_DICT_ENCODE_ENABLED
    )
    try:
        bodo.io.snowflake.SF_READ_AUTO_DICT_ENCODE_ENABLED = False
        out_df = test_hash_join(conn_str)
    finally:
        bodo.io.snowflake.SF_READ_AUTO_DICT_ENCODE_ENABLED = (
            saved_SF_READ_AUTO_DICT_ENCODE_ENABLED
        )
    expected_df = pd.DataFrame(
        {
            "p_partkey": [183728],
            "p_comment": ["bold deposi"],
            "p_name": ["yellow turquoise cornflower coral saddle"],
            "p_size": [37],
            "l_partkey": [183728],
            "l_comment": ["bold deposi"],
            "l_orderkey": [685476],
        }
    )
    out_df = bodo.allgatherv(out_df)
    _test_equal(
        out_df,
        expected_df,
        check_dtype=False,
    )
