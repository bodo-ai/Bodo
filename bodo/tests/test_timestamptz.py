import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import _get_dist_arg, check_func, pytest_mark_one_rank

if bodo.test_compiler:
    from bodo.utils.typing import ColNamesMetaType, MetaType


@pytest.mark.parametrize(
    "pattern",
    [
        pytest.param([4, 3, 2, 1, 0], id="small-no_null"),
        pytest.param([1, 11, 4, 11, 2, 11, 0, 11, 3], id="small-some_null"),
        pytest.param([11, 11, 8, 11, 11] * 3, id="small-most_null"),
        pytest.param((np.arange(1000) ** 3) % 12, id="large-some_null"),
        pytest.param(
            np.round(np.tan(np.arange(1000))).astype(np.int64) % 11,
            id="large-no_null",
        ),
        pytest.param(([11] * 400 + [3]) * 5, id="large-most_null"),
    ],
)
def test_timestamptz_sort(pattern, memory_leak_check):
    """Test that sorting an array of TIMESTAMP_TZ values returns
    them in the correct order"""

    def impl(df):
        return df.sort_values(by=["T"], na_position="last")

    base_values = np.array(
        [
            bodo.types.TimestampTZ.fromLocal("2021-01-02 14:00:00", 300),
            bodo.types.TimestampTZ.fromLocal("2021-01-02 12:30:00", 0),
            bodo.types.TimestampTZ.fromLocal("2021-01-02 06:45:00", -630),
            bodo.types.TimestampTZ.fromLocal("2021-03-14 00:00:00", -1),
            bodo.types.TimestampTZ.fromLocal("2024-01-02 00:00:00", 1),
            bodo.types.TimestampTZ.fromLocal("2024-03-14 16:30:00", 600),
            bodo.types.TimestampTZ.fromLocal("2024-03-14 12:00:00", 120),
            bodo.types.TimestampTZ.fromLocal("2024-03-14 12:00:00", 60),
            bodo.types.TimestampTZ.fromLocal("2024-03-14 12:00:00", 0),
            bodo.types.TimestampTZ.fromLocal("2024-03-14 12:00:00", -60),
            bodo.types.TimestampTZ.fromLocal("2024-03-14 12:00:00", -120),
            None,
        ]
    )
    arr = base_values[pattern]
    expected = base_values[sorted(pattern)]
    df = pd.DataFrame({"T": arr})
    refsol = pd.DataFrame({"T": expected})
    check_func(
        impl,
        (df,),
        py_output=refsol,
        check_dtype=False,
        check_names=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "pattern",
    [
        pytest.param([4, 3, 2, 1, 0], id="small-no_null"),
        pytest.param([1, 11, 4, 11, 2, 11, 0, 11, 3], id="small-some_null"),
        pytest.param([11, 11, 8, 11, 11] * 3, id="small-most_null"),
        pytest.param((np.arange(1000) ** 3) % 12, id="large-some_null"),
        pytest.param(
            np.round(np.tan(np.arange(1000))).astype(np.int64) % 11,
            id="large-no_null",
        ),
        pytest.param(([11] * 400 + [3]) * 5, id="large-most_null"),
    ],
)
def test_timestamptz_sort(pattern, memory_leak_check):
    """Test that sorting an array of TIMESTAMP_TZ values returns
    them in the correct order"""

    def impl(df):
        return df.sort_values(by=["T"], na_position="last")

    base_values = np.array(
        [
            bodo.types.TimestampTZ.fromLocal("2021-01-02 14:00:00", 300),
            bodo.types.TimestampTZ.fromLocal("2021-01-02 12:30:00", 0),
            bodo.types.TimestampTZ.fromLocal("2021-01-02 06:45:00", -630),
            bodo.types.TimestampTZ.fromLocal("2021-03-14 00:00:00", -1),
            bodo.types.TimestampTZ.fromLocal("2024-01-02 00:00:00", 1),
            bodo.types.TimestampTZ.fromLocal("2024-03-14 16:30:00", 600),
            bodo.types.TimestampTZ.fromLocal("2024-03-14 12:00:00", 120),
            bodo.types.TimestampTZ.fromLocal("2024-03-14 12:00:00", 60),
            bodo.types.TimestampTZ.fromLocal("2024-03-14 12:00:00", 0),
            bodo.types.TimestampTZ.fromLocal("2024-03-14 12:00:00", -60),
            bodo.types.TimestampTZ.fromLocal("2024-03-14 12:00:00", -120),
            None,
        ]
    )
    arr = base_values[pattern]
    expected = base_values[sorted(pattern)]
    df = pd.DataFrame({"T": arr})
    refsol = pd.DataFrame({"T": expected})
    check_func(
        impl,
        (df,),
        py_output=refsol,
        check_dtype=False,
        check_names=False,
        reset_index=True,
    )


def test_timestamptz_array_creation(memory_leak_check):
    """Test creation of TimestampTZ array"""

    def f():
        arr = bodo.hiframes.timestamptz_ext.alloc_timestamptz_array(5)
        arr[0] = bodo.types.TimestampTZ(pd.Timestamp("2021-01-02 03:04:05"), 100)
        arr[1] = bodo.types.TimestampTZ(pd.Timestamp("2022-12-31 12:59:59"), 200)
        arr[2] = bodo.types.TimestampTZ(pd.Timestamp("2024-01-01 00:00:00"), 300)
        arr[3] = None
        arr[4] = bodo.types.TimestampTZ(pd.Timestamp("2022-12-31 12:59:59"), 200)
        return arr

    expected = np.array(
        [
            bodo.types.TimestampTZ.fromUTC("2021-01-02 03:04:05", 100),
            bodo.types.TimestampTZ.fromUTC("2022-12-31 12:59:59", 200),
            bodo.types.TimestampTZ.fromUTC("2024-01-01 00:00:00", 300),
            None,
            bodo.types.TimestampTZ.fromUTC("2022-12-31 12:59:59", 200),
        ]
    )
    check_func(f, (), py_output=expected)


def test_timestamptz_boxing_unboxing(memory_leak_check):
    """Test boxing and unboxing of TimestampTZ scalar"""

    def f(v):
        return v

    v = bodo.types.TimestampTZ.fromUTC("2021-01-02 03:04:05", 100)
    check_func(f, (v,))


def test_timestamptz_array_boxing_unboxing(memory_leak_check):
    """Test boxing and unboxing of TimestampTZ array"""

    def f(arr):
        return arr

    arr = pd.Series(
        [
            bodo.types.TimestampTZ.fromUTC("2021-01-02 03:04:05", 100),
            bodo.types.TimestampTZ.fromUTC("2022-02-03 04:05:06", 200),
        ]
        * 5
    )
    check_func(f, (arr,))


# Tests for core distributed api operations


def test_bcast_scalar(memory_leak_check):
    """Test that a scalar is broadcasted correctly"""
    expected = bodo.types.TimestampTZ.fromUTC("2021-01-02 03:04:05", 100)
    if bodo.get_rank() == 0:
        x = expected
    else:
        x = bodo.types.TimestampTZ.fromUTC("2021-01-02 03:05:13", 100)
    result = bodo.libs.distributed_api.bcast_scalar(x)
    # Test the values exactly to ensure no conversion occurs.
    assert result.utc_timestamp == expected.utc_timestamp
    assert result.offset_minutes == expected.offset_minutes


def test_scatterv(memory_leak_check):
    """Test that scatterv works correctly on a given array"""
    arr = np.array(
        [
            bodo.types.TimestampTZ.fromUTC("2021-01-02 03:04:05", 100),
            bodo.types.TimestampTZ.fromUTC("2022-02-03 04:05:06", 200),
            bodo.types.TimestampTZ.fromUTC("2023-03-04 05:06:07", 300),
            bodo.types.TimestampTZ.fromUTC("2024-04-05 06:07:08", 400),
            None,
            bodo.types.TimestampTZ.fromUTC("2022-02-03 04:05:06", 200),
        ]
    )
    scattered_arr = bodo.libs.distributed_api.scatterv(arr)
    np.testing.assert_array_equal(
        scattered_arr, _get_dist_arg(arr, False, False, False)
    )


def test_gatherv(memory_leak_check):
    """Test that gatherv works correctly on a given array"""
    expected = np.array(
        [
            bodo.types.TimestampTZ.fromUTC("2021-01-02 03:04:05", 100),
            bodo.types.TimestampTZ.fromUTC("2022-02-03 04:05:06", 200),
            bodo.types.TimestampTZ.fromUTC("2023-03-04 05:06:07", 300),
            bodo.types.TimestampTZ.fromUTC("2024-04-05 06:07:08", 400),
            None,
            bodo.types.TimestampTZ.fromUTC("2022-02-03 04:05:06", 200),
        ]
    )
    section = _get_dist_arg(expected, False, False, False)
    gathered_array = bodo.libs.distributed_api.gatherv(section)
    if bodo.get_rank() == 0:
        np.testing.assert_array_equal(expected, gathered_array)


def test_distributed_getitem(memory_leak_check):
    """Test that getitem works correctly on a distributed array"""

    def impl(arr):
        return arr[0]

    arr = np.array(
        [
            bodo.types.TimestampTZ.fromUTC("2021-01-02 03:04:05", 100),
            bodo.types.TimestampTZ.fromUTC("2022-02-03 04:05:06", 200),
            bodo.types.TimestampTZ.fromUTC("2023-03-04 05:06:07", 300),
            bodo.types.TimestampTZ.fromUTC("2024-04-05 06:07:08", 400),
            None,
            bodo.types.TimestampTZ.fromUTC("2022-02-03 04:05:06", 200),
        ]
    )
    py_output = arr[0]
    check_func(impl, (arr,), py_output=py_output)


def test_distributed_scalar_optional_getitem(memory_leak_check):
    """Test that getitem works correctly on a distributed array"""

    def impl(arr, i):
        return bodo.utils.indexing.scalar_optional_getitem(arr, i)

    arr = np.array(
        [
            bodo.types.TimestampTZ.fromUTC("2021-01-02 03:04:05", 100),
            bodo.types.TimestampTZ.fromUTC("2022-02-03 04:05:06", 200),
            bodo.types.TimestampTZ.fromUTC("2023-03-04 05:06:07", 300),
            bodo.types.TimestampTZ.fromUTC("2024-04-05 06:07:08", 400),
            None,
            bodo.types.TimestampTZ.fromUTC("2022-02-03 04:05:06", 200),
        ]
    )
    for i in range(len(arr)):
        py_output = arr[i]
        check_func(impl, (arr, i), py_output=py_output)


# Test for lowering to C++ and back via the table builder API
def test_table_builder(memory_leak_check):
    global_1 = bodo.utils.typing.MetaType((0,))

    def impl(arr):
        T1 = bodo.hiframes.table.logical_table_to_table((arr,), (), global_1, 1)
        table_builder = bodo.libs.table_builder.init_table_builder_state(-1)
        bodo.libs.table_builder.table_builder_append(table_builder, T1)
        T2 = bodo.libs.table_builder.table_builder_finalize(table_builder)
        return T2

    arr = np.array(
        [
            bodo.types.TimestampTZ.fromUTC("2021-01-02 03:04:05", 100),
            bodo.types.TimestampTZ.fromUTC("2022-02-03 04:05:06", 200),
            bodo.types.TimestampTZ.fromUTC("2023-03-04 05:06:07", 300),
            bodo.types.TimestampTZ.fromUTC("2024-04-05 06:07:08", 400),
            None,
            bodo.types.TimestampTZ.fromUTC("2022-02-03 04:05:06", 200),
        ]
    )
    check_func(impl, (arr,), py_output=bodo.hiframes.table.Table([arr]))


@pytest.fixture
def timestamptz_join_data():
    """
    Returns three DataFrames used for testing joining
    when there happens to be TIMESTAMP_TZ data columns:

    df1:
        A: integer keys
        B: data column with timestamptz values

    df2:
        C: integer keys

    py_output: what should happen if df1 and df2 are joined
    on A=C.
    """
    key1_arr = np.arange(6)
    key2_arr = np.arange(4, 8)
    data_arr = np.array(
        [
            bodo.types.TimestampTZ.fromUTC("2021-01-02 03:04:05", 100),
            bodo.types.TimestampTZ.fromUTC("2022-02-03 04:05:06", 200),
            bodo.types.TimestampTZ.fromUTC("2023-03-04 05:06:07", 300),
            bodo.types.TimestampTZ.fromUTC("2024-04-05 06:07:08", 400),
            None,
            bodo.types.TimestampTZ.fromUTC("2022-02-03 04:05:06", 200),
        ]
    )
    df1 = pd.DataFrame({"A": key1_arr, "B": data_arr})
    df2 = pd.DataFrame({"C": key2_arr})
    py_output = pd.DataFrame({"A": key1_arr[4:], "B": data_arr[4:], "C": key2_arr[:2]})
    return df1, df2, py_output


def test_nonstreaming_join_timestamptz_data(timestamptz_join_data, memory_leak_check):
    """
    Tests a non-streaming join when one of the data columns is TIMESTAMPTZ.
    """

    def impl(df1, df2):
        return df1.merge(df2, left_on="A", right_on="C")[["A", "B", "C"]]

    df1, df2, py_output = timestamptz_join_data

    check_func(
        impl, (df1, df2), py_output=py_output, reset_index=True, sort_output=True
    )


def test_streaming_join_timestamptz_data(timestamptz_join_data, memory_leak_check):
    """
    Tests a streaming join when one of the data columns is TIMESTAMPTZ.
    """
    global_1 = MetaType((0, 1))
    global_2 = MetaType((0,))
    global_3 = MetaType((0,))
    global_4 = ColNamesMetaType(("A", "B"))
    global_5 = ColNamesMetaType(("C",))
    global_6 = ColNamesMetaType(("A", "B", "C"))
    global_7 = MetaType((1, 2, 0))
    global_build_outer = False
    global_probe_outer = False
    build_interval_cols = bodo.utils.typing.MetaType(())

    def impl(df1, df2):
        # Setup memory budgets and convert dataframes to tables
        bodo.libs.memory_budget.init_operator_comptroller()
        bodo.libs.memory_budget.register_operator(
            0, bodo.libs.memory_budget.OperatorType.JOIN, 0, 1, 4000
        )
        bodo.libs.memory_budget.register_operator(
            6, bodo.libs.memory_budget.OperatorType.ACCUMULATE_TABLE, 1, 1, 2880
        )
        bodo.libs.memory_budget.compute_satisfiable_budgets()
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), global_1, 2
        )
        T2 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2), (), global_2, 1
        )

        # Initialize the join state
        join_state = bodo.libs.streaming.join.init_join_state(
            0,
            global_2,
            global_3,
            global_4,
            global_5,
            global_build_outer,
            global_probe_outer,
            build_interval_cols,
            False,
        )

        # Build loop: add all of T1 to the build table
        is_last_build = False
        iter_build = 0
        length_build = bodo.hiframes.table.local_len(T1)
        done_build = False
        while not (done_build):
            T3 = bodo.hiframes.table.table_local_filter(
                T1, slice((iter_build * 3), ((iter_build + 1) * 3))
            )
            is_last_build = (iter_build * 3) >= length_build
            done_build, _ = bodo.libs.streaming.join.join_build_consume_batch(
                join_state, T3, is_last_build
            )
            iter_build = iter_build + 1

        # Probe loop: probe the join state with all of T2 and store the result in a table builder
        is_last_probe = False
        iter_probe = 0
        length_probe = bodo.hiframes.table.local_len(T2)
        done_probe = False
        produce_probe = True
        builder_state = bodo.libs.table_builder.init_table_builder_state(6)
        while not (done_probe):
            T4 = bodo.hiframes.table.table_local_filter(
                T2, slice((iter_probe * 3), ((iter_probe + 1) * 3))
            )
            is_last_probe = (iter_probe * 3) >= length_probe
            (T5, done_probe, _) = bodo.libs.streaming.join.join_probe_consume_batch(
                join_state, T4, is_last_probe, produce_probe
            )
            bodo.libs.table_builder.table_builder_append(builder_state, T5)
            iter_probe = iter_probe + 1
        bodo.libs.streaming.join.delete_join_state(join_state)

        # Extract the final table from the table builder and convert back to a DataFrame
        T6 = bodo.libs.table_builder.table_builder_finalize(builder_state)
        T7 = bodo.hiframes.table.table_subset(T6, global_7, False)
        index = bodo.hiframes.pd_index_ext.init_range_index(0, len(T7), 1, None)
        df3 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T7,), index, global_6)
        return df3

    df1, df2, py_output = timestamptz_join_data

    check_func(
        impl,
        (df1, df2),
        py_output=py_output,
        check_dtype=False,
        check_names=False,
        reset_index=True,
        sort_output=True,
    )


@pytest.fixture
def timestamptz_join_keys():
    """
    Returns three DataFrames used for testing joining on
    TIMESTAMP_TZ data:

    df1:
        A: timestamptz keys
        B: data column with unique values

    df2:
        C: timestamptz keys
        D: data column with unique values (different from df1["B"])

    py_output: what should happen if df1 and df2 are joined
    on A=C (without null matches).

    NOTE: not all rows where A=C are actually the same TIMESTAMPTZ,
    they may have different local timestamp but the same UTC timestamp.
    """
    key_arr_1 = np.array(
        [
            bodo.types.TimestampTZ.fromLocal("2024-02-29 12:00:00", 0),
            None,
            bodo.types.TimestampTZ.fromLocal("2024-02-29 16:30:00", 0),
            bodo.types.TimestampTZ.fromLocal("2024-02-29 14:30:00", 120),
            bodo.types.TimestampTZ.fromLocal("2024-07-04 00:00:00", 0),
            bodo.types.TimestampTZ.fromLocal("2024-07-04 12:00:00", 0),
            bodo.types.TimestampTZ.fromLocal("2024-07-04 23:18:01.123456789", 0),
            None,
            bodo.types.TimestampTZ.fromLocal("2024-07-05 00:00:00", 30),
            None,
        ]
    )
    key_arr_2 = np.array(
        [
            bodo.types.TimestampTZ.fromLocal("2024-07-04 00:00:00", 0),
            bodo.types.TimestampTZ.fromLocal("2024-07-04 11:15:00", -45),
            bodo.types.TimestampTZ.fromLocal("2024-07-04 20:18:01.123456789", -180),
            bodo.types.TimestampTZ.fromLocal("2024-02-29 15:30:00", -60),
            None,
            bodo.types.TimestampTZ.fromLocal("2024-07-04 20:03:01.123456789", -195),
            None,
            None,
            bodo.types.TimestampTZ.fromLocal("2024-07-05 00:00:00", 60),
            bodo.types.TimestampTZ.fromLocal("2024-02-29 16:30:00", 0),
            bodo.types.TimestampTZ.fromLocal("2024-02-29 11:30:00", -60),
            bodo.types.TimestampTZ.fromLocal("2024-07-05 00:00:00", 60),
            bodo.types.TimestampTZ.fromLocal("2024-07-04 06:00:00", 360),
            bodo.types.TimestampTZ.fromLocal("2024-02-29 12:30:00", 0),
            bodo.types.TimestampTZ.fromLocal("2024-07-04 12:00:00", 720),
            bodo.types.TimestampTZ.fromLocal("2024-07-04 00:00:00", 0),
            bodo.types.TimestampTZ.fromLocal("2024-07-04 00:00:00", 0),
            bodo.types.TimestampTZ.fromLocal("2024-07-05 00:00:00", 60),
            bodo.types.TimestampTZ.fromLocal("2024-07-04 12:00:00", 720),
            None,
            None,
            bodo.types.TimestampTZ.fromLocal("2024-07-04 00:00:00", 0),
            None,
            None,
            None,
            None,
        ]
    )
    data_arr_1 = np.arange(len(key_arr_1))
    data_arr_2 = np.arange(len(key_arr_1), len(key_arr_1) + len(key_arr_2))
    df1 = pd.DataFrame({"A": key_arr_1, "B": data_arr_1})
    df2 = pd.DataFrame({"C": key_arr_2, "D": data_arr_2})
    keep_idx_1 = [2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 6, 6]
    keep_idx_2 = [3, 9, 10, 13, 0, 12, 14, 15, 16, 18, 21, 1, 2, 5]
    py_output = pd.DataFrame(
        {
            "A": key_arr_1[keep_idx_1],
            "B": data_arr_1[keep_idx_1],
            "C": key_arr_2[keep_idx_2],
            "D": data_arr_2[keep_idx_2],
        }
    )
    return df1, df2, py_output


def test_nonstreaming_join_timestamptz_keys(timestamptz_join_keys, memory_leak_check):
    """
    Tests a non-streaming join when the keys are TIMESTAMPTZ.
    """

    def impl(df1, df2):
        return df1.merge(df2, left_on="A", right_on="C", _bodo_na_equal=False)[
            ["A", "B", "C", "D"]
        ]

    df1, df2, py_output = timestamptz_join_keys

    check_func(
        impl,
        (df1, df2),
        py_output=py_output,
        check_dtype=False,
        check_names=False,
        reset_index=True,
        sort_output=True,
    )


def test_streaming_join_timestamptz_keys(timestamptz_join_keys, memory_leak_check):
    """
    Tests a streaming join when the keys are TIMESTAMPTZ.
    """
    global_1 = MetaType((0, 1))
    global_2 = MetaType((0,))
    global_3 = MetaType((0,))
    global_4 = ColNamesMetaType(("A", "B"))
    global_5 = ColNamesMetaType(("C", "D"))
    global_6 = ColNamesMetaType(("A", "B", "C", "D"))
    global_7 = MetaType((2, 3, 0, 1))
    global_build_outer = False
    global_probe_outer = False
    build_interval_cols = bodo.utils.typing.MetaType(())

    def impl(df1, df2):
        # Setup memory budgets and convert dataframes to tables
        bodo.libs.memory_budget.init_operator_comptroller()
        bodo.libs.memory_budget.register_operator(
            0, bodo.libs.memory_budget.OperatorType.JOIN, 0, 1, 4000
        )
        bodo.libs.memory_budget.register_operator(
            6, bodo.libs.memory_budget.OperatorType.ACCUMULATE_TABLE, 1, 1, 2880
        )
        bodo.libs.memory_budget.compute_satisfiable_budgets()
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df1), (), global_1, 2
        )
        T2 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df2), (), global_1, 1
        )

        # Initialize the join state
        join_state = bodo.libs.streaming.join.init_join_state(
            0,
            global_2,
            global_3,
            global_4,
            global_5,
            global_build_outer,
            global_probe_outer,
            build_interval_cols,
            False,
        )

        # Build loop: add all of T1 to the build table
        is_last_build = False
        iter_build = 0
        length_build = bodo.hiframes.table.local_len(T1)
        done_build = False
        while not (done_build):
            T3 = bodo.hiframes.table.table_local_filter(
                T1, slice((iter_build * 3), ((iter_build + 1) * 3))
            )
            is_last_build = (iter_build * 3) >= length_build
            done_build, _ = bodo.libs.streaming.join.join_build_consume_batch(
                join_state, T3, is_last_build
            )
            iter_build = iter_build + 1

        # Probe loop: probe the join state with all of T2 and store the result in a table builder
        is_last_probe = False
        iter_probe = 0
        length_probe = bodo.hiframes.table.local_len(T2)
        done_probe = False
        produce_probe = True
        builder_state = bodo.libs.table_builder.init_table_builder_state(6)
        while not (done_probe):
            T4 = bodo.hiframes.table.table_local_filter(
                T2, slice((iter_probe * 3), ((iter_probe + 1) * 3))
            )
            is_last_probe = (iter_probe * 3) >= length_probe
            (T5, done_probe, _) = bodo.libs.streaming.join.join_probe_consume_batch(
                join_state, T4, is_last_probe, produce_probe
            )
            bodo.libs.table_builder.table_builder_append(builder_state, T5)
            iter_probe = iter_probe + 1
        bodo.libs.streaming.join.delete_join_state(join_state)

        # Extract the final table from the table builder and convert back to a DataFrame
        T6 = bodo.libs.table_builder.table_builder_finalize(builder_state)
        T7 = bodo.hiframes.table.table_subset(T6, global_7, False)
        index = bodo.hiframes.pd_index_ext.init_range_index(0, len(T7), 1, None)
        df3 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T7,), index, global_6)
        return df3

    df1, df2, py_output = timestamptz_join_keys

    check_func(
        impl,
        (df1, df2),
        py_output=py_output,
        check_dtype=False,
        check_names=False,
        reset_index=True,
        sort_output=True,
    )


@pytest_mark_one_rank
def test_concat(memory_leak_check):
    """
    Test that the bodo concat function works correctly. We test on 1 rank
    because order is not strictly defined in the output.
    """

    def impl(arr1, arr2):
        return bodo.libs.array_kernels.concat([arr1, arr2])

    arr1 = np.array(
        [
            bodo.types.TimestampTZ.fromUTC("2021-01-02 03:04:05", 100),
            bodo.types.TimestampTZ.fromUTC("2022-02-03 04:05:06", 200),
            bodo.types.TimestampTZ.fromUTC("2023-03-04 05:06:07", 300),
            bodo.types.TimestampTZ.fromUTC("2024-04-05 06:07:08", 400),
            None,
            bodo.types.TimestampTZ.fromUTC("2022-02-03 04:05:06", 200),
        ]
    )
    arr2 = np.array(
        [
            bodo.types.TimestampTZ.fromUTC("2021-01-02 03:04:05", 100),
            None,
            None,
            bodo.types.TimestampTZ.fromUTC("2022-02-03 14:45:06", 350),
        ]
    )
    py_output = np.concatenate([arr1, arr2])
    check_func(impl, (arr1, arr2), py_output=py_output)


@pytest.mark.parametrize(
    "idx",
    [
        pytest.param(
            np.array([True, False, True, False, True, True]), id="Boolean array index"
        ),
        pytest.param(np.array([1, 3, 0, 4]), id="Integer array index"),
        pytest.param(slice(1, 5, 1), id="Slice index"),
    ],
)
def test_getitem_complex(idx, memory_leak_check):
    """Test that getitem works with various idx inputs."""

    def impl(arr, idx):
        return arr[idx]

    arr = np.array(
        [
            bodo.types.TimestampTZ.fromUTC("2021-01-02 03:04:05", 100),
            bodo.types.TimestampTZ.fromUTC("2022-02-03 04:05:06", 200),
            bodo.types.TimestampTZ.fromUTC("2023-03-04 05:06:07", 300),
            bodo.types.TimestampTZ.fromUTC("2024-04-05 06:07:08", 400),
            None,
            bodo.types.TimestampTZ.fromUTC("2022-02-03 04:05:06", 200),
        ]
    )
    # Integer array indexing isn't supported on distributed data.
    only_seq = isinstance(idx, np.ndarray) and np.issubdtype(idx.dtype, np.integer)
    check_func(impl, (arr, idx), only_seq=only_seq)


@pytest.mark.parametrize(
    "idx",
    [
        pytest.param(
            np.array([True, False, True, False, True, True]), id="Boolean array index"
        ),
        pytest.param(np.array([1, 3, 0, 4]), id="Integer array index"),
        pytest.param(slice(1, 5, 1), id="Slice index"),
    ],
)
def test_setitem_complex(idx, memory_leak_check):
    """Test that setitem works with various idx inputs."""

    def impl(arr, idx, val):
        arr[idx] = val
        return arr

    arr = np.array(
        [
            bodo.types.TimestampTZ.fromUTC("2021-01-02 03:04:05", 100),
            bodo.types.TimestampTZ.fromUTC("2022-02-03 04:05:06", 200),
            bodo.types.TimestampTZ.fromUTC("2023-03-04 05:06:07", 300),
            bodo.types.TimestampTZ.fromUTC("2024-04-05 06:07:08", 400),
            None,
            bodo.types.TimestampTZ.fromUTC("2022-02-03 04:05:06", 200),
        ]
    )
    scalar_val = bodo.types.TimestampTZ.fromUTC("2021-01-02 14:21:17", 700)
    # Integer array indexing isn't supported on distributed data.
    only_seq = isinstance(idx, np.ndarray) and np.issubdtype(idx.dtype, np.integer)
    check_func(impl, (arr, idx, scalar_val), copy_input=True, only_seq=only_seq)
    arr_val = np.array(
        [
            bodo.types.TimestampTZ.fromUTC("2021-01-02 14:21:17", 700),
            None,
            None,
            bodo.types.TimestampTZ.fromUTC("2024-12-17 14:45:06", 350),
        ]
    )
    # arr_val cannot be distributed differently, so this only works with 1 rank.
    check_func(impl, (arr, idx, arr_val), copy_input=True, only_seq=True)


def test_cmp_with_timestamp(memory_leak_check):
    @bodo.jit
    def gt(a, b):
        return a > b

    a = bodo.types.TimestampTZ.fromLocal("2024-01-01 00:00:00", -420)
    b = pd.Timestamp("2024-01-01 00:00:00")
    assert gt(a, b)
    assert not gt(b, a)

    c = bodo.types.TimestampTZ.fromLocal("2024-01-01 00:00:00", 420)
    assert not gt(c, b)
    assert gt(b, c)
