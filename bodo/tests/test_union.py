from numba.core import types  # noqa TID253
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
from bodo.tests.utils import (
    check_func,
    pytest_mark_one_rank,
    pytest_mark_pandas,
    temp_env_override,
)


@pytest_mark_pandas
def test_union_dict_encoding(memory_leak_check):
    """
    Tests that a union of 3 tables with 1 dictionary encoded column
    works and results in a dictionary encoded output.
    """
    new_cols_tup = bodo.utils.typing.ColNamesMetaType(("A",))

    def impl(df1, df2, df3, drop_duplicates):
        return bodo.hiframes.pd_dataframe_ext.union_dataframes(
            (df1, df2, df3), drop_duplicates, new_cols_tup
        )

    # Generate the arrays for the columns.
    dict_list = ["abc", "b", None, "abc", None, "b", "cde", "россия "]
    dict_arr = pd.arrays.ArrowStringArray(
        pa.array(
            dict_list,
            type=pa.dictionary(pa.int32(), pa.string()),
        ).cast(pa.dictionary(pa.int32(), pa.large_string()))
    )
    str_list1 = ["ac", "e", None, "abc", None, "b", "cde"]
    str_list2 = ["россия", "alxfe", None, "abc", None, "b", "cde", "b", "d"]
    df1 = pd.DataFrame({"A": str_list1})
    df2 = pd.DataFrame({"B": dict_arr})
    df3 = pd.DataFrame({"C": str_list2})
    for drop_duplicates in (True, False):
        concat_list = dict_list + str_list1 + str_list2
        py_output = pd.DataFrame({"A": concat_list})
        if drop_duplicates:
            py_output = py_output.drop_duplicates()
        check_func(
            impl,
            (df1, df2, df3, drop_duplicates),
            py_output=py_output,
            sort_output=True,
            reset_index=True,
            use_dict_encoded_strings=False,
        )


@pytest_mark_pandas
def test_union_integer_promotion(memory_leak_check):
    """
    Tests that a union of 3 tables will promote to the maximum
    bit width of the integer columns and will be nullable if any is
    nullable.
    """
    new_cols_tup = bodo.utils.typing.ColNamesMetaType(("A",))

    def impl(df1, df2, df3, drop_duplicates):
        return bodo.hiframes.pd_dataframe_ext.union_dataframes(
            (df1, df2, df3), drop_duplicates, new_cols_tup
        )

    # Generate the arrays for the columns.
    int_list1 = [1, 2, 6, 4, 3, 4, 6]
    int_list2 = [1, 2, 5, 4, None] * 3
    int_list3 = [155, 2134, 231313, 4532532, 5425422, 532]
    df1 = pd.DataFrame({"A": pd.Series(int_list1, dtype="int32")})
    df2 = pd.DataFrame({"B": pd.Series(int_list2, dtype="Int8")})
    df3 = pd.DataFrame({"C": pd.Series(int_list3, dtype="int64")})
    for drop_duplicates in (True, False):
        concat_list = int_list1 + int_list2 + int_list3
        py_output = pd.DataFrame({"A": pd.Series(concat_list, dtype="Int64")})
        if drop_duplicates:
            py_output = py_output.drop_duplicates()
        check_func(
            impl,
            (df1, df2, df3, drop_duplicates),
            py_output=py_output,
            sort_output=True,
            reset_index=True,
        )


@pytest_mark_pandas
def test_union_nullable_boolean(memory_leak_check):
    """
    Tests a union of two DataFrames with boolean columns where 1 column is
    nullable and the other is non-nullable.
    """
    new_cols_tup = bodo.utils.typing.ColNamesMetaType(("A",))

    def impl(df1):
        # Create df2 in JIT so we keep the type as a non-nullable boolean.
        df2 = pd.DataFrame({"A": np.zeros(6, dtype="bool")})
        return bodo.hiframes.pd_dataframe_ext.union_dataframes(
            (df1, df2), False, new_cols_tup
        )

    df1 = pd.DataFrame(
        {
            "A": pd.array([True, False, None] * 3, dtype="boolean"),
        }
    )
    py_output = pd.DataFrame(
        {
            "A": pd.array([True, False, None] * 3 + [False] * 6, dtype="boolean"),
        }
    )
    check_func(
        impl,
        (df1,),
        py_output=py_output,
        sort_output=True,
        reset_index=True,
    )


@pytest_mark_one_rank
def test_stream_union_integer_promotion(memory_leak_check):
    """
    Test Union Casting between:
    - Non-null Integer Promotion
    - Null and non-null Integer Promotion
    - Non-null, nullable, and null array with promotion

    Logic is based on investigation from:
    https://bodo.atlassian.net/wiki/spaces/B/pages/1474134034/Numeric+Casting+Investigation+for+Union
    """
    import bodo.decorators  # isort:skip # noqa
    from bodo.hiframes.table import TableType
    from bodo.libs.streaming.union import UnionStateType

    nn_int8_arr = types.Array(types.int8, 1, "C")
    null_int8_arr = bodo.types.IntegerArrayType(types.int8)
    nn_int32_arr = types.Array(types.int32, 1, "C")
    null_int32_arr = bodo.types.IntegerArrayType(types.int32)
    nn_int64_arr = types.Array(types.int64, 1, "C")
    null_int64_arr = bodo.types.IntegerArrayType(types.int64)

    state = UnionStateType(
        in_table_types=(
            TableType((nn_int32_arr, null_int8_arr, nn_int64_arr)),
            TableType((nn_int8_arr, nn_int32_arr, bodo.types.null_array_type)),
            TableType((nn_int64_arr, null_int8_arr, null_int32_arr)),
        )
    )

    assert state.out_table_type == TableType(
        (
            nn_int64_arr,
            null_int32_arr,
            null_int64_arr,
        )
    )


@pytest_mark_one_rank
def test_stream_union_float_promotion(memory_leak_check):
    """
    Test Union Casting between:
    - Floats
    - Integer and Float

    Logic is based on investigation from:
    https://bodo.atlassian.net/wiki/spaces/B/pages/1474134034/Numeric+Casting+Investigation+for+Union
    but assuming float == Snowflake Float != Snowflake Number
    so can never cast float => decimal
    """
    import bodo.decorators  # isort:skip # noqa
    from bodo.hiframes.table import TableType
    from bodo.libs.streaming.union import UnionStateType

    # Float and Integer + Float Tests
    nn_int8_arr = types.Array(types.int8, 1, "C")
    null_int32_arr = bodo.types.IntegerArrayType(types.int32)
    null_int64_arr = bodo.types.IntegerArrayType(types.int64)
    nn_f32_arr = types.Array(types.float32, 1, "C")
    null_f64_arr = bodo.types.FloatingArrayType(types.float64)

    state = UnionStateType(
        in_table_types=(
            TableType((nn_f32_arr, nn_f32_arr, nn_f32_arr, null_f64_arr)),
            TableType((null_f64_arr, nn_int8_arr, null_int32_arr, null_int64_arr)),
        )
    )

    assert state.out_table_type == TableType(
        (
            null_f64_arr,
            nn_f32_arr,
            null_f64_arr,
            null_f64_arr,
        )
    )


@pytest_mark_one_rank
def test_stream_union_decimal_promotion(memory_leak_check):
    """
    Test Union Casting between:
    - Decimal
    - Integer and Decimal
    - Integer, Float, and Decimal

    Logic is based on investigation from:
    https://bodo.atlassian.net/wiki/spaces/B/pages/1474134034/Numeric+Casting+Investigation+for+Union
    but assuming:
    - float == Snowflake Float != Snowflake Number
        In general, cast float => decimal is unsafe
        (since float can have values outside decimals range)
    - Truncation of integer + decimal => float allowed
        Decimal has at most 38 sig-figs, float has up to 15, integer 18
    """
    import bodo.decorators  # isort:skip # noqa
    from bodo.hiframes.table import TableType
    from bodo.libs.streaming.union import UnionStateType

    state = UnionStateType(
        in_table_types=(
            TableType(
                (
                    bodo.types.DecimalArrayType(12, 5),
                    bodo.types.DecimalArrayType(15, 6),
                    bodo.types.DecimalArrayType(16, 0),
                    bodo.types.IntegerArrayType(types.int32),
                    types.Array(types.float32, 1, "C"),
                    bodo.types.FloatingArrayType(types.float64),
                )
            ),
            TableType(
                (
                    bodo.types.DecimalArrayType(15, 6),
                    bodo.types.DecimalArrayType(12, 5),
                    bodo.types.IntegerArrayType(types.int64),
                    bodo.types.DecimalArrayType(26, 0),
                    bodo.types.DecimalArrayType(15, 6),
                    bodo.types.DecimalArrayType(38, 18),
                )
            ),
            TableType(
                (
                    bodo.types.DecimalArrayType(25, 0),
                    bodo.types.DecimalArrayType(15, 14),
                    types.Array(types.int32, 1, "C"),
                    types.Array(types.int8, 1, "C"),
                    bodo.types.IntegerArrayType(types.int32),
                    bodo.types.DecimalArrayType(38, 4),
                )
            ),
        )
    )

    assert state.out_table_type == TableType(
        (
            # All sources are decimal, so continue as decimal
            # Max Scale=6, Max Non-Scale=25
            bodo.types.DecimalArrayType(31, 6),
            # All sources are decimal, so continue as decimal
            # This is not converted to float64 for Numeric safety
            bodo.types.DecimalArrayType(23, 14),
            # All are decimal with 0 scale or integer
            # Scale=0, Max Non-Scale=16 => int64
            bodo.types.IntegerArrayType(types.int64),
            # All are decimal with 0 scale or integer
            # Scale=0, Max Non-Scale=26 which is above max int size
            bodo.types.DecimalArrayType(26, 0),
            # Merging a float, so must cast to float at end
            # Max Precision=15 => float64
            bodo.types.FloatingArrayType(types.float64),
            # Merging a float, so must cast to float at end
            # Max Precision=38 => overflow but truncate to float64
            bodo.types.FloatingArrayType(types.float64),
        )
    )


@pytest_mark_one_rank
def test_stream_union_dict_encoding_combo(memory_leak_check):
    """
    Test Union Casting between:
    - String and Dictionary Encoded Arrays
    - Dictionary Encoded and Null Array
    - Dictionary Encoded, String, and Null Arrays
    """
    import bodo.decorators  # isort:skip # noqa
    from bodo.hiframes.table import TableType
    from bodo.libs.streaming.union import UnionStateType

    state = UnionStateType(
        in_table_types=(
            TableType(
                (
                    bodo.types.string_array_type,
                    bodo.types.dict_str_arr_type,
                    bodo.types.string_array_type,
                )
            ),
            TableType(
                (
                    bodo.types.dict_str_arr_type,
                    bodo.types.null_array_type,
                    bodo.types.dict_str_arr_type,
                )
            ),
            TableType(
                (
                    bodo.types.string_array_type,
                    bodo.types.null_array_type,
                    bodo.types.string_array_type,
                )
            ),
        )
    )

    assert state.out_table_type == TableType(
        (
            bodo.types.dict_str_arr_type,
            bodo.types.dict_str_arr_type,
            bodo.types.dict_str_arr_type,
        )
    )


@pytest_mark_one_rank
def test_stream_union_null(memory_leak_check):
    """
    Test Union Casting between:
    - Non-nullable and null array
    - Nullable and null array
    - Non-nullable and nullable array (of same type)
    - 2 Null Arrays
    """
    import bodo.decorators  # isort:skip # noqa
    from bodo.hiframes.table import TableType
    from bodo.libs.streaming.union import UnionStateType

    non_null_bool_arr = types.Array(types.bool_, 1, "C")

    state = UnionStateType(
        in_table_types=(
            TableType(
                (
                    non_null_bool_arr,
                    bodo.types.string_array_type,
                    bodo.types.boolean_array_type,
                    bodo.types.null_array_type,
                )
            ),
            TableType(
                (
                    bodo.types.null_array_type,
                    bodo.types.null_array_type,
                    non_null_bool_arr,
                    bodo.types.null_array_type,
                )
            ),
        )
    )

    assert state.out_table_type == TableType(
        (
            bodo.types.boolean_array_type,
            bodo.types.string_array_type,
            bodo.types.boolean_array_type,
            bodo.types.null_array_type,
        )
    )


@pytest.mark.parametrize("all", [True, False])
def test_stream_union_distinct_basic(all, datapath, memory_leak_check):
    """
    Basic test for Streaming Union, especially for testing coverage
    The BodoSQL UNION tests cover edge cases
    """
    from bodo.utils.typing import ColNamesMetaType, MetaType

    customer_path: str = datapath("tpch-test_data/parquet/customer.pq")
    orders_path: str = datapath("tpch-test_data/parquet/orders.pq")
    global_1 = ColNamesMetaType(("c_custkey",))
    global_2 = MetaType((0,))
    global_3 = MetaType((1,))

    def impl(customer_path, orders_path):
        is_last1 = False
        _iter_1 = 0
        state_1 = pd.read_parquet(
            customer_path, _bodo_chunksize=4000, dtype_backend="pyarrow"
        )
        state_2 = bodo.libs.streaming.union.init_union_state(-1, all=all)
        while not is_last1:
            T1, is_last1 = bodo.io.arrow_reader.read_arrow_next(state_1, True)
            T3 = bodo.hiframes.table.table_subset(T1, global_2, False)
            is_last1, _ = bodo.libs.streaming.union.union_consume_batch(
                state_2, T3, is_last1, False
            )
            _iter_1 = _iter_1 + 1
        bodo.io.arrow_reader.arrow_reader_del(state_1)

        is_last2 = False
        _iter_2 = 0
        state_3 = pd.read_parquet(
            orders_path, _bodo_chunksize=4000, dtype_backend="pyarrow"
        )
        _temp1 = False
        while not _temp1:
            T2, is_last2 = bodo.io.arrow_reader.read_arrow_next(state_3, True)
            T4 = bodo.hiframes.table.table_subset(T2, global_3, False)
            _temp1, _ = bodo.libs.streaming.union.union_consume_batch(
                state_2, T4, is_last2, True
            )
            _iter_2 = _iter_2 + 1
        bodo.io.arrow_reader.arrow_reader_del(state_3)

        is_last3 = False
        _iter_3 = 0
        table_builder = bodo.libs.table_builder.init_table_builder_state(-1)
        while not is_last3:
            T5, is_last3 = bodo.libs.streaming.union.union_produce_batch(state_2, True)
            bodo.libs.table_builder.table_builder_append(table_builder, T5)
            _iter_3 = _iter_3 + 1

        bodo.libs.streaming.union.delete_union_state(state_2)
        T6 = bodo.libs.table_builder.table_builder_finalize(table_builder)
        index_1 = bodo.hiframes.pd_index_ext.init_range_index(0, len(T6), 1, None)
        df1 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T6,), index_1, global_1)
        return df1

    cust_df = pd.read_parquet(
        customer_path, columns=["C_CUSTKEY"], dtype_backend="pyarrow"
    ).rename(columns={"C_CUSTKEY": "c_custkey"})
    ord_df = pd.read_parquet(
        orders_path, columns=["O_CUSTKEY"], dtype_backend="pyarrow"
    ).rename(columns={"O_CUSTKEY": "c_custkey"})

    py_output = pd.concat([cust_df, ord_df], ignore_index=True)
    if not all:
        py_output = py_output.drop_duplicates()

    check_func(
        impl,
        (customer_path, orders_path),
        py_output=py_output,
        sort_output=True,
        reset_index=True,
        check_dtype=False,
    )


@pytest.mark.skipif(bodo.get_size() != 2, reason="Only calibrated for 2 ranks")
def test_stream_union_distinct_sync(datapath, memory_leak_check):
    """
    Test that streaming union synchronization works as expected across
    multiple pipelines where the number of input batches on different ranks
    might be different.
    """
    from bodo.utils.typing import ColNamesMetaType, MetaType

    customer_path: str = datapath("tpch-test_data/parquet/customer.pq")
    orders_path: str = datapath("tpch-test_data/parquet/orders.pq")
    global_1 = ColNamesMetaType(("c_custkey",))
    global_2 = MetaType((0,))
    global_3 = MetaType((0,))

    def impl(customer_df, orders_df):
        is_last1 = False
        _iter_1 = 0
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(customer_df),
            (),
            global_2,
            1,
        )
        T1_len = bodo.hiframes.table.local_len(T1)
        union_state = bodo.libs.streaming.union.init_union_state(-1, all=False)
        while not is_last1:
            # We use a small batch size to force different number of iterations.
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_iter_1 * 200), ((_iter_1 + 1) * 200))
            )
            is_last1 = ((_iter_1 + 1) * 200) >= T1_len
            is_last1, _ = bodo.libs.streaming.union.union_consume_batch(
                union_state, T2, is_last1, False
            )
            _iter_1 = _iter_1 + 1

        is_last2 = False
        _iter_2 = 0
        T3 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(orders_df),
            (),
            global_3,
            1,
        )
        T3_len = bodo.hiframes.table.local_len(T3)
        while not is_last2:
            T4 = bodo.hiframes.table.table_local_filter(
                T3, slice((_iter_2 * 200), ((_iter_2 + 1) * 200))
            )
            is_last2 = ((_iter_2 + 1) * 200) >= T3_len
            is_last2, _ = bodo.libs.streaming.union.union_consume_batch(
                union_state, T4, is_last2, True
            )
            _iter_2 = _iter_2 + 1

        is_last3 = False
        _iter_3 = 0
        table_builder = bodo.libs.table_builder.init_table_builder_state(-1)
        while not is_last3:
            T5, is_last3 = bodo.libs.streaming.union.union_produce_batch(
                union_state, True
            )
            bodo.libs.table_builder.table_builder_append(table_builder, T5)
            _iter_3 = _iter_3 + 1

        bodo.libs.streaming.union.delete_union_state(union_state)
        T6 = bodo.libs.table_builder.table_builder_finalize(table_builder)
        index_1 = bodo.hiframes.pd_index_ext.init_range_index(0, len(T6), 1, None)
        df1 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T6,), index_1, global_1)
        return df1

    cust_df = pd.read_parquet(
        customer_path, columns=["C_CUSTKEY"], dtype_backend="pyarrow"
    ).rename(columns={"C_CUSTKEY": "c_custkey"})
    ord_df = pd.read_parquet(
        orders_path, columns=["O_CUSTKEY"], dtype_backend="pyarrow"
    ).rename(columns={"O_CUSTKEY": "c_custkey"})

    py_output = pd.concat([cust_df, ord_df], ignore_index=True)
    py_output = py_output.drop_duplicates()

    len_cust_df = cust_df.shape[0]
    len_ord_df = ord_df.shape[0]

    # Distribute the input unevenly between the two ranks
    # to force potential hang from shuffle and is_last sync
    if bodo.get_rank() == 0:
        cust_df = cust_df.iloc[: (len_cust_df // 4)]
        ord_df = ord_df.iloc[(len_ord_df // 4) :]
    else:
        cust_df = cust_df.iloc[(len_cust_df // 4) :]
        ord_df = ord_df.iloc[: (len_ord_df // 4)]

    # Test with a very low shuffle threshold to force many syncs.
    with temp_env_override({"BODO_SHUFFLE_THRESHOLD": str(1024)}):
        out_df = bodo.jit(distributed=["customer_df", "orders_df"])(impl)(
            cust_df, ord_df
        )
        # Verify that the output is correct
        out_df = bodo.allgatherv(out_df)
        pd.testing.assert_frame_equal(
            out_df.sort_values(by="c_custkey").reset_index(drop=True),
            py_output.sort_values(by="c_custkey").reset_index(drop=True),
            check_dtype=False,
            check_index_type=False,
        )


@pytest.mark.parametrize(
    "df,use_map_arrays",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "a": np.arange(50),
                    "b": pd.array(
                        [
                            [
                                1,
                            ],
                            [3],
                            None,
                            [4, 5, None],
                            [6, 7, 8, 9],
                        ]
                        * 10,
                        dtype=pd.ArrowDtype(pa.large_list(pa.int64())),
                    ),
                }
            ),
            False,
            id="nested_array",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "a": np.arange(50),
                    "b": pd.array(
                        [
                            {"1": 38.7, "2": "xyz"},
                            {"1": 11.0, "2": "pqr"},
                            None,
                            {"1": 329.1, "2": "abc"},
                            {"1": 329.1, "2": "abc"},
                        ]
                        * 10,
                        dtype=pd.ArrowDtype(
                            pa.struct(
                                [
                                    pa.field("1", pa.float64()),
                                    pa.field("2", pa.string()),
                                ]
                            )
                        ),
                    ),
                }
            ),
            False,
            id="struct_array",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "a": np.arange(50),
                    "b": pd.Series(
                        [
                            {"1": 38.7, "2": 33.2},
                            {"3": 11.0},
                            None,
                            {"abc": 398.21, "jakasf": None},
                            {},
                        ]
                        * 10,
                        dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.float64())),
                    ),
                }
            ),
            True,
            id="map_array",
        ),
        pytest.param(
            pd.DataFrame(
                {"a": np.arange(50), "b": [(1, 2), None, (3, 4), (2, 9), None] * 10}
            ),
            False,
            id="tuple_array",
            marks=pytest.mark.skip(
                "TODO[BSE-2076]: support tuple arrays in Arrow boxing/unboxing"
            ),
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "a": np.arange(50),
                    "b": pd.array(
                        [[[1, 2]], [[3, 4]], [[2, 9]]] * 16 + [[[]], None],
                        dtype=pd.ArrowDtype(pa.large_list(pa.large_list(pa.int64()))),
                    ),
                }
            ),
            False,
            id="nested_nested_array",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "a": np.arange(50),
                    "b": pd.array(
                        [
                            [
                                {"1": 38.7, "2": "xyz"},
                                {"1": 11.0, "2": "pqr"},
                            ],
                            [
                                {"1": 38.7, "2": "xyz"},
                                {"1": 11.0, "2": "pqr"},
                            ],
                            None,
                            [{"1": 2819.2, "2": "abc"}],
                            None,
                        ]
                        * 10,
                        dtype=pd.ArrowDtype(
                            pa.large_list(
                                pa.struct(
                                    [
                                        pa.field("1", pa.float64()),
                                        pa.field("2", pa.string()),
                                    ]
                                )
                            )
                        ),
                    ),
                }
            ),
            False,
            id="nested_struct_array",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "a": np.arange(50),
                    "b": pd.array(
                        [
                            [{"1": 38.7, "2": 33.2}, None],
                            [{"3": 11.0}],
                            [{"abc": 398.21, "jakasf": None}],
                            [],
                            None,
                        ]
                        * 10,
                        dtype=pd.ArrowDtype(
                            pa.large_list(pa.map_(pa.string(), pa.float64()))
                        ),
                    ),
                }
            ),
            True,
            id="nested_map_array",
            marks=pytest.mark.skip(
                "Boxing map arrays inside of nested arrays is not supported yet"
            ),
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "a": np.arange(50),
                    "b": [
                        [["a", "b", "c"], ["d", "e", "f"]],
                        [["g", "h", "i"]],
                        [["j", "k", "l"], ["m", "n", "o"]],
                    ]
                    * 16
                    + [[[]], None],
                }
            ),
            False,
            marks=pytest.mark.skip(
                "Check func can't compare doubly nested string arrays yet"
            ),
            id="nested_string_array",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "a": np.arange(50),
                    "b": pd.array(
                        [
                            {"1": "38.7", "2": "33.2"},
                            {"1": "11.0", "2": "oarnsio"},
                            {"1": "398.21", "2": "pqr"},
                            None,
                            None,
                        ]
                        * 10,
                        dtype=pd.ArrowDtype(
                            pa.struct(
                                [pa.field("1", pa.string()), pa.field("2", pa.string())]
                            )
                        ),
                    ),
                }
            ),
            False,
            id="struct_string_array",
        ),
    ],
)
@pytest.mark.parametrize("all", [True, False])
def test_nested_array_stream_union(all, df, use_map_arrays, memory_leak_check):
    from bodo.utils.typing import ColNamesMetaType, MetaType

    if not all:
        pytest.skip("Nested Arrays don't support equality yet")
    global_1 = ColNamesMetaType(("a", "b"))
    global_2 = MetaType((0, 1))

    def impl(df):
        is_last1 = False
        _iter_1 = 0
        state_2 = bodo.libs.streaming.union.init_union_state(-1, all=all)
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df), (), global_2, 1
        )
        while not is_last1:
            T3 = bodo.hiframes.table.table_local_filter(
                T1, slice((_iter_1 * 3), ((_iter_1 + 1) * 3))
            )
            is_last1 = (_iter_1 * 3) >= len(T1)
            is_last1, _ = bodo.libs.streaming.union.union_consume_batch(
                state_2, T3, is_last1, False
            )
            _iter_1 = _iter_1 + 1

        is_last2 = False
        _iter_2 = 0
        _temp1 = False
        while not _temp1:
            T5 = bodo.hiframes.table.table_local_filter(
                T1, slice((_iter_2 * 3), ((_iter_2 + 1) * 3))
            )
            _iter_2 = _iter_2 + 1
            is_last2 = (_iter_2 * 3) >= len(T1)
            _temp1, _ = bodo.libs.streaming.union.union_consume_batch(
                state_2, T5, is_last2, True
            )

        is_last3 = False
        _iter_3 = 0
        table_builder = bodo.libs.table_builder.init_table_builder_state(-1)
        while not is_last3:
            T6, is_last3 = bodo.libs.streaming.union.union_produce_batch(state_2, True)
            bodo.libs.table_builder.table_builder_append(table_builder, T6)
            _iter_3 = _iter_3 + 1

        bodo.libs.streaming.union.delete_union_state(state_2)
        T7 = bodo.libs.table_builder.table_builder_finalize(table_builder)
        index_1 = bodo.hiframes.pd_index_ext.init_range_index(0, len(T7), 1, None)
        df1 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T7,), index_1, global_1)
        # Sort for testing equality, can't use check_func's sort
        # because not implemented for nested arrays
        df1 = df1.sort_values(by="a")

        return df1

    py_output = pd.concat([df, df], ignore_index=True)
    if not all:
        py_output = py_output.drop_duplicates()
    py_output = py_output.sort_values(by="a").reset_index(drop=True)
    check_func(
        impl,
        (df,),
        py_output=py_output,
        reset_index=True,
        check_dtype=False,
        use_map_arrays=use_map_arrays,
    )
