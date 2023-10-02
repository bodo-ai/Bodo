import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from numba.core import types

import bodo
from bodo.hiframes.table import TableType
from bodo.libs.stream_union import UnionStateType
from bodo.tests.utils import check_func, pytest_mark_pandas
from bodo.utils.typing import ColNamesMetaType, MetaType


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
        df2 = pd.DataFrame(
            {
                "A": np.zeros(6, dtype="bool"),
            }
        )
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


@pytest.mark.skipif(bodo.get_size() > 1, reason="Only run on 1 rank")
def test_stream_union_integer_promotion(memory_leak_check):
    """
    Test Union Casting between:
    - Non-null Integer Promotion
    - Null and non-null Integer Promotion
    - Non-null, nullable, and null array with promotion
    """

    non_null_int8_arr = types.Array(types.int8, 1, "C")
    null_int8_arr = bodo.IntegerArrayType(types.int8)
    non_null_int32_arr = types.Array(types.int32, 1, "C")
    null_int32_arr = bodo.IntegerArrayType(types.int32)
    non_null_int64_arr = types.Array(types.int64, 1, "C")
    null_int64_arr = bodo.IntegerArrayType(types.int64)

    state = UnionStateType(
        in_table_types=(
            TableType((non_null_int32_arr, null_int8_arr, non_null_int64_arr)),
            TableType((non_null_int8_arr, non_null_int32_arr, bodo.null_array_type)),
            TableType((non_null_int64_arr, null_int8_arr, null_int32_arr)),
        )
    )

    assert state.out_table_type == TableType(
        (
            non_null_int64_arr,
            null_int32_arr,
            null_int64_arr,
        )
    )


@pytest.mark.skipif(bodo.get_size() > 1, reason="Only run on 1 rank")
def test_stream_union_dict_encoding_combo(memory_leak_check):
    """
    Test Union Casting between:
    - String and Dictionary Encoded Arrays
    - Dictionary Encoded and Null Array
    - Dictionary Encoded, String, and Null Arrays
    """

    state = UnionStateType(
        in_table_types=(
            TableType(
                (bodo.string_array_type, bodo.dict_str_arr_type, bodo.string_array_type)
            ),
            TableType(
                (bodo.dict_str_arr_type, bodo.null_array_type, bodo.dict_str_arr_type)
            ),
            TableType(
                (bodo.string_array_type, bodo.null_array_type, bodo.string_array_type)
            ),
        )
    )

    assert state.out_table_type == TableType(
        (
            bodo.dict_str_arr_type,
            bodo.dict_str_arr_type,
            bodo.dict_str_arr_type,
        )
    )


@pytest.mark.skipif(bodo.get_size() > 1, reason="Only run on 1 rank")
def test_stream_union_null(memory_leak_check):
    """
    Test Union Casting between:
    - Non-nullable and null array
    - Nullable and null array
    - Non-nullable and nullable array (of same type)
    - 2 Null Arrays
    """

    non_null_bool_arr = types.Array(types.bool_, 1, "C")

    state = UnionStateType(
        in_table_types=(
            TableType(
                (
                    non_null_bool_arr,
                    bodo.string_array_type,
                    bodo.boolean_array_type,
                    bodo.null_array_type,
                )
            ),
            TableType(
                (
                    bodo.null_array_type,
                    bodo.null_array_type,
                    non_null_bool_arr,
                    bodo.null_array_type,
                )
            ),
        )
    )

    assert state.out_table_type == TableType(
        (
            bodo.boolean_array_type,
            bodo.string_array_type,
            bodo.boolean_array_type,
            bodo.null_array_type,
        )
    )


@pytest.mark.parametrize("all", [True, False])
def test_stream_union_distinct_basic(all, datapath, memory_leak_check):
    """
    Basic test for Streaming Union, especially for testing coverage
    The BodoSQL UNION tests cover edge cases
    """
    customer_path: str = datapath("tpch-test_data/parquet/customer.parquet")
    orders_path: str = datapath("tpch-test_data/parquet/orders.parquet")
    global_1 = ColNamesMetaType(("c_custkey",))
    global_2 = MetaType((0,))
    global_3 = MetaType((1,))

    def impl(customer_path, orders_path):
        is_last1 = False
        _iter_1 = 0
        state_1 = pd.read_parquet(customer_path, _bodo_chunksize=4000)
        state_2 = bodo.libs.stream_union.init_union_state(-1, all=all)
        while not is_last1:
            T1, is_last1 = bodo.io.arrow_reader.read_arrow_next(state_1, True)
            T3 = bodo.hiframes.table.table_subset(T1, global_2, False)
            bodo.libs.stream_union.union_consume_batch(state_2, T3, False)
            _iter_1 = _iter_1 + 1
        bodo.io.arrow_reader.arrow_reader_del(state_1)

        is_last2 = False
        _iter_2 = 0
        state_3 = pd.read_parquet(orders_path, _bodo_chunksize=4000)
        _temp1 = False
        while not _temp1:
            T2, is_last2 = bodo.io.arrow_reader.read_arrow_next(state_3, True)
            T4 = bodo.hiframes.table.table_subset(T2, global_3, False)
            _temp1 = bodo.libs.stream_union.union_consume_batch(state_2, T4, is_last2)
            _iter_2 = _iter_2 + 1
        bodo.io.arrow_reader.arrow_reader_del(state_3)

        is_last3 = False
        _iter_3 = 0
        table_builder = bodo.libs.table_builder.init_table_builder_state(-1)
        while not is_last3:
            T5, is_last3 = bodo.libs.stream_union.union_produce_batch(state_2, True)
            bodo.libs.table_builder.table_builder_append(table_builder, T5)
            _iter_3 = _iter_3 + 1

        bodo.libs.stream_union.delete_union_state(state_2)
        T6 = bodo.libs.table_builder.table_builder_finalize(table_builder)
        index_1 = bodo.hiframes.pd_index_ext.init_range_index(0, len(T6), 1, None)
        df1 = bodo.hiframes.pd_dataframe_ext.init_dataframe((T6,), index_1, global_1)
        return df1

    cust_df = pd.read_parquet(customer_path, columns=["C_CUSTKEY"]).rename(
        columns={"C_CUSTKEY": "c_custkey"}
    )
    ord_df = pd.read_parquet(orders_path, columns=["O_CUSTKEY"]).rename(
        columns={"O_CUSTKEY": "c_custkey"}
    )

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
