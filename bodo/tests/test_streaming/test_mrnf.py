import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo

import bodo.decorators  # isort:skip # noqa
from bodo.libs.streaming.groupby import (
    delete_groupby_state,
    groupby_build_consume_batch,
    groupby_produce_output_batch,
    init_groupby_state,
)
from bodo.tests.utils import check_func, pytest_mark_one_rank, temp_env_override
from bodo.utils.typing import BodoError, ColNamesMetaType, MetaType


def _test_mrnf_helper(
    df,
    expected_df,
    keys_inds,
    fnames,
    f_in_offsets,
    f_in_cols,
    mrnf_sort_col_inds,
    mrnf_sort_col_asc,
    mrnf_sort_col_na,
    mrnf_col_inds_keep,
    input_table_kept_cols,
    batch_size,
    output_table_col_meta,
):
    """
    Helper function for the streaming MRNF unit tests.
    Since the MRNF codegen is nearly identical for all tests,
    we only define it once here.
    We use check_func to verify correctness.
    """
    n_cols = len(input_table_kept_cols.meta)

    def test_mrnf(df):
        mrnf_state = init_groupby_state(
            -1,
            keys_inds,
            fnames,
            f_in_offsets,
            f_in_cols,
            mrnf_sort_col_inds,
            mrnf_sort_col_asc,
            mrnf_sort_col_na,
            mrnf_col_inds_keep,
        )

        is_last1 = False
        _iter_1 = 0
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df),
            (),
            input_table_kept_cols,
            n_cols,
        )
        _temp1 = bodo.hiframes.table.local_len(T1)
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_iter_1 * batch_size), ((_iter_1 + 1) * batch_size))
            )
            is_last1 = (_iter_1 * batch_size) >= _temp1
            T3 = bodo.hiframes.table.table_subset(T2, input_table_kept_cols, False)
            _iter_1 = _iter_1 + 1
            is_last1, _ = groupby_build_consume_batch(mrnf_state, T3, is_last1, True)
        out_dfs = []
        is_last2 = False
        while not is_last2:
            out_table, is_last2 = groupby_produce_output_batch(mrnf_state, True)
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(out_table), 1, None
            )
            df_final = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (out_table,), index_var, output_table_col_meta
            )
            out_dfs.append(df_final)
        delete_groupby_state(mrnf_state)
        return pd.concat(out_dfs)

    check_func(
        test_mrnf,
        (df,),
        py_output=expected_df,
        check_dtype=False,
        sort_output=True,
        reset_index=True,
        convert_columns_to_pandas=True,
    )


def get_random_col(col_type: str, nrows: int) -> pd.Series:
    """
    Helper function to generate a random column
    of the specified type. This is used for
    generating the partition/order-by columns.
    We generate columns with only a few unique
    values to ensure that there are groups
    with multiple rows and the order-by has
    overlaps.

    Args:
        col_type (str): Column type to generate.
            Supported values are: "Int64", "string",
            "int32", "bool", "Float64" and "timestamp".
        nrows (int): Number of rows to put in the output.

    Returns:
        pd.Series: Column of specified type with 'nrows' rows.
    """
    # Fix the seed for deterministic data generation across ranks.
    np.random.seed(234)
    if col_type == "Int64":
        choices = [pd.NA, 45, 23, 67, 234, 0, -4, -10]
        s = pd.Series(np.random.choice(choices, nrows), dtype="Int64")
    elif col_type == "string":
        choices = [pd.NA, "apple", "pie", "egg", "salad", "banana"]
        s = pd.Series(np.random.choice(choices, nrows))
    elif col_type == "int32":
        choices = [-45, 267, 90, 1000, 0, -4, -10]
        s = pd.Series(np.random.choice(choices, nrows), dtype="int32")
    elif col_type == "bool":
        s = pd.Series(np.random.choice([True, False], nrows), dtype="bool")
    elif col_type == "Float64":
        choices = [pd.NA, 4.5, 23.43, 67.64, 234.0, 0.32, -4.43, -10.98]
        s = pd.Series(np.random.choice(choices, nrows), dtype="Float64")
    elif col_type == "timestamp":
        choices = [
            pd.NaT,
            pd.Timestamp(year=2024, month=1, day=1, second=5),
            pd.Timestamp(year=1992, month=11, day=5, minute=34, second=52),
            pd.Timestamp(year=2025, month=6, day=9),
            pd.Timestamp(year=2024, month=6, day=9),
        ]
        s = pd.Series(np.random.choice(choices, nrows))
    else:
        raise ValueError(f"Unsupported partition column type: {col_type}.")
    return s


@pytest.mark.slow
@pytest.mark.parametrize(
    "mrnf_sort_col_asc",
    [
        pytest.param((False, True), id="desc_asc"),
        pytest.param((True, False), marks=pytest.mark.slow, id="asc_desc"),
    ],
)
@pytest.mark.parametrize(
    "mrnf_sort_col_na",
    [
        pytest.param((True, False), id="na_last_first"),
        pytest.param((False, True), marks=pytest.mark.slow, id="na_first_last"),
    ],
)
@pytest.mark.parametrize(
    "partition_col_dtypes",
    [
        pytest.param(["Int64", "string"], id="part_nullable_int_string"),
        pytest.param(["int32", "bool"], marks=pytest.mark.slow, id="part_np_int_bool"),
        pytest.param(
            ["Float64", "timestamp"],
            marks=pytest.mark.slow,
            id="part_nullable_float_timestamp",
        ),
    ],
)
@pytest.mark.parametrize(
    "sort_col_dtypes",
    [
        pytest.param(
            ["Int64", "string"],
            id="sort_nullable_int_string",
        ),
        pytest.param(["int32", "bool"], marks=pytest.mark.slow, id="sort_np_int_bool"),
        pytest.param(
            ["Float64", "timestamp"],
            marks=pytest.mark.slow,
            id="sort_nullable_float_timestamp",
        ),
    ],
)
def test_mrnf_basic(
    mrnf_sort_col_asc: MetaType,
    mrnf_sort_col_na: MetaType,
    partition_col_dtypes: list,
    sort_col_dtypes: list,
    memory_leak_check,
):
    """
    Basic tests for streaming MRNF functionality. We test various
    data types (for both the partition and order-by columns) and
    sort directions.
    We use multiple different dtypes for the "data" columns
    (i.e. non-partition non-order-by columns) to verify correct
    pass-through behavior for many different types.
    """
    mrnf_sort_col_asc = MetaType(mrnf_sort_col_asc)
    mrnf_sort_col_na = MetaType(mrnf_sort_col_na)
    n_cols = 10
    col_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    assert len(col_names) == n_cols
    # Partition columns
    keys_inds = MetaType((3, 5))
    # Sort columns
    mrnf_sort_col_inds = MetaType((6, 1))
    fnames = MetaType(("min_row_number_filter",))
    f_in_cols = MetaType((0, 1, 2, 4, 6, 7, 8, 9))
    f_in_offsets = MetaType((0, n_cols - 2))
    batch_size = 40
    input_table_kept_cols = MetaType(tuple(range(n_cols)))

    mrnf_col_inds_keep = MetaType((0, 2, 3, 4, 6, 7, 8, 9))
    output_table_col_meta = ColNamesMetaType(
        tuple(np.take(col_names, mrnf_col_inds_keep.meta))
    )

    nrows = 403
    part_cols = []
    sort_cols = []

    for col_type in partition_col_dtypes:
        part_cols.append(get_random_col(col_type, nrows))

    for col_type in sort_col_dtypes:
        sort_cols.append(get_random_col(col_type, nrows))

    # We will set the data columns to the same value in the entire array
    # to avoid non-deterministic behavior during comparison.
    df = pd.DataFrame(
        {
            # Nullable int64 data column:
            "A": pd.Series([1] * nrows, dtype="Int64"),
            # Sort column 2:
            "B": sort_cols[1],
            # String data column
            "C": pd.Series(["abc"] * nrows),
            # PARTITION COLUMN 1:
            "D": part_cols[0],
            # Non-nullable int32 data column
            "E": pd.Series([65] * nrows, dtype="int32"),
            # PARTITION COLUMN 2:
            "F": part_cols[1],
            # Sort column 1:
            "G": sort_cols[0],
            # Float data column:
            "H": pd.Series([67.32] * nrows, dtype="float32"),
            # Bool data column:
            "I": pd.Series([True] * nrows, dtype="bool"),
            # Timestamp data column:
            "J": pd.Series([pd.Timestamp(year=2024, month=1, day=1, second=5)] * nrows),
        }
    )

    # We use a bodo function since Python cannot take a list for "na".
    @bodo.jit(distributed=False)
    def py_mrnf(x: pd.DataFrame, asc: list[bool], na: list[str]):
        return x.sort_values(by=["G", "B"], ascending=asc, na_position=na)

    expected_df = df.groupby(["D", "F"], as_index=False, dropna=False).apply(
        lambda x: py_mrnf(
            x,
            list(mrnf_sort_col_asc.meta),
            [("last" if f else "first") for f in list(mrnf_sort_col_na.meta)],
        ).iloc[0]
    )[list(output_table_col_meta.meta)]

    _test_mrnf_helper(
        df,
        expected_df,
        keys_inds,
        fnames,
        f_in_offsets,
        f_in_cols,
        mrnf_sort_col_inds,
        mrnf_sort_col_asc,
        mrnf_sort_col_na,
        mrnf_col_inds_keep,
        input_table_kept_cols,
        batch_size,
        output_table_col_meta,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "mrnf_col_inds_keep",
    [
        pytest.param(
            (0, 1, 2, 3, 4, 5, 6), marks=pytest.mark.slow, id="keep_all"
        ),  # Keep all columns
        pytest.param(
            (0, 2, 3, 4), marks=pytest.mark.slow, id="skip_sort"
        ),  # Keep partition col, skip all sort columns
        pytest.param(
            (0, 1, 2, 4, 5, 6), marks=pytest.mark.slow, id="skip_part"
        ),  # Keep all sort cols, skip the partition column
        pytest.param(
            (0, 2, 3, 4, 6),
            id="one_of_each",
        ),  # Keep one of each
        pytest.param(
            (0, 2, 4), marks=pytest.mark.slow, id="skip_all"
        ),  # Keep none of the partition/sort columns
    ],
)
def test_mrnf_skipped_cols(mrnf_col_inds_keep, memory_leak_check):
    """
    Test that skipping partition-cols and order-by columns in the
    output works as expected. We test various settings for full
    coverage.
    """
    mrnf_col_inds_keep = MetaType(mrnf_col_inds_keep)
    n_cols = 7
    col_names = ["A", "B", "C", "D", "E", "F", "G"]
    assert len(col_names) == n_cols
    # Partition columns
    keys_inds = MetaType((3,))
    # Sort columns
    mrnf_sort_col_inds = MetaType((6, 1, 5))
    fnames = MetaType(("min_row_number_filter",))
    f_in_cols = MetaType((0, 1, 2, 4, 5, 6))
    f_in_offsets = MetaType((0, n_cols - len(keys_inds.meta)))
    batch_size = 40
    input_table_kept_cols = MetaType(tuple(range(n_cols)))
    output_table_col_meta = ColNamesMetaType(
        tuple(np.take(col_names, mrnf_col_inds_keep.meta))
    )

    nrows = 103
    part_cols = []
    sort_cols = []

    for col_type in ["Int64"]:
        part_cols.append(get_random_col(col_type, nrows))

    for col_type in ["Int64", "string", "Float64"]:
        sort_cols.append(get_random_col(col_type, nrows))

    # We will set the data columns to the same value in the entire array
    # to avoid non-deterministic behavior during comparison.
    df = pd.DataFrame(
        {
            # Nullable int64 data column:
            "A": pd.Series([1] * nrows, dtype="Int64"),
            # Sort column 2:
            "B": sort_cols[1],
            # String data column
            "C": pd.Series(["abc"] * nrows),
            # PARTITION COLUMN 1:
            "D": part_cols[0],
            # Non-nullable int32 data column
            "E": pd.Series([65] * nrows, dtype="int32"),
            # SORT COLUMN 3:
            "F": sort_cols[2],
            # Sort column 1:
            "G": sort_cols[0],
        }
    )

    mrnf_sort_col_asc = MetaType((True, True, True))
    mrnf_sort_col_na = MetaType((False, False, False))

    @bodo.jit(distributed=False)
    def py_mrnf(x: pd.DataFrame, asc: list[bool], na: list[bool]):
        return x.sort_values(by=["G", "B", "F"], ascending=asc, na_position=na)

    expected_df = df.groupby(["D"], as_index=False, dropna=False).apply(
        lambda x: py_mrnf(
            x,
            list(mrnf_sort_col_asc.meta),
            [("last" if f else "first") for f in list(mrnf_sort_col_na.meta)],
        ).iloc[0]
    )[list(output_table_col_meta.meta)]

    _test_mrnf_helper(
        df,
        expected_df,
        keys_inds,
        fnames,
        f_in_offsets,
        f_in_cols,
        mrnf_sort_col_inds,
        mrnf_sort_col_asc,
        mrnf_sort_col_na,
        mrnf_col_inds_keep,
        input_table_kept_cols,
        batch_size,
        output_table_col_meta,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "data_col",
    [
        pytest.param(
            pd.Series(
                [
                    {11: ["A"], 21: ["B", None], 9: ["C"]},
                ]
                * 103,
                dtype=pd.ArrowDtype(pa.map_(pa.int64(), pa.large_list(pa.string()))),
            ),
            id="map",
        ),
        pytest.param(
            pd.array(
                [
                    {
                        "X": "C",
                        "Y": [1.1],
                        "Z": [[11], None],
                        "W": {"A": 1, "B": "ABC"},
                        "Q": None,
                    },
                ]
                * 103,
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [
                            pa.field("X", pa.string()),
                            pa.field("Y", pa.large_list(pa.float64())),
                            pa.field("Z", pa.large_list(pa.large_list(pa.int64()))),
                            pa.field(
                                "W",
                                pa.struct(
                                    [
                                        pa.field("A", pa.int8()),
                                        pa.field("B", pa.string()),
                                    ]
                                ),
                            ),
                            pa.field("Q", pa.large_list(pa.string())),
                        ]
                    )
                ),
            ),
            id="struct",
        ),
        pytest.param(
            pd.array(
                [["A", None, "B"]] * 103,
                dtype=pd.ArrowDtype(pa.large_list(pa.string())),
            ),
            id="array",
        ),
    ],
)
def test_mrnf_nested_array_data(data_col, memory_leak_check):
    """
    Test that MRNF works as expected when some of the "data"
    columns are semi-structured data type.
    """
    n_cols = 3
    col_names = ["A", "B", "C"]
    assert len(col_names) == n_cols
    # Partition columns
    keys_inds = MetaType((1,))
    # Sort columns
    mrnf_sort_col_inds = MetaType((0,))
    fnames = MetaType(("min_row_number_filter",))
    f_in_cols = MetaType((0, 2))
    f_in_offsets = MetaType((0, n_cols - 1))
    batch_size = 40
    input_table_kept_cols = MetaType(tuple(range(n_cols)))
    mrnf_col_inds_keep = MetaType((0, 2))
    output_table_col_meta = ColNamesMetaType(
        tuple(np.take(col_names, mrnf_col_inds_keep.meta))
    )
    nrows = 103

    df = pd.DataFrame(
        {
            "A": get_random_col("string", nrows),
            "B": get_random_col("Int64", nrows),
            "C": data_col,
        }
    )
    mrnf_sort_col_asc = MetaType((True,))
    mrnf_sort_col_na = MetaType((False,))

    @bodo.jit(distributed=False)
    def py_mrnf(x: pd.DataFrame):
        return x.sort_values(by="A", ascending=True, na_position="first")

    expected_df = df.groupby("B", as_index=False, dropna=False).apply(
        lambda x: py_mrnf(x).iloc[0]
    )[list(output_table_col_meta.meta)]

    _test_mrnf_helper(
        df,
        expected_df,
        keys_inds,
        fnames,
        f_in_offsets,
        f_in_cols,
        mrnf_sort_col_inds,
        mrnf_sort_col_asc,
        mrnf_sort_col_na,
        mrnf_col_inds_keep,
        input_table_kept_cols,
        batch_size,
        output_table_col_meta,
    )


@pytest.mark.parametrize(
    "key_col_choices, dtype",
    [
        pytest.param(
            [{1: 1.4, 2: 3.1}, None, {}],
            pd.ArrowDtype(pa.map_(pa.int64(), pa.float64())),
            id="map",
        ),
        pytest.param(
            [{"a": "xyz", "b": "abc"}, None, {}],
            pd.ArrowDtype(
                pa.struct([pa.field("a", pa.string()), pa.field("b", pa.string())])
            ),
            id="struct",
        ),
        pytest.param(
            [[{}, None, {1: 1, 2: 2}], [{3: 3}], None],
            pd.ArrowDtype(pa.large_list(pa.map_(pa.int64(), pa.int64()))),
            id="list_of_map",
        ),
    ],
)
def test_mrnf_nested_array_key(key_col_choices, dtype, memory_leak_check):
    """
    Test that MRNF works correctly when the partition-by column
    is a semi-structured data type.
    """
    n_cols = 3
    col_names = ["A", "B", "C"]
    assert len(col_names) == n_cols
    # Partition columns
    keys_inds = MetaType((0,))
    # Sort columns
    mrnf_sort_col_inds = MetaType((1,))
    fnames = MetaType(("min_row_number_filter",))
    f_in_cols = MetaType((1, 2))
    f_in_offsets = MetaType((0, n_cols - 1))
    batch_size = 40
    input_table_kept_cols = MetaType(tuple(range(n_cols)))
    mrnf_col_inds_keep = MetaType((0, 1, 2))
    output_table_col_meta = ColNamesMetaType(
        tuple(np.take(col_names, mrnf_col_inds_keep.meta))
    )
    nrows = 48

    df = pd.DataFrame(
        {
            "A": pd.Series(key_col_choices * 16, dtype=dtype),
            "B": pd.Series([10] * nrows, dtype="Int64"),
            "C": pd.Series(["abc"] * nrows),
        }
    )
    mrnf_sort_col_asc = MetaType((True,))
    mrnf_sort_col_na = MetaType((False,))

    expected_df = pd.DataFrame(
        {
            "A": pd.Series(key_col_choices, dtype=dtype),
            "B": pd.Series([10] * 3, dtype="Int64"),
            "C": pd.Series(["abc"] * 3),
        }
    )
    expected_df = expected_df[list(output_table_col_meta.meta)]

    _test_mrnf_helper(
        df,
        expected_df,
        keys_inds,
        fnames,
        f_in_offsets,
        f_in_cols,
        mrnf_sort_col_inds,
        mrnf_sort_col_asc,
        mrnf_sort_col_na,
        mrnf_col_inds_keep,
        input_table_kept_cols,
        batch_size,
        output_table_col_meta,
    )


def test_data_col_correctness(memory_leak_check):
    """
    Test that the output is correct with respect to the
    data columns. This is done by having unique values
    in the data and order-by columns so that the data column
    output is non-trivial but still deterministic
    (unlike the other tests where the data column
    has the same value in all rows to avoid ambiguity
    in the output).
    """
    nrows = 100
    df = pd.DataFrame(
        {
            # Unique data column:
            "A": pd.Series(np.arange(nrows), dtype="Int64"),
            # Partition Col:
            "B": get_random_col("string", nrows),
            # Order-by column
            "C": pd.Series([f"{i}" for i in range(nrows)]),
        }
    )

    def py_mrnf(x: pd.DataFrame):
        return x.sort_values(by="C", ascending=True, na_position="last").iloc[0]

    expected_df = df.groupby(["B"], as_index=False, dropna=False).apply(py_mrnf)
    expected_df = expected_df[["A", "B", "C"]]

    _test_mrnf_helper(
        df,
        expected_df,
        MetaType((1,)),
        MetaType(("min_row_number_filter",)),
        MetaType((0, 2)),
        MetaType((0, 2)),
        MetaType((2,)),
        MetaType((True,)),
        MetaType((True,)),
        MetaType((0, 1, 2)),
        MetaType((0, 1, 2)),
        40,
        ColNamesMetaType(tuple(df.columns)),
    )


def test_mrnf_no_order(memory_leak_check):
    """
    Test that the output is correct when there are no orderby columns.
    """
    nrows = 1000
    lengths = np.abs(np.round(np.tan(np.arange(nrows)))).astype(np.int32) % 20
    df = pd.DataFrame(
        {
            # Partition Col:
            "A": ["ABCDEFGHIJKLMNOPQRST"[:length] for length in lengths],
            # Data column (only 1 distinct value per partition):
            "B": [int(str(length) * ((length) % 4 + 1)) for length in lengths],
        }
    )

    def py_mrnf(x: pd.DataFrame):
        return x.iloc[0]

    expected_df = df.groupby(["B"], as_index=False, dropna=False).apply(py_mrnf)
    expected_df = expected_df[["A", "B"]]

    _test_mrnf_helper(
        df,
        expected_df,
        MetaType((0,)),
        MetaType(("min_row_number_filter",)),
        MetaType((0, 1)),
        MetaType((1,)),
        MetaType(()),
        MetaType(()),
        MetaType(()),
        MetaType((0, 1)),
        MetaType((0, 1)),
        40,
        ColNamesMetaType(tuple(df.columns)),
    )


@pytest_mark_one_rank
def test_mrnf_err_handling(memory_leak_check):
    """
    Test that the expected compiler validation errors are
    raised as expected when the inputs are invalid.
    """
    n_cols = 3
    col_names = ["A", "B", "C"]
    assert len(col_names) == n_cols
    keys_inds = MetaType((0,))
    mrnf_sort_col_inds = MetaType((1,))
    fnames = MetaType(("min_row_number_filter",))
    f_in_cols = MetaType((1, 2))
    f_in_offsets = MetaType((0, n_cols - 1))
    batch_size = 40
    input_table_kept_cols = MetaType(tuple(range(n_cols)))
    mrnf_col_inds_keep = MetaType((0, 1, 2))
    output_table_col_meta = ColNamesMetaType(
        tuple(np.take(col_names, mrnf_col_inds_keep.meta))
    )
    nrows = 10

    df = pd.DataFrame(
        {
            "A": get_random_col("string", nrows),
            "B": get_random_col("Int64", nrows),
            "C": pd.Series([10] * nrows, dtype="Int64"),
        }
    )
    mrnf_sort_col_asc = MetaType((True,))
    mrnf_sort_col_na = MetaType((False,))

    def test_mrnf(df):
        mrnf_state = init_groupby_state(
            -1,
            keys_inds,
            fnames,
            f_in_offsets,
            f_in_cols,
            mrnf_sort_col_inds,
            mrnf_sort_col_asc,
            mrnf_sort_col_na,
            mrnf_col_inds_keep,
        )

        is_last1 = False
        _iter_1 = 0
        T1 = bodo.hiframes.table.logical_table_to_table(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df),
            (),
            input_table_kept_cols,
            n_cols,
        )
        _temp1 = bodo.hiframes.table.local_len(T1)
        while not is_last1:
            T2 = bodo.hiframes.table.table_local_filter(
                T1, slice((_iter_1 * batch_size), ((_iter_1 + 1) * batch_size))
            )
            is_last1 = (_iter_1 * batch_size) >= _temp1
            T3 = bodo.hiframes.table.table_subset(T2, input_table_kept_cols, False)
            _iter_1 = _iter_1 + 1
            is_last1, _ = groupby_build_consume_batch(mrnf_state, T3, is_last1, True)
        out_dfs = []
        is_last2 = False
        while not is_last2:
            out_table, is_last2 = groupby_produce_output_batch(mrnf_state, True)
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(out_table), 1, None
            )
            df_final = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (out_table,), index_var, output_table_col_meta
            )
            out_dfs.append(df_final)
        delete_groupby_state(mrnf_state)
        return pd.concat(out_dfs)

    ## 1. Verify that an error is raised when a column is both a partition
    ## and orderby column.
    keys_inds = MetaType((0, 1))
    mrnf_sort_col_inds = MetaType((1,))
    with pytest.raises(
        BodoError,
        match=re.escape(
            "Groupby (Min Row-Number Filter): A column cannot be both a partition column and a sort column."
        ),
    ):
        bodo.jit((bodo.typeof(df),))(test_mrnf)
    # Restore:
    keys_inds = MetaType((0,))
    mrnf_sort_col_inds = MetaType((1,))

    ## 2. More than one fnames
    fnames = MetaType(("min_row_number_filter", "sum"))
    f_in_cols = MetaType((1, 2, 2))
    f_in_offsets = MetaType((0, 2, 3))
    with pytest.raises(
        BodoError,
        match=re.escape(
            "Streaming Groupby: Min Row-Number Filter cannot be combined with other aggregation functions."
        ),
    ):
        bodo.jit((bodo.typeof(df),))(test_mrnf)
    # Restore:
    fnames = MetaType(("min_row_number_filter",))
    f_in_cols = MetaType((1, 2))
    f_in_offsets = MetaType((0, n_cols - 1))

    ## 3. f_in_cols doesn't have all the columns
    f_in_cols = MetaType((2,))
    with pytest.raises(
        BodoError,
        match=re.escape(
            "Groupby (Min Row-Number Filter): All columns except the partition columns must be in f_in_cols!"
        ),
    ):
        bodo.jit((bodo.typeof(df),))(test_mrnf)

    # Restore:
    f_in_cols = MetaType((1, 2))

    ## 4. f_in_offsets isn't what's expected
    f_in_offsets = MetaType((0, n_cols - 2, n_cols - 1))
    with pytest.raises(
        BodoError,
        match=re.escape(
            "Groupby (Min Row-Number Filter): Expected f_in_offsets to be '[0, 2]', but got '[0, 1, 2]' instead"
        ),
    ):
        bodo.jit((bodo.typeof(df),))(test_mrnf)

    # Restore:
    f_in_offsets = MetaType((0, n_cols - 1))

    ## 5. len(mrnf_sort_col_asc) isn't correct
    mrnf_sort_col_asc = MetaType((True, False))
    with pytest.raises(
        BodoError,
        match=re.escape(
            "Groupby (Min Row-Number Filter): Mismatch in expected sizes of arguments!"
        ),
    ):
        bodo.jit((bodo.typeof(df),))(test_mrnf)

    # Restore:
    mrnf_sort_col_asc = MetaType((True,))

    ## 6. len(mrnf_sort_col_na) isn't correct
    mrnf_sort_col_na = MetaType((False, True))
    with pytest.raises(
        BodoError,
        match=re.escape(
            "Groupby (Min Row-Number Filter): Mismatch in expected sizes of arguments!"
        ),
    ):
        bodo.jit((bodo.typeof(df),))(test_mrnf)

    # Restore:
    mrnf_sort_col_na = MetaType((False,))

    ## 7. mrnf_col_inds_keep doesn't have a data column
    mrnf_col_inds_keep = MetaType((0,))
    with pytest.raises(
        BodoError,
        match=re.escape(
            "Groupby (Min Row-Number Filter): Column 2 must be in the list of indices to keep "
            "since it's neither a partition columns nor a sort column!"
        ),
    ):
        bodo.jit((bodo.typeof(df),))(test_mrnf)

    # Restore:
    mrnf_col_inds_keep = MetaType((0, 1, 2))

    ## 8. Using a semi-structured sort array
    semi_df = pd.DataFrame(
        {
            "A": get_random_col("string", nrows),
            "B": pd.Series(
                [
                    {11: ["A"], 21: ["B", None], 9: ["C"]},
                ]
                * nrows,
                dtype=pd.ArrowDtype(pa.map_(pa.int64(), pa.large_list(pa.string()))),
            ),
            "C": pd.Series([10] * nrows, dtype="Int64"),
        }
    )
    with pytest.raises(
        BodoError,
        match=re.escape(
            "Groupby (Min Row-Number Filter): Sorting on semi-structured arrays is not supported."
        ),
    ):
        bodo.jit((bodo.typeof(semi_df),))(test_mrnf)


@pytest.mark.parametrize(
    "sort_col_dtypes",
    [
        pytest.param(["bool"], id="bool"),
        pytest.param(["int32"], marks=pytest.mark.slow, id="int32"),
        pytest.param(["Int64"], marks=pytest.mark.slow, id="Int64"),
        pytest.param(["string"], marks=pytest.mark.slow, id="string"),
        pytest.param(["bool", "Int64"], marks=pytest.mark.slow, id="bool_Int64"),
        pytest.param(
            ["string", "int32", "bool"], marks=pytest.mark.slow, id="string_int32_bool"
        ),
    ],
)
@pytest.mark.parametrize(
    "sort_na_last",
    [
        pytest.param(False, id="na_first"),
        pytest.param(True, marks=pytest.mark.slow, id="na_last"),
    ],
)
@pytest.mark.parametrize(
    "sort_asc",
    [
        pytest.param(False, id="desc"),
        pytest.param(True, marks=pytest.mark.slow, id="asc"),
    ],
)
@pytest.mark.parametrize(
    "all_na",
    [
        pytest.param(False, id="not_all_na"),
        pytest.param(True, id="all_na"),
    ],
)
def test_mrnf_all_ties(
    sort_col_dtypes: list[str],
    all_na: bool,
    sort_na_last: bool,
    sort_asc: bool,
    memory_leak_check,
):
    """
    Test that the output is correct even when all the sort values are the
    same, i.e. it's all ties.
    """

    # Construct a dataframe with the specified sort column(s):
    df_cols = {"A": pd.array(list(np.arange(10)) * 10, dtype="Int32")}
    for i, col_dtype in enumerate(sort_col_dtypes):
        if col_dtype == "bool":
            if all_na:
                df_cols[f"orderby_{i}"] = pd.Series([pd.NA] * 100, dtype="boolean")
            else:
                df_cols[f"orderby_{i}"] = pd.Series([True] * 100, dtype="bool")
        elif col_dtype == "Int64":
            if all_na:
                df_cols[f"orderby_{i}"] = pd.Series([pd.NA] * 100, dtype="Int64")
            else:
                df_cols[f"orderby_{i}"] = pd.Series([657] * 100, dtype="Int64")
        elif col_dtype == "string":
            if all_na:
                df_cols[f"orderby_{i}"] = pd.Series([None] * 100, dtype="string")
            else:
                df_cols[f"orderby_{i}"] = pd.Series(["abc"] * 100, dtype="string")
        elif col_dtype == "int32":
            if all_na:
                df_cols[f"orderby_{i}"] = pd.Series([pd.NA] * 100, dtype="Int32")
            else:
                df_cols[f"orderby_{i}"] = pd.Series([907] * 100, dtype="int32")
        else:
            raise ValueError(f"Unsupported dtype: {col_dtype}!")
    df_cols["PT"] = pd.Series(["train"] * 100)

    # Construct the MRNF args:
    df = pd.DataFrame(df_cols)
    n_cols = len(df.columns)
    col_names = list(df.columns)
    n_sort_cols = len(sort_col_dtypes)
    sort_cols_list = [f"orderby_{i}" for i in range(n_sort_cols)]
    key_inds_list = [0]
    sort_inds_list = [i + 1 for i in range(n_sort_cols)]
    sort_asc_list = [sort_asc] * n_sort_cols
    sort_na_list = [sort_na_last] * n_sort_cols
    keep_inds_list = list(range(n_cols))
    f_in_cols_list = list(range(1, n_cols))
    f_in_offsets_list = [0, n_cols - 1]

    def py_mrnf(x: pd.DataFrame):
        return x.sort_values(
            by=sort_cols_list,
            ascending=sort_asc_list,
            na_position=("last" if sort_na_last else "first"),
        ).iloc[0]

    expected_df = df.groupby(["A"], as_index=False, dropna=False).apply(py_mrnf)
    expected_df = expected_df[list(np.take(list(df.columns), keep_inds_list))]

    _test_mrnf_helper(
        df,
        expected_df,
        MetaType(tuple(key_inds_list)),
        MetaType(("min_row_number_filter",)),
        MetaType(tuple(f_in_offsets_list)),
        MetaType(tuple(f_in_cols_list)),
        MetaType(tuple(sort_inds_list)),
        MetaType(tuple(sort_asc_list)),
        MetaType(tuple(sort_na_list)),
        MetaType(tuple(keep_inds_list)),
        MetaType(tuple(range(n_cols))),
        20,
        ColNamesMetaType(tuple(np.take(col_names, keep_inds_list))),
    )


@temp_env_override({"BODO_STREAM_LOOP_SYNC_ITERS": "1"})
@pytest.mark.parametrize(
    "n_unique_keys",
    [
        pytest.param(2, id="few_unique_local_reduction"),
        pytest.param(100, id="many_unique_no_local_reduction"),
    ],
)
@pytest.mark.skipif(
    bodo.get_size() == 1,
    reason="Test for shuffle behavior only that only happens when size > 1",
)
def test_mrnf_shuffle_reduction(n_unique_keys, memory_leak_check):
    """
    Test that the MRNF works correctly when there are many unique keys
    and the local reduction is disabled as well as when there are few
    unique keys and the local reduction is enabled.
    """
    nrows = 1000
    np.random.seed(234)
    keys = np.random.choice(list(range(n_unique_keys)), nrows)
    df = pd.DataFrame(
        {
            "A": pd.Series(keys, dtype="Int64"),
            "B": pd.Series(list(range(nrows)), dtype="Int64"),
        }
    )
    n_cols = len(df.columns)
    col_names = list(df.columns)
    n_sort_cols = 1
    sort_cols_list = ["B"]
    sort_inds_list = [i + 1 for i in range(n_sort_cols)]
    sort_asc_list = [True] * n_sort_cols
    sort_na_list = [True] * n_sort_cols
    keep_inds_list = list(range(n_cols))
    f_in_cols_list = list(range(1, n_cols))
    f_in_offsets_list = [0, n_cols - 1]

    def py_mrnf(x: pd.DataFrame):
        return x.sort_values(
            by=sort_cols_list,
            ascending=sort_asc_list,
            na_position=("last"),
        ).iloc[0]

    expected_df = df.groupby(["A"], as_index=False, dropna=False).apply(py_mrnf)
    expected_df = expected_df[list(np.take(list(df.columns), keep_inds_list))]
    _test_mrnf_helper(
        df,
        expected_df,
        MetaType((0,)),
        MetaType(("min_row_number_filter",)),
        MetaType(tuple(f_in_offsets_list)),
        MetaType(tuple(f_in_cols_list)),
        MetaType(tuple(sort_inds_list)),
        MetaType(tuple(sort_asc_list)),
        MetaType(tuple(sort_na_list)),
        MetaType(tuple(keep_inds_list)),
        MetaType(tuple(range(n_cols))),
        20,
        ColNamesMetaType(tuple(np.take(col_names, keep_inds_list))),
    )
