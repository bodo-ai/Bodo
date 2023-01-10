import datetime
import io
import os
import re
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from mpi4py import MPI
from numba import types
from pandas.core.dtypes.common import is_list_like

import bodo
from bodo.tests.dataframe_common import df_value  # noqa
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    SeriesOptTestPipeline,
    _ensure_func_calls_optimized_out,
    _get_dist_arg,
    _test_equal_guard,
    check_func,
    reduce_sum,
)
from bodo.utils.typing import BodoError


def test_dataframe_apply_method_str(memory_leak_check):
    """
    Test running DataFrame.apply with a string literal that
    matches a DataFrame method. Note by default all of these
    will run with axis=0 if the argument exists.

    """

    def impl1(df):
        # Test a DataFrame method that returns a Series without axis=1.
        return df.apply("nunique")

    def impl2(df):
        # Test a DataFrame method that conflicts with a numpy function
        return df.apply("sum", axis=1)

    def impl3(df):
        # Test a DataFrame method that returns a DataFrame
        return df.apply("abs")

    df = pd.DataFrame(
        {
            "A": np.arange(100) % 10,
            "B": -np.arange(100),
        }
    )

    check_func(impl1, (df,), is_out_distributed=False)
    check_func(impl2, (df,))
    check_func(impl3, (df,))


@pytest.mark.skip("[BE-1198] Support numpy ufuncs on DataFrames")
def test_dataframe_apply_numpy_str(memory_leak_check):
    """
    Test running dataframe.apply with a string literal that
    matches a Numpy function.
    """

    def impl1(df):
        return df.apply("sin")

    def impl2(df):
        # Test with axis=1 (unused)
        return df.apply("log", axis=1)

    df = pd.DataFrame(
        {
            "A": np.arange(100) % 10,
            "B": -np.arange(100),
        }
    )

    check_func(impl1, (df,))
    check_func(impl2, (df,))


@pytest.mark.slow
def test_dataframe_apply_no_func(memory_leak_check):
    """
    Test running dataframe.apply with a string literal that
    doesn't match a method or Numpy function raises an
    Exception.
    """

    def impl1(df):
        # This function doesn't exist in Numpy or as a
        # DataFrame method.
        return df.apply("concat", axis=1)

    df = pd.DataFrame(
        {
            "A": np.arange(100) % 10,
            "B": -np.arange(100),
        }
    )
    with pytest.raises(BodoError, match="user-defined function not supported"):
        bodo.jit(impl1)(df)


@pytest.mark.slow
def test_dataframe_apply_pandas_unsupported_method(memory_leak_check):
    """
    Test running dataframe.apply with a string literal that
    matches an unsupported DataFrame method raises an appropriate
    exception.
    """

    def impl1(df):
        return df.apply("argmin", axis=1)

    df = pd.DataFrame(
        {
            "A": np.arange(100) % 10,
            "B": -np.arange(100),
        }
    )
    with pytest.raises(BodoError, match="user-defined function not supported"):
        bodo.jit(impl1)(df)


@pytest.mark.slow
def test_dataframe_apply_numpy_unsupported_ufunc(memory_leak_check):
    """
    Test running dataframe.apply with a string literal that
    matches an unsupported ufunc raises an appropriate
    exception.
    """

    def impl1(df):
        return df.apply("cbrt", axis=1)

    df = pd.DataFrame(
        {
            "A": np.arange(100) % 10,
            "B": -np.arange(100),
        }
    )
    with pytest.raises(BodoError, match="user-defined function not supported"):
        bodo.jit(impl1)(df)


@pytest.mark.slow
def test_dataframe_apply_pandas_unsupported_type(memory_leak_check):
    """
    Test running dataframe.apply with a string literal that
    matches a method but has an unsupported type
    raises an appropriate exception.
    """

    def impl1(df):
        # Mean is unsupported for string types
        return df.apply("mean", axis=1)

    df = pd.DataFrame(
        {
            "A": ["ABC", "21", "231", "21dwcp"] * 25,
            "B": ["feq", "3243412rfe", "fonie wqw   ", "3c", "r32r23fc"] * 20,
        }
    )
    with pytest.raises(BodoError, match="user-defined function not supported"):
        bodo.jit(impl1)(df)


@pytest.mark.slow
def test_dataframe_apply_pandas_unsupported_axis(memory_leak_check):
    """
    Test running dataframe.apply with a method using
    axis=1 when Bodo doesn't support axis=1 yet.
    """

    def impl1(df):
        # nunique is unsupported for axis=1
        return df.apply("nunique", axis=1)

    df = pd.DataFrame(
        {
            "A": ["ABC", "21", "231", "21dwcp"] * 25,
            "B": -np.arange(100),
        }
    )
    with pytest.raises(BodoError, match="user-defined function not supported"):
        bodo.jit(impl1)(df)


@pytest.mark.slow
def test_dataframe_apply_numpy_unsupported_type(memory_leak_check):
    """
    Test running dataframe.apply with a string literal that
    matches a Numpy ufunc but has an unsupported type
    raises an appropriate exception.
    """

    def impl1(df):
        # radians is unsupported for string types
        return df.apply("radians", axis=1)

    df = pd.DataFrame(
        {
            "A": ["ABC", "21", "231", "21dwcp"] * 25,
            "B": -np.arange(100),
        }
    )
    with pytest.raises(BodoError, match="user-defined function not supported"):
        bodo.jit(impl1)(df)


def test_dataframe_optional_scalar(memory_leak_check):
    """
    Test calling pd.DataFrame with a scalar that is an optional type.
    """

    def impl(table1):
        df1 = pd.DataFrame({"A": table1["A"], "$f3": table1["A"] == np.int32(1)})
        S0 = df1["A"][df1["$f3"]]
        df2 = pd.DataFrame(
            {"col1_sum_a": S0.sum() if len(S0) > 0 else None},
            index=pd.RangeIndex(0, 1, 1),
        )
        return df2

    df = pd.DataFrame({"A": [1, 2, 3] * 4})

    # Pandas can avoid nullable so the types don't match
    check_func(impl, (df,), check_dtype=False)


def test_dataframe_is_none(memory_leak_check):
    """
    Test that dataframe is None can compile and keep the dataframe distributed.
    """

    def impl1(df):
        return df is None

    def impl2(df):
        return df is not None

    df = pd.DataFrame(
        {
            "A": np.arange(100) % 10,
            "B": -np.arange(100),
        }
    )
    check_func(impl1, (df,))
    check_func(impl2, (df,))
    # Test that none still works
    check_func(impl1, (None,))
    check_func(impl2, (None,))


@pytest.mark.slow
def test_describe_many_columns(memory_leak_check):
    """
    Runs df.describe on a dataframe with 25 columns.
    If df.describe is inlined, this will take more than
    1 hour.
    If df.describe is not inlined as expected, this should take
    closer to 12 seconds. To avoid failures due to small variations,
    we will check this tests take no longer than 1 minute.
    """

    def impl(df):
        return df.describe()

    df = pd.DataFrame(
        columns=np.arange(25), data=np.arange(1000 * 25).reshape(1000, 25)
    )
    import time

    t0 = time.time()
    check_func(impl, (df,), is_out_distributed=False)
    compilation_time = time.time() - t0
    # Determine the max compilation time on any rank to avoid hangs.
    comm = MPI.COMM_WORLD
    compilation_time = comm.allreduce(compilation_time, op=MPI.MAX)
    assert (
        compilation_time < 60
    ), "df.describe() took too long to compile. Possible regression?"


@pytest.mark.parametrize(
    "dt_like_series",
    [
        pd.Series([pd.Timestamp(2021, 4, 3), None] * 10),
        pd.Series([pd.Timedelta(days=-2), None] * 10),
    ],
)
def test_optional_fusion(memory_leak_check, dt_like_series):
    """
    Checks that pd.DataFrame can be used on multiple series
    operations with optional types. This triggers a parfor fusion
    that keeps values as optional types when merging functions
    (see BE-1396)
    """

    def impl(S):
        return pd.DataFrame(
            {"A": S.apply(lambda x: (None if (pd.isna(x)) else x)).isna()}
        )

    check_func(impl, (dt_like_series,))


def test_astype_str_null(memory_leak_check):
    """
    Checks that astype(str) converts Null values to strings
    """

    def impl(df):
        return df.astype(str)

    df = pd.DataFrame(
        {
            "A": pd.Series([1, 2, 4, None, 7] * 10, dtype="Int64"),
            "B": [pd.Timestamp(2021, 5, 4, 1), None] * 25,
        }
    )
    check_func(impl, (df,))


def test_astype_str_keep_null(memory_leak_check):
    """
    Checks that astype(str) keeps null values null when _bodo_nan_to_str=False
    """

    def impl(S):
        return S.astype(str, _bodo_nan_to_str=False)

    df = pd.DataFrame(
        {
            "A": pd.Series([1, 2, 4, None, 7] * 10, dtype="Int64"),
            "B": [pd.Timestamp(2021, 5, 4, 1), None] * 25,
        }
    )
    # This is a Bodo specific arg so use py_output
    py_output = df.astype(str)
    py_output["A"][py_output["A"] == "<NA>"] = None
    py_output["B"][py_output["B"] == "NaT"] = None
    check_func(impl, (df,), py_output=py_output)


def test_categorical_astype(memory_leak_check):
    """
    Test that astype with categorical columns to the underlying
    elem type works as expected. Needed for BodoSQL.
    """

    def impl(categorical_table):
        categorical_table = pd.DataFrame(
            {
                "A": categorical_table["A"].astype("str"),
                "B": categorical_table["B"].astype("Int64"),
                "C": categorical_table["C"].astype("UInt64"),
                "D": categorical_table["D"].astype("float64"),
                "E": categorical_table["E"].astype("datetime64[ns]"),
                "F": categorical_table["F"].astype("timedelta64[ns]"),
                "G": categorical_table["G"].astype("boolean"),
            }
        )
        return categorical_table

    df = pd.DataFrame(
        {
            # String category
            "A": pd.Categorical(["anve", "Er2"] * 5),
            # int64
            "B": pd.Categorical([5, -32] * 5),
            # uint64
            "C": pd.Categorical(pd.array([5, 2] * 5, "uint64")),
            # float64
            "D": pd.Categorical([1.1, 2.7] * 5),
            # dt64
            "E": pd.Categorical(
                [pd.Timestamp(2021, 4, 5), pd.Timestamp(2021, 4, 4)] * 5
            ),
            # td64
            "F": pd.Categorical([pd.Timedelta(2), pd.Timedelta(seconds=-4)] * 5),
            # boolean
            "G": pd.Categorical([True, False] * 5),
        }
    )

    check_func(impl, (df,))


def test_avoid_static_getitem_const(memory_leak_check):
    """Test for avoiding Numba's static_getitem const idx inference bug in typing pass.
    See https://github.com/numba/numba/issues/7592
    """

    def impl(df):
        cols = [col for col in df.columns if "B" in col]
        return df[cols]

    df = pd.DataFrame({"ABC": [1, 2, 3]})
    check_func(impl, (df,))


def test_df_gatherv_table_format(memory_leak_check):
    """test gathering a distributed dataframe with table format"""

    def impl(df):
        return bodo.gatherv(df)

    df = pd.DataFrame(
        {
            "A": [1, 2, 3] * 20,
            "B": ["abc", "bc", "cd"] * 20,
            "C": [5, 6, 7] * 20,
            "D": [1.1, 2.2, 3.3] * 20,
        }
    )
    # add a bunch of columns to trigger table format
    for i in range(bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD):
        df[f"F{i}"] = 11

    distributed_df = _get_dist_arg(df, copy=True, var_length=True)
    bodo_df = bodo.jit(distributed=["df"])(impl)(distributed_df)
    passed = 1
    if bodo.get_rank() == 0:
        passed = _test_equal_guard(
            bodo_df,
            df,
        )
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size()


def test_map_array_concat(memory_leak_check):
    """Tests for pd.concat on MapArrays with a sequential implementation."""

    def impl(df, n):
        df_list = []
        for i in range(1, n + 1):
            df_new = pd.DataFrame(
                {
                    "A": df.A,
                    "B": df.A.apply(lambda val, i: {j: j % i for j in range(val)}, i=i),
                }
            )
            df_list.append(df_new)
        return pd.concat(df_list)

    def impl_str(df, n):
        df_list = []
        for i in range(1, n + 1):
            df_new = pd.DataFrame(
                {
                    "A": df.A,
                    "B": df.A.apply(
                        lambda val, i: {str(j): str(j % i) for j in range(val)}, i=i
                    ),
                }
            )
            df_list.append(df_new)
        return pd.concat(df_list)

    n = 10
    df = pd.DataFrame({"A": np.arange(n)})

    # Test correctness sequentially, where ordering matters.
    # We cannot test parallel right now because we would need to sort
    # the output.
    check_func(impl, (df, n), is_out_distributed=False, only_seq=True)
    check_func(impl_str, (df, n), is_out_distributed=False, only_seq=True)


def test_update_df_type(memory_leak_check):
    """
    Test updating a DataFrame type works as expected and
    can be used in objmode.
    """

    def py_func(n):
        return pd.DataFrame(
            {
                "A": [1, 2, 4, 5, 9, 22],
                "B": [{str(j): j % n for j in range(n)} for i in range(1, 7)],
            }
        )

    dummy_df = pd.DataFrame(
        {
            "A": [1],
            "B": [{"a": 1}],
        }
    )
    infered_dtype = bodo.typeof(dummy_df)
    new_dtype = infered_dtype.replace_col_type(
        "B",
        bodo.MapArrayType(bodo.string_array_type, bodo.IntegerArrayType(types.int64)),
    )

    def bodo_func(n):
        with bodo.objmode(df=new_dtype):
            df = py_func(n)
        return df

    check_func(
        bodo_func, (15,), py_output=py_func(15), only_seq=True, is_out_distributed=False
    )


def test_index_col_assign(memory_leak_check):
    def test_impl(df):
        new_df = pd.DataFrame({"A": np.arange(len(df))})
        new_df["id"] = new_df.reset_index().index + 1
        return new_df

    df = pd.DataFrame({"A": [1, 2, 1, 4, 512, 1, 4]})
    check_func(test_impl, (df,))


def test_df_type_change_iloc_loc(memory_leak_check):
    """make sure df.loc and df.iloc work correctly when df type changes"""

    def impl1(df):
        df["B"] = 3
        df.iloc[:, 1] = 4
        return df

    def impl2(df):
        df["B"] = 3
        df.loc[:, [True, False]] = 4
        return df

    def impl3(df):
        df["B"] = 3
        return df.iloc[:, 1:]

    def impl4(df):
        df["B"] = 3
        return df.loc[:, [True, False]]

    df = pd.DataFrame({"A": [1, 2, 1, 4, 512, 1, 4]})
    check_func(impl1, (df,), copy_input=True, only_seq=True)
    check_func(impl2, (df,), copy_input=True, only_seq=True)
    check_func(impl3, (df,), copy_input=True, only_seq=True)
    check_func(impl4, (df,), copy_input=True, only_seq=True)


@pytest.mark.parametrize(
    "A",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 2, 3, 4.1, 5],
                    "B": ["a", "d", "c", "e", "d"],
                    "C": [4.3, 5.6, 3.2, 8.4, 9.6],
                }
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(pd.DataFrame({"A": [1, 2, 3, 4, 5]}), marks=pytest.mark.slow),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 2, 3, None, 5],
                    "B": ["a", "d", "c", "e", "d"],
                    "C": [4.3, 5.6, None, 8.4, 9.6],
                }
            ),
            marks=pytest.mark.slow,
        ),
    ],
)
def test_dataframe_infer_objects(A, memory_leak_check):
    def test_impl(df):
        return df.infer_objects()

    check_func(test_impl, (A,))


def test_na_df_apply_homogeneous_no_inline(memory_leak_check):
    """
    Checks that NA handling inside Bodo UDFs works properly
    for homogeneous series without inlining.
    """

    def impl(df):
        def f(row):
            # Add an objectmode call to prevent inlining
            with bodo.objmode(ret_val="int64"):
                ret_val = -1
            if pd.isna(row["A"]):
                return ret_val
            else:
                total = 0
                x = 8
                for i in range(x):
                    total += i * row["A"]
                return total

        return df.apply(f, axis=1)

    df = pd.DataFrame(
        {
            "A": pd.Series([1, None, None] * 5, dtype="Int64"),
            "B": pd.Series([None, 2, 3] * 5, dtype="Int64"),
        }
    )
    check_func(impl, (df,))


def test_apply_inline_optimization(memory_leak_check):
    """
    Tests that inlining a df.apply call with a single value
    properly optimizes out the intermediate series and tuple
    values that are used if the call can't be inlined.
    """

    def impl(df):
        def f(row):
            if pd.isna(row["A"]):
                return -1
            return row["A"]

        return df.apply(f, axis=1)

    df = pd.DataFrame(
        {
            "A": pd.Series([1, None, None] * 5, dtype="Int64"),
            "B": pd.Series([None, 2, 3] * 5, dtype="Int64"),
        }
    )
    check_func(impl, (df,))
    # Check to ensure build_nullable_tuple has been optimized out
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl)
    bodo_func(df)
    _ensure_func_calls_optimized_out(
        bodo_func, {("build_nullable_tuple", "bodo.libs.nullable_tuple_ext")}
    )


# TODO [BE-974]: Add memory_leak_check
def test_df_merge_type_reset():
    df1 = pd.DataFrame(
        {"key": ["bar", "baz", "foo", "foo"] * 3, "value": [1, 2, 3, 5] * 3}
    )
    df2 = pd.DataFrame(
        {"key": ["bar", "baz", "foo", "foo"] * 3, "value": [5, 6, 7, 8] * 3}
    )

    def test(df1, df2):
        df1a = df1.groupby("key").apply(lambda x: 3).reset_index()
        df2a = df2.groupby("key").apply(lambda x: 4).reset_index()
        ans = df1a.merge(df2a, left_on=["key"], right_on=["key"])
        return ans

    check_func(test, (df1, df2), reset_index=True, sort_output=True)


def test_df_itertuples(memory_leak_check):
    def test_impl(df):
        res = 0.0
        for r in df.itertuples():
            res += r[1]
        return res

    df = pd.DataFrame({"A": [13, 31, 42] * 4, "B": [1, 2, 3, 5] * 3})
    check_func(test_impl, (df,), dist_test=False)


@pytest.mark.parametrize(
    "func,err_regex",
    [
        (
            lambda df1, df2: df1.merge(df2, left_on="x", right_on="x"),
            r".* use of index .* is unsupported(\n|.)*",
        ),
        (
            lambda df1, df2: df1.merge(df2, on="non_existant_key"),
            r".* invalid key (\n|.)*",
        ),
    ],
)
# TODO: [BE-1738]: Add memory_leak_check
def test_df_merge_error_handling(func, err_regex):
    df1 = pd.DataFrame({"key": ["bar", "baz", "foo", "foo"], "value": [1, 2, 3, 5]})
    df2 = pd.DataFrame({"key": ["bar", "baz", "foo", "foo"], "value": [5, 6, 7, 8]})
    df1.index.name = "x"
    df2.index.name = "x"

    with pytest.raises(BodoError, match=err_regex):
        bodo.jit(func)(df1, df2)


pd_supported_merge_cols = [
    pytest.param(pd.Categorical([1, 1, 1, 2, 3]), id="CategoricalArrayType"),
    pytest.param(np.array([1, 2, 3, 4, 5]), id="Array"),
    pytest.param(
        pd.array([1, 2, 3, 4, 5], dtype=pd.Int32Dtype()), id="IntegerArrayType"
    ),
    pytest.param(
        pa.Decimal128Array.from_pandas(
            pd.array([1, 2, 3, 4, 5], dtype=pd.Int32Dtype())
        ),
        id="DecimalArrayType",
    ),
    pytest.param(
        pd.Series(({1: 1, 2: 2}, {2: 2}, {3: 3}, {4: 4}, {5: 5})), id="MapArrayType"
    ),
    pytest.param(("a", "b", "c", "d", "e"), id="StringArrayType"),
    pytest.param((b"a", b"b", b"c", b"d", b"e"), id="BinaryArrayType"),
    pytest.param((False, False, False, True, True), id="BooleanArray"),
    pytest.param(
        tuple(datetime.date(year, 7, 22) for year in range(1999, 2004)),
        id="DatetimeDateArrayType",
    ),
    pytest.param(
        tuple(
            datetime.date(2021, 7, 22) - datetime.date(year, 7, 22)
            for year in range(1999, 2004)
        ),
        id="DatetimeTimedeltaArrayType",
    ),
    # TODO [BE-1804]: test once intervals are supported
    # pytest.param(
    #     pd.arrays.IntervalArray(
    #         [
    #             pd.Interval(0, 1),
    #             pd.Interval(1, 2),
    #             pd.Interval(2, 3),
    #             pd.Interval(3, 4),
    #             pd.Interval(4, 5),
    #         ]
    #     ),
    #     id="IntervalArray"
    # ),
]


bodo_only_merge_cols = [
    pytest.param(
        pd.Series(({"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5})),
        id="StructArrayType",
    ),
    pytest.param(pd.Series(([1], [2], [3], [4], [5])), id="ArrayItemArrayType"),
]


@pytest.mark.parametrize("key", pd_supported_merge_cols)
def test_df_merge_col_key_types(key, memory_leak_check):
    def impl(df1, df2):
        return df1.merge(df2, on="key")

    df1 = pd.DataFrame({"key": key, "value": [1, 2, 3, 4, 5]})
    df2 = pd.DataFrame({"key": key, "value": [5, 6, 7, 8, 9]})

    try:
        check_func(impl, (df1, df2), reset_index=True, sort_output=True)
    except TypeError:
        with pytest.raises(BodoError, match=r".* MapArrayType unsupported(.|\n)*"):
            bodo.jit(impl)(df1, df2)


@pytest.mark.parametrize("val", pd_supported_merge_cols + bodo_only_merge_cols)
# TODO: [BE-1738]: Add memory_leak_check
def test_df_merge_col_value_types(val):
    def impl(df1, df2):
        return df1.merge(df2, on="key")

    df1 = pd.DataFrame({"key": ["bar", "bar", "baz", "foo", "foo"], "value": val})
    df2 = pd.DataFrame({"key": ["bar", "bar", "baz", "foo", "foo"], "value": val})

    check_func(impl, (df1, df2))


@pytest.mark.parametrize("key", bodo_only_merge_cols)
# TODO: [BE-1738]: Add memory_leak_check
def test_df_merge_col_key_bodo_only(key):
    def impl(df1, df2):
        return df1.merge(df2, on="key")

    val_x = [1, 2, 3, 4, 5]
    val_y = [5, 6, 7, 8, 9]
    df1 = pd.DataFrame({"key": key, "value": val_x})
    df2 = pd.DataFrame({"key": key, "value": val_y})

    df_exp = pd.DataFrame({"key": key, "value_x": val_x, "value_y": val_y})
    df_act = bodo.jit(impl)(df1, df2)
    pd.testing.assert_frame_equal(df_act, df_exp, check_column_type=False)


@pytest.mark.parametrize(
    "method_name",
    [
        "isna",
        pytest.param("isnull", marks=pytest.mark.slow),
        "notna",
        pytest.param("notnull", marks=pytest.mark.slow),
    ],
)
def test_table_isna(method_name, datapath, memory_leak_check):
    """
    Tests df.isna, df.isnull, df.notna, and df.notnull with
    a DataFrame in table format.
    """
    filename = datapath("many_columns.csv")

    func_text = f"""def impl():
        df = pd.read_csv({filename!r})
        df1 = df.{method_name}()
        return df1[["Column1", "Column3"]]
        """

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl)()
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['Column1', 'Column3']")


@pytest.mark.parametrize(
    "method_name",
    [
        "head",
        "tail",
    ],
)
def test_table_head_tail(method_name, datapath, memory_leak_check):
    """
    Tests df.head and df.tail with a DataFrame
    in table format.
    """
    filename = datapath("many_columns.csv")

    func_text = f"""def impl():
        df = pd.read_csv({filename!r})
        df1 = df.{method_name}()
        return df1[["Column1", "Column3"]]
        """

    local_vars = {}
    exec(func_text, globals(), local_vars)
    impl = local_vars["impl"]

    check_func(impl, ())
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl)()
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['Column1', 'Column3']")


def test_astype_dtypes_optimization(memory_leak_check):
    """
    Tests that doing df1.astype(df2.dtypes) is optimized
    to use the Bodo type information and perform the necessary
    type conversions. To test this, we check converting datetime64
    to dates, which will otherwise show up as objects in regular
    Pandas.
    """

    def test_impl(df1, df2):
        return df1.astype(dtype=df2.dtypes)

    df1 = pd.DataFrame(
        {
            # timestamp to cast to date
            "A": pd.date_range("2018-04-09", periods=5, freq="2D1H"),
            # Column that shouldn't be converted
            "B": ["a", "b", "c", "d", "e"],
            # Int to cast to string
            "C": [1, 2, 3, 4, 5],
        }
    )
    df2 = pd.DataFrame(
        {
            "C": ["A", "B", "C", "D", "E"],
            "A": pd.date_range("2018-04-09", periods=5, freq="2D").date,
        }
    )
    # Pandas can't change types with an object, so we use a py_output
    py_output = df1.apply(
        lambda x: pd.Series(
            [x["A"].date(), x["B"], str(x["C"])], index=["A", "B", "C"]
        ),
        axis=1,
    )
    # use_dict_encoded_strings=False since cannot cast int to dict-encoded strings
    check_func(
        test_impl, (df1, df2), py_output=py_output, use_dict_encoded_strings=False
    )


def test_df_getitem_columname_func(memory_leak_check):
    """
    Tests using a string literal in df[] after passing the
    string through a function
    """

    g = bodo.jit(lambda a: a)

    def impl(df, a):
        return df[g(a)]

    df = pd.DataFrame({"A": [1, 2, 3] * 5})
    check_func(impl, (df, "A"))

    df = pd.DataFrame({1: [1, 2, 3] * 5})
    check_func(impl, (df, 1))


def test_df_getitem_columname_list_func(memory_leak_check):
    """
    Tests using a list in df[] after passing the
    list through a function
    """
    g = bodo.jit(lambda a: a)

    def impl(df, a):
        return df[g(a)]

    df = pd.DataFrame({"A": [1, 2, 3] * 5})
    check_func(impl, (df, ["A"]))

    df = pd.DataFrame({1: [1, 2, 3] * 5})
    check_func(impl, (df, [1]))


def test_df_loc_getitem_list_func(memory_leak_check):
    """
    Tests using a list in df.loc after passing the
    list through a function
    """
    g = bodo.jit(lambda a: a)

    def impl(df, a):
        return df.loc[1:, g(a)]

    df = pd.DataFrame({"A": np.arange(10)})
    check_func(impl, (df, ["A"]))


def test_df_iloc_getitem_slice_func(memory_leak_check):
    """
    Tests using a slice in df.iloc after passing the
    slice through a function
    """
    g = bodo.jit(lambda a: a)

    def impl(df, a):
        x = df.iloc[1:, g(a)]
        return x

    df = pd.DataFrame({"A": np.arange(10)})

    # Use py_output because regular slices cannot be boxed yet.
    py_output = df.iloc[1:, 0:1]
    check_func(impl, (df, slice(0, 1)), py_output=py_output)


def test_df_loc_col_slice_assign(memory_leak_check):
    """
    Test assignment to a columns slice of a dataframe using df.loc
    """

    def impl1(df):
        df.loc[1, :] = -1
        return df

    def impl2(df):
        df.loc[1, ::2] = -1
        return df

    df = pd.DataFrame(
        {
            "A": np.arange(0, 5),
            "B": np.arange(5, 10),
            "C": np.arange(10, 15),
            "D": np.arange(15, 20),
        }
    )

    check_func(impl1, (df,), copy_input=True)
    check_func(impl2, (df,), copy_input=True)


def test_df_iloc_col_slice_assign(memory_leak_check):
    """
    Test assignment to a columns slice of a dataframe using df.iloc
    """

    def impl1(df):
        df.iloc[1, :] = -1
        return df

    def impl2(df):
        df.iloc[1, 1:4:2] = -1
        return df

    df = pd.DataFrame(
        {
            "A": np.arange(0, 5),
            "B": np.arange(5, 10),
            "C": np.arange(10, 15),
            "D": np.arange(15, 20),
        }
    )

    check_func(impl1, (df,), copy_input=True)
    check_func(impl2, (df,), copy_input=True)


def test_df_mask_where_df(df_value, memory_leak_check):
    def test_where(df, cond, val):
        return df.where(cond, val)

    def test_mask(df, cond, val):
        return df.mask(cond, val)

    np.random.seed(42)
    cond = np.random.ranf(df_value.shape) < 0.5
    assert len(cond[:, 0]) == len(df_value)

    new_cols = {c: str(c) + "_" for c in df_value.columns}
    df_renamed = df_value.rename(columns=new_cols)
    df_flipped = pd.DataFrame(
        {c: df_value[::-1].get(c) for c in df_value.columns}, index=df_value.index
    )
    assert set(df_value.columns).isdisjoint(df_renamed.columns)

    # TODO: support series.where with two categorical inputs
    # `str` equivalent of `lambda dtype: dtype.name`
    if "category" in df_value.dtypes.apply(str).values:
        with pytest.raises(
            BodoError,
            match=f"DataFrame.where.* 'other' must be a scalar, non-categorical series, 1-dim numpy array or StringArray with a matching type for Series.",
        ):
            bodo.jit(test_where)(df_value, cond, df_value)
        with pytest.raises(
            BodoError,
            match=f"DataFrame.mask.* 'other' must be a scalar, non-categorical series, 1-dim numpy array or StringArray with a matching type for Series.",
        ):
            bodo.jit(test_mask)(df_value, cond, df_value)
        return

    check_func(
        test_where,
        (
            df_value,
            cond,
            df_flipped,
        ),
        copy_input=True,
    )
    check_func(
        test_mask,
        (
            df_value,
            cond,
            df_flipped,
        ),
        copy_input=True,
    )
    check_func(
        test_where,
        (
            df_value,
            cond,
            df_renamed,
        ),
        copy_input=True,
    )
    check_func(
        test_mask,
        (
            df_value,
            cond,
            df_renamed,
        ),
        copy_input=True,
    )


def test_df_mask_where_series_other(memory_leak_check):
    """
    Test df.mask and df.where with pd.Series `other`.
    """

    def test_mask(df, cond, val):
        return df.mask(cond, val)

    def test_where(df, cond, val):
        return df.where(cond, val)

    np.random.seed(42)
    df = pd.DataFrame(
        {
            "A": [1, 8, 4, 11, -3],
            "B": [1.1, np.nan, 4.2, 3.1, -1.3],
            "C": [True, False, False, True, True],
        }
    )
    df2 = df.copy()
    df2.B = ["a", "b", "c", "d", "e"]
    cond = np.random.ranf(df.shape) < 0.5

    check_func(
        test_mask,
        (df, cond, df.iloc[:, 0]),
        copy_input=True,
        py_output=df.mask(cond, df.iloc[:, 0], axis=0),
    )
    check_func(
        test_where,
        (df, cond, df.iloc[:, 0]),
        copy_input=True,
        py_output=df.where(cond, df.iloc[:, 0], axis=0),
    )
    with pytest.raises(
        BodoError,
        match=f"DataFrame.where.* series and 'other' must share a common type.",
    ):
        check_func(
            test_where,
            (df2, cond, df2.iloc[:, 0]),
            copy_input=True,
            py_output=df2.where(cond, df2.iloc[:, 0], axis=0),
        )
    with pytest.raises(
        BodoError,
        match=f"DataFrame.mask.* series and 'other' must share a common type.",
    ):
        check_func(
            test_mask,
            (df2, cond, df2.iloc[:, 0]),
            copy_input=True,
            py_output=df2.where(cond, df2.iloc[:, 0], axis=0),
        )


def test_df_mask_where_scalar_other(memory_leak_check):
    """
    Test df.where and df.mask with a scalar value for `other`.
    """

    def test_mask(df, cond, val):
        return df.mask(cond, val)

    def test_where(df, cond, val):
        return df.where(cond, val)

    np.random.seed(42)
    df = pd.DataFrame({"A": np.arange(10), "B": np.arange(10)})
    cond = np.random.ranf(df.shape) < 0.5
    other = 5

    check_func(test_mask, (df, cond, other))
    check_func(test_where, (df, cond, other))


@pytest.mark.parametrize("offset", ("15D", pd.DateOffset(days=15), "0D"))
def test_df_first_last(memory_leak_check, offset):
    """
    Test df.first() and last() with string and DateOffset offsets (for DataFrames with DateTimeIndex)
    """

    def impl_first(df):
        return df.first(offset)

    def impl_last(df):
        return df.last(offset)

    n = 30
    df = pd.DataFrame(
        {"A": pd.Series(np.arange(n))},
        index=pd.date_range("2018-04-09", periods=n, freq="2D"),
    )

    if isinstance(offset, pd.DateOffset):
        end_date = end = df.index[0] + offset
        # Tick-like, e.g. 3 weeks
        if isinstance(offset, pd._libs.tslibs.Tick) and end_date in df.index:
            end = df.index.searchsorted(end_date, side="left")
            py_output = df.iloc[:end]
        else:
            py_output = df.loc[:end]
    else:
        py_output = None

    check_func(impl_first, (df,), py_output=py_output)
    check_func(impl_last, (df,))


def test_empty_df_first_last(memory_leak_check):
    """
    Test Series.first() and Series.last() with an empty dataframe.
    """

    def impl_first(df):
        return df.first("5D")

    def impl_last(df):
        return df.last("5D")

    def one_empty_rank(df):
        # tests df.last() with at least rank being empty when run with 3+ ranks
        res = df.groupby("A").sum().sort_values("B")
        return res.last("1D")

    n = 10
    df = pd.DataFrame(
        {"A": pd.Series(np.arange(n))},
        index=pd.date_range("2018-04-09", periods=n, freq="1D"),
    )
    empty_df = df[df.A > n]
    df2 = pd.DataFrame(
        {
            "A": np.tile(pd.date_range("2018-04-09", periods=2, freq="1D"), 15),
            "B": np.arange(30),
        }
    )
    check_func(impl_first, (empty_df,))
    check_func(impl_last, (empty_df,))
    check_func(one_empty_rank, (df2,))


def test_dataframe_explode():
    """
    Tests for df.explode() on single and multiple columns.
    """
    # NOTE: a helper is necessary as Pandas df.explode() will fail for the tests below
    # because the `mylen` function used by Pandas internally marks scalars/NAs as having
    # length -1 and takes empty arrays as 0 despite each of the aforementioned taking up
    # one entry in the exploded result.
    def _explode_helper(df, cols):
        mylen = lambda x: max(1, len(x)) if is_list_like(x) else 1
        counts = df[cols[0]].apply(mylen)
        assert all(all(counts == df[c].apply(mylen)) for c in cols)
        res = pd.DataFrame(
            {
                c: df[c].explode() if c in cols else df[c].repeat(counts)
                for c in df.columns
            },
            index=df[cols[0]].explode().index,
        )
        return res

    def test_impl(df, col):
        return df.explode(col)

    def test_impl_single(df):
        return df.explode("A")

    def test_impl_list(df):
        return df.explode(["A", "C"])

    def test_impl_tuple(df):
        return df.explode(("A", "C"))

    def test_str_split_explode(S):
        df = pd.DataFrame({"A": S.str.split()})
        return df.explode("A")

    df = pd.DataFrame(
        {
            "A": [[0, 1, 2], [5], [], [3, 4]] * 3,
            "B": [1, 7, 2, 4] * 3,
            "C": [[1, 2, 3], np.nan, [], [1, 2]] * 3,
        }
    )
    S = pd.Series(["a b c", "d e", "f", "g h"] * 3)

    check_func(
        test_impl,
        (
            df,
            "A",
        ),
        py_output=_explode_helper(df, "A"),
    )
    check_func(
        test_impl,
        (
            df,
            ["A", "C"],
        ),
        py_output=_explode_helper(df, ["A", "C"]),
    )
    # TODO: tuple passed in as argument
    # check_func(test_impl, (df, ("A", "C")), py_output=_explode_helper(df, ("A", "C")))
    check_func(test_impl_single, (df,), py_output=_explode_helper(df, "A"))
    check_func(test_impl_list, (df,), py_output=_explode_helper(df, ["A", "C"]))
    check_func(test_impl_tuple, (df,), py_output=_explode_helper(df, ("A", "C")))
    check_func(test_str_split_explode, (S,))


def test_empty_dataframe_creation(memory_leak_check):
    """
    Test empty dataframe.
    """

    def impl():
        df = pd.DataFrame({})
        return df

    check_func(impl, (), check_dtype=False)


def test_unify_dict_string_dataframes():
    """
    Tests unifying DataFrames when the inputs consist of
    dict_string_array columns.
    """
    # TODO: Add memory leak check, casting bug

    def impl1(df):
        # Tuple to tuple
        if len(df) > 200:
            # This implementation should always be False so
            # at runtime we check the cast
            df = df.sort_values(by="A")
        return df

    def impl2(source_file):
        # table to tuple
        df = pd.read_parquet(source_file)
        return bodo.hiframes.pd_dataframe_ext._table_to_tuple_format_decoded(df)

    def impl3(source_file):
        # Table to table
        df = pd.read_parquet(source_file)
        if len(df) > 200:
            # This implementation should always be False so
            # at runtime we check the cast
            df = bodo.utils.typing.decode_if_dict_array(df)
        return df

    def impl4(df):
        # tuple to table
        return bodo.hiframes.pd_dataframe_ext._tuple_to_table_format_decoded(df)

    df = pd.DataFrame({"A": ["aefw", "ER", None, "Few"] * 25, "B": ["#$@4", "!1"] * 50})
    temp_file = "temp_file.pq"
    if bodo.get_rank() == 0:
        # Send to a file to create table source
        df.to_parquet(temp_file, index=False)
    bodo.barrier()
    # use dictionary-encoded arrays for strings
    saved_read_as_dict_threshold = bodo.io.parquet_pio.READ_STR_AS_DICT_THRESHOLD
    try:
        bodo.hiframes.boxing._use_dict_str_type = True
        bodo.io.parquet_pio.READ_STR_AS_DICT_THRESHOLD = 1000.0
        check_func(impl1, (df,), use_table_format=False, use_dict_encoded_strings=True)
        check_func(
            impl2,
            (temp_file,),
            py_output=df,
            use_table_format=False,
            use_dict_encoded_strings=True,
        )
        check_func(
            impl3, (temp_file,), use_table_format=False, use_dict_encoded_strings=True
        )
        if bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD > len(df.columns):
            check_func(
                impl4,
                (df,),
                py_output=df,
                use_table_format=False,
                use_dict_encoded_strings=True,
            )
    finally:
        bodo.io.parquet_pio.READ_STR_AS_DICT_THRESHOLD = saved_read_as_dict_threshold
        if bodo.get_rank() == 0:
            os.remove(temp_file)


@pytest.mark.parametrize(
    "id_arr,value_arr",
    [
        (["A", "B"], ["C"]),
        (None, ["C", "D"]),
        (["A"], ["C"]),
        (["A", "B", "C"], ["C", "D"]),
        (["A", "B"], ["C", "D"]),
        (["A", "B"], None),
    ],
)
def test_df_melt(id_arr, value_arr, memory_leak_check):
    """
    Tests for df.melt / pd.melt with various id_vars value_vars of string/integer types
    """

    def test_impl(df, id_vars, value_vars):
        return df.melt(id_vars=id_vars, value_vars=value_vars)

    def test_impl2(df, id_vars, value_vars):
        return pd.melt(df, id_vars=id_vars, value_vars=value_vars)

    df = pd.DataFrame(
        {
            "A": ["a", "b", "c"] * 4,
            "B": ["c", "d", "e"] * 4,
            "C": [1, 3, 5] * 4,
            "D": [2, 4, 6] * 4,
        }
    )

    check_func(test_impl, (df, id_arr, value_arr), sort_output=True, reset_index=True)
    if id_arr and value_arr and len(id_arr) == len(value_arr) == 2:
        # No need to check all test cases again for pd.melt()
        check_func(
            test_impl2, (df, id_arr, value_arr), sort_output=True, reset_index=True
        )


@pytest.mark.parametrize(
    "df_dict",
    [
        {
            "A": ["a", "b", "c"] * 3,
            "B": ["d", "e", "f"] * 3,
            "C": [1, 2, 3] * 3,
            "D": [4, 5, 6] * 3,
        },
        {
            "A": ["a", "b", "c"] * 3,
            "B": ["d", "e", "f"] * 3,
            "C": [1.1, 2.1, 3.1] * 3,
            "D": [4, 5, 6] * 3,
        },
        {
            "A": ["a", "b", "c"] * 3,
            "B": ["d", "e", "f"] * 3,
            "C": [1, 2, 3] * 3,
            "D": [4.1, 5.1, 6.1] * 3,
        },
        {
            "A": ["a", "b", "c"] * 3,
            "B": ["d", "e", "f"] * 3,
            "C": np.array([1, 2, 3] * 3, dtype="int8"),
            "D": [37423, 583305, 32343] * 3,
        },
        {
            "A": ["a", "b", "c"] * 3,
            "B": ["d", "e", "f"] * 3,
            "C": np.array([1, 2, 3] * 3, dtype="uint8"),
            "D": [37423, -583305, 32343] * 3,
        },
        {
            "A": ["a", "b", "c"] * 3,
            "B": ["d", "e", "f"] * 3,
            3: np.array([1, 2, 3] * 3, dtype="uint8"),
            4: [37423, -583305, 32343] * 3,
        },
        {
            "A": [1, 2, 3] * 3,
            "B": [4, 5, 6] * 3,
            "C": ["a", "b", "c"] * 3,
            "D": ["d", "e", "f"] * 3,
        },
    ],
)
def test_df_melt_diff_types(df_dict, memory_leak_check):
    """
    Test df.melt() with different value_vars and id_vars column and label data types
    """

    def test_impl(df, id_vars, value_vars):
        return df.melt(id_vars=id_vars, value_vars=value_vars)

    df = pd.DataFrame(df_dict)

    if "C" in df.columns:
        check_func(
            test_impl, (df, ["A", "B"], ["C", "D"]), sort_output=True, reset_index=True
        )
        check_func(
            test_impl, (df, ["B", "A"], ["D", "C"]), sort_output=True, reset_index=True
        )
    elif 3 in df.columns:
        check_func(
            test_impl, (df, ["B", "A"], [3, 4]), sort_output=True, reset_index=True
        )


def test_df_melt_var_value_names(memory_leak_check):
    def impl(df, var_name, value_name):
        return df.melt(["A", "B"], "C", var_name=var_name, value_name=value_name)

    df = pd.DataFrame(
        {
            "A": ["a", "b", "c"] * 4,
            "B": ["c", "d", "e"] * 4,
            "C": [1, 3, 5] * 4,
            "D": [2, 4, 6] * 4,
        }
    )

    check_func(impl, (df, "some", "name"), sort_output=True, reset_index=True)
    check_func(
        impl,
        (df, 2, "name"),
        sort_output=True,
        reset_index=True,
        py_output=df.melt(["A", "B"], "C").rename(
            columns={"variable": 2, "value": "name"}
        ),
    )
    # TODO: Pandas produces KeyError
    # check_func(impl, (df, "some", None), sort_output=True, reset_index=True, py_output=df.melt(["A","B"], "C").rename(columns={"variable":"some","value":None}))


def test_df_melt_many_columns(memory_leak_check):
    def impl(df):
        return df.melt(id_vars=["col0", "col1"], value_vars=None)

    num_cols = max(1000, bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD)
    df = pd.DataFrame(
        data={f"col{i}": np.arange(10) for i in range(num_cols)}, index=np.arange(10)
    )

    t0 = time.time()
    check_func(impl, (df,), sort_output=True, reset_index=True)
    compilation_time = time.time() - t0
    # Determine the max compilation time on any rank to avoid hangs.
    comm = MPI.COMM_WORLD
    compilation_time = comm.allreduce(compilation_time, op=MPI.MAX)
    assert compilation_time < 60, "df.melt() took too long to compile."


@pytest.mark.parametrize("use_index", [True, False])
def test_df_table_memory_usage(use_index, datapath, memory_leak_check):
    """
    Test memory usage when using table format on distributed data.
    """

    filename = datapath("many_columns.parquet")

    @bodo.jit
    def memory_usage_table(use_index):
        df = pd.read_parquet(filename)
        return df.memory_usage(use_index)

    # Precompute the py_output for distributed data. These
    # numbers are taken from nbytes_no_table
    num_rows = 1000
    num_cols = 99
    total_int_data = 8 * num_rows
    total_float_data = 8 * num_rows
    # Number of bytes in the null bitmap is a function of nranks.
    # Here data is loaded as 1D, so we repeat the calculation here.
    # Each rank is ceil(my_chunk / 8)
    # my_chunk = num_rows // num_ranks + 1 if remainder
    null_bytes = 0
    for i in range(bodo.get_size()):
        # Compute bits evenly distributed
        my_chunk = num_rows // bodo.get_size()
        # Address remainder
        if i > num_rows % bodo.get_size():
            my_chunk += 1
        null_bytes += np.int64(np.ceil(my_chunk / 8))
    nbytes_list = []
    if use_index:
        # Append the index num bytes, which should just be a range index,
        # so 3 integers per rank.
        nbytes_list.append((8 * 3 * bodo.get_size()))
    for i in range(33):
        # Int columns are nullable integers
        nbytes_list.append(total_int_data + null_bytes)
        nbytes_list.append(total_float_data)
        # Rows: 0, 14, 21, 57 contain null data
        null_data_rows = set([0, 14, 21, 57])
        # Calculate the data field
        total_string_data = sum(
            [
                len(f"value{k}")
                for j, k in enumerate((np.arange(num_rows) * (i + 1)))
                if j not in null_data_rows
            ]
        )
        # Offsets has an value on every rank but otherwise are filled with 64bit integers
        total_offset_size = 8 * (num_rows + bodo.get_size())
        nbytes_list.append(total_string_data + total_offset_size + null_bytes)
    col_names = [f"Column{i}" for i in range(num_cols)]
    if use_index:
        col_names = ["Index"] + col_names
    py_output = pd.Series(nbytes_list, index=pd.Index(col_names))
    bodo_out = memory_usage_table(use_index)
    pd.testing.assert_series_equal(bodo_out, py_output, check_index_type=False)


@pytest.mark.parametrize("use_deep", [True, False])
def test_df_table_copy(use_deep, datapath, memory_leak_check):
    """
    Test copy when using table format.
    """

    filename = datapath("many_columns.parquet")

    def copy_table(use_deep):
        df = pd.read_parquet(filename)
        df = df.copy(deep=use_deep)
        return df[["Column1", "Column3"]]

    check_func(copy_table, (use_deep,))
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(copy_table)(use_deep)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['Column1', 'Column3']")


@pytest.mark.parametrize("use_copy", [True, False])
def test_df_table_rename(use_copy, datapath, memory_leak_check):
    """
    Test rename when using table format.
    """

    filename = datapath("many_columns.parquet")

    def rename_table(use_copy):
        df = pd.read_parquet(filename)
        df = df.rename(
            copy=use_copy, columns={"Column1": "Columnx23", "Column5": "Column1"}
        )
        return df[["Column1", "Columnx23"]]

    check_func(rename_table, (use_copy,))
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(rename_table)(use_copy)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['Column1', 'Column5']")


@pytest.mark.parametrize(
    "method",
    ["average", "first", "dense"],
)
@pytest.mark.parametrize(
    "na_option",
    ["top", "bottom", "keep"],
)
@pytest.mark.parametrize("ascending", [False, True])
@pytest.mark.parametrize("pct", [True, False])
def test_df_rank(method, na_option, ascending, pct):
    def impl(df):
        return df.rank(method=method, na_option=na_option, ascending=ascending, pct=pct)

    df = pd.DataFrame(
        {
            "A": [2, 1, 2, 4, 8, None, 4],
            "B": [None, "b", "a", "b", "c", "d", "c"],
            "C": [np.nan, 2, 1, 4.2, 8, 4.2, np.nan],
            "D": [None, True, False, False, True, True, None],
        }
    )

    if method == "first":
        if not ascending:
            with pytest.raises(
                BodoError,
                match=re.escape(
                    # TODO: separate Series and Dataframe rank() error messages
                    "Series.rank(): method='first' with ascending=False is currently unsupported."
                ),
            ):
                bodo.jit(impl)(df)
        else:
            py_output = df.apply(
                lambda c: c.rank(
                    method=method, na_option=na_option, ascending=ascending, pct=pct
                )
            )
            check_func(impl, (df,), dist_test=False, py_output=py_output)
    else:
        check_func(impl, (df,), dist_test=False)


@pytest.mark.slow
def test_concat_1d_1dvar():
    """Tests that 1D input variables are not set to 1DVar if the input is 1dvar"""

    def impl(table1):
        oned_df = pd.DataFrame({"D": table1["D"].sum()}, pd.RangeIndex(0, 1, 1))
        out_df = pd.concat(
            [
                table1,
                oned_df,
            ]
        )
        return out_df

    df = pd.DataFrame({"A": [1, 2, 3], "D": [3, 4, 5]})

    check_func(
        impl,
        (df,),
        check_names=False,
        sort_output=True,
        reset_index=True,
        check_dtype=False,
    )

@pytest.mark.tz_aware
def test_tz_aware_dataframe_getitem(memory_leak_check):
    def impl_iat(df):
        return df.iat[2, 0]

    def impl_iloc(df):
        return df.iloc[2, 0]

    # Note we don't test loc + regular getitem
    # because they cannot return a scalar

    df = pd.DataFrame(
        {"colname": pd.array([pd.Timestamp("2000-01-01", tz="US/Pacific")] * 10)}
    )
    check_func(impl_iat, (df,))
    check_func(impl_iloc, (df,))
