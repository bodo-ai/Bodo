# Copyright (C) 2022 Bodo Inc. All rights reserved.
import datetime
import operator
from decimal import Decimal

import numba
import numba.np.ufunc_db
import numpy as np
import pandas as pd
import pytest
from numba.core.ir_utils import find_callname, guard

import bodo
from bodo.pandas_compat import pandas_version
from bodo.tests.series_common import (  # noqa
    SeriesReplace,
    numeric_series_val,
    series_stat,
    series_val,
)
from bodo.tests.utils import (
    SeriesOptTestPipeline,
    _test_equal,
    check_func,
    is_bool_object_series,
)
from bodo.utils.typing import BodoError
from bodo.utils.utils import is_call_assign


# TODO: integer Null and other Nulls
# TODO: list of datetime.datetime, categorical, timedelta, ...
@pytest.mark.slow
@pytest.mark.parametrize(
    "data",
    [
        555,
        [2, 3, 5],
        [2.1, 3.2, 5.4],
        [True, False, True],
        ["A", "C", "AB"],
        np.array([2, 3, 5]),
        pd.Series([2, 5, 6]),
        pd.Series([2.1, 5.3, 6.1], name="C"),
        pd.Series(["A", "B", "CC"]),
        pd.Series(["A", "B", "CC"], name="A"),
        pd.date_range(start="2018-04-24", end="2018-04-27", periods=3),
        pd.date_range(start="2018-04-24", end="2018-04-27", periods=3, name="A"),
        pd.Index([10, 12, 13], dtype="Int64"),
        pd.Index([10, 12, 14], dtype="Int64", name="A"),
    ],
)
@pytest.mark.parametrize(
    "index",
    [
        [2, 3, 5],
        [2.1, 3.2, 5.4],
        ["A", "C", "AB"],
        np.array([2, 3, 5]),
        pd.date_range(start="2018-04-24", end="2018-04-27", periods=3),
        pd.date_range(start="2018-04-24", end="2018-04-27", periods=3, name="A"),
        pd.Index([10, 12, 13], dtype="Int64"),
        pd.Index([10, 12, 14], dtype="Int64", name="A"),
        pd.RangeIndex(1, 4, 1),
        None,
    ],
)
@pytest.mark.parametrize("name", [None, "ABC"])
def test_series_constructor(data, index, name, memory_leak_check):
    # set Series index to avoid implicit alignment in Pandas case
    if isinstance(data, pd.Series) and index is not None:
        index = None

    # bypass literal as data and index = None
    if isinstance(data, int) and index is None:
        return

    def impl(d, i, n):
        return pd.Series(d, i, name=n)

    bodo_func = bodo.jit(impl)
    pd.testing.assert_series_equal(
        bodo_func(data, index, name),
        impl(data, index, name),
        check_dtype=False,
        check_index_type=False,
    )


@pytest.mark.slow
def test_series_constructor2(memory_leak_check):
    def impl(d, i, n):
        return pd.Series(d, i, name=n)

    bodo_func = bodo.jit(impl)
    data1 = pd.Series(["A", "B", "CC"], name="A")
    pd.testing.assert_series_equal(
        bodo_func(data1, None, None), impl(data1, None, None), check_dtype=False
    )

    data2 = pd.date_range(start="2018-04-24", end="2018-04-27", periods=3, name="A")
    pd.testing.assert_series_equal(
        bodo_func(data2, None, None), impl(data2, None, None), check_dtype=False
    )


@pytest.mark.slow
def test_series_constructor_dtype1(memory_leak_check):
    def impl(d):
        return pd.Series(d, dtype=np.int32)

    check_func(impl, ([3, 4, 1, -3, 0],), is_out_distributed=False)
    check_func(impl, (np.array([3, 4, 1, -3, 0]),))


@pytest.mark.slow
def test_series_constructor_dtype2(memory_leak_check):
    def impl(d):
        return pd.Series(d, dtype="int32")

    check_func(impl, ([3, 4, 1, -3, 0],), is_out_distributed=False)
    check_func(impl, (np.array([3, 4, 1, -3, 0]),))


@pytest.mark.slow
def test_series_constructor_init_str(memory_leak_check):
    """test passing "str" to Series constructor to populate a Series"""

    def impl(n):
        I = np.arange(n)
        return pd.Series(index=I, data=np.nan, dtype="str")

    check_func(impl, (111,))


@pytest.mark.slow
def test_series_constructor_int_arr(memory_leak_check):
    def impl(d):
        return pd.Series(d, dtype="Int32")

    check_func(impl, ([3, 4, 1, -3, 0],), is_out_distributed=False)
    check_func(impl, (np.array([3, 4, 1, -3, 0]),))
    check_func(impl, (np.array([1, 4, 1, np.nan, 0], dtype=np.float32),))


@pytest.mark.slow
def test_series_constructor_range(memory_leak_check):
    def impl(start, stop, step):
        return pd.Series(range(10))

    check_func(impl, (0, 10, 1))
    check_func(impl, (10, -1, -3))


@pytest.mark.slow
def test_series_cov_ddof(memory_leak_check):
    def test_impl(s1, s2, ddof=1):
        return s1.cov(s2, ddof=ddof)

    s1 = pd.Series([0.90010907, 0.13484424, 0.62036035])
    s2 = pd.Series([0.12528585, 0.26962463, 0.51111198])
    # passing py_output here since check_func may convert input to nullable
    # float, which Pandas doesn't handle in cov()
    check_func(test_impl, (s1, s2, 0), py_output=test_impl(s1, s2, 0))
    check_func(test_impl, (s1, s2, 1), py_output=test_impl(s1, s2, 1))
    check_func(test_impl, (s1, s2, 2), py_output=test_impl(s1, s2, 2))
    check_func(test_impl, (s1, s2, 3), py_output=test_impl(s1, s2, 3))
    check_func(test_impl, (s1, s2, 4), py_output=test_impl(s1, s2, 4))
    check_func(test_impl, (pd.Series([], dtype=float), pd.Series([], dtype=float)))


def test_series_fillna_series_val(series_val):
    def impl(S):
        val = S.iat[0]
        return S.fillna(val)

    if isinstance(series_val.iat[0], list):
        message = '"value" parameter cannot be a list'
        with pytest.raises(BodoError, match=message):
            bodo.jit(impl)(series_val)
    else:
        # TODO: Set dist_test=True once distributed getitem is supported
        # for Nullable and Categorical
        check_func(impl, (series_val,), dist_test=False, check_dtype=False)


@pytest.mark.skipif(
    bodo.hiframes.boxing._use_dict_str_type, reason="not supported for dict string type"
)
def test_string_series_fillna_inplace():
    """tests fillna on string series with inplace = True"""

    A = pd.Series(["sakdhjlf", "a", None, "sadljgksd", "", None] * 2)
    A2 = pd.Series(["hsldf", "avsjbdhjof", "bjknjoiuh", "abnsdgd", "", "sadf"] * 2)

    def impl0(S):
        return S.fillna("", inplace=True)

    def impl1(S):
        return S.fillna("hello", inplace=True)

    def impl2(S):
        val = S.iat[0]
        return S.fillna(val, inplace=True)

    def impl3(S1, S2):
        return S1.fillna(S2, inplace=True)

    check_func(impl0, (A,), check_dtype=False, use_dict_encoded_strings=False)
    check_func(
        impl0, (A2,), dist_test=False, check_dtype=False, use_dict_encoded_strings=False
    )
    check_func(impl1, (A,), check_dtype=False, use_dict_encoded_strings=False)
    check_func(impl1, (A2,), check_dtype=False, use_dict_encoded_strings=False)
    check_func(impl2, (A,), check_dtype=False, use_dict_encoded_strings=False)
    check_func(impl2, (A2,), check_dtype=False, use_dict_encoded_strings=False)
    check_func(impl3, (A, A2), check_dtype=False, use_dict_encoded_strings=False)
    check_func(impl3, (A2, A), check_dtype=False, use_dict_encoded_strings=False)


def test_binary_series_fillna_inplace():
    """tests fillna on binary series with inplace = True"""

    A = pd.Series([b"sakdhjlf", b"a", np.nan, b"asdjg", bytes(1)] * 3)
    A2 = pd.Series([bytes(1), b"avsjbdhjof", b"bjknjoiuh", b"abnsdgd", b""] * 3)

    def impl0(S):
        return S.fillna(b"", inplace=True)

    def impl1(S):
        return S.fillna(b"hello", inplace=True)

    def impl2(S):
        val = S.iat[0]
        return S.fillna(val, inplace=True)

    def impl3(S1, S2):
        return S1.fillna(S2, inplace=True)

    check_func(impl0, (A,), check_dtype=False)
    check_func(impl0, (A2,), check_dtype=False)
    check_func(impl1, (A,), check_dtype=False)
    check_func(impl1, (A2,), check_dtype=False)
    check_func(impl2, (A,), check_dtype=False)
    check_func(impl2, (A2,), check_dtype=False)
    check_func(impl3, (A, A2), check_dtype=False)
    check_func(impl3, (A2, A), check_dtype=False)


@pytest.mark.skipif(
    bodo.hiframes.boxing._use_dict_str_type, reason="not supported for dict string type"
)
def test_str_binary_series_fillna_inplace_mismatch():
    """test that fillna doesn't accept mismatched sizes"""
    A = pd.Series(["a", None, "B"] * 2)
    A2 = pd.Series(["b", None] * 12)

    def impl(S1, S2):
        return S1.fillna(S2, inplace=True)

    with pytest.raises(AssertionError, match="requires same length"):
        check_func(impl, (A, A2), check_dtype=False)


def series_replace_impl(series, to_replace):
    return series.replace(to_replace)


def series_replace_value_impl(series, to_replace, value):
    return series.replace(to_replace, value)


@pytest.mark.slow
def test_replace_series_val(series_val):
    """Run series.replace on the types in the series_val fixture. Catch
    expected failures from lack of coverage.
    """
    series = series_val.dropna()
    to_replace = series.iat[0]
    value = series.iat[1]

    message = ""
    if any(
        isinstance(x, (datetime.date, pd.Timedelta, pd.Timestamp))
        for x in [to_replace, value]
    ):
        # TODO: [BE-469]
        message = "Not supported for types"
    elif any(isinstance(x, pd.Categorical) for x in [to_replace, value]) or any(
        isinstance(x, list) for x in series
    ):
        message = "only support with Scalar"

    if message:
        with pytest.raises(BodoError, match=message):
            bodo.jit(series_replace_value_impl)(series, to_replace, value)
    else:
        check_func(series_replace_value_impl, (series, to_replace, value))


def test_series_replace_bitwidth(memory_leak_check):
    """Checks that series.replace succeeds on integers with
    different bitwidths."""

    def impl(S):
        return S.replace(np.int8(3), np.int64(24))

    S = pd.Series([1, 2, 3, 4, 5] * 4, dtype=np.int16)
    # Bodo dtype won't match because pandas casts everything to int64
    check_func(impl, (S,), check_dtype=False)


@pytest.mark.slow
def test_series_float_literal(memory_leak_check):
    """Checks that series.replace with an integer and a float
    that can never be equal will return a copy."""
    # Tests for [BE-468]
    def impl(S):
        return S.replace(np.inf, np.nan)

    S = pd.Series([1, 2, 3, 4, 5] * 4, dtype=np.int16)
    check_func(impl, (S,))


@pytest.mark.slow
def test_series_replace_str_literal(memory_leak_check):
    """Checks that series.replace works with str literals"""

    def impl(S):
        return S.replace("a", "A")

    S = pd.Series(["a", "BB", "32e", "Ewrew"] * 4)
    check_func(impl, (S,))


@pytest.mark.slow
@pytest.mark.parametrize(
    "series_replace",
    [
        # Int
        pytest.param(
            SeriesReplace(series=pd.Series([1, 8, 4, 11, -3]), to_replace=1, value=10),
        ),
        # Float
        pytest.param(
            SeriesReplace(
                series=pd.Series([1.1, 8.2, 4.3, 11.4, -3.5]),
                to_replace=11.4,
                value=4.11,
            ),
        ),
        # String
        pytest.param(
            SeriesReplace(
                series=pd.Series(["A", "B", "CG", "ACDE"] * 4),
                to_replace="A",
                value="B",
            ),
        ),
        # Bool
        pytest.param(
            SeriesReplace(
                series=pd.Series([False, True, True, False, False]),
                to_replace=True,
                value=False,
            ),
        ),
        # List of strings
        pytest.param(
            SeriesReplace(
                pd.Series(["abc", "def"] * 4),
                to_replace=["abc"],
                value=["ghi"],
            ),
        ),
        # pd.Categorical pass
        pytest.param(
            SeriesReplace(
                pd.Series(pd.Categorical([1, 2, 5, None, 2], ordered=True)),
                to_replace=5,
                value=15,
            ),
        ),
        # to_replace=dictionary success
        pytest.param(
            SeriesReplace(
                pd.Series([1, 2, 3, 4] * 4),
                to_replace={1: 10, 2: 20, 3: 30},
                value=None,
            ),
        ),
    ],
)
def test_replace_types_supported(series_replace):
    """Run series.replace on particular types that all pass."""
    series = series_replace.series
    to_replace = series_replace.to_replace
    value = series_replace.value
    if value is None:
        check_func(series_replace_impl, (series, to_replace))
    else:
        check_func(series_replace_value_impl, (series, to_replace, value))


@pytest.mark.slow
@pytest.mark.parametrize(
    "series_replace",
    [
        # Timestamp
        pytest.param(
            SeriesReplace(
                series=pd.Series(
                    pd.date_range(start="2018-04-24", end="2018-04-29", periods=5)
                ),
                to_replace=pd.Timestamp("2018-04-24 00:00:00"),
                value=pd.Timestamp("2020-01-01 01:01:01"),
            ),
        ),
        # List of list of ints
        pytest.param(
            SeriesReplace(
                pd.Series([[1, 2], [3], [5, 4, 6], [-1, 3, 4]]),
                to_replace=[3],
                value=[1],
            ),
        ),
        # pd.Categorical expected fail to_replace
        pytest.param(
            SeriesReplace(
                pd.Series(pd.Categorical([1, 2, 5, None, 2], ordered=True)),
                to_replace=pd.Categorical([5]),
                value=pd.Categorical([15]),
            ),
        ),
        # pd.Categorical expected fail value
        pytest.param(
            SeriesReplace(
                pd.Series(pd.Categorical([1, 2, 5, None, 2], ordered=True)),
                to_replace=5,
                value=pd.Categorical([15]),
            ),
        ),
    ],
)
def test_replace_types_unsupported(series_replace):
    """Run series.replace on particular types that all fail."""
    series = series_replace.series
    to_replace = series_replace.to_replace
    value = series_replace.value

    if any(isinstance(x, pd.Timestamp) for x in [to_replace, value]):
        message = "Not supported for types"
    elif any(isinstance(x, (pd.Categorical)) for x in [to_replace, value]) or any(
        isinstance(x, list) for x in series
    ):
        message = "only support with Scalar"
    elif (
        (isinstance(to_replace, int) and isinstance(value, float))
        or isinstance(to_replace, bool)
        and (isinstance(value, (int, float)) and not isinstance(value, bool))
    ):
        message = "cannot replace type"
    elif series.dtype is np.dtype("int64") and isinstance(to_replace, float):
        message = "'to_replace' type must match series type"

    with pytest.raises(BodoError, match=message):
        bodo.jit(series_replace_value_impl)(series, to_replace, value)


@pytest.mark.slow
def test_replace_float_int_scalar_scalar():
    series = pd.Series([1.0, 2.0, 3.0] * 4)
    to_replace = 1
    value = 2.0
    check_func(series_replace_value_impl, (series, to_replace, value))


@pytest.mark.slow
def test_replace_float_int_list_scalar():
    series = pd.Series([1.0, 2.0, 3.0] * 4)
    to_replace = [1, 3, 6]
    value = 4.0
    check_func(series_replace_value_impl, (series, to_replace, value))


@pytest.mark.slow
def test_replace_float_int_list_list():
    series = pd.Series([1.0, 2.0, 3.0] * 4)
    to_replace = [1, 3]
    value = [2.0, 4.0]
    check_func(series_replace_value_impl, (series, to_replace, value))


@pytest.mark.slow
def test_replace_string_int():
    series = pd.Series(["AZ", "BY", "CX"] * 4)
    to_replace = 1
    value = "DW"
    check_func(series_replace_value_impl, (series, to_replace, value))


@pytest.mark.slow
def test_replace_inf_nan():
    series = pd.Series([0, 1, 2, 3] * 4)
    to_replace = [np.inf, -np.inf]
    value = np.nan
    check_func(series_replace_value_impl, (series, to_replace, value))


@pytest.mark.slow
def test_series_concat(series_val, memory_leak_check):
    """test of concatenation of series.
    We convert to dataframe in order to reset the index.
    """

    def f(S1, S2):
        return pd.concat([S1, S2])

    S1 = series_val.copy()
    S2 = series_val.copy()
    df1 = pd.DataFrame({"A": S1.values})
    df2 = pd.DataFrame({"A": S2.values})
    if isinstance(series_val.values[0], list):
        check_func(
            f,
            (df1, df2),
            sort_output=True,
            reset_index=True,
            convert_columns_to_pandas=True,
        )
    else:
        check_func(f, (df1, df2), sort_output=True, reset_index=True)


@pytest.mark.slow
def test_series_between(memory_leak_check):
    def impl_inclusive(S):
        return S.between(1, 4)

    def impl(S):
        return S.between(1, 4, inclusive="neither")

    S = pd.Series([2, 0, 4, 8, np.nan], dtype="Float64")
    check_func(impl, (S,), check_dtype=False)
    check_func(impl_inclusive, (S,))


@pytest.mark.slow
def test_datetime_series_between(memory_leak_check):
    def impl(S):
        lower = pd.Timestamp(year=2020, month=10, day=5)
        upper = pd.Timestamp(year=2020, month=10, day=20)
        return S.between(lower, upper)

    datatime_arr = [datetime.datetime(year=2020, month=10, day=x) for x in range(1, 32)]
    S = pd.Series(datatime_arr)
    check_func(impl, (S,))

    datatime_arr_nan = [datetime.datetime(year=2020, month=10, day=1), np.nan]
    S_with_nan = pd.Series(datatime_arr)
    check_func(impl, (S_with_nan,))


@pytest.mark.slow
def test_series_concat_categorical(memory_leak_check):
    """test of concatenation of categorical series.
    TODO: refactor concat tests
    """

    def f(S1, S2):
        return pd.concat([S1, S2])

    S = pd.Series(["AA", "BB", "", "AA", None, "AA"], dtype="category")
    check_func(f, (S, S), sort_output=True, reset_index=True)


#   The code that we want to have.
#    bodo_f = bodo.jit(f)
#    pd.testing.assert_series_equal(
#        bodo_f(S1, S2), f(S1, S2), check_dtype=False, check_index_type=False
#    )
#    check_func(f, (S1, S2), reset_index=True)


# TODO: readd memory leak check when PDCategorical type constant lowering
# no longer leaks memory
@pytest.mark.slow
def test_dataframe_concat(series_val):
    """This is actually a dataframe test that adds empty
    column when missing
    """
    # Pandas converts Integer arrays to int object arrays when adding an all NaN
    # chunk, which we cannot handle in our parallel testing.
    if isinstance(series_val.dtype, pd.core.arrays.integer._IntegerDtype):
        return

    def f(df1, df2):
        return pd.concat([df1, df2])

    S1 = series_val.copy()
    S2 = series_val.copy()
    df1 = pd.DataFrame({"A": S1.values})
    df2 = pd.DataFrame({"B": S2.values})
    # Categorical data seems to revert to object arrays when concat in Python
    # As a result we will compute the output directly and do the comparison
    # We use the actual series_val dtype to capture ordered info.
    if isinstance(series_val.dtype, pd.CategoricalDtype):
        # Pandas 1.2.2 causes the conversion to obj to alter the representation
        # on dt64 and td64
        py_output = f(df1, df2)
        if series_val.dtype.categories.dtype in [
            np.dtype("datetime64[ns]"),
            np.dtype("timedelta64[ns]"),
        ]:
            py_output = py_output.astype(series_val.dtype.categories.dtype)
        py_output = py_output.astype(series_val.dtype)
    else:
        py_output = None
    if isinstance(series_val.values[0], list):
        check_func(
            f,
            (df1, df2),
            sort_output=True,
            reset_index=True,
            convert_columns_to_pandas=True,
            py_output=py_output,
        )
    else:
        check_func(
            f, (df1, df2), sort_output=True, reset_index=True, py_output=py_output
        )


# TODO: readd memory leak check when PDCategorical type constant lowering
# no longer leaks memory
@pytest.mark.slow
def test_dataframe_concat_cat_dynamic():
    """This is actually a dataframe test that adds empty
    column when missing if categories change dynamically
    """

    def f(df1, df2, to_replace, value):
        df1.replace(to_replace, value)
        df2.replace(to_replace, value)
        return pd.concat([df1, df2])

    S = pd.Series(["AA", "BB", "", "AA", None, "AA"], dtype="category")
    S1 = S.copy()
    S2 = S.copy()
    df1 = pd.DataFrame({"A": S1.values})
    df2 = pd.DataFrame({"B": S2.values})
    to_replace = "AA"
    value = "AAAA"
    # Categorical data seems to revert to object arrays when concat in Python
    # As a result we will compute the output directly and do the comparison
    py_output = f(df1, df2, to_replace, value).astype("category")
    check_func(
        f,
        (df1, df2, to_replace, value),
        sort_output=True,
        reset_index=True,
        py_output=py_output,
    )


def test_series_concat_convert_to_nullable(memory_leak_check):
    """make sure numpy integer/bool arrays are converted to nullable integer arrays in
    concatenation properly
    """

    def impl1(S1, S2):
        return pd.concat([S1, S2])

    # Integer case
    S1 = pd.Series([3, 2, 1, -4, None, 11, 21, 31, None] * 2, dtype="Int64")
    S2 = pd.Series(np.arange(11) * 2, dtype="int32")
    check_func(impl1, (S1, S2), sort_output=True, reset_index=True)

    # calling pd.Series inside the function to force values to be Numpy bool since Bodo
    # assumes nullable for bool input arrays during unboxing by default
    def impl2(S1, S2):
        return pd.concat([S1, pd.Series(S2)])

    # Boolean case
    S1 = pd.Series(
        [True, False, False, True, None, False, False, True, None] * 2, dtype="boolean"
    )
    S2 = pd.Series([True, False, False, True, True, False, False], dtype="bool").values
    check_func(impl2, (S1, S2), sort_output=True, reset_index=True)

    # Float case
    if bodo.libs.float_arr_ext._use_nullable_float:
        S1 = pd.Series([3, 2, 1, -4, None, 11, 21, 31, None] * 2, dtype="Float64")
        S2 = pd.Series(np.arange(11) * 2.1, dtype="float32")
        check_func(impl1, (S1, S2), sort_output=True, reset_index=True)


def test_series_notna(series_val, memory_leak_check):
    def test_impl(S):
        return S.notna()

    check_func(test_impl, (series_val,))


def test_series_notnull(series_val, memory_leak_check):
    def test_impl(S):
        return S.notnull()

    check_func(test_impl, (series_val,))


@pytest.mark.slow
def test_box(series_val, memory_leak_check):
    # unbox and box
    def impl(S):
        return S

    check_func(impl, (series_val,))


@pytest.mark.slow
def test_series_index(series_val, memory_leak_check):
    def test_impl(S):
        return S.index

    check_func(test_impl, (series_val,))


def test_series_index_none(memory_leak_check):
    def test_impl():
        S = pd.Series([1, 4, 8])
        return S.index

    check_func(test_impl, (), only_seq=True)


@pytest.mark.slow
def test_series_values(series_val, memory_leak_check):
    def test_impl(S):
        return S.values

    check_func(test_impl, (series_val,))


@pytest.mark.slow
def test_series_dtype(numeric_series_val, memory_leak_check):
    def test_impl(S):
        return S.dtype

    check_func(test_impl, (numeric_series_val,))


@pytest.mark.slow
def test_series_shape(series_val, memory_leak_check):
    def test_impl(S):
        return S.shape

    check_func(test_impl, (series_val,))


@pytest.mark.slow
def test_series_ndim(series_val, memory_leak_check):
    def test_impl(S):
        return S.ndim

    check_func(test_impl, (series_val,))


@pytest.mark.slow
def test_series_size(series_val, memory_leak_check):
    def test_impl(S):
        return S.size

    check_func(test_impl, (series_val,))


@pytest.mark.slow
def test_series_T(series_val, memory_leak_check):
    def test_impl(S):
        return S.T

    check_func(test_impl, (series_val,))


def test_series_hasnans(series_val, memory_leak_check):
    def test_impl(S):
        return S.hasnans

    check_func(test_impl, (series_val,))


def test_series_empty(series_val, memory_leak_check):
    def test_impl(S):
        return S.empty

    check_func(test_impl, (series_val,))


def test_series_dtypes(numeric_series_val, memory_leak_check):
    def test_impl(S):
        return S.dtypes

    check_func(test_impl, (numeric_series_val,))


@pytest.mark.slow
def test_series_name(series_val, memory_leak_check):
    def test_impl(S):
        return S.name

    check_func(test_impl, (series_val,))


def test_series_astype_numeric(numeric_series_val, memory_leak_check):
    # datetime can't be converted to float
    if numeric_series_val.dtype == np.dtype("datetime64[ns]"):
        return

    def test_impl(S):
        return S.astype(np.float64)

    check_func(test_impl, (numeric_series_val,))


# TODO: add memory_leak_check
def test_series_astype_str(series_val):

    # not supported for list(string) and array(item)
    if isinstance(series_val.values[0], list):
        return

    # not supported for Datetime.date yet, TODO: support and test
    if isinstance(series_val.values[0], datetime.date):
        return

    # XXX str(float) not consistent with Python yet
    if series_val.dtype in (
        np.float32,
        np.float64,
        pd.Float32Dtype(),
        pd.Float64Dtype(),
    ):
        return

    if series_val.dtype == np.dtype("datetime64[ns]"):
        return

    if series_val.dtype == np.dtype("timedelta64[ns]"):
        return

    # XXX Pandas 1.2. Null value in categorical input gets
    # set as a null value and not "nan". It is unclear if
    # this is unique to certain types. np.nan is still
    # converted to "nan"
    if isinstance(series_val.dtype, pd.CategoricalDtype):
        return

    def test_impl1(S):
        return S.astype(str)

    def test_impl2(S):
        return S.astype(pd.StringDtype())

    # In pandas, binarySeries.astype(str) will call str on each of the bytes objects,
    # returning a string array.
    # For example:
    # Pandas behavior:
    #   pd.Series([b"a", b"c"]).astypes(str) == pd.Series(["b'a'", "b'c'"])
    # Desired Bodo Behavior:
    #   pd.Series([b"a", b"c"]).astypes(str) == pd.Series(["a", "c"])
    if isinstance(series_val.values[0], bytes):
        expected_out = series_val.apply(
            lambda bin_val: bin_val if pd.isna(bin_val) else bin_val.decode("utf-8")
        )
        check_func(test_impl1, (series_val,), py_output=expected_out)
        check_func(test_impl2, (series_val,), py_output=expected_out)
        return

    check_func(test_impl1, (series_val,))
    # Bodo doesn't unbox string dtypes as pd.StringDtype(), so provide a py_output
    check_func(test_impl2, (series_val,), py_output=series_val.astype(str))


@pytest.mark.slow
def test_series_int_astype_str_long(memory_leak_check):
    """test a Series with a lot of int/NA values to make sure string array capacity is
    adjusted properly.
    """

    def impl(S):
        return S.astype(str)

    S = pd.Series([11111, 22, None, -3222222, None, 445] * 1000, dtype="Int64")
    check_func(impl, (S,))


def test_series_astype_int_arr(numeric_series_val, memory_leak_check):
    # only integers can be converted safely
    if not pd.api.types.is_integer_dtype(numeric_series_val):
        return

    def test_impl(S):
        return S.astype("Int64")

    check_func(test_impl, (numeric_series_val,))


def test_series_astype_int_arr_nullable(memory_leak_check):
    """Test for converting between nullable integer arrays
    (TODO(ehsan): add nulls to numeric_series_val and remove this extra test)
    """

    def test_impl(S):
        return S.astype("Int64")

    S = pd.Series([1, 4, None, 5, 3, 1] * 2)
    check_func(test_impl, (S,))


@pytest.mark.parametrize(
    "S",
    [
        pd.Series([1.0, np.nan, 3.0, 2.0], dtype="float32"),
        pytest.param(
            pd.Series([1.0, np.nan, 3.0, 2.0], dtype="float64"), marks=pytest.mark.slow
        ),
    ],
)
def test_series_astype_float_to_int_arr(S, memory_leak_check):
    """Test converting float data to nullable int array"""
    # TODO: support converting string to int

    def test_impl(S):
        return S.astype("Int64")

    check_func(test_impl, (S,))


@pytest.mark.parametrize(
    "S",
    [
        pytest.param(
            pd.Series([True, False, False, True, True]), marks=pytest.mark.slow
        ),
        pd.Series([True, False, False, np.nan, True]),
    ],
)
def test_series_astype_bool_arr(S, memory_leak_check):
    # TODO: int, Int

    def test_impl(S):
        return S.astype("float32")

    check_func(test_impl, (S,))


@pytest.mark.parametrize(
    "S",
    [
        pd.Series(["a", "b", "aa", "bb", "A", "a", "BB"]),
        pd.Series([1, 2, 41, 2, 1, 4, 2, 1, 1, 25, 5, 3]),
    ],
)
def test_series_drop_duplicates(S, memory_leak_check):
    def test_impl(S):
        return S.drop_duplicates()

    check_func(test_impl, (S,), sort_output=True)


@pytest.mark.parametrize(
    "S", [pd.Series(["BB", "C", "A", None, "A", "BBB", None, "C", "BB", "A"])]
)
# TODO: add memory_leak_check
def test_series_astype_cat(S):
    def test_impl(S, ctype):
        return S.astype(ctype)

    def test_impl2(S):
        return S.astype("category")

    # full dtype values
    ctype = pd.CategoricalDtype(S.dropna().unique())
    check_func(test_impl, (S, ctype))
    # partial dtype values to force NA
    new_cats = ctype.categories.tolist()[:-1]
    np.random.seed(0)
    np.random.shuffle(new_cats)
    ctype = pd.CategoricalDtype(new_cats)
    check_func(test_impl, (S, ctype))
    # dtype not specified
    check_func(test_impl2, (S,))


@pytest.mark.slow
@pytest.mark.parametrize(
    "S", [pd.Series(["A", "BB", "A", "BBB", "BB", "A"]).astype("category")]
)
def test_series_cat_box(S, memory_leak_check):
    def test_impl(S):
        return S

    check_func(test_impl, (S,))


@pytest.mark.parametrize(
    "S", [pd.Series(["A", "BB", "A", "BBB", "BB", "A"]).astype("category")]
)
def test_series_cat_comp(S, memory_leak_check):
    def test_impl(S):
        return S == "BB"

    check_func(test_impl, (S,))


@pytest.mark.parametrize(
    "S",
    [
        pd.Series(["BB", "C", "A", None, "A", "BBB", None, "C", "BB", "A"]).astype(
            "category"
        )
    ],
)
def test_series_cat_codes(S, memory_leak_check):
    def test_impl(S):
        return S.cat.codes

    check_func(test_impl, (S,))


def test_series_copy(series_val, memory_leak_check):
    # TODO: test deep/shallow cases properly
    def test_deep(S):
        return S.copy()

    def test_shallow(S):
        return S.copy(deep=False)

    check_func(test_deep, (series_val,))
    check_func(test_shallow, (series_val,))
    check_func(test_deep, (series_val.values,))


def test_series_to_list(series_val, memory_leak_check):
    """Test Series.to_list(): non-float NAs throw a runtime error since can't be
    represented in lists.
    """
    # TODO: [BE-498] Correctly convert nan
    def impl(S):
        return S.to_list()

    if series_val.hasnans and not isinstance(series_val.iat[0], np.floating):
        message = "Not supported for NA values"
        with pytest.raises(ValueError, match=message):
            bodo.jit(impl)(series_val)
    # Bodo uses nan for nullable float arrays due to type stability
    elif series_val.dtype in (
        pd.Float32Dtype(),
        pd.Float64Dtype(),
        np.float32,
        np.float64,
    ):
        check_func(
            impl,
            (series_val,),
            only_seq=True,
            py_output=series_val.astype("float64").to_list(),
        )
    else:
        check_func(impl, (series_val,), only_seq=True)


def test_series_to_numpy(numeric_series_val, memory_leak_check):
    def test_impl(S):
        return S.to_numpy()

    check_func(test_impl, (numeric_series_val,))


# TODO: add memory_leak_check (it leaks with Decimal array)
@pytest.mark.smoke
def test_series_iat_getitem(series_val):
    def test_impl(S):
        return S.iat[2]

    bodo_func = bodo.jit(test_impl)
    bodo_out = bodo_func(series_val)
    py_out = test_impl(series_val)
    _test_equal(bodo_out, py_out)
    # fix distributed
    # check_func(test_impl, (series_val,))


@pytest.mark.slow
def test_series_iat_getitem_datetime(memory_leak_check):
    """Test to check that series.iat properly converts
    unboxes datetime.date, dt64, and td64"""

    def test_impl(S):
        return S.iat[2]

    date_series = pd.Series(pd.date_range("2020-01-14", "2020-01-17").date)
    datetime_series = pd.Series(pd.date_range("2020-01-14", "2020-01-17"))
    timedelta_series = pd.Series(
        np.append(
            [datetime.timedelta(days=5, seconds=4, weeks=4)] * 2,
            [datetime.timedelta(seconds=202, hours=5)] * 2,
        )
    )
    # TODO: Ensure parallel implementation works
    check_func(test_impl, (date_series,), dist_test=False)
    check_func(test_impl, (datetime_series,), dist_test=False)
    check_func(test_impl, (timedelta_series,), dist_test=False)


@pytest.mark.smoke
def test_series_iat_setitem(series_val, memory_leak_check):

    val = series_val.iat[0]

    def test_impl(S, val):
        S.iat[2] = val
        return S

    # string setitem not supported yet
    if isinstance(series_val.iat[0], str) and not isinstance(
        series_val.dtype, pd.CategoricalDtype
    ):
        with pytest.raises(BodoError, match="Series string setitem not supported yet"):
            bodo.jit(test_impl)(series_val, val)
        return
    elif isinstance(series_val.iat[0], bytes) and not isinstance(
        series_val.dtype, pd.CategoricalDtype
    ):
        with pytest.raises(BodoError, match="Series binary setitem not supported yet"):
            bodo.jit(test_impl)(series_val, val)
        return

    # not supported for list(string) and array(item)
    if isinstance(series_val.values[0], list):
        with pytest.raises(
            BodoError,
            match="Series setitem not supported for Series with immutable array type .*",
        ):
            bodo.jit(test_impl)(series_val, val)
        return

    # TODO: Test distributed implementation
    check_func(test_impl, (series_val, val), copy_input=True, dist_test=False)


@pytest.mark.slow
def test_series_iat_setitem_datetime(memory_leak_check):
    """
    Test that series.iat supports datetime.date, datetime.datetime, and datetime.timedelta
    scalar values.
    """

    def test_impl(S, val):
        S.iat[2] = val
        return S

    S1 = pd.Series(pd.date_range(start="2018-04-24", end="2018-04-29", periods=5))
    val1 = datetime.datetime(2011, 11, 4, 0, 0)
    S2 = pd.Series(pd.date_range(start="2018-04-24", end="2018-04-29", periods=5).date)
    val2 = datetime.date(2002, 12, 26)
    S3 = pd.Series(
        [
            datetime.timedelta(3, 3, 3),
            datetime.timedelta(2, 2, 2),
            datetime.timedelta(1, 1, 1),
            np.nan,
            datetime.timedelta(5, 5, 5),
        ]
    )
    val3 = datetime.timedelta(days=64, seconds=29156, microseconds=10)
    check_func(test_impl, (S1, val1), copy_input=True, dist_test=False)
    check_func(test_impl, (S2, val2), copy_input=True, dist_test=False)
    check_func(test_impl, (S3, val3), copy_input=True, dist_test=False)


@pytest.mark.smoke
def test_series_iloc_getitem_int(series_val):
    def test_impl(S):
        return S.iloc[2]

    bodo_func = bodo.jit(test_impl)
    bodo_out = bodo_func(series_val)
    py_out = test_impl(series_val)
    _test_equal(bodo_out, py_out)
    # fix distributed
    # check_func(test_impl, (series_val,))


@pytest.mark.slow
def test_series_iloc_getitem_datetime(memory_leak_check):
    """Test to check that series.iloc properly converts
    unboxes datetime.date, dt64, and td64"""

    def test_impl(S):
        return S.iloc[2]

    date_series = pd.Series(pd.date_range("2020-01-14", "2020-01-17").date)
    datetime_series = pd.Series(pd.date_range("2020-01-14", "2020-01-17"))
    timedelta_series = pd.Series(
        np.append(
            [datetime.timedelta(days=5, seconds=4, weeks=4)] * 2,
            [datetime.timedelta(seconds=202, hours=5)] * 2,
        )
    )
    # TODO: Ensure parallel implementation works
    check_func(test_impl, (date_series,), dist_test=False)
    check_func(test_impl, (datetime_series,), dist_test=False)
    check_func(test_impl, (timedelta_series,), dist_test=False)


def test_series_iloc_getitem_slice(series_val, memory_leak_check):
    def test_impl(S):
        return S.iloc[1:4]

    # fix distributed
    check_func(test_impl, (series_val,), check_dtype=False, dist_test=False)


def test_series_iloc_getitem_array_int(series_val, memory_leak_check):
    def test_impl(S):
        return S.iloc[[1, 3]]

    check_func(test_impl, (series_val,), check_dtype=False, dist_test=False)


def test_series_iloc_getitem_array_bool(series_val, memory_leak_check):
    """Tests that getitem with Series.iloc works with a Boolean List index"""

    def test_impl(S):
        return S.iloc[[True, True, False, True, False]]

    # Make sure cond always matches length
    if len(series_val) > 5:
        series_val = series_val[:5]

    check_func(test_impl, (series_val,), check_dtype=False, dist_test=False)


def test_series_loc_getitem_array_bool(series_val, memory_leak_check):
    """Tests that getitem with Series.loc works with a Boolean List index"""

    def test_impl(S):
        return S.loc[[True, True, False, True, False]]

    # Make sure cond always matches length
    if len(series_val) > 5:
        series_val = series_val[:5]

    check_func(test_impl, (series_val,), check_dtype=False, dist_test=False)


@pytest.mark.slow
def test_series_loc_getitem_int_range(memory_leak_check):
    def test_impl(S):
        return S.loc[2]

    S = pd.Series(
        data=[1, 12, 1421, -241, 4214], index=pd.RangeIndex(start=0, stop=5, step=1)
    )
    check_func(test_impl, (S,))


@pytest.mark.skipif(
    bodo.hiframes.boxing._use_dict_str_type, reason="not supported for dict string type"
)
@pytest.mark.slow
def test_series_loc_setitem_array_bool(series_val, memory_leak_check):
    """Tests that setitem with Series.loc works with a Boolean List index"""

    def test_impl(S, val):
        S.loc[[True, True, False, True, False]] = val
        return S

    # Make sure cond always matches length
    if len(series_val) > 5:
        series_val = series_val[:5]

    err_msg = "Series setitem not supported for Series with immutable array type .*"

    # Test with an array
    val = series_val.iloc[0:3].values.copy()
    if isinstance(series_val.iat[0], list):
        with pytest.raises(BodoError, match=err_msg):
            bodo.jit(test_impl)(series_val.copy(deep=True), val)
    elif isinstance(series_val.iat[0], bytes):
        with pytest.raises(
            BodoError,
            match="Series binary setitem not supported yet",
        ):
            bodo.jit(test_impl)(series_val.copy(deep=True), val)
    else:
        check_func(
            test_impl,
            (series_val, val),
            check_dtype=False,
            copy_input=True,
            dist_test=False,
        )
    # Test with a list
    val = list(val)
    # avoid pd.NA typing issues for nullable float arrays
    if series_val.dtype in (
        pd.Float32Dtype(),
        pd.Float64Dtype(),
        np.float32,
        np.float64,
    ):
        pass
    elif isinstance(series_val.iat[0], list):
        with pytest.raises(BodoError, match=err_msg):
            bodo.jit(test_impl)(series_val.copy(deep=True), val)
    elif isinstance(series_val.iat[0], bytes):
        with pytest.raises(
            BodoError,
            match="Series binary setitem not supported yet",
        ):
            bodo.jit(test_impl)(series_val.copy(deep=True), val)
    else:
        check_func(
            test_impl,
            (series_val, val),
            check_dtype=False,
            copy_input=True,
            dist_test=False,
        )
    # Test with a scalar
    val = val[0]
    if isinstance(series_val.iat[0], list):
        with pytest.raises(BodoError, match=err_msg):
            bodo.jit(test_impl)(series_val.copy(deep=True), val)
    elif isinstance(series_val.iat[0], bytes):
        with pytest.raises(
            BodoError,
            match="Series binary setitem not supported yet",
        ):
            bodo.jit(test_impl)(series_val.copy(deep=True), val)
    else:
        check_func(
            test_impl,
            (series_val, val),
            check_dtype=False,
            copy_input=True,
            dist_test=False,
        )


def test_series_diff(numeric_series_val, memory_leak_check):
    """test Series.diff()"""
    # # Pandas as of 1.2.2 is buggy for uint8 and produces wrong results
    if numeric_series_val.dtype == np.uint8:
        return

    def impl(S):
        return S.diff()

    # TODO: Support nullable arrays
    if isinstance(
        numeric_series_val.dtype,
        (pd.core.arrays.integer._IntegerDtype, pd.Float32Dtype, pd.Float64Dtype),
    ):
        with pytest.raises(
            BodoError, match="Series.diff.* column input type .* not supported"
        ):
            bodo.jit(impl)(numeric_series_val)
    else:
        check_func(impl, (numeric_series_val,))


@pytest.mark.slow
def test_series_iloc_setitem_datetime_scalar(memory_leak_check):
    """
    Test that series.iloc supports datetime.date, datetime.datetime, and datetime.timedelta
    scalar values.
    """

    def test_impl(S, idx, val):
        S.iloc[idx] = val
        return S

    S1 = pd.Series(pd.date_range(start="2018-04-24", end="2018-04-29", periods=5))
    val1 = datetime.datetime(2011, 11, 4, 0, 0)
    S2 = pd.Series(pd.date_range(start="2018-04-24", end="2018-04-29", periods=5).date)
    val2 = datetime.date(2002, 12, 26)
    S3 = pd.Series(
        [
            datetime.timedelta(3, 3, 3),
            datetime.timedelta(2, 2, 2),
            datetime.timedelta(1, 1, 1),
            np.nan,
            datetime.timedelta(5, 5, 5),
        ]
    )
    val3 = datetime.timedelta(days=64, seconds=29156, microseconds=10)

    arr_idx = np.array([True, True, False, True, False])

    for idx in (1, slice(1, 4), arr_idx, list(arr_idx)):
        check_func(test_impl, (S1, idx, val1), copy_input=True, dist_test=False)
        check_func(test_impl, (S2, idx, val2), copy_input=True, dist_test=False)
        check_func(test_impl, (S3, idx, val3), copy_input=True, dist_test=False)


@pytest.mark.smoke
def test_series_iloc_setitem_int(series_val, memory_leak_check):
    """
    Test setitem for Series.iloc with int idx.
    """

    val = series_val.iat[0]

    def test_impl(S, val):
        S.iloc[2] = val
        # print(S) TODO: fix crash
        return S

    # string setitem not supported yet
    if isinstance(series_val.iat[0], str) and not isinstance(
        series_val.dtype, pd.CategoricalDtype
    ):
        with pytest.raises(BodoError, match="Series string setitem not supported yet"):
            bodo.jit(test_impl)(series_val, val)
        return
    elif isinstance(series_val.iat[0], bytes) and not isinstance(
        series_val.dtype, pd.CategoricalDtype
    ):
        with pytest.raises(BodoError, match="Series binary setitem not supported yet"):
            bodo.jit(test_impl)(series_val, val)
        return

    # not supported for list(string) and array(item)
    if isinstance(series_val.values[0], list):
        with pytest.raises(
            BodoError,
            match="Series setitem not supported for Series with immutable array type .*",
        ):
            bodo.jit(test_impl)(series_val, val)
        return

    # TODO: Test distributed implementation
    check_func(test_impl, (series_val, val), copy_input=True, dist_test=False)


@pytest.mark.skipif(
    bodo.hiframes.boxing._use_dict_str_type, reason="not supported for dict string type"
)
def test_series_iloc_setitem_list_bool(series_val, memory_leak_check):
    """
    Test setitem for Series.iloc and Series with bool arr/list idx.
    """

    if isinstance(series_val.dtype, pd.CategoricalDtype):
        # TODO: [BE-49] support conversion between dt64/Timestamp
        return

    def test_impl(S, idx, val):
        S.iloc[idx] = val
        return S

    # test Series.values since it is used instead of iloc sometimes
    def test_impl2(S, idx, val):
        S.values[idx] = val
        return S

    idx = np.array([True, True, False, True, False] + [False] * (len(series_val) - 5))
    # value is array
    val = series_val.iloc[0:3].values.copy()  # values to avoid alignment
    if series_val.hasnans:
        # extra NA to keep dtype nullable like bool arr
        val[0] = None

    # string setitem not supported yet
    if isinstance(series_val.iat[0], str) and not isinstance(
        series_val.dtype, pd.CategoricalDtype
    ):
        with pytest.raises(BodoError, match="Series string setitem not supported yet"):
            bodo.jit(test_impl)(series_val, idx, val)
        check_func(test_impl2, (series_val, idx, val), copy_input=True, dist_test=False)

    # binary setitem not supported yet
    elif isinstance(series_val.iat[0], bytes) and not isinstance(
        series_val.dtype, pd.CategoricalDtype
    ):
        with pytest.raises(
            BodoError,
            match="Series binary setitem not supported yet",
        ):
            bodo.jit(test_impl)(series_val, idx, val)
        with pytest.raises(
            BodoError,
            match="Binary array setitem with index .* and value .* not supported.",
        ):
            bodo.jit(test_impl2)(series_val, idx, val)

    # not supported for list(string) and array(item)
    elif isinstance(series_val.values[0], list):
        with pytest.raises(
            BodoError,
            match="Series setitem not supported for Series with immutable array type .*",
        ):
            bodo.jit(test_impl)(series_val, idx, val)
        with pytest.raises(
            BodoError,
            match="only setitem with scalar index is currently supported for list arrays",
        ):
            bodo.jit(test_impl2)(series_val, idx, val)

    else:
        # TODO: Test distributed implementation
        check_func(test_impl, (series_val, idx, val), copy_input=True, dist_test=False)
        check_func(test_impl2, (series_val, idx, val), copy_input=True, dist_test=False)

    val = series_val.dropna().iloc[0:3].to_list()

    # string setitem not supported yet
    if isinstance(series_val.iat[0], str) and not isinstance(
        series_val.dtype, pd.CategoricalDtype
    ):
        with pytest.raises(BodoError, match="Series string setitem not supported yet"):
            bodo.jit(test_impl)(series_val, idx, val)
        with pytest.raises(
            BodoError,
            match="StringArray setitem with index .* and value .* not supported.",
        ):
            bodo.jit(test_impl2)(series_val, idx, val)

    # binary setitem not supported for list of bytes
    elif isinstance(series_val.iat[0], bytes) and not isinstance(
        series_val.dtype, pd.CategoricalDtype
    ):
        with pytest.raises(
            BodoError,
            match="Series binary setitem not supported yet",
        ):
            bodo.jit(test_impl)(series_val, idx, val)
        with pytest.raises(
            BodoError,
            match="Binary array setitem with index .* and value .* not supported.",
        ):
            bodo.jit(test_impl2)(series_val, idx, val)

    # not supported for list(string) and array(item)
    elif isinstance(series_val.values[0], list):
        with pytest.raises(
            BodoError,
            match="Series setitem not supported for Series with immutable array type .*",
        ):
            bodo.jit(test_impl)(series_val, idx, val)
        with pytest.raises(
            BodoError,
            match="only setitem with scalar index is currently supported for list arrays",
        ):
            bodo.jit(test_impl2)(series_val, idx, val)

    else:
        # TODO: Test distributed implementation
        # Pandas promotes to float64. This is most likely a bug and Bodo
        # keeps the same, smaller type.
        if series_val.dtype == np.uint8:
            check_dtype = False
        else:
            check_dtype = True
        check_func(
            test_impl,
            (series_val, idx, val),
            copy_input=True,
            dist_test=False,
            check_dtype=check_dtype,
        )
        # setitem dt64/td64 array with list Timestamp/Timedelta values not supported
        if not (
            series_val.dtype == np.dtype("datetime64[ns]")
            or series_val.dtype == np.dtype("timedelta64[ns]")
        ):
            check_func(
                test_impl2,
                (series_val, idx, val),
                copy_input=True,
                dist_test=False,
                check_dtype=check_dtype,
            )


def test_series_iloc_setitem_scalar(series_val, memory_leak_check):
    """
    Tests that series.iloc setitem with array/index/slice
    properly supports a Scalar RHS
    """

    if isinstance(series_val.dtype, pd.CategoricalDtype):
        # TODO: [BE-49] support setitem array idx, scalar value for Categorical arrays
        return

    val = series_val.iat[0]

    def test_impl(S, idx, val):
        S.iloc[idx] = val
        return S

    arr_idx = np.array(
        [True, True, False, True, False] + [False] * (len(series_val) - 5)
    )

    for idx in (slice(1, 4), arr_idx, list(arr_idx)):
        # string setitem not supported yet
        if isinstance(series_val.iat[0], str) and not isinstance(
            series_val.dtype, pd.CategoricalDtype
        ):
            with pytest.raises(
                BodoError, match="Series string setitem not supported yet"
            ):
                bodo.jit(test_impl)(series_val, idx, val)
            return
        elif isinstance(series_val.iat[0], bytes) and not isinstance(
            series_val.dtype, pd.CategoricalDtype
        ):
            with pytest.raises(
                BodoError, match="Series binary setitem not supported yet"
            ):
                bodo.jit(test_impl)(series_val, idx, val)
            return

        # not supported for list(string) and array(item)
        elif isinstance(series_val.values[0], list):
            with pytest.raises(
                BodoError,
                match="Series setitem not supported for Series with immutable array type .*",
            ):
                bodo.jit(test_impl)(series_val, idx, val)
            return
        else:
            # TODO: Test distributed implementation
            check_func(
                test_impl,
                (series_val, idx, val),
                copy_input=True,
                dist_test=False,
            )


def test_series_iloc_setitem_slice(series_val, memory_leak_check):
    """
    Test setitem for Series.iloc and Series.values with slice idx.
    """

    if isinstance(series_val.dtype, pd.CategoricalDtype):
        # TODO: [BE-49] support conversion between dt64/Timestamp
        return

    def test_impl(S, val):
        S.iloc[1:4] = val
        return S

    # test Series.values since it is used instead of iloc sometimes
    def test_impl2(S, val):
        S.values[1:4] = val
        return S

    # value is array
    val = series_val.iloc[0:3].values.copy()  # values to avoid alignment
    if series_val.hasnans:
        # extra NA to keep dtype nullable like bool arr
        val[0] = None

    # string setitem not supported yet
    if isinstance(series_val.iat[0], str) and not isinstance(
        series_val.dtype, pd.CategoricalDtype
    ):
        with pytest.raises(BodoError, match="Series string setitem not supported yet"):
            bodo.jit(test_impl)(series_val, val)
        # TODO: Prevent writing to immutable array for test_impl2.

    # binary setitem not supported yet
    elif isinstance(series_val.iat[0], bytes) and not isinstance(
        series_val.dtype, pd.CategoricalDtype
    ):
        with pytest.raises(BodoError, match="Series binary setitem not supported yet"):
            bodo.jit(test_impl)(series_val, val)
        # TODO: Prevent writing to immutable array for test_impl2.

    # not supported for list(string) and array(item)
    elif isinstance(series_val.values[0], list):
        with pytest.raises(
            BodoError,
            match="Series setitem not supported for Series with immutable array type .*",
        ):
            bodo.jit(test_impl)(series_val, val)
        # TODO: Prevent writing to immutable array for test_impl2.
    else:
        check_func(
            test_impl,
            (series_val, val),
            copy_input=True,
            check_dtype=False,
            dist_test=False,
        )
        check_func(
            test_impl2,
            (series_val, val),
            copy_input=True,
            check_dtype=False,
            dist_test=False,
        )

    # value is a list

    val = series_val.dropna().iloc[0:3].to_list()

    # string setitem not supported yet
    if isinstance(series_val.iat[0], str) and not isinstance(
        series_val.dtype, pd.CategoricalDtype
    ):
        with pytest.raises(BodoError, match="Series string setitem not supported yet"):
            bodo.jit(test_impl)(series_val, val)
        # TODO: Prevent writing to immutable array for test_impl2.

    # binary setitem not supported yet
    elif isinstance(series_val.iat[0], bytes) and not isinstance(
        series_val.dtype, pd.CategoricalDtype
    ):
        with pytest.raises(
            BodoError,
            match="Series binary setitem not supported yet",
        ):
            bodo.jit(test_impl)(series_val, val)

    # not supported for list(string) and array(item)
    elif isinstance(series_val.values[0], list):
        with pytest.raises(
            BodoError,
            match="Series setitem not supported for Series with immutable array type .*",
        ):
            bodo.jit(test_impl)(series_val, val)
        # TODO: Prevent writing to immutable array for test_impl2.
    else:
        check_func(
            test_impl,
            (series_val, val),
            copy_input=True,
            check_dtype=False,
            dist_test=False,
        )
        # setitem dt64/td64 array with list Timestamp/Timedelta values not supported
        if not (
            series_val.dtype == np.dtype("datetime64[ns]")
            or series_val.dtype == np.dtype("timedelta64[ns]")
        ):
            check_func(
                test_impl2,
                (series_val, val),
                copy_input=True,
                check_dtype=False,
                dist_test=False,
            )


@pytest.mark.parametrize("idx", [[1, 3], np.array([1, 3]), pd.Series([1, 3])])
def test_series_iloc_setitem_list_int(series_val, idx, memory_leak_check):
    """
    Test setitem for Series.iloc and Series.values with list/array
    of ints idx.
    """

    if isinstance(series_val.dtype, pd.CategoricalDtype):
        # TODO: [BE-49] support conversion between dt64/Timestamp
        return

    def test_impl(S, val, idx):
        S.iloc[idx] = val
        return S

    # test Series.values which may be used as alternative to Series.iloc for setitem
    # tests underlying array setitem essentially
    def test_impl2(S, val, idx):
        S.values[idx] = val
        return S

    # value is an array
    val = series_val.iloc[0:2].values.copy()  # values to avoid alignment
    if series_val.hasnans:
        # extra NA to keep dtype nullable like bool arr
        val[0] = None
    # string setitem not supported yet
    if isinstance(series_val.iat[0], str) and not isinstance(
        series_val.dtype, pd.CategoricalDtype
    ):
        with pytest.raises(BodoError, match="Series string setitem not supported yet"):
            bodo.jit(test_impl)(series_val, val, idx)
        # TODO: Prevent writing to immutable array for test_impl2.

    # binary setitem not supported yet
    elif isinstance(series_val.iat[0], bytes) and not isinstance(
        series_val.dtype, pd.CategoricalDtype
    ):
        with pytest.raises(
            BodoError,
            match="Series binary setitem not supported yet",
        ):
            bodo.jit(test_impl)(series_val, val, idx)
        # TODO: Prevent writing to immutable array for test_impl2.

    # not supported for list(string) and array(item)
    elif isinstance(series_val.values[0], list):
        with pytest.raises(
            BodoError,
            match="Series setitem not supported for Series with immutable array type .*",
        ):
            bodo.jit(test_impl)(series_val, val, idx)
        # TODO: Prevent writing to immutable array for test_impl2.
    else:
        check_func(
            test_impl,
            (series_val, val, idx),
            copy_input=True,
            check_dtype=False,
            dist_test=False,
        )
        # setitem dt64/td64 array with list(int) idx not supported
        if not (
            series_val.dtype == np.dtype("datetime64[ns]")
            or series_val.dtype == np.dtype("timedelta64[ns]")
        ):
            check_func(
                test_impl2,
                (series_val, val, idx),
                copy_input=True,
                check_dtype=False,
                dist_test=False,
            )

    val = series_val.dropna().iloc[0:2].to_list()
    # string setitem not supported yet
    if isinstance(series_val.iat[0], str) and not isinstance(
        series_val.dtype, pd.CategoricalDtype
    ):
        with pytest.raises(BodoError, match="Series string setitem not supported yet"):
            bodo.jit(test_impl)(series_val, val, idx)
        # TODO: Prevent writing to immutable array for test_impl2.

    # binary setitem not supported yet
    elif isinstance(series_val.iat[0], bytes) and not isinstance(
        series_val.dtype, pd.CategoricalDtype
    ):
        with pytest.raises(
            BodoError,
            match="Series binary setitem not supported yet",
        ):
            bodo.jit(test_impl)(series_val, val, idx)
        # TODO: Prevent writing to immutable array for test_impl2.

    # not supported for list(string) and array(item)
    elif isinstance(series_val.values[0], list):
        with pytest.raises(
            BodoError,
            match="Series setitem not supported for Series with immutable array type .*",
        ):
            bodo.jit(test_impl)(series_val, val, idx)
        # TODO: Prevent writing to immutable array for test_impl2.
    else:
        check_func(
            test_impl,
            (series_val, val, idx),
            copy_input=True,
            check_dtype=False,
            dist_test=False,
        )
        # setitem dt64/td64 array with list(int) idx not supported
        if not (
            series_val.dtype == np.dtype("datetime64[ns]")
            or series_val.dtype == np.dtype("timedelta64[ns]")
        ):
            check_func(
                test_impl2,
                (series_val, val, idx),
                copy_input=True,
                check_dtype=False,
                dist_test=False,
            )


####### getitem tests ###############


# TODO: add memory_leak_check
@pytest.mark.smoke
def test_series_getitem_int(series_val):
    # timedelta setitem not supported yet
    if series_val.dtype == np.dtype("timedelta64[ns]"):
        return

    def test_impl(S):
        return S[2]

    bodo_func = bodo.jit(test_impl)
    # integer label-based indexing should raise error
    if type(series_val.index) in (pd.Int64Index, pd.UInt64Index):
        with pytest.raises(BodoError, match="not supported yet"):
            bodo_func(series_val)
    else:
        bodo_out = bodo_func(series_val)
        py_out = test_impl(series_val)
        _test_equal(bodo_out, py_out)


def test_series_getitem_slice(series_val, memory_leak_check):
    def test_impl(S):
        return S[1:4]

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(
        bodo_func(series_val),
        test_impl(series_val),
        check_dtype=False,
        check_categorical=False,
        check_index_type=False,
    )


@pytest.mark.parametrize("idx", [[1, 3], np.array([1, 3]), pd.Series([1, 3])])
def test_series_getitem_list_int(series_val, idx, memory_leak_check):
    def test_impl(S, idx):
        return S[idx]

    bodo_func = bodo.jit(test_impl)
    # integer label-based indexing should raise error
    if type(series_val.index) in (pd.Int64Index, pd.UInt64Index):
        with pytest.raises(BodoError, match="not supported yet"):
            bodo_func(series_val, idx)
    else:
        pd.testing.assert_series_equal(
            bodo_func(series_val, idx),
            test_impl(series_val, idx),
            check_dtype=False,
            check_categorical=False,
            check_index_type=False,
        )


def test_series_getitem_array_bool(series_val, memory_leak_check):
    def test_impl(S):
        return S[[True, True, False, True, False]]

    def test_impl2(S, cond):
        # using .values to test boolean_array
        return S[cond.values]

    # Make sure cond always matches length
    if len(series_val) > 5:
        series_val = series_val[:5]

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(
        bodo_func(series_val),
        test_impl(series_val),
        check_dtype=False,
        check_categorical=False,
        check_index_type=False,
    )
    cond = pd.Series([True, True, False, True, False])
    bodo_func = bodo.jit(test_impl2)
    pd.testing.assert_series_equal(
        bodo_func(series_val, cond),
        test_impl2(series_val, cond),
        check_dtype=False,
        check_categorical=False,
        check_index_type=False,
    )


############### setitem tests #################


@pytest.mark.smoke
def test_series_setitem_int(series_val, memory_leak_check):
    # not supported for list(string) and array(item)
    if isinstance(series_val.values[0], list):
        return

    # string setitem not supported yet
    if isinstance(series_val.iat[0], str) or isinstance(series_val.iat[0], bytes):
        return
    val = series_val.iat[0]

    def test_impl(S, val):
        S[2] = val
        return S

    bodo_func = bodo.jit(test_impl)
    # integer label-based indexing should raise error
    if type(series_val.index) in (pd.Int64Index, pd.UInt64Index):
        with pytest.raises(BodoError, match="not supported yet"):
            bodo_func(series_val, val)
    else:
        check_func(test_impl, (series_val, val), dist_test=False, copy_input=True)


def test_series_setitem_slice(series_val, memory_leak_check):
    # not supported for list(string) and array(item)
    if isinstance(series_val.values[0], list):
        return

    # string/binary setitem not supported yet. Binary JIRA: BE-1256
    if isinstance(series_val.iat[0], (str, bytes)):
        return

    val = series_val.iloc[0:3].values.copy()  # values to avoid alignment
    if series_val.hasnans:
        # extra NA to keep dtype nullable like bool arr
        val[0] = None

    def test_impl(S, val):
        S[1:4] = val
        return S

    check_func(test_impl, (series_val, val), dist_test=False, copy_input=True)


@pytest.mark.parametrize("idx", [[1, 4], np.array([1, 4]), pd.Series([1, 4])])
@pytest.mark.parametrize("list_val_arg", [True, False])
def test_series_setitem_list_int(series_val, idx, list_val_arg, memory_leak_check):
    # not supported for list(string) and array(item)
    if isinstance(series_val.values[0], list):
        return

    # string/binary setitem not supported yet. Binary JIRA: BE-1256
    if isinstance(series_val.iat[0], (str, bytes)):
        return
    val = series_val.iloc[0:2].values.copy()  # values to avoid alignment
    if list_val_arg:
        val = list(val)

    def test_impl(S, val, idx):
        S[idx] = val
        return S

    bodo_func = bodo.jit(test_impl)
    # integer label-based indexing should raise error
    if type(series_val.index) in (pd.Int64Index, pd.UInt64Index):
        with pytest.raises(BodoError, match="not supported yet"):
            bodo_func(series_val, val, idx)
    else:
        # Pandas coerces Series type to set values, so avoid low precision
        # TODO: warn or error?
        if list_val_arg and (
            series_val.dtype
            in (
                np.int8,
                np.uint8,
                np.int16,
                np.uint16,
                np.int32,
                np.uint32,
            )
        ):
            return
        check_func(test_impl, (series_val, val, idx), dist_test=False, copy_input=True)


############################ Series.loc indexing ##########################


@pytest.mark.skipif(
    bodo.hiframes.boxing._use_dict_str_type, reason="not supported for dict string type"
)
def test_series_loc_setitem_bool(memory_leak_check):
    """test Series.loc[bool_arr] setitem"""

    def impl(S, idx, val):
        S.loc[idx] = val
        return S

    S = pd.Series(["A", "AB", None, "ABC", "AB", "D", ""])
    check_func(
        impl, (S, S == "AB", "ABC"), copy_input=True, use_dict_encoded_strings=False
    )
    S = pd.Series([4.8, 1.1, 2.2, np.nan, 3.3, 4.4])
    check_func(impl, (S, S < 4.0, 1.8), copy_input=True, use_dict_encoded_strings=False)
    S = pd.Series([4, 1, 2, None, 3, 4], dtype="Int64")
    check_func(impl, (S, S < 4, -2), copy_input=True, use_dict_encoded_strings=False)


############################ binary ops #############################


def test_series_operations(memory_leak_check, is_slow_run):
    def f_rpow(S1, S2):
        return S1.rpow(S2)

    def f_rsub(S1, S2):
        return S1.rsub(S2)

    def f_rmul(S1, S2):
        return S1.rmul(S2)

    def f_radd(S1, S2):
        return S1.radd(S2)

    def f_rdiv(S1, S2):
        return S1.rdiv(S2)

    def f_truediv(S1, S2):
        return S1.truediv(S2)

    def f_rtruediv(S1, S2):
        return S1.rtruediv(S2)

    def f_floordiv(S1, S2):
        return S1.floordiv(S2)

    def f_rfloordiv(S1, S2):
        return S1.rfloordiv(S2)

    def f_mod(S1, S2):
        return S1.mod(S2)

    def f_rmod(S1, S2):
        return S1.rmod(S2)

    S1 = pd.Series([2, 3, 4])
    S2 = pd.Series([6, 7, 8])
    check_func(f_rpow, (S1, S2))
    if not is_slow_run:
        return
    check_func(f_rsub, (S1, S2))
    check_func(f_rmul, (S1, S2))
    check_func(f_radd, (S1, S2))
    check_func(f_rdiv, (S1, S2))
    check_func(f_truediv, (S1, S2))
    check_func(f_rtruediv, (S1, S2))
    check_func(f_floordiv, (S1, S2))
    check_func(f_rfloordiv, (S1, S2))
    check_func(f_mod, (S1, S2))
    check_func(f_rmod, (S1, S2))


def test_series_add(memory_leak_check):
    def test_impl(s):
        return s.add(s, None)

    s = pd.Series([1, 8, 4, 10, 3], [3, 7, 9, 2, 1], dtype="Int32")
    check_func(test_impl, (s,))


@pytest.mark.slow
@pytest.mark.parametrize(
    "op",
    [
        "add",
        "sub",
        "mul",
        "truediv",
        "floordiv",
        "mod",
        "pow",
        "lt",
        "gt",
        "le",
        "ge",
        "ne",
        "eq",
    ],
)
@pytest.mark.parametrize("fill", [None, True])
def test_series_explicit_binary_op(numeric_series_val, op, fill, memory_leak_check):
    # dt64 not supported here
    if numeric_series_val.dtype == np.dtype("datetime64[ns]"):
        return
    # XXX ne operator is buggy in Pandas and doesn't set NaNs in output
    # when both inputs are NaNs
    if op == "ne" and numeric_series_val.hasnans:
        return
    # Numba returns float32 for truediv but Numpy returns float64
    if op == "truediv" and numeric_series_val.dtype == np.uint8:
        return
    # Pandas 1.2.0 converts floordiv and truediv to Float64 when input is
    # nullable integer
    # TODO: Support FloatingArray
    if op in ("truediv", "floordiv") and isinstance(
        numeric_series_val.dtype, pd.core.arrays.integer._IntegerDtype
    ):
        check_dtype = False
    else:
        check_dtype = True
    if op == "pow" and numeric_series_val.dtype in (
        np.int8,
        np.int16,
        np.int32,
        np.int64,
    ):
        # negative numbers not supported in integer pow
        numeric_series_val = numeric_series_val.abs()

    func_text = "def test_impl(S, other, fill_val):\n"
    func_text += "  return S.{}(other, fill_value=fill_val)\n".format(op)
    loc_vars = {}
    exec(func_text, {"bodo": bodo}, loc_vars)
    test_impl = loc_vars["test_impl"]

    if fill is not None:
        fill = numeric_series_val.iloc[0]
    check_func(
        test_impl,
        (numeric_series_val, numeric_series_val, fill),
        check_dtype=check_dtype,
    )


@pytest.mark.parametrize("fill", [None, 1.6])
def test_series_explicit_binary_op_nan(fill, memory_leak_check):
    # test nan conditions (both nan, left nan, right nan)
    def test_impl(S, other, fill_val):
        return S.add(other, fill_value=fill_val)

    L1 = pd.Series([1.0, np.nan, 2.3, np.nan])
    L2 = pd.Series([1.0, np.nan, np.nan, 1.1], name="ABC")
    check_func(test_impl, (L1, L2, fill))


def test_series_explicit_binary_op_nullable_int_bool(memory_leak_check):
    """Make comparison operation on nullable int array returns a BooleanArray
    (Pandas 1.0)
    """

    def test_impl(S, other):
        return S.lt(other)

    S1 = pd.Series([1, 8, 4, 10, 3], [3, 7, 9, 2, 1], dtype="Int32")
    S2 = pd.Series([1, -1, 3, 11, 7], [3, 7, 9, 2, 1], dtype="Int32")
    check_func(test_impl, (S1, S2))
    check_func(test_impl, (S1, 5))


@pytest.mark.slow
@pytest.mark.parametrize("op", bodo.hiframes.pd_series_ext.series_binary_ops)
def test_series_binary_op(op, memory_leak_check):
    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(S, other):\n"
    func_text += "  return S {} other\n".format(op_str)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars["test_impl"]

    S = pd.Series([4, 6, 7, 1], [3, 5, 0, 7], name="ABC")
    check_func(test_impl, (S, S))
    check_func(test_impl, (S, 2))
    check_func(test_impl, (2, S))


@pytest.mark.slow
@pytest.mark.parametrize("op", bodo.hiframes.pd_series_ext.series_inplace_binary_ops)
def test_series_inplace_binary_op(op, memory_leak_check):
    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(S, other):\n"
    func_text += "  S {} other\n".format(op_str)
    func_text += "  return S\n"
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars["test_impl"]

    S = pd.Series([4, 6, 7, 1], [3, 5, 0, 7], name="ABC")
    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(
        bodo_func(S.copy(), S.copy()), test_impl(S.copy(), S.copy())
    )
    pd.testing.assert_series_equal(bodo_func(S.copy(), 2), test_impl(S.copy(), 2))
    # XXX: A**=S doesn't work in Pandas for some reason
    if op != operator.ipow:
        np.testing.assert_array_equal(
            bodo_func(S.values.copy(), S.copy()), test_impl(S.values.copy(), S.copy())
        )


@pytest.mark.parametrize("op", bodo.hiframes.pd_series_ext.series_unary_ops)
def test_series_unary_op(op, memory_leak_check):
    # TODO: fix operator.pos
    if op == operator.pos:
        return

    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(S):\n"
    func_text += "  return {} S\n".format(op_str)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars["test_impl"]

    S = pd.Series([4, 6, 7, 1], [3, 5, 0, 7], name="ABC")
    check_func(test_impl, (S,))


def test_series_ufunc(memory_leak_check):
    ufunc = np.negative

    def test_impl(S):
        return ufunc(S)

    S = pd.Series([4, 6, 7, 1], [3, 5, 0, 7], name="ABC")
    check_func(test_impl, (S,))


@pytest.mark.slow
@pytest.mark.parametrize(
    # avoiding isnat since only supported for datetime/timedelta
    "ufunc",
    [f for f in numba.np.ufunc_db.get_ufuncs() if f.nin == 1 and f != np.isnat],
)
def test_series_unary_ufunc(ufunc, memory_leak_check):
    def test_impl(S):
        return ufunc(S)

    S = pd.Series([4, 6, 7, 1], [3, 5, 0, 7], name="ABC")
    check_func(test_impl, (S,))


def test_series_unary_ufunc_np_call(memory_leak_check):
    # a ufunc called explicitly, since the above test sets module name as
    # 'ufunc' instead of 'numpy'
    def test_impl(S):
        return np.negative(S)

    S = pd.Series([4, 6, 7, 1], [3, 5, 0, 7], name="ABC")
    check_func(test_impl, (S,))


@pytest.mark.slow
@pytest.mark.parametrize(
    "ufunc", [f for f in numba.np.ufunc_db.get_ufuncs() if f.nin == 2 and f.nout == 1]
)
def test_series_binary_ufunc(ufunc, memory_leak_check):
    def test_impl(S1, S2):
        return ufunc(S1, S2)

    S = pd.Series([4, 6, 7, 1], [3, 5, 0, 7], name="ABC")
    A = np.array([1, 3, 7, 11])
    check_func(test_impl, (S, S))
    check_func(test_impl, (S, A))
    check_func(test_impl, (A, S))


@pytest.mark.slow
@pytest.mark.parametrize(
    "op", [operator.eq, operator.ne, operator.ge, operator.gt, operator.le, operator.lt]
)
@pytest.mark.parametrize(
    "S",
    [
        # dtype="boolean" makes these nullable Boolean arrays
        pd.Series(
            [True, False, False, True, True, True, False, False], dtype="boolean"
        ),
        pd.Series(
            [True, False, np.nan, True, True, False, True, False], dtype="boolean"
        ),
    ],
)
def test_series_bool_cmp_op(S, op, memory_leak_check):
    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(S, other):\n"
    func_text += "  return S {} other\n".format(op_str)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars["test_impl"]

    check_func(test_impl, (S, S))
    check_func(test_impl, (S, True))
    check_func(test_impl, (True, S))


@pytest.mark.slow
@pytest.mark.parametrize(
    "op", [operator.eq, operator.ne, operator.ge, operator.gt, operator.le, operator.lt]
)
@pytest.mark.parametrize(
    "S",
    [
        pd.Series(
            [True, False, False, True, True, True, False, False], dtype="boolean"
        ),
        pd.Series(
            [True, False, np.nan, True, True, False, True, False], dtype="boolean"
        ),
    ],
)
def test_series_bool_vals_cmp_op(S, op, memory_leak_check):
    """The comparison operation for missing value can be somewhat complicated:
    ---In Python/C++/Fortran/Javascript, the comparison NaN == NaN returns false.
    ---In Pandas the nan or None or missing values are all considered in the same
    way and the result is a missing value in output.
         -
    The following is badly treated in pandas 1.1.0 currently:
    pd.Series([True, False, np.nan, True, True, False, True, False]) .
    Instead of [True, True, <NA>, True, ..., True], pandas returns
    [True, True, False, True, ..., True].
    Adding the ', dtype="boolean"' resolves the issue.
    """
    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(S, other):\n"
    func_text += "  return S.values {} other.values\n".format(op_str)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars["test_impl"]

    check_func(test_impl, (S, S))


def test_series_str_add(memory_leak_check):
    """Test addition for string Series"""

    def test_impl(S, other):
        return S + other

    S = pd.Series(["AA", "D", None, "녁은", "AA🐍"] * 2, [3, 5, 0, 7, -1] * 2)
    S2 = pd.Series(["C🐍", "녁은", "D", None, "AA"] * 2, [3, 5, 0, 7, -1] * 2)
    check_func(test_impl, (S, S2))
    check_func(test_impl, (S, "CC"))
    check_func(test_impl, ("CC", S))


def test_series_str_cmp(memory_leak_check):
    """Test basic comparison for string Series (#1381)"""

    def test_impl(S):
        return S == "A"

    S = pd.Series(pd.array(["AA", "D", None, "녁은", "AA🐍"] * 2), [3, 5, 0, 7, -1] * 2)
    check_func(test_impl, (S,))


@pytest.mark.slow
@pytest.mark.parametrize(
    "S1,S2,fill,raises",
    [
        # float64 input
        (
            pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
            pd.Series([6.0, 21.0, 3.6, 5.0]),
            None,
            False,
        ),
        # index, name
        (
            pd.Series([1.0, 2.0, 3.0, 4.0], [3, 5, 0, 7], name="ABC"),
            pd.Series([6.0, 21.0, 3.6, 5.0], [3, 5, 0, 7]),
            None,
            False,
        ),
        # combine float64/32
        (
            pd.Series([1, 4, 5], dtype="float64"),
            pd.Series([3, 1, 2], dtype="float32"),
            None,
            False,
        ),
        # raise on size mismatch
        (pd.Series([1, 2, 3]), pd.Series([6.0, 21.0, 3.0, 5.0]), None, True),
        (pd.Series([6.0, 21.0, 3.0, 5.0]), pd.Series([1, 2, 3]), None, True),
        # integer case
        (pd.Series([1, 2, 3, 4, 5]), pd.Series([6, 21, 3, 5]), 16, False),
        # different types
        (
            pd.Series([6.1, 21.2, 3.3, 5.4, 6.7]),
            pd.Series([1, 2, 3, 4, 5]),
            None,
            False,
        ),
        # same len integer
        (pd.Series([1, 2, 3, 4, 5]), pd.Series([6, 21, 17, -5, 4]), None, False),
        # same len
        (
            pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
            pd.Series([6.0, 21.0, 3.6, 5.0, 0.0]),
            None,
            False,
        ),
        # fill value
        (
            pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
            pd.Series([6.0, 21.0, 3.6, 5.0]),
            1237.56,
            False,
        ),
        # fill value same len
        (
            pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
            pd.Series([6.0, 21.0, 3.6, 5.0, 0.0]),
            1237.56,
            False,
        ),
    ],
)
def test_series_combine(S1, S2, fill, raises, memory_leak_check):
    def test_impl(S1, S2, fill_val):
        return S1.combine(S2, lambda a, b: 2 * a + b, fill_val)

    bodo_func = bodo.jit(test_impl)
    if raises:
        with pytest.raises(AssertionError):
            bodo_func(S1, S2, fill)
    else:
        # TODO: fix 1D_Var chunk size mismatch on inputs with different sizes
        pd.testing.assert_series_equal(bodo_func(S1, S2, fill), test_impl(S1, S2, fill))


@pytest.mark.slow
def test_series_combine_kws(memory_leak_check):
    def test_impl(S1, S2, fill_val):
        return S1.combine(other=S2, func=lambda a, b: 2 * a + b, fill_value=fill_val)

    S1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    S2 = pd.Series([6.0, 21.0, 3.6, 5.0, 0.0])
    fill = 1237.56
    check_func(test_impl, (S1, S2, fill), check_dtype=False)


@pytest.mark.slow
def test_series_combine_kws_int(memory_leak_check):
    def test_impl(S1, S2, fill_val):
        return S1.combine(other=S2, func=lambda a, b: 2 * a + b, fill_value=fill_val)

    S1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    S2 = pd.Series([6.0, 21.0, 3.6, 5.0, 0.0])
    fill = 2
    check_func(test_impl, (S1, S2, fill), check_dtype=False)


def test_series_combine_no_fill(memory_leak_check):
    def test_impl(S1, S2):
        return S1.combine(other=S2, func=lambda a, b: 2 * a + b)

    S1 = pd.Series([1, 2, 3, 4, 5])
    S2 = pd.Series([6, 21, 3, 5, 0])
    check_func(test_impl, (S1, S2))


def g(a, b):
    return 2 * a + b


@pytest.mark.slow
def test_series_combine_global_func(memory_leak_check):
    def test_impl(S1, S2):
        return S1.combine(other=S2, func=g)

    S1 = pd.Series([1, 2, 3, 4, 5])
    S2 = pd.Series([6, 21, 3, 5, 0])
    check_func(test_impl, (S1, S2))


@pytest.mark.slow
@pytest.mark.parametrize(
    "S",
    [
        pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
        pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], [3, 1, 0, 2, 4], name="ABC"),
    ],
)
def test_series_apply(S, memory_leak_check):
    def test_impl(S):
        return S.apply(lambda a: 2 * a)

    check_func(test_impl, (S,))


def test_series_apply_kw(memory_leak_check):
    def test_impl(S):
        return S.apply(func=lambda a: 2 * a)

    S = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    check_func(test_impl, (S,))


def test_series_pipe():
    """
    Test Series.pipe()
    """

    def impl1(S):
        return S.pipe(lambda S: S.sum())

    # test *args, **kwargs
    def impl2(S, a, b):
        return S.pipe(lambda S, a, b: S.sum() + a + b, a, b=b)

    S = pd.Series([1, 2, 3, 4, 5, 6])
    check_func(impl1, (S,))
    check_func(impl2, (S, 1, 2))


def test_series_heter_constructor(memory_leak_check):
    """
    test creating Series with heterogeneous values
    """

    def impl1():
        return pd.Series([1, "A"], ["B", "C"])

    check_func(impl1, (), dist_test=False)


def test_series_apply_df_output(memory_leak_check):
    """test Series.apply() with dataframe output"""

    def impl1(S):
        return S.apply(lambda a: pd.Series([a, "AA"]))

    def impl2(S):
        def g(a):
            # TODO: support assert in UDFs properly
            # assert a > 0.0
            if a > 3:
                return pd.Series([a, 2 * a], ["A", "B"])
            return pd.Series([a, 3 * a], ["A", "B"])

        return S.apply(g)

    def impl3(S):
        return S.apply(lambda a: pd.Series((str(a), str(a + 1.2))))

    def impl4(S):
        return S.apply(lambda a: pd.Series((a, a + 1.2)))

    S = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    check_func(impl1, (S,))
    check_func(impl2, (S,), check_dtype=False)
    _check_IR_no_const_arr(impl3, (S,))
    _check_IR_no_const_arr(impl4, (S,))


def _check_IR_no_const_arr(test_impl, args):
    """makes sure there is no const array call left in the IR after optimization"""
    bodo_func = numba.njit(pipeline_class=SeriesOptTestPipeline, parallel=True)(
        test_impl
    )
    bodo_func(*args)  # calling the function to get function IR
    fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    # make sure there is no const call in IR
    for block in fir.blocks.values():
        for stmt in block.body:
            assert not (
                is_call_assign(stmt)
                and (
                    guard(find_callname, fir, stmt.value)
                    in (
                        ("str_arr_from_sequence", "bodo.libs.str_arr_ext"),
                        ("asarray", "numpy"),
                    )
                )
            )


@pytest.mark.slow
def test_series_apply_extra_arg(memory_leak_check):
    def test_impl(S, D):
        return S.apply(lambda a, d: a not in d, args=(D,))

    d = ("A", "B")
    S = pd.Series(["A", "C", "FF", "AA", "CC", "B", "DD", "ABC", "A"])
    check_func(test_impl, (S, d))


@pytest.mark.slow
def test_series_apply_kwargs(memory_leak_check):
    def test_impl(S, D):
        return S.apply(lambda a, d: a not in d, d=D)

    d = ("A", "B")
    S = pd.Series(["A", "C", "FF", "AA", "CC", "B", "DD", "ABC", "A"])
    check_func(test_impl, (S, d))


def test_series_apply_args_and_kwargs(memory_leak_check):
    def test_impl(S, b, d):
        return S.apply(lambda x, c=1, a=2: x == a + c, a=b, args=(d,))

    n = 121
    S = pd.Series(np.arange(n))
    check_func(test_impl, (S, 3, 2))


@pytest.mark.slow
def test_series_apply_supported_types(series_val, memory_leak_check):
    """Test Series.apply with all Bodo supported Types"""

    def test_impl(S, val):
        return S.apply(lambda a, val: a if a != val else None, val=val)

    if isinstance(series_val.dtype, pd.CategoricalDtype):
        # TODO: [BE-130] support apply for pd.Categorical
        return
    # Bodo gives different value for NaN from Pandas.(e.g. NaN vs. False for boolean)
    S = series_val.dropna()
    # Increase data size to pass testing with 3 ranks.
    S = S.repeat(3)
    # TODO: Fails with array of lists. Test when equality operation support list comparison.
    if isinstance(S.values[0], list):
        return
    val = S.iloc[0]
    # Disable check_dtype since Pandas return uint8 as int64
    check_func(test_impl, (S, val), check_dtype=False)


@pytest.mark.slow
def test_series_apply_args(memory_leak_check):
    """Test Series.apply with unsupported and wrong arguments"""

    def test_convert_dtype_false(S):
        return S.apply(lambda a: a, convert_dtype=False)

    def test_convert_dtype_true(S):
        return S.apply(lambda a: a, convert_dtype=True)

    def test_wrong_func(S):
        return S.apply("XX")

    def test_wrong_arg(S):
        return S.apply(lambda x: x, axis=1)

    S = pd.Series([2, 1, 3])
    with pytest.raises(
        BodoError, match="Series.apply.* only supports default value True"
    ):
        bodo.jit(test_convert_dtype_false)(S)

    bodo.jit(test_convert_dtype_true)(S)

    with pytest.raises(
        BodoError, match="Series.apply.*: user-defined function not supported"
    ):
        bodo.jit(test_wrong_func)(S)

    with pytest.raises(
        BodoError,
        match=r"Series.apply.*: user-defined function not supported: got an unexpected keyword argument",
    ):
        bodo.jit(test_wrong_arg)(S)


# TODO: Add memory leak check once constant lowering memory leak is fixed
@pytest.mark.slow
def test_series_map_supported_types(series_val):
    """Test Series.map with all Bodo supported Types"""

    def test_impl(S):
        return S.map(lambda a: a)  # if not pd.isna(a) else None)

    # nan vs. 0
    S = series_val.dropna()
    # Increase data size to pass testing with 3 ranks
    S = S.repeat(3)
    # Disable check_dtype since Pandas return int64 vs. Int64
    check_func(test_impl, (S,), check_dtype=False)


@pytest.mark.slow
def test_series_map_args(memory_leak_check):
    """Test Series.map with unsupported and wrong arguments"""

    def test_na_action_ignore(S):
        return S.map(lambda a: a, na_action="ignore")

    def test_na_action_none(S):
        return S.map(lambda a: a, na_action=None)

    def test_wrong_func(S):
        return S.map("XX")

    S = pd.Series([2, 1, 3])
    with pytest.raises(BodoError, match="Series.map.* only supports default value"):
        bodo.jit(test_na_action_ignore)(S)

    bodo.jit(test_na_action_none)(S)

    with pytest.raises(
        BodoError, match="Series.map.*: user-defined function not supported"
    ):
        bodo.jit(test_wrong_func)(S)


# TODO: add memory_leak_check after fix its failure with Categorical
@pytest.mark.slow
def test_series_groupby_supported_types(series_val):
    """Test Series.groupby with all Bodo supported Types"""

    def test_impl(S):
        return S.groupby(level=0).max()

    if isinstance(series_val.dtype, pd.CategoricalDtype):
        series_val = series_val.cat.as_ordered()

    # TODO: [BE-346] Gives nan with array of lists.
    # series_val16: bodo_output: [nan, nan, nan, nan, nan]
    if isinstance(series_val.values[0], list):
        return

    check_func(
        test_impl,
        (series_val,),
        check_names=False,
        sort_output=True,
        reset_index=True,
        check_categorical=False,  # Bodo keeps categorical type, Pandas changes series type
        check_dtype=False,
    )


@pytest.mark.slow
def test_series_groupby_by_arg_supported_types(series_val, memory_leak_check):
    """Test Series.groupby by argument with all Bodo supported Types"""

    def test_impl_by(S, byS):
        return S.groupby(byS).mean()

    # TODO: [BE-347]
    if series_val.dtype == np.bool_ or is_bool_object_series(series_val):
        return

    if isinstance(series_val.values[0], list):
        return

    if isinstance(series_val.dtype, pd.CategoricalDtype):
        return

    # not supported for Decimal yet, TODO: support and test
    if isinstance(series_val.values[0], Decimal):
        return

    # matches length for both series_val and by argument
    if len(series_val) > 5:
        series_val = series_val[:5]

    S = pd.Series([390.0, 350.0, 30.0, 20.0, 5.5])
    check_func(
        test_impl_by,
        (S, series_val.values),
        check_names=False,
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )

    # TODO: [BE-347] support boolean arrays for `by` argument
    # def test_impl_by_types(S):
    #    return S.groupby(S>100).mean()
    # S = pd.Series([390., 350., 30., 20.])
    # check_func(test_impl_by_types, (S, ))


@pytest.mark.parametrize(
    "S",
    [
        pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
        pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], [3, 1, 0, 2, 4], name="ABC"),
    ],
)
def test_series_map(S, memory_leak_check):
    def test_impl(S):
        return S.map(lambda a: 2 * a)

    check_func(test_impl, (S,))


@pytest.mark.parametrize(
    "arg1",
    [
        pd.Series([1, 2, 3] * 4),
        pd.Series([-12, None, 14, None] * 3, dtype="Int64"),
    ],
)
@pytest.mark.parametrize(
    "arg2",
    [
        pd.Series([0, -1, 2, None] * 3, dtype="Int64"),
        pd.Series([1, 2, 3] * 4),
        np.array([3, -1, 5] * 4),
        1,
        0,
    ],
)
def test_series_and_or_int(arg1, arg2, memory_leak_check):
    """Tests support for pd.Series &/|, with integer data"""

    # https://github.com/pandas-dev/pandas/issues/34463
    # Pandas doesn't currently support and/or between
    # two Nullable Integer Arrays, or nullable integer Arrays and
    # scalars but it most likely will in the future.
    assert pandas_version in (
        (1, 3),
        (1, 4),
    ), "Check support for pd.IntegerArray's and/or"

    # Lambda functions used for setting expected output:
    or_scalar = lambda v, x: v if pd.isna(v) else v | x
    and_scalar = lambda v, x: v if pd.isna(v) else v & x

    # Workaround for the aformentioned issue with and/or with Nullable Integer Arrays
    arg1_null_series = isinstance(arg1, pd.Series) and arg1.dtype.name == "Int64"
    arg2_null_series = isinstance(arg2, pd.Series) and arg2.dtype.name == "Int64"
    if arg1_null_series and arg2_null_series:
        pyOutputOr = arg1.fillna(1).astype(np.int64) | arg2.fillna(1).astype(np.int64)
        pyOutputOr[arg1.isna() | arg2.isna()] = None
        pyOutputOr = pyOutputOr.astype(arg1.dtype)
        pyOutputAnd = arg1.fillna(1).astype(np.int64) & arg2.fillna(1).astype(np.int64)
        pyOutputAnd[arg1.isna() | arg2.isna()] = None
        pyOutputAnd = pyOutputAnd.astype(arg1.dtype)
    elif arg1_null_series and isinstance(arg2, int):
        pyOutputOr = arg1.apply(or_scalar, x=arg2)
        pyOutputAnd = arg1.apply(and_scalar, x=arg2)
    elif arg2_null_series and isinstance(arg2, int):
        pyOutputOr = arg2.apply(or_scalar, x=arg1)
        pyOutputAnd = arg2.apply(and_scalar, x=arg1)
    else:
        pyOutputOr = None
        pyOutputAnd = None

    def impl_and(lhs, rhs):
        return lhs & rhs

    def impl_or(lhs, rhs):
        return lhs | rhs

    check_func(impl_and, (arg1, arg2), py_output=pyOutputAnd)
    check_func(impl_and, (arg2, arg1), py_output=pyOutputAnd)
    check_func(impl_or, (arg1, arg2), py_output=pyOutputOr)
    check_func(impl_or, (arg2, arg1), py_output=pyOutputOr)


def test_series_init_dict_int():
    """tests initializing a series with a constant str->int dict"""

    def impl():
        S = pd.Series({"r1": 1})
        return S

    def impl2():
        S = pd.Series({"r1": 1, "r2": 2, "r3": 3})
        return S

    # output should not be distributed, if initialized with a constant dict
    check_func(impl, (), is_out_distributed=False)
    check_func(impl2, (), is_out_distributed=False)


def test_series_init_dict_kwd():
    """tests that we handle kwd argument correctly in the typing pass transform"""

    def impl1():
        S = pd.Series(
            {"r1": 1, "r2": 2, "r3": 3}, None, np.int32, "my_series", False, False
        )
        return S

    def impl2():
        S = pd.Series({"r1": 1, "r2": 2, "r3": 3}, None, "Int32", name="my_series")
        return S

    def impl3():
        S = pd.Series({"r1": 1, "r2": 2, "r3": 3}, None, name="my_series")
        return S

    check_func(impl1, (), is_out_distributed=False)
    check_func(impl2, (), is_out_distributed=False)
    check_func(impl3, (), is_out_distributed=False)


def test_series_init_dict_grpby_lambda():
    """
    Tests intializing a series with a dict containing constant keys and values inside of
    a groupby apply
    """

    df1 = pd.DataFrame({"A": [1, 2, 3] * 5, "B": [1, 2, 3, 4, 5] * 3})

    def impl1(df):
        return df.groupby(
            "A", as_index=False, sort=False, observed=True, dropna=False
        ).apply(lambda x: pd.Series({"r2": 10}))

    df2 = pd.DataFrame(
        {
            "id2": ["id016", "id045", "id023", "id057", "id040"],
            "id4": [15, 40, 68, 32, 9],
            "v1": [5, 5, 2, 1, 2],
            "v2": [11, 4, 14, 15, 9],
        }
    )

    def impl2(df):
        return df.groupby(
            ["id2", "id4"], as_index=False, sort=False, observed=True, dropna=False
        ).apply(lambda x: pd.Series({"r2": 10}))

    # TODO [BE-2246]: Match output dtype by checking null info.
    check_func(impl1, (df1,), sort_output=True, reset_index=True, check_dtype=False)
    check_func(impl2, (df2,), sort_output=True, reset_index=True)


def test_series_getitem_string_index(memory_leak_check):
    """Tests that we can do a getitem on a Series with a string index"""
    S = pd.Series([1, 2, 3, 4, 5, 6], index=["A", "B", "C", "D", "E", "F"])

    def test_impl(S):
        return S["A"]

    # needed so sonar doesn't complain
    def test_impl2(S, idx):
        return S[idx]

    # TODO: support in distributed case
    check_func(test_impl, (S,), only_seq=True)
    check_func(test_impl2, (S, "A"), only_seq=True)

    # TODO: re add if/when idx.get_loc supports non-unique index
    # S2 = pd.Series([1, 2, 3] * 3, index=["A", "B", "C"] * 3)
    # check_func(test_impl, (S2,), only_seq=True)


def test_series_getitem_str_grpby_apply():
    """Tests that getitem works inside of groupby apply"""

    # example taken from H20 benchmark Q09
    data = {
        "id1": ["id016", "id039", "id047", "id043", "id054"],
        "id2": ["id016", "id045", "id023", "id057", "id040"],
        "v1": [5, 5, 2, 1, 2],
        "v2": [11, 4, 14, 15, 9],
    }
    df = pd.DataFrame(data)

    def test_impl(df):
        return df.groupby(
            ["id1", "id2"], as_index=False, sort=False, observed=True, dropna=False
        ).apply(lambda x: x.corr()["v1"]["v2"] ** 2)

    # in Bodo, when doing groupby apply, we set the output column to empty string
    # if we have a string index. In Pandas, it's normally None
    py_out = test_impl(df)
    py_out.columns = ["id1", "id2", ""]

    check_func(test_impl, (df,), py_output=py_out, sort_output=True, reset_index=True)


def test_series_getitem_str_and_series_init_dict_grpby_apply():
    """
    Tests both str getitem on series, and intializing series with constant key and non constant value dicts
    work within a groupby apply.
    This example is taken from h20 benchmark Q7
    """
    d = {
        "id1": ["id016", "id039", "id047", "id043", "id054"],
        "id2": ["id016", "id045", "id023", "id057", "id040"],
        "id3": [
            "id0000042202",
            "id0000029558",
            "id0000071286",
            "id0000015141",
            "id0000011083",
        ],
        "id4": [15, 40, 68, 32, 9],
        "id5": [24, 49, 20, 43, 25],
        "id6": [5971, 39457, 74463, 63743, 16920],
        "v1": [5, 5, 2, 1, 2],
        "v2": [11, 4, 14, 15, 9],
        "v3": [37.211254, 48.951141, 60.469241, 7.692145, 22.863525],
    }
    df = pd.DataFrame(data=d)

    def test_impl(df):
        ans = (
            df[["id2", "id4", "v1", "v2"]]
            .groupby(
                ["id2", "id4"], as_index=False, sort=False, observed=True, dropna=False
            )
            .apply(
                lambda x: pd.Series(
                    {"r2": x.corr()["v1"]["v2"] ** 2, "r2_2": x.corr()["v1"]["v2"]}
                )
            )
        )
        return ans

    check_func(test_impl, (df,), sort_output=True, reset_index=True)


def test_series_init_empty_dict():
    """tests intialzing series with an empty dict works"""

    def test_impl():
        # this is needed so the dict has a type
        d = {"a": 1}
        d.pop("a")
        pd.Series(d)

    check_func(
        test_impl,
        (),
    )
