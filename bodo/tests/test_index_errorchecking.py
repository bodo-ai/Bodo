# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Tests for pd.Index error checking
"""


import re

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.hiframes.pd_index_ext import (
    IntervalIndexType,
    cat_idx_unsupported_atrs,
    cat_idx_unsupported_methods,
    dt_index_unsupported_atrs,
    dt_index_unsupported_methods,
    index_unsupported_atrs,
    index_unsupported_methods,
    interval_idx_unsupported_atrs,
    interval_idx_unsupported_methods,
    multi_index_unsupported_atrs,
    multi_index_unsupported_methods,
    period_index_unsupported_atrs,
    period_index_unsupported_methods,
    td_index_unsupported_atrs,
    td_index_unsupported_methods,
)
from bodo.hiframes.pd_multi_index_ext import MultiIndexType
from bodo.utils.typing import BodoError


def test_object_dtype(memory_leak_check):
    """
    Test that providing object as the dtype raises a reasonable
    error.
    """

    def impl():
        a = pd.Index(["a", "b", "c"], dtype="object")
        return a[0]

    with pytest.raises(
        BodoError,
        match="pd.Index\\(\\) object 'dtype' is not specific enough for typing. Please provide a more exact type \\(e.g. str\\).",
    ):
        bodo.jit(impl)()


@pytest.fixture(
    params=[
        pytest.param(
            (pd.Index(range(15)), "pandas.RangeIndex.{}"), id="RangeIndexType"
        ),
        pytest.param(
            (pd.Index([1, 2, 3, 4] * 3), "pandas.Index.{} with numeric data"),
            id="NumericIndexType",
        ),
        pytest.param(
            (
                pd.Index(["hello", "world", "how are", "you"] * 3),
                "pandas.Index.{} with string data",
            ),
            id="StringIndexType",
        ),
        pytest.param(
            (
                pd.Index([b"hello", b"world", b"how are", b"you"] * 3),
                "pandas.Index.{} with binary data",
            ),
            id="BinaryIndexType",
        ),
        pytest.param(
            (
                pd.Index(
                    [
                        pd.Timedelta(12, unit="d"),
                        pd.Timedelta(123, unit="ns"),
                        pd.Timedelta(100, unit="h"),
                    ]
                    * 3
                ),
                "pandas.TimedeltaIndex.{}",
            ),
            id="TimedeltaIndexType",
        ),
        pytest.param(
            (
                pd.Index(["hello", "world", "how are", "you"] * 3).astype("category"),
                "pandas.CategoricalIndex.{}",
            ),
            id="CategoricalIndexType",
        ),
        pytest.param(
            (
                pd.PeriodIndex(year=[2015, 2016, 2018], month=[1, 2, 3], freq="M"),
                "pandas.PeriodIndex.{}",
            ),
            id="PeriodIndexType",
        ),
        pytest.param(
            (
                pd.Index(
                    [
                        pd.Timestamp("2020-02-23"),
                        pd.Timestamp("2017-11-02"),
                        pd.Timestamp("2000-8-18"),
                    ]
                    * 3
                ),
                "pandas.DatetimeIndex.{}",
            ),
            id="DatetimeIndexType",
        ),
        pytest.param(
            (pd.interval_range(start=0, end=10), "pandas.IntervalIndex.{}"),
            id="IntervalIndexType",
        ),
        pytest.param(
            (
                pd.MultiIndex.from_arrays(
                    [[1, 1, 2, 2], ["red", "blue", "red", "blue"]],
                    names=("number", "color"),
                ),
                "pandas.MultiIndex.{}",
            ),
            id="MultiIndexType",
        ),
    ]
)
def all_index_types(request):
    """fixture that contains all supported index types (excluding hetrogenous)"""
    return request.param


@pytest.fixture(params=index_unsupported_methods)
def index_unsuported_methods_fixture(request):
    """fixture around the methods that are unsupported for all index types"""
    return request.param


@pytest.fixture(params=index_unsupported_atrs)
def index_unsupported_atrs_fixture(request):
    """fixture around the attributes that are unsupported for all index types"""
    return request.param


@pytest.fixture(params=cat_idx_unsupported_atrs)
def cat_idx_unsupported_atrs_fixture(request):
    """fixture around the attributes that are unsupported for categorical index types"""
    return request.param


@pytest.fixture(params=interval_idx_unsupported_atrs)
def interval_idx_unsupported_atrs_fixture(request):
    """fixture around the attributes that are unsupported for interval index types"""
    return request.param


@pytest.fixture(params=multi_index_unsupported_atrs)
def multi_index_unsupported_atrs_fixture(request):
    """fixture around the attributes that are unsupported for multi_index types"""
    return request.param


@pytest.fixture(params=dt_index_unsupported_atrs)
def dt_index_unsupported_atrs_fixture(request):
    """fixture around the attributes that are unsupported for datetime index types"""
    return request.param


@pytest.fixture(params=td_index_unsupported_atrs)
def td_index_unsupported_atrs_fixture(request):
    """fixture around the attributes that are unsupported for timedelta index types"""
    return request.param


@pytest.fixture(params=period_index_unsupported_atrs)
def period_index_unsupported_atrs_fixture(request):
    """fixture around the attributes that are unsupported for period index types"""
    return request.param


@pytest.fixture(params=cat_idx_unsupported_methods)
def cat_idx_unsupported_methods_fixture(request):
    """fixture around the methods that are unsupported for categorical index types"""
    return request.param


@pytest.fixture(params=interval_idx_unsupported_methods)
def interval_idx_unsupported_methods_fixture(request):
    """fixture around the methods that are unsupported for interval index types"""
    return request.param


@pytest.fixture(params=multi_index_unsupported_methods)
def multi_index_unsupported_methods_fixture(request):
    """fixture around the methods that are unsupported for multi_index types"""
    return request.param


@pytest.fixture(params=dt_index_unsupported_methods)
def dt_index_unsupported_methods_fixture(request):
    """fixture around the methods that are unsupported for datetime index types"""
    return request.param


@pytest.fixture(params=td_index_unsupported_methods)
def td_index_unsupported_methods_fixture(request):
    """fixture around the methods that are unsupported for timedelta index types"""
    return request.param


@pytest.fixture(params=period_index_unsupported_methods)
def period_index_unsupported_methods_fixture(request):
    """fixture around the methods that are unsupported for timedelta index types"""
    return request.param


@pytest.mark.slow
def test_all_idx_unsupported_methods(all_index_types, index_unsuported_methods_fixture):
    """tests that the unsupported index methods raise the propper errors"""
    idx_val = all_index_types[0]
    idx_formatstr = all_index_types[1]
    check_unsupported_method(idx_formatstr, idx_val, index_unsuported_methods_fixture)


@pytest.mark.slow
def test_all_idx_unsupported_attrs(all_index_types, index_unsupported_atrs_fixture):
    """tests that the unsupported index attributes raise the propper errors"""

    idx_val = all_index_types[0]
    idx_formatstr = all_index_types[1]
    check_unsupported_atr(idx_formatstr, idx_val, index_unsupported_atrs_fixture)


@pytest.mark.slow
def test_cat_idx_unsupported_methods(cat_idx_unsupported_methods_fixture):
    """tests that the unsupported categorical index methods raise the propper errors"""
    check_unsupported_method(
        "pandas.CategoricalIndex.{}",
        pd.Index(["hello", "world", "how are", "you"] * 3).astype("category"),
        cat_idx_unsupported_methods_fixture,
    )


@pytest.mark.slow
def test_interval_idx_unsupported_methods(interval_idx_unsupported_methods_fixture):
    """tests that the unsupported interval index methods raise the propper errors"""
    check_unsupported_method(
        "pandas.IntervalIndex.{}",
        pd.interval_range(start=0, end=10),
        interval_idx_unsupported_methods_fixture,
    )


@pytest.mark.slow
def test_multi_idx_unsupported_methods(multi_index_unsupported_methods_fixture):
    """tests that the unsupported multi_index methods raise the propper errors"""
    check_unsupported_method(
        "pandas.MultiIndex.{}",
        pd.MultiIndex.from_arrays(
            [[1, 1, 2, 2], ["red", "blue", "red", "blue"]], names=("number", "color")
        ),
        multi_index_unsupported_methods_fixture,
    )


@pytest.mark.slow
def test_dt_idx_unsupported_methods(dt_index_unsupported_methods_fixture):
    """tests that the unsupported datetime index methods raise the propper errors"""
    check_unsupported_method(
        "pandas.DatetimeIndex.{}",
        pd.Index(
            [
                pd.Timestamp("2020-02-23"),
                pd.Timestamp("2017-11-02"),
                pd.Timestamp("2000-8-18"),
            ]
            * 3
        ),
        dt_index_unsupported_methods_fixture,
    )


@pytest.mark.slow
def test_td_idx_unsupported_methods(td_index_unsupported_methods_fixture):
    """tests that the unsupported timedelta index methods raise the propper errors"""
    check_unsupported_method(
        "pandas.TimedeltaIndex.{}",
        pd.Index(
            [
                pd.Timedelta(12, unit="d"),
                pd.Timedelta(123, unit="ns"),
                pd.Timedelta(100, unit="h"),
            ]
            * 3
        ),
        td_index_unsupported_methods_fixture,
    )


@pytest.mark.slow
def test_period_idx_unsupported_methods(period_index_unsupported_methods_fixture):
    """tests that the unsupported period index methods raise the propper errors"""
    check_unsupported_method(
        "pandas.PeriodIndex.{}",
        pd.PeriodIndex(year=[2015, 2016, 2018], month=[1, 2, 3], freq="M"),
        period_index_unsupported_methods_fixture,
    )


@pytest.mark.slow
def test_cat_idx_unsupported_atrs(cat_idx_unsupported_atrs_fixture):
    """tests that the categorical index attributes raise the propper errors"""
    check_unsupported_atr(
        "pandas.CategoricalIndex.{}",
        pd.Index(["hello", "world", "how are", "you"] * 3).astype("category"),
        cat_idx_unsupported_atrs_fixture,
    )


@pytest.mark.slow
def test_interval_idx_unsupported_atrs(interval_idx_unsupported_atrs_fixture):
    """tests that the interval index attributes raise the propper errors"""
    check_unsupported_atr(
        "pandas.IntervalIndex.{}",
        pd.interval_range(start=0, end=10),
        interval_idx_unsupported_atrs_fixture,
    )


@pytest.mark.slow
def test_multi_idx_unsupported_atrs(multi_index_unsupported_atrs_fixture):
    """tests that the categorical index attributes raise the propper errors"""
    check_unsupported_atr(
        "pandas.MultiIndex.{}",
        pd.MultiIndex.from_arrays(
            [[1, 1, 2, 2], ["red", "blue", "red", "blue"]], names=("number", "color")
        ),
        multi_index_unsupported_atrs_fixture,
    )


@pytest.mark.slow
def test_dt_idx_unsupported_atrs(dt_index_unsupported_atrs_fixture):
    """tests that the datetime index attributes raise the propper errors"""
    check_unsupported_atr(
        "pandas.DatetimeIndex.{}",
        pd.Index(
            [
                pd.Timestamp("2020-02-23"),
                pd.Timestamp("2017-11-02"),
                pd.Timestamp("2000-8-18"),
            ]
            * 3
        ),
        dt_index_unsupported_atrs_fixture,
    )


@pytest.mark.slow
def test_td_idx_unsupported_atrs(td_index_unsupported_atrs_fixture):
    """tests that the timedelta index attributes raise the propper errors"""
    check_unsupported_atr(
        "pandas.TimedeltaIndex.{}",
        pd.Index(
            [
                pd.Timedelta(12, unit="d"),
                pd.Timedelta(123, unit="ns"),
                pd.Timedelta(100, unit="h"),
            ]
            * 3
        ),
        td_index_unsupported_atrs_fixture,
    )


@pytest.mark.slow
def test_period_idx_unsupported_atrs(period_index_unsupported_atrs_fixture):
    """tests that the period index attributes raise the propper errors"""
    check_unsupported_atr(
        "pandas.PeriodIndex.{}",
        pd.PeriodIndex(year=[2015, 2016, 2018], month=[1, 2, 3], freq="M"),
        period_index_unsupported_atrs_fixture,
    )


@pytest.mark.slow
def test_multiindex_from_arrays():
    def impl():
        return pd.MultiIndex.from_arrays([[1, 2, 3], ["red", "blue", "red"]])

    err_msg = re.escape("pandas.MultiIndex.from_arrays() is not yet supported")

    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_multiindex_from_tuples():
    def impl():
        return pd.MultiIndex.from_tuples(
            [(1, "red"), (1, "blue"), (2, "red"), (2, "blue")]
        )

    err_msg = re.escape("pandas.MultiIndex.from_tuples() is not yet supported")

    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_multiindex_from_frame():
    def impl():
        return pd.MultiIndex.from_frame(
            pd.DataFrame(
                [["HI", "Temp"], ["HI", "Precip"], ["NJ", "Temp"], ["NJ", "Precip"]],
                columns=["a", "b"],
            )
        )

    err_msg = re.escape("pandas.MultiIndex.from_frame() is not yet supported")

    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_interval_index_from_arrays():
    def impl():
        return pd.IntervalIndex.from_arrays([0, 1, 2], [1, 2, 3])

    err_msg = re.escape("pandas.IntervalIndex.from_arrays() is not yet supported")

    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_interval_index_from_tuples():
    def impl():
        return pd.IntervalIndex.from_tuples([(0, 1), (1, 2)])

    err_msg = re.escape("pandas.IntervalIndex.from_tuples() is not yet supported")

    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_index_to_numpy_args():
    def impl1(index):
        return index.to_numpy(dtype="unicode")

    def impl2(index):
        return index.to_numpy(na_value=42)

    def impl3(index):
        return index.to_numpy(copy=7.5)

    err_msg1 = re.escape(
        "Index.to_numpy(): dtype parameter only supports default value None"
    )
    err_msg2 = re.escape(
        "Index.to_numpy(): na_value parameter only supports default value None"
    )
    err_msg3 = re.escape("Index.to_numpy(): copy argument must be a boolean")

    I = pd.Index([1, 2, 3, 4, 5])

    with pytest.raises(BodoError, match=err_msg1):
        bodo.jit(impl1)(I)

    with pytest.raises(BodoError, match=err_msg2):
        bodo.jit(impl2)(I)

    with pytest.raises(BodoError, match=err_msg3):
        bodo.jit(impl3)(I)


@pytest.mark.slow
def test_index_to_series_malformed():
    def impl1(I, J):
        return I.to_series(index=J)

    def impl2(I, n):
        return I.to_series(name=n)

    err_msg1 = re.escape(
        "Index.to_series(): unsupported type for argument index: UnicodeType"
    )
    err_msg2 = re.escape("Index.to_series(): unsupported type for argument index: Set")
    err_msg3 = re.escape(
        "Index.to_series(): unsupported type for argument index: DictType"
    )
    err_msg4 = re.escape(
        "Index.to_series(): unsupported type for argument index: MultiIndexType"
    )
    err_msg5 = re.escape(
        "Index.to_series(): only constant string/int are supported for argument name"
    )

    I = pd.Index([1, 2, 3, 4, 5])
    J = "ABCDE"
    K = {6, 7, 8, 9, 10}
    L = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
    M = pd.MultiIndex.from_product([[1, 2], ["A", "B", "C"]])

    with pytest.raises(BodoError, match=err_msg1):
        bodo.jit(impl1)(I, J)

    with pytest.raises(BodoError, match=err_msg2):
        bodo.jit(impl1)(I, K)

    with pytest.raises(BodoError, match=err_msg3):
        bodo.jit(impl1)(I, L)

    with pytest.raises(BodoError, match=err_msg4):
        bodo.jit(impl1)(I, M)

    with pytest.raises(BodoError, match=err_msg5):
        bodo.jit(impl2)(I, ["A", "B", "C"])


@pytest.mark.slow
def test_index_to_frame_malformed():
    def impl1(I):
        return I.to_frame(name=["Chocolate", "Vanilla"])

    def impl2(I):
        return I.to_frame(index=len(I) > 4)

    def impl3(I):
        return I.to_frame(index=I)

    def impl4(I):
        return I.to_frame(name=("x", "y", "z"))

    def impl5(I, n):
        return I.to_frame(name=n)

    err_msg1 = re.escape(
        "Index.to_frame(): only constant string/int are supported for argument name"
    )
    err_msg2 = re.escape(
        "Index.to_frame(): index argument must be a compile time constant"
    )
    err_msg3 = re.escape("Index.to_frame(): index argument must be a constant boolean")
    err_msg4 = re.escape("MultiIndex.to_frame(): expected 2 names, not 3")
    err_msg5 = re.escape(
        "MultiIndex.to_frame(): only constant string/int list/tuple are supported for argument name"
    )

    I = pd.Index([1, 2, 3, 4, 5])
    M = pd.MultiIndex.from_product([["A", "B", "C"], [1, 2, 3]])

    with pytest.raises(BodoError, match=err_msg1):
        bodo.jit(impl1)(I)

    with pytest.raises(BodoError, match=err_msg2):
        bodo.jit(impl2)(I)

    with pytest.raises(BodoError, match=err_msg3):
        bodo.jit(impl3)(I)

    with pytest.raises(BodoError, match=err_msg4):
        bodo.jit(impl4)(M)

    with pytest.raises(BodoError, match=err_msg5):
        bodo.jit(impl5)(M, ("i", "ii", "iii"))


@pytest.mark.slow
def test_interval_index_from_breaks():
    def impl():
        return pd.IntervalIndex.from_breaks([0, 1, 2, 3])

    err_msg = re.escape("pandas.IntervalIndex.from_breaks() is not yet supported")

    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)()


def test_argmin_argmax_min_max_edgecases():
    def impl1(I):
        return I.argmin()

    def impl2(I):
        return I.argmax()

    def impl3(I):
        return I.min()

    def impl4(I):
        return I.max()

    def impl5(I):
        return I.argmin(axis=1)

    def impl6(I):
        return I.argmax(axis=1)

    def impl7(I):
        return I.min(axis=1)

    def impl8(I):
        return I.max(axis=1)

    def impl9(I):
        return I.argmin(skipna=False)

    def impl10(I):
        return I.argmax(skipna=False)

    def impl11(I):
        return I.min(skipna=False)

    def impl12(I):
        return I.max(skipna=False)

    err_msg1 = re.escape("Index.argmin(): only ordered categoricals are possible")
    err_msg2 = re.escape("Index.argmax(): only ordered categoricals are possible")
    err_msg3 = re.escape("Index.min(): only ordered categoricals are possible")
    err_msg4 = re.escape("Index.max(): only ordered categoricals are possible")
    err_msg5 = re.escape("Index.argmin(): axis parameter only supports default value 0")
    err_msg6 = re.escape("Index.argmax(): axis parameter only supports default value 0")
    err_msg7 = re.escape("Index.min(): axis parameter only supports default value None")
    err_msg8 = re.escape("Index.max(): axis parameter only supports default value None")
    err_msg9 = re.escape(
        "Index.argmin(): skipna parameter only supports default value True"
    )
    err_msg10 = re.escape(
        "Index.argmax(): skipna parameter only supports default value True"
    )
    err_msg11 = re.escape(
        "Index.min(): skipna parameter only supports default value True"
    )
    err_msg12 = re.escape(
        "Index.max(): skipna parameter only supports default value True"
    )

    I1 = pd.CategoricalIndex(list("ABCAACAB"))
    I2 = pd.Index([1, 5, 2, 1, 0, 1, 5, 2, 1, 3])

    with pytest.raises(BodoError, match=err_msg1):
        bodo.jit(impl1)(I1)

    with pytest.raises(BodoError, match=err_msg2):
        bodo.jit(impl2)(I1)

    with pytest.raises(BodoError, match=err_msg3):
        bodo.jit(impl3)(I1)

    with pytest.raises(BodoError, match=err_msg4):
        bodo.jit(impl4)(I1)

    with pytest.raises(BodoError, match=err_msg5):
        bodo.jit(impl5)(I2)

    with pytest.raises(BodoError, match=err_msg6):
        bodo.jit(impl6)(I2)

    with pytest.raises(BodoError, match=err_msg7):
        bodo.jit(impl7)(I2)

    with pytest.raises(BodoError, match=err_msg8):
        bodo.jit(impl8)(I2)

    with pytest.raises(BodoError, match=err_msg9):
        bodo.jit(impl9)(I2)

    with pytest.raises(BodoError, match=err_msg10):
        bodo.jit(impl10)(I2)

    with pytest.raises(BodoError, match=err_msg11):
        bodo.jit(impl11)(I2)

    with pytest.raises(BodoError, match=err_msg12):
        bodo.jit(impl12)(I2)


@pytest.mark.slow
def test_range_index_from_range():
    def impl():
        return pd.RangeIndex.from_range(range(10))

    err_msg = re.escape("pandas.RangeIndex.from_range() is not yet supported")

    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)()


@pytest.mark.slow
def test_sort_values_args():
    def impl1(I):
        return I.sort_values(return_indexer=True)

    def impl2(I):
        return I.sort_values(key=lambda x: x + 1)

    def impl3(I):
        return I.sort_values(ascending=42)

    def impl4(I, na_position):
        return I.sort_values(na_position=na_position)

    err_msg1 = re.escape(
        "Index.sort_values(): return_indexer parameter only supports default value False"
    )
    err_msg2 = re.escape(
        "Index.sort_values(): key parameter only supports default value None"
    )
    err_msg3 = re.escape(
        "Index.sort_values(): 'ascending' parameter must be of type bool"
    )
    err_msg4 = re.escape(
        "Index.sort_values(): 'na_position' should either be 'first' or 'last'"
    )

    I = pd.Index([1, 8, -9, 0, 3, 1, 9, -8])

    with pytest.raises(BodoError, match=err_msg1):
        bodo.jit(impl1)(I)

    with pytest.raises(BodoError, match=err_msg2):
        bodo.jit(impl2)(I)

    with pytest.raises(BodoError, match=err_msg3):
        bodo.jit(impl3)(I)

    with pytest.raises(BodoError, match=err_msg4):
        bodo.jit(impl4)(I, "start")

    with pytest.raises(BodoError, match=err_msg4):
        bodo.jit(impl4)(I, 42)


@pytest.mark.slow
def test_where_args():
    def impl(I, C, O):
        return I.where(C, O)

    err_msg1 = re.escape("where() index and 'other' must share a common type.")
    err_msg2 = re.escape(
        "where() 'other' must be a scalar, non-categorical series, 1-dim numpy array or StringArray with a matching type for Index."
    )

    I = pd.Index([1, 8, -9, 0, 3, 1, 9, -8])
    C = pd.array([True, False] * 4)

    with pytest.raises(BodoError, match=err_msg1):
        bodo.jit(impl)(I, C, None)

    with pytest.raises(BodoError, match=err_msg1):
        bodo.jit(impl)(I, C, "A")

    with pytest.raises(BodoError, match=err_msg2):
        bodo.jit(impl)(I, C, list(range(8)))

    with pytest.raises(BodoError, match=err_msg2):
        bodo.jit(impl)(I, C, I)

    with pytest.raises(BodoError, match=err_msg2):
        bodo.jit(impl)(I, C, pd.Series(list("ABCDEFGH")))


@pytest.mark.slow
def test_putmask_args():
    def impl(I, C, O):
        return I.putmask(C, O)

    err_msg1 = re.escape("putmask() index and 'other' must share a common type.")
    err_msg2 = re.escape(
        "putmask() 'other' must be a scalar, non-categorical series, 1-dim numpy array or StringArray with a matching type for Index."
    )

    I = pd.Index([1, 8, -9, 0, 3, 1, 9, -8])
    C = pd.array([True, False] * 4)

    with pytest.raises(BodoError, match=err_msg1):
        bodo.jit(impl)(I, C, None)

    with pytest.raises(BodoError, match=err_msg1):
        bodo.jit(impl)(I, C, "A")

    with pytest.raises(BodoError, match=err_msg2):
        bodo.jit(impl)(I, C, list(range(8)))

    with pytest.raises(BodoError, match=err_msg2):
        bodo.jit(impl)(I, C, I)

    with pytest.raises(BodoError, match=err_msg2):
        bodo.jit(impl)(I, C, pd.Series(list("ABCDEFGH")))


@pytest.mark.slow
def test_set_operations_unsupported():
    def impl1(I, J):
        return I.union(J)

    def impl2(I, J):
        return I.intersection(J)

    def impl3(I, J):
        return I.difference(J)

    def impl4(I, J):
        return I.symmetric_difference(J)

    def impl5(I, J):
        return I.union(J, sort=False)

    def impl6(I, J):
        return I.intersection(J, sort=False)

    def impl7(I, J):
        return I.difference(J, sort=False)

    def impl8(I, J):
        return I.symmetric_difference(J, sort=False)

    def impl9(I, J):
        return I.symmetric_difference(J, result_name="sym")

    err_msg1 = re.escape("Index.union(): incompatible types int64 and unicode_type")
    err_msg2 = re.escape(
        "Index.intersection(): incompatible types unicode_type and readonly bytes(uint8, 1d, C)"
    )
    err_msg3 = re.escape("Index.difference(): incompatible types int64 and float64")
    err_msg4 = re.escape(
        "Index.symmetric_difference(): incompatible types datetime64[ns] and timedelta64[ns]"
    )
    err_msg5 = re.escape(
        "Index.union(): sort parameter only supports default value None"
    )
    err_msg6 = re.escape(
        "Index.intersection(): sort parameter only supports default value None"
    )
    err_msg7 = re.escape(
        "Index.difference(): sort parameter only supports default value None"
    )
    err_msg8 = re.escape(
        "Index.symmetric_difference(): sort parameter only supports default value None"
    )
    err_msg9 = re.escape(
        "Index.symmetric_difference(): result_name parameter only supports default value None"
    )
    err_msg10 = re.escape(
        "pd.Index.union(): unsupported type for argument other: reflected list(int64)<iv=None>"
    )
    err_msg11 = re.escape(
        "Index.union(): unsupported type for argument other: UniTuple(int64 x 4)"
    )
    err_msg12 = re.escape(
        "pd.Index.union(): unsupported type for argument other: IntegerArrayType(int64)"
    )
    err_msg13 = re.escape(
        "pd.Index.union(): unsupported type for argument other: unicode_type"
    )

    with pytest.raises(BodoError, match=err_msg1):
        bodo.jit(impl1)(pd.Index([1, 2, 3, 4, 5]), pd.Index(list("ABCDE")))

    with pytest.raises(BodoError, match=err_msg2):
        bodo.jit(impl2)(
            pd.Index(list("AEIOU")), pd.Index([b"A", b"E", b"I", b"O", b"U"])
        )

    with pytest.raises(BodoError, match=err_msg3):
        bodo.jit(impl3)(pd.RangeIndex(0, 100, 7), pd.Index([1.0, 7.0, 20.0, 21.0]))

    with pytest.raises(BodoError, match=err_msg4):
        bodo.jit(impl4)(
            pd.date_range("2018-01-01", "2018-12-01", freq="M"),
            pd.TimedeltaIndex(
                ["1 days", "2 days", "3 days", "4 days", "5 days", "6 days"]
            ),
        )

    with pytest.raises(BodoError, match=err_msg5):
        bodo.jit(impl5)(pd.Index([1, 2, 3, 4, 5]), pd.Index([2, 4, 6, 8, 10]))

    with pytest.raises(BodoError, match=err_msg6):
        bodo.jit(impl6)(pd.Index([1, 2, 3, 4, 5]), pd.Index([2, 4, 6, 8, 10]))

    with pytest.raises(BodoError, match=err_msg7):
        bodo.jit(impl7)(pd.Index([1, 2, 3, 4, 5]), pd.Index([2, 4, 6, 8, 10]))

    with pytest.raises(BodoError, match=err_msg8):
        bodo.jit(impl8)(pd.Index([1, 2, 3, 4, 5]), pd.Index([2, 4, 6, 8, 10]))

    with pytest.raises(BodoError, match=err_msg9):
        bodo.jit(impl9)(pd.Index([1, 2, 3, 4, 5]), pd.Index([2, 4, 6, 8, 10]))

    with pytest.raises(BodoError, match=err_msg10):
        bodo.jit(impl1)(pd.Index([1, 2, 3, 4, 5]), [1, 3, 5, 7])

    with pytest.raises(BodoError, match=err_msg11):
        bodo.jit(impl1)(pd.Index([1, 2, 3, 4, 5]), (1, 3, 5, 7))

    # TODO: allow pd.array [BE-3063]
    with pytest.raises(BodoError, match=err_msg12):
        bodo.jit(impl1)(pd.Index([1, 2, 3, 4, 5]), pd.array([3, 5, 7, 9, 11]))

    with pytest.raises(BodoError, match=err_msg13):
        bodo.jit(impl1)(pd.Index([1, 2, 3, 4, 5]), "foo")


# TODO: handle cases where n is not the same length as I
@pytest.mark.slow
def test_repeat_malformed():
    def impl(I, n):
        return I.repeat(n)

    err_msg1 = re.escape("negative dimensions not allowed")
    err_msg2 = re.escape(
        "Index.repeat(): 'repeats' should be an integer or array of integers"
    )

    I = pd.Index([1, 2, 3])

    with pytest.raises(ValueError, match=err_msg1):
        bodo.jit(impl)(I, -4)

    with pytest.raises(BodoError, match=err_msg2):
        bodo.jit(impl)(I, "ABC")


def check_unsupported_atr(idx_format_str, idx_val, unsupported_atr):
    func_text = f"""
def impl(I):
    return I.{unsupported_atr}
"""

    loc_vars = {}
    exec(func_text, {"bodo": bodo, "np": np}, loc_vars)
    impl = loc_vars["impl"]

    unsupported_str = idx_format_str.format(unsupported_atr)
    err_msg = f"{unsupported_str} not supported yet"

    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)(idx_val)


def check_unsupported_method(idx_format_str, idx_val, unsupported_method):
    # The overload matches any combination of arguments, so we don't have to worry
    func_text = f"""
def impl(I):
    return I.{unsupported_method}()
"""

    loc_vars = {}
    exec(func_text, {"bodo": bodo, "np": np}, loc_vars)
    impl = loc_vars["impl"]

    unsupported_str = idx_format_str.format(unsupported_method + "()")
    err_msg = re.escape(f"{unsupported_str} not supported yet")

    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)(idx_val)


def test_index_copy_kwd_arg_err_msg(all_index_types):
    """tests that the proper errors are raised when doing Index.copy() with unsupported kwd args"""
    from bodo.hiframes.pd_index_ext import idx_typ_to_format_str_map

    idx_val = all_index_types[0]

    idx_typ_str = idx_typ_to_format_str_map[type(bodo.typeof(idx_val))].format("copy()")
    if isinstance(bodo.typeof(idx_val), (MultiIndexType, IntervalIndexType)):
        err_string = f"{idx_typ_str}: not yet supported"
    else:
        err_string = f"{idx_typ_str}: dtype parameter only supports default value None"

    def impl(I):
        return I.copy(dtype=np.int32)

    #  default value none.

    err_string = idx_typ_to_format_str_map[type(bodo.typeof(idx_val))].format("copy()")
    full_err_msg = ".*" + re.escape(err_string) + ".*"

    with pytest.raises(
        BodoError,
        match=full_err_msg,
    ):
        bodo.jit(impl)(idx_val)


def test_index_take_kwd_arg_err_msg(all_index_types):
    """tests that the proper errors are raised when doing Index.copy() with unsupported kwd args"""
    from bodo.hiframes.pd_index_ext import idx_typ_to_format_str_map

    idx_val = all_index_types[0]

    if isinstance(bodo.typeof(idx_val), (MultiIndexType, IntervalIndexType)):
        idx_typ_str = idx_typ_to_format_str_map[type(bodo.typeof(idx_val))].format(
            "take()"
        )
        err_string = f"{idx_typ_str} not supported yet"
    else:
        err_string = (
            "Index.take(): fill_value parameter only supports default value None"
        )

    def impl(I):
        return I.take(slice(0, 10), fill_value=5)

    full_err_msg = ".*" + re.escape(err_string) + ".*"

    with pytest.raises(
        BodoError,
        match=full_err_msg,
    ):
        bodo.jit(impl)(idx_val)


def test_cat_idx_init_err():
    def impl():
        pd.CategoricalIndex(["A", "B", "C"])

    err_msg = (
        ".*" + re.escape("pd.CategoricalIndex() initializer not yet supported.") + ".*"
    )

    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)()


def test_dti_init_kwd_err():
    def impl():
        pd.DatetimeIndex(
            pd.date_range(start="2018-04-24", end="2018-04-25", periods=5),
            normalize=True,
        )

    err_msg = (
        ".*"
        + re.escape(
            "pandas.DatetimeIndex(): normalize parameter only supports default value False"
        )
        + ".*"
    )

    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)()


def test_tdi_init_kwd_err():
    def impl():
        pd.TimedeltaIndex(np.arange(100), unit="s")

    err_msg = (
        ".*"
        + re.escape(
            "pandas.TimedeltaIndex(): unit parameter only supports default value None"
        )
        + ".*"
    )

    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)()


def test_idx_int64_init_err():
    def impl():
        pd.Int64Index(np.arange(100), dtype=np.int32)

    err_msg = (
        ".*"
        + re.escape(
            "pandas.Int64Index(): dtype parameter only supports default value None"
        )
        + ".*"
    )
    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)()


def test_idx_uint64_init_err():
    def impl():
        pd.UInt64Index(np.arange(100), dtype=np.uint32)

    err_msg = (
        ".*"
        + re.escape(
            "pandas.UInt64Index(): dtype parameter only supports default value None"
        )
        + ".*"
    )
    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)()


def test_idx_float64_init_err():
    def impl():
        pd.Float64Index(np.arange(100), dtype=np.float32)

    err_msg = (
        ".*"
        + re.escape(
            "pandas.Float64Index(): dtype parameter only supports default value None"
        )
        + ".*"
    )
    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)()


@pytest.mark.skip("TODO")
def test_idx_map_tup_return():
    index = pd.Index(np.arange(10))

    def test_impl(I):
        return I.map(lambda a: (1, a))

    check_func(test_impl, (index,))


@pytest.mark.parametrize(
    "index",
    [
        pd.Index(["A", "B", "C", "D"]),
        pytest.param(
            pd.Index([b"hello", b"world", b"", b"test", bytes(2), b"CC"]),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.CategoricalIndex(["A", "B", "B", "A", "C", "A", "B", "C"]),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.IntervalIndex.from_arrays(np.arange(11), np.arange(11) + 1),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.PeriodIndex(
                year=[2015, 2015, 2016, 1026, 2018, 2018, 2019],
                month=[1, 2, 3, 1, 2, 3, 4],
                freq="M",
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.MultiIndex.from_arrays([[1, 5, 9], [2, 1, 8]]), marks=pytest.mark.slow
        ),
    ],
)
def test_monotonic_unsupported(index):
    """
    Checks that is_monotonic, is_monotonic_increasing, and is_monotonic_decreasing attributes
    throw error for unsupported index types (i.e. not a NumericIndex, DatetimeIndex,
    TimedeltaIndex, or RangeIndex).
    """

    def test_unsupp_is_monotonic(idx):
        return idx.is_monotonic

    def test_unsupp_is_monotonic_increasing(idx):
        return idx.is_monotonic_increasing

    def test_unsupp_is_monotonic_decreasing(idx):
        return idx.is_monotonic_decreasing

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_unsupp_is_monotonic)(index)

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_unsupp_is_monotonic_increasing)(index)

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_unsupp_is_monotonic_decreasing)(index)
