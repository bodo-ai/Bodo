# Copyright (C) 2021 Bodo Inc. All rights reserved.
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
def test_interval_index_from_breaks():
    def impl():
        return pd.IntervalIndex.from_breaks([0, 1, 2, 3])

    err_msg = re.escape("pandas.IntervalIndex.from_breaks() is not yet supported")

    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)()


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
