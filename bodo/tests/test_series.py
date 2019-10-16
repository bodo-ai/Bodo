# Copyright (C) 2019 Bodo Inc.
import unittest
import os
import operator
import pandas as pd
import numpy as np
import random
import string
import pyarrow.parquet as pq
import numba
import numba.targets.ufunc_db
import bodo
from bodo.libs.str_arr_ext import StringArray
from bodo.tests.utils import (
    count_array_REPs,
    count_parfor_REPs,
    count_parfor_OneDs,
    count_array_OneDs,
    dist_IR_contains,
    get_start_end,
    check_func,
    is_bool_object_series,
)
import pytest


_cov_corr_series = [
    (pd.Series(x), pd.Series(y))
    for x, y in [
        ([np.nan, -2.0, 3.0, 9.1], [np.nan, -2.0, 3.0, 5.0]),
        # TODO(quasilyte): more intricate data for complex-typed series.
        # Some arguments make assert_almost_equal fail.
        # Functions that yield mismaching results: _column_corr_impl and _column_cov_impl.
        (
            [complex(-2.0, 1.0), complex(3.0, 1.0)],
            [complex(-3.0, 1.0), complex(2.0, 1.0)],
        ),
        ([complex(-2.0, 1.0), complex(3.0, 1.0)], [1.0, -2.0]),
        ([1.0, -4.5], [complex(-4.5, 1.0), complex(3.0, 1.0)]),
    ]
]


GLOBAL_VAL = 2


# TODO: integer Null and other Nulls
# TODO: list of datetime.datetime, categorical, timedelta, ...
@pytest.mark.parametrize(
    "data",
    [
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
        pd.Int64Index([10, 12, 13]),
        pd.Int64Index([10, 12, 14], name="A"),
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
        pd.Int64Index([10, 12, 13]),
        pd.Int64Index([10, 12, 14], name="A"),
        pd.RangeIndex(1, 4, 1),
        None,
    ],
)
@pytest.mark.parametrize("name", [None, "ABC"])
def test_series_constructor(data, index, name):
    # set Series index to avoid implicit alignment in Pandas case
    if isinstance(data, pd.Series) and index is not None:
        data.index = index

    def impl(d, i, n):
        return pd.Series(d, i, name=n)

    bodo_func = bodo.jit(impl)
    pd.testing.assert_series_equal(
        bodo_func(data, index, name), impl(data, index, name), check_dtype=False
    )


def test_series_constructor_dtype1():
    def impl(d):
        return pd.Series(d, dtype=np.int32)

    check_func(impl, ([3, 4, 1, -3, 0],), is_out_distributed=False)
    check_func(impl, (np.array([3, 4, 1, -3, 0]),))


def test_series_constructor_dtype2():
    def impl(d):
        return pd.Series(d, dtype="int32")

    check_func(impl, ([3, 4, 1, -3, 0],), is_out_distributed=False)
    check_func(impl, (np.array([3, 4, 1, -3, 0]),))


def test_series_constructor_int_arr():
    def impl(d):
        return pd.Series(d, dtype="Int32")

    check_func(impl, ([3, 4, 1, -3, 0],), is_out_distributed=False)
    check_func(impl, (np.array([3, 4, 1, -3, 0]),))
    check_func(impl, (np.array([1, 4, 1, np.nan, 0], dtype=np.float32),))


# using length of 5 arrays to enable testing on 3 ranks (2, 2, 1 distribution)
# zero length chunks on any rank can cause issues, TODO: fix
# TODO: other possible Series types like Categorical, dt64, td64, ...
@pytest.fixture(
    params=[
        pd.Series([1, 8, 4, 11, -3]),
        pd.Series([1.1, np.nan, 4.2, 3.1, -3.5]),
        pd.Series([True, False, False, True, True]),  # bool array without NA
        pd.Series([True, False, False, np.nan, True]),  # bool array with NA
        pd.Series([1, 8, 4, 0, 3], dtype=np.uint8),
        pd.Series([1, 8, 4, 10, 3], dtype="Int32"),
        pd.Series([1, 8, 4, -1, 2], name="ACD"),
        pd.Series([1, 8, 4, 1, -3], [3, 7, 9, 2, 1]),
        pd.Series([1, 8, 4, 11, -3], [3, 7, 9, 2, 1], name="AAC"),
        pd.Series([1, 2, 3, -1, 6], ["A", "BA", "", "DD", "GGG"]),
        pd.Series(
            ["A", "B", "CDD", "AA", "GGG"]
        ),  # TODO: string with Null (np.testing fails)
        pd.Series(["A", "B", "CG", "ACDE", "C"], [4, 7, 0, 1, -2]),
        pd.Series(pd.date_range(start="2018-04-24", end="2018-04-29", periods=5)),
        pd.Series(
            [3, 5, 1, -1, 2],
            pd.date_range(start="2018-04-24", end="2018-04-29", periods=5),
        ),
        # TODO: timedelta
    ]
)
def series_val(request):
    return request.param


# TODO: timedelta, period, tuple, etc.
@pytest.fixture(
    params=[
        pd.Series([1, 8, 4, 11, -3]),
        pd.Series([1.1, np.nan, 4.1, 1.4, -2.1]),
        pd.Series([1, 8, 4, 10, 3], dtype=np.uint8),
        pd.Series([1, 8, 4, 10, 3], dtype="Int32"),
        pd.Series([1, 8, 4, -1, 2], [3, 7, 9, 2, 1], name="AAC"),
        pd.Series(pd.date_range(start="2018-04-24", end="2018-04-29", periods=5)),
    ]
)
def numeric_series_val(request):
    return request.param


def test_box(series_val):
    # unbox and box
    def impl(S):
        return S

    check_func(impl, (series_val,))


def test_series_index(series_val):
    def test_impl(S):
        return S.index

    check_func(test_impl, (series_val,))


def test_series_index_none():
    def test_impl():
        S = pd.Series([1, 4, 8])
        return S.index

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_index_equal(bodo_func(), test_impl())


def test_series_values(series_val):
    def test_impl(S):
        return S.values

    check_func(test_impl, (series_val,))


def test_series_dtype(numeric_series_val):
    def test_impl(S):
        return S.dtype

    check_func(test_impl, (numeric_series_val,))


def test_series_shape(series_val):
    def test_impl(S):
        return S.shape

    check_func(test_impl, (series_val,))


def test_series_ndim(series_val):
    def test_impl(S):
        return S.ndim

    check_func(test_impl, (series_val,))


def test_series_size(series_val):
    def test_impl(S):
        return S.size

    check_func(test_impl, (series_val,))


def test_series_T(series_val):
    def test_impl(S):
        return S.T

    check_func(test_impl, (series_val,))


def test_series_hasnans(series_val):
    def test_impl(S):
        return S.hasnans

    check_func(test_impl, (series_val,))


def test_series_empty(series_val):
    def test_impl(S):
        return S.empty

    check_func(test_impl, (series_val,))


def test_series_dtypes(numeric_series_val):
    def test_impl(S):
        return S.dtypes

    check_func(test_impl, (numeric_series_val,))


def test_series_name(series_val):
    def test_impl(S):
        return S.name

    check_func(test_impl, (series_val,))


def test_series_put(series_val):
    # IntegerArray doesn't have put
    if isinstance(series_val.dtype, pd.core.arrays.integer._IntegerDtype):
        return

    def test_impl(S):
        S.put(0, S.values[1])
        return S

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(
        bodo_func(series_val.copy()), test_impl(series_val.copy()), check_dtype=False
    )


def test_series_astype_numeric(numeric_series_val):
    # datetime can't be converted to float
    if numeric_series_val.dtype == np.dtype("datetime64[ns]"):
        return

    def test_impl(S):
        return S.astype(np.float64)

    check_func(test_impl, (numeric_series_val,))


def test_series_astype_str(series_val):
    # XXX str(float) not consistent with Python yet
    if series_val.dtype == np.float64:
        return

    if series_val.dtype == np.dtype("datetime64[ns]"):
        return

    def test_impl(S):
        return S.astype(str)

    check_func(test_impl, (series_val,))


def test_series_astype_int_arr(numeric_series_val):
    # only integers can be converted safely
    if not pd.api.types.is_integer_dtype(numeric_series_val):
        return

    def test_impl(S):
        return S.astype("Int64")

    check_func(test_impl, (numeric_series_val,))


@pytest.mark.parametrize(
    "S",
    [
        pd.Series([True, False, False, True, True]),
        pd.Series([True, False, False, np.nan, True]),
    ],
)
def test_series_astype_bool_arr(S):
    # TODO: int, Int

    def test_impl(S):
        return S.astype("float32")

    check_func(test_impl, (S,))


@pytest.mark.skip(reason="categorical feature gaps")
@pytest.mark.parametrize("S", [pd.Series(["A", "BB", "A", "BBB", "BB", "A"])])
def test_series_astype_cat(S):
    ctype = pd.CategoricalDtype(S.unique())

    def test_impl(S):
        return S.astype(ctype)

    check_func(test_impl, (S,))


@pytest.mark.parametrize(
    "S", [pd.Series(["A", "BB", "A", "BBB", "BB", "A"]).astype("category")]
)
def test_series_cat_box(S):
    def test_impl(S):
        return S

    check_func(test_impl, (S,))


@pytest.mark.parametrize(
    "S", [pd.Series(["A", "BB", "A", "BBB", "BB", "A"]).astype("category")]
)
def test_series_cat_comp(S):
    def test_impl(S):
        return S == "BB"

    check_func(test_impl, (S,))


def test_series_copy_deep(series_val):
    # TODO: test deep/shallow cases properly
    def test_impl(S):
        return S.copy()

    check_func(test_impl, (series_val,))


def test_series_copy_shallow(series_val):
    # TODO: test deep/shallow cases properly
    def test_impl(S):
        return S.copy(deep=False)

    check_func(test_impl, (series_val,))


def test_series_to_list(series_val):
    # XXX can't compare nans here, TODO: fix
    if series_val.hasnans:
        return

    def test_impl(S):
        return S.to_list()

    bodo_func = bodo.jit(test_impl)
    assert bodo_func(series_val) == test_impl(series_val)


def test_series_get_values(series_val):
    def test_impl(S):
        return S.get_values()

    check_func(test_impl, (series_val,))


def test_series_iat_getitem(series_val):
    def test_impl(S):
        return S.iat[2]

    bodo_func = bodo.jit(test_impl)
    assert bodo_func(series_val) == test_impl(series_val)
    # fix distributed
    # check_func(test_impl, (series_val,))


def test_series_iat_setitem(series_val):
    # string setitem not supported yet
    if isinstance(series_val.iat[0], str):
        return
    val = series_val.iat[0]

    def test_impl(S, val):
        S.iat[2] = val
        # print(S) TODO: fix crash
        return S

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(
        bodo_func(series_val.copy(), val),
        test_impl(series_val.copy(), val),
        check_dtype=False,
    )


def test_series_iloc_getitem_int(series_val):
    def test_impl(S):
        return S.iloc[2]

    bodo_func = bodo.jit(test_impl)
    assert bodo_func(series_val) == test_impl(series_val)
    # fix distributed
    # check_func(test_impl, (series_val,))


def test_series_iloc_getitem_slice(series_val):
    def test_impl(S):
        return S.iloc[1:4]

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(
        bodo_func(series_val), test_impl(series_val), check_dtype=False
    )
    # fix distributed
    # check_func(test_impl, (series_val,))


def test_series_iloc_getitem_array_int(series_val):
    def test_impl(S):
        return S.iloc[[1, 3]]

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(
        bodo_func(series_val), test_impl(series_val), check_dtype=False
    )


def test_series_iloc_getitem_array_bool(series_val):
    def test_impl(S):
        return S.iloc[[True, True, False, True, False]]

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(
        bodo_func(series_val), test_impl(series_val), check_dtype=False
    )


def test_series_iloc_setitem_int(series_val):
    # string setitem not supported yet
    if isinstance(series_val.iat[0], str):
        return
    val = series_val.iat[0]

    def test_impl(S, val):
        S.iloc[2] = val
        # print(S) TODO: fix crash
        return S

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(
        bodo_func(series_val.copy(), val),
        test_impl(series_val.copy(), val),
        check_dtype=False,
    )


def test_series_iloc_setitem_slice(series_val):
    # string setitem not supported yet
    if isinstance(series_val.iat[0], str):
        return

    val = series_val.iloc[0:3].values.copy()  # values to avoid alignment
    if series_val.hasnans:
        val[0] = np.nan  # extra NA to keep dtype nullable like bool arr

    def test_impl(S, val):
        S.iloc[1:4] = val
        return S

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(
        bodo_func(series_val.copy(), val),
        test_impl(series_val.copy(), val),
        check_dtype=False,
    )


@pytest.mark.parametrize("idx", [[1, 3], np.array([1, 3]), pd.Series([1, 3])])
def test_series_iloc_setitem_list_int(series_val, idx):
    # string setitem not supported yet
    if isinstance(series_val.iat[0], str):
        return

    val = series_val.iloc[0:2].values.copy()  # values to avoid alignment
    if series_val.hasnans:
        val[0] = np.nan  # extra NA to keep dtype nullable like bool arr

    def test_impl(S, val, idx):
        S.iloc[idx] = val
        return S

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(
        bodo_func(series_val.copy(), val, idx),
        test_impl(series_val.copy(), val, idx),
        check_dtype=False,
    )


####### getitem tests ###############


def test_series_getitem_int(series_val):
    def test_impl(S):
        return S[2]

    bodo_func = bodo.jit(test_impl)
    # integer label-based indexing should raise error
    if type(series_val.index) in (pd.Int64Index, pd.UInt64Index):
        with pytest.raises(numba.TypingError):  # TODO: ValueError
            bodo_func(series_val)
    else:
        assert bodo_func(series_val) == test_impl(series_val)


def test_series_getitem_slice(series_val):
    def test_impl(S):
        return S[1:4]

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(
        bodo_func(series_val), test_impl(series_val), check_dtype=False
    )


@pytest.mark.parametrize("idx", [[1, 3], np.array([1, 3]), pd.Series([1, 3])])
def test_series_getitem_list_int(series_val, idx):
    def test_impl(S, idx):
        return S[idx]

    bodo_func = bodo.jit(test_impl)
    # integer label-based indexing should raise error
    if type(series_val.index) in (pd.Int64Index, pd.UInt64Index):
        with pytest.raises(numba.TypingError):  # TODO: ValueError
            bodo_func(series_val, idx)
    else:
        pd.testing.assert_series_equal(
            bodo_func(series_val, idx), test_impl(series_val, idx), check_dtype=False
        )


def test_series_getitem_array_bool(series_val):
    def test_impl(S):
        return S[[True, True, False, True, False]]

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(
        bodo_func(series_val), test_impl(series_val), check_dtype=False
    )


############### setitem tests #################


def test_series_setitem_int(series_val):
    # string setitem not supported yet
    if isinstance(series_val.iat[0], str):
        return
    val = series_val.iat[0]

    def test_impl(S, val):
        S[2] = val
        return S

    bodo_func = bodo.jit(test_impl)
    # integer label-based indexing should raise error
    if type(series_val.index) in (pd.Int64Index, pd.UInt64Index):
        with pytest.raises(numba.TypingError):  # TODO: ValueError
            bodo_func(series_val, val)
    else:
        pd.testing.assert_series_equal(
            bodo_func(series_val.copy(), val),
            test_impl(series_val.copy(), val),
            check_dtype=False,
        )


def test_series_setitem_slice(series_val):
    # string setitem not supported yet
    if isinstance(series_val.iat[0], str):
        return

    val = series_val.iloc[0:3].values.copy()  # values to avoid alignment
    if series_val.hasnans:
        val[0] = np.nan  # extra NA to keep dtype nullable like bool arr

    def test_impl(S, val):
        S[1:4] = val
        return S

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(
        bodo_func(series_val.copy(), val),
        test_impl(series_val.copy(), val),
        check_dtype=False,
    )


@pytest.mark.parametrize("idx", [[1, 4], np.array([1, 4]), pd.Series([1, 4])])
@pytest.mark.parametrize("list_val_arg", [True, False])
def test_series_setitem_list_int(series_val, idx, list_val_arg):
    # string setitem not supported yet
    if isinstance(series_val.iat[0], str):
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
        with pytest.raises(numba.TypingError):  # TODO: ValueError
            bodo_func(series_val, val, idx)
    else:
        # Pandas coerces Series type to set values, so avoid low precision
        # TODO: warn or error?
        if list_val_arg and series_val.dtype in (
            np.int8,
            np.uint8,
            np.int16,
            np.uint16,
            np.int32,
            np.uint32,
        ):
            return
        pd.testing.assert_series_equal(
            bodo_func(series_val.copy(), val, idx),
            test_impl(series_val.copy(), val, idx),
            check_dtype=False,
        )


############################ binary ops #############################


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
def test_series_explicit_binary_op(numeric_series_val, op, fill):
    # dt64 not supported here
    if numeric_series_val.dtype == np.dtype("datetime64[ns]"):
        return
    # XXX ne operator is buggy in Pandas and doesn't set NaNs in output
    # when both inputs are NaNs
    if op is "ne" and numeric_series_val.hasnans:
        return
    # Numba returns float32 for truediv but Numpy returns float64
    if op is "truediv" and numeric_series_val.dtype == np.uint8:
        return
    if op is "pow" and numeric_series_val.dtype in (
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
    check_func(test_impl, (numeric_series_val, numeric_series_val, fill))


@pytest.mark.parametrize("fill", [None, 1.6])
def test_series_explicit_binary_op_nan(fill):
    # test nan conditions (both nan, left nan, right nan)
    def test_impl(S, other, fill_val):
        return S.add(other, fill_value=fill_val)

    L1 = pd.Series([1.0, np.nan, 2.3, np.nan])
    L2 = pd.Series([1.0, np.nan, np.nan, 1.1], name="ABC")
    check_func(test_impl, (L1, L2, fill))


@pytest.mark.parametrize("op", bodo.hiframes.pd_series_ext.series_binary_ops)
def test_series_binary_op(op):
    op_str = numba.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(S, other):\n"
    func_text += "  return S {} other\n".format(op_str)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars["test_impl"]

    S = pd.Series([4, 6, 7, 1], [3, 5, 0, 7], name="ABC")
    check_func(test_impl, (S, S))
    check_func(test_impl, (S, 2))
    check_func(test_impl, (2, S))


@pytest.mark.parametrize("op", bodo.hiframes.pd_series_ext.series_inplace_binary_ops)
def test_series_inplace_binary_op(op):
    op_str = numba.utils.OPERATORS_TO_BUILTINS[op]
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
def test_series_unary_op(op):
    # TODO: fix operator.pos
    if op == operator.pos:
        return

    op_str = numba.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(S):\n"
    func_text += "  return {} S\n".format(op_str)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars["test_impl"]

    S = pd.Series([4, 6, 7, 1], [3, 5, 0, 7], name="ABC")
    check_func(test_impl, (S,))


@pytest.mark.parametrize(
    "ufunc", [f for f in numba.targets.ufunc_db.get_ufuncs() if f.nin == 1]
)
def test_series_unary_ufunc(ufunc):
    def test_impl(S):
        return ufunc(S)

    S = pd.Series([4, 6, 7, 1], [3, 5, 0, 7], name="ABC")
    check_func(test_impl, (S,))


def test_series_unary_ufunc_np_call():
    # a ufunc called explicitly, since the above test sets module name as
    # 'ufunc' instead of 'numpy'
    def test_impl(S):
        return np.negative(S)

    S = pd.Series([4, 6, 7, 1], [3, 5, 0, 7], name="ABC")
    check_func(test_impl, (S,))


@pytest.mark.parametrize(
    "ufunc", [f for f in numba.targets.ufunc_db.get_ufuncs() if f.nin == 2]
)
def test_series_binary_ufunc(ufunc):
    def test_impl(S1, S2):
        return ufunc(S1, S2)

    S = pd.Series([4, 6, 7, 1], [3, 5, 0, 7], name="ABC")
    A = np.array([1, 3, 7, 11])
    check_func(test_impl, (S, S))
    check_func(test_impl, (S, A))
    check_func(test_impl, (A, S))


@pytest.mark.parametrize(
    "op", [operator.eq, operator.ne, operator.ge, operator.gt, operator.le, operator.lt]
)
@pytest.mark.parametrize(
    "S",
    [
        pd.Series([True, False, False, True, True]),
        pd.Series([True, False, np.nan, True, True]),
    ],
)
def test_series_bool_cmp_op(S, op):
    op_str = numba.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(S, other):\n"
    func_text += "  return S {} other\n".format(op_str)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars["test_impl"]

    check_func(test_impl, (S, S))
    check_func(test_impl, (S, True))
    check_func(test_impl, (True, S))


@pytest.mark.parametrize(
    "op", [operator.eq, operator.ne, operator.ge, operator.gt, operator.le, operator.lt]
)
@pytest.mark.parametrize(
    "S",
    [
        pd.Series([True, False, False, True, True]),
        pd.Series([True, False, np.nan, True, True]),
    ],
)
def test_series_bool_vals_cmp_op(S, op):
    op_str = numba.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(S, other):\n"
    func_text += "  return S.values {} other.values\n".format(op_str)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars["test_impl"]

    check_func(test_impl, (S, S))


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
def test_series_combine(S1, S2, fill, raises):
    def test_impl(S1, S2, fill_val):
        return S1.combine(S2, lambda a, b: 2 * a + b, fill_val)

    bodo_func = bodo.jit(test_impl)
    if raises:
        with pytest.raises(AssertionError):
            bodo_func(S1, S2, fill)
    else:
        # TODO: fix 1D_Var chunk size mismatch on inputs with different sizes
        pd.testing.assert_series_equal(bodo_func(S1, S2, fill), test_impl(S1, S2, fill))


def test_series_combine_kws():
    def test_impl(S1, S2, fill_val):
        return S1.combine(other=S2, func=lambda a, b: 2 * a + b, fill_value=fill_val)

    S1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    S2 = pd.Series([6.0, 21.0, 3.6, 5.0, 0.0])
    fill = 1237.56
    check_func(test_impl, (S1, S2, fill))


def test_series_combine_no_fill():
    def test_impl(S1, S2):
        return S1.combine(other=S2, func=lambda a, b: 2 * a + b)

    S1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    S2 = pd.Series([6.0, 21.0, 3.6, 5.0, 0.0])
    check_func(test_impl, (S1, S2))


@pytest.mark.parametrize(
    "S",
    [
        pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
        pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], [3, 1, 0, 2, 4], name="ABC"),
    ],
)
def test_series_apply(S):
    def test_impl(S):
        return S.apply(lambda a: 2 * a)

    check_func(test_impl, (S,))


def test_series_apply_kw():
    def test_impl(S):
        return S.apply(func=lambda a: 2 * a)

    S = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    check_func(test_impl, (S,))


@pytest.mark.parametrize(
    "S",
    [
        pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
        pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], [3, 1, 0, 2, 4], name="ABC"),
    ],
)
def test_series_map(S):
    def test_impl(S):
        return S.map(lambda a: 2 * a)

    check_func(test_impl, (S,))


def test_series_map_global1():
    def test_impl(S):
        return S.map(arg=lambda a: a + GLOBAL_VAL)

    S = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    check_func(test_impl, (S,))


def test_series_map_tup1():
    def test_impl(S):
        return S.map(lambda a: (a, 2 * a))

    S = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(bodo_func(S), test_impl(S))
    # TODO: support unbox for column of tuples
    # check_func(test_impl, (S,))


def test_series_map_tup_map1():
    def test_impl(S):
        A = S.map(lambda a: (a, 2 * a))
        return A.map(lambda a: a[1])

    S = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    check_func(test_impl, (S,))


@pytest.mark.parametrize(
    "S",
    [
        pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
        pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], [3, 1, 0, 2, 4], name="ABC"),
    ],
)
def test_series_rolling(S):
    def test_impl(S):
        return S.rolling(3).sum()

    check_func(test_impl, (S,))


def test_series_rolling_kw():
    def test_impl(S):
        return S.rolling(window=3, center=True).sum()

    S = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    check_func(test_impl, (S,))


@pytest.mark.parametrize(
    "S",
    [
        pd.Series([1.0, 2.2, 3.1, 4.6, 5.9]),
        pd.Series([1.0, 2.2, 3.1, 4.6, 5.9], [3, 1, 0, 2, 4], name="ABC"),
    ],
)
def test_series_cumsum(S):
    # TODO: datetime64, timedelta64
    # TODO: support skipna
    def test_impl(S):
        return S.cumsum()

    check_func(test_impl, (S,))


@pytest.mark.parametrize(
    "S",
    [
        pd.Series([1.0, 2.2, 3.1, 4.6, 5.9]),
        pd.Series([1.0, 2.2, 3.1, 4.6, 5.9], [3, 1, 0, 2, 4], name="ABC"),
    ],
)
def test_series_cumprod(S):
    # TODO: datetime64, timedelta64
    # TODO: support skipna
    def test_impl(S):
        return S.cumprod()

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(bodo_func(S), test_impl(S))
    # TODO: implement distributed cumprod
    # check_func(test_impl, (S,))


def test_series_rename():
    # TODO: renaming labels, etc.
    def test_impl(A):
        return A.rename("B")

    S = pd.Series([1.0, 2.0, np.nan, 1.0], name="A")
    check_func(test_impl, (S,))


def test_series_abs():
    def test_impl(S):
        return S.abs()

    S = pd.Series([np.nan, -2.0, 3.0])
    check_func(test_impl, (S,))


def test_series_min(series_val):
    # skip strings, TODO: handle strings
    if isinstance(series_val.values[0], str):
        return

    def test_impl(A):
        return A.min()

    check_func(test_impl, (series_val,))


def test_series_max(series_val):
    # skip strings, TODO: handle strings
    if isinstance(series_val.values[0], str):
        return

    def test_impl(A):
        return A.max()

    check_func(test_impl, (series_val,))


def test_series_idxmin(series_val):
    # IntegerArray doesn't have argmin yet, TODO: implement
    if isinstance(series_val.dtype, pd.core.arrays.integer._IntegerDtype):
        return

    # skip strings, TODO: handle strings
    if isinstance(series_val.values[0], str):
        return

    # argmin() not supported for bools in Pandas
    if series_val.dtype == np.bool_ or is_bool_object_series(series_val):
        return

    def test_impl(A):
        return A.idxmin()

    bodo_func = bodo.jit(test_impl)
    assert bodo_func(series_val) == test_impl(series_val)
    # TODO: support more distribtued types and test


def test_series_idxmax(series_val):
    # IntegerArray doesn't have argmin yet, TODO: implement
    if isinstance(series_val.dtype, pd.core.arrays.integer._IntegerDtype):
        return

    # skip strings, TODO: handle strings
    if isinstance(series_val.values[0], str):
        return

    # argmax() not supported for bools in Pandas
    if series_val.dtype == np.bool_ or is_bool_object_series(series_val):
        return

    def test_impl(A):
        return A.idxmax()

    bodo_func = bodo.jit(test_impl)
    assert bodo_func(series_val) == test_impl(series_val)
    # TODO: support more distribtued types and test


def test_series_median(numeric_series_val):
    # NA not supported yet, TODO: support
    if numeric_series_val.dtype == np.float:
        return

    # TODO: support nullable int
    if isinstance(numeric_series_val.dtype, pd.core.arrays.integer._IntegerDtype):
        return

    # median not supported for dt64
    if numeric_series_val.dtype == np.dtype("datetime64[ns]"):
        return

    def test_impl(A):
        return A.median()

    check_func(test_impl, (numeric_series_val,))


def test_series_head(series_val):
    def test_impl(S):
        return S.head(3)

    check_func(test_impl, (series_val,), False)


def test_series_tail(series_val):
    def test_impl(S):
        return S.tail(3)

    check_func(test_impl, (series_val,), False)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_series_nlargest(numeric_series_val, k):
    # TODO: support nullable int
    if isinstance(numeric_series_val.dtype, pd.core.arrays.integer._IntegerDtype):
        return

    def test_impl(S, k):
        return S.nlargest(k)

    check_func(test_impl, (numeric_series_val, k), False)


def test_series_nlargest_non_index():
    # test Series with None as Index
    def test_impl(k):
        S = pd.Series([3, 5, 6, 1, 9])
        return S.nlargest(k)

    bodo_func = bodo.jit(test_impl)
    k = 3
    pd.testing.assert_series_equal(bodo_func(k), test_impl(k))


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_series_nsmallest(numeric_series_val, k):
    # TODO: support nullable int
    if isinstance(numeric_series_val.dtype, pd.core.arrays.integer._IntegerDtype):
        return

    def test_impl(S, k):
        return S.nsmallest(k)

    check_func(test_impl, (numeric_series_val, k), False)


def test_series_nsmallest_non_index():
    # test Series with None as Index
    def test_impl(k):
        S = pd.Series([3, 5, 6, 1, 9])
        return S.nsmallest(k)

    bodo_func = bodo.jit(test_impl)
    k = 3
    pd.testing.assert_series_equal(bodo_func(k), test_impl(k))


def test_series_take(series_val):
    def test_impl(A):
        return A.take([2, 3])

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(
        bodo_func(series_val), test_impl(series_val), check_dtype=False
    )
    # TODO: dist support for selection with index list


def test_series_argsort(series_val):
    def test_impl(A):
        return A.argsort()

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(bodo_func(series_val), test_impl(series_val))
    # TODO: support distributed argsort()
    # check_func(test_impl, (series_val,))


def test_series_sort_values(series_val):
    # XXX can't push NAs to the end, TODO: fix
    if series_val.hasnans:
        return

    # BooleanArray can't be key in sort, TODO: handle
    if series_val.dtype == np.bool_:
        return

    def test_impl(A):
        return A.sort_values()

    check_func(test_impl, (series_val,))


@pytest.mark.parametrize("ignore_index", [True, False])
def test_series_append_single(series_val, ignore_index):

    func_text = "def test_impl(A, B):\n"
    func_text += "  return A.append(B, {})\n".format(ignore_index)
    loc_vars = {}
    exec(func_text, {"bodo": bodo}, loc_vars)
    test_impl = loc_vars["test_impl"]

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(
        bodo_func(series_val, series_val),
        test_impl(series_val, series_val),
        check_dtype=False,
        check_names=False,
    )  # XXX append can't set name yet


@pytest.mark.parametrize("ignore_index", [True, False])
def test_series_append_multi(series_val, ignore_index):
    func_text = "def test_impl(A, B, C):\n"
    func_text += "  return A.append([B, C], {})\n".format(ignore_index)
    loc_vars = {}
    exec(func_text, {"bodo": bodo}, loc_vars)
    test_impl = loc_vars["test_impl"]

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(
        bodo_func(series_val, series_val, series_val),
        test_impl(series_val, series_val, series_val),
        check_dtype=False,
        check_names=False,
    )  # XXX append can't set name yet


def test_series_quantile(numeric_series_val):
    # quantile not supported for dt64 yet, TODO: support and test
    if numeric_series_val.dtype == np.dtype("datetime64[ns]"):
        return

    def test_impl(A):
        return A.quantile(0.30)

    # TODO: needs np.testing.assert_almost_equal?
    check_func(test_impl, (numeric_series_val,))


def test_series_nunique(series_val):
    # doesn't support NAs yet, TODO: support and test
    if series_val.hasnans:
        return
    # not supported for dt64 yet, TODO: support and test
    if series_val.dtype == np.dtype("datetime64[ns]"):
        return

    # BooleanArray can't be key in shuffle, TODO: handle
    if series_val.dtype == np.bool_:
        return

    def test_impl(A):
        return A.nunique()

    check_func(test_impl, (series_val,))


def test_series_unique(series_val):
    # not supported for dt64 yet, TODO: support and test
    if series_val.dtype == np.dtype("datetime64[ns]"):
        return

    # np.testing.assert_array_equal() throws division by zero for bool arrays
    # with nans for some reason
    if series_val.dtype == np.dtype("O") and series_val.hasnans:
        return

    # BooleanArray can't be key in shuffle, TODO: handle
    if series_val.dtype == np.bool_:
        return

    def test_impl(A):
        return A.unique()

    # sorting since output order is not consistent
    check_func(test_impl, (series_val,), sort_output=True)


def test_series_describe(numeric_series_val):
    # not supported for dt64 yet, TODO: support and test
    if numeric_series_val.dtype == np.dtype("datetime64[ns]"):
        return

    def test_impl(A):
        return A.describe()

    check_func(test_impl, (numeric_series_val,), False)


@pytest.mark.parametrize(
    "S,value",
    [
        (pd.Series([1.0, 2.0, np.nan, 1.0], [3, 4, 2, 1], name="A"), 5.0),
        (pd.Series(["aa", "b", None, "ccc"], [3, 4, 2, 1], name="A"), "dd"),
    ],
)
def test_series_fillna(S, value):
    def test_impl(A, val):
        return A.fillna(val)

    check_func(test_impl, (S, value))


@pytest.mark.parametrize(
    "S",
    [
        pd.Series([1.0, 2.0, np.nan, 1.0], [3, 4, 2, 1], name="A"),
        pd.Series(["aa", "b", None, "ccc"], [3, 4, 2, 1], name="A"),
    ],
)
def test_series_dropna(S):
    def test_impl(A):
        return A.dropna()

    check_func(test_impl, (S,))


def test_series_shift(numeric_series_val):
    # not supported for dt64 yet, TODO: support and test
    if numeric_series_val.dtype == np.dtype("datetime64[ns]"):
        return

    # TODO: support nullable int
    if isinstance(numeric_series_val.dtype, pd.core.arrays.integer._IntegerDtype):
        return

    def test_impl(A):
        return A.shift(2)

    check_func(test_impl, (numeric_series_val,))


def test_series_pct_change(numeric_series_val):
    # not supported for dt64 yet, TODO: support and test
    if numeric_series_val.dtype == np.dtype("datetime64[ns]"):
        return

    # TODO: support nullable int
    if isinstance(numeric_series_val.dtype, pd.core.arrays.integer._IntegerDtype):
        return

    def test_impl(A):
        return A.pct_change(2)

    check_func(test_impl, (numeric_series_val,))


def test_series_index_cast():
    # cast None index to integer index if necessary
    def test_impl(n):
        if n < 5:
            S = pd.Series([3, 4], [2, 3])
        else:
            S = pd.Series([3, 6])
        return S

    bodo_func = bodo.jit(test_impl)
    n = 10
    pd.testing.assert_series_equal(bodo_func(n), test_impl(n))


############################### old tests ###############################


def test_create_series1():
    def test_impl():
        A = pd.Series([1, 2, 3])
        return A.values

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_array_equal(bodo_func(), test_impl())


def test_create_series_index1():
    # create and box an indexed Series
    def test_impl():
        A = pd.Series([1, 2, 3], ["A", "C", "B"])
        return A

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(bodo_func(), test_impl())


def test_create_series_index2():
    def test_impl():
        A = pd.Series([1, 2, 3], index=["A", "C", "B"])
        return A

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(bodo_func(), test_impl())


def test_create_series_index3():
    def test_impl():
        A = pd.Series([1, 2, 3], index=["A", "C", "B"], name="A")
        return A

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(bodo_func(), test_impl())


def test_create_series_index4():
    def test_impl(name):
        A = pd.Series([1, 2, 3], index=["A", "C", "B"], name=name)
        return A

    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_series_equal(bodo_func("A"), test_impl("A"))


class TestSeries(unittest.TestCase):
    def test_create1(self):
        def test_impl():
            df = pd.DataFrame({"A": [1, 2, 3]})
            return (df.A == 1).sum()

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), test_impl())

    def test_create2(self):
        def test_impl(n):
            df = pd.DataFrame({"A": np.arange(n)})
            return (df.A == 2).sum()

        n = 11
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(n), test_impl(n))

    def test_create_str(self):
        def test_impl():
            df = pd.DataFrame({"A": ["a", "b", "c"]})
            return (df.A == "a").sum()

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), test_impl())

    def test_pass_df1(self):
        def test_impl(df):
            return (df.A == 2).sum()

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df), test_impl(df))

    def test_pass_df_str(self):
        def test_impl(df):
            return (df.A == "a").sum()

        df = pd.DataFrame({"A": ["a", "b", "c"]})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df), test_impl(df))

    def test_pass_series1(self):
        # TODO: check to make sure it is series type
        def test_impl(A):
            return (A == 2).sum()

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A), test_impl(df.A))

    def test_pass_series2(self):
        # test creating dataframe from passed series
        def test_impl(A):
            df = pd.DataFrame({"A": A})
            return (df.A == 2).sum()

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A), test_impl(df.A))

    def test_pass_series_str(self):
        def test_impl(A):
            return (A == "a").sum()

        df = pd.DataFrame({"A": ["a", "b", "c"]})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A), test_impl(df.A))

    def test_pass_series_index1(self):
        def test_impl(A):
            return A

        S = pd.Series([3, 5, 6], ["a", "b", "c"], name="A")
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S))

    def test_series_attr1(self):
        def test_impl(A):
            return A.size

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A), test_impl(df.A))

    def test_series_attr2(self):
        def test_impl(A):
            return A.copy().values

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(df.A), test_impl(df.A))

    def test_series_attr3(self):
        def test_impl(A):
            return A.min()

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A), test_impl(df.A))

    def test_series_attr4(self):
        def test_impl(A):
            return A.cumsum().values

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(df.A), test_impl(df.A))

    def test_series_argsort1(self):
        def test_impl(A):
            return A.argsort()

        n = 11
        A = pd.Series(np.random.ranf(n))
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(A), test_impl(A))

    def test_series_attr6(self):
        def test_impl(A):
            return A.take([2, 3]).values

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(df.A), test_impl(df.A))

    def test_series_attr7(self):
        def test_impl(A):
            return A.astype(np.float64)

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(df.A), test_impl(df.A))

    def test_series_copy_str1(self):
        def test_impl(A):
            return A.copy()

        S = pd.Series(["aa", "bb", "cc"])
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S))

    def test_series_astype_str1(self):
        def test_impl(A):
            return A.astype(str)

        n = 11
        S = pd.Series(np.arange(n))
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S))

    def test_series_astype_str2(self):
        def test_impl(A):
            return A.astype(str)

        S = pd.Series(["aa", "bb", "cc"])
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S))

    def test_np_call_on_series1(self):
        def test_impl(A):
            return np.min(A)

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(df.A), test_impl(df.A))

    def test_series_values1(self):
        def test_impl(A):
            return (A == 2).values

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(df.A), test_impl(df.A))

    def test_series_shape1(self):
        def test_impl(A):
            return A.shape

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A), test_impl(df.A))

    def test_static_setitem_series1(self):
        def test_impl(A):
            A[0] = 2
            return (A == 2).sum()

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A), test_impl(df.A))

    def test_setitem_series1(self):
        def test_impl(A, i):
            A[i] = 2
            return (A == 2).sum()

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A.copy(), 0), test_impl(df.A.copy(), 0))

    def test_setitem_series2(self):
        def test_impl(A, i):
            A[i] = 100

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        A1 = df.A.copy()
        A2 = df.A
        bodo_func = bodo.jit(test_impl)
        bodo_func(A1, 0)
        test_impl(A2, 0)
        np.testing.assert_array_equal(A1.values, A2.values)

    @unittest.skip("enable after remove dead in hiframes is removed")
    def test_setitem_series3(self):
        def test_impl(A, i):
            S = pd.Series(A)
            S[i] = 100

        n = 11
        A = np.arange(n)
        A1 = A.copy()
        A2 = A
        bodo_func = bodo.jit(test_impl)
        bodo_func(A1, 0)
        test_impl(A2, 0)
        np.testing.assert_array_equal(A1, A2)

    def test_setitem_series_bool1(self):
        def test_impl(A):
            A[A > 3] = 100

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        A1 = df.A.copy()
        A2 = df.A
        bodo_func = bodo.jit(test_impl)
        bodo_func(A1)
        test_impl(A2)
        np.testing.assert_array_equal(A1.values, A2.values)

    def test_setitem_series_bool2(self):
        def test_impl(A, B):
            A[A > 3] = B[A > 3]

        n = 11
        df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2})
        A1 = df.A.copy()
        A2 = df.A
        bodo_func = bodo.jit(test_impl)
        bodo_func(A1, df.B)
        test_impl(A2, df.B)
        np.testing.assert_array_equal(A1.values, A2.values)

    def test_static_getitem_series1(self):
        def test_impl(A):
            return A[0]

        n = 11
        A = pd.Series(np.arange(n))
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(A), test_impl(A))

    def test_getitem_series1(self):
        def test_impl(A, i):
            return A[i]

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A, 0), test_impl(df.A, 0))

    def test_getitem_series_str1(self):
        def test_impl(A, i):
            return A[i]

        df = pd.DataFrame({"A": ["aa", "bb", "cc"]})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A, 0), test_impl(df.A, 0))

    def test_series_iat1(self):
        def test_impl(A):
            return A.iat[3]

        n = 11
        S = pd.Series(np.arange(n) ** 2)
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_iat2(self):
        def test_impl(A):
            A.iat[3] = 1
            return A

        n = 11
        S = pd.Series(np.arange(n) ** 2)
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S))

    def test_series_iloc1(self):
        def test_impl(A):
            return A.iloc[3]

        n = 11
        S = pd.Series(np.arange(n) ** 2)
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_iloc2(self):
        def test_impl(A):
            return A.iloc[3:8]

        n = 11
        S = pd.Series(np.arange(n) ** 2)
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S))

    def test_series_op1(self):
        def test_impl(A, i):
            return A + A

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(
            bodo_func(df.A, 0), test_impl(df.A, 0), check_names=False
        )

    def test_series_op2(self):
        def test_impl(A, i):
            return A + i

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(
            bodo_func(df.A, 1), test_impl(df.A, 1), check_names=False
        )

    def test_series_op3(self):
        def test_impl(A, i):
            A += i
            return A

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(
            bodo_func(df.A.copy(), 1), test_impl(df.A, 1), check_names=False
        )

    def test_series_op4(self):
        def test_impl(A):
            return A.add(A)

        n = 11
        A = pd.Series(np.arange(n))
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(A), test_impl(A))

    def test_series_op5(self):
        def test_impl(A):
            return A.pow(A)

        n = 11
        A = pd.Series(np.arange(n))
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(A), test_impl(A))

    def test_series_op6(self):
        def test_impl(A, B):
            return A.eq(B)

        n = 11
        A = pd.Series(np.arange(n))
        B = pd.Series(np.arange(n) ** 2)
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(A, B), test_impl(A, B))

    def test_series_op7(self):
        def test_impl(A):
            return -A

        n = 11
        A = pd.Series(np.arange(n))
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(A), test_impl(A))

    def test_series_inplace_binop_array(self):
        def test_impl(A, B):
            A += B
            return A

        n = 11
        A = np.arange(n) ** 2.0  # TODO: use 2 for test int casting
        B = pd.Series(np.ones(n))
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(A.copy(), B), test_impl(A, B))

    def test_series_fusion1(self):
        def test_impl(A, B):
            return A + B + 1

        n = 11
        A = pd.Series(np.arange(n))
        B = pd.Series(np.arange(n) ** 2)
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(A, B), test_impl(A, B))
        self.assertEqual(count_parfor_REPs(), 1)

    def test_series_fusion2(self):
        # make sure getting data var avoids incorrect single def assumption
        def test_impl(A, B):
            S = B + 2
            if A[0] == 0:
                S = A + 1
            return S + B

        n = 11
        A = pd.Series(np.arange(n))
        B = pd.Series(np.arange(n) ** 2)
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(A, B), test_impl(A, B))
        self.assertEqual(count_parfor_REPs(), 3)

    def test_series_len(self):
        def test_impl(A, i):
            return len(A)

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(df.A, 0), test_impl(df.A, 0))

    def test_series_box(self):
        def test_impl():
            A = pd.Series([1, 2, 3])
            return A

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(), test_impl())

    def test_series_box2(self):
        def test_impl():
            A = pd.Series(["1", "2", "3"])
            return A

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(), test_impl())

    def test_series_list_str_unbox1(self):
        def test_impl(A):
            return A.iloc[0]

        S = pd.Series([["aa", "b"], ["ccc"], []])
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(S), test_impl(S))
        # call twice to test potential refcount errors
        np.testing.assert_array_equal(bodo_func(S), test_impl(S))

    def test_np_typ_call_replace(self):
        # calltype replacement is tricky for np.typ() calls since variable
        # type can't provide calltype
        def test_impl(i):
            return np.int32(i)

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(1), test_impl(1))

    def test_series_ufunc1(self):
        def test_impl(A, i):
            return np.isinf(A).values

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(df.A, 1), test_impl(df.A, 1))

    def test_list_convert(self):
        def test_impl():
            df = pd.DataFrame(
                {
                    "one": np.array([-1, np.nan, 2.5]),
                    "two": ["foo", "bar", "baz"],
                    "three": [True, False, True],
                }
            )
            return df.one.values, df.two.values, df.three.values

        bodo_func = bodo.jit(test_impl)
        one, two, three = bodo_func()
        self.assertTrue(isinstance(one, np.ndarray))
        self.assertTrue(isinstance(two, np.ndarray))
        self.assertTrue(isinstance(three, np.ndarray))

    @unittest.skip("needs empty_like typing fix in npydecl.py")
    def test_series_empty_like(self):
        def test_impl(A):
            return np.empty_like(A)

        n = 11
        df = pd.DataFrame({"A": np.arange(n)})
        bodo_func = bodo.jit(test_impl)
        self.assertTrue(isinstance(bodo_func(df.A), np.ndarray))

    def test_series_fillna1(self):
        def test_impl(A):
            return A.fillna(5.0)

        df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 1.0]})
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(
            bodo_func(df.A), test_impl(df.A), check_names=False
        )

    def test_series_fillna_str1(self):
        def test_impl(A):
            return A.fillna("dd")

        df = pd.DataFrame({"A": ["aa", "b", None, "ccc"]})
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(
            bodo_func(df.A), test_impl(df.A), check_names=False
        )

    def test_series_fillna_str_inplace1(self):
        def test_impl(A):
            A.fillna("dd", inplace=True)
            return A

        S1 = pd.Series(["aa", "b", None, "ccc"])
        S2 = S1.copy()
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(S1), test_impl(S2))
        # TODO: handle string array reflection
        # bodo_func(S1)
        # test_impl(S2)
        # np.testing.assert_array_equal(S1, S2)

    def test_series_fillna_str_inplace_empty1(self):
        def test_impl(A):
            A.fillna("", inplace=True)
            return A

        S1 = pd.Series(["aa", "b", None, "ccc"])
        S2 = S1.copy()
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(S1), test_impl(S2))

    def test_series_dropna_float1(self):
        def test_impl(A):
            return A.dropna().values

        S1 = pd.Series([1.0, 2.0, np.nan, 1.0])
        S2 = S1.copy()
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(S1), test_impl(S2))

    def test_series_dropna_str1(self):
        def test_impl(A):
            return A.dropna().values

        S1 = pd.Series(["aa", "b", None, "ccc"])
        S2 = S1.copy()
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(S1), test_impl(S2))

    def test_series_dropna_str_parallel1(self):
        def test_impl(A):
            B = A.dropna()
            return (B == "gg").sum()

        S1 = pd.Series(["aa", "b", None, "ccc", "dd", "gg"])
        bodo_func = bodo.jit(distributed=["A"])(test_impl)
        start, end = get_start_end(len(S1))
        # TODO: gatherv
        self.assertEqual(bodo_func(S1[start:end]), test_impl(S1))

    def test_series_dropna_float_inplace1(self):
        def test_impl(A):
            A.dropna(inplace=True)
            return A.values

        S1 = pd.Series([1.0, 2.0, np.nan, 1.0])
        S2 = S1.copy()
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(S1), test_impl(S2))

    def test_series_dropna_str_inplace1(self):
        def test_impl(A):
            A.dropna(inplace=True)
            return A.values

        S1 = pd.Series(["aa", "b", None, "ccc"])
        S2 = S1.copy()
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(S1), test_impl(S2))

    def test_series_sum1(self):
        def test_impl(S):
            return S.sum()

        bodo_func = bodo.jit(test_impl)
        # column with NA
        S = pd.Series([np.nan, 2.0, 3.0])
        self.assertEqual(bodo_func(S), test_impl(S))
        # all NA case should produce 0
        S = pd.Series([np.nan, np.nan])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_sum2(self):
        def test_impl(S):
            return (S + S).sum()

        bodo_func = bodo.jit(test_impl)
        S = pd.Series([np.nan, 2.0, 3.0])
        self.assertEqual(bodo_func(S), test_impl(S))
        S = pd.Series([np.nan, np.nan])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_prod1(self):
        def test_impl(S):
            return S.prod()

        bodo_func = bodo.jit(test_impl)
        # column with NA
        S = pd.Series([np.nan, 2.0, 3.0])
        self.assertEqual(bodo_func(S), test_impl(S))
        # all NA case should produce 1
        S = pd.Series([np.nan, np.nan])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_count1(self):
        def test_impl(S):
            return S.count()

        bodo_func = bodo.jit(test_impl)
        S = pd.Series([np.nan, 2.0, 3.0])
        self.assertEqual(bodo_func(S), test_impl(S))
        S = pd.Series([np.nan, np.nan])
        self.assertEqual(bodo_func(S), test_impl(S))
        S = pd.Series(["aa", "bb", np.nan])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_mean1(self):
        def test_impl(S):
            return S.mean()

        bodo_func = bodo.jit(test_impl)
        S = pd.Series([np.nan, 2.0, 3.0])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_var1(self):
        def test_impl(S):
            return S.var()

        bodo_func = bodo.jit(test_impl)
        S = pd.Series([np.nan, 2.0, 3.0])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_std1(self):
        def test_impl(S):
            return S.std()

        bodo_func = bodo.jit(test_impl)
        S = pd.Series([np.nan, 2.0, 3.0])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_min1(self):
        def test_impl(S):
            return S.min()

        bodo_func = bodo.jit(test_impl)
        S = pd.Series([np.nan, 2.0, 3.0])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_max1(self):
        def test_impl(S):
            return S.max()

        bodo_func = bodo.jit(test_impl)
        S = pd.Series([np.nan, 2.0, 3.0])
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_value_counts(self):
        def test_impl(S):
            return S.value_counts()

        bodo_func = bodo.jit(test_impl)
        S = pd.Series(["AA", "BB", "C", "AA", "C", "AA"])
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S))

    def test_series_dist_input1(self):
        def test_impl(S):
            return S.max()

        bodo_func = bodo.jit(distributed={"S"})(test_impl)
        n = 111
        S = pd.Series(np.arange(n))
        start, end = get_start_end(n)
        self.assertEqual(bodo_func(S[start:end]), test_impl(S))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_series_tuple_input1(self):
        def test_impl(s_tup):
            return s_tup[0].max()

        bodo_func = bodo.jit(test_impl)
        n = 111
        S = pd.Series(np.arange(n))
        S2 = pd.Series(np.arange(n) + 1.0)
        s_tup = (S, 1, S2)
        self.assertEqual(bodo_func(s_tup), test_impl(s_tup))

    @unittest.skip("pending handling of build_tuple in dist pass")
    def test_series_tuple_input_dist1(self):
        def test_impl(s_tup):
            return s_tup[0].max()

        bodo_func = bodo.jit(locals={"s_tup:input": "distributed"})(test_impl)
        n = 111
        S = pd.Series(np.arange(n))
        S2 = pd.Series(np.arange(n) + 1.0)
        start, end = get_start_end(n)
        s_tup = (S, 1, S2)
        h_s_tup = (S[start:end], 1, S2[start:end])
        self.assertEqual(bodo_func(h_s_tup), test_impl(s_tup))

    def test_series_concat1(self):
        def test_impl(S1, S2):
            return pd.concat([S1, S2]).values

        bodo_func = bodo.jit(test_impl)
        S1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        S2 = pd.Series([6.0, 7.0])
        np.testing.assert_array_equal(bodo_func(S1, S2), test_impl(S1, S2))

    def test_series_concat_str1(self):
        def test_impl(S1, S2):
            return pd.concat([S1, S2])

        bodo_func = bodo.jit(test_impl)
        S1 = pd.Series(["aa", "bb", np.nan, "", "GGG"])
        S2 = pd.Series(["1", "12", "", np.nan, "A"])
        # TODO: handle index in concat
        pd.testing.assert_series_equal(
            bodo_func(S1, S2), test_impl(S1, S2).reset_index(drop=True)
        )

    def test_series_cov1(self):
        def test_impl(S1, S2):
            return S1.cov(S2)

        bodo_func = bodo.jit(test_impl)
        for pair in _cov_corr_series:
            S1, S2 = pair
            np.testing.assert_almost_equal(
                bodo_func(S1, S2),
                test_impl(S1, S2),
                err_msg="S1={}\nS2={}".format(S1, S2),
            )

    def test_series_corr1(self):
        def test_impl(S1, S2):
            return S1.corr(S2)

        bodo_func = bodo.jit(test_impl)
        for pair in _cov_corr_series:
            S1, S2 = pair
            np.testing.assert_almost_equal(
                bodo_func(S1, S2),
                test_impl(S1, S2),
                err_msg="S1={}\nS2={}".format(S1, S2),
            )

    def test_series_str_len1(self):
        def test_impl(S):
            return S.str.len()

        S = pd.Series(["aa", "abc", "c", "cccd"])
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S), check_dtype=False)

    def test_series_str2str(self):
        str2str_methods = (
            "capitalize",
            "lower",
            "lstrip",
            "rstrip",
            "strip",
            "swapcase",
            "title",
            "upper",
        )
        for method in str2str_methods:
            func_text = "def test_impl(S):\n"
            func_text += "  return S.str.{}()\n".format(method)
            loc_vars = {}
            exec(func_text, {"bodo": bodo}, loc_vars)
            test_impl = loc_vars["test_impl"]
            # XXX: \t support pending Numba #4188
            # S = pd.Series([' \tbbCD\t ', 'ABC', ' mCDm\t', 'abc'])
            S = pd.Series([" bbCD ", "ABC", " mCDm ", np.nan, "abc"])
            check_func(test_impl, (S,))

    def test_series_str2bool(self):
        str2bool_methods = (
            "isalnum",
            "isalpha",
            "isdigit",
            "isspace",
            "isupper",
            "islower",
            "istitle",
            "isnumeric",
            "isdecimal",
        )
        for method in str2bool_methods:
            func_text = "def test_impl(S):\n"
            func_text += "  return S.str.{}()\n".format(method)
            loc_vars = {}
            exec(func_text, {"bodo": bodo}, loc_vars)
            test_impl = loc_vars["test_impl"]
            S = pd.Series(
                [" 1aB ", "982", "ABC", "  ", np.nan, "abc", "Hi There", "100.20"]
            )
            check_func(test_impl, (S,))

    def test_series_append1(self):
        def test_impl(S, other):
            return S.append(other).values

        bodo_func = bodo.jit(test_impl)
        S1 = pd.Series([-2.0, 3.0, 9.1])
        S2 = pd.Series([-2.0, 5.0])
        # Test single series
        np.testing.assert_array_equal(bodo_func(S1, S2), test_impl(S1, S2))

    def test_series_append2(self):
        def test_impl(S1, S2, S3):
            return S1.append([S2, S3]).values

        bodo_func = bodo.jit(test_impl)
        S1 = pd.Series([-2.0, 3.0, 9.1])
        S2 = pd.Series([-2.0, 5.0])
        S3 = pd.Series([1.0])
        # Test series tuple
        np.testing.assert_array_equal(bodo_func(S1, S2, S3), test_impl(S1, S2, S3))

    def test_series_isna1(self):
        def test_impl(S):
            return S.isna()

        # column with NA
        S = pd.Series([np.nan, 2.0, 3.0])
        check_func(test_impl, (S,))

    def test_series_isnull1(self):
        def test_impl(S):
            return S.isnull()

        bodo_func = bodo.jit(test_impl)
        # column with NA
        S = pd.Series([np.nan, 2.0, 3.0])
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S))

    def test_series_notna1(self):
        def test_impl(S):
            return S.notna()

        bodo_func = bodo.jit(test_impl)
        # column with NA
        S = pd.Series([np.nan, 2.0, 3.0])
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S))

    def test_series_str_isna1(self):
        def test_impl(S):
            return S.isna()

        S = pd.Series(["aa", None, "c", "cccd"])
        check_func(test_impl, (S,))

    def test_series_nlargest1(self):
        def test_impl(S):
            return S.nlargest(4)

        bodo_func = bodo.jit(test_impl)
        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        np.testing.assert_array_equal(bodo_func(S).values, test_impl(S).values)

    def test_series_nlargest_default1(self):
        def test_impl(S):
            return S.nlargest()

        bodo_func = bodo.jit(test_impl)
        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        np.testing.assert_array_equal(bodo_func(S).values, test_impl(S).values)

    def test_series_nlargest_nan1(self):
        def test_impl(S):
            return S.nlargest(4)

        bodo_func = bodo.jit(test_impl)
        S = pd.Series([1.0, np.nan, 3.0, 2.0, np.nan, 4.0])
        np.testing.assert_array_equal(bodo_func(S).values, test_impl(S).values)

    def test_series_nlargest_parallel1(self):
        fname = os.path.join("bodo", "tests", "data", "kde.parquet")

        def test_impl():
            df = pq.read_table(fname).to_pandas()
            S = df.points
            return S.nlargest(4)

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func().values, test_impl().values)

    def test_series_nsmallest1(self):
        def test_impl(S):
            return S.nsmallest(4)

        bodo_func = bodo.jit(test_impl)
        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        np.testing.assert_array_equal(bodo_func(S).values, test_impl(S).values)

    def test_series_nsmallest_default1(self):
        def test_impl(S):
            return S.nsmallest()

        bodo_func = bodo.jit(test_impl)
        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        np.testing.assert_array_equal(bodo_func(S).values, test_impl(S).values)

    def test_series_nsmallest_nan1(self):
        def test_impl(S):
            return S.nsmallest(4)

        bodo_func = bodo.jit(test_impl)
        S = pd.Series([1.0, np.nan, 3.0, 2.0, np.nan, 4.0])
        np.testing.assert_array_equal(bodo_func(S).values, test_impl(S).values)

    def test_series_nsmallest_parallel1(self):
        fname = os.path.join("bodo", "tests", "data", "kde.parquet")

        def test_impl():
            df = pq.read_table(fname).to_pandas()
            S = df.points
            return S.nsmallest(4)

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func().values, test_impl().values)

    def test_series_head1(self):
        def test_impl(S):
            return S.head(4)

        bodo_func = bodo.jit(test_impl)
        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        np.testing.assert_array_equal(bodo_func(S).values, test_impl(S).values)

    def test_series_head_default1(self):
        def test_impl(S):
            return S.head()

        bodo_func = bodo.jit(test_impl)
        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        np.testing.assert_array_equal(bodo_func(S).values, test_impl(S).values)

    def test_series_head_index1(self):
        def test_impl():
            S = pd.Series([6, 9, 2, 3, 6, 4, 5], [8, 1, 6, 0, 9, 1, 3])
            return S.head(3)

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(), test_impl())

    def test_series_head_index2(self):
        def test_impl():
            S = pd.Series([6, 9, 2, 3, 6, 4, 5], ["a", "ab", "abc", "c", "f", "hh", ""])
            return S.head(3)

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(), test_impl())

    def test_series_head_index_parallel1(self):
        def test_impl(S):
            return S.head(3)

        S = pd.Series([6, 9, 2, 3, 6, 4, 5], ["a", "ab", "abc", "c", "f", "hh", ""])
        bodo_func = bodo.jit(distributed={"S"})(test_impl)
        start, end = get_start_end(len(S))
        pd.testing.assert_series_equal(bodo_func(S[start:end]), test_impl(S))
        self.assertTrue(count_array_OneDs() > 0)

    def test_series_median1(self):
        def test_impl(S):
            return S.median()

        bodo_func = bodo.jit(test_impl)
        m = 100
        np.random.seed(0)
        S = pd.Series(np.random.randint(-30, 30, m))
        self.assertEqual(bodo_func(S), test_impl(S))
        S = pd.Series(np.random.ranf(m))
        self.assertEqual(bodo_func(S), test_impl(S))
        # odd size
        m = 101
        S = pd.Series(np.random.randint(-30, 30, m))
        self.assertEqual(bodo_func(S), test_impl(S))
        S = pd.Series(np.random.ranf(m))
        self.assertEqual(bodo_func(S), test_impl(S))

    def test_series_median_parallel1(self):
        fname = os.path.join("bodo", "tests", "data", "kde.parquet")

        def test_impl():
            df = pq.read_table(fname).to_pandas()
            S = df.points
            return S.median()

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), test_impl())

    def test_series_argsort_parallel(self):
        fname = os.path.join("bodo", "tests", "data", "kde.parquet")

        def test_impl():
            df = pq.read_table(fname).to_pandas()
            S = df.points
            return S.argsort().values

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(), test_impl())

    def test_series_idxmin1(self):
        def test_impl(A):
            return A.idxmin()

        n = 11
        np.random.seed(0)
        S = pd.Series(np.random.ranf(n))
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(S), test_impl(S))

    def test_series_idxmax1(self):
        def test_impl(A):
            return A.idxmax()

        n = 11
        np.random.seed(0)
        S = pd.Series(np.random.ranf(n))
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(S), test_impl(S))

    def test_series_sort_values1(self):
        def test_impl(A):
            return A.sort_values()

        n = 11
        np.random.seed(0)
        S = pd.Series(np.random.ranf(n))
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S))

    def test_series_sort_values_index1(self):
        def test_impl(A, B):
            S = pd.Series(A, B)
            return S.sort_values()

        n = 11
        np.random.seed(0)
        # TODO: support passing Series with Index
        # S = pd.Series(np.random.ranf(n), np.random.randint(0, 100, n))
        A = np.random.ranf(n)
        B = np.random.ranf(n)
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(A, B), test_impl(A, B))

    def test_series_sort_values_parallel1(self):
        fname = os.path.join("bodo", "tests", "data", "kde.parquet")

        def test_impl():
            df = pq.read_table(fname).to_pandas()
            S = df.points
            return S.sort_values()

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(), test_impl())

    def test_series_shift_default1(self):
        def test_impl(S):
            return S.shift()

        bodo_func = bodo.jit(test_impl)
        S = pd.Series([np.nan, 2.0, 3.0, 5.0, np.nan, 6.0, 7.0])
        pd.testing.assert_series_equal(bodo_func(S), test_impl(S))


if __name__ == "__main__":
    unittest.main()
