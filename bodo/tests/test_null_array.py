"""
Tests for the null array type, which is an array of all nulls
that can be cast to any type. See null_arr_ext.py for the
core implementation.
"""

import datetime
from decimal import Decimal

import numba  # noqa TID253
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
from bodo.tests.utils import check_func


def test_nullable_bool_cast(memory_leak_check):
    """
    Tests casting a nullable array to a boolean array.
    """

    def impl(n):
        null_arr = bodo.libs.null_arr_ext.init_null_array(n)
        return null_arr.astype(pd.BooleanDtype())

    n = 10
    arr = pd.array([None] * n, dtype=pd.BooleanDtype())
    check_func(impl, [n], py_output=arr)


def test_null_arr_getitem(memory_leak_check):
    """Test getitem with nullable arrays

    Args:
        memory_leak_check: A context manager fixture that makes sure there is no memory leak in the test.
    """

    def impl(n, idx):
        null_arr = bodo.libs.null_arr_ext.init_null_array(n)
        return [null_arr[idx]]

    n = 10
    np.random.seed(0)

    # A single integer
    idx = 0
    check_func(
        impl,
        (n, idx),
        py_output=[None],
        check_dtype=False,
        dist_test=False,
    )

    def impl2(n, idx):
        null_arr = bodo.libs.null_arr_ext.init_null_array(n)
        return null_arr[idx]

    # Array of integers
    idx = np.random.randint(0, n, 11)
    check_func(
        impl2,
        (n, idx),
        py_output=pd.array([None] * len(idx), dtype=pd.ArrowDtype(pa.null())),
        check_dtype=False,
        dist_test=False,
    )

    # Array of booleans
    idx = [True, False, True, False, False]
    arr = pd.array([None] * len(idx))
    expected_output = arr[idx]
    check_func(
        impl2,
        (n, idx),
        py_output=expected_output,
        check_dtype=False,
        dist_test=False,
    )

    # slice
    idx = slice(5)
    check_func(
        impl2,
        (n, idx),
        py_output=pd.array([None] * 5, dtype=pd.ArrowDtype(pa.null())),
        check_dtype=False,
    )


def test_isna_check(memory_leak_check):
    """
    Test that isna works properly with NullArrayType
    """

    def test_impl(n):
        null_arr = bodo.libs.null_arr_ext.init_null_array(n)
        return pd.isna(null_arr)

    n = 10
    py_output = pd.array([True] * n)
    check_func(test_impl, (n,), py_output=py_output)


@pytest.mark.slow
def test_astype_check(memory_leak_check):
    """
    Test that astype works properly with NullArrayType
    """

    def test_impl(n):
        null_arr = bodo.libs.null_arr_ext.init_null_array(n)
        S = pd.Series(null_arr)
        return S.astype(np.dtype("datetime64[ns]"))

    n = 10
    py_output = pd.Series([None] * n, dtype=pd.ArrowDtype(pa.timestamp("ns")))
    check_func(test_impl, (n,), py_output=py_output)


def test_nullarray_cast():
    """Test casting arrays to null array, which is necessary for BodoSQL"""

    def impl(A, flag):
        n = len(A)
        if flag:
            B = A
        else:
            B = bodo.libs.null_arr_ext.init_null_array(n)
        return B

    n = 5
    for t in bodo.libs.array_kernels.BODO_ARRAY_TYPE_CLASSES:
        py_output = np.array([None] * n, object)
        if t == numba.core.types.Array:
            # Numpy arrays are tested below
            continue
        elif t == bodo.libs.str_arr_ext.StringArrayType:
            A = pd.array(["A"] * n, "string[pyarrow]")
        elif t == bodo.libs.binary_arr_ext.BinaryArrayType:
            A = np.array([b"A"] * n, dtype=object)
        elif t == bodo.hiframes.datetime_date_ext.DatetimeDateArrayType:
            A = np.array([datetime.date(2011, 8, 9)] * n, object)
        elif t == bodo.hiframes.datetime_timedelta_ext.TimeDeltaArrayType:
            A = pd.array([pd.Timedelta(33)] * n)
        elif t == bodo.libs.bool_arr_ext.BooleanArrayType:
            A = pd.array([True] * n, "boolean")
        elif t == bodo.libs.int_arr_ext.IntegerArrayType:
            A = pd.array(np.arange(n), "Int32")
        elif t == bodo.libs.float_arr_ext.FloatingArrayType:
            A = pd.array(np.arange(n) + 1.1, "Float64")
        elif t == bodo.libs.decimal_arr_ext.DecimalArrayType:
            A = np.array([Decimal("32.1")] * n, object)
        elif t == bodo.libs.array_item_arr_ext.ArrayItemArrayType:
            A = pd.array([[1]] * n, dtype=pd.ArrowDtype(pa.large_list(pa.int32())))
        elif t == bodo.libs.struct_arr_ext.StructArrayType:
            A = pd.array(
                [{"A": 1, "B": 2}] * n,
                pd.ArrowDtype(
                    pa.struct([pa.field("A", pa.int64()), pa.field("B", pa.int32())])
                ),
            )
        elif t == bodo.libs.map_arr_ext.MapArrayType:
            A = pd.array(
                [{"A": 1, "B": 2}] * n,
                pd.ArrowDtype(pa.map_(pa.large_string(), pa.int64())),
            )
        elif t == bodo.types.DatetimeArrayType:
            A = pd.array([pd.Timestamp("2000-01-01", tz="UTC")] * 5)
            py_output = pd.array([pd.NaT] * n, pd.DatetimeTZDtype(tz="UTC"))
        elif t == bodo.types.TimeArrayType:
            A = np.array([bodo.types.Time(12, 0, precision=6)] * n, object)
        elif t == bodo.hiframes.timestamptz_ext.TimestampTZArrayType:
            A = np.array(
                [bodo.types.TimestampTZ.fromUTC("2021-01-02 03:04:05", 100)] * n, object
            )
        else:
            # Ignore array types that are not used in BodoSQL
            assert t in (
                bodo.libs.primitive_arr_ext.PrimitiveArrayType,
                bodo.hiframes.pd_categorical_ext.CategoricalArrayType,
                bodo.libs.interval_arr_ext.IntervalArrayType,
                bodo.hiframes.split_impl.StringArraySplitViewType,
                bodo.libs.tuple_arr_ext.TupleArrayType,
                bodo.libs.str_ext.RandomAccessStringArrayType,
                # Dictionary type is tested with regular string input automatically
                bodo.libs.dict_arr_ext.DictionaryArrayType,
                bodo.libs.null_arr_ext.NullArrayType,
                bodo.libs.matrix_ext.MatrixType,
                bodo.libs.csr_matrix_ext.CSRMatrixType,
            ), f"test_nullarray_cast: unsupported array type {t}"

        check_func(impl, (A, False), py_output=py_output, only_seq=True)
        check_func(impl, (A, True), py_output=A, only_seq=True)

    # Numpy integer input (needs Numpy to nullable cast)
    A = np.arange(n)
    check_func(impl, (A, False), py_output=py_output, only_seq=True)
    check_func(impl, (A, True), py_output=A, only_seq=True)

    # Numpy float input (needs Numpy to nullable cast)
    A = np.arange(n) + 1.1
    check_func(impl, (A, False), py_output=py_output, only_seq=True)
    check_func(impl, (A, True), py_output=A, only_seq=True)

    # Numpy bool input (needs Numpy to nullable cast)
    A = np.arange(n) > n // 2
    check_func(impl, (A, False), py_output=py_output, only_seq=True)
    check_func(impl, (A, True), py_output=A, only_seq=True)


def test_nullarray_invalid_cast():
    """Make sure invalid null array cast raises a proper error"""

    def impl(A, flag):
        if flag:
            B = A
        else:
            B = bodo.libs.null_arr_ext.init_null_array(3)
        return B

    with pytest.raises(bodo.utils.typing.BodoError, match="Unable to unify"):
        bodo.jit(impl)(3, True)


def test_nullable_decimal_cast(memory_leak_check):
    """
    Tests casting a nullable array to a decimal array.
    """
    dtype = bodo.types.Decimal128Type(30, 10)

    def impl(n):
        null_arr = bodo.libs.null_arr_ext.init_null_array(n)
        return null_arr.astype(dtype)

    n = 10
    arr = np.array([None] * n, dtype=object)
    check_func(impl, [n], py_output=arr)
