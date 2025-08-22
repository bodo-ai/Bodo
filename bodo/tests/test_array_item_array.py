"""Tests for array of list of fixed size items."""

import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
from bodo.tests.utils import check_func, pytest_mark_one_rank


@pytest.fixture(
    params=[
        pd.arrays.ArrowExtensionArray(
            pa.array(
                [[1, 3, None], [2], None, [4, None, 5, 6], [], [1, 1], None] * 2,
                pa.large_list(pa.int64()),
            )
        ),
        pytest.param(
            pd.arrays.ArrowExtensionArray(
                pa.array(
                    [[2.0, -3.2], [2.2, 1.3], None, [4.1, np.nan, 6.3], [], [1.1, 1.2]]
                    * 2,
                    pa.large_list(pa.float64()),
                )
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.arrays.ArrowExtensionArray(
                pa.array(
                    [
                        [True, False, None],
                        [False, False],
                        None,
                        [True, False, None] * 4,
                        [],
                        [True, True],
                    ]
                    * 2,
                    pa.large_list(pa.bool_()),
                )
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.arrays.ArrowExtensionArray(
                pa.array(
                    [
                        [datetime.date(2018, 1, 24), datetime.date(1983, 1, 3)],
                        [datetime.date(1966, 4, 27), datetime.date(1999, 12, 7)],
                        None,
                        [datetime.date(1966, 4, 27), datetime.date(2004, 7, 8)],
                        [],
                        [datetime.date(2020, 11, 17)],
                    ]
                    * 2,
                    pa.large_list(pa.date32()),
                )
            ),
            marks=pytest.mark.slow,
        ),
        # data from Spark-generated Parquet files can have array elements
        pytest.param(
            pd.arrays.ArrowExtensionArray(
                pa.array(
                    [
                        np.array([1, 3], np.int32),
                        np.array([2], np.int32),
                        None,
                        np.array([4, 5, 6], np.int32),
                        np.array([], np.int32),
                        np.array([1, 1], np.int32),
                    ]
                    * 2,
                    pa.large_list(pa.int32()),
                )
            ),
            marks=pytest.mark.slow,
        ),
        # TODO: enable Decimal test when memory leaks and test equality issues are fixed
        # np.array(
        #     [
        #         [Decimal("1.6"), Decimal("-0.222")],
        #         [Decimal("1111.316"), Decimal("1234.00046"), Decimal("5.1")],
        #         None,
        #         [Decimal("-11131.0056"), Decimal("0.0")],
        #         [],
        #         [Decimal("-11.00511")],
        #     ]
        # ),
        # nested list case with NA elems
        pytest.param(
            pd.arrays.ArrowExtensionArray(
                pa.array(
                    [
                        [[1, 3], [2]],
                        [[3, 1]],
                        None,
                        [[4, 5, 6], [1], [1, 2]],
                        [],
                        [[1], None, [1, 4], []],
                    ]
                    * 2,
                    pa.large_list(pa.large_list(pa.int64())),
                )
            ),
            marks=pytest.mark.slow,
        ),
        # string data with NA
        pytest.param(
            pd.arrays.ArrowExtensionArray(
                pa.array(
                    [[["1", "2", "8"], ["3"]], [["2", None]]] * 4,
                    pa.large_list(pa.large_list(pa.string())),
                )
            ),
            marks=pytest.mark.slow,
        ),
        # two level nesting
        pytest.param(
            pd.arrays.ArrowExtensionArray(
                pa.array(
                    [
                        [[[1, 2], [3]], [[2, None]]],
                        [[[3], [], [1, None, 4]]],
                        None,
                        [[[4, 5, 6], []], [[1]], [[1, 2]]],
                        [],
                        [[[], [1]], None, [[1, 4]], []],
                    ]
                    * 2,
                    pa.large_list(pa.large_list(pa.large_list(pa.int64()))),
                )
            ),
            marks=pytest.mark.slow,
        ),
        # struct data
        pytest.param(
            np.array(
                [
                    [{"A": 1, "B": 2}, {"A": 10, "B": 20}],
                    [{"A": 3, "B": 4}],
                    [{"A": 5, "B": 6}, {"A": 50, "B": 60}, {"A": 500, "B": 600}],
                    [{"A": 10, "B": 20}, {"A": 100, "B": 200}],
                    [{"A": 30, "B": 40}],
                    [{"A": 50, "B": 60}, {"A": 500, "B": 600}, {"A": 5000, "B": 6000}],
                ],
                dtype=object,
            ),
            marks=pytest.mark.slow,
        ),
    ]
)
def array_item_arr_value(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            np.array(
                [[1, 3, None], [2.2], None, ["bodo"], [1, 1], None] * 2, dtype=object
            ),
            marks=pytest.mark.skip("[BE-57]"),
        )
    ]
)
def bad_array_item_arr_value(request):
    return request.param


def test_bad_unbox(bad_array_item_arr_value, memory_leak_check):
    # just unbox
    def impl(arr_arg):
        return True

    # TODO(Nick): Capture this as an error when the segfault is avoided.
    check_func(impl, (bad_array_item_arr_value,))


@pytest.mark.slow
def test_unbox(array_item_arr_value, memory_leak_check):
    # just unbox
    def impl(arr_arg):
        return True

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl, (array_item_arr_value,))
    check_func(impl2, (array_item_arr_value,))


def test_unbox_dict_str(memory_leak_check):
    """Test boxing/unboxing array(array) with dict-encoded data (see [BSE-1155])"""

    def impl(arr_arg):
        return arr_arg

    A1 = np.array([["a1", None, "a2"], None, ["a3"]], object)
    A2 = np.array([[["1", "2", "8"], ["3"]], [["2", None]]] * 4, dtype=object)
    check_func(impl, (A1,), use_dict_encoded_strings=True, only_seq=True)
    check_func(impl, (A2,), use_dict_encoded_strings=True, only_seq=True)


@pytest.mark.smoke
def test_getitem_int(array_item_arr_value, memory_leak_check):
    def test_impl(A, i):
        return A[i]

    i = 1
    bodo_out = bodo.jit(test_impl)(array_item_arr_value, i)
    py_out = test_impl(array_item_arr_value, i)
    # cannot compare nested cases properly yet since comparison functions fail (TODO)
    if isinstance(
        bodo.typeof(bodo_out), bodo.libs.array_item_arr_ext.ArrayItemArrayType
    ):
        return
    pd.testing.assert_series_equal(
        pd.Series(bodo_out), pd.Series(py_out), check_dtype=False
    )


def test_getitem_bool(array_item_arr_value, memory_leak_check):
    def test_impl(A, ind):
        return A[ind]

    np.random.seed(1)
    ind = np.random.ranf(len(array_item_arr_value)) < 0.2
    bodo_out = bodo.jit(test_impl)(array_item_arr_value, ind)
    py_out = test_impl(array_item_arr_value, ind)
    pd.testing.assert_series_equal(
        pd.Series(py_out), pd.Series(bodo_out), check_dtype=False
    )


def test_getitem_slice(array_item_arr_value, memory_leak_check):
    def test_impl(A, ind):
        return A[ind]

    ind = slice(1, 4)
    bodo_out = bodo.jit(test_impl)(array_item_arr_value, ind)
    py_out = test_impl(array_item_arr_value, ind)
    pd.testing.assert_series_equal(
        pd.Series(py_out), pd.Series(bodo_out), check_dtype=False
    )


@pytest.mark.slow
def test_ndim(memory_leak_check):
    def test_impl(A):
        return A.ndim

    A = np.array([[1, 2, 3], [2]], object)
    assert bodo.jit(test_impl)(A) == test_impl(A)


@pytest.mark.slow
def test_shape(memory_leak_check):
    def test_impl(A):
        return A.shape

    A = np.array([[1, 2, 3], [2], None, []], object)
    assert bodo.jit(test_impl)(A) == test_impl(A)


@pytest.mark.slow
def test_dtype(memory_leak_check):
    def test_impl(A):
        return A.dtype

    A = np.array([[1, 2, 3], [2], None, []], object)
    assert bodo.jit(test_impl)(A) == test_impl(A)


@pytest.mark.slow
def test_copy(array_item_arr_value, memory_leak_check):
    def test_impl(A):
        return A.copy()

    check_func(test_impl, (array_item_arr_value,))


@pytest.mark.skipif(
    bodo.tests.utils.test_spawn_mode_enabled,
    reason="Spawn workers don't set _use_dict_str_type",
)
def test_nested_arr_dict_getitem(memory_leak_check):
    """Make sure dictionary-encoded arrays inside nested arrays support allocation and
    setitem that are necessary for operations like array(item) bool getitem
    """

    def test_impl(A, ind):
        return A[ind]

    # Struct array
    A1 = pd.arrays.ArrowExtensionArray(
        pa.array(
            [[{"A": "a1", "B": 2}], [{"A": "a1", "B": 2}, {"A": "a2", "B": 2}]],
            pa.large_list(
                pa.struct([pa.field("A", pa.large_string()), pa.field("B", pa.int64())])
            ),
        )
    )
    # Map array
    A2 = pd.arrays.ArrowExtensionArray(
        pa.array(
            [
                [{1: "abc", 4: "h"}],
                [
                    {1: "aa", 3: "m"},
                    {1: "abc", 4: "abc"},
                ],
            ],
            pa.large_list(pa.map_(pa.int64(), pa.large_string())),
        )
    )
    B = np.array([False, True], np.bool_)

    orig_use_dict_str_type = bodo.hiframes.boxing._use_dict_str_type
    bodo.hiframes.boxing._use_dict_str_type = True
    try:
        check_func(test_impl, (A1, B), only_seq=True)
        check_func(test_impl, (A2, B), only_seq=True)
    finally:
        bodo.hiframes.boxing._use_dict_str_type = orig_use_dict_str_type


@pytest_mark_one_rank
def test_nested_arr_pyarrow_typeof():
    """
    Test that we are properly typing nested arrays containing pyarrow types
    """

    precision, scale = 38, 2

    A = pd.array([Decimal("0.3")], dtype=pd.ArrowDtype(pa.decimal128(precision, scale)))
    S = pd.Series([A])
    typ = bodo.typeof(S)

    assert (
        isinstance(typ.dtype, bodo.types.DecimalArrayType)
        and typ.dtype.precision == precision
        and typ.dtype.scale == scale
    )
