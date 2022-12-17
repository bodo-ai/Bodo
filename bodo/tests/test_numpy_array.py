import datetime
from decimal import Decimal
from enum import IntEnum

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func
from bodo.utils.typing import BodoError


@pytest.fixture(
    params=[
        pytest.param(
            (
                np.array(
                    [
                        True,
                        False,
                        True,
                        True,
                        False,
                        False,
                        True,
                    ]
                    * 8
                ),
                np.array(
                    [
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ]
                    * 2
                ),
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                np.array(
                    [1.121, 0.0, 35.13431, -2414.4242, 23211.22] * 8, dtype=np.float32
                ),
                np.array(
                    [
                        1.121,
                        0.0,
                        35.13431,
                        -2414.4242,
                        23211.22,
                        1.111,
                        232.2,
                        0.0,
                        232.2,
                    ]
                    * 2,
                    dtype=np.float32,
                ),
            ),
        ),
        pytest.param(
            (
                np.array(
                    [1.121, 0.0, 35.13431, -2414.4242, 23211.22] * 8, dtype=np.float64
                ),
                np.array(
                    [
                        1.121,
                        0.0,
                        35.13431,
                        -2414.4242,
                        23211.22,
                        1.111,
                        232.2,
                        0.0,
                        232.2,
                    ]
                    * 2,
                    dtype=np.float32,
                ),
            ),
            marks=pytest.mark.skip("No support for differing bitwidths"),
        ),
        pytest.param(
            (
                np.array(
                    [
                        3,
                        5,
                        123,
                        24,
                        42,
                        24,
                        123,
                        254,
                    ]
                    * 8,
                    dtype=np.int32,
                ),
                np.array(
                    [
                        -3,
                        -5,
                        -123,
                        24,
                        -42,
                        24,
                        -123,
                        1254,
                    ]
                    * 2,
                    dtype=np.int32,
                ),
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                np.array(
                    [
                        3,
                        5,
                        123,
                        24,
                        42,
                        24,
                        123,
                        254,
                    ]
                    * 8,
                    dtype=np.uint8,
                ),
                np.array(
                    [
                        -3,
                        -5,
                        -123,
                        24,
                        -42,
                        24,
                        -123,
                        1254,
                    ]
                    * 2,
                    dtype=np.int16,
                ),
            ),
            marks=pytest.mark.skip("No support for differing bitwidths"),
        ),
        pytest.param(
            (
                np.array(
                    [
                        "True",
                        "False",
                        "go",
                        "bears",
                        "u",
                        "who",
                        "power",
                        "trip",
                    ]
                    * 8
                ),
                np.array(
                    [
                        "hi",
                        "go",
                        "to",
                        "you",
                        "who",
                        "hi",
                        "u",
                        "power",
                    ]
                    * 2
                ),
            ),
            marks=pytest.mark.skip("No support for unichr in our unique"),
        ),
        pytest.param(
            (
                np.array(
                    [
                        "True",
                        "False",
                        "go",
                        "bears",
                        "u",
                        "who",
                        "power",
                        "trip",
                    ]
                    * 8
                ),
                np.array(
                    [
                        "hi",
                        "go",
                        "to",
                        "you",
                        "who",
                        "hi",
                        "u",
                    ]
                    * 2
                ),
            ),
            marks=pytest.mark.skip("No support for differing bitwidths"),
        ),
        pytest.param(
            (
                pd.array(
                    [
                        True,
                        False,
                        True,
                        True,
                        False,
                        False,
                        True,
                    ]
                    * 8
                ),
                pd.array(
                    [
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ]
                    * 2
                ),
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                np.append(
                    pd.date_range("2017-07-03", "2017-07-17").date,
                    [datetime.date(2016, 3, 3)],
                ),
                np.append(
                    pd.date_range("2017-07-15", "2017-09-02").date,
                    [datetime.date(2018, 6, 7)],
                ),
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                np.array(
                    [
                        datetime.timedelta(days=5, seconds=4, weeks=4),
                        datetime.timedelta(days=5, seconds=5, weeks=4),
                        datetime.timedelta(days=11, seconds=4, weeks=4),
                        datetime.timedelta(days=5, seconds=4, weeks=4),
                        datetime.timedelta(days=5, seconds=4, weeks=4),
                        datetime.timedelta(days=5, seconds=4, weeks=4),
                        datetime.timedelta(days=5, seconds=64, weeks=4),
                        datetime.timedelta(days=11, seconds=4, weeks=4),
                        datetime.timedelta(days=42, seconds=11, weeks=4),
                        datetime.timedelta(days=5, seconds=123, weeks=4),
                    ]
                ),
                np.array(
                    [
                        datetime.timedelta(days=5, seconds=4, weeks=4),
                        datetime.timedelta(days=5, seconds=5, weeks=4),
                        datetime.timedelta(days=11, seconds=4, weeks=4),
                        datetime.timedelta(days=151, seconds=64, weeks=4),
                        datetime.timedelta(days=11, seconds=4, weeks=4),
                        datetime.timedelta(days=42, seconds=11, weeks=4),
                        datetime.timedelta(days=5, seconds=123, weeks=123),
                    ]
                ),
            ),
            marks=pytest.mark.skip(
                "TODO(Nick): Add support for timedelta arrays inside C++ code"
            ),
        ),
        pytest.param(
            (
                np.array(
                    [
                        Decimal("1.6"),
                        Decimal("-0.222"),
                        Decimal("1111.316"),
                        Decimal("1234.00046"),
                        Decimal("5.1"),
                        Decimal("-11131.0056"),
                        Decimal("0.0"),
                    ]
                ),
                np.array(
                    [
                        Decimal("0.0"),
                        Decimal("-0.222"),
                        Decimal("1111.316"),
                        Decimal("1"),
                        Decimal("5.1"),
                        Decimal("-1"),
                        Decimal("-1"),
                    ]
                ),
            ),
            marks=pytest.mark.skip("Issue with eq operator (fails intersect)"),
        ),
        pytest.param(
            (
                pd.arrays.IntegerArray(
                    np.array([1, -3, 2, 3, 10] * 10, np.int8),
                    np.array([False, False, False, False, False] * 10),
                ),
                pd.arrays.IntegerArray(
                    np.array([-3, -3, 20, 3, 5] * 6, np.int8),
                    np.array([False, False, False, False, False] * 6),
                ),
            ),
        ),
        pytest.param(
            (
                pd.arrays.IntegerArray(
                    np.array([1, -3, 2, 3, 10] * 10, np.int8),
                    np.array([False, False, False, False, False] * 10),
                ),
                pd.arrays.IntegerArray(
                    np.array([-3, -3, 20, 3, 5] * 6, np.int16),
                    np.array([False, False, False, False, False] * 6),
                ),
            ),
            marks=pytest.mark.skip("No support for differing bitwidths"),
        ),
        pytest.param(
            (
                pd.array(
                    [
                        "¿abc¡Y tú, quién te crees?",
                        "ÕÕÕú¡úú,úũ¿ééé",
                        "россия очень, холодная страна",
                        "مرحبا, العالم ، هذا هو بودو",
                        "Γειά σου ,Κόσμε",
                        "Español es agra,dable escuchar",
                        "한국,가,고싶다ㅠ",
                        "🢇🄐,🏈𠆶💑😅",
                    ],
                ),
                pd.array(
                    [
                        "Γειά σου ,Κόσμε",
                        "Español es agra,dable escuchar",
                        "한국,가,고싶다ㅠ",
                        "🢇🄐,🏈𠆶💑😅",
                        "isspace",
                        "islower",
                        "isupper",
                        "istitle",
                        "isnumeric",
                        "isdecimal",
                        "🢇🄐,🏈𠆶💑😅",
                        "isspace",
                        "islower",
                        "isupper",
                    ],
                ),
            ),
            marks=pytest.mark.skip("Issue with eq operator (fails intersect)"),
        ),
    ]
)
def arr_tuple_val(request):
    return request.param


@pytest.mark.slow
def test_np_union1d(arr_tuple_val, memory_leak_check):
    def impl(A1, A2):
        return np.union1d(A1, A2)

    # Keep as array tuple to allow for scaling to larger array tuple
    # sizes if needed for other Numpy functions.
    A1 = arr_tuple_val[0]
    A2 = arr_tuple_val[1]

    check_func(impl, (A1, A2))


@pytest.mark.slow
def test_np_intersect1d(arr_tuple_val, memory_leak_check):
    def impl(A1, A2):
        return np.intersect1d(A1, A2)

    # Keep as array tuple to allow for scaling to larger array tuple
    # sizes if needed for other Numpy functions.
    A1 = arr_tuple_val[0]
    A2 = arr_tuple_val[1]

    # TODO(Nick): Add parallel test when there is parallel support.
    check_func(impl, (A1, A2), dist_test=False)


@pytest.mark.slow
def test_np_setdiff1d(arr_tuple_val, memory_leak_check):
    def impl(A1, A2):
        return np.setdiff1d(A1, A2)

    # Keep as array tuple to allow for scaling to larger array tuple
    # sizes if needed for other Numpy functions.
    A1 = arr_tuple_val[0]
    A2 = arr_tuple_val[1]

    # TODO(Nick): Add parallel test when there is parallel support.
    check_func(impl, (A1, A2), dist_test=False)


@pytest.mark.slow
def test_np_hstack_list(arr_tuple_val, memory_leak_check):
    def impl(A1, A2):
        # Sort values because np.hstack order won't match.
        # This uses Series because types.Array don't have a parallel
        # implementation of np.sort
        return pd.Series(data=np.hstack([A1, A2])).sort_values().values

    check_func(impl, (*arr_tuple_val,))


@pytest.mark.slow
def test_np_hstack_tuple(arr_tuple_val, memory_leak_check):
    def impl(A1, A2):
        # Sort values because np.hstack order won't match.
        # This uses Series because types.Array don't have a parallel
        # implementation of np.sort
        return pd.Series(data=np.hstack((A1, A2))).sort_values().values

    check_func(impl, (*arr_tuple_val,))


@pytest.mark.slow
def test_np_hstack_tuple_heterogenous(memory_leak_check):
    """Test to merge float and int arrays. These can legally merge, and as
    a result should pass the type checking.
    """

    def impl(A1, A2):
        # Sort values because np.hstack order won't match.
        # This uses Series because types.Array don't have a parallel
        # implementation of np.sort
        return pd.Series(data=np.hstack((A1, A2))).sort_values().values

    A1 = np.array(
        [
            3,
            5,
            123,
            24,
            42,
            24,
            123,
            254,
        ]
        * 8,
        dtype=np.int32,
    )
    A2 = np.array([1.121, 0.0, 35.13431, -2414.4242, 23211.22] * 8, dtype=np.float64)

    check_func(impl, (A1, A2))


def test_np_linspace(memory_leak_check):
    def test_impl(start, stop, num):
        return np.linspace(start, stop, num=num)

    check_func(test_impl, (0, 1000, 100000))
    check_func(test_impl, (-2000, -4000, 100000))
    check_func(test_impl, (-5, 4.5252, 1000))


@pytest.mark.slow
def test_np_linspace_int(memory_leak_check):
    def test_impl(start, stop, num, dtype):
        return np.linspace(start, stop, num=num, dtype=dtype)

    check_func(test_impl, (0, 1000, 100000, np.int32))
    check_func(test_impl, (-2000, -4000, 100000, np.int32))
    check_func(test_impl, (-5, 4.5252, 1000, np.int32))


@pytest.mark.slow
def test_np_linspace_float(memory_leak_check):
    def test_impl(start, stop, num, dtype):
        return np.linspace(start, stop, num=num, dtype=dtype)

    check_func(test_impl, (0, 1000, 100000, np.float32))
    check_func(test_impl, (-2000, -4000, 100000, np.float32))
    check_func(test_impl, (-5, 4.5252, 1000, np.float32))


@pytest.mark.slow
def test_np_linspace_kwargs(memory_leak_check):
    def test_impl(start, stop, num, dtype, endpoint):
        return np.linspace(
            start, stop, num=num, dtype=dtype, endpoint=endpoint, retstep=False, axis=0
        )

    check_func(test_impl, (0, 1000, 100000, np.int64, False))
    check_func(test_impl, (-2000, -4000, 100000, np.int64, False))
    check_func(test_impl, (-5, 4.5252, 100000, np.int64, False))


@pytest.mark.slow
@pytest.mark.parametrize(
    "num_arr",
    [
        pd.arrays.IntegerArray(
            np.array([1, -3, 2, 3, 10] * 10, np.int64),
            np.array([False, False, False, False, True] * 10),
        ),
        np.array(
            [
                -3,
                -5,
                -123,
                24,
                -42,
                24,
                -123,
                1254,
            ]
            * 2,
            dtype=np.int64,
        ),
        np.array(
            [
                1.121,
                0.0,
                35.13431,
                -2414.4242,
                23211.22,
                1.111,
                232.2,
                0.0,
                232.2,
            ]
            * 2,
            dtype=np.float32,
        ),
    ],
)
def test_cbrt(num_arr):
    def test_impl(A):
        return np.cbrt(A)

    # Numpy uses different floating point libaries in
    # different platforms so precision may vary. This
    # should be fixed when numba adds support for it.
    check_func(test_impl, (num_arr,), atol=2e-06, rtol=2e-07)


@pytest.fixture(
    params=[
        pd.array(
            [
                "¿abc¡Y tú, quién te crees?",
                "ÕÕÕú¡úú,úũ¿ééé",
                "россия очень, холодная страна",
                "مرحبا, العالم ، هذا هو بودو",
                "Γειά σου ,Κόσμε",
                "Español es agra,dable escuchar",
                "한국,가,고싶다ㅠ",
                "🢇🄐,🏈𠆶💑😅",
            ],
        ),
        np.array(
            [
                Decimal("0.0"),
                Decimal("-0.222"),
                Decimal("1111.316"),
                Decimal("1"),
                Decimal("5.1"),
                Decimal("-1"),
                Decimal("-1"),
            ]
        ),
        np.array(
            [
                datetime.timedelta(days=5, seconds=4, weeks=4),
                datetime.timedelta(days=5, seconds=5, weeks=4),
                datetime.timedelta(days=11, seconds=4, weeks=4),
                datetime.timedelta(days=5, seconds=4, weeks=4),
                datetime.timedelta(days=5, seconds=4, weeks=4),
                datetime.timedelta(days=5, seconds=4, weeks=4),
                datetime.timedelta(days=5, seconds=64, weeks=4),
                datetime.timedelta(days=11, seconds=4, weeks=4),
                datetime.timedelta(days=42, seconds=11, weeks=4),
                datetime.timedelta(days=5, seconds=123, weeks=4),
            ]
        ),
        np.append(
            pd.date_range("2017-07-03", "2017-07-17").date,
            [datetime.date(2016, 3, 3)],
        ),
        pd.array(
            [
                True,
                False,
                True,
                True,
                False,
                False,
                True,
            ]
        ),
        pd.arrays.IntegerArray(
            np.array([1, -3, 2, 3, 10] * 10, np.int8),
            np.array([False, False, False, False, False] * 10),
        ),
    ],
)
def bodo_arr_val(request):
    return request.param


@pytest.mark.slow
def test_in(bodo_arr_val, memory_leak_check):
    def test_impl(A, val):
        return val in A

    init_val = bodo_arr_val[1]
    check_func(test_impl, (bodo_arr_val, init_val))
    # Remove all locations of init_val. In all arrays elements 0 and 1 are distinct
    np.where(bodo_arr_val == init_val, bodo_arr_val[0], bodo_arr_val)
    check_func(test_impl, (bodo_arr_val, init_val))


@pytest.mark.slow
def test_any(bodo_arr_val, memory_leak_check):
    def test_impl(A):
        # Python's logical or won't return a bool so set to bool
        return bool(np.any(A))

    if isinstance(bodo_arr_val, pd.arrays.IntegerArray):
        # Reduce op is not supported on integer arrays
        # This tests that there is a parallel version for a Numpy Array type
        bodo_arr_val = np.array(bodo_arr_val)

    check_func(test_impl, (bodo_arr_val,))


@pytest.mark.slow
def test_all(bodo_arr_val, memory_leak_check):
    def test_impl(A):
        # Python's logical and won't return a bool so set to bool
        return bool(np.all(A))

    if isinstance(bodo_arr_val, pd.arrays.IntegerArray):
        # Reduce op is not supported on integer arrays
        # This tests that there is a parallel version for a Numpy Array type
        bodo_arr_val = np.array(bodo_arr_val)

    check_func(test_impl, (bodo_arr_val,))


@pytest.mark.slow
def test_any_all_numpy_2d(memory_leak_check):
    """Check that a multidimensional Numpy array outputs the
    correct result for a 2D array. This shouldn't use our kernel.
    """

    def test_impl_any(A):
        return bool(np.any(A))

    def test_impl_all(A):
        return bool(np.all(A))

    arr = np.array([[False, False, True], [False, False, False], [True, True, True]])
    assert test_impl_any(arr) == bodo.jit(test_impl_any)(arr)
    assert test_impl_all(arr) == bodo.jit(test_impl_all)(arr)


def test_np_random_multivariate_normal(memory_leak_check):
    def test_impl(nvars, nrows):
        mu = np.zeros(nvars, dtype=np.float64)
        S = np.random.uniform(-5.0, 5.0, (nvars, nvars))
        cov = np.dot(S.T, S)
        A = np.random.multivariate_normal(mu, cov, nrows)
        return A

    nvars = 10
    nrows = 20
    np.random.seed(2)
    # Seeding doesn't seem to work properly so we can't check
    # equality. Instead, test the shape by setting the tolerance very high
    check_func(test_impl, (nvars, nrows), atol=1000.0, rtol=1000.0)


@pytest.fixture(
    params=[
        pytest.param(
            pd.arrays.IntegerArray(
                np.array([1, -3, 2, 3, 10] * 2, np.int8),
                np.array([False, True, True, False, False] * 2),
            ),
            id="IntegerArray",
        ),
        pytest.param(
            pd.arrays.FloatingArray(
                np.array([1.0, -3.0, 2.0, 3.0, 10.0] * 2, np.float64),
                np.array([False, True, True, False, False] * 2),
            ),
            id="FloatingArray",
            marks=pytest.mark.skip(reason="FloatingArray not fully implemented"),
        ),
        pytest.param(
            pd.array([True, False, True, pd.NA, False] * 2), id="BooleanArray"
        ),
        pytest.param(
            np.array(
                [
                    Decimal("1.6"),
                    None,
                    Decimal("-0.222"),
                    Decimal("1111.316"),
                    Decimal("5.1"),
                ]
                * 2
            ),
            id="DecimalArray",
        ),
        pytest.param(
            np.append(pd.date_range("2020-01-14", "2020-01-22").date, [None]),
            id="DateArray",
        ),
        pytest.param(
            np.append(
                [
                    datetime.timedelta(days=5, seconds=4, weeks=4),
                    datetime.timedelta(days=2, microseconds=121),
                ],
                [None, datetime.timedelta(microseconds=100000001213131, hours=5)] * 4,
            ),
            id="TimedeltaArray",
        ),
        pytest.param(pd.Categorical([1, 2, 5, None, 2] * 2), marks=pytest.mark.slow),
        pytest.param(pd.Categorical(["AA", "BB", "", "AA", None] * 2)),
        pytest.param(
            pd.Categorical(
                np.array(
                    ["2020-01-14", "2020-01-15", "2020-01-16", "2020-01-17", "NAT"],
                    dtype="datetime64[ns]",
                )
            )
        ),
        pytest.param(
            pd.Categorical(
                np.append(
                    pd.timedelta_range(start="1 day", periods=9),
                    [np.timedelta64("NaT")],
                )
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Categorical([3, 1.321, 0.0122, -1.321, 0.0, 1, 3, 2]),
            marks=pytest.mark.slow,
        ),
    ]
)
def mutable_bodo_arr(request):
    return request.param


# TODO: Add immutable bodo arrays


@pytest.mark.slow
def test_setitem_none(mutable_bodo_arr, memory_leak_check):
    def test_impl(A, idx):
        A[idx] = None
        return A

    np.random.seed(0)

    # scalar idx
    idx = np.random.randint(0, len(mutable_bodo_arr), 1)[0]
    check_func(
        test_impl, (mutable_bodo_arr.copy(), idx), copy_input=True, dist_test=False
    )

    # int arr idx
    idx = np.random.randint(0, len(mutable_bodo_arr), 11)
    check_func(
        test_impl, (mutable_bodo_arr.copy(), idx), copy_input=True, dist_test=False
    )

    # bool arr idx
    idx = np.random.ranf(len(mutable_bodo_arr)) < 0.2
    check_func(
        test_impl, (mutable_bodo_arr.copy(), idx), copy_input=True, dist_test=False
    )

    # slice idx
    idx = slice(1, 4)
    check_func(
        test_impl, (mutable_bodo_arr.copy(), idx), copy_input=True, dist_test=False
    )


@pytest.mark.slow
def test_setitem_optional(mutable_bodo_arr, memory_leak_check):
    def test_impl(A, i, flag, val):
        if flag:
            x = None
        else:
            x = val
        A[i] = x
        return A

    np.random.seed(0)

    # scalar idx
    idx = np.random.randint(0, len(mutable_bodo_arr), 1)[0]
    val = mutable_bodo_arr[0]
    check_func(
        test_impl,
        (mutable_bodo_arr.copy(), idx, False, val),
        copy_input=True,
        dist_test=False,
    )
    check_func(
        test_impl,
        (mutable_bodo_arr.copy(), idx, True, val),
        copy_input=True,
        dist_test=False,
    )

    # int arr idx
    idx = np.random.randint(0, len(mutable_bodo_arr), 11)
    check_func(
        test_impl,
        (mutable_bodo_arr.copy(), idx, False, val),
        copy_input=True,
        dist_test=False,
    )
    check_func(
        test_impl,
        (mutable_bodo_arr.copy(), idx, True, val),
        copy_input=True,
        dist_test=False,
    )

    # bool arr idx
    idx = np.random.ranf(len(mutable_bodo_arr)) < 0.2
    check_func(
        test_impl,
        (mutable_bodo_arr.copy(), idx, False, val),
        copy_input=True,
        dist_test=False,
    )
    check_func(
        test_impl,
        (mutable_bodo_arr.copy(), idx, True, val),
        copy_input=True,
        dist_test=False,
    )

    # slice idx
    idx = slice(1, 4)
    check_func(
        test_impl,
        (mutable_bodo_arr.copy(), idx, False, val),
        copy_input=True,
        dist_test=False,
    )
    check_func(
        test_impl,
        (mutable_bodo_arr.copy(), idx, True, val),
        copy_input=True,
        dist_test=False,
    )


@pytest.mark.slow
def test_getitem(mutable_bodo_arr, memory_leak_check):
    """
    Tests that getitem works properly for mutable arrays in bodo
    """

    def test_impl(A, idx):
        return A[idx]

    np.random.seed(0)

    # A single integer
    idx = 0
    check_func(test_impl, (mutable_bodo_arr, idx), dist_test=False)

    # Array of integers
    idx = np.random.randint(0, len(mutable_bodo_arr), 11)
    check_func(test_impl, (mutable_bodo_arr, idx), dist_test=False)

    # Array of booleans
    idx = np.random.ranf(len(mutable_bodo_arr)) < 0.2
    check_func(test_impl, (mutable_bodo_arr, idx), dist_test=False)

    # Check nullable with pd.array and insert NA values
    # Indexing with pd.Array into Numpy array not supported in regular Numpy/Pandas
    if isinstance(mutable_bodo_arr, np.ndarray):
        idx[1] = False
        idx[-1] = False
        py_output = mutable_bodo_arr[idx]
    else:
        py_output = None
    idx = pd.array(idx)
    idx[1] = None
    idx[-1] = None
    check_func(test_impl, (mutable_bodo_arr, idx), py_output=py_output, dist_test=False)

    # Slice
    idx = slice(1, 4)
    check_func(test_impl, (mutable_bodo_arr, idx), dist_test=False)


@pytest.mark.slow
def test_int_getitem_none(mutable_bodo_arr, memory_leak_check):
    """
    Test that getitem with None won't fail at runtime.
    It will give a garbage value because users need to check for
    NaN with pd.isnan
    """

    def test_impl(A, idx):
        return A[idx]

    # Find the first nan value in the array
    idx = pd.Series(mutable_bodo_arr).isna().idxmax()

    # Test that we don't get a runtime error. The result is garbage,
    # so we ignore it.
    bodo.jit(test_impl)(mutable_bodo_arr, idx)


@pytest.mark.slow
def test_isna_check(mutable_bodo_arr, memory_leak_check):
    """
    Test that isna works properly for each array
    """
    if isinstance(mutable_bodo_arr, pd.Categorical):
        # TODO: [BE-130] Support Categorical in UDFs.
        return

    def test_impl(S):
        return S.map(lambda a: a if not pd.isna(a) else None).values

    if isinstance(mutable_bodo_arr, pd.core.arrays.IntegerArray):
        # Pandas converts IntegerArrays into floating point arrays.
        # As a result, None is set to np.nan, so simply setting check_dtype=False
        # fails.
        py_output = mutable_bodo_arr
    else:
        py_output = None
    check_func(test_impl, (pd.Series(mutable_bodo_arr),), py_output=py_output)


@pytest.mark.slow
def test_setna_getitem(mutable_bodo_arr, memory_leak_check):
    """
    Test that setna works properly for each array and can be used
    with getitem.
    """

    def test_impl(arr):
        # Use range to prevent loop fusion.
        # We want to verify setna actually modifies
        # the data array to a valid value.
        n = len(arr)
        out_arr = arr.copy()
        for i in range(n):
            arr[i] = None
        for i in range(n):
            out_arr[i] = arr[i]
        return out_arr

    # Test that we don't get a runtime error. The result is garbage,
    # because arr[i] will return a garbage value if pd.isna(arr[i])
    bodo.jit(test_impl)(mutable_bodo_arr)


# TODO: Add memory leak check when constant lowering memory leak is fixed.
@pytest.mark.slow
def test_bad_setitem(mutable_bodo_arr):
    """
    Tests that a type mismatch gives a reasonable error message and doesn't just fail
    randomly in Numba.

    These tests check that non-integer values (i.e. floats) aren't accepted.
    """

    def test_impl_scalar(A):
        A[2] = 9.8
        return A

    def test_impl_arr_like(A, ind):
        A[ind] = np.random.rand(2)
        return A

    def test_impl_series_like(A, ind):
        A[ind] = pd.Series(np.random.rand(2))
        return A

    def test_impl_list_like(A, ind):
        A[ind] = [1.1, 1.4]
        return A

    if isinstance(mutable_bodo_arr, pd.Categorical) and np.issubdtype(
        mutable_bodo_arr.categories.dtype, np.float64
    ):
        err_typ = ValueError
        error_msg = "Cannot setitem on a Categorical with a new category, set the categories first"
    else:
        err_typ = BodoError
        error_msg = "received an incorrect 'value' type"

    # we support setting a float scalar value to IntegerArray (see int_arr_setitem)
    if not isinstance(mutable_bodo_arr, pd.arrays.IntegerArray):
        with pytest.raises(err_typ, match=error_msg):
            bodo.jit(test_impl_scalar)(mutable_bodo_arr)

    indices = [
        np.array([False, True, True, False, False]),
        np.random.randint(0, len(mutable_bodo_arr), 2),
        [1, 2],
        slice(0, 2),
    ]
    for ind in indices:
        with pytest.raises(err_typ, match=error_msg):
            bodo.jit(test_impl_arr_like)(mutable_bodo_arr, ind)
        with pytest.raises(err_typ, match=error_msg):
            bodo.jit(test_impl_series_like)(mutable_bodo_arr, ind)
        with pytest.raises(err_typ, match=error_msg):
            bodo.jit(test_impl_list_like)(mutable_bodo_arr, ind)


@pytest.mark.slow
def test_numpy_contains_inline(memory_leak_check):
    """Checks that we inline the in operator if a valid impl exists
    supported by Bodo (not Numba)."""

    def test_impl(arr):
        return "a" in arr

    arr = np.array([3, 4, 0, 2, 5])
    # contains operator isn't support on distributed Numpy
    # arrays yet.
    check_func(test_impl, (arr,), dist_test=False)


def test_np_select(arr_tuple_val, memory_leak_check):
    """tests np select for non null ndarrays and pandas arrays"""
    np.random.seed(42)
    if isinstance(arr_tuple_val[0][0], datetime.date):
        pytest.skip("need support for np.where with datetime date types see BE-1494")

    minsize = arr_tuple_val[0].size
    if minsize > arr_tuple_val[1].size:
        minsize = arr_tuple_val[1].size

    cond1 = np.random.randint(2, size=minsize).astype(bool)
    cond2 = np.random.randint(2, size=minsize).astype(bool)

    # test all combinations of tuple/list

    def impl_1(A1, A2, cond1, cond2):
        choicelist = (A1, A2)
        condlist = (cond1, cond2)
        return np.select(condlist, choicelist)

    def impl_2(A1, A2, cond1, cond2):
        choicelist = (A1, A2)
        condlist = [cond1, cond2]
        return np.select(condlist, choicelist)

    def impl_3(A1, A2, cond1, cond2):
        choicelist = [A1, A2]
        condlist = (cond1, cond2)
        return np.select(condlist, choicelist)

    def impl_4(A1, A2, cond1, cond2):
        choicelist = [A1, A2]
        condlist = [cond1, cond2]
        return np.select(condlist, choicelist)

    # for numeric/bool, we default to 0/false. This is to keep the expected behavior of np select
    # in situations that a user might resonably want/expect to have the default set to 0.
    # for all other types, we default to NA, for type stability.
    def na_impl(A1, A2, cond1, cond2):
        choicelist = [A1, A2]
        condlist = [cond1, cond2]
        return np.select(condlist, choicelist, default=np.NAN)

    if not isinstance(
        arr_tuple_val[0][0], (np.integer, np.bool_, np.float64, np.float32)
    ):
        py_out = na_impl(
            arr_tuple_val[0][:minsize], arr_tuple_val[1][:minsize], cond1, cond2
        )
    else:
        py_out = None

    for impl in [impl_1, impl_2, impl_3, impl_4]:
        check_func(
            impl,
            (arr_tuple_val[0][:minsize], arr_tuple_val[1][:minsize], cond1, cond2),
            check_dtype=False,
            py_output=py_out,
        )


def test_np_select_nullable(arr_tuple_val, memory_leak_check):
    """tests np select for nullable pandas arrays"""
    np.random.seed(42)
    if isinstance(arr_tuple_val[0][0], datetime.date):
        pytest.skip("need support for np.where with datetime date types see BE-1494")

    minsize = arr_tuple_val[0].size
    if minsize > arr_tuple_val[1].size:
        minsize = arr_tuple_val[1].size

    cond1 = np.random.randint(2, size=minsize).astype(bool)
    cond2 = np.random.randint(2, size=minsize).astype(bool)

    def impl1(A1, A2, cond1, cond2):
        choicelist = (A1, A2)
        condlist = (cond1, cond2)
        return np.select(condlist, choicelist)

    def impl2(A1, A2, cond1, cond2):
        choicelist = (A1, A2)
        condlist = [cond1, cond2]
        return np.select(condlist, choicelist)

    def impl3(A1, A2, cond1, cond2):
        choicelist = [A1, A2]
        condlist = (cond1, cond2)
        return np.select(condlist, choicelist)

    def impl4(A1, A2, cond1, cond2):
        choicelist = [A1, A2]
        condlist = [cond1, cond2]
        return np.select(condlist, choicelist)

    # for numeric/bool, we default to 0/false. This is to keep the expected behavior of np select
    # in situations that a user might resonably want/expect to have the default set to 0.
    # for all other types, we default to NA, for type stability.
    def na_impl(A1, A2, cond1, cond2):
        choicelist = [A1, A2]
        condlist = [cond1, cond2]
        return np.select(condlist, choicelist, default=np.NAN)

    if not isinstance(
        arr_tuple_val[0][0], (np.integer, np.bool_, np.float64, np.float32)
    ):
        py_out = na_impl(
            arr_tuple_val[0][:minsize], arr_tuple_val[1][:minsize], cond1, cond2
        )
    else:
        py_out = None

    A1 = pd.array(arr_tuple_val[0][:minsize])
    A2 = pd.array(arr_tuple_val[1][:minsize])
    A1[np.random.choice([True, False], size=minsize, p=[0.5, 0.5])] = np.NaN
    A2[np.random.choice([True, False], size=minsize, p=[0.5, 0.5])] = np.NaN

    if isinstance(A1, pd.arrays.FloatingArray) or isinstance(
        A2, pd.arrays.FloatingArray
    ):
        pytest.skip(
            "Needs support for nullable floating point arrays in Bodo. See BE-41"
        )
    for impl in [impl1, impl2, impl3, impl4]:
        check_func(impl, (A1, A2, cond1, cond2), check_dtype=False, py_output=py_out)


def test_np_where_impl(arr_tuple_val, memory_leak_check):
    """tests np where on non null numpy arrays"""
    np.random.seed(42)
    if isinstance(arr_tuple_val[0][0], datetime.date):
        pytest.skip("need support for np.where with datetime date types see BE-1494")

    minsize = arr_tuple_val[0].size
    if minsize > arr_tuple_val[1].size:
        minsize = arr_tuple_val[1].size

    cond = np.random.randint(2, size=minsize).astype(bool)

    def impl(A1, A2, cond):
        return np.where(cond, A1, A2)

    check_func(impl, (arr_tuple_val[0][:minsize], arr_tuple_val[1][:minsize], cond))


def test_np_where_impl_nullable(arr_tuple_val, memory_leak_check):
    """tests np where on non nullable pandas arrays"""
    np.random.seed(42)
    if isinstance(arr_tuple_val[0][0], datetime.date):
        pytest.skip("need support for np.where with datetime date types see BE-1494")

    minsize = arr_tuple_val[0].size
    if minsize > arr_tuple_val[1].size:
        minsize = arr_tuple_val[1].size

    cond = np.random.randint(2, size=minsize).astype(bool)

    cond[np.random.choice([True, False], size=minsize, p=[0.5, 0.5])] = np.NaN

    def impl(A1, A2, cond):
        return np.where(cond, A1, A2)

    A1 = pd.array(arr_tuple_val[0][:minsize])
    A2 = pd.array(arr_tuple_val[1][:minsize])
    A1[np.random.choice([True, False], size=minsize, p=[0.5, 0.5])] = np.NaN
    A2[np.random.choice([True, False], size=minsize, p=[0.5, 0.5])] = np.NaN

    if isinstance(A1, pd.arrays.FloatingArray) or isinstance(
        A2, pd.arrays.FloatingArray
    ):
        pytest.skip(
            "Needs support for nullable floating point arrays in Bodo. See BE-41"
        )

    check_func(impl, (A1, A2, cond))


def test_np_select_none_default(arr_tuple_val, memory_leak_check):
    """tests np select when passing "None" as a default value with numpy arrays"""
    np.random.seed(42)
    if isinstance(arr_tuple_val[0][0], datetime.date):
        pytest.skip("need support for np.where with datetime date types see BE-1494")

    minsize = arr_tuple_val[0].size
    if minsize > arr_tuple_val[1].size:
        minsize = arr_tuple_val[1].size

    cond1 = np.random.randint(2, size=minsize).astype(bool)
    cond2 = np.random.randint(2, size=minsize).astype(bool)

    # test all combinations of tuple/list

    def impl(A1, A2, cond1, cond2):
        choicelist = [A1, A2]
        condlist = [cond1, cond2]
        return np.select(condlist, choicelist, default=None)

    def py_impl(A1, A2, cond1, cond2):
        choicelist = [A1, A2]
        condlist = [cond1, cond2]
        return np.select(condlist, choicelist, default=pd.NA)

    py_out = py_impl(
        arr_tuple_val[0][:minsize], arr_tuple_val[1][:minsize], cond1, cond2
    )  # .astype(arr_tuple_val[0].dtype)

    if arr_tuple_val[0].dtype.name.startswith("float"):
        py_out[pd.isna(py_out)] = np.NAN
        py_out = py_out.astype(float)

    check_func(
        impl,
        (arr_tuple_val[0][:minsize], arr_tuple_val[1][:minsize], cond1, cond2),
        check_dtype=False,
        py_output=py_out,
    )


def test_np_select_set_default(arr_tuple_val, memory_leak_check):
    """tests np select when passing a default value with numpy arrays"""
    np.random.seed(42)
    if isinstance(arr_tuple_val[0][0], datetime.date):
        pytest.skip("need support for np.where with datetime date types see BE-1494")

    minsize = arr_tuple_val[0].size
    if minsize > arr_tuple_val[1].size:
        minsize = arr_tuple_val[1].size

    cond1 = np.random.randint(2, size=minsize).astype(bool)
    cond2 = np.random.randint(2, size=minsize).astype(bool)

    # test all combinations of tuple/list

    def impl(A1, A2, cond1, cond2, default):
        choicelist = [A1, A2]
        condlist = [cond1, cond2]
        return np.select(condlist, choicelist, default)

    check_func(
        impl,
        (
            arr_tuple_val[0][:minsize],
            arr_tuple_val[1][:minsize],
            cond1,
            cond2,
            arr_tuple_val[0][0],
        ),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "arr_fun",
    [
        pytest.param(lambda x: np.empty((x, x), dtype="int64").fill(-1), id="empty"),
        pytest.param(lambda x: np.zeros((x, x), dtype="int64"), id="zeros"),
        pytest.param(lambda x: np.ones((x, x), dtype="int64"), id="ones"),
    ],
)
def test_np_array_from_enum(arr_fun):
    """tests np array from IntEnum"""

    class TestEnum(IntEnum):
        A = 1
        B = 2
        C = 3

    for member in TestEnum:
        check_func(arr_fun, (member,))


def test_np_ravel(memory_leak_check):
    """
    [BE-2795] Tests support np.ravel
    on 2-dim numpy array with various
    layouts.
    """

    def impl(df):
        values = df.values
        values = values.reshape(-1, 1)
        result = values.ravel()
        return result

    df = pd.DataFrame(
        {
            "A": np.arange(100),
            "B": np.arange(100, 200) / 0.25,
        }
    )
    check_func(impl, (df,))
