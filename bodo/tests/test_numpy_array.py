import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func


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


def test_np_union1d(arr_tuple_val, memory_leak_check):
    def impl(A1, A2):
        return np.union1d(A1, A2)

    # Keep as array tuple to allow for scaling to larger array tuple
    # sizes if needed for other Numpy functions.
    A1 = arr_tuple_val[0]
    A2 = arr_tuple_val[1]

    check_func(impl, (A1, A2))


def test_np_intersect1d(arr_tuple_val, memory_leak_check):
    def impl(A1, A2):
        return np.intersect1d(A1, A2)

    # Keep as array tuple to allow for scaling to larger array tuple
    # sizes if needed for other Numpy functions.
    A1 = arr_tuple_val[0]
    A2 = arr_tuple_val[1]

    # TODO(Nick): Add parallel test when there is parallel support.
    check_func(impl, (A1, A2), dist_test=False)


def test_np_setdiff1d(arr_tuple_val, memory_leak_check):
    def impl(A1, A2):
        return np.setdiff1d(A1, A2)

    # Keep as array tuple to allow for scaling to larger array tuple
    # sizes if needed for other Numpy functions.
    A1 = arr_tuple_val[0]
    A2 = arr_tuple_val[1]

    # TODO(Nick): Add parallel test when there is parallel support.
    check_func(impl, (A1, A2), dist_test=False)


def test_np_linspace(memory_leak_check):
    def test_impl(start, stop, num):
        return np.linspace(start, stop, num=num)

    check_func(test_impl, (0, 1000, 100000))
    check_func(test_impl, (-2000, -4000, 100000))
    check_func(test_impl, (-5, 4.5252, 1000))


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


@pytest.mark.parametrize(
    "arr",
    [
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
def test_in(arr, memory_leak_check):
    def test_impl(A, val):
        return val in A

    init_val = arr[1]
    check_func(test_impl, (arr, init_val))
    # Remove all locations of init_val. In all arrays elements 0 and 1 are distinct
    np.where(arr == init_val, arr[0], arr)
    check_func(test_impl, (arr, init_val))
