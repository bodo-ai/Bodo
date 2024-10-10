# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Tests for interval arrays"""

import numpy as np
import pandas as pd
import pytest

from bodo.tests.utils import check_func


@pytest.fixture(
    params=[pd.arrays.IntervalArray.from_arrays(np.arange(11), np.arange(11) + 1)]
)
def interval_array_value(request):
    return request.param


@pytest.mark.slow
def test_unbox(interval_array_value, memory_leak_check):
    # just unbox
    def impl(arr_arg):
        return True

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl, (interval_array_value,))
    check_func(impl2, (interval_array_value,))


@pytest.mark.slow
def test_nbytes(interval_array_value, memory_leak_check):
    """Test IntervalArrayType nbytes"""

    def impl(arr):
        return arr.nbytes

    check_func(impl, (interval_array_value,))


@pytest.fixture(
    params=[
        pytest.param((pd.Interval(10, 12), pd.Interval(20, 30)), id="integers"),
        pytest.param(
            (pd.Interval(2.71828, 3.1415), pd.Interval(-12.7, 12.8)), id="floats"
        ),
        pytest.param(
            (
                pd.Interval(pd.Timestamp("2024"), pd.Timestamp("2025")),
                pd.Interval(pd.Timestamp("2023-7-4"), pd.Timestamp("2024-4-1")),
            ),
            id="timestamps",
        ),
        pytest.param(
            (
                pd.Interval(pd.Timedelta("1H"), pd.Timedelta("1D")),
                pd.Interval(pd.Timedelta("1H"), pd.Timedelta("2D")),
            ),
            id="timedeltas",
        ),
    ]
)
def interval_scalars(request):
    """
    Returns a pair of interval scalars for several types.
    """
    return request.param


def test_interval_scalar_box_unbox(interval_scalars, memory_leak_check):
    """
    Tests the correctness of boxing and unboxing on interval scalars
    """

    def test_unbox(interval_a, interval_b):
        return True

    def test_box_1(interval_a, interval_b):
        return interval_a

    def test_box_2(interval_a, interval_b):
        return interval_b

    check_func(test_unbox, interval_scalars)
    check_func(test_box_1, interval_scalars)
    check_func(test_box_2, interval_scalars)


def test_interval_scalar_fields_constructor(interval_scalars, memory_leak_check):
    """
    Tests the correctness of accessing the left/right fields and the constructor
    for interval scalars.
    """

    def test_left_right(interval_a, interval_b):
        return (interval_a.left, interval_a.right, interval_b.left, interval_b.right)

    def test_constructor(interval_a, interval_b):
        lhs = min(interval_a.left, interval_b.left)
        rhs = max(interval_a.right, interval_b.right)
        return pd.Interval(lhs, rhs)

    check_func(test_left_right, interval_scalars)
    check_func(test_constructor, interval_scalars)


def test_interval_scalar_comparisons(interval_scalars, memory_leak_check):
    """
    Tests the correctness of comparisons of intervals.
    """

    def test_comparisons(interval_a, interval_b):
        answer = [False]
        q = min(interval_a.left, interval_b.left)
        r = max(interval_a.left, interval_b.left)
        s = min(interval_a.right, interval_b.right)
        t = max(interval_a.right, interval_b.right)
        interval_c = pd.Interval(q, r)
        interval_d = pd.Interval(q, s)
        interval_e = pd.Interval(q, t)
        interval_f = pd.Interval(r, t)
        intervals = [
            interval_a,
            interval_b,
            interval_c,
            interval_d,
            interval_e,
            interval_f,
        ]
        for first_interval in intervals:
            for second_interval in intervals:
                answer.append(first_interval == second_interval)
                answer.append(first_interval != second_interval)
                answer.append(first_interval < second_interval)
                answer.append(first_interval <= second_interval)
                answer.append(first_interval > second_interval)
                answer.append(first_interval >= second_interval)
        return answer

    check_func(test_comparisons, interval_scalars)
