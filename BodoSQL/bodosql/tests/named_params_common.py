"""
Common file used by tests that evaluate queries with named parameters.
"""
# Copyright (C) 2022 Bodo Inc. All rights reserved.

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(
    params=[
        pytest.param((np.uint64(2), np.uint64(3)), marks=pytest.mark.slow),
        pytest.param((np.uint8(5), np.uint32(2)), marks=pytest.mark.slow),
        (2, 0),
        pytest.param((np.int8(1), np.int32(1)), marks=pytest.mark.slow),
    ]
)
def int_named_params(request):
    """
    Fixture for integer named params. These always contain 2 values:
    @a, and @b, which refer to elements 0 and 1 of the tuple.
    """
    params_tuple = request.param
    return {"a": params_tuple[0], "b": params_tuple[1]}


@pytest.fixture(
    params=[
        (5.6, np.float32(17.7)),
        pytest.param((np.float32(5.5), np.float64(0.05)), marks=pytest.mark.slow),
    ]
)
def float_named_params(request):
    """
    Fixture for float named params. These always contain 2 values:
    @a, and @b, which refer to elements 0 and 1 of the tuple.
    """
    params_tuple = request.param
    return {"a": params_tuple[0], "b": params_tuple[1]}


@pytest.fixture(
    params=[
        ("hello", "world"),
    ]
)
def string_named_params(request):
    """
    Fixture for string named params. These always contain 2 values:
    @a, and @b, which refer to elements 0 and 1 of the tuple.
    """
    params_tuple = request.param
    return {"a": params_tuple[0], "b": params_tuple[1]}


@pytest.fixture(
    params=[
        # Check date like values
        (pd.Timestamp(2021, 8, 16), pd.Timestamp(2020, 2, 2)),
        pytest.param(
            (pd.Timestamp(2022, 1, 31, second=15), pd.Timestamp(2000, 1, 1, hour=3)),
            marks=pytest.mark.slow,
        ),
    ]
)
def timestamp_named_params(request):
    """
    Fixture for Timestamp named params. These always contain 2 values:
    @a, and @b, which refer to elements 0 and 1 of the tuple.
    """
    params_tuple = request.param
    return {"a": params_tuple[0], "b": params_tuple[1]}


@pytest.fixture(
    params=[
        # Check positive values
        (pd.Timedelta(days=1), pd.Timedelta(hours=5)),
        pytest.param(
            (pd.Timedelta(days=-2, hours=1), pd.Timedelta(minutes=-15)),
            marks=pytest.mark.slow,
        ),
    ]
)
def timedelta_named_params(request):
    """
    Fixture for Timedelta named params. These always contain 2 values:
    @a, and @b, which refer to elements 0 and 1 of the tuple.
    """
    params_tuple = request.param
    return {"a": params_tuple[0], "b": params_tuple[1]}


@pytest.fixture(
    params=[
        # Check positive values
        (pd.DateOffset(years=1, months=2), pd.DateOffset(months=-3)),
        (pd.DateOffset(years=-3, months=1), pd.DateOffset(years=4)),
    ]
)
def dateoffset_named_params(request):
    """
    Fixture for Dateoffset named params. These always contain 2 values:
    @a, and @b, which refer to elements 0 and 1 of the tuple.
    """
    params_tuple = request.param
    return {"a": params_tuple[0], "b": params_tuple[1]}


@pytest.fixture(
    params=[
        (2, np.uint8(0)),
        (5.6, np.float32(17.7)),
        ("hello", "world"),
        (pd.Timestamp(2021, 8, 16), pd.Timestamp(year=2020, month=2, day=2, minute=4)),
        (pd.Timedelta(days=-1), pd.Timedelta(hours=5)),
    ]
)
def named_params_all_column_types(request):
    """
    Fixture for named params in all different column types,
    not considering differences in bitwidth.
    """
    params_tuple = request.param
    return {"a": params_tuple[0], "b": params_tuple[1]}
