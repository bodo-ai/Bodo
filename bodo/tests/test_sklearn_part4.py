# Copyright (C) 2022 Bodo Inc. All rights reserved.
""" Test miscellaneous supported sklearn models and methods
    Currently this file tests:
    Robust Scaler
    The Robust Scaler tests need to be done across two files (this and test_sklearn_part5)
    due to the large ammount of test_robust_scalar tests,
    which can cause OOM issues on nightly due to numba caching artifacts.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

import bodo
from bodo.tests.test_sklearn_part3 import gen_sklearn_scalers_random_data
from bodo.tests.utils import _get_dist_arg, check_func

# ---------------------RobustScaler Tests, part 1--------------------


@pytest.fixture(
    params=[
        # Test one with numpy array and one with df
        (
            gen_sklearn_scalers_random_data(15, 5, 0.2, 4),
            gen_sklearn_scalers_random_data(60, 5, 0.5, 2),
        ),
        (
            pd.DataFrame(gen_sklearn_scalers_random_data(20, 3)),
            gen_sklearn_scalers_random_data(100, 3),
        ),
        # The other combinations are marked slow
        pytest.param(
            (
                gen_sklearn_scalers_random_data(20, 3),
                gen_sklearn_scalers_random_data(100, 3),
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                gen_sklearn_scalers_random_data(20, 3),
                pd.DataFrame(gen_sklearn_scalers_random_data(100, 3)),
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.DataFrame(gen_sklearn_scalers_random_data(20, 3)),
                pd.DataFrame(gen_sklearn_scalers_random_data(100, 3)),
            ),
            marks=pytest.mark.slow,
        ),
    ]
)
def robust_scalar_data(request):
    """
    Returns data for the robust scalar tests
    """
    return request.param


@pytest.mark.parametrize(
    "with_centering", [True, pytest.param(False, marks=pytest.mark.slow)]
)
@pytest.mark.parametrize(
    "with_scaling", [True, pytest.param(False, marks=pytest.mark.slow)]
)
@pytest.mark.parametrize(
    "quantile_range",
    [
        (25.0, 75.0),
        pytest.param((10.0, 85.0), marks=pytest.mark.slow),
        pytest.param((40.0, 60.0), marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize(
    "unit_variance", [False, pytest.param(True, marks=pytest.mark.slow)]
)
@pytest.mark.parametrize("copy", [True, pytest.param(False, marks=pytest.mark.slow)])
def test_robust_scaler_fit(
    robust_scalar_data,
    with_centering,
    with_scaling,
    quantile_range,
    unit_variance,
    copy,
    memory_leak_check,
):
    """
    Tests for sklearn.preprocessing.RobustScaler.fit implementation in Bodo.
    """

    def test_fit(X):
        m = RobustScaler(
            with_centering=with_centering,
            with_scaling=with_scaling,
            quantile_range=quantile_range,
            unit_variance=unit_variance,
            copy=copy,
        )
        m = m.fit(X)
        return m

    py_output = test_fit(robust_scalar_data[0])
    bodo_output = bodo.jit(distributed=["X"])(test_fit)(
        _get_dist_arg(robust_scalar_data[0])
    )

    if with_centering:
        assert np.allclose(
            py_output.center_, bodo_output.center_, atol=1e-4, equal_nan=True
        )
    if with_scaling:
        assert np.allclose(
            py_output.scale_, bodo_output.scale_, atol=1e-4, equal_nan=True
        )


@pytest.mark.parametrize(
    "with_centering", [True, pytest.param(False, marks=pytest.mark.slow)]
)
@pytest.mark.parametrize(
    "with_scaling", [True, pytest.param(False, marks=pytest.mark.slow)]
)
@pytest.mark.parametrize(
    "quantile_range",
    [
        (25.0, 75.0),
        pytest.param((10.0, 85.0), marks=pytest.mark.slow),
        pytest.param((40.0, 60.0), marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize(
    "unit_variance", [False, pytest.param(True, marks=pytest.mark.slow)]
)
@pytest.mark.parametrize("copy", [True, pytest.param(False, marks=pytest.mark.slow)])
def test_robust_scaler_transform(
    robust_scalar_data,
    with_centering,
    with_scaling,
    quantile_range,
    unit_variance,
    copy,
    memory_leak_check,
):
    """
    Tests for sklearn.preprocessing.RobustScaler.transform implementation in Bodo.
    """

    def test_transform(X, X1):
        m = RobustScaler(
            with_centering=with_centering,
            with_scaling=with_scaling,
            quantile_range=quantile_range,
            unit_variance=unit_variance,
            copy=copy,
        )
        m = m.fit(X)
        X1_transformed = m.transform(X1)
        return X1_transformed

    check_func(
        test_transform,
        robust_scalar_data,
        is_out_distributed=True,
        atol=1e-4,
        copy_input=True,
    )
