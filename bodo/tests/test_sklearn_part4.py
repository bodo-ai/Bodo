# Copyright (C) 2019 Bodo Inc. All rights reserved.
""" Test miscellaneous supported sklearn models and methods
    Currently this file tests:
    Robust Scaler
    This needs to be done due to the large ammount of test_robust_scalar tests,
    which can cause OOM issues on nightly due to numba caching artifacts.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

import bodo
from bodo.tests.test_sklearn_part3 import gen_sklearn_scalers_random_data
from bodo.tests.utils import _get_dist_arg, check_func

# ---------------------RobustScaler Tests--------------------


@pytest.mark.parametrize(
    "data",
    [
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
    ],
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
def test_robust_scaler(
    data,
    with_centering,
    with_scaling,
    quantile_range,
    unit_variance,
    copy,
    memory_leak_check,
):
    """
    Tests for sklearn.preprocessing.RobustScaler implementation in Bodo.
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

    py_output = test_fit(data[0])
    bodo_output = bodo.jit(distributed=["X"])(test_fit)(_get_dist_arg(data[0]))

    if with_centering:
        assert np.allclose(
            py_output.center_, bodo_output.center_, atol=1e-4, equal_nan=True
        )
    if with_scaling:
        assert np.allclose(
            py_output.scale_, bodo_output.scale_, atol=1e-4, equal_nan=True
        )

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
        test_transform, data, is_out_distributed=True, atol=1e-4, copy_input=True
    )

    def test_inverse_transform(X, X1):
        m = RobustScaler(
            with_centering=with_centering,
            with_scaling=with_scaling,
            quantile_range=quantile_range,
            unit_variance=unit_variance,
            copy=copy,
        )
        m = m.fit(X)
        X1_inverse_transformed = m.inverse_transform(X1)
        return X1_inverse_transformed

    check_func(
        test_inverse_transform,
        data,
        is_out_distributed=True,
        atol=1e-4,
        copy_input=True,
    )


@pytest.mark.parametrize(
    "bool_val",
    [True, pytest.param(False, marks=pytest.mark.slow)],
)
def test_robust_scaler_bool_attrs(bool_val, memory_leak_check):
    def impl_with_centering():
        m = RobustScaler(with_centering=bool_val)
        return m.with_centering

    def impl_with_scaling():
        m = RobustScaler(with_scaling=bool_val)
        return m.with_scaling

    def impl_unit_variance():
        m = RobustScaler(unit_variance=bool_val)
        return m.unit_variance

    def impl_copy():
        m = RobustScaler(copy=bool_val)
        return m.copy

    check_func(impl_with_centering, ())
    check_func(impl_with_scaling, ())
    check_func(impl_unit_variance, ())
    check_func(impl_copy, ())


def test_robust_scaler_array_and_quantile_range_attrs(memory_leak_check):

    data = gen_sklearn_scalers_random_data(20, 3)

    def impl_center_(X):
        m = RobustScaler()
        m.fit(X)
        return m.center_

    def impl_scale_(X):
        m = RobustScaler()
        m.fit(X)
        return m.scale_

    def impl_quantile_range():
        m = RobustScaler()
        return m.quantile_range

    check_func(impl_center_, (data,), is_out_distributed=False)
    check_func(impl_scale_, (data,), is_out_distributed=False)
    check_func(impl_quantile_range, (), is_out_distributed=False)
