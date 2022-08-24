# Copyright (C) 2022 Bodo Inc. All rights reserved.
""" Test miscellaneous supported sklearn models and methods
    Currently this file tests:
    Robust Scaler
    The Robust Scaler tests need to be done across two files (this and test_sklearn_part4)
    due to the large ammount of test_robust_scalar tests,
    which can cause OOM issues on nightly due to numba caching artifacts.
"""

import pytest
from sklearn.preprocessing import RobustScaler

from bodo.tests.test_sklearn_part3 import gen_sklearn_scalers_random_data
from bodo.tests.test_sklearn_part4 import robust_scalar_data  # pragma: no cover
from bodo.tests.utils import check_func

# ---------------------RobustScaler Tests, part 2--------------------


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
def test_robust_scaler_inverse_transform(
    robust_scalar_data,
    with_centering,
    with_scaling,
    quantile_range,
    unit_variance,
    copy,
    memory_leak_check,
):
    """
    Tests for sklearn.preprocessing.RobustScaler.inverse_transform implementation in Bodo.
    """

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
        robust_scalar_data,
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
