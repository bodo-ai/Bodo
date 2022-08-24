# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Tests scikit-learn error checking for unsupported cases """

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import bodo
from bodo.tests.utils import _get_dist_arg
from bodo.utils.typing import BodoError


def generate_X_y_regression(n_samples, n_features):
    """Generate X (info) and y (labels) for regression dataset
    n_samples: number of instances (row)
    n_features: number of features (column)
    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        random_state=42,
    )
    return X, y


def generate_X_y_classification(n_samples, n_features, n_classes):
    """Generate X (info) and y (labels) for regression dataset
    n_samples: number of instances (row)
    n_features: number of features (column)
    n_classes: total number of labels in the dataset
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        random_state=42,
        n_informative=3,
    )
    return X, y


def generate_sample_weight(n_samples):
    """Generate sample_weight data"""
    return np.random.random_sample(size=n_samples)


@pytest.mark.slow
def test_sgdr_sample_weight(memory_leak_check):
    """Test BodoError is raised for SGDRegressor.fit():
    'sample_weight' unsupported argument for distributed case."""

    def impl(X_train, y_train, sample_weight):
        clf = SGDRegressor()
        clf.fit(X_train, y_train, sample_weight=sample_weight)
        return clf

    X, y = generate_X_y_regression(6, 4)
    sample_weight = generate_sample_weight(6)

    err_msg = "'sample_weight' is not supported."

    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(distributed=["X_train", "y_train", "sample_weight"])(impl)(
            _get_dist_arg(X), _get_dist_arg(y), _get_dist_arg(sample_weight)
        )


@pytest.mark.slow
def test_sgdr_coef_init(memory_leak_check):
    """Test BodoError is raised for SGDRegressor.fit():
    'coef_init' unsupported argument for distributed case."""

    def impl(X_train, y_train, coef_init):
        clf = SGDRegressor()
        clf.fit(X_train, y_train, coef_init=coef_init)
        return clf

    X, y = generate_X_y_regression(6, 4)
    coef_init = np.array([-1, -1])

    err_msg = "'coef_init' is not supported."

    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(distributed=["X_train", "y_train", "coef_init"])(impl)(
            _get_dist_arg(X), _get_dist_arg(y), _get_dist_arg(coef_init)
        )


@pytest.mark.slow
def test_sgdr_intercept_init(memory_leak_check):
    """Test BodoError is raised for SGDRegressor.fit():
    'intercept_init' unsupported argument for distributed case."""

    def impl(X_train, y_train, intercept_init):
        clf = SGDRegressor()
        clf.fit(X_train, y_train, intercept_init=intercept_init)
        return clf

    X, y = generate_X_y_regression(6, 4)
    intercept_init = np.array([-1])

    err_msg = "'intercept_init' is not supported."

    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(distributed=["X_train", "y_train", "intercept_init"])(impl)(
            _get_dist_arg(X), _get_dist_arg(y), _get_dist_arg(intercept_init)
        )


@pytest.mark.slow
def test_sgdc_sample_weight(memory_leak_check):
    """Test BodoError is raised for SGDClassifier.fit():
    'sample_weight' unsupported argument for distributed case."""

    def impl(X_train, y_train, sample_weight):
        clf = SGDClassifier()
        clf.fit(X_train, y_train, sample_weight=sample_weight)
        return clf

    X, y = generate_X_y_classification(8, 6, 3)
    sample_weight = generate_sample_weight(8)

    err_msg = "'sample_weight' is not supported."

    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(distributed=["X_train", "y_train", "sample_weight"])(impl)(
            _get_dist_arg(X), _get_dist_arg(y), _get_dist_arg(sample_weight)
        )


@pytest.mark.slow
def test_lr_sample_weight(memory_leak_check):
    """Test BodoError is raised for LogisticRegression.fit():
    'sample_weight' unsupported argument for distributed case."""

    def impl(X_train, y_train, sample_weight):
        clf = LogisticRegression()
        clf.fit(X_train, y_train, sample_weight=sample_weight)
        return clf

    X, y = generate_X_y_classification(8, 6, 2)
    sample_weight = generate_sample_weight(8)

    err_msg = "'sample_weight' is not supported."

    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(distributed=["X_train", "y_train", "sample_weight"])(impl)(
            _get_dist_arg(X), _get_dist_arg(y), _get_dist_arg(sample_weight)
        )


@pytest.mark.slow
def test_linearReg_sample_weight(memory_leak_check):
    """Test BodoError is raised for LinearRegression.fit():
    'sample_weight' unsupported argument for distributed case."""

    def impl(X_train, y_train, sample_weight):
        clf = LinearRegression()
        clf.fit(X_train, y_train, sample_weight=sample_weight)
        return clf

    X, y = generate_X_y_regression(6, 4)
    sample_weight = generate_sample_weight(6)

    err_msg = "'sample_weight' is not supported."

    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(distributed=["X_train", "y_train", "sample_weight"])(impl)(
            _get_dist_arg(X), _get_dist_arg(y), _get_dist_arg(sample_weight)
        )


@pytest.mark.slow
def test_lasso_sample_weight(memory_leak_check):
    """Test BodoError is raised for Lasso.fit():
    'sample_weight' unsupported argument for distributed case."""

    def impl(X_train, y_train, sample_weight):
        clf = Lasso()
        clf.fit(X_train, y_train, sample_weight=sample_weight)
        return clf

    X, y = generate_X_y_regression(6, 4)
    sample_weight = generate_sample_weight(6)

    err_msg = "'sample_weight' is not supported."

    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(distributed=["X_train", "y_train", "sample_weight"])(impl)(
            _get_dist_arg(X), _get_dist_arg(y), _get_dist_arg(sample_weight)
        )


@pytest.mark.slow
def test_ridge_sample_weight(memory_leak_check):
    """Test BodoError is raised for Ridge.fit():
    'sample_weight' unsupported argument for distributed case."""

    def impl(X_train, y_train, sample_weight):
        clf = Ridge()
        clf.fit(X_train, y_train, sample_weight=sample_weight)
        return clf

    X, y = generate_X_y_regression(6, 4)
    sample_weight = generate_sample_weight(6)

    err_msg = "'sample_weight' is not supported."

    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(distributed=["X_train", "y_train", "sample_weight"])(impl)(
            _get_dist_arg(X), _get_dist_arg(y), _get_dist_arg(sample_weight)
        )


@pytest.mark.slow
def test_lsvc_sample_weight(memory_leak_check):
    """Test BodoError is raised for LinearSVC.fit():
    'sample_weight' unsupported argument for distributed case."""

    def impl(X_train, y_train, sample_weight):
        clf = LinearSVC()
        clf.fit(X_train, y_train, sample_weight=sample_weight)
        return clf

    X, y = generate_X_y_classification(8, 6, 3)
    sample_weight = generate_sample_weight(8)

    err_msg = "'sample_weight' is not supported."

    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(distributed=["X_train", "y_train", "sample_weight"])(impl)(
            _get_dist_arg(X), _get_dist_arg(y), _get_dist_arg(sample_weight)
        )


@pytest.mark.slow
def test_standardscaler_sample_weight(memory_leak_check):
    """Test BodoError is raised for StandardScaler.fit():
    'sample_weight' unsupported argument for distributed case."""

    def impl(X_train, sample_weight):
        clf = StandardScaler()
        clf.fit(X_train, sample_weight=sample_weight)
        return clf

    X, _ = generate_X_y_classification(8, 6, 3)
    sample_weight = generate_sample_weight(8)

    err_msg = "'sample_weight' is not supported."

    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(distributed=["X_train", "sample_weight"])(impl)(
            _get_dist_arg(X), _get_dist_arg(sample_weight)
        )


@pytest.mark.slow
def test_randomforestclassifier_sample_weight(memory_leak_check):
    """Test BodoError is raised for RandomForestClassifier.fit():
    'sample_weight' unsupported argument for distributed case."""

    def impl(X_train, y_train, sample_weight):
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train, sample_weight=sample_weight)
        return clf

    X, y = generate_X_y_classification(8, 6, 3)
    sample_weight = generate_sample_weight(8)

    err_msg = "'sample_weight' is not supported."

    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(distributed=["X_train", "y_train", "sample_weight"])(impl)(
            _get_dist_arg(X), _get_dist_arg(y), _get_dist_arg(sample_weight)
        )


@pytest.mark.slow
def test_randomforestregressor_sample_weight(memory_leak_check):
    """Test BodoError is raised for RandomForestRegressor.fit():
    'sample_weight' unsupported argument for distributed case."""

    def impl(X_train, y_train, sample_weight):
        clf = RandomForestRegressor()
        clf.fit(X_train, y_train, sample_weight=sample_weight)
        return clf

    X, y = generate_X_y_regression(8, 6)
    sample_weight = generate_sample_weight(8)

    err_msg = "'sample_weight' is not supported."

    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(distributed=["X_train", "y_train", "sample_weight"])(impl)(
            _get_dist_arg(X), _get_dist_arg(y), _get_dist_arg(sample_weight)
        )


@pytest.mark.parametrize("value", [10, "wrong"])
@pytest.mark.slow
def test_count_vectorizer_min_max_df(value, memory_leak_check):
    """Test BodoError is raised for CountVectorizer():
    'min_df', 'max_df' unsupported arguments."""

    cat_in_the_hat_docs = [
        "One Cent, Two Cents, Old Cent, New Cent: All About Money (Cat in the Hat's Learning Library",
        "Inside Your Outside: All About the Human Body (Cat in the Hat's Learning Library)",
        "Oh, The Things You Can Do That Are Good for You: All About Staying Healthy (Cat in the Hat's Learning Library)",
        "On Beyond Bugs: All About Insects (Cat in the Hat's Learning Library)",
        "There's No Place Like Space: All About Our Solar System (Cat in the Hat's Learning Library)",
    ]
    df = pd.DataFrame({"A": cat_in_the_hat_docs})

    def impl(df, value):
        v = CountVectorizer(min_df=value)
        return v

    with pytest.raises(BodoError, match="'min_df' is not supported"):
        bodo.jit(impl)(df, value)

    def impl2(df, value):
        v = CountVectorizer(max_df=value)
        return v

    with pytest.raises(BodoError, match="'max_df' is not supported"):
        bodo.jit(impl2)(df, value)
