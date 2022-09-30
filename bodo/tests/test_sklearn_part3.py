# Copyright (C) 2022 Bodo Inc. All rights reserved.
""" Test miscellaneous supported sklearn models and methods
    Currently this file tests:
    MultinomialNB, OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
"""

import random

import numpy as np
import pandas as pd
import pytest
import scipy
from scipy.sparse import csr_matrix, issparse
from scipy.special import comb
from sklearn.model_selection import LeavePOut
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import (
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.utils._testing import assert_array_equal

import bodo
from bodo.tests.utils import _get_dist_arg, check_func
from bodo.utils.typing import BodoError

# ----------------------- LeavePOut -----------------------------


@pytest.mark.parametrize(
    "X, y, groups",
    [
        (
            np.arange(100).reshape((20, 5)).astype(np.int64),
            np.arange(20).astype(np.int64),
            None,
        ),
        (
            np.arange(110).reshape((22, 5)).astype(np.float64),
            np.arange(22).astype(np.float64),
            None,
        ),
        (
            np.arange(100).reshape((20, 5)).astype(np.int64),
            np.arange(20).astype(np.int64),
            np.array([0, 1] * 10).astype(np.int64),
        ),
    ],
)
@pytest.mark.parametrize("p", [1, 2, 5])
def test_leave_p_out(X, y, groups, p, memory_leak_check):
    """Test sklearn.model_selection.LeavePOut's split method."""
    # Compute expected number of splits
    n_splits = int(comb(len(X), p, exact=True))

    def test_split(X, y, groups, p):
        m = LeavePOut(p=p)
        out = m.split(X, y, groups)
        return out

    dist_impl = bodo.jit(distributed=["X", "y", "groups"])(test_split)
    rep_impl = bodo.jit(replicated=True)(test_split)

    # Test that all indices are returned in distributed split
    test_idxs_dist = []
    for train_idxs, test_idxs in dist_impl(
        _get_dist_arg(X), _get_dist_arg(y), groups, p
    ):
        train_idxs = bodo.allgatherv(train_idxs)
        test_idxs = bodo.allgatherv(test_idxs)
        test_idxs_dist.append(test_idxs)

        idxs = np.sort(np.concatenate((train_idxs, test_idxs), axis=0), axis=0)
        assert_array_equal(idxs, np.arange(X.shape[0]))

    assert n_splits == len(test_idxs_dist)

    # Test that all indices are returned in replicated split
    test_idxs_rep = []
    for train_idxs, test_idxs in rep_impl(X, y, groups, p):
        test_idxs_rep.append(test_idxs)

        idxs = np.sort(np.concatenate((train_idxs, test_idxs), axis=0), axis=0)
        assert_array_equal(idxs, np.arange(X.shape[0]))

    assert n_splits == len(test_idxs_rep)

    # Test that bodo's folds are equivalent to sklearn's
    test_idxs_sklearn = []
    for train_idxs, test_idxs in test_split(X, y, groups, p):
        test_idxs_sklearn.append(test_idxs)

    for dist_idxs, rep_idxs, sklearn_idxs in zip(
        test_idxs_dist, test_idxs_rep, test_idxs_sklearn
    ):
        assert_array_equal(dist_idxs, sklearn_idxs)
        assert_array_equal(rep_idxs, sklearn_idxs)

    # Test that get_n_splits returns the correct number of folds
    def test_n_splits(X):
        m = LeavePOut(p=p)
        out = m.get_n_splits(X)
        return out

    check_func(test_n_splits, (X,))


def test_leave_p_out_error(memory_leak_check):
    """Test error handling of KFold.split()"""

    def test_split(X, p):
        m = LeavePOut(p=p)
        out = m.split(X)
        return out

    dist_impl = bodo.jit(distributed=["X"])(test_split)

    # X has too few items for p=3
    X = np.array([[1, 2], [3, 4], [5, 6]])
    error_str = "p=3 must be strictly less than the number of samples"
    with pytest.raises(ValueError, match=error_str):
        for train_idxs, test_idxs in dist_impl(_get_dist_arg(X), 3):
            pass


# ---------------------- LabelEncoder -----------------------


@pytest.mark.parametrize(
    "values, classes ",
    [
        (
            np.array([2, 1, 3, 1, 3], dtype="int64"),
            np.array([1, 2, 3], dtype="int64"),
        ),
        (
            np.array([2.2, 1.1, 3.3, 1.1, 3.3], dtype="float64"),
            np.array([1.1, 2.2, 3.3], dtype="float64"),
        ),
        (
            np.array(["b", "a", "c", "a", "c"], dtype=object),
            np.array(["a", "b", "c"], dtype=object),
        ),
        (
            np.array(["bb", "aa", "cc", "aa", "cc"], dtype=object),
            np.array(["aa", "bb", "cc"], dtype=object),
        ),
    ],
)
def test_label_encoder(values, classes, memory_leak_check):
    """Test LabelEncoder's transform, fit_transform and inverse_transform methods.
    Taken from here (https://github.com/scikit-learn/scikit-learn/blob/8ea176ae0ca535cdbfad7413322bbc3e54979e4d/sklearn/preprocessing/tests/test_label.py#L193)
    """

    def test_fit(values):
        le = LabelEncoder()
        le.fit(values)
        return le

    le = bodo.jit(distributed=["values"])(test_fit)(_get_dist_arg(values))
    assert_array_equal(le.classes_, classes)

    def test_transform(values):
        le = LabelEncoder()
        le.fit(values)
        result = le.transform(values)
        return result

    check_func(test_transform, (values,))

    def test_fit_transform(values):
        le = LabelEncoder()
        result = le.fit_transform(values)
        return result

    check_func(test_fit_transform, (values,))


def test_naive_mnnb_csr(memory_leak_check):
    """Test csr matrix with MultinomialNB
    Taken from here (https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tests/test_naive_bayes.py#L461)
    """

    def test_mnnb(X, y2):
        clf = MultinomialNB()
        clf.fit(X, y2)
        y_pred = clf.predict(X)
        return y_pred

    rng = np.random.RandomState(42)

    # Data is 6 random integer points in a 100 dimensional space classified to
    # three classes.
    X2 = rng.randint(5, size=(6, 100))
    y2 = np.array([1, 1, 2, 2, 3, 3])
    X = scipy.sparse.csr_matrix(X2)
    y_pred = bodo.jit(distributed=["X", "y2", "y_pred"])(test_mnnb)(
        _get_dist_arg(X), _get_dist_arg(y2)
    )
    y_pred = bodo.allgatherv(y_pred)
    assert_array_equal(y_pred, y2)

    check_func(test_mnnb, (X, y2))


# ----------------------- OneHotEncoder -----------------------


@pytest.mark.parametrize(
    "X",
    [
        np.array([["a", "a"], ["ac", "b"]] * 5 + [["b", "a"]], dtype=object),
        np.array([[0, 0], [2, 1]] * 5 + [[1, 0]], dtype=np.int64),
        pd.DataFrame({"A": ["a", "b", "a"] * 5 + ["c"], "B": [1, np.nan, 3] * 5 + [2]}),
    ],
)
def test_one_hot_encoder(X):
    """
    Test OneHotEncoder's fit and transform methods, as well as the categories_
    attribute.
    """

    def test_fit(X):
        m = OneHotEncoder(sparse=False)
        m.fit(X)
        result = m.categories_
        return result

    check_func(test_fit, (X,), is_out_distributed=False)

    def test_transform(X):
        m = OneHotEncoder(sparse=False)
        m.fit(X)
        result = m.transform(X)
        return result

    check_func(test_transform, (X,))


def test_one_hot_encoder_fit_attributes():
    """
    Test to make sure that the fit function results in an object with the
    correct attributes to conform with version 1.1.*.
    """

    def get_model_fit(X):
        m = OneHotEncoder(sparse=False)
        m.fit(X)
        return m

    X = np.array([["a", "a"], ["ac", "b"]] * 5 + [["b", "a"]], dtype=object)
    python_m = get_model_fit(X)
    bodo_m = bodo.jit()(get_model_fit)(X)

    for py_attr_name in dir(python_m):
        if (
            py_attr_name.startswith("_")
            and not py_attr_name.startswith("__")
            and not callable(getattr(python_m, py_attr_name))
        ):
            assert hasattr(bodo_m, py_attr_name), f"{py_attr_name} not found in bodo_m"
            assert getattr(python_m, py_attr_name) == getattr(
                bodo_m, py_attr_name
            ), f"Attribute {py_attr_name} does not match between python and bodo"


@pytest.mark.parametrize(
    "X, categories",
    [
        (
            np.array([[0, 0], [2, 1]] * 5 + [[1, 0]], dtype=np.int64),
            [[0, 2], [0, 1]],
        ),
        (
            pd.DataFrame(
                {"A": ["a", "b", "a"] * 5 + ["c"], "B": ["1", "1", "3"] * 5 + ["2"]}
            ),
            [["a", "b"], ["1", "2"]],
        ),
        (
            # Unknown categories only on last rank
            pd.DataFrame(
                {"A": ["a", "b", "a"] * 5 + ["c"], "B": ["1", "1", "2"] * 5 + ["3"]}
            ),
            [["a", "b"], ["1", "2"]],
        ),
    ],
)
def test_one_hot_encoder_categories(X, categories):
    """Test OneHotEncoder's categories and handle_unknown parameters."""

    def test_transform_error(X):
        m = OneHotEncoder(sparse=False, categories=categories, handle_unknown="error")
        m.fit(X)
        result = m.transform(X)
        return result

    dist_impl_error = bodo.jit(distributed=["X"])(test_transform_error)
    rep_impl_error = bodo.jit(replicated=True)(test_transform_error)

    error_str = "Found unknown categories"
    with pytest.raises(ValueError, match=error_str):
        dist_impl_error(X)
    with pytest.raises(ValueError, match=error_str):
        rep_impl_error(X)

    def test_transform_ignore(X):
        m = OneHotEncoder(sparse=False, categories=categories, handle_unknown="ignore")
        m.fit(X)
        result = m.transform(X)
        return result

    check_func(test_transform_ignore, (X,))


@pytest.mark.parametrize(
    "X, drop",
    [
        (
            pd.DataFrame(
                {"A": ["a", "b", "a"] * 5 + ["c"], "B": ["1", "1", "3"] * 5 + ["2"]}
            ),
            ["c", "3"],
        ),
        (
            np.array([[0, 0], [2, 1]] * 5 + [[1, 0]], dtype=np.int64),
            [2, 1],
        ),
        (
            np.array([["a", "a"], ["ac", "b"]] * 5 + [["b", "a"]], dtype=object),
            ["ac", "a"],
        ),
    ],
)
def test_one_hot_encoder_attributes(X, drop):
    """Test OneHotEncoder's drop_idx_ and n_features_in_ attributes."""

    def test_drop_idx_1(X):
        m = OneHotEncoder(sparse=False, dtype=np.float64)
        m.fit(X)
        result = m.drop_idx_
        return result

    def test_drop_idx_2(X):
        m = OneHotEncoder(sparse=False, drop="first")
        m.fit(X)
        result = m.drop_idx_
        return result

    def test_drop_idx_3(X):
        m = OneHotEncoder(sparse=False, drop="if_binary")
        m.fit(X)
        result = m.drop_idx_
        return result

    def test_drop_idx_4(X):
        m = OneHotEncoder(sparse=False, drop=drop)
        m.fit(X)
        result = m.drop_idx_
        return result

    check_func(test_drop_idx_1, (X,), is_out_distributed=False)
    check_func(test_drop_idx_2, (X,), is_out_distributed=False)
    check_func(test_drop_idx_3, (X,), is_out_distributed=False)
    check_func(test_drop_idx_4, (X,), is_out_distributed=False)

    def test_n_features_in_(X):
        m = OneHotEncoder(sparse=False)
        m.fit(X)
        result = m.n_features_in_
        return result

    check_func(test_n_features_in_, (X,))


def test_one_hot_encoder_unsupported():
    """
    Test OneHotEncoder's unsupported arguments.
    """

    def impl1():
        m = OneHotEncoder()
        return m

    def impl2():
        m = OneHotEncoder(sparse=True)
        return m

    def impl3():
        m = OneHotEncoder(dtype=np.float32)
        return m

    def impl4():
        m = OneHotEncoder(sparse=False, dtype=np.int64)
        return m

    err_msg_1 = "sparse parameter only supports default value False"
    err_msg_2 = "sparse parameter only supports default value False"
    err_msg_3 = "sparse parameter only supports default value False"
    err_msg_4 = "dtype parameter only supports default value float64"

    with pytest.raises(BodoError, match=err_msg_1):
        bodo.jit()(impl1)()

    with pytest.raises(BodoError, match=err_msg_2):
        bodo.jit()(impl2)()

    with pytest.raises(BodoError, match=err_msg_3):
        bodo.jit()(impl3)()

    with pytest.raises(BodoError, match=err_msg_4):
        bodo.jit()(impl4)()


# ---------------------StandardScaler Tests--------------------


def gen_sklearn_scalers_random_data(
    num_samples, num_features, frac_Nans=0.0, scale=1.0
):
    """
    Generate random data of shape (num_samples, num_features), where each number
    is in the range (-scale, scale), and frac_Nans fraction of entries are np.nan.
    """
    random.seed(5)
    np.random.seed(5)
    X = np.random.rand(num_samples, num_features)
    X = 2 * X - 1
    X = X * scale
    mask = np.random.choice([1, 0], X.shape, p=[frac_Nans, 1 - frac_Nans]).astype(bool)
    X[mask] = np.nan
    return X


def gen_sklearn_scalers_edge_case(
    num_samples, num_features, frac_Nans=0.0, scale=1.0, dim_to_nan=0
):
    """
    Helper function to generate random data for testing an edge case of sklearn scalers.
    In this edge case, along a specified dimension (dim_to_nan), all but one entry is
    set to np.nan.
    """
    X = gen_sklearn_scalers_random_data(
        num_samples, num_features, frac_Nans=frac_Nans, scale=scale
    )
    X[1:, dim_to_nan] = np.nan
    return X


@pytest.mark.parametrize(
    "data",
    [
        (
            gen_sklearn_scalers_random_data(20, 3),
            gen_sklearn_scalers_random_data(100, 3),
        ),
        (
            gen_sklearn_scalers_random_data(15, 5, 0.2, 4),
            gen_sklearn_scalers_random_data(60, 5, 0.5, 2),
        ),
        (
            gen_sklearn_scalers_random_data(20, 1, 0, 2),
            gen_sklearn_scalers_random_data(50, 1, 0.1, 1),
        ),
        (
            gen_sklearn_scalers_random_data(20, 1, 0.2, 5),
            gen_sklearn_scalers_random_data(50, 1, 0.1, 2),
        ),
        (
            gen_sklearn_scalers_edge_case(20, 5, 0, 4, 2),
            gen_sklearn_scalers_random_data(40, 5, 0.1, 3),
        ),
        (
            gen_sklearn_scalers_random_data(20, 3),
            gen_sklearn_scalers_random_data(100, 3),
        ),
        (
            csr_matrix(gen_sklearn_scalers_random_data(20, 3)),
            csr_matrix(gen_sklearn_scalers_random_data(100, 3)),
        ),
        (
            csr_matrix(gen_sklearn_scalers_random_data(15, 5, 0.2, 4)),
            csr_matrix(gen_sklearn_scalers_random_data(60, 5, 0.5, 2)),
        ),
    ],
)
@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("with_mean", [True, False])
@pytest.mark.parametrize("with_std", [True, False])
def test_standard_scaler(data, copy, with_mean, with_std, memory_leak_check):
    """
    Tests for sklearn.preprocessing.StandardScaler implementation in Bodo.
    """
    if issparse(data[0]) and (with_mean or with_std):
        # Skip sparse test cases that would result in an error, since
        # memory_leak_check behaves inconsistently with numba errors. These are
        # tested in test_standard_scaler_sparse_error without memory_leak_check.
        return

    def test_fit(X):
        m = StandardScaler(with_mean=with_mean, with_std=with_std, copy=copy)
        m = m.fit(X)
        return m

    py_output = test_fit(data[0])
    bodo_output = bodo.jit(distributed=["X"])(test_fit)(_get_dist_arg(data[0]))

    assert np.array_equal(py_output.n_samples_seen_, bodo_output.n_samples_seen_)
    if with_mean or with_std:
        assert np.allclose(
            py_output.mean_, bodo_output.mean_, atol=1e-4, equal_nan=True
        )
    if with_std:
        assert np.allclose(py_output.var_, bodo_output.var_, atol=1e-4, equal_nan=True)
        assert np.allclose(
            py_output.scale_, bodo_output.scale_, atol=1e-4, equal_nan=True
        )

    def test_transform(X, X1):
        m = StandardScaler(with_mean=with_mean, with_std=with_std, copy=copy)
        m = m.fit(X)
        X1_transformed = m.transform(X1)
        return X1_transformed

    check_func(
        test_transform, data, is_out_distributed=True, atol=1e-4, copy_input=True
    )

    def test_inverse_transform(X, X1):
        m = StandardScaler(with_mean=with_mean, with_std=with_std, copy=copy)
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
    "data",
    [
        (
            csr_matrix(gen_sklearn_scalers_random_data(20, 3)),
            csr_matrix(gen_sklearn_scalers_random_data(100, 3)),
        ),
        (
            csr_matrix(gen_sklearn_scalers_random_data(15, 5, 0.2, 4)),
            csr_matrix(gen_sklearn_scalers_random_data(60, 5, 0.5, 2)),
        ),
    ],
)
@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("with_mean", [True, False])
@pytest.mark.parametrize("with_std", [True, False])
def test_standard_scaler_sparse_error(data, copy, with_mean, with_std):
    def test_fit(X):
        m = StandardScaler(with_mean=with_mean, with_std=with_std, copy=copy)
        m = m.fit(X)
        return m

    if with_mean:
        # Test that sparse matrices cannot be centered
        error_str = "Cannot center sparse matrices: pass `with_mean=False` instead."
        with pytest.raises(ValueError, match=error_str):
            py_output = test_fit(data[0])
        with pytest.raises(ValueError, match=error_str):
            bodo_output = bodo.jit(distributed=["X"])(test_fit)(_get_dist_arg(data[0]))

    if with_std:
        # In bodo's implementation, `with_std=True` causes an underlying call to fit
        # on each rank where `with_mean=True`, which also triggers the ValueError
        # above when X is a sparse matrix
        error_str = "Cannot center sparse matrices: pass `with_mean=False` instead."
        with pytest.raises(ValueError, match=error_str):
            bodo_output = bodo.jit(distributed=["X"])(test_fit)(_get_dist_arg(data[0]))


# ---------------------MaxAbsScaler Tests--------------------


@pytest.mark.parametrize(
    "data",
    [
        # Test one with numpy array, one with df, and one with sparse matrix
        (
            gen_sklearn_scalers_random_data(15, 5, 0.2, 4),
            gen_sklearn_scalers_random_data(60, 5, 0.5, 2),
        ),
        (
            pd.DataFrame(gen_sklearn_scalers_random_data(20, 3)),
            gen_sklearn_scalers_random_data(100, 3),
        ),
        (
            csr_matrix(gen_sklearn_scalers_random_data(20, 3)),
            csr_matrix(gen_sklearn_scalers_random_data(100, 3)),
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
        pytest.param(
            (
                csr_matrix(gen_sklearn_scalers_random_data(15, 5, 0.2, 4)),
                csr_matrix(gen_sklearn_scalers_random_data(60, 5, 0.5, 2)),
            ),
            marks=pytest.mark.slow,
        ),
    ],
)
@pytest.mark.parametrize("copy", [True, False])
def test_max_abs_scaler(data, copy, memory_leak_check):
    """
    Tests for sklearn.preprocessing.MaxAbsScaler implementation in Bodo.
    """

    def test_fit(X):
        m = MaxAbsScaler(copy=copy)
        m = m.fit(X)
        return m

    py_output = test_fit(data[0])
    bodo_output = bodo.jit(distributed=["X"])(test_fit)(_get_dist_arg(data[0]))

    assert np.array_equal(py_output.n_samples_seen_, bodo_output.n_samples_seen_)
    assert np.allclose(
        py_output.max_abs_, bodo_output.max_abs_, atol=1e-4, equal_nan=True
    )

    def test_partial_fit(X, X1):
        m = MaxAbsScaler(copy=copy)
        m = m.partial_fit(X)
        m = m.partial_fit(X1)
        return m

    py_output = test_partial_fit(data[0], data[1])
    bodo_output = bodo.jit(distributed=["X", "X1"])(test_partial_fit)(
        _get_dist_arg(data[0]), _get_dist_arg(data[1])
    )

    assert np.array_equal(py_output.n_samples_seen_, bodo_output.n_samples_seen_)
    assert np.allclose(
        py_output.max_abs_, bodo_output.max_abs_, atol=1e-4, equal_nan=True
    )

    def test_transform(X, X1):
        m = MaxAbsScaler(copy=copy)
        m = m.fit(X)
        X1_transformed = m.transform(X1)
        return X1_transformed

    check_func(
        test_transform, data, is_out_distributed=True, atol=1e-4, copy_input=True
    )

    def test_inverse_transform(X, X1):
        m = MaxAbsScaler(copy=copy)
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


def test_max_abs_scaler_attrs(memory_leak_check):
    """
    Tests for attributes of sklearn.preprocessing.MaxAbsScaler in Bodo.
    """

    data = gen_sklearn_scalers_random_data(20, 5)

    def impl_scale_(X):
        m = MaxAbsScaler()
        m.fit(X)
        return m.scale_

    def impl_max_abs_(X):
        m = MaxAbsScaler()
        m.fit(X)
        return m.max_abs_

    check_func(impl_scale_, (data,), is_out_distributed=False)
    check_func(impl_max_abs_, (data,), is_out_distributed=False)


# ---------------------MinMaxScaler Tests--------------------


@pytest.mark.parametrize(
    "data",
    [
        (
            gen_sklearn_scalers_random_data(20, 3),
            gen_sklearn_scalers_random_data(100, 3),
        ),
        (
            gen_sklearn_scalers_random_data(15, 5, 0.2, 4),
            gen_sklearn_scalers_random_data(60, 5, 0.5, 2),
        ),
        (
            gen_sklearn_scalers_random_data(20, 1, 0, 2),
            gen_sklearn_scalers_random_data(50, 1, 0.1, 1),
        ),
        (
            gen_sklearn_scalers_random_data(20, 1, 0.2, 5),
            gen_sklearn_scalers_random_data(50, 1, 0.1, 2),
        ),
        (
            gen_sklearn_scalers_edge_case(20, 5, 0, 4, 2),
            gen_sklearn_scalers_random_data(40, 5, 0.1, 3),
        ),
    ],
)
@pytest.mark.parametrize("feature_range", [(0, 1), (-2, 2)])
@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("clip", [True, False])
def test_minmax_scaler(data, feature_range, copy, clip, memory_leak_check):
    """
    Tests for sklearn.preprocessing.MinMaxScaler implementation in Bodo.
    """

    def test_fit(X):
        m = MinMaxScaler(feature_range=feature_range, copy=copy, clip=clip)
        m = m.fit(X)
        return m

    py_output = test_fit(data[0])
    bodo_output = bodo.jit(distributed=["X"])(test_fit)(_get_dist_arg(data[0]))

    assert py_output.n_samples_seen_ == bodo_output.n_samples_seen_
    assert np.array_equal(py_output.min_, bodo_output.min_, equal_nan=True)
    assert np.array_equal(py_output.scale_, bodo_output.scale_, equal_nan=True)
    assert np.array_equal(py_output.data_min_, bodo_output.data_min_, equal_nan=True)
    assert np.array_equal(py_output.data_max_, bodo_output.data_max_, equal_nan=True)
    assert np.array_equal(
        py_output.data_range_, bodo_output.data_range_, equal_nan=True
    )

    def test_transform(X, X1):
        m = MinMaxScaler(feature_range=feature_range, copy=copy, clip=clip)
        m = m.fit(X)
        X1_transformed = m.transform(X1)
        return X1_transformed

    check_func(
        test_transform, data, is_out_distributed=True, atol=1e-8, copy_input=True
    )

    def test_inverse_transform(X, X1):
        m = MinMaxScaler(feature_range=feature_range, copy=copy, clip=clip)
        m = m.fit(X)
        X1_inverse_transformed = m.inverse_transform(X1)
        return X1_inverse_transformed

    check_func(
        test_inverse_transform,
        data,
        is_out_distributed=True,
        atol=1e-8,
        copy_input=True,
    )
