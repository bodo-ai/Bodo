# Copyright (C) 2019 Bodo Inc. All rights reserved.
""" Test miscellaneous supported sklearn models and methods
    Currently this file tests:
    train_test_split, MultinomialNB, LinearSVC, 
    LabelEncoder, MinMaxScaler, StandardScaler
"""

import random

import numpy as np
import pandas as pd
import pytest
import scipy
from sklearn import datasets
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.validation import check_random_state

import bodo
from bodo.tests.utils import _get_dist_arg, check_func
from bodo.utils.typing import BodoError


# --------------------Multinomial Naive Bayes Tests-----------------#
def test_multinomial_nb():
    """Test Multinomial Naive Bayes
    Taken from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tests/test_naive_bayes.py#L442
    """
    rng = np.random.RandomState(0)
    X = rng.randint(5, size=(6, 100))
    y = np.array([1, 1, 2, 2, 3, 3])

    def impl_fit(X, y):
        clf = MultinomialNB()
        clf.fit(X, y)
        return clf

    clf = bodo.jit(distributed=["X", "y"])(impl_fit)(
        _get_dist_arg(np.array(X)),
        _get_dist_arg(np.array(y)),
    )
    # class_log_prior_: Smoothed empirical log probability for each class.
    # It's computation is replicated by all ranks
    np.testing.assert_array_almost_equal(
        np.log(np.array([2, 2, 2]) / 6.0), clf.class_log_prior_, 8
    )

    def impl_predict(X, y):
        clf = MultinomialNB()
        y_pred = clf.fit(X, y).predict(X)
        return y_pred

    check_func(
        impl_predict,
        (X, y),
        py_output=y,
        is_out_distributed=True,
    )

    X = np.array([[1, 0, 0], [1, 1, 0]])
    y = np.array([0, 1])

    def test_alpha_vector(X, y):
        # Setting alpha=np.array with same length
        # as number of features should be fine
        alpha = np.array([1, 2, 1])
        nb = MultinomialNB(alpha=alpha)
        nb.fit(X, y)
        return nb

    # Test feature probabilities uses pseudo-counts (alpha)
    nb = bodo.jit(distributed=["X", "y"])(test_alpha_vector)(
        _get_dist_arg(np.array(X)),
        _get_dist_arg(np.array(y)),
    )
    feature_prob = np.array([[2 / 5, 2 / 5, 1 / 5], [1 / 3, 1 / 2, 1 / 6]])
    # feature_log_prob_: Empirical log probability of features given a class, P(x_i|y).
    # Computation is distributed and then gathered and replicated in all ranks.
    np.testing.assert_array_almost_equal(nb.feature_log_prob_, np.log(feature_prob))

    # Test dataframe.
    train = pd.DataFrame(
        {"A": range(20), "B": range(100, 120), "C": range(20, 40), "D": range(40, 60)}
    )
    train_labels = pd.Series(range(20))

    check_func(impl_predict, (train, train_labels))


def test_multinomial_nb_score():
    rng = np.random.RandomState(0)
    X = rng.randint(5, size=(6, 100))
    y = np.array([1, 1, 2, 2, 3, 3])

    def impl(X, y):
        clf = MultinomialNB()
        clf.fit(X, y)
        score = clf.score(X, y)
        return score

    check_func(impl, (X, y))


# --------------------Linear SVC -----------------#
# also load the iris dataset
# and randomly permute it
iris = datasets.load_iris()
rng = check_random_state(0)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]


def test_svm_linear_svc(memory_leak_check):
    """
    Test LinearSVC
    """
    # Toy dataset where features correspond directly to labels.
    X = iris.data
    y = iris.target
    classes = [0, 1, 2]

    def impl_fit(X, y):
        clf = LinearSVC()
        clf.fit(X, y)
        return clf

    clf = bodo.jit(distributed=["X", "y"])(impl_fit)(
        _get_dist_arg(X),
        _get_dist_arg(y),
    )
    np.testing.assert_array_equal(clf.classes_, classes)

    def impl_pred(X, y):
        clf = LinearSVC()
        clf.fit(X, y)
        y_pred = clf.predict(X)
        score = precision_score(y, y_pred, average="micro")
        return score

    bodo_score_result = bodo.jit(distributed=["X", "y"])(impl_pred)(
        _get_dist_arg(X),
        _get_dist_arg(y),
    )

    sklearn_score_result = impl_pred(X, y)
    np.allclose(sklearn_score_result, bodo_score_result, atol=0.1)

    def impl_score(X, y):
        clf = LinearSVC()
        clf.fit(X, y)
        return clf.score(X, y)

    bodo_score_result = bodo.jit(distributed=["X", "y"])(impl_score)(
        _get_dist_arg(X),
        _get_dist_arg(y),
    )

    sklearn_score_result = impl_score(X, y)
    np.allclose(sklearn_score_result, bodo_score_result, atol=0.1)


# ------------------------train_test_split------------------------
def test_train_test_split(memory_leak_check):
    def impl_shuffle(X, y):
        # simple test
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        return X_train, X_test, y_train, y_test

    def impl_no_shuffle(X, y):
        # simple test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, train_size=0.6, shuffle=False
        )
        return X_train, X_test, y_train, y_test

    X = np.arange(100).reshape((10, 10))
    y = np.arange(10)

    # Test shuffle with numpy arrays
    X_train, X_test, y_train, y_test = bodo.jit(
        distributed=["X", "y", "X_train", "X_test", "y_train", "y_test"], cache=True
    )(impl_shuffle)(
        _get_dist_arg(X),
        _get_dist_arg(y),
    )
    # Test correspondence of X and y
    assert_array_equal(X_train[:, 0], y_train * 10)
    assert_array_equal(X_test[:, 0], y_test * 10)

    bodo_X_train = bodo.allgatherv(X_train)
    bodo_X_test = bodo.allgatherv(X_test)
    bodo_X = np.sort(np.concatenate((bodo_X_train, bodo_X_test), axis=0), axis=0)
    assert_array_equal(bodo_X, X)

    # Test without shuffle with numpy arrays
    X_train, X_test, y_train, y_test = bodo.jit(
        distributed=["X", "y", "X_train", "X_test", "y_train", "y_test"], cache=True
    )(impl_no_shuffle)(
        _get_dist_arg(X),
        _get_dist_arg(y),
    )
    # Test correspondence of X and y
    assert_array_equal(X_train[:, 0], y_train * 10)
    assert_array_equal(X_test[:, 0], y_test * 10)

    bodo_X_train = bodo.allgatherv(X_train)
    bodo_X_test = bodo.allgatherv(X_test)
    bodo_X = np.sort(np.concatenate((bodo_X_train, bodo_X_test), axis=0), axis=0)
    assert_array_equal(bodo_X, X)

    # Test replicated shuffle with numpy arrays
    X_train, X_test, y_train, y_test = bodo.jit(impl_shuffle)(X, y)
    # Test correspondence of X and y
    assert_array_equal(X_train[:, 0], y_train * 10)
    assert_array_equal(X_test[:, 0], y_test * 10)


@pytest.mark.parametrize(
    "train_size, test_size", [(0.6, None), (None, 0.3), (None, None), (0.7, 0.3)]
)
def test_train_test_split_df(train_size, test_size, memory_leak_check):
    """ Test train_test_split with DataFrame dataset and train_size/test_size variation"""

    def impl_shuffle(X, y, train_size, test_size):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, test_size=test_size
        )
        return X_train, X_test, y_train, y_test

    # Test replicated shuffle with DataFrame
    train = pd.DataFrame({"A": range(20), "B": range(100, 120)})
    train_labels = pd.Series(range(20))
    X_train, X_test, y_train, y_test = bodo.jit(impl_shuffle)(
        train, train_labels, train_size, test_size
    )
    assert_array_equal(X_train.iloc[:, 0], y_train)
    assert_array_equal(X_test.iloc[:, 0], y_test)

    # Test when labels is series but data is array
    train = np.arange(100).reshape((10, 10))
    train_labels = pd.Series(range(10))

    # Replicated
    X_train, X_test, y_train, y_test = bodo.jit(impl_shuffle)(
        train, train_labels, train_size, test_size
    )
    assert_array_equal(X_train[:, 0], y_train * 10)
    assert_array_equal(X_test[:, 0], y_test * 10)

    # Distributed
    X_train, X_test, y_train, y_test = bodo.jit(
        distributed=["X", "y", "X_train", "X_test", "y_train", "y_test"], cache=True
    )(impl_shuffle)(
        _get_dist_arg(train), _get_dist_arg(train_labels), train_size, test_size
    )
    assert_array_equal(X_train[:, 0], y_train * 10)
    assert_array_equal(X_test[:, 0], y_test * 10)
    bodo_X_train = bodo.allgatherv(X_train)
    bodo_X_test = bodo.allgatherv(X_test)
    bodo_X = np.sort(np.concatenate((bodo_X_train, bodo_X_test), axis=0), axis=0)
    assert_array_equal(bodo_X, train)

    # Test distributed DataFrame
    train = pd.DataFrame({"A": range(20), "B": range(100, 120)})
    train_labels = pd.Series(range(20))
    X_train, X_test, y_train, y_test = bodo.jit(
        distributed=["X", "y", "X_train", "X_test", "y_train", "y_test"]
    )(impl_shuffle)(
        _get_dist_arg(train), _get_dist_arg(train_labels), train_size, test_size
    )
    assert_array_equal(X_train.iloc[:, 0], y_train)
    assert_array_equal(X_test.iloc[:, 0], y_test)
    bodo_X_train = bodo.allgatherv(X_train)
    bodo_X_test = bodo.allgatherv(X_test)
    bodo_X = np.sort(np.concatenate((bodo_X_train, bodo_X_test), axis=0), axis=0)
    assert_array_equal(bodo_X, train)

    from sklearn import model_selection

    def impl_shuffle_import(X, y):
        """ Test to verify that both import styles work for model_selection"""
        # simple test
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)
        return X_train, X_test, y_train, y_test

    # Test with change in import
    train = pd.DataFrame({"A": range(20), "B": range(100, 120)})
    train_labels = pd.Series(range(20))
    X_train, X_test, y_train, y_test = bodo.jit(
        distributed=["X", "y", "X_train", "X_test", "y_train", "y_test"]
    )(impl_shuffle_import)(
        _get_dist_arg(train),
        _get_dist_arg(train_labels),
    )
    assert_array_equal(X_train.iloc[:, 0], y_train)
    assert_array_equal(X_test.iloc[:, 0], y_test)
    bodo_X_train = bodo.allgatherv(X_train)
    bodo_X_test = bodo.allgatherv(X_test)
    bodo_X = np.sort(np.concatenate((bodo_X_train, bodo_X_test), axis=0), axis=0)
    assert_array_equal(bodo_X, train)


def test_train_test_split_unsupported(memory_leak_check):
    """
    Test an supported argument to train_test_split
    """

    def impl(X, y, train_size, test_size):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, test_size=test_size, stratify=True
        )
        return X_train, X_test, y_train, y_test

    train = pd.DataFrame({"A": range(20), "B": range(100, 120)})
    train_labels = pd.Series(range(20))
    train_size = 0.6
    test_size = 0.3

    err_msg = "stratify parameter only supports default value None"
    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl)(train, train_labels, train_size, test_size)


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
def test_label_encoder(values, classes):
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


def test_naive_mnnb_csr():
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
    ],
)
@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("with_mean", [True, False])
@pytest.mark.parametrize("with_std", [True, False])
def test_standard_scaler(data, copy, with_mean, with_std, memory_leak_check):
    """
    Tests for sklearn.preprocessing.StandardScaler implementation in Bodo.
    """

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
