# Copyright (C) 2019 Bodo Inc. All rights reserved.
""" Test miscellaneous supported sklearn models and methods
    Currently this file tests:
    train_test_split, MultinomialNB, LinearSVC, 
    OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
"""

import random

import numpy as np
import pandas as pd
import pytest
import scipy
from pandas.testing import assert_frame_equal
from scipy.special import comb
from sklearn import datasets
from sklearn.metrics import precision_score
from sklearn.model_selection import KFold, LeavePOut, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import (
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from sklearn.utils._testing import assert_array_equal, assert_raises
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


# ----------------------------shuffle-----------------------------
@pytest.mark.parametrize(
    "data",
    [
        np.arange(100).reshape((10, 10)).astype(np.int64),
        np.arange(100).reshape((10, 10)).astype(np.float64),
    ],
)
def test_shuffle_np(data, memory_leak_check):
    """
    Test sklearn.utils.shuffle for np arrays. For each test, check that the
    output is a permutation of the input (i.e. didn't lose/duplicate data),
    and not equal to the input (i.e. the order actually changed).
    """

    def impl(data):
        out = shuffle(data, n_samples=None)
        return out

    dist_impl = bodo.jit(distributed=["data"])(impl)
    rep_impl = bodo.jit(replicated=True)(impl)

    # Test distributed shuffle using np arrays
    bodo_shuffle_1 = dist_impl(_get_dist_arg(data))
    bodo_shuffle_1 = bodo.allgatherv(bodo_shuffle_1)
    bodo_shuffle_1_sorted = np.sort(bodo_shuffle_1.flatten())
    bodo_shuffle_2 = dist_impl(_get_dist_arg(data))
    bodo_shuffle_2 = bodo.allgatherv(bodo_shuffle_2)

    # Check that we didn't lose or duplicate any data
    assert_array_equal(bodo_shuffle_1_sorted, data.flatten())
    # Check that shuffled output is different from original data
    assert_raises(AssertionError, assert_array_equal, bodo_shuffle_1, data)
    # Check that different shuffles give different results
    assert_raises(AssertionError, assert_array_equal, bodo_shuffle_1, bodo_shuffle_2)

    # Test replicated shuffle using np arrays
    bodo_shuffle_1 = rep_impl(data)
    bodo_shuffle_1_sorted = np.sort(bodo_shuffle_1.flatten())
    bodo_shuffle_2 = rep_impl(data)

    # Check that we didn't lose or duplicate any data
    assert_array_equal(bodo_shuffle_1_sorted, data.flatten())
    # Check that shuffled output is different from original data
    assert_raises(AssertionError, assert_array_equal, bodo_shuffle_1, data)
    # Check that different shuffles give different results
    assert_raises(AssertionError, assert_array_equal, bodo_shuffle_1, bodo_shuffle_2)


@pytest.mark.parametrize(
    "data",
    [
        pd.DataFrame(
            {
                "A": np.arange(20).astype(np.int64),
                "B": np.arange(100, 120).astype(np.int64),
            }
        ),
        pd.DataFrame(
            {
                "A": np.arange(20).astype(np.float64),
                "B": np.arange(100, 120).astype(np.float64),
            }
        ),
    ],
)
def test_shuffle_pd(data, memory_leak_check):
    """
    Test sklearn.utils.shuffle for dataframes. For each test, check that the
    output is a permutation of the input (i.e. didn't lose/duplicate data),
    and not equal to the input (i.e. the order actually changed).
    """

    def impl(data):
        out = shuffle(data, n_samples=None)
        return out

    dist_impl = bodo.jit(distributed=["data"])(impl)
    rep_impl = bodo.jit(replicated=True)(impl)

    # Test distributed shuffle using dataframes
    bodo_shuffle_1 = dist_impl(_get_dist_arg(data))
    bodo_shuffle_1 = bodo.allgatherv(bodo_shuffle_1)
    bodo_shuffle_1_sorted = bodo_shuffle_1.sort_values("A")
    bodo_shuffle_2 = dist_impl(_get_dist_arg(data))
    bodo_shuffle_2 = bodo.allgatherv(bodo_shuffle_2)

    # Check that we didn't lose or duplicate any data
    assert_frame_equal(bodo_shuffle_1_sorted, data)
    # Check that shuffled output is different from original data
    assert_raises(AssertionError, assert_frame_equal, bodo_shuffle_1, data)
    # Check that different shuffles give different results
    assert_raises(AssertionError, assert_frame_equal, bodo_shuffle_1, bodo_shuffle_2)

    # Test replicated shuffle using dataframes
    bodo_shuffle_1 = rep_impl(data)
    bodo_shuffle_1_sorted = bodo_shuffle_1.sort_values("A")
    bodo_shuffle_2 = rep_impl(data)

    # Check that we didn't lose or duplicate any data
    assert_frame_equal(bodo_shuffle_1_sorted, data)
    # Check that shuffled output is different from original data
    assert_raises(AssertionError, assert_frame_equal, bodo_shuffle_1, data)
    # Check that different shuffles give different results
    assert_raises(AssertionError, assert_frame_equal, bodo_shuffle_1, bodo_shuffle_2)


@pytest.mark.parametrize(
    "data",
    [
        np.arange(100).reshape((10, 10)).astype(np.int64),
        np.arange(100).reshape((10, 10)).astype(np.float64),
    ],
)
@pytest.mark.parametrize("random_state", [0, 1, 2])
def test_shuffle_random_state(data, random_state, memory_leak_check):
    """
    Test that shuffle returns deterministic results with given random states
    """

    def impl(data, random_state):
        out = shuffle(data, random_state=random_state)

    dist_impl = bodo.jit(distributed=["data"], cache=True)(impl)

    bodo_shuffle_1 = dist_impl(_get_dist_arg(data), random_state)
    bodo_shuffle_1 = bodo.allgatherv(bodo_shuffle_1)
    bodo_shuffle_2 = dist_impl(_get_dist_arg(data), random_state)
    bodo_shuffle_2 = bodo.allgatherv(bodo_shuffle_2)

    assert_array_equal(bodo_shuffle_1, bodo_shuffle_2)


@pytest.mark.slow
@pytest.mark.parametrize("n_samples", [0, 1, 8, 14, 15])
@pytest.mark.parametrize("nitems, niters", [(15, 10000)])
def test_shuffle_n_samples(nitems, niters, n_samples, memory_leak_check):
    """
    Test the `n_samples` argument to sklearn.utils.shuffle
    """

    def impl(data, n_samples, random_state):
        out = shuffle(data, n_samples=n_samples, random_state=random_state)
        return out

    dist_impl = bodo.jit(distributed=["data"])(impl)

    data = np.arange(nitems)

    bodo_shuffle_n = dist_impl(_get_dist_arg(data), n_samples, 0)
    bodo_shuffle_n = bodo.allgatherv(bodo_shuffle_n)
    bodo_shuffle_all = dist_impl(_get_dist_arg(data), None, 0)
    bodo_shuffle_all = bodo.allgatherv(bodo_shuffle_all)

    # Check that number of samples is correct
    assert bodo_shuffle_n.shape[0] == min(n_samples, data.shape[0])

    # Check that our samples are a subset of `data` with no repetitions
    bodo_shuffle_n_elts = set(bodo_shuffle_n.flatten())
    bodo_shuffle_all_elts = set(bodo_shuffle_all.flatten())
    assert len(bodo_shuffle_n_elts) == bodo_shuffle_n.shape[0] * np.prod(data.shape[1:])
    assert bodo_shuffle_n_elts.issubset(bodo_shuffle_all_elts)

    # Check that output is close to a uniform distribution
    if n_samples > 0:
        # output_freqs[i, j] indicates the number of times that
        # element i gets moved to index j
        output_freqs = np.zeros((n_samples, nitems))
        for i in range(niters):
            output = dist_impl(_get_dist_arg(data), n_samples, i)
            output = bodo.allgatherv(output)
            for j in range(n_samples):
                output_freqs[j, output[j]] += 1

        expected_freq = niters / nitems
        assert np.all(3 / 4 * expected_freq < output_freqs)
        assert np.all(output_freqs < 4 / 3 * expected_freq)


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
    """Test train_test_split with DataFrame dataset and train_size/test_size variation"""

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
        """Test to verify that both import styles work for model_selection"""
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
    Test an unsupported argument to train_test_split
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


# ----------------------- KFold -----------------------------


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
@pytest.mark.parametrize("n_splits", [2, 3, 5])
@pytest.mark.parametrize("shuffle", [True, False])
def test_kfold(X, y, groups, n_splits, shuffle, memory_leak_check):
    """Test sklearn.model_selection.KFold's split method."""

    if shuffle:

        def test_split(X, y, groups, n_splits, random_state):
            m = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            out = m.split(X, y, groups)
            return out

    else:

        def test_split(X, y, groups, n_splits, random_state):
            m = KFold(n_splits=n_splits, shuffle=False)
            out = m.split(X, y, groups)
            return out

    dist_impl = bodo.jit(distributed=["X", "y", "groups"])(test_split)
    rep_impl = bodo.jit(replicated=True)(test_split)

    # Test that all indices are returned in distributed split
    test_idxs_dist = []
    for train_idxs, test_idxs in dist_impl(
        _get_dist_arg(X), _get_dist_arg(y), groups, n_splits, 0
    ):
        train_idxs = bodo.allgatherv(train_idxs)
        test_idxs = bodo.allgatherv(test_idxs)
        test_idxs_dist.append(test_idxs)

        idxs = np.sort(np.concatenate((train_idxs, test_idxs), axis=0), axis=0)
        assert_array_equal(idxs, np.arange(X.shape[0]))

    assert n_splits == len(test_idxs_dist)
    all_test_idxs_dist = np.sort(np.concatenate(test_idxs_dist, axis=0), axis=0)
    assert_array_equal(all_test_idxs_dist, np.arange(X.shape[0]))

    # Test that all indices are returned in replicated split
    test_idxs_rep = []
    for train_idxs, test_idxs in rep_impl(X, y, groups, n_splits, 0):
        test_idxs_rep.append(test_idxs)

        idxs = np.sort(np.concatenate((train_idxs, test_idxs), axis=0), axis=0)
        assert_array_equal(idxs, np.arange(X.shape[0]))

    assert n_splits == len(test_idxs_rep)
    all_test_idxs_rep = np.sort(np.concatenate(test_idxs_rep, axis=0), axis=0)
    assert_array_equal(all_test_idxs_rep, np.arange(X.shape[0]))

    # Test that bodo's fold sizes are equivalent to sklearn's
    test_idxs_sklearn = []
    for train_idxs, test_idxs in test_split(X, y, groups, n_splits, 0):
        test_idxs_sklearn.append(test_idxs)

    for dist_idxs, rep_idxs, sklearn_idxs in zip(
        test_idxs_dist, test_idxs_rep, test_idxs_sklearn
    ):
        assert len(dist_idxs) == len(sklearn_idxs)
        assert len(rep_idxs) == len(sklearn_idxs)

    # Test that get_n_splits returns the correct number of folds
    def test_n_splits():
        m = KFold(n_splits=n_splits)
        out = m.get_n_splits()
        return out

    check_func(test_n_splits, ())


@pytest.mark.parametrize("n_splits", [2, 3, 5])
def test_kfold_random_state(n_splits, memory_leak_check):
    """Test that KFold.split() gives deterministic outputs when random_state is passed"""

    def test_split(X, n_splits, random_state):
        m = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        out = m.split(X)
        return out

    dist_impl = bodo.jit(distributed=["X", "y", "groups"])(test_split)

    X = np.arange(100)
    for (train1, test1), (train2, test2), (train3, test3) in zip(
        dist_impl(X, n_splits, 0),
        dist_impl(X, n_splits, 0),
        dist_impl(X, n_splits, None),
    ):
        # Check that train1/train2 and test1/test2 are equal
        assert_array_equal(train1, train2)
        assert_array_equal(test1, test2)
        # Check that train1/train3 and test1/test3 are not equal
        assert_raises(AssertionError, assert_array_equal, train1, train3)
        assert_raises(AssertionError, assert_array_equal, test1, test3)


def test_kfold_error():
    """Test error handling of KFold.split()"""

    def test_split(X, n_splits):
        m = KFold(n_splits=n_splits, shuffle=False)
        out = m.split(X)
        return out

    dist_impl = bodo.jit(distributed=["X"])(test_split)

    # X has too few items for 4-way fold
    X = np.array([[1, 2], [3, 4], [5, 6]])
    error_str = "number of splits n_splits=4 greater than the number of samples"
    with pytest.raises(ValueError, match=error_str):
        for train_idxs, test_idxs in dist_impl(_get_dist_arg(X), 4):
            pass


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


def test_leave_p_out_error():
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


# ---------------------MaxAbsScaler Tests--------------------


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


## TODO Fix memory leak [BE-2825]
def test_robust_scaler_array_and_quantile_range_attrs():

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
