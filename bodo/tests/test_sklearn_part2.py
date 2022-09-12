# Copyright (C) 2022 Bodo Inc. All rights reserved.
""" Test miscellaneous supported sklearn models and methods
    Currently this file tests:
    train_test_split, MultinomialNB, LinearSVC,
"""


import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn import datasets
from sklearn.metrics import precision_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from sklearn.utils._testing import assert_array_equal, assert_raises
from sklearn.utils.validation import check_random_state

import bodo
from bodo.tests.utils import _get_dist_arg, check_func
from bodo.utils.typing import BodoError


# --------------------Multinomial Naive Bayes Tests-----------------#
def test_multinomial_nb(memory_leak_check):
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


def test_multinomial_nb_score(memory_leak_check):
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
    assert_frame_equal(bodo_shuffle_1_sorted, data, check_column_type=False)
    # Check that shuffled output is different from original data
    assert_raises(AssertionError, assert_frame_equal, bodo_shuffle_1, data)
    # Check that different shuffles give different results
    assert_raises(AssertionError, assert_frame_equal, bodo_shuffle_1, bodo_shuffle_2)

    # Test replicated shuffle using dataframes
    bodo_shuffle_1 = rep_impl(data)
    bodo_shuffle_1_sorted = bodo_shuffle_1.sort_values("A")
    bodo_shuffle_2 = rep_impl(data)

    # Check that we didn't lose or duplicate any data
    assert_frame_equal(bodo_shuffle_1_sorted, data, check_column_type=False)
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

            # Make sure we can iterate through the output of split() within JIT
            out = []
            for train_idx, test_idx in m.split(X, y, groups):
                out.append((train_idx, test_idx))
            return out

    else:

        def test_split(X, y, groups, n_splits, random_state):
            m = KFold(n_splits=n_splits, shuffle=False)

            # Make sure we can iterate through the output of split() within JIT
            out = []
            for train_idx, test_idx in m.split(X, y, groups):
                out.append((train_idx, test_idx))
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
