"""Test supported sklearn naive bayes models"""

import numpy as np
import pandas as pd
import pytest
import scipy
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils._testing import assert_array_equal

import bodo
from bodo.tests.utils import _get_dist_arg, check_func

pytestmark = [pytest.mark.ml, pytest.mark.weekly]


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
