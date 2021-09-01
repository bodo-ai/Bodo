# Copied and adapted from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/tests/test_forest.py

import time

import numpy as np
import pandas as pd
import pytest
import scipy
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import precision_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils._testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
)
from sklearn.utils.validation import check_random_state

import bodo
from bodo.tests.utils import _get_dist_arg, check_func

# ---------------------- RandomForestClassifier tests ----------------------

# toy sample
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y = [-1, -1, -1, 1, 1, 1]
T = [[-1, -1], [2, 2], [3, 2]] * 3
true_result = [-1, 1, 1] * 3

# also load the iris dataset
# and randomly permute it
iris = datasets.load_iris()
rng = check_random_state(0)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# --------------------KMeans Clustering Tests-----------------#


def test_kmeans(memory_leak_check):
    """
    Shamelessly copied from the sklearn tests:
    https://github.com/scikit-learn/scikit-learn/blob/0fb307bf39bbdacd6ed713c00724f8f871d60370/sklearn/cluster/tests/test_k_means.py#L57
    """

    X = np.array([[0, 0], [0.5, 0], [0.5, 1], [1, 1]], dtype=np.float64)
    sample_weight = np.array([3, 1, 1, 3])
    init_centers = np.array([[0, 0], [1, 1]], dtype=np.float64)

    expected_labels = [0, 0, 1, 1]
    expected_inertia = 0.375
    expected_centers = np.array([[0.125, 0], [0.875, 1]], dtype=np.float64)
    expected_n_iter = 2

    def impl_fit(X_, sample_weight_, init_centers_):
        kmeans = KMeans(n_clusters=2, n_init=1, init=init_centers_)
        kmeans.fit(X_, sample_weight=sample_weight_)
        return kmeans

    clf = bodo.jit(distributed=["X_", "sample_weight_"])(impl_fit)(
        _get_dist_arg(np.array(X)),
        _get_dist_arg(np.array(sample_weight)),
        np.array(init_centers),
    )

    dist_expected_labels = _get_dist_arg(np.array(expected_labels))

    assert_array_equal(clf.labels_, dist_expected_labels)
    assert_allclose(clf.inertia_, expected_inertia)
    assert_allclose(clf.cluster_centers_, expected_centers)
    assert clf.n_iter_ == expected_n_iter

    def impl_predict0(X_, sample_weight_):
        kmeans = KMeans(n_clusters=2, n_init=1, init=init_centers)
        kmeans.fit(X_, None, sample_weight_)
        return kmeans.predict(X_, sample_weight_)

    check_func(
        impl_predict0,
        (
            X,
            sample_weight,
        ),
        is_out_distributed=True,
    )

    def impl_predict1(X_):
        kmeans = KMeans(n_clusters=2, n_init=1, init=init_centers)
        kmeans.fit(X_)
        return kmeans.predict(X_)

    check_func(
        impl_predict1,
        (X,),
        is_out_distributed=True,
    )

    def impl_predict2(X_, sample_weight_):
        kmeans = KMeans(n_clusters=2, n_init=1, init=init_centers)
        kmeans.fit(X_)
        return kmeans.predict(X_, sample_weight=sample_weight_)

    check_func(
        impl_predict2,
        (
            X,
            sample_weight,
        ),
        is_out_distributed=True,
    )

    def impl_score0(X_, sample_weight_):
        kmeans = KMeans(n_clusters=2, n_init=1, init=init_centers)
        kmeans.fit(X_, sample_weight=sample_weight_)
        return kmeans.score(X_, sample_weight=sample_weight_)

    check_func(
        impl_score0,
        (
            X,
            sample_weight,
        ),
    )

    def impl_score1(X_, sample_weight_):
        kmeans = KMeans(n_clusters=2, n_init=1, init=init_centers)
        kmeans.fit(X_, sample_weight=sample_weight_)
        return kmeans.score(X_, None, sample_weight_)

    check_func(
        impl_score1,
        (
            X,
            sample_weight,
        ),
    )

    def impl_score2(X_):
        kmeans = KMeans(n_clusters=2, n_init=1, init=init_centers)
        kmeans.fit(X_)
        return kmeans.score(X_)

    check_func(
        impl_score2,
        (X,),
    )

    def impl_transform(X_, sample_weight_):
        kmeans = KMeans(n_clusters=2, n_init=1, init=init_centers)
        kmeans.fit(X_, sample_weight=sample_weight_)
        return kmeans.transform(X_)

    check_func(
        impl_transform,
        (
            X,
            sample_weight,
        ),
        is_out_distributed=True,
    )


# --------------------Logistic Regression Tests-----------------#


def test_logistic_regression(memory_leak_check):
    """
    Shamelessly copied from the sklearn tests:
    https://github.com/scikit-learn/scikit-learn/blob/0fb307bf39bbdacd6ed713c00724f8f871d60370/sklearn/tests/test_multiclass.py#L240
    """
    # Toy dataset where features correspond directly to labels.
    X = np.array([[0, 0, 5], [0, 5, 0], [3, 0, 0], [0, 0, 6], [6, 0, 0]])
    y = np.array([1, 2, 2, 1, 2])
    # When testing with string, with predict this error comes
    # >           bodo_out = bodo_func(*call_args)
    # E           ValueError: invalid literal for int() with base 10: 'eggs'
    # y = np.array(["eggs", "spam", "spam", "eggs", "spam"])
    # classes = np.array(["eggs", "spam"])
    classes = np.array([1, 2])
    # Y = np.array([[0, 1, 1, 0, 1]]).T

    def impl_fit(X, y):
        clf = LogisticRegression()
        clf.fit(X, y)
        return clf

    clf = bodo.jit(impl_fit)(X, y)
    np.testing.assert_array_equal(clf.classes_, classes)

    def impl_pred(X, y):
        clf = LogisticRegression()
        clf.fit(X, y)
        y_pred = clf.predict(np.array([[0, 0, 4]]))[0]
        return y_pred

    check_func(
        impl_pred,
        (
            X,
            y,
        ),
    )

    def impl_score(X, y):
        # TODO (Hadia, Sahil) When n_jobs is set to 8, it's (recently been) failing on CodeBuild (but not Azure) for some
        # reason, so we need to investigate and fix the issue.
        clf = LogisticRegression(n_jobs=1)
        clf.fit(X, y)
        return clf.score(X, y)

    check_func(
        impl_score,
        (
            X,
            y,
        ),
    )

    def impl(X_train, y_train, X_test, y_test, name="Logistic Regression BODO"):
        # Bodo ignores n_jobs. This is set for scikit-learn (non-bodo) run. It should be set to number of cores available.
        # TODO (Hadia, Sahil) When n_jobs is set to 8, it's (recently been) failing on CodeBuild (but not Azure) for some
        # reason, so we need to investigate and fix the issue.
        clf = LogisticRegression(n_jobs=1)
        start_time = time.time()
        clf.fit(X_train, y_train)
        end_time = time.time()
        y_pred = clf.predict(X_test)
        score = precision_score(y_test, y_pred, average="weighted")
        if bodo.get_rank() == 0:
            print(
                "\n", name, "Time: ", (end_time - start_time), "\tScore: ", score, "\n"
            )
        return score

    def impl_coef(X_train, y_train):
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        return clf.coef_

    splitN = 60
    n_samples = 1000
    n_features = 10
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    if bodo.get_rank() == 0:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=2,
            n_clusters_per_class=1,
            flip_y=0.03,
            n_informative=5,
            n_redundant=0,
            n_repeated=0,
            random_state=42,
        )
        sklearn_predict_result = impl(
            X[:splitN],
            y[:splitN],
            X[splitN:],
            y[splitN:],
            "Real Logistic Regression SK",
        )
        sklearn_coef_ = impl_coef(X[:splitN], y[:splitN])
        X_train = bodo.scatterv(X[:splitN])
        y_train = bodo.scatterv(y[:splitN])
        X_test = bodo.scatterv(X[splitN:])
        y_test = bodo.scatterv(y[splitN:])
    else:
        X_train = bodo.scatterv(None)
        y_train = bodo.scatterv(None)
        X_test = bodo.scatterv(None)
        y_test = bodo.scatterv(None)

    bodo_predict_result = bodo.jit(
        distributed=["X_train", "y_train", "X_test", "y_test"]
    )(impl)(X_train, y_train, X_test, y_test)
    if bodo.get_rank() == 0:
        assert np.allclose(sklearn_predict_result, bodo_predict_result, atol=0.1)
    bodo_coef_ = bodo.jit(distributed=["X_train", "y_train"])(impl_coef)(
        X_train, y_train
    )
    if bodo.get_rank() == 0:
        bodo_R = np.dot(X_train, bodo_coef_[0]) > 0.0
        bodo_accuracy = np.sum(bodo_R == y_train) / len(X_train)
        sk_R = np.dot(X_train, sklearn_coef_[0]) > 0.0
        sk_accuracy = np.sum(sk_R == y_train) / len(X_train)

        assert np.allclose(bodo_accuracy, sk_accuracy, atol=0.1)


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


# --------------------Linear Regression Tests-----------------#


def test_linear_regression():
    """Test Linear Regression wrappers"""

    def impl(X_train, y_train, X_test, y_test):
        clf = LinearRegression()
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        return score

    def impl_pred(X_train, y_train, X_test, y_test):
        clf = LinearRegression()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return y_pred

    def impl_coef(X_train, y_train, X_test, y_test):
        clf = LinearRegression()
        clf.fit(X_train, y_train)
        return clf.coef_

    splitN = 500
    n_samples = 10000
    n_features = 100
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    if bodo.get_rank() == 0:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features,
        )
        X_train = X[:splitN]
        y_train = y[:splitN]
        X_test = X[splitN:]
        y_test = y[splitN:]
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        sklearn_score_result = impl(X_train, y_train, X_test, y_test)
        sklearn_predict_result = impl_pred(X_train, y_train, X_test, y_test)
        sklearn_coef_ = impl_coef(X_train, y_train, X_test, y_test)
        X_train = bodo.scatterv(X_train)
        y_train = bodo.scatterv(y_train)
        X_test = bodo.scatterv(X_test)
        y_test = bodo.scatterv(y_test)
    else:
        X_train = bodo.scatterv(None)
        y_train = bodo.scatterv(None)
        X_test = bodo.scatterv(None)
        y_test = bodo.scatterv(None)

    bodo_score_result = bodo.jit(
        distributed=["X_train", "y_train", "X_test", "y_test"]
    )(impl)(X_train, y_train, X_test, y_test)
    bodo_predict_result = bodo.jit(
        distributed=["X_train", "y_train", "X_test", "y_test", "y_pred"]
    )(impl_pred)(X_train, y_train, X_test, y_test)
    # Can't compare y_pred of bodo vs sklearn
    # So, we need to use a score metrics. However, current supported scores are
    # classification metrics only.
    # Gather output in rank 0. This can go away when r2_score is supported
    # TODO: return r2_score directly once it's supported.
    total_predict_result = bodo.gatherv(bodo_predict_result, root=0)
    total_y_test = bodo.gatherv(y_test, root=0)
    bodo_coef_ = bodo.jit(distributed=["X_train", "y_train", "X_test", "y_test"])(
        impl_coef
    )(X_train, y_train, X_test, y_test)
    if bodo.get_rank() == 0:
        assert np.allclose(sklearn_score_result, bodo_score_result, atol=0.1)
        b_score = r2_score(total_y_test, total_predict_result)
        sk_score = r2_score(total_y_test, sklearn_predict_result)
        assert np.allclose(b_score, sk_score, atol=0.1)
        # coef_ tolerance??? This example can be upto 0.9. Not sure if this is a good threshold
        assert np.allclose(bodo_coef_, sklearn_coef_, atol=0.9)


@pytest.mark.skip(
    reason="TODO: support Multivariate Regression (SGDRegressor doesn't support it yet"
)
def test_lr_multivariate(memory_leak_check):
    """Test Multivariate Linear Regression
    Taken from sklearn tests
    https://github.com/scikit-learn/scikit-learn/blob/0fb307bf39bbdacd6ed713c00724f8f871d60370/sklearn/tests/test_multiclass.py#L278
    """

    def test_pred(X_train, y_train):
        clf = LinearRegression()
        clf.fit(X_train, y_train)
        y_pred = clf.predict([[0, 4, 4], [0, 1, 1], [3, 3, 3]])  # [0]
        print(y_pred)
        return y_pred

    # Toy dataset where features correspond directly to labels.
    X = np.array([[0, 4, 5], [0, 5, 0], [3, 3, 3], [4, 0, 6], [6, 0, 0]])
    y = np.array([[0, 1, 1], [0, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0]])
    check_func(test_pred, (X, y))  # , only_seq=True)


# --------------------Lasso Regression Tests-----------------#
def test_lasso():
    """Test Lasso wrappers"""

    def impl(X_train, y_train, X_test, y_test):
        clf = Lasso()
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        return score

    def impl_pred(X_train, y_train, X_test, y_test):
        clf = Lasso()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return y_pred

    splitN = 500
    n_samples = 10000
    n_features = 100
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    if bodo.get_rank() == 0:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features,
        )
        X_train = X[:splitN]
        y_train = y[:splitN]
        X_test = X[splitN:]
        y_test = y[splitN:]
        sklearn_score_result = impl(X_train, y_train, X_test, y_test)
        sklearn_predict_result = impl_pred(X_train, y_train, X_test, y_test)
        X_train = bodo.scatterv(X_train)
        y_train = bodo.scatterv(y_train)
        X_test = bodo.scatterv(X_test)
        y_test = bodo.scatterv(y_test)
    else:
        X_train = bodo.scatterv(None)
        y_train = bodo.scatterv(None)
        X_test = bodo.scatterv(None)
        y_test = bodo.scatterv(None)

    bodo_score_result = bodo.jit(
        distributed=["X_train", "y_train", "X_test", "y_test"]
    )(impl)(X_train, y_train, X_test, y_test)
    bodo_predict_result = bodo.jit(
        distributed=["X_train", "y_train", "X_test", "y_test", "y_pred"]
    )(impl_pred)(X_train, y_train, X_test, y_test)
    # Can't compare y_pred of bodo vs sklearn
    # So, we need to use a score metrics. However, current supported scores are
    # classification metrics only.
    # Gather output in rank 0. This can go away when r2_score is supported
    # TODO: return r2_score directly once it's supported.
    total_predict_result = bodo.gatherv(bodo_predict_result, root=0)
    total_y_test = bodo.gatherv(y_test, root=0)
    if bodo.get_rank() == 0:
        assert np.allclose(sklearn_score_result, bodo_score_result, atol=0.1)
        b_score = r2_score(total_y_test, total_predict_result)
        sk_score = r2_score(total_y_test, sklearn_predict_result)
        assert np.allclose(b_score, sk_score, atol=0.1)


# --------------------Ridge Regression Tests-----------------#
def test_ridge_regression():
    """Test Ridge Regression wrapper"""

    def impl(X_train, y_train, X_test, y_test):
        clf = Ridge()
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        return score

    def impl_pred(X_train, y_train, X_test, y_test):
        clf = Ridge()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return y_pred

    def impl_coef(X_train, y_train, X_test, y_test):
        clf = Ridge()
        clf.fit(X_train, y_train)
        return clf.coef_

    splitN = 500
    n_samples = 10000
    n_features = 100
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    if bodo.get_rank() == 0:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features,
        )
        X_train = X[:splitN]
        y_train = y[:splitN]
        X_test = X[splitN:]
        y_test = y[splitN:]
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        sklearn_score_result = impl(X_train, y_train, X_test, y_test)
        sklearn_predict_result = impl_pred(X_train, y_train, X_test, y_test)
        sklearn_coef_ = impl_coef(X_train, y_train, X_test, y_test)
        X_train = bodo.scatterv(X_train)
        y_train = bodo.scatterv(y_train)
        X_test = bodo.scatterv(X_test)
        y_test = bodo.scatterv(y_test)
    else:
        X_train = bodo.scatterv(None)
        y_train = bodo.scatterv(None)
        X_test = bodo.scatterv(None)
        y_test = bodo.scatterv(None)

    bodo_score_result = bodo.jit(
        distributed=["X_train", "y_train", "X_test", "y_test"]
    )(impl)(X_train, y_train, X_test, y_test)
    bodo_predict_result = bodo.jit(
        distributed=["X_train", "y_train", "X_test", "y_test", "y_pred"]
    )(impl_pred)(X_train, y_train, X_test, y_test)
    # Can't compare y_pred of bodo vs sklearn
    # So, we need to use a score metrics. However, current supported scores are
    # classification metrics only.
    # Gather output in rank 0. This can go away when r2_score is supported
    # TODO: return r2_score directly once it's supported.
    total_predict_result = bodo.gatherv(bodo_predict_result, root=0)
    total_y_test = bodo.gatherv(y_test, root=0)
    bodo_coef_ = bodo.jit(distributed=["X_train", "y_train", "X_test", "y_test"])(
        impl_coef
    )(X_train, y_train, X_test, y_test)
    if bodo.get_rank() == 0:
        assert np.allclose(sklearn_score_result, bodo_score_result, atol=0.1)
        b_score = r2_score(total_y_test, total_predict_result)
        sk_score = r2_score(total_y_test, sklearn_predict_result)
        assert np.allclose(b_score, sk_score, atol=0.1)
        assert np.allclose(bodo_coef_, sklearn_coef_, atol=0.9)


# --------------------Linear SVC -----------------#
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


def test_hashing_vectorizer():
    """Test HashingVectorizer's fit_transform method.
    Taken from here (https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/feature_extraction/tests/test_text.py#L573)
    """

    JUNK_FOOD_DOCS = (
        "the pizza pizza beer copyright",
        "the pizza burger beer copyright",
        "the the pizza beer beer copyright",
        "the burger beer beer copyright",
        "the coke burger coke copyright",
        "the coke burger burger",
    )

    NOTJUNK_FOOD_DOCS = (
        "the salad celeri copyright",
        "the salad salad sparkling water copyright",
        "the the celeri celeri copyright",
        "the tomato tomato salad water",
        "the tomato salad water copyright",
    )

    ALL_FOOD_DOCS = JUNK_FOOD_DOCS + NOTJUNK_FOOD_DOCS

    def test_fit_transform(X):
        v = HashingVectorizer()
        X_transformed = v.fit_transform(X)
        return X_transformed

    result = bodo.jit(
        test_fit_transform,
        all_args_distributed_block=True,
        all_returns_distributed=True,
    )(_get_dist_arg(np.array(ALL_FOOD_DOCS), False))
    result = bodo.allgatherv(result)
    token_nnz = result.nnz
    assert result.shape == (len(ALL_FOOD_DOCS), (2 ** 20))
    # By default the hashed values receive a random sign and l2 normalization
    # makes the feature values bounded
    assert np.min(result.data) > -1
    assert np.min(result.data) < 0
    assert np.max(result.data) > 0
    assert np.max(result.data) < 1
    # Check that the rows are normalized
    for i in range(result.shape[0]):
        assert_almost_equal(np.linalg.norm(result[0].data, 2), 1.0)

    check_func(test_fit_transform, (np.array(ALL_FOOD_DOCS),))

    # Check vectorization with some non-default parameters
    def test_fit_transform_args(X):
        v = HashingVectorizer(ngram_range=(1, 2), norm="l1")
        ans = v.fit_transform(X)
        return ans

    X = bodo.jit(distributed=["X", "ans"])(test_fit_transform_args)(
        _get_dist_arg(np.array(ALL_FOOD_DOCS))
    )
    X = bodo.allgatherv(X)
    assert X.shape == (len(ALL_FOOD_DOCS), (2 ** 20))

    # ngrams generate more non zeros
    ngrams_nnz = X.nnz
    assert ngrams_nnz > token_nnz
    assert ngrams_nnz < 2 * token_nnz

    # makes the feature values bounded
    assert np.min(X.data) > -1
    assert np.max(X.data) < 1

    # Check that the rows are normalized
    for i in range(X.shape[0]):
        assert_almost_equal(np.linalg.norm(X[0].data, 1), 1.0)


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


# ---------------------- RandomForestRegressor tests ----------------------
def generate_dataset(n_train, n_test, n_features, noise=0.1, verbose=False):
    """Generate a regression dataset with the given parameters."""
    """ Copied from https://scikit-learn.org/0.16/auto_examples/applications/plot_prediction_latency.html """
    if verbose:
        print("generating dataset...")
    # IMPORTANT (Bodo change): This is called on all ranks to generate the same
    # data so they must call with the same random_state.
    # By the way we use our tests in this module, it's possible that sklearn's
    # internal random state is out of sync across processes when we get here
    X, y, coef = make_regression(
        n_samples=n_train + n_test,
        n_features=n_features,
        noise=noise,
        coef=True,
        random_state=7,
    )
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]
    idx = np.arange(n_train)
    np.random.seed(13)
    np.random.shuffle(idx)
    X_train = X_train[idx]
    y_train = y_train[idx]

    std = X_train.std(axis=0)
    mean = X_train.mean(axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    std = y_train.std(axis=0)
    mean = y_train.mean(axis=0)
    y_train = (y_train - mean) / std
    y_test = (y_test - mean) / std

    import gc

    gc.collect()
    if verbose:
        print("ok")
    return X_train, y_train, X_test, y_test


def test_rf_regressor():
    """
    Test RandomForestRegressor model, fit, predict, and score
    """

    # Test model
    # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/tests/test_forest.py#L1402
    X = np.zeros((10, 10))
    y = np.ones((10,))

    def test_model(X, y):
        return RandomForestRegressor(n_estimators=10, random_state=57).fit(X, y)

    gbr = bodo.jit(test_model)(_get_dist_arg(X), _get_dist_arg(y))
    assert_array_equal(gbr.feature_importances_, np.zeros(10, dtype=np.float64))

    # Test predict and score
    X_train, y_train, X_test, y_test = generate_dataset(100, 20, 30)

    def test_predict(X_train, y_train, X_test):
        rfr = RandomForestRegressor(random_state=7)
        rfr.fit(X_train, y_train)
        y_pred = rfr.predict(X_test)
        return y_pred

    check_func(test_predict, (X_train, y_train, X_test))

    def test_score(X_train, y_train, X_test, y_test):
        rfr = RandomForestRegressor(random_state=7)
        rfr.fit(X_train, y_train)
        y_pred = rfr.predict(X_test)
        score = r2_score(y_test, y_pred)
        return score

    bodo_score = bodo.jit(distributed=["X_train", "y_train", "X_test", "y_test"])(
        test_score
    )(
        _get_dist_arg(np.array(X_train)),
        _get_dist_arg(np.array(y_train)),
        _get_dist_arg(np.array(X_test)),
        _get_dist_arg(np.array(y_test)),
    )
    sklearn_score = test_score(X_train, y_train, X_test, y_test)
    assert np.allclose(sklearn_score, bodo_score, atol=0.1)


def test_count_vectorizer():
    """Test CountVectorizer's vocabulary and fit_transform"""

    cat_in_the_hat_docs = [
        "One Cent, Two Cents, Old Cent, New Cent: All About Money (Cat in the Hat's Learning Library",
        "Inside Your Outside: All About the Human Body (Cat in the Hat's Learning Library)",
        "Oh, The Things You Can Do That Are Good for You: All About Staying Healthy (Cat in the Hat's Learning Library)",
        "On Beyond Bugs: All About Insects (Cat in the Hat's Learning Library)",
        "There's No Place Like Space: All About Our Solar System (Cat in the Hat's Learning Library)",
    ]
    df = pd.DataFrame({"A": cat_in_the_hat_docs})

    def impl(df):
        v = CountVectorizer()
        v.fit_transform(df["A"])
        ans = v.get_feature_names()
        return ans

    check_func(impl, (df,))

    # Test vocabulary_ and stop_words
    def impl2(df):
        v = CountVectorizer(stop_words="english")
        v.fit_transform(df["A"])
        return sorted(v.vocabulary_)

    check_func(impl2, (df,))

    # Test user-defined vocabulary
    # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/feature_extraction/tests/test_text.py#L315
    def impl3(docs, vocab):
        v = CountVectorizer(vocabulary=vocab)
        ans = v.fit_transform(docs)
        return ans

    JUNK_FOOD_DOCS = (
        "the pizza pizza beer copyright",
        "the pizza burger beer copyright",
        "the the pizza beer beer copyright",
        "the burger beer beer copyright",
        "the coke burger coke copyright",
        "the coke burger burger",
    )
    vocab = {"pizza": 0, "beer": 1}
    result = bodo.jit(
        impl3,
        all_args_distributed_block=True,
        all_returns_distributed=True,
    )(_get_dist_arg(np.array(JUNK_FOOD_DOCS), False), vocab)
    X = bodo.allgatherv(result)
    terms = set(vocab.keys())
    assert X.shape[1] == len(terms)
    # assert same values in both sklearn and Bodo
    X_sk = impl3(JUNK_FOOD_DOCS, vocab)
    assert np.array_equal(X.todense(), X_sk.todense())

    # Test replicated
    check_func(impl3, (JUNK_FOOD_DOCS, vocab), only_seq=True)
