# Copyright (C) 2022 Bodo Inc. All rights reserved.
""" Test supported sklearn.cluster and sklearn.ensemble models"""

import time

import numpy as np
import pandas as pd
import pytest
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.validation import check_random_state

import bodo
from bodo.tests.utils import _get_dist_arg, check_func

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


# ---------------------- RandomForestClassifier tests ----------------------
# Copied and adapted from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/tests/test_forest.py
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


def test_simple_pandas_input(memory_leak_check):
    """Check classification against sklearn with toy data from pandas"""

    def impl(X, y, T):
        m = RandomForestClassifier(n_estimators=10, random_state=57)
        m.fit(X, y)
        return m.predict(T)

    def impl_predict_proba(X, y, T):
        m = RandomForestClassifier(n_estimators=10, random_state=57)
        m.fit(X, y)
        return m.predict_proba(T)

    def impl_predict_log_proba(X, y, T):
        m = RandomForestClassifier(n_estimators=10, random_state=57)
        m.fit(X, y)
        return m.predict_log_proba(T)

    train = pd.DataFrame({"A": range(20), "B": range(100, 120)})
    train_labels = pd.Series(range(20))
    predict_test = pd.DataFrame({"A": range(10), "B": range(100, 110)})

    check_func(impl, (train, train_labels, predict_test))
    check_func(impl_predict_proba, (train, train_labels, predict_test))
    check_func(impl_predict_log_proba, (train, train_labels, predict_test))


def test_classification_toy(memory_leak_check):
    """Check classification on a toy dataset."""

    def impl0(X, y, T):
        clf = RandomForestClassifier(n_estimators=10, random_state=1)
        clf.fit(X, y)
        return clf

    clf = bodo.jit(distributed=["X", "y", "T"])(impl0)(
        _get_dist_arg(np.array(X)),
        _get_dist_arg(np.array(y)),
        _get_dist_arg(np.array(T)),
    )
    np.testing.assert_array_equal(clf.predict(T), true_result)
    assert 10 == len(clf)

    def impl1(X, y, T):
        clf = RandomForestClassifier(n_estimators=10, random_state=1)
        clf.fit(X, y)
        # assert 10 == len(clf)  # TODO support len of RandomForestClassifier
        return clf.predict(T)

    check_func(impl1, (np.array(X), np.array(y), np.array(T)))

    def impl2(X, y, T):
        clf = RandomForestClassifier(n_estimators=10, max_features=1, random_state=1)
        clf.fit(X, y)
        # assert 10 == len(clf)  # TODO support len of RandomForestClassifier
        return clf.predict(T)

    check_func(impl2, (np.array(X), np.array(y), np.array(T)))

    def impl_predict_proba(X, y, T):
        clf = RandomForestClassifier(n_estimators=10, max_features=1, random_state=1)
        clf.fit(X, y)
        # assert 10 == len(clf)  # TODO support len of RandomForestClassifier
        return clf.predict_proba(T)

    check_func(impl_predict_proba, (np.array(X), np.array(y), np.array(T)))

    def impl_predict_log_proba(X, y, T):
        clf = RandomForestClassifier(n_estimators=10, max_features=1, random_state=1)
        clf.fit(X, y)
        # assert 10 == len(clf)  # TODO support len of RandomForestClassifier
        return clf.predict_log_proba(T)

    check_func(impl_predict_log_proba, (np.array(X), np.array(y), np.array(T)))

    # TODO sklearn test does more stuff that we don't support currently:
    # also test apply
    # leaf_indices = clf.apply(X)
    # assert leaf_indices.shape == (len(X), clf.n_estimators)


def check_iris_criterion(criterion):
    # Check consistency on dataset iris.

    def impl(data, target, criterion):
        clf = RandomForestClassifier(
            n_estimators=10, criterion=criterion, random_state=1
        )
        clf.fit(data, target)
        score = clf.score(data, target)
        return score

    check_func(impl, (iris.data, iris.target, criterion))

    def impl2(data, target, criterion):
        clf = RandomForestClassifier(
            n_estimators=10, criterion=criterion, max_features=2, random_state=1
        )
        clf.fit(data, target)
        score = clf.score(data, target)
        return score

    check_func(impl2, (iris.data, iris.target, criterion))


@pytest.mark.parametrize("criterion", ("gini", "entropy"))
def test_iris(criterion, memory_leak_check):
    check_iris_criterion(criterion)


def test_multioutput(memory_leak_check):
    # Check estimators on multi-output problems.

    X_train = [
        [-2, -1],
        [-1, -1],
        [-1, -2],
        [1, 1],
        [1, 2],
        [2, 1],
        [-2, 1],
        [-1, 1],
        [-1, 2],
        [2, -1],
        [1, -1],
        [1, -2],
    ]
    y_train = [
        [-1, 0],
        [-1, 0],
        [-1, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [-1, 2],
        [-1, 2],
        [-1, 2],
        [1, 3],
        [1, 3],
        [1, 3],
    ]
    X_test = [[-1, -1], [1, 1], [-1, 1], [1, -1]] * 3
    y_test = [[-1, 0], [1, 1], [-1, 2], [1, 3]] * 3

    def impl(X_train, y_train, X_test):
        est = RandomForestClassifier(random_state=0, bootstrap=False)
        y_pred = est.fit(X_train, y_train).predict(X_test)
        return y_pred

    # NOTE that sklearn test uses assert_array_almost_equal(y_pred, y_test)
    # and check_func uses assert_array_equal
    check_func(
        impl,
        (np.array(X_train), np.array(y_train), np.array(X_test)),
        py_output=np.array(y_test).flatten(),
    )

    # TODO sklearn test does more stuff that we don't support currently


@pytest.mark.skip(reason="TODO: predict needs to be able to return array of strings")
def test_multioutput_string(memory_leak_check):
    # Check estimators on multi-output problems with string outputs.

    X_train = [
        [-2, -1],
        [-1, -1],
        [-1, -2],
        [1, 1],
        [1, 2],
        [2, 1],
        [-2, 1],
        [-1, 1],
        [-1, 2],
        [2, -1],
        [1, -1],
        [1, -2],
    ]
    y_train = [
        ["red", "blue"],
        ["red", "blue"],
        ["red", "blue"],
        ["green", "green"],
        ["green", "green"],
        ["green", "green"],
        ["red", "purple"],
        ["red", "purple"],
        ["red", "purple"],
        ["green", "yellow"],
        ["green", "yellow"],
        ["green", "yellow"],
    ]
    X_test = [[-1, -1], [1, 1], [-1, 1], [1, -1]]
    y_test = [
        ["red", "blue"],
        ["green", "green"],
        ["red", "purple"],
        ["green", "yellow"],
    ]

    def impl(X_train, y_train, X_test):
        est = RandomForestClassifier(random_state=0, bootstrap=False)
        y_pred = est.fit(X_train, y_train).predict(X_test)
        return y_pred

    check_func(
        impl,
        (np.array(X_train), np.array(y_train), np.array(X_test)),
        py_output=np.array(y_test).flatten(),
    )


@pytest.mark.skip(reason="Run manually on multinode cluster.")
def test_multinode_bigdata(memory_leak_check):
    """Check classification against sklearn with big data on multinode cluster"""

    # name is used for distinguishing function printing time.
    def impl(X_train, y_train, X_test, y_test, name="BODO"):
        # Bodo ignores n_jobs. This is set for scikit-learn (non-bodo) run. It should be set to number of cores avialable.
        clf = RandomForestClassifier(
            n_estimators=100, random_state=None, n_jobs=8, verbose=3
        )
        start_time = time.time()
        clf.fit(X_train, y_train)
        end_time = time.time()
        if bodo.get_rank() == 0:
            print(name, "Time: ", (end_time - start_time))
        score = clf.score(X_test, y_test)
        return score

    splitN = 500
    n_samples = 5000000
    n_features = 500
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    if bodo.get_rank() == 0:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=3,
            n_clusters_per_class=2,
            n_informative=3,
        )
        sklearn_predict_result = impl(
            X[:splitN], y[:splitN], X[splitN:], y[splitN:], "SK"
        )
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
    # TODO sklearn test does more stuff that we don't support currently


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


def test_rf_regressor(memory_leak_check):
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
