# Copyright (C) 2022 Bodo Inc. All rights reserved.
""" Test supported sklearn.linear_model models"""
import time

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.metrics import precision_score, r2_score
from sklearn.preprocessing import StandardScaler

import bodo
from bodo.tests.utils import check_func


# ---------------------- SGDClassifer tests ----------------------
def test_sgdc_svm(memory_leak_check):
    """Check SGDClassifier SVM against sklearn with big data on multinode cluster"""

    # name is used for distinguishing function printing time.
    def impl(X_train, y_train, X_test, y_test, name="SVM BODO"):
        # Bodo ignores n_jobs. This is set for scikit-learn (non-bodo) run. It should be set to number of cores avialable.
        # Currently disabling any iteration breaks for fair comparison with partial_fit. Loop for max_iter
        clf = SGDClassifier(
            n_jobs=8,
            max_iter=10,
            early_stopping=False,
            verbose=0,
        )
        start_time = time.time()
        clf.fit(X_train, y_train)
        end_time = time.time()
        if bodo.get_rank() == 0:
            print("\n", name, "Time: ", (end_time - start_time), "\n")
        score = clf.score(X_test, y_test)
        return score

    def impl_coef(X_train, y_train):
        clf = SGDClassifier()
        clf.fit(X_train, y_train)
        return clf.coef_

    splitN = 500
    n_samples = 10000
    n_features = 50
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
            X[:splitN], y[:splitN], X[splitN:], y[splitN:], "SVM SK"
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
    bodo_coef_serial = bodo.jit(distributed=False)(impl_coef)(X_train, y_train)
    if bodo.get_rank() == 0:
        bodo_R = np.dot(X_train, bodo_coef_[0]) > 0.0
        bodo_accuracy = np.sum(bodo_R == y_train) / len(X_train)
        sk_R = np.dot(X_train, sklearn_coef_[0]) > 0.0
        sk_accuracy = np.sum(sk_R == y_train) / len(X_train)
        assert np.allclose(bodo_accuracy, sk_accuracy, atol=0.1)
        serial_bodo_R = np.dot(X_train, bodo_coef_serial[0]) > 0.0
        serial_bodo_accuracy = np.sum(serial_bodo_R == y_train) / len(X_train)
        assert np.allclose(serial_bodo_accuracy, sk_accuracy, atol=0.1)


def test_sgdc_lr(memory_leak_check):
    """Check SGDClassifier Logistic Regression against sklearn with big data on multinode cluster"""

    # name is used for distinguishing function printing time.
    def impl(X_train, y_train, X_test, y_test, name="Logistic Regression BODO"):
        # Bodo ignores n_jobs. This is set for scikit-learn (non-bodo) run. It should be set to number of cores avialable.
        clf = SGDClassifier(
            n_jobs=8,
            loss="log",
            max_iter=100,
            early_stopping=False,
            random_state=42,
        )
        start_time = time.time()
        clf.fit(X_train, y_train)
        end_time = time.time()
        # score = clf.score(X_test, y_test)
        y_pred = clf.predict(X_test)
        score = precision_score(y_test, y_pred, average="micro")
        if bodo.get_rank() == 0:
            print(
                "\n", name, "Time: ", (end_time - start_time), "\tScore: ", score, "\n"
            )
        return score

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
            X[:splitN], y[:splitN], X[splitN:], y[splitN:], "Logistic Regression SK"
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


def test_sgdc_predict_proba_log_proba(memory_leak_check):

    splitN = 500
    n_samples = 1000
    n_features = 50
    X_train = None
    y_train = None
    X_test = None
    # Create exact same dataset on all ranks
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=3,
        n_clusters_per_class=2,
        n_informative=3,
        random_state=10,
    )
    X_train = X[:splitN]
    y_train = y[:splitN]
    X_test = X[splitN:]
    y_test = y[splitN:]

    # Create exact same model on all ranks using sklearn python implementation
    # That way, we can test predict_proba and predict_log_proba implementation
    # independent of the model.
    # Bodo ignores n_jobs. This is set for scikit-learn (non-bodo) run. It should be set to number of cores avialable.
    clf = SGDClassifier(
        n_jobs=8,
        loss="log",
        max_iter=10,
        early_stopping=False,
        random_state=500,
    )
    clf.fit(X_train, y_train)

    def impl_predict_proba(clf, X_test):
        y_pred_proba = clf.predict_proba(X_test)
        return y_pred_proba

    def impl_predict_log_proba(clf, X_test):
        y_pred_log_proba = clf.predict_log_proba(X_test)
        return y_pred_log_proba

    check_func(impl_predict_proba, (clf, X_test))
    check_func(impl_predict_log_proba, (clf, X_test))


# ---------------------- SGDRegressor tests ----------------------
@pytest.mark.parametrize("penalty", ["l1", "l2", None])
def test_sgdr(penalty, memory_leak_check):
    """Check SGDRegressor against sklearn
    penalty identifies type of regression
    None:Linear, l2: Ridge, l1: Lasso"""

    def impl_predict(X_train, y_train, X_test):
        clf = SGDRegressor(
            alpha=0.01,
            max_iter=2,
            eta0=0.01,
            learning_rate="adaptive",
            shuffle=False,
            penalty=penalty,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return y_pred

    X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
    y = np.array([1, 1, 1, 2, 2, 2])
    T = np.array([[-1, -1], [2, 2], [3, 2]])
    sklearn_predict_result = impl_predict(X, y, T)
    # TODO [BE-528]: Refactor this code with a distributed implementation
    bodo_predict_result = bodo.jit()(impl_predict)(X, y, T)
    np.testing.assert_array_almost_equal(
        bodo_predict_result, sklearn_predict_result, decimal=2
    )

    # name is used for distinguishing function printing time.
    def impl(X_train, y_train, X_test, y_test, name="BODO"):
        # Bodo ignores n_jobs. This is set for scikit-learn (non-bodo) run. It should be set to number of cores avialable.
        # Currently disabling any iteration breaks for fair comparison with partial_fit. Loop for max_iter
        clf = SGDRegressor(
            penalty=penalty,
            early_stopping=False,
            verbose=0,
        )
        start_time = time.time()
        clf.fit(X_train, y_train)
        end_time = time.time()
        if bodo.get_rank() == 0:
            print("\n", name, "Time: ", (end_time - start_time), "\n")
        score = clf.score(X_test, y_test)
        return score

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
        sklearn_predict_result = impl(X_train, y_train, X_test, y_test, "SK")
        X_train = bodo.scatterv(X_train)
        y_train = bodo.scatterv(y_train)
        X_test = bodo.scatterv(X_test)
        y_test = bodo.scatterv(y_test)
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


# --------------------Linear Regression Tests-----------------#


def test_linear_regression(memory_leak_check):
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
def test_lasso(memory_leak_check):
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
def test_ridge_regression(memory_leak_check):
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


# --------------------Logistic Regression Tests-----------------#


def test_logistic_regression(memory_leak_check):
    """
    Shamelessly copied from the sklearn tests:
    https://github.com/scikit-learn/scikit-learn/blob/0fb307bf39bbdacd6ed713c00724f8f871d60370/sklearn/tests/test_multiclass.py#L240
    """
    np.random.seed(42)
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


def test_logistic_regression_predict_proba_log_proba(memory_leak_check):

    splitN = 500
    n_samples = 1000
    n_features = 50
    X_train = None
    y_train = None
    X_test = None
    # Create exact same dataset on all ranks
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=3,
        n_clusters_per_class=2,
        n_informative=3,
        random_state=20,
    )
    X_train = X[:splitN]
    y_train = y[:splitN]
    X_test = X[splitN:]

    # Create exact same model on all ranks using sklearn python implementation
    # That way, we can test predict_proba and predict_log_proba implementation
    # independent of the model.
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    def impl_predict_proba(clf, X_test):
        y_pred_proba = clf.predict_proba(X_test)
        return y_pred_proba

    def impl_predict_log_proba(clf, X_test):
        y_pred_log_proba = clf.predict_log_proba(X_test)
        return y_pred_log_proba

    check_func(impl_predict_proba, (clf, X_test))
    check_func(impl_predict_log_proba, (clf, X_test))
