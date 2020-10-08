# Copied and adapted from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/tests/test_forest.py

import random
import time

import numpy as np
import pandas as pd
import pytest
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
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


def test_simple_pandas_input(memory_leak_check):
    """Check classification against sklearn with toy data from pandas"""

    def impl(X, y, T):
        m = RandomForestClassifier(n_estimators=10, random_state=57)
        m.fit(X, y)
        return m.predict(T)

    train = pd.DataFrame({"A": range(20), "B": range(100, 120)})
    train_labels = pd.Series(range(20))
    predict_test = pd.DataFrame({"A": range(10), "B": range(100, 110)})

    check_func(impl, (train, train_labels, predict_test))


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

    # TODO sklearn test does more stuff that we don't support currently


# ---------------------- sklearn.metrics score tests ----------------------


def gen_random(n, true_chance, return_arrays=True):
    random.seed(5)
    y_true = [random.randint(-3, 3) for _ in range(n)]
    valid_cats = set(y_true)
    y_pred = []
    for i in range(n):
        if random.random() < true_chance:
            y_pred.append(y_true[i])
        else:
            y_pred.append(random.choice(list(valid_cats - {y_true[i]})))
    if return_arrays:
        return [np.array(y_true), np.array(y_pred)]
    else:
        return [y_true, y_pred]


@pytest.mark.parametrize(
    "data",
    [
        gen_random(10, 0.5, return_arrays=True),
        gen_random(50, 0.7, return_arrays=True),
        gen_random(76, 0.3, return_arrays=False),
        gen_random(11, 0.43, return_arrays=False),
    ],
)
@pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
# TODO: Add memory_leak when bug is solved (curently fails on data0 and data1)
def test_score(data, average):
    def test_precision(y_true, y_pred, average):
        return precision_score(y_true, y_pred, average=average)

    def test_recall(y_true, y_pred, average):
        return recall_score(y_true, y_pred, average=average)

    def test_f1(y_true, y_pred, average):
        return f1_score(y_true, y_pred, average=average)

    check_func(test_precision, tuple(data + [average]), is_out_distributed=False)
    check_func(test_recall, tuple(data + [average]), is_out_distributed=False)
    check_func(test_f1, tuple(data + [average]), is_out_distributed=False)


@pytest.mark.skip(reason="Run manually on multinode cluster.")
def test_multinode_bigdata():
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
        y_pred = clf.predict(X_test)
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
