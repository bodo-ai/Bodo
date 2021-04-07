# Copied and adapted from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/tests/test_forest.py

import random
import time

import numpy as np
import pandas as pd
import pytest
import scipy
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils._testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
)
from sklearn.utils.validation import check_random_state

import bodo
from bodo.tests.utils import _get_dist_arg, check_func
from bodo.utils.typing import BodoError

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


def gen_random_k_dims(n, k):
    """
    Generate a random array of shape (n, k).
    Each element is in [0,1).
    If k == 1, then it returns an array of shape (n,)
    """
    random.seed(5)
    if k > 1:
        y_true = np.random.rand(n, k)
        y_pred = np.random.rand(n, k)
    elif k == 1:
        y_true = np.random.random_sample(size=n)
        y_pred = np.random.random_sample(size=n)
    else:
        raise RuntimeError("k must be >=1")

    sample_weight = np.random.random_sample(size=n)
    return [y_true, y_pred, sample_weight]


def gen_random_with_sample_weight(n, true_chance, return_arrays=True):
    """
    Wrapper around the gen_random function. This one also has a third
    array/list for sample_weight, each element of which is in (0,1).
    Returns np arrays if return_arrays==True, else python lists.
    """
    [y_true, y_pred] = gen_random(n, true_chance, return_arrays)
    np.random.seed(5)
    sample_weight = np.random.random_sample(size=n)
    if not return_arrays:
        sample_weight = list(sample_weight)
    return [y_true, y_pred, sample_weight]


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

    from sklearn import metrics

    def test_metrics_f1(y_true, y_pred, average):
        """ Test to verify that both import styles work for classification metrics"""
        return metrics.f1_score(y_true, y_pred, average=average)

    check_func(test_precision, tuple(data + [average]), is_out_distributed=False)
    check_func(test_recall, tuple(data + [average]), is_out_distributed=False)
    check_func(test_f1, tuple(data + [average]), is_out_distributed=False)
    check_func(test_metrics_f1, tuple(data + [average]), is_out_distributed=False)


@pytest.mark.parametrize(
    "data",
    [
        gen_random_with_sample_weight(10, 0.5, return_arrays=True),
        gen_random_with_sample_weight(50, 0.7, return_arrays=True),
        gen_random_with_sample_weight(76, 0.3, return_arrays=False),
        gen_random_with_sample_weight(11, 0.43, return_arrays=False),
    ],
)
@pytest.mark.parametrize("normalize", [True, False])
# TODO: Add memory_leak when bug is solved (curently fails on data0 and data1)
def test_accuracy_score(data, normalize):
    """
    Tests for the sklearn.metrics.accuracy_score implementation in Bodo.
    """

    def test_accuracy_score_0(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def test_accuracy_score_1(y_true, y_pred):
        return accuracy_score(y_true, y_pred, normalize=normalize)

    def test_accuracy_score_2(y_true, y_pred, sample_weight_):
        return accuracy_score(y_true, y_pred, sample_weight=sample_weight_)

    def test_accuracy_score_3(y_true, y_pred, sample_weight_):
        return accuracy_score(
            y_true, y_pred, normalize=normalize, sample_weight=sample_weight_
        )

    def test_accuracy_score_4(y_true, y_pred, sample_weight_):
        return accuracy_score(
            y_true, y_pred, sample_weight=sample_weight_, normalize=normalize
        )

    check_func(test_accuracy_score_0, tuple(data[0:2]), is_out_distributed=False)
    check_func(test_accuracy_score_1, tuple(data[0:2]), is_out_distributed=False)
    check_func(test_accuracy_score_2, tuple(data), is_out_distributed=False)
    check_func(
        test_accuracy_score_3,
        tuple(data),
        is_out_distributed=False,
    )
    check_func(
        test_accuracy_score_4,
        tuple(data),
        is_out_distributed=False,
    )


@pytest.mark.parametrize(
    "data",
    [
        gen_random_k_dims(20, 1),
        gen_random_k_dims(20, 3),
    ],
)
@pytest.mark.parametrize("squared", [True, False])
@pytest.mark.parametrize("multioutput", ["uniform_average", "raw_values", "array"])
def test_mse(data, squared, multioutput, memory_leak_check):
    """
    Tests for the sklearn.metrics.mean_squared_error implementation in Bodo.
    """

    if multioutput == "array":
        if len(data[0].shape) > 1:
            multioutput = np.random.random_sample(size=data[0].shape[1])
        else:
            return

    def test_mse_0(y_true, y_pred):
        return mean_squared_error(
            y_true, y_pred, squared=squared, multioutput=multioutput
        )

    def test_mse_1(y_true, y_pred, sample_weight_):
        return mean_squared_error(
            y_true,
            y_pred,
            sample_weight=sample_weight_,
            squared=squared,
            multioutput=multioutput,
        )

    check_func(test_mse_0, tuple(data[0:2]), is_out_distributed=False)
    check_func(test_mse_1, tuple(data), is_out_distributed=False)


@pytest.mark.parametrize(
    "data",
    [
        gen_random_k_dims(20, 1),
        gen_random_k_dims(20, 3),
    ],
)
@pytest.mark.parametrize("multioutput", ["uniform_average", "raw_values", "array"])
def test_mae(data, multioutput, memory_leak_check):
    """
    Tests for the sklearn.metrics.mean_absolute_error implementation in Bodo.
    """

    if multioutput == "array":
        if len(data[0].shape) > 1:
            multioutput = np.random.random_sample(size=data[0].shape[1])
        else:
            return

    def test_mae_0(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred, multioutput=multioutput)

    def test_mae_1(y_true, y_pred, sample_weight_):
        return mean_absolute_error(
            y_true,
            y_pred,
            sample_weight=sample_weight_,
            multioutput=multioutput,
        )

    check_func(test_mae_0, tuple(data[0:2]), is_out_distributed=False)
    check_func(test_mae_1, tuple(data), is_out_distributed=False)


@pytest.mark.parametrize(
    "data",
    [
        gen_random_k_dims(20, 1),
        gen_random_k_dims(20, 3),
    ],
)
@pytest.mark.parametrize(
    "multioutput",
    [
        "uniform_average",
        "raw_values",
        "variance_weighted",
        "array",
        "some_unsupported_val",
    ],
)
def test_r2_score(data, multioutput, memory_leak_check):
    """
    Tests for the sklearn.metrics.r2_score implementation in Bodo.
    """

    if multioutput == "array":
        if len(data[0].shape) > 1:
            multioutput = np.random.random_sample(size=data[0].shape[1])
        else:
            return

    def test_r2_0(y_true, y_pred):
        return r2_score(y_true, y_pred, multioutput=multioutput)

    def test_r2_1(y_true, y_pred, sample_weight_):
        return r2_score(
            y_true,
            y_pred,
            sample_weight=sample_weight_,
            multioutput=multioutput,
        )

    from sklearn import metrics

    def test_metrics_r2_1(y_true, y_pred, sample_weight_):
        """ Test to verify that both import styles work for regression metrics"""
        return metrics.r2_score(
            y_true,
            y_pred,
            sample_weight=sample_weight_,
            multioutput=multioutput,
        )

    # To check that Bodo fails in compilation when an unsupported value is passed
    # in for multioutput
    if multioutput == "some_unsupported_val":
        with pytest.raises(BodoError, match="Unsupported argument"):
            bodo.jit(distributed=["y_true", "y_pred"])(test_r2_0)(
                _get_dist_arg(data[0]), _get_dist_arg(data[1])
            )
        return

    check_func(test_r2_0, tuple(data[0:2]), is_out_distributed=False)
    check_func(test_r2_1, tuple(data), is_out_distributed=False)
    check_func(test_metrics_r2_1, tuple(data), is_out_distributed=False)

    # Check that appropriate error is raised when number of samples in
    # y_true and y_pred are inconsistent
    with pytest.raises(
        ValueError,
        match="inconsistent number of samples",
    ):
        bodo.jit(distributed=["y_true", "y_pred"])(test_r2_0)(
            _get_dist_arg(data[0]), _get_dist_arg(data[1][:-1])
        )


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


# ---------------------- SGDClassifer tests ----------------------
def test_sgdc_svm():
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


def test_sgdc_lr():
    """Check SGDClassifier Logistic Regression against sklearn with big data on multinode cluster"""

    # name is used for distinguishing function printing time.
    def impl(X_train, y_train, X_test, y_test, name="Logistic Regression BODO"):
        # Bodo ignores n_jobs. This is set for scikit-learn (non-bodo) run. It should be set to number of cores avialable.
        clf = SGDClassifier(
            n_jobs=8,
            loss="log",
            max_iter=10,
            early_stopping=False,
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


# ---------------------- SGDRegressor tests ----------------------
@pytest.mark.parametrize("penalty", ["l1", "l2", None])
def test_sgdr(penalty):
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
