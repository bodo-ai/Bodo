# Copied and adapted from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/tests/test_forest.py

import random
import time

import numpy as np
import pandas as pd
import pytest
from sklearn import datasets
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
