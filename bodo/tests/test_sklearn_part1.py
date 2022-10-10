# Copyright (C) 2022 Bodo Inc. All rights reserved.
""" Test supported sklearn.metrics methods"""

import random

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.metrics.pairwise import cosine_similarity

import bodo
from bodo.tests.utils import _get_dist_arg, check_func
from bodo.utils.typing import BodoError

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


def gen_random_strings(n, true_chance, return_pd_series=False, return_pd_array=False):
    """
    Only one of return_pd_series and return_pd_array should be set to true.
    If both are set, a pd.Series is returned. If neither are set,
    a simple list is returned.
    """
    [y_true, y_pred] = gen_random(n, true_chance, return_arrays=False)
    cats = list(set(y_true).union(set(y_pred)))
    choices = {cat: str(chr(ord("a") + i)) for i, cat in enumerate(cats)}
    y_true = [choices[y] for y in y_true]
    y_pred = [choices[y] for y in y_pred]
    if return_pd_series:
        y_true = pd.Series(y_true)
        y_pred = pd.Series(y_pred)
    elif return_pd_array:
        y_true = pd.array(y_true)
        y_pred = pd.array(y_pred)

    return y_true, y_pred, list(choices.values())


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


def gen_random_sample_weights(n, return_arrays=True, integral_sample_weights=False):
    np.random.seed(5)
    sample_weight = np.random.random_sample(size=n)
    if integral_sample_weights:
        sample_weight = (sample_weight * 10).astype(np.int64)
    if not return_arrays:
        sample_weight = list(sample_weight)
    return sample_weight


def gen_random_with_sample_weight(
    n, true_chance, return_arrays=True, integral_sample_weights=False
):
    """
    Wrapper around the gen_random function. This one also has a third
    array/list for sample_weight, each element of which is in (0,1)
    when integral_sample_weights=False or integers in [0, 10) when
    integral_sample_weights=True.
    Returns np arrays if return_arrays=True, else python lists.
    """
    [y_true, y_pred] = gen_random(n, true_chance, return_arrays)
    sample_weight = gen_random_sample_weights(n, return_arrays, integral_sample_weights)
    return [y_true, y_pred, sample_weight]


def gen_random_strings_with_sample_weight(
    n,
    true_chance,
    return_pd_series=False,
    return_pd_array=False,
    integral_sample_weights=False,
):
    """
    Wrapper around the gen_random_strings function. This one also has a fourth
    array/list for sample_weight, each element of which is in (0,1)
    when integral_sample_weights=False or integers in [0, 10) when
    integral_sample_weights=True.
    Returns pd arrays if return_arrays==True, else python lists.
    """
    [y_true, y_pred, choices] = gen_random_strings(
        n, true_chance, return_pd_series, return_pd_array
    )
    sample_weight = gen_random_sample_weights(
        n,
        return_arrays=return_pd_series | return_pd_array,
        integral_sample_weights=integral_sample_weights,
    )
    return [y_true, y_pred, sample_weight, choices]


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
def test_score(data, average, memory_leak_check):
    def test_precision(y_true, y_pred, average):
        return precision_score(y_true, y_pred, average=average)

    def test_recall(y_true, y_pred, average):
        return recall_score(y_true, y_pred, average=average)

    def test_f1(y_true, y_pred, average):
        return f1_score(y_true, y_pred, average=average)

    from sklearn import metrics

    def test_metrics_f1(y_true, y_pred, average):
        """Test to verify that both import styles work for classification metrics"""
        return metrics.f1_score(y_true, y_pred, average=average)

    check_func(test_precision, tuple(data + [average]), is_out_distributed=False)
    check_func(test_recall, tuple(data + [average]), is_out_distributed=False)
    check_func(test_f1, tuple(data + [average]), is_out_distributed=False)
    check_func(test_metrics_f1, tuple(data + [average]), is_out_distributed=False)


# ---------------------------------- log_loss ----------------------------------


@pytest.mark.parametrize(
    "data",
    # Examples taken from https://github.com/scikit-learn/scikit-learn/blob/632384f4314d84d55de1ba8f4234f7cdc8f37824/sklearn/metrics/tests/test_classification.py#L2466
    [
        (
            np.array([0, 0, 0, 1, 1]),
            np.array([[0.5, 0.5], [0.1, 0.9], [0.01, 0.99], [0.9, 0.1], [0.75, 0.25]]),
            None,
            None,
        ),
        (
            np.array([1, 0, 2] * 3),
            np.array([[0.2, 0.7, 0.1], [0.6, 0.2, 0.2], [0.6, 0.1, 0.3]] * 3),
            None,
            None,
        ),
        (
            # We need to pass `y_true` and `y_pred` as np arrays here so that the
            # input can be distributed, and we need to use dtype=object here because
            # allgatherv doesn't yet support string arrays well [BE-2819]
            np.array(["no", "no", "no", "yes", "yes"], dtype=object),
            np.array([[0.5, 0.5], [0.1, 0.9], [0.01, 0.99], [0.9, 0.1], [0.75, 0.25]]),
            None,
            np.array([0.1, 0.1, 0.2, 0.3, 0.3]),
        ),
        (
            # Checking that the labels arg works. Without passing in labels,
            # there are only two classes in y_true but three classes in y_pred
            np.array([1, 2, 2] * 3),
            np.array([[0.2, 0.7, 0.1], [0.6, 0.2, 0.2], [0.6, 0.1, 0.3]] * 3),
            [1, 2, 3],  # Passing a list here so that the `labels` arg is replicated
            None,
        ),
        (
            pd.Series(np.array(["ham", "spam", "spam", "ham"] * 2)),
            pd.DataFrame(
                np.array([[0.2, 0.7], [0.6, 0.5], [0.4, 0.1], [0.7, 0.2]] * 2)
            ),
            None,
            None,
        ),
    ],
)
@pytest.mark.parametrize("normalize", [True, False])
def test_log_loss(data, normalize, memory_leak_check):
    """
    Tests for the sklearn.metrics.log_loss implementation in Bodo.
    """

    def test_log_loss_0(y_true, y_pred, labels, sample_weight):
        return log_loss(y_true, y_pred, labels=labels, sample_weight=sample_weight)

    def test_log_loss_1(y_true, y_pred, labels, sample_weight):
        return log_loss(
            y_true,
            y_pred,
            normalize=normalize,
            labels=labels,
            sample_weight=sample_weight,
        )

    check_func(test_log_loss_0, data, is_out_distributed=False)
    check_func(test_log_loss_1, data, is_out_distributed=False)


def test_log_loss_error():
    def impl(y_true, y_pred):
        return log_loss(y_true, y_pred)

    dist_impl = bodo.jit(distributed=["y_true", "y_pred"])(impl)

    # Number of classes are not equal
    y_true = np.array([1, 0, 2] * 2)
    y_pred = np.array([[0.2, 0.7], [0.6, 0.5], [0.4, 0.1]] * 2)
    with pytest.raises(ValueError):
        dist_impl(y_true, y_pred)

    # Only one label is provided
    y_true = np.array([2, 2] * 2)
    y_pred = np.array([[0.2, 0.7], [0.6, 0.5]] * 2)
    error_str = "The labels array needs to contain at least two labels for log_loss"
    with pytest.raises(ValueError, match=error_str):
        dist_impl(y_true, y_pred)

    # Inconsistent number of samples
    y_true = np.array([2, 2] * 2)
    y_pred = np.array([[0.2, 0.7], [0.6, 0.5]] * 3)
    error_str = "Found input variables with inconsistent numbers of samples"
    with pytest.raises(ValueError, match=error_str):
        dist_impl(y_true, y_pred)


# ---------------------------- cosine_similarity -------------------------------


def gen_random_2d(n, k, seed=None):
    """
    Generate a random 2d ndarray with shape (n, k).

    Args
        n (int): Number of rows
        k (int): Number of columns
        seed (None or int): Random seed, if specified
    """
    np.random.seed(seed)
    out = np.random.rand(n, k)
    return out


@pytest.mark.parametrize(
    "X, Y, pass_Y_as_kwarg",
    [
        (
            gen_random_2d(15, 6, seed=0),
            gen_random_2d(8, 6, seed=1),
            True,
        ),
        (
            gen_random_2d(7, 4, seed=2),
            gen_random_2d(14, 4, seed=3),
            False,
        ),
        (
            gen_random_2d(16, 5, seed=4),
            None,
            True,
        ),
        (
            np.eye(6, dtype=np.float64),
            None,
            True,
        ),
        (
            np.eye(5, dtype=np.int32),
            None,
            False,
        ),
    ],
)
def test_cosine_similarity(X, Y, pass_Y_as_kwarg, memory_leak_check):
    """
    Tests for sklearn.metrics.pairwise.cosine_similarity.
    This is manually tested since the distributions of X and Y are independent.
    """
    # For 100% coverage, check that Y can be passed as a positional arg and kwarg
    if pass_Y_as_kwarg:

        def impl(X, Y):
            out = cosine_similarity(X, Y=Y)
            return out

    else:

        def impl(X, Y):
            out = cosine_similarity(X, Y)
            return out

    # Can't use check_func here since X and Y possibly have different distributions
    # So, we manually test that all four combinations of X and Y distributions
    # are properly handled
    dist_dist_impl = bodo.jit(distributed=["X", "Y"])(impl)
    dist_rep_impl = bodo.jit(distributed=["X"])(impl)
    rep_dist_impl = bodo.jit(distributed=["Y"])(impl)
    rep_rep_impl = bodo.jit(replicated=True)(impl)

    py_output = impl(X, Y)
    bodo_output1 = bodo.allgatherv(dist_dist_impl(_get_dist_arg(X), _get_dist_arg(Y)))
    bodo_output2 = bodo.allgatherv(dist_rep_impl(_get_dist_arg(X), Y))
    bodo_output3 = rep_dist_impl(X, _get_dist_arg(Y))
    bodo_output4 = rep_rep_impl(X, Y)

    assert np.allclose(py_output, bodo_output1, atol=1e-4)
    assert np.allclose(py_output, bodo_output2, atol=1e-4)
    assert np.allclose(py_output, bodo_output3, atol=1e-4)
    assert np.allclose(py_output, bodo_output4, atol=1e-4)


def test_cosine_similarity_unsupported(memory_leak_check):
    """
    Tests for unsupported features in sklearn.metrics.pairwise.cosine_similarity.
    """

    def impl(X):
        out = cosine_similarity(X, dense_output=False)
        return out

    err_msg = "dense_output parameter only supports default value True"
    with pytest.raises(BodoError, match=err_msg):
        X = np.eye(15)
        bodo.jit(distributed=["X"])(impl)(_get_dist_arg(X))


# ----------------------------- accuracy_score ---------------------------------


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
def test_accuracy_score(data, normalize, memory_leak_check):
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

    check_func(
        test_accuracy_score_0, tuple(data[0:2]), only_seq=True, is_out_distributed=False
    )
    check_func(
        test_accuracy_score_1, tuple(data[0:2]), only_seq=True, is_out_distributed=False
    )
    check_func(
        test_accuracy_score_2, tuple(data), only_seq=True, is_out_distributed=False
    )
    check_func(
        test_accuracy_score_3,
        tuple(data),
        is_out_distributed=False,
        only_seq=True,
    )
    check_func(
        test_accuracy_score_4,
        tuple(data),
        is_out_distributed=False,
        only_seq=True,
    )


# TODO: Add memory_leak when bug is solved (curently fails on data0 and data1)
@pytest.mark.parametrize(
    "data",
    [
        gen_random_with_sample_weight(10, 0.5, return_arrays=True),
        gen_random_with_sample_weight(50, 0.7, return_arrays=True),
        gen_random_with_sample_weight(76, 0.3, return_arrays=False),
        gen_random_with_sample_weight(11, 0.43, return_arrays=False),
        gen_random_with_sample_weight(
            10, 0.5, return_arrays=True, integral_sample_weights=True
        ),
    ],
)
@pytest.mark.parametrize(
    "labels",
    # The range of outputs of gen_random_with_sample_weight is [-3, 3]
    [
        None,
        np.arange(-3, 3),
        np.arange(-5, 7),
        np.arange(-2, 2),
    ],
)
@pytest.mark.parametrize(
    "normalize",
    [
        "true",
        "pred",
        "all",
        None,
    ],
)
def test_confusion_matrix(data, labels, normalize, memory_leak_check):
    """
    Tests for the sklearn.metrics.confusion_matrix implementation in Bodo
    with integer labels.
    """

    if labels is not None:
        labels = list(labels)  # To force it to be seen as replicated

    def test_confusion_matrix_0(y_true, y_pred):
        return confusion_matrix(y_true, y_pred)

    def test_confusion_matrix_1(y_true, y_pred):
        return confusion_matrix(y_true, y_pred, normalize=normalize)

    def test_confusion_matrix_2(y_true, y_pred, sample_weight_):
        return confusion_matrix(y_true, y_pred, sample_weight=sample_weight_)

    def test_confusion_matrix_3(y_true, y_pred, labels_):
        return confusion_matrix(y_true, y_pred, labels=labels_)

    def test_confusion_matrix_4(y_true, y_pred, sample_weight_, labels_):
        return confusion_matrix(
            y_true,
            y_pred,
            normalize=normalize,
            sample_weight=sample_weight_,
            labels=labels_,
        )

    def test_confusion_matrix_5(y_true, y_pred, labels_):
        return confusion_matrix(y_true, y_pred, normalize=normalize, labels=labels_)

    check_func(
        test_confusion_matrix_0,
        tuple(data[0:2]),
        only_seq=True,
        is_out_distributed=False,
    )
    check_func(
        test_confusion_matrix_1,
        tuple(data[0:2]),
        only_seq=True,
        is_out_distributed=False,
    )
    check_func(
        test_confusion_matrix_2, tuple(data), only_seq=True, is_out_distributed=False
    )
    check_func(
        test_confusion_matrix_3,
        (data[0], data[1], labels),
        is_out_distributed=False,
        only_seq=True,
    )
    check_func(
        test_confusion_matrix_4,
        (*data, labels),
        is_out_distributed=False,
        only_seq=True,
    )
    check_func(
        test_confusion_matrix_5,
        (data[0], data[1], labels),
        is_out_distributed=False,
        only_seq=True,
    )


# TODO: Add memory_leak
@pytest.mark.parametrize(
    "data",
    [
        gen_random_strings_with_sample_weight(10, 0.5),
        gen_random_strings_with_sample_weight(50, 0.7, return_pd_series=True),
        gen_random_strings_with_sample_weight(76, 0.3, return_pd_array=True),
        gen_random_strings_with_sample_weight(
            10, 0.5, return_pd_series=True, integral_sample_weights=True
        ),
    ],
)
@pytest.mark.parametrize(
    "normalize",
    [
        "true",
        "pred",
        "all",
        None,
    ],
)
def test_confusion_matrix_string_labels(data, normalize, memory_leak_check):
    """
    Tests for the sklearn.metrics.confusion_matrix implementation in Bodo
    with string labels
    """

    [y_true, y_pred, sample_weight, labels] = data

    def test_confusion_matrix_0(y_true, y_pred):
        return confusion_matrix(y_true, y_pred)

    def test_confusion_matrix_1(y_true, y_pred):
        return confusion_matrix(y_true, y_pred, normalize=normalize)

    def test_confusion_matrix_2(y_true, y_pred, sample_weight_):
        return confusion_matrix(y_true, y_pred, sample_weight=sample_weight_)

    def test_confusion_matrix_3(y_true, y_pred, labels_):
        return confusion_matrix(y_true, y_pred, labels=labels_)

    def test_confusion_matrix_4(y_true, y_pred, sample_weight_, labels_):
        return confusion_matrix(
            y_true,
            y_pred,
            normalize=normalize,
            sample_weight=sample_weight_,
            labels=labels_,
        )

    def test_confusion_matrix_5(y_true, y_pred, labels_):
        return confusion_matrix(y_true, y_pred, normalize=normalize, labels=labels_)

    check_func(
        test_confusion_matrix_0,
        (y_true, y_pred),
        only_seq=True,
        is_out_distributed=False,
    )
    check_func(
        test_confusion_matrix_1,
        (y_true, y_pred),
        only_seq=True,
        is_out_distributed=False,
    )
    check_func(
        test_confusion_matrix_2,
        (y_true, y_pred, sample_weight),
        only_seq=True,
        is_out_distributed=False,
    )
    check_func(
        test_confusion_matrix_3,
        (y_true, y_pred, labels),
        only_seq=True,
        is_out_distributed=False,
    )
    check_func(
        test_confusion_matrix_4,
        (y_true, y_pred, sample_weight, labels),
        only_seq=True,
        is_out_distributed=False,
    )
    check_func(
        test_confusion_matrix_5,
        (y_true, y_pred, labels),
        only_seq=True,
        is_out_distributed=False,
    )
    bodo.barrier()
    import gc

    del data
    del y_true, y_pred, sample_weight, labels
    gc.collect()


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

    bodo.barrier()
    import gc

    del data
    gc.collect()


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
        """Test to verify that both import styles work for regression metrics"""
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

    bodo.barrier()
    import gc

    del data
    gc.collect()


def test_r2_score_inconsistent(memory_leak_check):
    """
    Check that appropriate error is raised when number of samples in
    y_true and y_pred are inconsistent
    """

    def test_r2_0(y_true, y_pred):
        return r2_score(y_true, y_pred, multioutput="uniform_average")

    data = gen_random_k_dims(20, 1)

    with pytest.raises(
        ValueError,
        match="inconsistent number of samples",
    ):
        bodo.jit(distributed=["y_true", "y_pred"])(test_r2_0)(
            _get_dist_arg(data[0]), _get_dist_arg(data[1][:-1])
        )

    bodo.barrier()
    import gc

    del data
    gc.collect()
