"""Support scikit-learn metrics methods."""

import sys
import warnings

import numba
import numpy as np
import sklearn.metrics
from mpi4py import MPI
from numba.core import types
from numba.extending import overload
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.utils.validation import column_or_1d

import bodo
from bodo.libs.distributed_api import Reduce_Type
from bodo.utils.typing import (
    BodoError,
    check_unsupported_args,
    get_overload_const_str,
    is_overload_constant_str,
    is_overload_false,
    is_overload_none,
    is_overload_true,
)

this_module = sys.modules[__name__]


def precision_recall_fscore_support_helper(MCM, average):
    def multilabel_confusion_matrix(
        y_true, y_pred, *, sample_weight=None, labels=None, samplewise=False
    ):
        return MCM

    # Dynamic monkey patching: here we temporarily swap scikit-learn's
    # implementation of multilabel_confusion_matrix function for our own. This
    # is done in order to allow us to call sklearn's precision_recall_fscore_support
    # function and thus reuse most of their implementation.
    # The downside of this approach is that it could break in the future with
    # changes in scikit-learn, since we call precision_recall_fscore_support
    # with dummy values, but maybe it is easy to make more robust.
    f = sklearn.metrics._classification.multilabel_confusion_matrix
    result = -1.0
    try:
        sklearn.metrics._classification.multilabel_confusion_matrix = (
            multilabel_confusion_matrix
        )

        result = sklearn.metrics._classification.precision_recall_fscore_support(
            [], [], average=average
        )
    finally:
        sklearn.metrics._classification.multilabel_confusion_matrix = f
    return result


@numba.njit
def precision_recall_fscore_parallel(
    y_true, y_pred, operation, average="binary"
):  # pragma: no cover
    labels = bodo.libs.array_kernels.unique(y_true, parallel=True)
    labels = bodo.allgatherv(labels, False)
    labels = bodo.libs.array_kernels.sort(labels, ascending=True, inplace=False)

    nlabels = len(labels)
    # true positive for each label
    tp_sum = np.zeros(nlabels, np.int64)
    # count of label appearance in y_true
    true_sum = np.zeros(nlabels, np.int64)
    # count of label appearance in y_pred
    pred_sum = np.zeros(nlabels, np.int64)
    label_dict = bodo.hiframes.pd_categorical_ext.get_label_dict_from_categories(labels)
    for i in range(len(y_true)):
        true_sum[label_dict[y_true[i]]] += 1
        if y_pred[i] not in label_dict:
            # TODO: Seems like this warning needs to be printed:
            # sklearn/metrics/_classification.py:1221: UndefinedMetricWarning: Recall is ill-defined and
            # being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control
            # this behavior.
            # _warn_prf(average, modifier, msg_start, len(result))
            # TODO: it looks like the warning is only thrown for recall but I would
            # double-check carefully
            continue
        label = label_dict[y_pred[i]]
        pred_sum[label] += 1
        if y_true[i] == y_pred[i]:
            tp_sum[label] += 1

    # gather global tp_sum, true_sum and pred_sum on every process
    tp_sum = bodo.libs.distributed_api.dist_reduce(
        tp_sum, np.int32(Reduce_Type.Sum.value)
    )
    true_sum = bodo.libs.distributed_api.dist_reduce(
        true_sum, np.int32(Reduce_Type.Sum.value)
    )
    pred_sum = bodo.libs.distributed_api.dist_reduce(
        pred_sum, np.int32(Reduce_Type.Sum.value)
    )

    # see https://github.com/scikit-learn/scikit-learn/blob/e0abd262ea3328f44ae8e612f5b2f2cece7434b6/sklearn/metrics/_classification.py#L526
    fp = pred_sum - tp_sum
    fn = true_sum - tp_sum
    tp = tp_sum
    # see https://github.com/scikit-learn/scikit-learn/blob/e0abd262ea3328f44ae8e612f5b2f2cece7434b6/sklearn/metrics/_classification.py#L541
    tn = y_true.shape[0] - tp - fp - fn

    with numba.objmode(result="float64[:]"):
        # see https://github.com/scikit-learn/scikit-learn/blob/e0abd262ea3328f44ae8e612f5b2f2cece7434b6/sklearn/metrics/_classification.py#L543
        MCM = np.array([tn, fp, fn, tp]).T.reshape(-1, 2, 2)
        if operation == "precision":
            result = precision_recall_fscore_support_helper(MCM, average)[0]
        elif operation == "recall":
            result = precision_recall_fscore_support_helper(MCM, average)[1]
        elif operation == "f1":
            result = precision_recall_fscore_support_helper(MCM, average)[2]
        if average is not None:
            # put result in an array so that the return type of this function
            # is array of floats regardless of value of 'average'
            result = np.array([result])

    return result


@overload(sklearn.metrics.precision_score, no_unliteral=True)
def overload_precision_score(
    y_true,
    y_pred,
    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
    zero_division="warn",
    _is_data_distributed=False,
):
    if is_overload_none(average):
        # this case returns an array of floats, one for each label
        if is_overload_false(_is_data_distributed):

            def _precision_score_impl(
                y_true,
                y_pred,
                labels=None,
                pos_label=1,
                average="binary",
                sample_weight=None,
                zero_division="warn",
                _is_data_distributed=False,
            ):
                # user could pass lists and numba throws error if passing lists
                # to object mode, so we convert to arrays
                y_true = bodo.utils.conversion.coerce_to_array(y_true)
                y_pred = bodo.utils.conversion.coerce_to_array(y_pred)
                with numba.objmode(score="float64[:]"):
                    score = sklearn.metrics.precision_score(
                        y_true,
                        y_pred,
                        labels=labels,
                        pos_label=pos_label,
                        average=average,
                        sample_weight=sample_weight,
                        zero_division=zero_division,
                    )
                return score

            return _precision_score_impl
        else:

            def _precision_score_impl(
                y_true,
                y_pred,
                labels=None,
                pos_label=1,
                average="binary",
                sample_weight=None,
                zero_division="warn",
                _is_data_distributed=False,
            ):
                return precision_recall_fscore_parallel(
                    y_true, y_pred, "precision", average=average
                )

            return _precision_score_impl
    else:
        # this case returns one float
        if is_overload_false(_is_data_distributed):

            def _precision_score_impl(
                y_true,
                y_pred,
                labels=None,
                pos_label=1,
                average="binary",
                sample_weight=None,
                zero_division="warn",
                _is_data_distributed=False,
            ):
                # user could pass lists and numba throws error if passing lists
                # to object mode, so we convert to arrays
                y_true = bodo.utils.conversion.coerce_to_array(y_true)
                y_pred = bodo.utils.conversion.coerce_to_array(y_pred)
                with numba.objmode(score="float64"):
                    score = sklearn.metrics.precision_score(
                        y_true,
                        y_pred,
                        labels=labels,
                        pos_label=pos_label,
                        average=average,
                        sample_weight=sample_weight,
                        zero_division=zero_division,
                    )
                return score

            return _precision_score_impl
        else:

            def _precision_score_impl(
                y_true,
                y_pred,
                labels=None,
                pos_label=1,
                average="binary",
                sample_weight=None,
                zero_division="warn",
                _is_data_distributed=False,
            ):
                score = precision_recall_fscore_parallel(
                    y_true, y_pred, "precision", average=average
                )
                return score[0]

            return _precision_score_impl


@overload(sklearn.metrics.recall_score, no_unliteral=True)
def overload_recall_score(
    y_true,
    y_pred,
    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
    zero_division="warn",
    _is_data_distributed=False,
):
    if is_overload_none(average):
        # this case returns an array of floats, one for each label
        if is_overload_false(_is_data_distributed):

            def _recall_score_impl(
                y_true,
                y_pred,
                labels=None,
                pos_label=1,
                average="binary",
                sample_weight=None,
                zero_division="warn",
                _is_data_distributed=False,
            ):
                # user could pass lists and numba throws error if passing lists
                # to object mode, so we convert to arrays
                y_true = bodo.utils.conversion.coerce_to_array(y_true)
                y_pred = bodo.utils.conversion.coerce_to_array(y_pred)
                with numba.objmode(score="float64[:]"):
                    score = sklearn.metrics.recall_score(
                        y_true,
                        y_pred,
                        labels=labels,
                        pos_label=pos_label,
                        average=average,
                        sample_weight=sample_weight,
                        zero_division=zero_division,
                    )
                return score

            return _recall_score_impl
        else:

            def _recall_score_impl(
                y_true,
                y_pred,
                labels=None,
                pos_label=1,
                average="binary",
                sample_weight=None,
                zero_division="warn",
                _is_data_distributed=False,
            ):
                return precision_recall_fscore_parallel(
                    y_true, y_pred, "recall", average=average
                )

            return _recall_score_impl
    else:
        # this case returns one float
        if is_overload_false(_is_data_distributed):

            def _recall_score_impl(
                y_true,
                y_pred,
                labels=None,
                pos_label=1,
                average="binary",
                sample_weight=None,
                zero_division="warn",
                _is_data_distributed=False,
            ):
                # user could pass lists and numba throws error if passing lists
                # to object mode, so we convert to arrays
                y_true = bodo.utils.conversion.coerce_to_array(y_true)
                y_pred = bodo.utils.conversion.coerce_to_array(y_pred)
                with numba.objmode(score="float64"):
                    score = sklearn.metrics.recall_score(
                        y_true,
                        y_pred,
                        labels=labels,
                        pos_label=pos_label,
                        average=average,
                        sample_weight=sample_weight,
                        zero_division=zero_division,
                    )
                return score

            return _recall_score_impl
        else:

            def _recall_score_impl(
                y_true,
                y_pred,
                labels=None,
                pos_label=1,
                average="binary",
                sample_weight=None,
                zero_division="warn",
                _is_data_distributed=False,
            ):
                score = precision_recall_fscore_parallel(
                    y_true, y_pred, "recall", average=average
                )
                return score[0]

            return _recall_score_impl


@overload(sklearn.metrics.f1_score, no_unliteral=True)
def overload_f1_score(
    y_true,
    y_pred,
    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
    zero_division="warn",
    _is_data_distributed=False,
):
    if is_overload_none(average):
        # this case returns an array of floats, one for each label
        if is_overload_false(_is_data_distributed):

            def _f1_score_impl(
                y_true,
                y_pred,
                labels=None,
                pos_label=1,
                average="binary",
                sample_weight=None,
                zero_division="warn",
                _is_data_distributed=False,
            ):
                # user could pass lists and numba throws error if passing lists
                # to object mode, so we convert to arrays
                y_true = bodo.utils.conversion.coerce_to_array(y_true)
                y_pred = bodo.utils.conversion.coerce_to_array(y_pred)
                with numba.objmode(score="float64[:]"):
                    score = sklearn.metrics.f1_score(
                        y_true,
                        y_pred,
                        labels=labels,
                        pos_label=pos_label,
                        average=average,
                        sample_weight=sample_weight,
                        zero_division=zero_division,
                    )
                return score

            return _f1_score_impl
        else:

            def _f1_score_impl(
                y_true,
                y_pred,
                labels=None,
                pos_label=1,
                average="binary",
                sample_weight=None,
                zero_division="warn",
                _is_data_distributed=False,
            ):
                return precision_recall_fscore_parallel(
                    y_true, y_pred, "f1", average=average
                )

            return _f1_score_impl
    else:
        # this case returns one float
        if is_overload_false(_is_data_distributed):

            def _f1_score_impl(
                y_true,
                y_pred,
                labels=None,
                pos_label=1,
                average="binary",
                sample_weight=None,
                zero_division="warn",
                _is_data_distributed=False,
            ):
                # user could pass lists and numba throws error if passing lists
                # to object mode, so we convert to arrays
                y_true = bodo.utils.conversion.coerce_to_array(y_true)
                y_pred = bodo.utils.conversion.coerce_to_array(y_pred)
                with numba.objmode(score="float64"):
                    score = sklearn.metrics.f1_score(
                        y_true,
                        y_pred,
                        labels=labels,
                        pos_label=pos_label,
                        average=average,
                        sample_weight=sample_weight,
                        zero_division=zero_division,
                    )
                return score

            return _f1_score_impl
        else:

            def _f1_score_impl(
                y_true,
                y_pred,
                labels=None,
                pos_label=1,
                average="binary",
                sample_weight=None,
                zero_division="warn",
                _is_data_distributed=False,
            ):
                score = precision_recall_fscore_parallel(
                    y_true, y_pred, "f1", average=average
                )
                return score[0]

            return _f1_score_impl


def mse_mae_dist_helper(y_true, y_pred, sample_weight, multioutput, metric):
    """
    Helper for distributed mse calculation.
    metric must be one of ['mse', 'mae']
    """

    if metric == "mse":
        # This is basically `np.average((y_true-y_pred)**2, axis=0, weights=sample_weight)`
        # except we get some type-checking like length matching for free from sklearn
        local_raw_values_metric = sklearn.metrics.mean_squared_error(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            multioutput="raw_values",
        )
    elif metric == "mae":
        # This is basically `np.average(np.abs(y_true-y_pred), axis=0, weights=sample_weight)`
        # except we get some type-checking like length matching for free from sklearn
        local_raw_values_metric = sklearn.metrics.mean_absolute_error(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            multioutput="raw_values",
        )
    else:  # pragma: no cover
        raise RuntimeError(
            f"Unrecognized metric {metric}. Must be one of 'mae' and 'mse'"
        )

    comm = MPI.COMM_WORLD
    num_pes = comm.Get_size()

    # Calculate sum of sample weights on each rank
    if sample_weight is not None:
        local_weights_sum = np.sum(sample_weight)
    else:
        local_weights_sum = np.float64(y_true.shape[0])

    # Do an all-gather of all the sample weight sums
    rank_weights = np.zeros(num_pes, dtype=type(local_weights_sum))
    comm.Allgather(local_weights_sum, rank_weights)

    # Do an all-gather of the local metric values
    local_raw_values_metric_by_rank = np.zeros(
        (num_pes, *local_raw_values_metric.shape),
        dtype=local_raw_values_metric.dtype,
    )
    comm.Allgather(local_raw_values_metric, local_raw_values_metric_by_rank)

    # Calculate global metric by doing a weighted average using rank_weights
    global_raw_values_metric = np.average(
        local_raw_values_metric_by_rank, weights=rank_weights, axis=0
    )

    if isinstance(multioutput, str) and multioutput == "raw_values":
        return global_raw_values_metric
    elif isinstance(multioutput, str) and multioutput == "uniform_average":
        return np.average(global_raw_values_metric)
    else:  # multioutput must be weights
        return np.average(global_raw_values_metric, weights=multioutput)


@overload(sklearn.metrics.mean_squared_error, no_unliteral=True)
def overload_mean_squared_error(
    y_true,
    y_pred,
    sample_weight=None,
    multioutput="uniform_average",
    _is_data_distributed=False,
):
    """
    Provide implementations for the mean_squared_error computation.
    If data is not distributed, we simply call sklearn on each rank.
    Else we compute in a distributed way.
    Provide separate impl for case where sample_weight is provided
    vs not provided for type unification purposes.
    """

    if (
        is_overload_constant_str(multioutput)
        and get_overload_const_str(multioutput) == "raw_values"
    ):
        # this case returns an array of floats (one for each dimension)

        if is_overload_none(sample_weight):

            def _mse_impl(
                y_true,
                y_pred,
                sample_weight=None,
                multioutput="uniform_average",
                _is_data_distributed=False,
            ):  # pragma: no cover
                y_true = bodo.utils.conversion.coerce_to_array(y_true)
                y_pred = bodo.utils.conversion.coerce_to_array(y_pred)
                with numba.objmode(err="float64[:]"):
                    if _is_data_distributed:
                        err = mse_mae_dist_helper(
                            y_true,
                            y_pred,
                            sample_weight=sample_weight,
                            multioutput=multioutput,
                            metric="mse",
                        )
                    else:
                        err = sklearn.metrics.mean_squared_error(
                            y_true,
                            y_pred,
                            sample_weight=sample_weight,
                            multioutput=multioutput,
                        )
                return err

            return _mse_impl
        else:

            def _mse_impl(
                y_true,
                y_pred,
                sample_weight=None,
                multioutput="uniform_average",
                _is_data_distributed=False,
            ):  # pragma: no cover
                y_true = bodo.utils.conversion.coerce_to_array(y_true)
                y_pred = bodo.utils.conversion.coerce_to_array(y_pred)
                sample_weight = bodo.utils.conversion.coerce_to_array(sample_weight)
                with numba.objmode(err="float64[:]"):
                    if _is_data_distributed:
                        err = mse_mae_dist_helper(
                            y_true,
                            y_pred,
                            sample_weight=sample_weight,
                            multioutput=multioutput,
                            metric="mse",
                        )
                    else:
                        err = sklearn.metrics.mean_squared_error(
                            y_true,
                            y_pred,
                            sample_weight=sample_weight,
                            multioutput=multioutput,
                        )
                return err

            return _mse_impl

    else:
        # this case returns a single float value

        if is_overload_none(sample_weight):

            def _mse_impl(
                y_true,
                y_pred,
                sample_weight=None,
                multioutput="uniform_average",
                _is_data_distributed=False,
            ):  # pragma: no cover
                y_true = bodo.utils.conversion.coerce_to_array(y_true)
                y_pred = bodo.utils.conversion.coerce_to_array(y_pred)
                with numba.objmode(err="float64"):
                    if _is_data_distributed:
                        err = mse_mae_dist_helper(
                            y_true,
                            y_pred,
                            sample_weight=sample_weight,
                            multioutput=multioutput,
                            metric="mse",
                        )
                    else:
                        err = sklearn.metrics.mean_squared_error(
                            y_true,
                            y_pred,
                            sample_weight=sample_weight,
                            multioutput=multioutput,
                        )
                return err

            return _mse_impl
        else:

            def _mse_impl(
                y_true,
                y_pred,
                sample_weight=None,
                multioutput="uniform_average",
                _is_data_distributed=False,
            ):  # pragma: no cover
                y_true = bodo.utils.conversion.coerce_to_array(y_true)
                y_pred = bodo.utils.conversion.coerce_to_array(y_pred)
                sample_weight = bodo.utils.conversion.coerce_to_array(sample_weight)
                with numba.objmode(err="float64"):
                    if _is_data_distributed:
                        err = mse_mae_dist_helper(
                            y_true,
                            y_pred,
                            sample_weight=sample_weight,
                            multioutput=multioutput,
                            metric="mse",
                        )
                    else:
                        err = sklearn.metrics.mean_squared_error(
                            y_true,
                            y_pred,
                            sample_weight=sample_weight,
                            multioutput=multioutput,
                        )
                return err

            return _mse_impl


@overload(sklearn.metrics.mean_absolute_error, no_unliteral=True)
def overload_mean_absolute_error(
    y_true,
    y_pred,
    sample_weight=None,
    multioutput="uniform_average",
    _is_data_distributed=False,
):
    """
    Provide implementations for the mean_absolute_error computation.
    If data is not distributed, we simply call sklearn on each rank.
    Else we compute in a distributed way.
    Provide separate impl for case where sample_weight is provided
    vs not provided for type unification purposes.
    """

    if (
        is_overload_constant_str(multioutput)
        and get_overload_const_str(multioutput) == "raw_values"
    ):
        # this case returns an array of floats (one for each dimension)

        if is_overload_none(sample_weight):

            def _mae_impl(
                y_true,
                y_pred,
                sample_weight=None,
                multioutput="uniform_average",
                _is_data_distributed=False,
            ):  # pragma: no cover
                y_true = bodo.utils.conversion.coerce_to_array(y_true)
                y_pred = bodo.utils.conversion.coerce_to_array(y_pred)
                with numba.objmode(err="float64[:]"):
                    if _is_data_distributed:
                        err = mse_mae_dist_helper(
                            y_true,
                            y_pred,
                            sample_weight=sample_weight,
                            multioutput=multioutput,
                            metric="mae",
                        )
                    else:
                        err = sklearn.metrics.mean_absolute_error(
                            y_true,
                            y_pred,
                            sample_weight=sample_weight,
                            multioutput=multioutput,
                        )
                return err

            return _mae_impl
        else:

            def _mae_impl(
                y_true,
                y_pred,
                sample_weight=None,
                multioutput="uniform_average",
                _is_data_distributed=False,
            ):  # pragma: no cover
                y_true = bodo.utils.conversion.coerce_to_array(y_true)
                y_pred = bodo.utils.conversion.coerce_to_array(y_pred)
                sample_weight = bodo.utils.conversion.coerce_to_array(sample_weight)
                with numba.objmode(err="float64[:]"):
                    if _is_data_distributed:
                        err = mse_mae_dist_helper(
                            y_true,
                            y_pred,
                            sample_weight=sample_weight,
                            multioutput=multioutput,
                            metric="mae",
                        )
                    else:
                        err = sklearn.metrics.mean_absolute_error(
                            y_true,
                            y_pred,
                            sample_weight=sample_weight,
                            multioutput=multioutput,
                        )
                return err

            return _mae_impl

    else:
        # this case returns a single float value

        if is_overload_none(sample_weight):

            def _mae_impl(
                y_true,
                y_pred,
                sample_weight=None,
                multioutput="uniform_average",
                _is_data_distributed=False,
            ):  # pragma: no cover
                y_true = bodo.utils.conversion.coerce_to_array(y_true)
                y_pred = bodo.utils.conversion.coerce_to_array(y_pred)
                with numba.objmode(err="float64"):
                    if _is_data_distributed:
                        err = mse_mae_dist_helper(
                            y_true,
                            y_pred,
                            sample_weight=sample_weight,
                            multioutput=multioutput,
                            metric="mae",
                        )
                    else:
                        err = sklearn.metrics.mean_absolute_error(
                            y_true,
                            y_pred,
                            sample_weight=sample_weight,
                            multioutput=multioutput,
                        )
                return err

            return _mae_impl
        else:

            def _mae_impl(
                y_true,
                y_pred,
                sample_weight=None,
                multioutput="uniform_average",
                _is_data_distributed=False,
            ):  # pragma: no cover
                y_true = bodo.utils.conversion.coerce_to_array(y_true)
                y_pred = bodo.utils.conversion.coerce_to_array(y_pred)
                sample_weight = bodo.utils.conversion.coerce_to_array(sample_weight)
                with numba.objmode(err="float64"):
                    if _is_data_distributed:
                        err = mse_mae_dist_helper(
                            y_true,
                            y_pred,
                            sample_weight=sample_weight,
                            multioutput=multioutput,
                            metric="mae",
                        )
                    else:
                        err = sklearn.metrics.mean_absolute_error(
                            y_true,
                            y_pred,
                            sample_weight=sample_weight,
                            multioutput=multioutput,
                        )
                return err

            return _mae_impl


# ----------------------------------log_loss-------------------------------------


def log_loss_dist_helper(y_true, y_pred, normalize, sample_weight, labels):
    """
    Helper for distributed log_loss computation.
    Call sklearn on each rank with normalize=False to get
    counts (i.e. sum(accuracy_bits))
    (or sample_weight.T @ accuracy_bits when sample_weight != None
    """
    import pandas as pd

    # Workaround np.all() issue with ArrowStringArray
    # See test_sklearn_metrics.py::test_log_loss
    if isinstance(labels, pd.arrays.ArrowStringArray):
        labels = labels.to_numpy()
    if isinstance(sample_weight, pd.arrays.ArrowExtensionArray):
        sample_weight = sample_weight.to_numpy()

    loss = sklearn.metrics.log_loss(
        y_true,
        y_pred,
        normalize=False,
        sample_weight=sample_weight,
        labels=labels,
    )
    comm = MPI.COMM_WORLD
    loss = comm.allreduce(loss, op=MPI.SUM)
    if normalize:
        sum_of_weights = (
            np.sum(sample_weight) if (sample_weight is not None) else len(y_true)
        )
        sum_of_weights = comm.allreduce(sum_of_weights, op=MPI.SUM)
        loss = loss / sum_of_weights

    return loss


@overload(sklearn.metrics.log_loss, no_unliteral=True)
def overload_log_loss(
    y_true,
    y_pred,
    normalize=True,
    sample_weight=None,
    labels=None,
    _is_data_distributed=False,
):
    """Provide implementations for the log_loss computation.
    If data is not distributed, we simply call sklearn on each rank.
    Else we compute in a distributed way.
    Provide separate impl for case where sample_weight is provided
    vs not provided for type unification purposes.
    """

    if isinstance(y_pred, numba.core.types.containers.List) and isinstance(
        y_pred.dtype, numba.core.types.List
    ):
        raise BodoError(
            "log_loss does not support list input for 2D y_pred, please use numpy array instead"
        )

    func_text = "def _log_loss_impl(\n"
    func_text += "    y_true,\n"
    func_text += "    y_pred,\n"
    func_text += "    normalize=True,\n"
    func_text += "    sample_weight=None,\n"
    func_text += "    labels=None,\n"
    func_text += "    _is_data_distributed=False,\n"
    func_text += "):\n"
    # User could pass lists and numba throws error if passing lists
    # to object mode, so we convert to arrays
    func_text += "    y_true = bodo.utils.conversion.coerce_to_array(y_true)\n"
    if bodo.hiframes.pd_series_ext.is_series_type(y_pred) or isinstance(
        y_pred, numba.core.types.containers.List
    ):
        func_text += "    y_pred = bodo.utils.conversion.coerce_to_array(y_pred)\n"
    # Coerce optional args from lists to arrays if needed
    if not is_overload_none(sample_weight):
        func_text += (
            "    sample_weight = bodo.utils.conversion.coerce_to_array(sample_weight)\n"
        )
    if not is_overload_none(labels):
        func_text += "    labels = bodo.utils.conversion.coerce_to_array(labels)\n"

    # For distributed data, pre-compute labels globally, then call our implementation
    if is_overload_true(_is_data_distributed) and is_overload_none(labels):
        # No need to sort the labels here since LabelBinarizer sorts them anyways
        # when we call sklearn.metrics.log_loss within log_loss_dist_helper
        func_text += (
            "    labels = bodo.libs.array_kernels.unique(y_true, parallel=True)\n"
        )
        func_text += "    labels = bodo.allgatherv(labels, False)\n"

    func_text += "    with numba.objmode(loss='float64'):\n"
    if is_overload_false(_is_data_distributed):
        # For replicated data, directly call sklearn
        func_text += "        loss = sklearn.metrics.log_loss(\n"
    else:
        # For distributed data, pre-compute labels globally, then call our implementation
        func_text += "        loss = log_loss_dist_helper(\n"
    func_text += "            y_true, y_pred, normalize=normalize,\n"
    func_text += "            sample_weight=sample_weight, labels=labels\n"
    func_text += "        )\n"
    func_text += "    return loss\n"
    loc_vars = {}
    exec(func_text, globals(), loc_vars)
    _log_loss_impl = loc_vars["_log_loss_impl"]
    return _log_loss_impl


# ----------------------------- cosine_similarity ------------------------------


@overload(sklearn.metrics.pairwise.cosine_similarity, no_unliteral=True)
def overload_metrics_cosine_similarity(
    X,
    Y=None,
    dense_output=True,
    _is_Y_distributed=False,  # Second-to-last argument specifies if Y is distributed
    _is_X_distributed=False,  # Last argument specifies if X is distributed
):
    """
    Provide implementations for cosine_similarity computation.
    If X is replicated, we simply call sklearn on each rank (after calling
    allgather on Y if it's distributed).
    If X is distributed, we provide a native implementation.
    Our native implementation only supports dense output.
    The current implementation of our algorithm requires Y to be replicated
    internally, as the output distribution matches X's distribution.
    If Y is passed in as distributed, we call allgatherv to replicate it.

    Args
        X (ndarray of shape (n_samples_X, n_features)): Input data
        Y (ndarray of shape (n_samples_Y, n_features) or None): Input data.
          If None, the output will be pairwise similarities between all
          samples in X.
        dense_output (bool): Whether to return dense output even
          when the input is sparse. Only True is supported.

    Returns:
        kernel_matrix (ndarray of shape (n_samples_X, n_samples_Y):
          Pairwise cosine similarities between elements in X and Y.
    """

    # We only support dense_output=True
    args_dict = {
        "dense_output": dense_output,
    }
    args_default_dict = {
        "dense_output": True,
    }
    check_unsupported_args("cosine_similarity", args_dict, args_default_dict, "ml")

    if is_overload_false(_is_X_distributed):
        # If X is replicated, directly call sklearn
        # See note in sklearn_utils_shuffle for how we define an entry in
        # numba.core.types for the underlying numba type of X
        X_type_name = (
            f"metrics_cosine_similarity_type_{numba.core.ir_utils.next_label()}"
        )
        setattr(types, X_type_name, X)

        func_text = "def _metrics_cosine_similarity_impl(\n"
        func_text += "    X, Y=None, dense_output=True, _is_Y_distributed=False, _is_X_distributed=False\n"
        func_text += "):\n"
        if (not is_overload_none(Y)) and is_overload_true(_is_Y_distributed):
            # If Y exists and is distributed, use allgatherv to replicate it before
            # passing to sklearn. No need to do anything in the replicated case.
            func_text += "    Y = bodo.allgatherv(Y)\n"

        # Indexing by [:,::1] instead of [:,:] forces C-style memory layout.
        # This is needed to prevent ambiguous typing of output array
        func_text += "    with numba.objmode(out='float64[:,::1]'):\n"
        func_text += "        out = sklearn.metrics.pairwise.cosine_similarity(\n"
        func_text += "            X, Y, dense_output=dense_output\n"
        func_text += "        )\n"
        func_text += "    return out\n"

        loc_vars = {}
        exec(func_text, globals(), loc_vars)
        _metrics_cosine_similarity_impl = loc_vars["_metrics_cosine_similarity_impl"]

    else:
        # If X is distributed, use native implementation.
        if is_overload_none(Y):
            # If Y is None, use specialized implementation for cosine_similarity(X, Y)
            def _metrics_cosine_similarity_impl(
                X,
                Y=None,
                dense_output=True,
                _is_Y_distributed=False,
                _is_X_distributed=False,
            ):  # pragma: no cover
                # No need to use object mode since our native implementation is
                # fully compilable by numba

                # Normalize each feature within X. No communication is needed as X is
                # distributed across samples (axis=0), not features (axis=1).
                # See the following links for a derivation: `cosine_similarity`
                # calls `normalize`, which calls `extmath.row_norms`.
                # https://github.com/scikit-learn/scikit-learn/blob/1.0.X/sklearn/metrics/pairwise.py#L1253
                # https://github.com/scikit-learn/scikit-learn/blob/1.0.X/sklearn/preprocessing/_data.py#L1823
                # https://github.com/scikit-learn/scikit-learn/blob/1.0.X/sklearn/utils/extmath.py#L51
                X_norms = np.sqrt((X * X).sum(axis=1)).reshape(-1, 1)
                X_normalized = X / X_norms

                # Compute X.T by communicating across all ranks
                X_normalized_T = bodo.allgatherv(X_normalized).T

                # Compute dot product between local chunk of X and X.T
                kernel_matrix = np.dot(X_normalized, X_normalized_T)
                return kernel_matrix

        else:
            # If Y is not None, use implementation of cosine_similarity(X, Y)
            func_text = "def _metrics_cosine_similarity_impl(\n"
            func_text += "    X, Y=None, dense_output=True, _is_Y_distributed=False, _is_X_distributed=False\n"
            func_text += "):\n"
            # No need to use object mode since our native implementation is
            # fully compilable by numba

            # Normalize each feature within X, Y. No communication is needed as X and Y
            # are distributed across samples (axis=0), not features (axis=1).
            # See the following links for a derivation: `cosine_similarity`
            # calls `normalize`, which calls `extmath.row_norms`.
            # https://github.com/scikit-learn/scikit-learn/blob/1.0.X/sklearn/metrics/pairwise.py#L1253
            # https://github.com/scikit-learn/scikit-learn/blob/1.0.X/sklearn/preprocessing/_data.py#L1823
            # https://github.com/scikit-learn/scikit-learn/blob/1.0.X/sklearn/utils/extmath.py#L51
            func_text += "    X_norms = np.sqrt((X * X).sum(axis=1)).reshape(-1, 1)\n"
            func_text += "    X_normalized = X / X_norms\n"
            func_text += "    Y_norms = np.sqrt((Y * Y).sum(axis=1)).reshape(-1, 1)\n"
            func_text += "    Y_normalized = Y / Y_norms\n"

            if is_overload_true(_is_Y_distributed):
                # If Y exists and is distributed, use allgatherv to replicate it before
                # passing to sklearn. No need to do anything in the replicated case.
                # No need to do an explicit None check because we are already in the `else`
                # branch of `is_overload_none(Y)`.
                func_text += "    Y_normalized = bodo.allgatherv(Y_normalized)\n"

            # Compute Y.T, where Y_normalized is already replicated across all ranks
            func_text += "    Y_normalized_T = Y_normalized.T\n"

            # Compute dot product between local chunk of X and Y.T
            func_text += "    kernel_matrix = np.dot(X_normalized, Y_normalized_T)\n"
            func_text += "    return kernel_matrix\n"

            loc_vars = {}
            exec(func_text, globals(), loc_vars)
            _metrics_cosine_similarity_impl = loc_vars[
                "_metrics_cosine_similarity_impl"
            ]

    return _metrics_cosine_similarity_impl


# ------------------------------ accuracy_score --------------------------------


def accuracy_score_dist_helper(y_true, y_pred, normalize, sample_weight):
    """
    Helper for distributed accuracy_score computation.
    Call sklearn on each rank with normalize=False to get
    counts (i.e. sum(accuracy_bits))
    (or sample_weight.T @ accuracy_bits) when sample_weight != None
    """
    score = sklearn.metrics.accuracy_score(
        y_true, y_pred, normalize=False, sample_weight=sample_weight
    )
    comm = MPI.COMM_WORLD
    score = comm.allreduce(score, op=MPI.SUM)
    if normalize:
        sum_of_weights = (
            np.sum(sample_weight) if (sample_weight is not None) else len(y_true)
        )
        sum_of_weights = comm.allreduce(sum_of_weights, op=MPI.SUM)
        score = score / sum_of_weights

    return score


@overload(sklearn.metrics.accuracy_score, no_unliteral=True)
def overload_accuracy_score(
    y_true, y_pred, normalize=True, sample_weight=None, _is_data_distributed=False
):
    """
    Provide implementations for the accuracy_score computation.
    If data is not distributed, we simply call sklearn on each rank.
    Else we compute in a distributed way.
    Provide separate impl for case where sample_weight is provided
    vs not provided for type unification purposes.
    """

    if is_overload_false(_is_data_distributed):
        if is_overload_none(sample_weight):

            def _accuracy_score_impl(
                y_true,
                y_pred,
                normalize=True,
                sample_weight=None,
                _is_data_distributed=False,
            ):  # pragma: no cover
                # user could pass lists and numba throws error if passing lists
                # to object mode, so we convert to arrays
                y_true = bodo.utils.conversion.coerce_to_array(y_true)
                y_pred = bodo.utils.conversion.coerce_to_array(y_pred)

                with numba.objmode(score="float64"):
                    score = sklearn.metrics.accuracy_score(
                        y_true, y_pred, normalize=normalize, sample_weight=sample_weight
                    )
                return score

            return _accuracy_score_impl
        else:

            def _accuracy_score_impl(
                y_true,
                y_pred,
                normalize=True,
                sample_weight=None,
                _is_data_distributed=False,
            ):  # pragma: no cover
                # user could pass lists and numba throws error if passing lists
                # to object mode, so we convert to arrays
                y_true = bodo.utils.conversion.coerce_to_array(y_true)
                y_pred = bodo.utils.conversion.coerce_to_array(y_pred)
                sample_weight = bodo.utils.conversion.coerce_to_array(sample_weight)
                with numba.objmode(score="float64"):
                    score = sklearn.metrics.accuracy_score(
                        y_true, y_pred, normalize=normalize, sample_weight=sample_weight
                    )
                return score

            return _accuracy_score_impl

    else:
        if is_overload_none(sample_weight):

            def _accuracy_score_impl(
                y_true,
                y_pred,
                normalize=True,
                sample_weight=None,
                _is_data_distributed=False,
            ):  # pragma: no cover
                # user could pass lists and numba throws error if passing lists
                # to object mode, so we convert to arrays
                y_true = bodo.utils.conversion.coerce_to_array(y_true)
                y_pred = bodo.utils.conversion.coerce_to_array(y_pred)
                with numba.objmode(score="float64"):
                    score = accuracy_score_dist_helper(
                        y_true,
                        y_pred,
                        normalize=normalize,
                        sample_weight=sample_weight,
                    )
                return score

            return _accuracy_score_impl
        else:

            def _accuracy_score_impl(
                y_true,
                y_pred,
                normalize=True,
                sample_weight=None,
                _is_data_distributed=False,
            ):  # pragma: no cover
                # user could pass lists and numba throws error if passing lists
                # to object mode, so we convert to arrays
                y_true = bodo.utils.conversion.coerce_to_array(y_true)
                y_pred = bodo.utils.conversion.coerce_to_array(y_pred)
                sample_weight = bodo.utils.conversion.coerce_to_array(sample_weight)
                with numba.objmode(score="float64"):
                    score = accuracy_score_dist_helper(
                        y_true,
                        y_pred,
                        normalize=normalize,
                        sample_weight=sample_weight,
                    )
                return score

            return _accuracy_score_impl


def check_consistent_length_parallel(*arrays):
    """
    Checks that the length of each of the arrays is the same (on each rank).
    If it is inconsistent on any rank, the function returns False
    on all ranks.
    Nones are ignored.
    """
    comm = MPI.COMM_WORLD
    is_consistent = True
    lengths = [len(arr) for arr in arrays if arr is not None]
    if len(np.unique(lengths)) > 1:
        is_consistent = False
    is_consistent = comm.allreduce(is_consistent, op=MPI.LAND)
    return is_consistent


def r2_score_dist_helper(
    y_true,
    y_pred,
    sample_weight,
    multioutput,
):
    """
    Helper for distributed r2_score calculation.
    The code is very similar to the sklearn source code for this function,
    except we've made it parallelizable using MPI operations.
    Return values is always an array. When output is a single float value,
    we wrap it around an array, and unwrap it in the caller function.
    """

    comm = MPI.COMM_WORLD

    # Shamelessly copied from https://github.com/scikit-learn/scikit-learn/blob/4afd4fba6/sklearn/metrics/_regression.py#L676-#L723

    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))

    # Check that the lengths are consistent on each process
    if not check_consistent_length_parallel(y_true, y_pred, sample_weight):
        raise ValueError(
            "y_true, y_pred and sample_weight (if not None) have inconsistent number of samples"
        )

    # Check that number of samples > 2, else raise Warning and return nan.
    # This is a pathological scenario and hasn't been heavily tested.
    local_num_samples = y_true.shape[0]
    num_samples = comm.allreduce(local_num_samples, op=MPI.SUM)
    if num_samples < 2:
        warnings.warn(
            "R^2 score is not well-defined with less than two samples.",
            UndefinedMetricWarning,
        )
        return np.array([float("nan")])

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
        weight = sample_weight[:, np.newaxis]
    else:
        # This is the local sample_weight, which is just the number of samples
        sample_weight = np.float64(y_true.shape[0])
        weight = 1.0

    # Calculate the numerator
    local_numerator = (weight * ((y_true - y_pred) ** 2)).sum(axis=0, dtype=np.float64)
    numerator = np.zeros(local_numerator.shape, dtype=local_numerator.dtype)
    comm.Allreduce(local_numerator, numerator, op=MPI.SUM)

    # Calculate the y_true_avg (needed for denominator calculation)
    # Do a weighted sum of y_true for each dimension
    local_y_true_avg_numerator = np.nansum(y_true * weight, axis=0, dtype=np.float64)
    y_true_avg_numerator = np.zeros_like(local_y_true_avg_numerator)
    comm.Allreduce(local_y_true_avg_numerator, y_true_avg_numerator, op=MPI.SUM)

    local_y_true_avg_denominator = np.nansum(sample_weight, dtype=np.float64)
    y_true_avg_denominator = comm.allreduce(local_y_true_avg_denominator, op=MPI.SUM)

    y_true_avg = y_true_avg_numerator / y_true_avg_denominator

    # Calculate the denominator
    local_denominator = (weight * ((y_true - y_true_avg) ** 2)).sum(
        axis=0, dtype=np.float64
    )
    denominator = np.zeros(local_denominator.shape, dtype=local_denominator.dtype)
    comm.Allreduce(local_denominator, denominator, op=MPI.SUM)

    # Compute the output scores, same as sklearn
    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = np.ones([y_true.shape[1] if len(y_true.shape) > 1 else 1])
    output_scores[valid_score] = 1 - (numerator[valid_score] / denominator[valid_score])
    # arbitrary set to zero to avoid -inf scores, having a constant
    # y_true is not interesting for scoring a regression anyway
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            # return scores individually
            return output_scores
        elif multioutput == "uniform_average":
            # passing None as weights results in uniform mean
            avg_weights = None
        elif multioutput == "variance_weighted":
            avg_weights = denominator
            # avoid fail on constant y or one-element arrays.
            # NOTE: This part hasn't been heavily tested
            if not np.any(nonzero_denominator):
                if not np.any(nonzero_numerator):
                    return np.array([1.0])
                else:
                    return np.array([0.0])
    else:
        avg_weights = multioutput

    return np.array([np.average(output_scores, weights=avg_weights)])


@overload(sklearn.metrics.r2_score, no_unliteral=True)
def overload_r2_score(
    y_true,
    y_pred,
    sample_weight=None,
    multioutput="uniform_average",
    _is_data_distributed=False,
):
    """
    Provide implementations for the r2_score computation.
    If data is not distributed, we simply call sklearn on each rank.
    Else we compute in a distributed way.
    Provide separate impl for case where sample_weight is provided
    vs not provided for type unification purposes.
    """

    # Check that value of multioutput is valid
    if is_overload_constant_str(multioutput) and get_overload_const_str(
        multioutput
    ) not in ["raw_values", "uniform_average", "variance_weighted"]:
        raise BodoError(
            f"Unsupported argument {get_overload_const_str(multioutput)} specified for 'multioutput'"
        )

    if (
        is_overload_constant_str(multioutput)
        and get_overload_const_str(multioutput) == "raw_values"
    ):
        # this case returns an array of floats (one for each dimension)

        if is_overload_none(sample_weight):

            def _r2_score_impl(
                y_true,
                y_pred,
                sample_weight=None,
                multioutput="uniform_average",
                _is_data_distributed=False,
            ):  # pragma: no cover
                # user could pass lists and numba throws error if passing lists
                # to object mode, so we convert to arrays
                y_true = bodo.utils.conversion.coerce_to_array(y_true)
                y_pred = bodo.utils.conversion.coerce_to_array(y_pred)
                with numba.objmode(score="float64[:]"):
                    if _is_data_distributed:
                        score = r2_score_dist_helper(
                            y_true,
                            y_pred,
                            sample_weight=sample_weight,
                            multioutput=multioutput,
                        )
                    else:
                        score = sklearn.metrics.r2_score(
                            y_true,
                            y_pred,
                            sample_weight=sample_weight,
                            multioutput=multioutput,
                        )
                return score

            return _r2_score_impl

        else:

            def _r2_score_impl(
                y_true,
                y_pred,
                sample_weight=None,
                multioutput="uniform_average",
                _is_data_distributed=False,
            ):  # pragma: no cover
                # user could pass lists and numba throws error if passing lists
                # to object mode, so we convert to arrays
                y_true = bodo.utils.conversion.coerce_to_array(y_true)
                y_pred = bodo.utils.conversion.coerce_to_array(y_pred)
                sample_weight = bodo.utils.conversion.coerce_to_array(sample_weight)
                with numba.objmode(score="float64[:]"):
                    if _is_data_distributed:
                        score = r2_score_dist_helper(
                            y_true,
                            y_pred,
                            sample_weight=sample_weight,
                            multioutput=multioutput,
                        )
                    else:
                        score = sklearn.metrics.r2_score(
                            y_true,
                            y_pred,
                            sample_weight=sample_weight,
                            multioutput=multioutput,
                        )
                return score

            return _r2_score_impl

    else:
        # this case returns a single float value

        if is_overload_none(sample_weight):

            def _r2_score_impl(
                y_true,
                y_pred,
                sample_weight=None,
                multioutput="uniform_average",
                _is_data_distributed=False,
            ):  # pragma: no cover
                # user could pass lists and numba throws error if passing lists
                # to object mode, so we convert to arrays
                y_true = bodo.utils.conversion.coerce_to_array(y_true)
                y_pred = bodo.utils.conversion.coerce_to_array(y_pred)
                with numba.objmode(score="float64"):
                    if _is_data_distributed:
                        score = r2_score_dist_helper(
                            y_true,
                            y_pred,
                            sample_weight=sample_weight,
                            multioutput=multioutput,
                        )
                        score = score[0]
                    else:
                        score = sklearn.metrics.r2_score(
                            y_true,
                            y_pred,
                            sample_weight=sample_weight,
                            multioutput=multioutput,
                        )
                return score

            return _r2_score_impl

        else:

            def _r2_score_impl(
                y_true,
                y_pred,
                sample_weight=None,
                multioutput="uniform_average",
                _is_data_distributed=False,
            ):  # pragma: no cover
                # user could pass lists and numba throws error if passing lists
                # to object mode, so we convert to arrays
                y_true = bodo.utils.conversion.coerce_to_array(y_true)
                y_pred = bodo.utils.conversion.coerce_to_array(y_pred)
                sample_weight = bodo.utils.conversion.coerce_to_array(sample_weight)
                with numba.objmode(score="float64"):
                    if _is_data_distributed:
                        score = r2_score_dist_helper(
                            y_true,
                            y_pred,
                            sample_weight=sample_weight,
                            multioutput=multioutput,
                        )
                        score = score[0]
                    else:
                        score = sklearn.metrics.r2_score(
                            y_true,
                            y_pred,
                            sample_weight=sample_weight,
                            multioutput=multioutput,
                        )
                return score

            return _r2_score_impl


def confusion_matrix_dist_helper(
    y_true, y_pred, labels=None, sample_weight=None, normalize=None
):
    """
    Distributed confusion matrix computation.
    The basic idea is to compute the confusion matrix locally, and then
    do an element-wise summation across all ranks (which is what
    AllReduce(SUM) does). We don't normalize during local confusion
    matrix computation, instead we normalize after aggregating
    the raw confusion matrices for correctness.
    The rest is to handle edge cases, etc.
    """

    # Shamelessly copied from https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/metrics/_classification.py#L322-#L324
    if normalize not in ["true", "pred", "all", None]:  # pragma: no cover
        raise ValueError("normalize must be one of {'true', 'pred', 'all', None}")

    comm = MPI.COMM_WORLD

    try:
        # Get local confusion_matrix with normalize=None
        local_cm_or_e = sklearn.metrics.confusion_matrix(
            y_true, y_pred, labels=labels, sample_weight=sample_weight, normalize=None
        )
    except ValueError as e:  # pragma: no cover
        local_cm_or_e = e

    # Handle the case where some but not all ranks raise this ValueError
    # https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/metrics/_classification.py#L312
    # This can only occur when the labels are explicitly provided by the user. For instance, say the
    # user provides the labels: [0, 1, 2]. Since the data is distributed, it could be the case
    # that on some rank y_true is say [5, 6, 7], i.e. none of the provided labels are in y_true on
    # this rank. If this happens for all ranks, we need to raise an error, same as sklearn.
    # If it happens on some ranks, but not all, that means the inputs are still valid, we just need to
    # capture the exception and on those ranks that it was raised, the local confusion matrix
    # will be all 0s (and therefore we can do AllReduce(SUM) on it and get the correct result globally).
    error_on_this_rank = (
        isinstance(local_cm_or_e, ValueError)
        and "At least one label specified must be in y_true" in local_cm_or_e.args[0]
    )
    error_on_all_ranks = comm.allreduce(error_on_this_rank, op=MPI.LAND)
    if error_on_all_ranks:  # pragma: no cover
        # If it's an error on all ranks, then reraise it
        raise local_cm_or_e
    elif error_on_this_rank:  # pragma: no cover
        # Determine the dtype based on sample_weight.
        # Choose the accumulator dtype to always have high precision
        dtype = np.int64
        if sample_weight is not None and sample_weight.dtype.kind not in {
            "i",
            "u",
            "b",
        }:
            dtype = np.float64
        # If on this rank, but not all ranks, set it to an all zero array
        local_cm = np.zeros((labels.size, labels.size), dtype=dtype)
    else:
        local_cm = local_cm_or_e

    # Create buffer for global confusion_matrix
    global_cm = np.zeros_like(local_cm)
    # Do element-wise sum across all ranks to get the global confusion_matrix
    comm.Allreduce(local_cm, global_cm)

    # Handle the normalize parameter on the global_cm
    # Shamelessly copied from https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/metrics/_classification.py#L349-#L356
    with np.errstate(all="ignore"):
        if normalize == "true":
            global_cm = global_cm / global_cm.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            global_cm = global_cm / global_cm.sum(axis=0, keepdims=True)
        elif normalize == "all":
            global_cm = global_cm / global_cm.sum()
        global_cm = np.nan_to_num(global_cm)

    return global_cm


@overload(sklearn.metrics.confusion_matrix, no_unliteral=True)
def overload_confusion_matrix(
    y_true,
    y_pred,
    labels=None,
    sample_weight=None,
    normalize=None,
    _is_data_distributed=False,
):
    """
    Provide implementations for the confusion_matrix computation.
    If data is not distributed, we simply call sklearn on each rank.
    Else we compute in a distributed way.
    Provide separate impl for case where sample_weight is provided
    vs not provided for type unificaiton purposes
    """

    func_text = "def _confusion_matrix_impl(\n"
    func_text += "    y_true, y_pred, labels=None,\n"
    func_text += "    sample_weight=None, normalize=None,\n"
    func_text += "    _is_data_distributed=False,\n"
    func_text += "):\n"

    # user could pass lists and numba throws error if passing lists
    # to object mode, so we convert to arrays
    func_text += "    y_true = bodo.utils.conversion.coerce_to_array(y_true)\n"
    func_text += "    y_pred = bodo.utils.conversion.coerce_to_array(y_pred)\n"
    func_text += "    y_true = bodo.utils.typing.decode_if_dict_array(y_true)\n"
    func_text += "    y_pred = bodo.utils.typing.decode_if_dict_array(y_pred)\n"

    cm_dtype = ("int64[:,:]", "np.int64")
    if not is_overload_none(normalize):
        cm_dtype = ("float64[:,:]", "np.float64")
    if not is_overload_none(sample_weight):
        func_text += (
            "    sample_weight = bodo.utils.conversion.coerce_to_array(sample_weight)\n"
        )
        # Choose the accumulator dtype to always have high precision
        # Copied from https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/metrics/_classification.py#L339-#L343
        # (with slight modification)
        # This works for both numpy arrays and pd.Series. Lists are not distributable
        # so we can't support them anyway.
        if numba.np.numpy_support.as_dtype(sample_weight.dtype).kind not in {
            "i",
            "u",
            "b",
        }:
            cm_dtype = ("float64[:,:]", "np.float64")

    if not is_overload_none(labels):
        func_text += "    labels = bodo.utils.conversion.coerce_to_array(labels)\n"
    else:
        if is_overload_true(_is_data_distributed):
            # TODO (Check while benchmarking) Maybe do unique on y_true and y_pred individually first?
            func_text += (
                "    labels = bodo.libs.array_kernels.concat([y_true, y_pred])\n"
            )
            func_text += (
                "    labels = bodo.libs.array_kernels.unique(labels, parallel=True)\n"
            )
            func_text += "    labels = bodo.allgatherv(labels, False)\n"
            func_text += "    labels = bodo.libs.array_kernels.sort(labels, ascending=True, inplace=False)\n"

    func_text += f"    with numba.objmode(cm='{cm_dtype[0]}'):\n"
    if is_overload_false(_is_data_distributed):
        func_text += "      cm = sklearn.metrics.confusion_matrix(\n"
    else:
        func_text += "      cm = confusion_matrix_dist_helper(\n"
    func_text += "        y_true, y_pred, labels=labels,\n"
    func_text += "        sample_weight=sample_weight, normalize=normalize,\n"
    # The datatype of local_cm should already be dtype, but forcing it anyway
    func_text += f"      ).astype({cm_dtype[1]})\n"
    func_text += "    return cm\n"

    loc_vars = {}
    exec(
        func_text,
        globals(),
        loc_vars,
    )
    _confusion_matrix_impl = loc_vars["_confusion_matrix_impl"]
    return _confusion_matrix_impl
