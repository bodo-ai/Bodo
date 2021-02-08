"""Support scikit-learn using object mode of Numba """
import itertools
import numbers
import types as pytypes
import warnings

import numba
import numpy as np
import pandas as pd
import sklearn.cluster
import sklearn.ensemble
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.naive_bayes
import sklearn.svm
import sklearn.utils
from mpi4py import MPI
from numba.core import types
from numba.extending import (
    NativeValue,
    box,
    models,
    overload,
    overload_attribute,
    overload_method,
    register_model,
    typeof_impl,
    unbox,
)
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import hinge_loss, log_loss, mean_squared_error
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing._data import (
    _handle_zeros_in_scale as sklearn_handle_zeros_in_scale,
)
from sklearn.utils.extmath import (
    _safe_accumulator_op as sklearn_safe_accumulator_op,
)
from sklearn.utils.validation import _check_sample_weight, column_or_1d

import bodo
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.hiframes.pd_index_ext import NumericIndexType
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.libs.distributed_api import (
    Reduce_Type,
    create_subcomm_mpi4py,
    get_host_ranks,
    get_nodes_first_ranks,
    get_num_nodes,
)
from bodo.utils.typing import (
    BodoError,
    check_unsupported_args,
    get_overload_const_int,
    get_overload_const_str,
    is_overload_constant_str,
    is_overload_false,
    is_overload_none,
    is_overload_true,
)


def model_fit(m, X, y):
    # TODO check that random_state behavior matches sklearn when
    # the training is distributed (does not apply currently)

    # Add temp var. for global number of trees.
    n_estimators_global = m.n_estimators
    # Split m.n_estimators across Nodes
    hostname = MPI.Get_processor_name()
    nodename_ranks = get_host_ranks()
    nnodes = len(nodename_ranks)
    my_rank = bodo.get_rank()
    m.n_estimators = bodo.libs.distributed_api.get_node_portion(
        n_estimators_global, nnodes, my_rank
    )

    # For each first rank in each node train the model
    if my_rank == (nodename_ranks[hostname])[0]:
        # train model on rank 0
        m.n_jobs = len(nodename_ranks[hostname])
        # To get different seed on each node. Default in MPI seed is generated on master and passed, hence random_state values are repeated.
        if m.random_state is None:
            m.random_state = np.random.RandomState()

        from sklearn.utils import parallel_backend

        with parallel_backend("threading"):
            m.fit(X, y)
        m.n_jobs = 1

    # Gather all trees from each first rank/node to rank 0 within subcomm. Then broadcast to all
    # Get lowest rank in each node
    with numba.objmode(first_rank_node="int32[:]"):
        first_rank_node = get_nodes_first_ranks()
    # Create subcommunicator with these ranks only
    subcomm = create_subcomm_mpi4py(first_rank_node)
    # Gather trees in chunks to avoid reaching memory threshold for MPI.
    if subcomm != MPI.COMM_NULL:
        CHUNK_SIZE = 10
        root_data_size = bodo.libs.distributed_api.get_node_portion(
            n_estimators_global, nnodes, 0
        )
        num_itr = root_data_size // CHUNK_SIZE
        if root_data_size % CHUNK_SIZE != 0:
            num_itr += 1
        forest = []
        for i in range(num_itr):
            trees = subcomm.gather(
                m.estimators_[i * CHUNK_SIZE : i * CHUNK_SIZE + CHUNK_SIZE]
            )
            if my_rank == 0:
                forest += list(itertools.chain.from_iterable(trees))
        if my_rank == 0:
            m.estimators_ = forest

    # rank 0 broadcast of forest to every rank
    comm = MPI.COMM_WORLD
    # Currently, we consider that the model that results from training is
    # replicated, i.e. every rank will have the whole forest.
    # So we gather all the trees (estimators) on every rank.
    # The forest doesn't seem to have a big memory footprint, so this makes
    # sense and allows data-parallel predictions.
    # sklearn with joblib seems to do a task-parallel approach where
    # every worker has all the data and there are n tasks with n being the
    # number of trees
    if my_rank == 0:
        # Do piece-wise broadcast to avoid huge messages that can result
        # from pickling the estimators
        # TODO investigate why the pickled estimators are so large. It
        # doesn't look like the unpickled estimators have a large memory
        # footprint
        for i in range(0, n_estimators_global, 10):
            comm.bcast(m.estimators_[i : i + 10])
        comm.bcast(m.n_classes_)
        comm.bcast(m.n_outputs_)
        comm.bcast(m.classes_)
    # Add no cover becuase coverage report is done by one rank only.
    else:  # pragma: no cover
        estimators = []
        for i in range(0, n_estimators_global, 10):
            estimators += comm.bcast(None)
        m.n_classes_ = comm.bcast(None)
        m.n_outputs_ = comm.bcast(None)
        m.classes_ = comm.bcast(None)
        m.estimators_ = estimators
    assert len(m.estimators_) == n_estimators_global
    m.n_estimators = n_estimators_global


# -----------------------------------------------------------------------------
# Typing and overloads to use RandomForestClassifier inside Bodo functions
# directly via sklearn's API


class BodoRandomForestClassifierType(types.Opaque):
    def __init__(self):
        super(BodoRandomForestClassifierType, self).__init__(
            name="BodoRandomForestClassifierType"
        )


random_forest_classifier_type = BodoRandomForestClassifierType()
types.random_forest_classifier_type = random_forest_classifier_type

register_model(BodoRandomForestClassifierType)(models.OpaqueModel)


@typeof_impl.register(sklearn.ensemble.RandomForestClassifier)
def typeof_random_forest_classifier(val, c):
    return random_forest_classifier_type


@box(BodoRandomForestClassifierType)
def box_random_forest_classifier(typ, val, c):
    # NOTE: we can't just let Python steal a reference since boxing can happen
    # at any point and even in a loop, which can make refcount invalid.
    # see implementation of str.contains and test_contains_regex
    # TODO: investigate refcount semantics of boxing in Numba when variable is returned
    # from function versus not returned
    c.pyapi.incref(val)
    return val


@unbox(BodoRandomForestClassifierType)
def unbox_random_forest_classifier(typ, obj, c):
    # borrow a reference from Python
    c.pyapi.incref(obj)
    return NativeValue(obj)


@overload(sklearn.ensemble.RandomForestClassifier, no_unliteral=True)
def sklearn_ensemble_RandomForestClassifier_overload(
    n_estimators=100,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features="auto",
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=None,
    verbose=0,
    warm_start=False,
    class_weight=None,
    ccp_alpha=0.0,
    max_samples=None,
):

    # TODO n_jobs should be left unspecified so should probably throw an error if used

    def _sklearn_ensemble_RandomForestClassifier_impl(
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="auto",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
    ):  # pragma: no cover
        with numba.objmode(m="random_forest_classifier_type"):
            if random_state is None and get_num_nodes() > 1:
                print("With multinode, fixed random_state seed values are ignored.\n")
                random_state = None
            m = sklearn.ensemble.RandomForestClassifier(
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                min_impurity_split=min_impurity_split,
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=1,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                class_weight=class_weight,
                ccp_alpha=ccp_alpha,
                max_samples=max_samples,
            )
        return m

    return _sklearn_ensemble_RandomForestClassifier_impl


@overload_method(BodoRandomForestClassifierType, "fit", no_unliteral=True)
def overload_model_fit(
    m,
    X,
    y,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    def _model_fit_impl(m, X, y, _is_data_distributed=False):  # pragma: no cover

        # Get lowest rank in each node
        with numba.objmode(first_rank_node="int32[:]"):
            first_rank_node = get_nodes_first_ranks()
        if _is_data_distributed:
            nnodes = len(first_rank_node)
            X = bodo.gatherv(X)
            y = bodo.gatherv(y)
            # Broadcast X, y to first rank in each node
            if nnodes > 1:
                X = bodo.libs.distributed_api.bcast_comm(
                    X, comm_ranks=first_rank_node, nranks=nnodes
                )
                y = bodo.libs.distributed_api.bcast_comm(
                    y, comm_ranks=first_rank_node, nranks=nnodes
                )

        with numba.objmode:
            model_fit(m, X, y)  # return value is m

        bodo.barrier()
        return m

    return _model_fit_impl


def parallel_predict_regression(m, X):
    """
    Implement the regression prediction operation in parallel.
    Each rank has its own copy of the model and predicts for its
    own set of data.
    """

    def _model_predict_impl(m, X):  # pragma: no cover

        with numba.objmode(result="float64[:]"):
            # currently we do data-parallel prediction
            m.n_jobs = 1
            if len(X) == 0:
                # TODO If X is replicated this should be an error (same as sklearn)
                result = np.empty(0, dtype=np.float64)
            else:

                result = m.predict(X).astype(np.float64).flatten()
        return result

    return _model_predict_impl


def parallel_predict(m, X):
    """
    Implement the prediction operation in parallel.
    Each rank has its own copy of the model and predicts for its
    own set of data.
    This strategy is the same for a lot of classifier estimators.
    """

    def _model_predict_impl(m, X):  # pragma: no cover
        with numba.objmode(result="int64[:]"):
            # currently we do data-parallel prediction
            m.n_jobs = 1
            if len(X) == 0:
                # TODO If X is replicated this should be an error (same as sklearn)
                result = np.empty(0, dtype=np.int64)
            else:
                result = m.predict(X).astype(np.int64).flatten()
        return result

    return _model_predict_impl


def parallel_score(
    m,
    X,
    y,
    sample_weight=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """
    Implement the score operation in parallel.
    Each rank has its own copy of the model and
    calculates the score for its own set of data.
    Then, gather and get mean of all scores.
    This strategy is the same for a lot of estimators.
    """

    def _model_score_impl(
        m, X, y, sample_weight=None, _is_data_distributed=False
    ):  # pragma: no cover

        with numba.objmode(result="float64[:]"):
            result = m.score(X, y, sample_weight=sample_weight)
            if _is_data_distributed:
                # replicate result so that the average is weighted based on
                # the data size on each rank
                result = np.full(len(y), result)
            else:
                result = np.array([result])
        if _is_data_distributed:
            result = bodo.allgatherv(result)
        return result.mean()

    return _model_score_impl


@overload_method(BodoRandomForestClassifierType, "predict", no_unliteral=True)
def overload_model_predict(m, X):
    """Overload Random Forest Classifier predict. (Data parallelization)"""
    return parallel_predict(m, X)


@overload_method(BodoRandomForestClassifierType, "score", no_unliteral=True)
def overload_model_score(
    m,
    X,
    y,
    sample_weight=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """Overload Random Forest Classifier score."""
    return parallel_score(m, X, y, sample_weight, _is_data_distributed)


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
    y_true, y_pred, average="binary", _is_data_distributed=False
):

    if is_overload_none(average):
        # this case returns an array of floats, one for each label
        if is_overload_false(_is_data_distributed):

            def _precision_score_impl(
                y_true, y_pred, average="binary", _is_data_distributed=False
            ):
                # user could pass lists and numba throws error if passing lists
                # to object mode, so we convert to arrays
                y_true = bodo.utils.conversion.coerce_to_array(y_true)
                y_pred = bodo.utils.conversion.coerce_to_array(y_pred)
                with numba.objmode(score="float64[:]"):
                    score = sklearn.metrics.precision_score(
                        y_true, y_pred, average=average
                    )
                return score

            return _precision_score_impl
        else:

            def _precision_score_impl(
                y_true, y_pred, average="binary", _is_data_distributed=False
            ):
                return precision_recall_fscore_parallel(
                    y_true, y_pred, "precision", average=average
                )

            return _precision_score_impl
    else:
        # this case returns one float
        if is_overload_false(_is_data_distributed):

            def _precision_score_impl(
                y_true, y_pred, average="binary", _is_data_distributed=False
            ):
                # user could pass lists and numba throws error if passing lists
                # to object mode, so we convert to arrays
                y_true = bodo.utils.conversion.coerce_to_array(y_true)
                y_pred = bodo.utils.conversion.coerce_to_array(y_pred)
                with numba.objmode(score="float64"):
                    score = sklearn.metrics.precision_score(
                        y_true, y_pred, average=average
                    )
                return score

            return _precision_score_impl
        else:

            def _precision_score_impl(
                y_true, y_pred, average="binary", _is_data_distributed=False
            ):
                score = precision_recall_fscore_parallel(
                    y_true, y_pred, "precision", average=average
                )
                return score[0]

            return _precision_score_impl


@overload(sklearn.metrics.recall_score, no_unliteral=True)
def overload_recall_score(y_true, y_pred, average="binary", _is_data_distributed=False):

    if is_overload_none(average):
        # this case returns an array of floats, one for each label
        if is_overload_false(_is_data_distributed):

            def _recall_score_impl(
                y_true, y_pred, average="binary", _is_data_distributed=False
            ):
                # user could pass lists and numba throws error if passing lists
                # to object mode, so we convert to arrays
                y_true = bodo.utils.conversion.coerce_to_array(y_true)
                y_pred = bodo.utils.conversion.coerce_to_array(y_pred)
                with numba.objmode(score="float64[:]"):
                    score = sklearn.metrics.recall_score(
                        y_true, y_pred, average=average
                    )
                return score

            return _recall_score_impl
        else:

            def _recall_score_impl(
                y_true, y_pred, average="binary", _is_data_distributed=False
            ):
                return precision_recall_fscore_parallel(
                    y_true, y_pred, "recall", average=average
                )

            return _recall_score_impl
    else:
        # this case returns one float
        if is_overload_false(_is_data_distributed):

            def _recall_score_impl(
                y_true, y_pred, average="binary", _is_data_distributed=False
            ):
                # user could pass lists and numba throws error if passing lists
                # to object mode, so we convert to arrays
                y_true = bodo.utils.conversion.coerce_to_array(y_true)
                y_pred = bodo.utils.conversion.coerce_to_array(y_pred)
                with numba.objmode(score="float64"):
                    score = sklearn.metrics.recall_score(
                        y_true, y_pred, average=average
                    )
                return score

            return _recall_score_impl
        else:

            def _recall_score_impl(
                y_true, y_pred, average="binary", _is_data_distributed=False
            ):
                score = precision_recall_fscore_parallel(
                    y_true, y_pred, "recall", average=average
                )
                return score[0]

            return _recall_score_impl


@overload(sklearn.metrics.f1_score, no_unliteral=True)
def overload_f1_score(y_true, y_pred, average="binary", _is_data_distributed=False):

    if is_overload_none(average):
        # this case returns an array of floats, one for each label
        if is_overload_false(_is_data_distributed):

            def _f1_score_impl(
                y_true, y_pred, average="binary", _is_data_distributed=False
            ):
                # user could pass lists and numba throws error if passing lists
                # to object mode, so we convert to arrays
                y_true = bodo.utils.conversion.coerce_to_array(y_true)
                y_pred = bodo.utils.conversion.coerce_to_array(y_pred)
                with numba.objmode(score="float64[:]"):
                    score = sklearn.metrics.f1_score(y_true, y_pred, average=average)
                return score

            return _f1_score_impl
        else:

            def _f1_score_impl(
                y_true, y_pred, average="binary", _is_data_distributed=False
            ):
                return precision_recall_fscore_parallel(
                    y_true, y_pred, "f1", average=average
                )

            return _f1_score_impl
    else:
        # this case returns one float
        if is_overload_false(_is_data_distributed):

            def _f1_score_impl(
                y_true, y_pred, average="binary", _is_data_distributed=False
            ):
                # user could pass lists and numba throws error if passing lists
                # to object mode, so we convert to arrays
                y_true = bodo.utils.conversion.coerce_to_array(y_true)
                y_pred = bodo.utils.conversion.coerce_to_array(y_pred)
                with numba.objmode(score="float64"):
                    score = sklearn.metrics.f1_score(y_true, y_pred, average=average)
                return score

            return _f1_score_impl
        else:

            def _f1_score_impl(
                y_true, y_pred, average="binary", _is_data_distributed=False
            ):
                score = precision_recall_fscore_parallel(
                    y_true, y_pred, "f1", average=average
                )
                return score[0]

            return _f1_score_impl


def mse_mae_dist_helper(y_true, y_pred, sample_weight, multioutput, squared, metric):
    """
    Helper for distributed mse calculation.
    metric must be one of ['mse', 'mae']
    squared: only for mse
    """

    if metric == "mse":
        # This is basically `np.average((y_true-y_pred)**2, axis=0, weights=sample_weight)`
        # except we get some type-checking like length matching for free from sklearn
        local_raw_values_metric = sklearn.metrics.mean_squared_error(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            multioutput="raw_values",
            squared=True,
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

    # Element-wise sqrt if squared=False in case of mse
    if metric == "mse" and (not squared):
        global_raw_values_metric = np.sqrt(global_raw_values_metric)

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
    squared=True,
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
                squared=True,
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
                            squared=squared,
                            metric="mse",
                        )
                    else:
                        err = sklearn.metrics.mean_squared_error(
                            y_true,
                            y_pred,
                            sample_weight=sample_weight,
                            multioutput=multioutput,
                            squared=squared,
                        )
                return err

            return _mse_impl
        else:

            def _mse_impl(
                y_true,
                y_pred,
                sample_weight=None,
                multioutput="uniform_average",
                squared=True,
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
                            squared=squared,
                            metric="mse",
                        )
                    else:
                        err = sklearn.metrics.mean_squared_error(
                            y_true,
                            y_pred,
                            sample_weight=sample_weight,
                            multioutput=multioutput,
                            squared=squared,
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
                squared=True,
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
                            squared=squared,
                            metric="mse",
                        )
                    else:
                        err = sklearn.metrics.mean_squared_error(
                            y_true,
                            y_pred,
                            sample_weight=sample_weight,
                            multioutput=multioutput,
                            squared=squared,
                        )
                return err

            return _mse_impl
        else:

            def _mse_impl(
                y_true,
                y_pred,
                sample_weight=None,
                multioutput="uniform_average",
                squared=True,
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
                            squared=squared,
                            metric="mse",
                        )
                    else:
                        err = sklearn.metrics.mean_squared_error(
                            y_true,
                            y_pred,
                            sample_weight=sample_weight,
                            multioutput=multioutput,
                            squared=squared,
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
                            squared=True,  # Is ignored when metric = mae
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
                            squared=True,  # Is ignored when metric = mae
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
                            squared=True,  # Is ignored when metric = mae
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
                            squared=True,  # Is ignored when metric = mae
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


# -------------------------------------SGDRegressor----------------------------------------
# Support sklearn.linear_model.SGDRegressorusing object mode of Numba
# Linear regression: sklearn.linear_model.SGDRegressor(loss="squared_loss", penalty=None)
# Ridge regression: sklearn.linear_model.SGDRegressor(loss="squared_loss", penalty='l2')
# Lasso: sklearn.linear_model.SGDRegressor(loss="squared_loss", penalty='l1')

# -----------------------------------------------------------------------------
# Typing and overloads to use SGDRegressor inside Bodo functions
# directly via sklearn's API


class BodoSGDRegressorType(types.Opaque):
    def __init__(self):
        super(BodoSGDRegressorType, self).__init__(name="BodoSGDRegressorType")


sgd_regressor_type = BodoSGDRegressorType()
types.sgd_regressor_type = sgd_regressor_type

register_model(BodoSGDRegressorType)(models.OpaqueModel)


@typeof_impl.register(sklearn.linear_model.SGDRegressor)
def typeof_sgd_regressor(val, c):
    return sgd_regressor_type


@box(BodoSGDRegressorType)
def box_sgd_regressor(typ, val, c):
    # See note in box_random_forest_classifier
    c.pyapi.incref(val)
    return val


@unbox(BodoSGDRegressorType)
def unbox_sgd_regressor(typ, obj, c):
    # borrow a reference from Python
    c.pyapi.incref(obj)
    return NativeValue(obj)


@overload(sklearn.linear_model.SGDRegressor, no_unliteral=True)
def sklearn_linear_model_SGDRegressor_overload(
    loss="squared_loss",
    penalty="l2",
    alpha=0.0001,
    l1_ratio=0.15,
    fit_intercept=True,
    max_iter=1000,
    tol=0.001,
    shuffle=True,
    verbose=0,
    epsilon=0.1,
    random_state=None,
    learning_rate="invscaling",
    eta0=0.01,
    power_t=0.25,
    early_stopping=False,
    validation_fraction=0.1,
    n_iter_no_change=5,
    warm_start=False,
    average=False,
):
    def _sklearn_linear_model_SGDRegressor_impl(
        loss="squared_loss",
        penalty="l2",
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        tol=0.001,
        shuffle=True,
        verbose=0,
        epsilon=0.1,
        random_state=None,
        learning_rate="invscaling",
        eta0=0.01,
        power_t=0.25,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        warm_start=False,
        average=False,
    ):  # pragma: no cover
        with numba.objmode(m="sgd_regressor_type"):
            m = sklearn.linear_model.SGDRegressor(
                loss=loss,
                penalty=penalty,
                alpha=alpha,
                l1_ratio=l1_ratio,
                fit_intercept=fit_intercept,
                max_iter=max_iter,
                tol=tol,
                shuffle=shuffle,
                verbose=verbose,
                epsilon=epsilon,
                random_state=random_state,
                learning_rate=learning_rate,
                eta0=eta0,
                power_t=power_t,
                early_stopping=early_stopping,
                validation_fraction=validation_fraction,
                n_iter_no_change=n_iter_no_change,
                warm_start=warm_start,
                average=average,
            )
        return m

    return _sklearn_linear_model_SGDRegressor_impl


@overload_method(BodoSGDRegressorType, "fit", no_unliteral=True)
def overload_sgdr_model_fit(
    m,
    X,
    y,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    def _model_sgdr_fit_impl(m, X, y, _is_data_distributed=False):  # pragma: no cover

        # TODO: Rebalance the data X and y to be the same size on every rank
        with numba.objmode(m="sgd_regressor_type"):
            m = fit_sgd(m, X, y, _is_data_distributed)

        bodo.barrier()

        return m

    return _model_sgdr_fit_impl


@overload_method(BodoSGDRegressorType, "predict", no_unliteral=True)
def overload_sgdr_model_predict(m, X):
    """Overload SGDRegressor predict. (Data parallelization)"""
    return parallel_predict_regression(m, X)


@overload_method(BodoSGDRegressorType, "score", no_unliteral=True)
def overload_sgdr_model_score(
    m,
    X,
    y,
    sample_weight=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """Overload SGDRegressor score."""
    return parallel_score(m, X, y, sample_weight, _is_data_distributed)


# -------------------------------------SGDClassifier----------------------------------------
# Support sklearn.linear_model.SGDClassifier using object mode of Numba
# The model it fits can be controlled with the loss parameter; by default, it fits a linear support vector machine (SVM).
# Logistic regression (loss='log')
# -----------------------------------------------------------------------------

# Typing and overloads to use SGDClassifier inside Bodo functions
# directly via sklearn's API
class BodoSGDClassifierType(types.Opaque):
    def __init__(self):
        super(BodoSGDClassifierType, self).__init__(name="BodoSGDClassifierType")


sgd_classifier_type = BodoSGDClassifierType()
types.sgd_classifier_type = sgd_classifier_type

register_model(BodoSGDClassifierType)(models.OpaqueModel)


@typeof_impl.register(sklearn.linear_model.SGDClassifier)
def typeof_sgd_classifier(val, c):
    return sgd_classifier_type


@box(BodoSGDClassifierType)
def box_sgd_classifier(typ, val, c):
    # NOTE: we can't just let Python steal a reference since boxing can happen
    # at any point and even in a loop, which can make refcount invalid.
    # see implementation of str.contains and test_contains_regex
    # TODO: investigate refcount semantics of boxing in Numba when variable is returned
    # from function versus not returned
    c.pyapi.incref(val)
    return val


@unbox(BodoSGDClassifierType)
def unbox_sgd_classifier(typ, obj, c):
    # borrow a reference from Python
    c.pyapi.incref(obj)
    return NativeValue(obj)


@overload(sklearn.linear_model.SGDClassifier, no_unliteral=True)
def sklearn_linear_model_SGDClassifier_overload(
    loss="hinge",
    penalty="l2",
    alpha=0.0001,
    l1_ratio=0.15,
    fit_intercept=True,
    max_iter=1000,
    tol=0.001,
    shuffle=True,
    verbose=0,
    epsilon=0.1,
    n_jobs=None,
    random_state=None,
    learning_rate="optimal",
    eta0=0.0,
    power_t=0.5,
    early_stopping=False,
    validation_fraction=0.1,
    n_iter_no_change=5,
    class_weight=None,
    warm_start=False,
    average=False,
):
    def _sklearn_linear_model_SGDClassifier_impl(
        loss="hinge",
        penalty="l2",
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        tol=0.001,
        shuffle=True,
        verbose=0,
        epsilon=0.1,
        n_jobs=None,
        random_state=None,
        learning_rate="optimal",
        eta0=0.0,
        power_t=0.5,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        class_weight=None,
        warm_start=False,
        average=False,
    ):  # pragma: no cover
        with numba.objmode(m="sgd_classifier_type"):
            m = sklearn.linear_model.SGDClassifier(
                loss=loss,
                penalty=penalty,
                alpha=alpha,
                l1_ratio=l1_ratio,
                fit_intercept=fit_intercept,
                max_iter=max_iter,
                tol=tol,
                shuffle=shuffle,
                verbose=verbose,
                epsilon=epsilon,
                n_jobs=n_jobs,
                random_state=random_state,
                learning_rate=learning_rate,
                eta0=eta0,
                power_t=power_t,
                early_stopping=early_stopping,
                validation_fraction=validation_fraction,
                n_iter_no_change=n_iter_no_change,
                class_weight=class_weight,
                warm_start=warm_start,
                average=average,
            )
        return m

    return _sklearn_linear_model_SGDClassifier_impl


def fit_sgd(m, X, y, y_classes=None, _is_data_distributed=False):
    """Fit a linear model classifier using SGD (parallel version)"""
    comm = MPI.COMM_WORLD
    # Get size of data on each rank
    total_datasize = comm.allreduce(len(X), op=MPI.SUM)
    rank_weight = len(X) / total_datasize
    nranks = comm.Get_size()
    m.n_jobs = 1
    # Currently early_stopping must be False.
    m.early_stopping = False
    best_loss = np.inf
    no_improvement_count = 0
    # TODO: Add other loss cases
    if m.loss == "hinge":
        loss_func = hinge_loss
    elif m.loss == "log":
        loss_func = log_loss
    elif m.loss == "squared_loss":
        loss_func = mean_squared_error
    else:
        raise ValueError("loss {} not supported".format(m.loss))

    regC = False
    if isinstance(m, sklearn.linear_model.SGDRegressor):
        regC = True
    for _ in range(m.max_iter):
        if regC:
            m.partial_fit(X, y)
        else:
            m.partial_fit(X, y, classes=y_classes)
        # Can be removed when rebalancing is done. Now, we have to give more weight to ranks with more data
        m.coef_ = m.coef_ * rank_weight
        m.coef_ = comm.allreduce(m.coef_, op=MPI.SUM)
        m.intercept_ = m.intercept_ * rank_weight
        m.intercept_ = comm.allreduce(m.intercept_, op=MPI.SUM)
        if regC:
            y_pred = m.predict(X)
            cur_loss = loss_func(y, y_pred)
        else:
            y_pred = m.decision_function(X)
            cur_loss = loss_func(y, y_pred, labels=y_classes)
        cur_loss_sum = comm.allreduce(cur_loss, op=MPI.SUM)
        cur_loss = cur_loss_sum / nranks
        # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/linear_model/_sgd_fast.pyx#L620
        if m.tol > np.NINF and cur_loss > best_loss - m.tol * total_datasize:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
        if cur_loss < best_loss:
            best_loss = cur_loss
        if no_improvement_count >= m.n_iter_no_change:
            break

    return m


@overload_method(BodoSGDClassifierType, "fit", no_unliteral=True)
def overload_sgdc_model_fit(
    m,
    X,
    y,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """
    Provide implementations for the fit function.
    In case input is replicated, we simply call sklearn,
    else we use partial_fit on each rank then use we re-compute the attributes using MPI operations.
    """
    if is_overload_true(_is_data_distributed):

        def _model_sgdc_fit_impl(
            m, X, y, _is_data_distributed=False
        ):  # pragma: no cover

            # TODO: Rebalance the data X and y to be the same size on every rank
            # y has to be an array
            y_classes = bodo.libs.array_kernels.unique(y)
            y_classes = bodo.allgatherv(y_classes, False)

            with numba.objmode(m="sgd_classifier_type"):
                m = fit_sgd(m, X, y, y_classes, _is_data_distributed)

            return m

        return _model_sgdc_fit_impl
    else:
        # If replicated, then just call sklearn
        def _model_sgdc_fit_impl(
            m, X, y, _is_data_distributed=False
        ):  # pragma: no cover
            with numba.objmode(m="sgd_classifier_type"):
                m = m.fit(X, y)
            return m

        return _model_sgdc_fit_impl


@overload_method(BodoSGDClassifierType, "predict", no_unliteral=True)
def overload_sgdc_model_predict(m, X):
    """Overload SGDClassifier predict. (Data parallelization)"""
    return parallel_predict(m, X)


@overload_method(BodoSGDClassifierType, "score", no_unliteral=True)
def overload_sgdc_model_score(
    m,
    X,
    y,
    sample_weight=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """Overload SGDClassifier score."""
    return parallel_score(m, X, y, sample_weight, _is_data_distributed)


@overload_attribute(BodoSGDClassifierType, "coef_")
def get_sgdc_coef(m):
    """ Overload coef_ attribute to be accessible inside bodo.jit """

    def impl(m):  # pragma: no cover
        with numba.objmode(result="float64[:,:]"):
            result = m.coef_
        return result

    return impl


# --------------------------------------------------------------------------------------------------#
# --------------------------------------- K-Means --------------------------------------------------#
# Support for sklearn.cluster.KMeans using objmode. We implement a basic wrapper around sklearn's
# implementation of KMeans.
# --------------------------------------------------------------------------------------------------#


class BodoKMeansClusteringType(types.Opaque):
    def __init__(self):
        super(BodoKMeansClusteringType, self).__init__(name="BodoKMeansClusteringType")


kmeans_clustering_type = BodoKMeansClusteringType()
types.kmeans_clustering_type = kmeans_clustering_type

register_model(BodoKMeansClusteringType)(models.OpaqueModel)


@typeof_impl.register(sklearn.cluster.KMeans)
def typeof_kmeans_clustering(val, c):
    return kmeans_clustering_type


@box(BodoKMeansClusteringType)
def box_kmeans_clustering(typ, val, c):
    # See note in box_random_forest_classifier
    c.pyapi.incref(val)
    return val


@unbox(BodoKMeansClusteringType)
def unbox_kmeans_clustering(typ, obj, c):
    # borrow reference from Python
    c.pyapi.incref(obj)
    return NativeValue(obj)


@overload(sklearn.cluster.KMeans, no_unliteral=True)
def sklearn_cluster_kmeans_overload(
    n_clusters=8,
    init="k-means++",
    n_init=10,
    max_iter=300,
    tol=1e-4,
    precompute_distances="deprecated",
    verbose=0,
    random_state=None,
    copy_x=True,
    n_jobs="deprecated",
    algorithm="auto",
):
    def _sklearn_cluster_kmeans_impl(
        n_clusters=8,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        precompute_distances="deprecated",
        verbose=0,
        random_state=None,
        copy_x=True,
        n_jobs="deprecated",
        algorithm="auto",
    ):  # pragma: no cover

        with numba.objmode(m="kmeans_clustering_type"):
            m = sklearn.cluster.KMeans(
                n_clusters=n_clusters,
                init=init,
                n_init=n_init,
                max_iter=max_iter,
                tol=tol,
                precompute_distances=precompute_distances,
                verbose=verbose,
                random_state=random_state,
                copy_x=copy_x,
                n_jobs=n_jobs,
                algorithm=algorithm,
            )
        return m

    return _sklearn_cluster_kmeans_impl


def kmeans_fit_helper(
    m, len_X, all_X, all_sample_weight, _is_data_distributed
):  # pragma: no cover
    """
    The KMeans algorithm is highly parallelizable.
    The training (fit) is already parallelized by Sklearn using OpenMP (for a single node)
    Therefore, we gather the data on rank0 and call sklearn's fit function
    which parallelizes the operation.
    """
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()

    hostname = MPI.Get_processor_name()
    nodename_ranks = get_host_ranks()
    orig_njobs = m.n_jobs if hasattr(m, "n_jobs") else None
    orig_nthreads = m._n_threads if hasattr(m, "_n_threads") else None

    # We run on only rank0, but we want that rank to use all the cores
    m.n_jobs = len(nodename_ranks[hostname])
    m._n_threads = len(nodename_ranks[hostname])

    # Call Sklearn's fit on the gathered data
    if my_rank == 0:
        m.fit(X=all_X, y=None, sample_weight=all_sample_weight)

    # Broadcast the public attributes of the model that must be replicated
    if my_rank == 0:
        comm.bcast(m.cluster_centers_)
        comm.bcast(m.inertia_)
        comm.bcast(m.n_iter_)
    else:
        # Acts as a barriers too
        m.cluster_centers_ = comm.bcast(None)
        m.inertia_ = comm.bcast(None)
        m.n_iter_ = comm.bcast(None)

    # Scatter the m.labels_ if _is_data_distributed
    if _is_data_distributed:
        X_counts = comm.allgather(len_X)
        if my_rank == 0:
            displs = np.empty(len(X_counts) + 1, dtype=int)
            np.cumsum(X_counts, out=displs[1:])
            displs[0] = 0
            send_data = [
                m.labels_[displs[r] : displs[r + 1]] for r in range(len(X_counts))
            ]
            my_labels = comm.scatter(send_data)
        else:
            my_labels = comm.scatter(None)
        m.labels_ = my_labels
    else:
        if my_rank == 0:
            comm.bcast(m.labels_)
        else:
            m.labels_ = comm.bcast(None)

    # Restore
    m.n_jobs = orig_njobs
    m._n_threads = orig_nthreads

    return m


@overload_method(BodoKMeansClusteringType, "fit", no_unliteral=True)
def overload_kmeans_clustering_fit(
    m,
    X,
    y=None,
    sample_weight=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    def _cluster_kmeans_fit_impl(
        m, X, y=None, sample_weight=None, _is_data_distributed=False
    ):  # pragma: no cover

        # If data is distributed, gather it on rank0
        # since that's where we call fit
        if _is_data_distributed:
            all_X = bodo.gatherv(X)
            if sample_weight is not None:
                all_sample_weight = bodo.gatherv(sample_weight)
            else:
                all_sample_weight = None
        else:
            all_X = X
            all_sample_weight = sample_weight

        with numba.objmode(m="kmeans_clustering_type"):
            m = kmeans_fit_helper(
                m, len(X), all_X, all_sample_weight, _is_data_distributed
            )

        return m

    return _cluster_kmeans_fit_impl


def kmeans_predict_helper(m, X, sample_weight):
    """
    We implement the prediction operation in parallel.
    Each rank has its own copy of the KMeans model and predicts for its
    own set of data.
    """

    # Get original n_threads value if it exists
    orig_nthreads = m._n_threads if hasattr(m, "_n_threads") else None
    orig_njobs = m.n_jobs if hasattr(m, "n_jobs") else None
    m._n_threads = 1
    m.n_jobs = 1

    if len(X) == 0:
        # TODO If X is replicated this should be an error (same as sklearn)
        preds = np.empty(0, dtype=np.int64)
    else:
        preds = m.predict(X, sample_weight).astype(np.int64).flatten()

    # Restore
    m._n_threads = orig_nthreads
    m.n_jobs = orig_njobs
    return preds


@overload_method(BodoKMeansClusteringType, "predict", no_unliteral=True)
def overload_kmeans_clustering_predict(
    m,
    X,
    sample_weight=None,
):
    def _cluster_kmeans_predict(m, X, sample_weight=None):  # pragma: no cover

        with numba.objmode(preds="int64[:]"):
            # TODO: Set _n_threads to 1, even though it shouldn't be necessary
            preds = kmeans_predict_helper(m, X, sample_weight)
        return preds

    return _cluster_kmeans_predict


@overload_method(BodoKMeansClusteringType, "score", no_unliteral=True)
def overload_kmeans_clustering_score(
    m,
    X,
    y=None,
    sample_weight=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """
    We implement the score operation in parallel.
    Each rank has its own copy of the KMeans model and
    calculates the score for its own set of data.
    We then add these scores up.
    """

    def _cluster_kmeans_score(
        m, X, y=None, sample_weight=None, _is_data_distributed=False
    ):  # pragma: no cover
        with numba.objmode(result="float64"):
            # Don't NEED to set _n_threads becasue
            # (a) it isn't used, (b) OMP_NUM_THREADS is set to 1 by bodo init
            # But we're do it anyway in case sklearn changes its behavior later
            orig_nthreads = m._n_threads if hasattr(m, "_n_threads") else None
            orig_njobs = m.n_jobs if hasattr(m, "n_jobs") else None
            m._n_threads = 1
            m.n_jobs = 1

            if len(X) == 0:
                # TODO If X is replicated this should be an error (same as sklearn)
                result = 0
            else:
                result = m.score(X, y=y, sample_weight=sample_weight)
            if _is_data_distributed:
                # If distributed, then add up all the scores
                comm = MPI.COMM_WORLD
                result = comm.allreduce(result, op=MPI.SUM)

            # Restore
            m._n_threads = orig_nthreads
            m.n_jobs = orig_njobs

        return result

    return _cluster_kmeans_score


@overload_method(BodoKMeansClusteringType, "transform", no_unliteral=True)
def overload_kmeans_clustering_transform(m, X):
    """
    We implement the transform operation in parallel.
    Each rank has its own copy of the KMeans model and
    computes the data transformation for its own set of data.
    """

    def _cluster_kmeans_transform(m, X):  # pragma: no cover

        with numba.objmode(X_new="float64[:,:]"):
            # Doesn't parallelize automatically afaik. Set n_jobs and n_threads to 1 anyway.
            orig_nthreads = m._n_threads if hasattr(m, "_n_threads") else None
            orig_njobs = m.n_jobs if hasattr(m, "n_jobs") else None
            m._n_threads = 1
            m.n_jobs = 1

            if len(X) == 0:
                # TODO If X is replicated this should be an error (same as sklearn)
                X_new = np.empty((0, m.n_clusters), dtype=np.int64)
            else:
                X_new = m.transform(X).astype(np.float64)

            # Restore
            m._n_threads = orig_nthreads
            m.n_jobs = orig_njobs

        return X_new

    return _cluster_kmeans_transform


# -------------------------------------MultinomialNB----------------------------------------
# Support sklearn.naive_bayes.MultinomialNB using object mode of Numba
# -----------------------------------------------------------------------------
# Typing and overloads to use MultinomialNB inside Bodo functions
# directly via sklearn's API


class BodoMultinomialNBType(types.Opaque):
    def __init__(self):
        super(BodoMultinomialNBType, self).__init__(name="BodoMultinomialNBType")


multinomial_nb_type = BodoMultinomialNBType()
types.multinomial_nb_type = multinomial_nb_type

register_model(BodoMultinomialNBType)(models.OpaqueModel)


@typeof_impl.register(sklearn.naive_bayes.MultinomialNB)
def typeof_multinomial_nb(val, c):
    return multinomial_nb_type


@box(BodoMultinomialNBType)
def box_multinomial_nb(typ, val, c):
    # See note in box_random_forest_classifier
    c.pyapi.incref(val)
    return val


@unbox(BodoMultinomialNBType)
def unbox_multinomial_nb(typ, obj, c):
    # borrow a reference from Python
    c.pyapi.incref(obj)
    return NativeValue(obj)


@overload(sklearn.naive_bayes.MultinomialNB, no_unliteral=True)
def sklearn_naive_bayes_multinomialnb_overload(
    alpha=1.0,
    fit_prior=True,
    class_prior=None,
):
    def _sklearn_naive_bayes_multinomialnb_impl(
        alpha=1.0,
        fit_prior=True,
        class_prior=None,
    ):  # pragma: no cover
        with numba.objmode(m="multinomial_nb_type"):
            m = sklearn.naive_bayes.MultinomialNB(
                alpha=alpha,
                fit_prior=fit_prior,
                class_prior=class_prior,
            )

        return m

    return _sklearn_naive_bayes_multinomialnb_impl


@overload_method(BodoMultinomialNBType, "fit", no_unliteral=True)
def overload_multinomial_nb_model_fit(
    m,
    X,
    y,
    sample_weight=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):

    """ MultinomialNB fit overload """
    # If data is replicated, run scikit-learn directly
    if is_overload_false(_is_data_distributed):

        def _naive_bayes_multinomial_impl(
            m, X, y, sample_weight=None, _is_data_distributed=False
        ):  # pragma: no cover
            with numba.objmode():
                m.fit(X, y, sample_weight)
            return m

        return _naive_bayes_multinomial_impl
    else:
        # TODO: sample_weight (future enhancement)
        func_text = "def _model_multinomial_nb_fit_impl(\n"
        func_text += "    m, X, y, sample_weight=None, _is_data_distributed=False\n"
        func_text += "):  # pragma: no cover\n"
        # Attempt to change data to numpy array. Any data that fails means, we don't support
        func_text += "    y = bodo.utils.conversion.coerce_to_ndarray(y)\n"
        if isinstance(X, DataFrameType):
            func_text += "    X = X.to_numpy()\n"
        else:
            func_text += "    X = bodo.utils.conversion.coerce_to_ndarray(X)\n"
        func_text += "    my_rank = bodo.get_rank()\n"
        func_text += "    nranks = bodo.get_size()\n"
        func_text += "    total_cols = X.shape[1]\n"
        # Gather specific columns to each rank. Each rank will have n consecutive columns
        func_text += "    for i in range(nranks):\n"
        func_text += "        start = bodo.libs.distributed_api.get_start(total_cols, nranks, i)\n"
        func_text += (
            "        end = bodo.libs.distributed_api.get_end(total_cols, nranks, i)\n"
        )
        # Only write when its your columns
        func_text += "        if i == my_rank:\n"
        func_text += "            X_train = bodo.gatherv(X[:, start:end:1], root=i)\n"
        func_text += "        else:\n"
        func_text += "            bodo.gatherv(X[:, start:end:1], root=i)\n"
        # Replicate y in all ranks
        func_text += "    y_train = bodo.allgatherv(y, False)\n"
        func_text += '    with numba.objmode(m="multinomial_nb_type"):\n'
        func_text += "        m = fit_multinomial_nb(\n"
        func_text += "            m, X_train, y_train, sample_weight, total_cols, _is_data_distributed\n"
        func_text += "        )\n"
        func_text += "    bodo.barrier()\n"
        func_text += "    return m\n"
        loc_vars = {}
        exec(
            func_text,
            {
                "bodo": bodo,
                "np": np,
                "numba": numba,
                "fit_multinomial_nb": fit_multinomial_nb,
            },
            loc_vars,
        )
        # print(func_text)
        _model_multinomial_nb_fit_impl = loc_vars["_model_multinomial_nb_fit_impl"]
        return _model_multinomial_nb_fit_impl


def fit_multinomial_nb(
    m, X_train, y_train, sample_weight=None, total_cols=0, _is_data_distributed=False
):
    """Fit naive bayes Multinomial(parallel version)
    Since this model depends on having lots of columns, we do parallelization by columns"""
    # 1. Compute class log probabilities
    # Taken as it's from sklearn https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/naive_bayes.py#L596
    m._check_X_y(X_train, y_train)
    _, n_features = X_train.shape
    m.n_features_ = n_features
    labelbin = LabelBinarizer()
    Y = labelbin.fit_transform(y_train)
    m.classes_ = labelbin.classes_
    if Y.shape[1] == 1:
        Y = np.concatenate((1 - Y, Y), axis=1)

    # LabelBinarizer().fit_transform() returns arrays with dtype=np.int64.
    # We convert it to np.float64 to support sample_weight consistently;
    # this means we also don't have to cast X to floating point
    # This is also part of it arguments
    if sample_weight is not None:
        Y = Y.astype(np.float64, copy=False)
        sample_weight = _check_sample_weight(sample_weight, X_train)
        sample_weight = np.atleast_2d(sample_weight)
        Y *= sample_weight.T
    class_prior = m.class_prior
    n_effective_classes = Y.shape[1]
    m._init_counters(n_effective_classes, n_features)
    m._count(X_train.astype("float64"), Y)
    alpha = m._check_alpha()
    m._update_class_log_prior(class_prior=class_prior)
    # 2. Computation for feature probabilities
    # Our own implementation for _update_feature_log_prob
    # Probability cannot be computed in parallel as we need total number of all features per class.
    # P(Feature | class) = #feature in class / #all features in class

    # 3. Compute feature probability
    # 3a. Add alpha and compute sum of all features each rank has per class
    smoothed_fc = m.feature_count_ + alpha
    sub_smoothed_cc = smoothed_fc.sum(axis=1)
    # 3b. Allreduce to get sum of all features / class
    comm = MPI.COMM_WORLD
    nranks = comm.Get_size()
    # (classes, )
    smoothed_cc = np.zeros(n_effective_classes)
    comm.Allreduce(sub_smoothed_cc, smoothed_cc, op=MPI.SUM)
    # 3c. Each rank compute log probability for its own set of features.
    # (classes, sub_features)
    sub_feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc.reshape(-1, 1))

    # 4. Allgather the log features so each rank has full model. This is the one used in predict
    # Allgather combines by rows. Therefore, transpose before sending and after receiving
    # Reshape as 1D after transposing (This is needed so numpy actually changes data layout to be transposed)
    sub_log_feature_T = sub_feature_log_prob_.T.reshape(
        n_features * n_effective_classes
    )
    # Get count of elements and displacements for each rank.
    sizes = np.ones(nranks) * (total_cols // nranks)
    remainder_cols = total_cols % nranks
    for rank in range(remainder_cols):
        sizes[rank] += 1
    sizes *= n_effective_classes
    offsets = np.zeros(nranks, dtype=np.int32)
    offsets[1:] = np.cumsum(sizes)[:-1]
    full_log_feature_T = np.zeros((total_cols, n_effective_classes), dtype=np.float64)
    comm.Allgatherv(
        sub_log_feature_T, [full_log_feature_T, sizes, offsets, MPI.DOUBLE_PRECISION]
    )
    # Retranspose to get final shape (n_classes, total_n_features)
    m.feature_log_prob_ = full_log_feature_T.T
    m.n_features_ = m.feature_log_prob_.shape[1]

    # Replicate feature_count. Not now. will see if users need it.
    # feature_count_T = (clf.feature_count_).T
    # feature_count_T = bodo.allgatherv(feature_count_T, False)
    # clf.feature_count_ = feature_count_T.T

    return m


@overload_method(BodoMultinomialNBType, "predict", no_unliteral=True)
def overload_multinomial_nb_model_predict(m, X):
    """Overload Multinomial predict. (Data parallelization)"""
    return parallel_predict(m, X)


@overload_method(BodoMultinomialNBType, "score", no_unliteral=True)
def overload_multinomial_nb_model_score(
    m,
    X,
    y,
    sample_weight=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """Overload Multinomial score."""
    return parallel_score(m, X, y, sample_weight, _is_data_distributed)


# -------------------------------------Logisitic Regression--------------------
# Support sklearn.linear_model.LogisticRegression object mode of Numba
# -----------------------------------------------------------------------------
# Typing and overloads to use LogisticRegression inside Bodo functions
# directly via sklearn's API


class BodoLogisticRegressionType(types.Opaque):
    def __init__(self):
        super(BodoLogisticRegressionType, self).__init__(
            name="BodoLogisticRegressionType"
        )


logistic_regression_type = BodoLogisticRegressionType()
types.logistic_regression_type = logistic_regression_type

register_model(BodoLogisticRegressionType)(models.OpaqueModel)


@typeof_impl.register(sklearn.linear_model.LogisticRegression)
def typeof_logistic_regression(val, c):
    return logistic_regression_type


@box(BodoLogisticRegressionType)
def box_logistic_regression(typ, val, c):
    # See note in box_random_forest_classifier
    c.pyapi.incref(val)
    return val


@unbox(BodoLogisticRegressionType)
def unbox_logistic_regression(typ, obj, c):
    # borrow a reference from Python
    c.pyapi.incref(obj)
    return NativeValue(obj)


@overload(sklearn.linear_model.LogisticRegression, no_unliteral=True)
def sklearn_linear_model_logistic_regression_overload(
    penalty="l2",
    dual=False,
    tol=0.0001,
    C=1.0,
    fit_intercept=True,
    intercept_scaling=1,
    class_weight=None,
    random_state=None,
    solver="lbfgs",
    max_iter=100,
    multi_class="auto",
    verbose=0,
    warm_start=False,
    n_jobs=None,
    l1_ratio=None,
):
    def _sklearn_linear_model_logistic_regression_impl(
        penalty="l2",
        dual=False,
        tol=0.0001,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="lbfgs",
        max_iter=100,
        multi_class="auto",
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None,
    ):  # pragma: no cover
        with numba.objmode(m="logistic_regression_type"):
            m = sklearn.linear_model.LogisticRegression(
                penalty=penalty,
                dual=dual,
                tol=tol,
                C=C,
                fit_intercept=fit_intercept,
                intercept_scaling=intercept_scaling,
                class_weight=class_weight,
                random_state=random_state,
                solver=solver,
                max_iter=max_iter,
                multi_class=multi_class,
                verbose=verbose,
                warm_start=warm_start,
                n_jobs=n_jobs,
                l1_ratio=l1_ratio,
            )
        return m

    return _sklearn_linear_model_logistic_regression_impl


@overload_method(BodoLogisticRegressionType, "fit", no_unliteral=True)
def overload_logistic_regression_fit(
    m,
    X,
    y,
    sample_weight=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """ Logistic Regression fit overload """
    # If data is replicated, run scikit-learn directly
    if is_overload_false(_is_data_distributed):

        def _logistic_regression_fit_impl(
            m, X, y, sample_weight=None, _is_data_distributed=False
        ):  # pragma: no cover
            with numba.objmode():
                m.fit(X, y, sample_weight)
            return m

        return _logistic_regression_fit_impl
    else:
        # Create and run SGDClassifier(loss='log')
        def _sgdc_logistic_regression_fit_impl(
            m, X, y, sample_weight=None, _is_data_distributed=False
        ):  # pragma: no cover
            if bodo.get_rank() == 0:
                print(
                    "WARNING: Data is distributed so Bodo will fit model with SGD solver optimization (SGDClassifier)"
                )
            with numba.objmode(clf="sgd_classifier_type"):
                # SGDClassifier doesn't allow l1_ratio to be None. default=0.15
                if m.l1_ratio is None:
                    l1_ratio = 0.15
                else:
                    l1_ratio = m.l1_ratio
                clf = sklearn.linear_model.SGDClassifier(
                    loss="log",
                    penalty=m.penalty,
                    tol=m.tol,
                    fit_intercept=m.fit_intercept,
                    class_weight=m.class_weight,
                    random_state=m.random_state,
                    max_iter=m.max_iter,
                    verbose=m.verbose,
                    warm_start=m.warm_start,
                    n_jobs=m.n_jobs,
                    l1_ratio=l1_ratio,
                )
            clf.fit(X, y, _is_data_distributed=True)
            with numba.objmode():
                m.coef_ = clf.coef_
                m.intercept_ = clf.intercept_
                m.n_iter_ = clf.n_iter_
                m.classes_ = clf.classes_
            return m

        return _sgdc_logistic_regression_fit_impl


@overload_method(BodoLogisticRegressionType, "predict", no_unliteral=True)
def overload_logistic_regression_predict(m, X):
    """Overload Logistic Regression predict. (Data parallelization)"""
    return parallel_predict(m, X)


@overload_method(BodoLogisticRegressionType, "score", no_unliteral=True)
def overload_logistic_regression_score(
    m,
    X,
    y,
    sample_weight=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """Overload Logistic Regression score."""
    return parallel_score(m, X, y, sample_weight, _is_data_distributed)


@overload_attribute(BodoLogisticRegressionType, "coef_")
def get_logisticR_coef(m):
    """ Overload coef_ attribute to be accessible inside bodo.jit """

    def impl(m):  # pragma: no cover
        with numba.objmode(result="float64[:,:]"):
            result = m.coef_
        return result

    return impl


# -------------------------------------Linear Regression--------------------
# Support sklearn.linear_model.LinearRegression object mode of Numba
# -----------------------------------------------------------------------------
# Typing and overloads to use LinearRegression inside Bodo functions
# directly via sklearn's API


class BodoLinearRegressionType(types.Opaque):
    def __init__(self):
        super(BodoLinearRegressionType, self).__init__(name="BodoLinearRegressionType")


linear_regression_type = BodoLinearRegressionType()
types.linear_regression_type = linear_regression_type

register_model(BodoLinearRegressionType)(models.OpaqueModel)


@typeof_impl.register(sklearn.linear_model.LinearRegression)
def typeof_linear_regression(val, c):
    return linear_regression_type


@box(BodoLinearRegressionType)
def box_linear_regression(typ, val, c):
    # See note in box_random_forest_classifier
    c.pyapi.incref(val)
    return val


@unbox(BodoLinearRegressionType)
def unbox_linear_regression(typ, obj, c):

    # borrow a reference from Python
    c.pyapi.incref(obj)
    return NativeValue(obj)


@overload(sklearn.linear_model.LinearRegression, no_unliteral=True)
def sklearn_linear_model_linear_regression_overload(
    fit_intercept=True,
    normalize=False,
    copy_X=True,
    n_jobs=None,
):
    def _sklearn_linear_model_linear_regression_impl(
        fit_intercept=True,
        normalize=False,
        copy_X=True,
        n_jobs=None,
    ):  # pragma: no cover
        with numba.objmode(m="linear_regression_type"):
            m = sklearn.linear_model.LinearRegression(
                fit_intercept=fit_intercept,
                normalize=normalize,
                copy_X=copy_X,
                n_jobs=n_jobs,
            )
        return m

    return _sklearn_linear_model_linear_regression_impl


@overload_method(BodoLinearRegressionType, "fit", no_unliteral=True)
def overload_linear_regression_fit(
    m,
    X,
    y,
    sample_weight=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """ Linear Regression fit overload """
    # If data is replicated, run scikit-learn directly
    if is_overload_false(_is_data_distributed):

        def _linear_regression_fit_impl(
            m, X, y, sample_weight=None, _is_data_distributed=False
        ):  # pragma: no cover
            with numba.objmode():
                m.fit(X, y, sample_weight)
            return m

        return _linear_regression_fit_impl
    else:
        # Create and run SGDRegressor(loss="squared_loss", penalty=None)
        def _sgdc_linear_regression_fit_impl(
            m, X, y, sample_weight=None, _is_data_distributed=False
        ):  # pragma: no cover
            if bodo.get_rank() == 0:
                print(
                    "WARNING: Data is distributed so Bodo will fit model with SGD solver optimization (SGDRegressor)"
                )
            with numba.objmode(clf="sgd_regressor_type"):
                clf = sklearn.linear_model.SGDRegressor(
                    loss="squared_loss",
                    penalty=None,
                    fit_intercept=m.fit_intercept,
                )
            clf.fit(X, y, _is_data_distributed=True)
            with numba.objmode():
                m.coef_ = clf.coef_
                m.intercept_ = clf.intercept_
            return m

        return _sgdc_linear_regression_fit_impl


@overload_method(BodoLinearRegressionType, "predict", no_unliteral=True)
def overload_linear_regression_predict(m, X):
    """Overload Linear Regression predict. (Data parallelization)"""
    return parallel_predict_regression(m, X)


@overload_method(BodoLinearRegressionType, "score", no_unliteral=True)
def overload_linear_regression_score(
    m,
    X,
    y,
    sample_weight=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """Overload Linear Regression score."""
    return parallel_score(m, X, y, sample_weight, _is_data_distributed)


@overload_attribute(BodoLinearRegressionType, "coef_")
def get_lr_coef(m):
    """ Overload coef_ attribute to be accessible inside bodo.jit """

    def impl(m):  # pragma: no cover
        with numba.objmode(result="float64[:]"):
            result = m.coef_
        return result

    return impl


# -------------------------------------Lasso Regression--------------------
# Support sklearn.linear_model.Lasso object mode of Numba
# -----------------------------------------------------------------------------
# Typing and overloads to use Lasso inside Bodo functions
# directly via sklearn's API


class BodoLassoType(types.Opaque):
    def __init__(self):
        super(BodoLassoType, self).__init__(name="BodoLassoType")


lasso_type = BodoLassoType()
types.lasso_type = lasso_type

register_model(BodoLassoType)(models.OpaqueModel)


@typeof_impl.register(sklearn.linear_model.Lasso)
def typeof_lasso(val, c):
    return lasso_type


@box(BodoLassoType)
def box_lasso(typ, val, c):
    # See note in box_random_forest_classifier
    c.pyapi.incref(val)
    return val


@unbox(BodoLassoType)
def unbox_lasso(typ, obj, c):
    # borrow a reference from Python
    c.pyapi.incref(obj)
    return NativeValue(obj)


@overload(sklearn.linear_model.Lasso, no_unliteral=True)
def sklearn_linear_model_lasso_overload(
    alpha=1.0,
    fit_intercept=True,
    normalize=False,
    precompute=False,
    copy_X=True,
    max_iter=1000,
    tol=0.0001,
    warm_start=False,
    positive=False,
    random_state=None,
    selection="cyclic",
):
    def _sklearn_linear_model_lasso_impl(
        alpha=1.0,
        fit_intercept=True,
        normalize=False,
        precompute=False,
        copy_X=True,
        max_iter=1000,
        tol=0.0001,
        warm_start=False,
        positive=False,
        random_state=None,
        selection="cyclic",
    ):  # pragma: no cover
        with numba.objmode(m="lasso_type"):
            m = sklearn.linear_model.Lasso(
                alpha=alpha,
                fit_intercept=fit_intercept,
                normalize=normalize,
                precompute=precompute,
                copy_X=copy_X,
                max_iter=max_iter,
                tol=tol,
                warm_start=warm_start,
                positive=positive,
                random_state=random_state,
                selection=selection,
            )
        return m

    return _sklearn_linear_model_lasso_impl


@overload_method(BodoLassoType, "fit", no_unliteral=True)
def overload_lasso_fit(
    m,
    X,
    y,
    sample_weight=None,
    check_input=True,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """ Lasso fit overload """
    # If data is replicated, run scikit-learn directly
    if is_overload_false(_is_data_distributed):

        def _lasso_fit_impl(
            m, X, y, sample_weight=None, check_input=True, _is_data_distributed=False
        ):  # pragma: no cover
            with numba.objmode():
                m.fit(X, y, sample_weight, check_input)
            return m

        return _lasso_fit_impl
    else:
        # Create and run SGDRegressor(loss="squared_loss", penalty='l1')
        def _sgdc_lasso_fit_impl(
            m, X, y, sample_weight=None, check_input=True, _is_data_distributed=False
        ):  # pragma: no cover
            if bodo.get_rank() == 0:
                print(
                    "WARNING: Data is distributed so Bodo will fit model with SGD solver optimization (SGDRegressor)"
                )
            with numba.objmode(clf="sgd_regressor_type"):
                clf = sklearn.linear_model.SGDRegressor(
                    loss="squared_loss",
                    penalty="l1",
                    alpha=m.alpha,
                    fit_intercept=m.fit_intercept,
                    max_iter=m.max_iter,
                    tol=m.tol,
                    warm_start=m.warm_start,
                    random_state=m.random_state,
                )
            clf.fit(X, y, _is_data_distributed=True)
            with numba.objmode():
                m.coef_ = clf.coef_
                m.intercept_ = clf.intercept_
                m.n_iter_ = clf.n_iter_
            return m

        return _sgdc_lasso_fit_impl


@overload_method(BodoLassoType, "predict", no_unliteral=True)
def overload_lass_predict(m, X):
    """Overload Lasso Regression predict. (Data parallelization)"""
    return parallel_predict_regression(m, X)


@overload_method(BodoLassoType, "score", no_unliteral=True)
def overload_lasso_score(
    m,
    X,
    y,
    sample_weight=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """Overload Lasso Regression score."""
    return parallel_score(m, X, y, sample_weight, _is_data_distributed)


# -------------------------------------Ridge Regression--------------------
# Support sklearn.linear_model.Ridge object mode of Numba
# -----------------------------------------------------------------------------
# Typing and overloads to use Ridge inside Bodo functions
# directly via sklearn's API


class BodoRidgeType(types.Opaque):
    def __init__(self):
        super(BodoRidgeType, self).__init__(name="BodoRidgeType")


ridge_type = BodoRidgeType()
types.ridge_type = ridge_type

register_model(BodoRidgeType)(models.OpaqueModel)


@typeof_impl.register(sklearn.linear_model.Ridge)
def typeof_ridge(val, c):
    return ridge_type


@box(BodoRidgeType)
def box_ridge(typ, val, c):
    # See note in box_random_forest_classifier
    c.pyapi.incref(val)
    return val


@unbox(BodoRidgeType)
def unbox_ridge(typ, obj, c):
    # borrow a reference from Python
    c.pyapi.incref(obj)
    return NativeValue(obj)


@overload(sklearn.linear_model.Ridge, no_unliteral=True)
def sklearn_linear_model_ridge_overload(
    alpha=1.0,
    fit_intercept=True,
    normalize=False,
    copy_X=True,
    max_iter=None,
    tol=0.001,
    solver="auto",
    random_state=None,
):
    def _sklearn_linear_model_ridge_impl(
        alpha=1.0,
        fit_intercept=True,
        normalize=False,
        copy_X=True,
        max_iter=None,
        tol=0.001,
        solver="auto",
        random_state=None,
    ):  # pragma: no cover
        with numba.objmode(m="ridge_type"):
            m = sklearn.linear_model.Ridge(
                alpha=alpha,
                fit_intercept=fit_intercept,
                normalize=normalize,
                copy_X=copy_X,
                max_iter=max_iter,
                tol=tol,
                solver=solver,
                random_state=random_state,
            )
        return m

    return _sklearn_linear_model_ridge_impl


@overload_method(BodoRidgeType, "fit", no_unliteral=True)
def overload_ridge_fit(
    m,
    X,
    y,
    sample_weight=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """ Ridge Regression fit overload """
    # If data is replicated, run scikit-learn directly
    if is_overload_false(_is_data_distributed):

        def _ridge_fit_impl(
            m, X, y, sample_weight=None, _is_data_distributed=False
        ):  # pragma: no cover
            with numba.objmode():
                m.fit(X, y, sample_weight)
            return m

        return _ridge_fit_impl
    else:
        # Create and run SGDRegressor(loss="squared_loss", penalty='l2')
        def _ridge_fit_impl(
            m, X, y, sample_weight=None, _is_data_distributed=False
        ):  # pragma: no cover
            if bodo.get_rank() == 0:
                print(
                    "WARNING: Data is distributed so Bodo will fit model with SGD solver optimization (SGDRegressor)"
                )
            with numba.objmode(clf="sgd_regressor_type"):
                if m.max_iter is None:
                    max_iter = 1000
                else:
                    max_iter = m.max_iter
                clf = sklearn.linear_model.SGDRegressor(
                    loss="squared_loss",
                    penalty="l2",
                    alpha=0.001,
                    fit_intercept=m.fit_intercept,
                    max_iter=max_iter,
                    tol=m.tol,
                    random_state=m.random_state,
                )
            clf.fit(X, y, _is_data_distributed=True)
            with numba.objmode():
                m.coef_ = clf.coef_
                m.intercept_ = clf.intercept_
                m.n_iter_ = clf.n_iter_
            return m

        return _ridge_fit_impl


@overload_method(BodoRidgeType, "predict", no_unliteral=True)
def overload_linear_regression_predict(m, X):
    """Overload Ridge Regression predict. (Data parallelization)"""
    return parallel_predict_regression(m, X)


@overload_method(BodoRidgeType, "score", no_unliteral=True)
def overload_linear_regression_score(
    m,
    X,
    y,
    sample_weight=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """Overload Ridge Regression score."""
    return parallel_score(m, X, y, sample_weight, _is_data_distributed)


# ------------------------Linear Support Vector Classification-----------------
# Support sklearn.svm.LinearSVC object mode of Numba
# -----------------------------------------------------------------------------
# Typing and overloads to use LinearSVC inside Bodo functions
# directly via sklearn's API


class BodoLinearSVCType(types.Opaque):
    def __init__(self):
        super(BodoLinearSVCType, self).__init__(name="BodoLinearSVCType")


linear_svc_type = BodoLinearSVCType()
types.linear_svc_type = linear_svc_type

register_model(BodoLinearSVCType)(models.OpaqueModel)


@typeof_impl.register(sklearn.svm.LinearSVC)
def typeof_linear_svc(val, c):
    return linear_svc_type


@box(BodoLinearSVCType)
def box_linear_svc(typ, val, c):
    # See note in box_random_forest_classifier
    c.pyapi.incref(val)
    return val


@unbox(BodoLinearSVCType)
def unbox_linear_svc(typ, obj, c):
    # borrow a reference from Python
    c.pyapi.incref(obj)
    return NativeValue(obj)


@overload(sklearn.svm.LinearSVC, no_unliteral=True)
def sklearn_svm_linear_svc_overload(
    penalty="l2",
    loss="squared_hinge",
    dual=True,
    tol=0.0001,
    C=1.0,
    multi_class="ovr",
    fit_intercept=True,
    intercept_scaling=1,
    class_weight=None,
    verbose=0,
    random_state=None,
    max_iter=1000,
):
    def _sklearn_svm_linear_svc_impl(
        penalty="l2",
        loss="squared_hinge",
        dual=True,
        tol=0.0001,
        C=1.0,
        multi_class="ovr",
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        verbose=0,
        random_state=None,
        max_iter=1000,
    ):  # pragma: no cover
        with numba.objmode(m="linear_svc_type"):
            m = sklearn.svm.LinearSVC(
                penalty=penalty,
                loss=loss,
                dual=dual,
                tol=tol,
                C=C,
                multi_class=multi_class,
                fit_intercept=fit_intercept,
                intercept_scaling=intercept_scaling,
                class_weight=class_weight,
                verbose=verbose,
                random_state=random_state,
                max_iter=max_iter,
            )
        return m

    return _sklearn_svm_linear_svc_impl


@overload_method(BodoLinearSVCType, "fit", no_unliteral=True)
def overload_linear_svc_fit(
    m,
    X,
    y,
    sample_weight=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):

    """ Linear SVC fit overload """
    # If data is replicated, run scikit-learn directly
    if is_overload_false(_is_data_distributed):

        def _svm_linear_svc_fit_impl(
            m, X, y, sample_weight=None, _is_data_distributed=False
        ):  # pragma: no cover
            with numba.objmode():
                m.fit(X, y, sample_weight)
            return m

        return _svm_linear_svc_fit_impl
    else:
        # Create and run SGDClassifier(loss='log')
        def _svm_linear_svc_fit_impl(
            m, X, y, sample_weight=None, _is_data_distributed=False
        ):  # pragma: no cover
            if bodo.get_rank() == 0:
                print(
                    "WARNING: Data is distributed so Bodo will fit model with SGD solver optimization (SGDClassifier)"
                )
            with numba.objmode(clf="sgd_classifier_type"):
                clf = sklearn.linear_model.SGDClassifier(
                    loss="hinge",
                    penalty=m.penalty,
                    tol=m.tol,
                    fit_intercept=m.fit_intercept,
                    class_weight=m.class_weight,
                    random_state=m.random_state,
                    max_iter=m.max_iter,
                    verbose=m.verbose,
                )
            clf.fit(X, y, _is_data_distributed=True)
            with numba.objmode():
                m.coef_ = clf.coef_
                m.intercept_ = clf.intercept_
                m.n_iter_ = clf.n_iter_
                m.classes_ = clf.classes_
            return m

        return _svm_linear_svc_fit_impl


@overload_method(BodoLinearSVCType, "predict", no_unliteral=True)
def overload_svm_linear_svc_predict(m, X):
    """Overload LinearSVC predict. (Data parallelization)"""
    return parallel_predict(m, X)


@overload_method(BodoLinearSVCType, "score", no_unliteral=True)
def overload_svm_linear_svc_score(
    m,
    X,
    y,
    sample_weight=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """Overload LinearSVC score."""
    return parallel_score(m, X, y, sample_weight, _is_data_distributed)


# ----------------------------------------------------------------------------------------
# ----------------------------------- Standard-Scaler ------------------------------------
# Support for sklearn.preprocessing.StandardScaler.
# Currently only fit, transform and inverse_transform functions are supported.
# Support for partial_fit will be added in the future since that will require a
# more native implementation. We use sklearn's transform and inverse_transform directly
# in their Bodo implementation. For fit, we use a combination of sklearn's fit function
# and a native implementation. We compute the mean and num_samples_seen on each rank
# using sklearn's fit implementation, then we compute the global values for these using
# MPI operations, and then calculate the variance using a native implementation.
# ----------------------------------------------------------------------------------------


class BodoPreprocessingStandardScalerType(types.Opaque):
    def __init__(self):
        super(BodoPreprocessingStandardScalerType, self).__init__(
            name="BodoPreprocessingStandardScalerType"
        )


preprocessing_standard_scaler_type = BodoPreprocessingStandardScalerType()
types.preprocessing_standard_scaler_type = preprocessing_standard_scaler_type

register_model(BodoPreprocessingStandardScalerType)(models.OpaqueModel)


@typeof_impl.register(sklearn.preprocessing.StandardScaler)
def typeof_preprocessing_standard_scaler(val, c):
    return preprocessing_standard_scaler_type


@box(BodoPreprocessingStandardScalerType)
def box_preprocessing_standard_scaler(typ, val, c):
    # See note in box_random_forest_classifier
    c.pyapi.incref(val)
    return val


@unbox(BodoPreprocessingStandardScalerType)
def unbox_preprocessing_standard_scaler(typ, obj, c):
    # borrow reference from Python
    c.pyapi.incref(obj)
    return NativeValue(obj)


@overload(sklearn.preprocessing.StandardScaler, no_unliteral=True)
def sklearn_preprocessing_standard_scaler_overload(
    copy=True, with_mean=True, with_std=True
):
    """
    Provide implementation for __init__ functions of StandardScaler.
    We simply call sklearn in objmode.
    """

    def _sklearn_preprocessing_standard_scaler_impl(
        copy=True, with_mean=True, with_std=True
    ):  # pragma: no cover

        with numba.objmode(m="preprocessing_standard_scaler_type"):
            m = sklearn.preprocessing.StandardScaler(
                copy=copy,
                with_mean=with_mean,
                with_std=with_std,
            )
        return m

    return _sklearn_preprocessing_standard_scaler_impl


def sklearn_preprocessing_standard_scaler_fit_dist_helper(m, X):
    """
    Distributed calculation of mean and variance for standard scaler.
    We use sklearn to calculate mean and n_samples_seen, combine the
    results appropriately to get the global mean and n_samples_seen.
    We then use these to calculate the variance (and std-dev i.e. scale)
    ourselves (using standard formulae for variance and some helper
    functions from sklearn)
    """

    comm = MPI.COMM_WORLD
    num_pes = comm.Get_size()

    # Get original value of with_std, with_mean
    original_with_std = m.with_std
    original_with_mean = m.with_mean

    # Call with with_std = False to get the mean and n_samples_seen
    m.with_std = False
    if original_with_std:
        m.with_mean = True  # Force set to True, since we'll need it for std calculation
    m = m.fit(X)

    # Restore with_std, with_mean
    m.with_std = original_with_std
    m.with_mean = original_with_mean

    # Handle n_samples_seen:
    # Sklearn returns an int if the same number of samples were found for all dimensions
    # and returns an array if different number of elements were found on different dimensions.
    # For ease of computation in upcoming steps, we convert them to arrays if it is currently an int.
    # We also check if it's an int on all the ranks, if it is, then we will convert it to int at the end
    # on all the ranks to be consistent with sklearn behavior.

    # From: https://github.com/scikit-learn/scikit-learn/blob/0fb307bf39bbdacd6ed713c00724f8f871d60370/sklearn/preprocessing/_data.py#L708
    if not isinstance(m.n_samples_seen_, numbers.Integral):
        n_samples_seen_ints_on_all_ranks = False
    else:
        n_samples_seen_ints_on_all_ranks = True
        # Convert to array if it is currently an integer
        # From: https://github.com/scikit-learn/scikit-learn/blob/0fb307bf39bbdacd6ed713c00724f8f871d60370/sklearn/preprocessing/_data.py#L709
        m.n_samples_seen_ = np.repeat(m.n_samples_seen_, X.shape[1]).astype(
            np.int64, copy=False
        )

    # And then AllGather on n_samples_seen_ to get the sum (and weights for later)
    n_samples_seen_by_rank = np.zeros(
        (num_pes, *m.n_samples_seen_.shape), dtype=m.n_samples_seen_.dtype
    )
    comm.Allgather(m.n_samples_seen_, n_samples_seen_by_rank)
    global_n_samples_seen = np.sum(n_samples_seen_by_rank, axis=0)

    # Set n_samples_seen as the sum
    m.n_samples_seen_ = global_n_samples_seen

    if m.with_mean or m.with_std:
        # AllGather on the mean, and then recompute using np.average and n_samples_seen_rank as weight
        mean_by_rank = np.zeros((num_pes, *m.mean_.shape), dtype=m.mean_.dtype)
        comm.Allgather(m.mean_, mean_by_rank)
        # Replace NaNs with 0 since np.average doesn't have NaN handling
        mean_by_rank[np.isnan(mean_by_rank)] = 0
        global_mean = np.average(mean_by_rank, axis=0, weights=n_samples_seen_by_rank)
        m.mean_ = global_mean

    # If with_std, then calculate (for each dim), np.nansum((X - mean)**2)/total_n_samples_seen on each rank
    if m.with_std:
        # Using _safe_accumulator_op (like in https://github.com/scikit-learn/scikit-learn/blob/0fb307bf39bbdacd6ed713c00724f8f871d60370/sklearn/utils/extmath.py#L776)
        local_variance_calc = (
            sklearn_safe_accumulator_op(np.nansum, (X - global_mean) ** 2, axis=0)
            / global_n_samples_seen
        )
        # Then AllReduce(op.SUM) these values, to get the global variance on each rank.
        global_variance = np.zeros_like(local_variance_calc)
        comm.Allreduce(local_variance_calc, global_variance, op=MPI.SUM)
        m.var_ = global_variance
        # Calculate scale_ from var_
        # From: https://github.com/scikit-learn/scikit-learn/blob/0fb307bf39bbdacd6ed713c00724f8f871d60370/sklearn/preprocessing/_data.py#L772
        m.scale_ = sklearn_handle_zeros_in_scale(np.sqrt(m.var_))

    # Logical AND across ranks on n_samples_seen_ints_on_all_ranks
    n_samples_seen_ints_on_all_ranks = comm.allreduce(
        n_samples_seen_ints_on_all_ranks, op=MPI.LAND
    )
    # If all are ints, then convert to int on all ranks, else let them be arrays
    if n_samples_seen_ints_on_all_ranks:
        m.n_samples_seen_ = m.n_samples_seen_[0]

    return m


@overload_method(BodoPreprocessingStandardScalerType, "fit", no_unliteral=True)
def overload_preprocessing_standard_scaler_fit(
    m,
    X,
    y=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position)
):
    """
    Provide implementations for the fit function.
    In case input is replicated, we simply call sklearn,
    else we use our native implementation.
    """

    def _preprocessing_standard_scaler_fit_impl(
        m, X, y=None, _is_data_distributed=False
    ):  # pragma: no cover

        with numba.objmode(m="preprocessing_standard_scaler_type"):
            if _is_data_distributed:
                # If distributed, then use native implementation
                m = sklearn_preprocessing_standard_scaler_fit_dist_helper(m, X)
            else:
                # If replicated, then just call sklearn
                m = m.fit(X, y)

        return m

    return _preprocessing_standard_scaler_fit_impl


@overload_method(BodoPreprocessingStandardScalerType, "transform", no_unliteral=True)
def overload_preprocessing_standard_scaler_transform(
    m,
    X,
    copy=None,
):
    """
    Provide implementation for the transform function.
    We simply call sklearn's transform on each rank.
    """

    def _preprocessing_standard_scaler_transform_impl(
        m,
        X,
        copy=None,
    ):  # pragma: no cover
        with numba.objmode(transformed_X="float64[:,:]"):
            transformed_X = m.transform(X, copy=copy)
        return transformed_X

    return _preprocessing_standard_scaler_transform_impl


@overload_method(
    BodoPreprocessingStandardScalerType, "inverse_transform", no_unliteral=True
)
def overload_preprocessing_standard_scaler_inverse_transform(
    m,
    X,
    copy=None,
):
    """
    Provide implementation for the inverse_transform function.
    We simply call sklearn's inverse_transform on each rank.
    """

    def _preprocessing_standard_scaler_inverse_transform_impl(
        m,
        X,
        copy=None,
    ):  # pragma: no cover
        with numba.objmode(inverse_transformed_X="float64[:,:]"):
            inverse_transformed_X = m.inverse_transform(X, copy=copy)
        return inverse_transformed_X

    return _preprocessing_standard_scaler_inverse_transform_impl


# -----------------------------------------------------------------------------
# ---------------------------train_test_split--------------------------------------------------
def get_data_slice_parallel(data, labels, len_train):  # pragma: no cover
    """When shuffle=False, just split the data/labels using slicing.
    Run in bodo.jit to do it across ranks"""
    data_train = data[:len_train]
    data_test = data[len_train:]
    data_train = bodo.rebalance(data_train)
    data_test = bodo.rebalance(data_test)
    # TODO: labels maynot be present
    labels_train = labels[:len_train]
    labels_test = labels[len_train:]
    labels_train = bodo.rebalance(labels_train)
    labels_test = bodo.rebalance(labels_test)
    return data_train, data_test, labels_train, labels_test


@numba.njit
def get_train_test_size(train_size, test_size):  # pragma: no cover
    """Set train_size and test_size values"""
    if train_size is None and test_size is None:
        return 0.75, 0.25
    elif train_size is not None and test_size is None:
        return train_size, 1.0 - train_size
    elif train_size is None and test_size is not None:
        return 1.0 - test_size, train_size
    elif train_size + test_size > 1:
        raise ValueError(
            "The sum of test_size and train_size, should be in the (0, 1) range. Reduce test_size and/or train_size."
        )
    else:
        return train_size, test_size


# TODO: labels can be 2D (We don't currently support multivariate in any ML algorithm.)


def set_labels_type(labels, label_type):  # pragma: no cover
    return labels


@overload(set_labels_type, no_unliteral=True)
def overload_set_labels_type(labels, label_type):
    """Change labels type to be same as data variable type if they are different"""
    if get_overload_const_int(label_type) == 1:

        def _set_labels(labels, label_type):  # pragma: no cover
            # Make it a series
            return pd.Series(labels)

        return _set_labels

    elif get_overload_const_int(label_type) == 2:

        def _set_labels(labels, label_type):  # pragma: no cover
            # Get array from labels series
            return labels.values

        return _set_labels
    else:

        def _set_labels(labels, label_type):  # pragma: no cover
            return labels

        return _set_labels


def reset_labels_type(labels, label_type):  # pragma: no cover
    return labels


@overload(reset_labels_type, no_unliteral=True)
def overload_reset_labels_type(labels, label_type):
    """ Reset labels to its original type if changed"""
    if get_overload_const_int(label_type) == 1:

        def _reset_labels(labels, label_type):  # pragma: no cover
            # Change back to array
            return labels.values

        return _reset_labels
    elif get_overload_const_int(label_type) == 2:

        def _reset_labels(labels, label_type):  # pragma: no cover
            # Change back to Series
            return pd.Series(labels, index=np.arange(len(labels)))

        return _reset_labels
    else:

        def _reset_labels(labels, label_type):  # pragma: no cover
            return labels

        return _reset_labels


# Overload to use train_test_split inside Bodo functions
# directly via sklearn's API
@overload(sklearn.model_selection.train_test_split, no_unliteral=True)
def overload_train_test_split(
    data,
    labels=None,
    train_size=None,
    test_size=None,
    random_state=None,
    shuffle=True,
    stratify=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """
    Implement train_test_split. If data is replicated, run sklearn version.
    If data is distributed and shuffle=False, use slicing and then rebalance across ranks
    If data is distributed and shuffle=True, generate a global train/test mask, shuffle, and rebalance across ranks.
    """
    # TODO: Check if labels is None and change output accordingly
    # no_labels = False
    # if is_overload_none(labels):
    #    no_labels = True
    args_dict = {
        "stratify": stratify,
    }

    args_default_dict = {
        "stratify": None,
    }
    check_unsupported_args("train_test_split", args_dict, args_default_dict)
    # If data is replicated, run scikit-learn directly

    if is_overload_false(_is_data_distributed):
        data_type_name = f"data_split_type_{numba.core.ir_utils.next_label()}"
        labels_type_name = f"labels_split_type_{numba.core.ir_utils.next_label()}"
        for d, d_type_name in ((data, data_type_name), (labels, labels_type_name)):
            if isinstance(d, (DataFrameType, SeriesType)):
                d_typ = d.copy(index=NumericIndexType(types.int64))
                setattr(types, d_type_name, d_typ)
            else:
                setattr(types, d_type_name, d)
        func_text = "def _train_test_split_impl(\n"
        func_text += "    data,\n"
        func_text += "    labels=None,\n"
        func_text += "    train_size=None,\n"
        func_text += "    test_size=None,\n"
        func_text += "    random_state=None,\n"
        func_text += "    shuffle=True,\n"
        func_text += "    stratify=None,\n"
        func_text += "    _is_data_distributed=False,\n"
        func_text += "):  # pragma: no cover\n"
        func_text += "    with numba.objmode(data_train='{}', data_test='{}', labels_train='{}', labels_test='{}'):\n".format(
            data_type_name, data_type_name, labels_type_name, labels_type_name
        )
        func_text += "        data_train, data_test, labels_train, labels_test = sklearn.model_selection.train_test_split(\n"
        func_text += "            data,\n"
        func_text += "            labels,\n"
        func_text += "            train_size=train_size,\n"
        func_text += "            test_size=test_size,\n"
        func_text += "            random_state=random_state,\n"
        func_text += "            shuffle=shuffle,\n"
        func_text += "            stratify=stratify,\n"
        func_text += "        )\n"
        func_text += "    return data_train, data_test, labels_train, labels_test\n"
        loc_vars = {}
        exec(func_text, {"numba": numba, "sklearn": sklearn}, loc_vars)
        _train_test_split_impl = loc_vars["_train_test_split_impl"]
        return _train_test_split_impl
    else:
        global get_data_slice_parallel
        if isinstance(get_data_slice_parallel, pytypes.FunctionType):
            get_data_slice_parallel = bodo.jit(
                distributed=[
                    "data",
                    "labels",
                    "data_train",
                    "data_test",
                    "labels_train",
                    "labels_test",
                ]
            )(get_data_slice_parallel)

        # Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.

        label_type = 0
        # 0: no change, 1: change to series, 2: change to array
        if isinstance(data, DataFrameType) and isinstance(labels, types.Array):
            label_type = 1
        elif isinstance(data, types.Array) and isinstance(labels, (SeriesType)):
            label_type = 2
        if is_overload_none(random_state):
            random_state = 42

        def _train_test_split_impl(
            data,
            labels=None,
            train_size=None,
            test_size=None,
            random_state=None,
            shuffle=True,
            stratify=None,
            _is_data_distributed=False,
        ):  # pragma: no cover
            if data.shape[0] != labels.shape[0]:
                raise ValueError(
                    "Found input variables with inconsistent number of samples\n"
                )
            train_size, test_size = get_train_test_size(train_size, test_size)
            # Get total size of data on each rank
            global_data_size = bodo.libs.distributed_api.dist_reduce(
                len(data), np.int32(Reduce_Type.Sum.value)
            )
            len_train = int(global_data_size * train_size)
            len_test = global_data_size - len_train

            if shuffle:
                # Check type. This is needed for shuffle behavior.
                labels = set_labels_type(labels, label_type)

                my_rank = bodo.get_rank()
                nranks = bodo.get_size()
                rank_data_len = np.empty(nranks, np.int64)
                bodo.libs.distributed_api.allgather(rank_data_len, len(data))
                rank_offset = np.cumsum(rank_data_len[0 : my_rank + 1])
                # Create mask where True is for training and False for testing
                global_mask = np.full(global_data_size, True)
                global_mask[:len_test] = False
                np.random.seed(42)
                np.random.permutation(global_mask)
                # Let each rank find its train/test dataset
                if my_rank:
                    start = rank_offset[my_rank - 1]
                else:
                    start = 0
                end = rank_offset[my_rank]
                local_mask = global_mask[start:end]

                data_train = data[local_mask]
                data_test = data[~local_mask]
                labels_train = labels[local_mask]
                labels_test = labels[~local_mask]

                data_train = bodo.random_shuffle(
                    data_train, seed=random_state, parallel=True
                )
                data_test = bodo.random_shuffle(
                    data_test, seed=random_state, parallel=True
                )
                labels_train = bodo.random_shuffle(
                    labels_train, seed=random_state, parallel=True
                )
                labels_test = bodo.random_shuffle(
                    labels_test, seed=random_state, parallel=True
                )

                # Restore type
                labels_train = reset_labels_type(labels_train, label_type)
                labels_test = reset_labels_type(labels_test, label_type)
            else:
                (
                    data_train,
                    data_test,
                    labels_train,
                    labels_test,
                ) = get_data_slice_parallel(data, labels, len_train)

            return data_train, data_test, labels_train, labels_test

        return _train_test_split_impl


# ----------------------------------------------------------------------------------------
# ----------------------------------- MinMax-Scaler ------------------------------------
# Support for sklearn.preprocessing.MinMaxScaler.
# Currently only fit, transform and inverse_transform functions are supported.
# Support for partial_fit will be added in the future since that will require a
# more native implementation (although not hard at all).
# We use sklearn's transform and inverse_transform directly in their Bodo implementation.
# For fit, we use a combination of sklearn's fit function and a native implementation.
# We compute the min/max and num_samples_seen on each rank using sklearn's fit
# implementation, then we compute the global values for these using MPI operations, and
# then re-calculate the rest of the attributes based on these global values.
# ----------------------------------------------------------------------------------------


class BodoPreprocessingMinMaxScalerType(types.Opaque):
    def __init__(self):
        super(BodoPreprocessingMinMaxScalerType, self).__init__(
            name="BodoPreprocessingMinMaxScalerType"
        )


preprocessing_minmax_scaler_type = BodoPreprocessingMinMaxScalerType()
types.preprocessing_minmax_scaler_type = preprocessing_minmax_scaler_type

register_model(BodoPreprocessingMinMaxScalerType)(models.OpaqueModel)


@typeof_impl.register(sklearn.preprocessing.MinMaxScaler)
def typeof_preprocessing_minmax_scaler(val, c):
    return preprocessing_minmax_scaler_type


@box(BodoPreprocessingMinMaxScalerType)
def box_preprocessing_minmax_scaler(typ, val, c):
    # See note in box_random_forest_classifier
    c.pyapi.incref(val)
    return val


@unbox(BodoPreprocessingMinMaxScalerType)
def unbox_preprocessing_minmax_scaler(typ, obj, c):
    # borrow reference from Python
    c.pyapi.incref(obj)
    return NativeValue(obj)


@overload(sklearn.preprocessing.MinMaxScaler, no_unliteral=True)
def sklearn_preprocessing_minmax_scaler_overload(
    feature_range=(0, 1),
    copy=True,
    clip=False,
):
    """
    Provide implementation for __init__ functions of MinMaxScaler.
    We simply call sklearn in objmode.
    """

    def _sklearn_preprocessing_minmax_scaler_impl(
        feature_range=(0, 1),
        copy=True,
        clip=False,
    ):  # pragma: no cover

        with numba.objmode(m="preprocessing_minmax_scaler_type"):
            m = sklearn.preprocessing.MinMaxScaler(
                feature_range=feature_range,
                copy=copy,
                clip=clip,
            )
        return m

    return _sklearn_preprocessing_minmax_scaler_impl


def sklearn_preprocessing_minmax_scaler_fit_dist_helper(m, X):
    """
    Distributed calculation of attributes for MinMaxScaler.
    We use sklearn to calculate min, max and n_samples_seen, combine the
    results appropriately to get the global min/max and n_samples_seen.
    """

    comm = MPI.COMM_WORLD
    num_pes = comm.Get_size()

    # Fit locally
    m = m.fit(X)

    # Compute global n_samples_seen_
    global_n_samples_seen = comm.allreduce(m.n_samples_seen_, op=MPI.SUM)
    m.n_samples_seen_ = global_n_samples_seen

    # Compute global data_min
    local_data_min_by_rank = np.zeros(
        (num_pes, *m.data_min_.shape), dtype=m.data_min_.dtype
    )
    comm.Allgather(m.data_min_, local_data_min_by_rank)
    global_data_min = np.nanmin(local_data_min_by_rank, axis=0)

    # Compute global data_max
    local_data_max_by_rank = np.zeros(
        (num_pes, *m.data_max_.shape), dtype=m.data_max_.dtype
    )
    comm.Allgather(m.data_max_, local_data_max_by_rank)
    global_data_max = np.nanmax(local_data_max_by_rank, axis=0)

    # Compute global data_range
    global_data_range = global_data_max - global_data_min

    # Re-compute the rest of the attributes
    # Similar to: https://github.com/scikit-learn/scikit-learn/blob/42aff4e2edd8e8887478f6ff1628f27de97be6a3/sklearn/preprocessing/_data.py#L409
    m.scale_ = (
        m.feature_range[1] - m.feature_range[0]
    ) / sklearn_handle_zeros_in_scale(global_data_range)
    m.min_ = m.feature_range[0] - global_data_min * m.scale_
    m.data_min_ = global_data_min
    m.data_max_ = global_data_max
    m.data_range_ = global_data_range

    return m


@overload_method(BodoPreprocessingMinMaxScalerType, "fit", no_unliteral=True)
def overload_preprocessing_minmax_scaler_fit(
    m,
    X,
    y=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position)
):
    """
    Provide implementations for the fit function.
    In case input is replicated, we simply call sklearn,
    else we use our native implementation.
    """

    def _preprocessing_minmax_scaler_fit_impl(
        m, X, y=None, _is_data_distributed=False
    ):  # pragma: no cover

        with numba.objmode(m="preprocessing_minmax_scaler_type"):
            if _is_data_distributed:
                # If distributed, then use native implementation
                m = sklearn_preprocessing_minmax_scaler_fit_dist_helper(m, X)
            else:
                # If replicated, then just call sklearn
                m = m.fit(X, y)

        return m

    return _preprocessing_minmax_scaler_fit_impl


@overload_method(BodoPreprocessingMinMaxScalerType, "transform", no_unliteral=True)
def overload_preprocessing_minmax_scaler_transform(
    m,
    X,
):
    """
    Provide implementation for the transform function.
    We simply call sklearn's transform on each rank.
    """

    def _preprocessing_minmax_scaler_transform_impl(
        m,
        X,
    ):  # pragma: no cover
        with numba.objmode(transformed_X="float64[:,:]"):
            transformed_X = m.transform(X)
        return transformed_X

    return _preprocessing_minmax_scaler_transform_impl


@overload_method(
    BodoPreprocessingMinMaxScalerType, "inverse_transform", no_unliteral=True
)
def overload_preprocessing_minmax_scaler_inverse_transform(
    m,
    X,
):
    """
    Provide implementation for the inverse_transform function.
    We simply call sklearn's inverse_transform on each rank.
    """

    def _preprocessing_minmax_scaler_inverse_transform_impl(
        m,
        X,
    ):  # pragma: no cover
        with numba.objmode(inverse_transformed_X="float64[:,:]"):
            inverse_transformed_X = m.inverse_transform(X)
        return inverse_transformed_X

    return _preprocessing_minmax_scaler_inverse_transform_impl


# ----------------------------------------------------------------------------------------
# ----------------------------------- LabelEncoder------------------------------------
# Support for sklearn.preprocessing.LabelEncoder.
# Currently only fit, fit_transform, transform and inverse_transform functions are supported.
# We use sklearn's transform and inverse_transform directly in their Bodo implementation.
# For fit, we use np.unique and then replicate its output to be classes_ attribute
# ----------------------------------------------------------------------------------------


class BodoPreprocessingLabelEncoderType(types.Opaque):
    def __init__(self):
        super(BodoPreprocessingLabelEncoderType, self).__init__(
            name="BodoPreprocessingLabelEncoderType"
        )


preprocessing_label_encoder_type = BodoPreprocessingLabelEncoderType()
types.preprocessing_label_encoder_type = preprocessing_label_encoder_type

register_model(BodoPreprocessingLabelEncoderType)(models.OpaqueModel)


@typeof_impl.register(sklearn.preprocessing.LabelEncoder)
def typeof_preprocessing_label_encoder(val, c):
    return preprocessing_label_encoder_type


@box(BodoPreprocessingLabelEncoderType)
def box_preprocessing_label_encoder(typ, val, c):
    # See note in box_random_forest_classifier
    c.pyapi.incref(val)
    return val


@unbox(BodoPreprocessingLabelEncoderType)
def unbox_preprocessing_label_encoder(typ, obj, c):
    # borrow reference from Python
    c.pyapi.incref(obj)
    return NativeValue(obj)


@overload(sklearn.preprocessing.LabelEncoder, no_unliteral=True)
def sklearn_preprocessing_label_encoder_overload():
    """
    Provide implementation for __init__ functions of LabelEncoder.
    We simply call sklearn in objmode.
    """

    def _sklearn_preprocessing_label_encoder_impl():  # pragma: no cover

        with numba.objmode(m="preprocessing_label_encoder_type"):
            m = sklearn.preprocessing.LabelEncoder()
        return m

    return _sklearn_preprocessing_label_encoder_impl


@overload_method(BodoPreprocessingLabelEncoderType, "fit", no_unliteral=True)
def overload_preprocessing_label_encoder_fit(
    m,
    y,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position)
):
    """
    Provide implementations for the fit function.
    In case input is replicated, we simply call sklearn,
    else we use our unique to get labels and assign them to classes_ attribute
    """
    if is_overload_true(_is_data_distributed):

        def _sklearn_preprocessing_label_encoder_fit_impl(
            m, y, _is_data_distributed=False
        ):  # pragma: no cover
            y_classes = bodo.libs.array_kernels.unique(y, parallel=True)
            y_classes = bodo.allgatherv(y_classes, False)
            y_classes = bodo.libs.array_kernels.sort(
                y_classes, ascending=True, inplace=False
            )
            with numba.objmode:
                m.classes_ = y_classes

            return m

        return _sklearn_preprocessing_label_encoder_fit_impl

    else:
        # If replicated, then just call sklearn
        def _sklearn_preprocessing_label_encoder_fit_impl(
            m, y, _is_data_distributed=False
        ):  # pragma: no cover
            with numba.objmode(m="preprocessing_label_encoder_type"):
                m = m.fit(y)

            return m

        return _sklearn_preprocessing_label_encoder_fit_impl


@overload_method(BodoPreprocessingLabelEncoderType, "transform", no_unliteral=True)
def overload_preprocessing_label_encoder_transform(
    m,
    y,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position)
):
    """
    Provide implementation for the transform function.
    We simply call sklearn's transform on each rank.
    """

    def _preprocessing_label_encoder_transform_impl(
        m, y, _is_data_distributed=False
    ):  # pragma: no cover
        with numba.objmode(transformed_y="int64[:]"):
            transformed_y = m.transform(y)
        return transformed_y

    return _preprocessing_label_encoder_transform_impl


@numba.njit
def le_fit_transform(m, y):  # pragma: no cover
    m = m.fit(y, _is_data_distributed=True)
    transformed_y = m.transform(y, _is_data_distributed=True)
    return transformed_y


@overload_method(BodoPreprocessingLabelEncoderType, "fit_transform", no_unliteral=True)
def overload_preprocessing_label_encoder_fit_transform(
    m,
    y,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position)
):
    """
    Provide implementation for the fit_transform function.
    If distributed repeat fit and then transform operation.
    If replicated simply call sklearn directly in objmode
    """
    if is_overload_true(_is_data_distributed):

        def _preprocessing_label_encoder_fit_transform_impl(
            m, y, _is_data_distributed=False
        ):  # pragma: no cover
            transformed_y = le_fit_transform(m, y)
            return transformed_y

        return _preprocessing_label_encoder_fit_transform_impl
    else:
        # If replicated, then just call sklearn
        def _preprocessing_label_encoder_fit_transform_impl(
            m, y, _is_data_distributed=False
        ):  # pragma: no cover
            with numba.objmode(transformed_y="int64[:]"):
                transformed_y = m.fit_transform(y)
            return transformed_y

        return _preprocessing_label_encoder_fit_transform_impl
