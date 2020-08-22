"""Support sklearn.ensemble.RandomForestClassifier using object mode of Numba
"""
import numpy as np
import pandas as pd
import numba
from numba.core import types
from numba.extending import (
    box,
    unbox,
    register_model,
    models,
    NativeValue,
    overload,
    overload_method,
    typeof_impl,
)

from bodo.utils.typing import (
    is_overload_constant_str,
    get_overload_const_str,
    is_overload_true,
    is_overload_false,
    is_overload_none,
)

import bodo
from bodo.libs.distributed_api import dist_reduce, Reduce_Type, get_node_portion
from mpi4py import MPI
import itertools
import sklearn.ensemble
import sklearn.metrics


def model_fit(m, X, y):
    # TODO multi-node:
    # Current implementation is not correct for number of nodes > 1 and
    # should abort in that case.
    # TODO check that random_state behavior matches sklearn when
    # the training is distributed (does not apply currently)

    if bodo.get_rank() == 0:
        # train model on rank 0
        # TODO for multinode: n_jobs should be number of ranks on this node,
        # NOT on MPI_COMM_WORLD
        m.n_jobs = bodo.get_size()
        from sklearn.utils import parallel_backend

        with parallel_backend("threading"):
            m.fit(X, y)
        m.n_jobs = 1

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
    if bodo.get_rank() == 0:
        # Do piece-wise broadcast to avoid huge messages that can result
        # from pickling the estimators
        # TODO investigate why the pickled estimators are so large. It
        # doesn't look like the unpickled estimators have a large memory
        # footprint
        for i in range(0, m.n_estimators, 10):
            comm.bcast(m.estimators_[i : i + 10])
        comm.bcast(m.n_classes_)
        comm.bcast(m.n_outputs_)
        comm.bcast(m.classes_)
    else:
        estimators = []
        for i in range(0, m.n_estimators, 10):
            estimators += comm.bcast(None)
        m.n_classes_ = comm.bcast(None)
        m.n_outputs_ = comm.bcast(None)
        m.classes_ = comm.bcast(None)
        m.estimators_ = estimators
    assert len(m.estimators_) == m.n_estimators


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

        if _is_data_distributed:
            # TODO for multinode: replicate on first rank of each node
            X = bodo.gatherv(X)
            y = bodo.gatherv(y)

        with numba.objmode:
            model_fit(m, X, y)  # return value is m

        bodo.barrier()
        return m

    return _model_fit_impl


@overload_method(BodoRandomForestClassifierType, "predict", no_unliteral=True)
def overload_model_predict(m, X):
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


@overload_method(BodoRandomForestClassifierType, "score", no_unliteral=True)
def overload_model_score(
    m,
    X,
    y,
    sample_weight=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
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
def precision_recall_fscore_parallel(y_true, y_pred, operation, average="binary"):
    labels = bodo.libs.array_kernels.unique_parallel(y_true)
    labels = bodo.allgatherv(labels, False)
    labels = pd.Series(labels).sort_values().values

    nlabels = len(labels)
    # true positive for each label
    tp_sum = np.zeros(nlabels, np.int64)
    # count of label appearance in y_true
    true_sum = np.zeros(nlabels, np.int64)
    # count of label appearance in y_pred
    pred_sum = np.zeros(nlabels, np.int64)
    label_dict = bodo.hiframes.pd_categorical_ext.get_label_dict_from_categories(labels)
    for i in range(len(y_true)):
        label = label_dict[y_pred[i]]
        pred_sum[label] += 1
        true_sum[label_dict[y_true[i]]] += 1
        if y_true[i] == y_pred[i]:
            tp_sum[label] += 1

    # gather global tp_sum, true_sum and pred_sum on every process
    tp_sum = bodo.libs.distributed_api.dist_reduce(tp_sum, np.int32(Reduce_Type.Sum.value))
    true_sum = bodo.libs.distributed_api.dist_reduce(true_sum, np.int32(Reduce_Type.Sum.value))
    pred_sum = bodo.libs.distributed_api.dist_reduce(pred_sum, np.int32(Reduce_Type.Sum.value))

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
