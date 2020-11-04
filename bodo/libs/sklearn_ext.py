"""Support scikit-learn using object mode of Numba """
import itertools

import numba
import numpy as np
import pandas as pd
import sklearn.cluster
import sklearn.ensemble
import sklearn.linear_model
import sklearn.metrics
import sklearn.utils
from mpi4py import MPI
from numba.core import types
from numba.extending import (
    NativeValue,
    box,
    models,
    overload,
    overload_method,
    register_model,
    typeof_impl,
    unbox,
)
from sklearn.metrics import hinge_loss, log_loss, mean_squared_error
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import _check_sample_weight

import bodo
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.libs.distributed_api import (
    Reduce_Type,
    create_subcomm_mpi4py,
    dist_reduce,
    get_host_ranks,
    get_node_portion,
    get_nodes_first_ranks,
    get_num_nodes,
)
from bodo.utils.typing import (
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
        if m.tol > np.NINF and cur_loss > best_loss - m.tol * len(X):
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
    def _model_sgdc_fit_impl(m, X, y, _is_data_distributed=False):  # pragma: no cover

        # TODO: Rebalance the data X and y to be the same size on every rank
        # y has to be an array
        y_classes = bodo.libs.array_kernels.unique(y)

        if _is_data_distributed:
            y_classes = bodo.allgatherv(y_classes, False)

        with numba.objmode(m="sgd_classifier_type"):
            m = fit_sgd(m, X, y, y_classes, _is_data_distributed)

        bodo.barrier()

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
