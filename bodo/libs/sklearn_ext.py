"""Support sklearn.ensemble.RandomForestClassifier using object mode of Numba
"""
import itertools

import numba
import numpy as np
import pandas as pd
import sklearn.ensemble
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
from sklearn.metrics import hinge_loss, log_loss

import bodo
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


def fit_sgdc(m, X, y, y_classes, _is_data_distributed=False):
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
    for _ in range(m.max_iter):
        m.partial_fit(X, y, classes=y_classes)
        # Can be removed when rebalancing is done. Now, we have to give more weight to ranks with more data
        m.coef_ = m.coef_ * rank_weight
        m.coef_ = comm.allreduce(m.coef_, op=MPI.SUM)
        m.intercept_ = m.intercept_ * rank_weight
        m.intercept_ = comm.allreduce(m.intercept_, op=MPI.SUM)
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
            m = fit_sgdc(m, X, y, y_classes, _is_data_distributed)

        bodo.barrier()

        return m

    return _model_sgdc_fit_impl


@overload_method(BodoSGDClassifierType, "predict", no_unliteral=True)
def overload_sgdc_model_predict(m, X):
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


@overload_method(BodoSGDClassifierType, "score", no_unliteral=True)
def overload_sgdc_model_score(
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
