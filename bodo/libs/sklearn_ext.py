"""Support scikit-learn using object mode of Numba """
import itertools
import numbers
import sys
import types as pytypes
import warnings
from itertools import combinations

import numba
import numpy as np
import pandas as pd
import sklearn.cluster
import sklearn.ensemble
import sklearn.feature_extraction
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.naive_bayes
import sklearn.svm
import sklearn.utils
from numba.core import types
from numba.extending import (
    overload,
    overload_attribute,
    overload_method,
    register_jitable,
)
from scipy import stats  # noqa
from scipy.special import comb
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import hinge_loss, log_loss, mean_squared_error
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing._data import (
    _handle_zeros_in_scale as sklearn_handle_zeros_in_scale,
)
from sklearn.utils._encode import _unique
from sklearn.utils.extmath import (
    _safe_accumulator_op as sklearn_safe_accumulator_op,
)
from sklearn.utils.validation import _check_sample_weight, column_or_1d

import bodo
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.hiframes.pd_index_ext import NumericIndexType
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.libs.csr_matrix_ext import CSRMatrixType
from bodo.libs.distributed_api import (
    Reduce_Type,
    create_subcomm_mpi4py,
    get_host_ranks,
    get_nodes_first_ranks,
    get_num_nodes,
)
from bodo.mpi4py import MPI
from bodo.utils.py_objs import install_py_obj_class
from bodo.utils.typing import (
    BodoError,
    BodoWarning,
    check_unsupported_args,
    get_overload_const,
    get_overload_const_int,
    get_overload_const_str,
    is_overload_constant_number,
    is_overload_constant_str,
    is_overload_false,
    is_overload_none,
    is_overload_true,
)

this_module = sys.modules[__name__]

_is_sklearn_supported_version = False
_min_sklearn_version = (1, 4, 0)
_min_sklearn_ver_str = ".".join(str(x) for x in _min_sklearn_version)
_max_sklearn_version_exclusive = (1, 5, 0)
_max_sklearn_ver_str = ".".join(str(x) for x in _max_sklearn_version_exclusive)
try:
    # Create capturing group for major, minor, and bugfix version numbers.
    # last number could have string e.g. `dev0`
    import re

    import sklearn  # noqa

    regex = re.compile(r"(\d+)\.(\d+)\..*(\d+)")
    sklearn_version = sklearn.__version__
    m = regex.match(sklearn_version)
    if m:
        ver = tuple(map(int, m.groups()))
        if ver >= _min_sklearn_version and ver < _max_sklearn_version_exclusive:
            _is_sklearn_supported_version = True

except ImportError:
    # TODO if sklearn is not installed, trying to use sklearn inside
    # bodo functions should give more meaningful errors than:
    # Untyped global name 'RandomForestClassifier': cannot determine Numba type of <class 'abc.ABCMeta'>
    pass


def check_sklearn_version():
    if not _is_sklearn_supported_version:
        msg = f" Bodo supports scikit-learn version >= {_min_sklearn_ver_str} and < {_max_sklearn_ver_str}.\n \
            Installed version is {sklearn.__version__}.\n"
        raise BodoError(msg)


def random_forest_model_fit(m, X, y):
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
    with bodo.objmode(first_rank_node="int32[:]"):
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
        if isinstance(m, sklearn.ensemble.RandomForestClassifier):
            comm.bcast(m.n_classes_)
            comm.bcast(m.classes_)
        comm.bcast(m.n_outputs_)
    # Add no cover becuase coverage report is done by one rank only.
    else:  # pragma: no cover
        estimators = []
        for i in range(0, n_estimators_global, 10):
            estimators += comm.bcast(None)
        if isinstance(m, sklearn.ensemble.RandomForestClassifier):
            m.n_classes_ = comm.bcast(None)
            m.classes_ = comm.bcast(None)
        m.n_outputs_ = comm.bcast(None)
        m.estimators_ = estimators
    assert len(m.estimators_) == n_estimators_global
    m.n_estimators = n_estimators_global
    m.n_features_in_ = X.shape[1]


# -----------------------------------------------------------------------------
# Typing and overloads to use RandomForestClassifier inside Bodo functions
# directly via sklearn's API

# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoRandomForestClassifierType = install_py_obj_class(
    types_name="random_forest_classifier_type",
    python_type=sklearn.ensemble.RandomForestClassifier,
    module=this_module,
    class_name="BodoRandomForestClassifierType",
    model_name="BodoRandomForestClassifierModel",
)


@overload(sklearn.ensemble.RandomForestClassifier, no_unliteral=True)
def sklearn_ensemble_RandomForestClassifier_overload(
    n_estimators=100,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features="sqrt",
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=None,
    verbose=0,
    warm_start=False,
    class_weight=None,
    ccp_alpha=0.0,
    max_samples=None,
    monotonic_cst=None,
):
    check_sklearn_version()

    # TODO n_jobs should be left unspecified so should probably throw an error if used

    def _sklearn_ensemble_RandomForestClassifier_impl(
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
    ):  # pragma: no cover
        with bodo.objmode(m="random_forest_classifier_type"):
            if random_state is not None and get_num_nodes() > 1:
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
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=1,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                class_weight=class_weight,
                ccp_alpha=ccp_alpha,
                max_samples=max_samples,
                monotonic_cst=monotonic_cst,
            )
        return m

    return _sklearn_ensemble_RandomForestClassifier_impl


def parallel_predict_regression(m, X):
    """
    Implement the regression prediction operation in parallel.
    Each rank has its own copy of the model and predicts for its
    own set of data.
    """
    check_sklearn_version()

    def _model_predict_impl(m, X):  # pragma: no cover
        with bodo.objmode(result="float64[:]"):
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
    check_sklearn_version()

    def _model_predict_impl(m, X):  # pragma: no cover
        with bodo.objmode(result="int64[:]"):
            # currently we do data-parallel prediction
            m.n_jobs = 1
            # len cannot be used with csr
            if X.shape[0] == 0:
                # TODO If X is replicated this should be an error (same as sklearn)
                result = np.empty(0, dtype=np.int64)
            else:
                result = m.predict(X).astype(np.int64).flatten()
        return result

    return _model_predict_impl


def parallel_predict_proba(m, X):
    """
    Implement the predict_proba operation in parallel.
    Each rank has its own copy of the model and computes results for its
    own set of data.
    This strategy is the same for a lot of classifier estimators.
    """
    check_sklearn_version()

    def _model_predict_proba_impl(m, X):  # pragma: no cover
        with bodo.objmode(result="float64[:,:]"):
            # currently we do data-parallel prediction
            m.n_jobs = 1
            # len cannot be used with csr
            if X.shape[0] == 0:
                # TODO If X is replicated this should be an error (same as sklearn)
                result = np.empty((0, 0), dtype=np.float64)
            else:
                result = m.predict_proba(X).astype(np.float64)
        return result

    return _model_predict_proba_impl


def parallel_predict_log_proba(m, X):
    """
    Implement the predict_log_proba operation in parallel.
    Each rank has its own copy of the model and computes results for its
    own set of data.
    This strategy is the same for a lot of classifier estimators.
    """
    check_sklearn_version()

    def _model_predict_log_proba_impl(m, X):  # pragma: no cover
        with bodo.objmode(result="float64[:,:]"):
            # currently we do data-parallel prediction
            m.n_jobs = 1
            # len cannot be used with csr
            if X.shape[0] == 0:
                # TODO If X is replicated this should be an error (same as sklearn)
                result = np.empty((0, 0), dtype=np.float64)
            else:
                result = m.predict_log_proba(X).astype(np.float64)
        return result

    return _model_predict_log_proba_impl


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
    check_sklearn_version()

    def _model_score_impl(
        m, X, y, sample_weight=None, _is_data_distributed=False
    ):  # pragma: no cover
        with bodo.objmode(result="float64[:]"):
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
    check_sklearn_version()
    """Overload Random Forest Classifier predict. (Data parallelization)"""
    return parallel_predict(m, X)


@overload_method(BodoRandomForestClassifierType, "predict_proba", no_unliteral=True)
def overload_rf_predict_proba(m, X):
    check_sklearn_version()
    """Overload Random Forest Classifier predict_proba. (Data parallelization)"""
    return parallel_predict_proba(m, X)


@overload_method(BodoRandomForestClassifierType, "predict_log_proba", no_unliteral=True)
def overload_rf_predict_log_proba(m, X):
    check_sklearn_version()
    """Overload Random Forest Classifier predict_log_proba. (Data parallelization)"""
    return parallel_predict_log_proba(m, X)


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

    with bodo.objmode(result="float64[:]"):
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
    check_sklearn_version()
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
                with bodo.objmode(score="float64[:]"):
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
                with bodo.objmode(score="float64"):
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
    check_sklearn_version()
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
                with bodo.objmode(score="float64[:]"):
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
                with bodo.objmode(score="float64"):
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
    check_sklearn_version()
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
                with bodo.objmode(score="float64[:]"):
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
                with bodo.objmode(score="float64"):
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

    check_sklearn_version()
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
                with bodo.objmode(err="float64[:]"):
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
                with bodo.objmode(err="float64[:]"):
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
                with bodo.objmode(err="float64"):
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
                with bodo.objmode(err="float64"):
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

    check_sklearn_version()
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
                with bodo.objmode(err="float64[:]"):
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
                with bodo.objmode(err="float64[:]"):
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
                with bodo.objmode(err="float64"):
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
                with bodo.objmode(err="float64"):
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


# ----------------------------------log_loss-------------------------------------


def log_loss_dist_helper(y_true, y_pred, eps, normalize, sample_weight, labels):
    """
    Helper for distributed log_loss computation.
    Call sklearn on each rank with normalize=False to get
    counts (i.e. sum(accuracy_bits))
    (or sample_weight.T @ accuracy_bits when sample_weight != None
    """
    loss = sklearn.metrics.log_loss(
        y_true,
        y_pred,
        eps=eps,
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
    eps=1e-15,
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
    check_sklearn_version()

    func_text = "def _log_loss_impl(\n"
    func_text += "    y_true,\n"
    func_text += "    y_pred,\n"
    func_text += "    eps=1e-15,\n"
    func_text += "    normalize=True,\n"
    func_text += "    sample_weight=None,\n"
    func_text += "    labels=None,\n"
    func_text += "    _is_data_distributed=False,\n"
    func_text += "):\n"
    # User could pass lists and numba throws error if passing lists
    # to object mode, so we convert to arrays
    func_text += "    y_true = bodo.utils.conversion.coerce_to_array(y_true)\n"
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

    func_text += "    with bodo.objmode(loss='float64'):\n"
    if is_overload_false(_is_data_distributed):
        # For replicated data, directly call sklearn
        func_text += "        loss = sklearn.metrics.log_loss(\n"
    else:
        # For distributed data, pre-compute labels globally, then call our implementation
        func_text += "        loss = log_loss_dist_helper(\n"
    func_text += "            y_true, y_pred, eps=eps, normalize=normalize,\n"
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

    check_sklearn_version()

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
        func_text += "    with bodo.objmode(out='float64[:,::1]'):\n"
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

    check_sklearn_version()
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

                with bodo.objmode(score="float64"):
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
                with bodo.objmode(score="float64"):
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
                with bodo.objmode(score="float64"):
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
                with bodo.objmode(score="float64"):
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

    check_sklearn_version()
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
                with bodo.objmode(score="float64[:]"):
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
                with bodo.objmode(score="float64[:]"):
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
                with bodo.objmode(score="float64"):
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
                with bodo.objmode(score="float64"):
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
        raise ValueError("normalize must be one of {'true', 'pred', " "'all', None}")

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

    check_sklearn_version()

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

    func_text += f"    with bodo.objmode(cm='{cm_dtype[0]}'):\n"
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


# -------------------------------------SGDRegressor----------------------------------------
# Support sklearn.linear_model.SGDRegressorusing object mode of Numba
# Linear regression: sklearn.linear_model.SGDRegressor(loss="squared_error", penalty=None)
# Ridge regression: sklearn.linear_model.SGDRegressor(loss="squared_error", penalty='l2')
# Lasso: sklearn.linear_model.SGDRegressor(loss="squared_error", penalty='l1')

# -----------------------------------------------------------------------------
# Typing and overloads to use SGDRegressor inside Bodo functions
# directly via sklearn's API

# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoSGDRegressorType = install_py_obj_class(
    types_name="sgd_regressor_type",
    python_type=sklearn.linear_model.SGDRegressor,
    module=this_module,
    class_name="BodoSGDRegressorType",
    model_name="BodoSGDRegressorModel",
)


@overload(sklearn.linear_model.SGDRegressor, no_unliteral=True)
def sklearn_linear_model_SGDRegressor_overload(
    loss="squared_error",
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
    check_sklearn_version()

    def _sklearn_linear_model_SGDRegressor_impl(
        loss="squared_error",
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
        with bodo.objmode(m="sgd_regressor_type"):
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
    coef_init=None,
    intercept_init=None,
    sample_weight=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    check_sklearn_version()

    if is_overload_true(_is_data_distributed):
        if not is_overload_none(sample_weight):
            raise BodoError(
                "sklearn.linear_model.SGDRegressor.fit() : 'sample_weight' is not supported for distributed data."
            )

        if not is_overload_none(coef_init):
            raise BodoError(
                "sklearn.linear_model.SGDRegressor.fit() : 'coef_init' is not supported for distributed data."
            )
        if not is_overload_none(intercept_init):
            raise BodoError(
                "sklearn.linear_model.SGDRegressor.fit() : 'intercept_init' is not supported for distributed data."
            )

        def _model_sgdr_fit_impl(
            m,
            X,
            y,
            coef_init=None,
            intercept_init=None,
            sample_weight=None,
            _is_data_distributed=False,
        ):  # pragma: no cover
            # TODO: Rebalance the data X and y to be the same size on every rank
            with bodo.objmode(m="sgd_regressor_type"):
                m = fit_sgd(m, X, y, _is_data_distributed)

            bodo.barrier()

            return m

        return _model_sgdr_fit_impl
    else:
        # If replicated, then just call sklearn
        def _model_sgdr_fit_impl(
            m,
            X,
            y,
            coef_init=None,
            intercept_init=None,
            sample_weight=None,
            _is_data_distributed=False,
        ):  # pragma: no cover
            with bodo.objmode(m="sgd_regressor_type"):
                m = m.fit(X, y, coef_init, intercept_init, sample_weight)
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
# Logistic regression (loss='log_loss')
# -----------------------------------------------------------------------------

# Typing and overloads to use SGDClassifier inside Bodo functions
# directly via sklearn's API

# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoSGDClassifierType = install_py_obj_class(
    types_name="sgd_classifier_type",
    python_type=sklearn.linear_model.SGDClassifier,
    module=this_module,
    class_name="BodoSGDClassifierType",
    model_name="BodoSGDClassifierModel",
)


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
    check_sklearn_version()

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
        with bodo.objmode(m="sgd_classifier_type"):
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
    elif m.loss == "log_loss":
        loss_func = log_loss
    elif m.loss == "squared_error":
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
    coef_init=None,
    intercept_init=None,
    sample_weight=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    check_sklearn_version()
    """
    Provide implementations for the fit function.
    In case input is replicated, we simply call sklearn,
    else we use partial_fit on each rank then use we re-compute the attributes using MPI operations.
    """
    if is_overload_true(_is_data_distributed):
        if not is_overload_none(sample_weight):
            raise BodoError(
                "sklearn.linear_model.SGDClassifier.fit() : 'sample_weight' is not supported for distributed data."
            )

        if not is_overload_none(coef_init):
            raise BodoError(
                "sklearn.linear_model.SGDClassifier.fit() : 'coef_init' is not supported for distributed data."
            )
        if not is_overload_none(intercept_init):
            raise BodoError(
                "sklearn.linear_model.SGDClassifier.fit() : 'intercept_init' is not supported for distributed data."
            )

        def _model_sgdc_fit_impl(
            m,
            X,
            y,
            coef_init=None,
            intercept_init=None,
            sample_weight=None,
            _is_data_distributed=False,
        ):  # pragma: no cover
            # TODO: Rebalance the data X and y to be the same size on every rank
            # y has to be an array
            y_classes = bodo.libs.array_kernels.unique(y, parallel=True)
            y_classes = bodo.allgatherv(y_classes, False)

            with bodo.objmode(m="sgd_classifier_type"):
                m = fit_sgd(m, X, y, y_classes, _is_data_distributed)

            return m

        return _model_sgdc_fit_impl
    else:
        # If replicated, then just call sklearn
        def _model_sgdc_fit_impl(
            m,
            X,
            y,
            coef_init=None,
            intercept_init=None,
            sample_weight=None,
            _is_data_distributed=False,
        ):  # pragma: no cover
            with bodo.objmode(m="sgd_classifier_type"):
                m = m.fit(X, y, coef_init, intercept_init, sample_weight)
            return m

        return _model_sgdc_fit_impl


@overload_method(BodoSGDClassifierType, "predict", no_unliteral=True)
def overload_sgdc_model_predict(m, X):
    """Overload SGDClassifier predict. (Data parallelization)"""
    return parallel_predict(m, X)


@overload_method(BodoSGDClassifierType, "predict_proba", no_unliteral=True)
def overload_sgdc_model_predict_proba(m, X):
    """Overload SGDClassifier predict_proba. (Data parallelization)"""
    return parallel_predict_proba(m, X)


@overload_method(BodoSGDClassifierType, "predict_log_proba", no_unliteral=True)
def overload_sgdc_model_predict_log_proba(m, X):
    """Overload SGDClassifier predict_log_proba. (Data parallelization)"""
    return parallel_predict_log_proba(m, X)


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
    """Overload coef_ attribute to be accessible inside bodo.jit"""

    def impl(m):  # pragma: no cover
        with bodo.objmode(result="float64[:,:]"):
            result = m.coef_
        return result

    return impl


# --------------------------------------------------------------------------------------------------#
# --------------------------------------- K-Means --------------------------------------------------#
# Support for sklearn.cluster.KMeans using objmode. We implement a basic wrapper around sklearn's
# implementation of KMeans.
# --------------------------------------------------------------------------------------------------#

# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoKMeansClusteringType = install_py_obj_class(
    types_name="kmeans_clustering_type",
    python_type=sklearn.cluster.KMeans,
    module=this_module,
    class_name="BodoKMeansClusteringType",
    model_name="BodoKMeansClusteringModel",
)


@overload(sklearn.cluster.KMeans, no_unliteral=True)
def sklearn_cluster_kmeans_overload(
    n_clusters=8,
    init="k-means++",
    n_init="auto",
    max_iter=300,
    tol=1e-4,
    verbose=0,
    random_state=None,
    copy_x=True,
    algorithm="lloyd",
):
    check_sklearn_version()

    def _sklearn_cluster_kmeans_impl(
        n_clusters=8,
        init="k-means++",
        n_init="auto",
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="lloyd",
    ):  # pragma: no cover
        with bodo.objmode(m="kmeans_clustering_type"):
            m = sklearn.cluster.KMeans(
                n_clusters=n_clusters,
                init=init,
                n_init=n_init,
                max_iter=max_iter,
                tol=tol,
                verbose=verbose,
                random_state=random_state,
                copy_x=copy_x,
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
    # _n_threads still used (https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/cluster/_kmeans.py#L1171)
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

        with bodo.objmode(m="kmeans_clustering_type"):
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
    m._n_threads = 1

    if len(X) == 0:
        # TODO If X is replicated this should be an error (same as sklearn)
        preds = np.empty(0, dtype=np.int64)
    else:
        preds = m.predict(X, sample_weight).astype(np.int64).flatten()

    # Restore
    m._n_threads = orig_nthreads
    return preds


@overload_method(BodoKMeansClusteringType, "predict", no_unliteral=True)
def overload_kmeans_clustering_predict(
    m,
    X,
    sample_weight=None,
):
    def _cluster_kmeans_predict(m, X, sample_weight=None):  # pragma: no cover
        with bodo.objmode(preds="int64[:]"):
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
        with bodo.objmode(result="float64"):
            # Don't NEED to set _n_threads becasue
            # (a) it isn't used, (b) OMP_NUM_THREADS is set to 1 by bodo init
            # But we're do it anyway in case sklearn changes its behavior later
            orig_nthreads = m._n_threads if hasattr(m, "_n_threads") else None
            m._n_threads = 1

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
        with bodo.objmode(X_new="float64[:,:]"):
            # Doesn't parallelize automatically afaik. Set n_threads to 1 anyway.
            orig_nthreads = m._n_threads if hasattr(m, "_n_threads") else None
            m._n_threads = 1

            if len(X) == 0:
                # TODO If X is replicated this should be an error (same as sklearn)
                X_new = np.empty((0, m.n_clusters), dtype=np.int64)
            else:
                X_new = m.transform(X).astype(np.float64)

            # Restore
            m._n_threads = orig_nthreads

        return X_new

    return _cluster_kmeans_transform


# -------------------------------------MultinomialNB----------------------------------------
# Support sklearn.naive_bayes.MultinomialNB using object mode of Numba
# -----------------------------------------------------------------------------
# Typing and overloads to use MultinomialNB inside Bodo functions
# directly via sklearn's API

# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoMultinomialNBType = install_py_obj_class(
    types_name="multinomial_nb_type",
    python_type=sklearn.naive_bayes.MultinomialNB,
    module=this_module,
    class_name="BodoMultinomialNBType",
    model_name="BodoMultinomialNBModel",
)


@overload(sklearn.naive_bayes.MultinomialNB, no_unliteral=True)
def sklearn_naive_bayes_multinomialnb_overload(
    alpha=1.0,
    fit_prior=True,
    class_prior=None,
):
    check_sklearn_version()

    def _sklearn_naive_bayes_multinomialnb_impl(
        alpha=1.0,
        fit_prior=True,
        class_prior=None,
    ):  # pragma: no cover
        with bodo.objmode(m="multinomial_nb_type"):
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
):  # pragma: no cover
    """MultinomialNB fit overload"""
    # If data is replicated, run scikit-learn directly
    if is_overload_false(_is_data_distributed):

        def _naive_bayes_multinomial_impl(
            m, X, y, sample_weight=None, _is_data_distributed=False
        ):  # pragma: no cover
            with bodo.objmode():
                m.fit(X, y, sample_weight)
            return m

        return _naive_bayes_multinomial_impl
    else:
        # TODO: sample_weight (future enhancement)
        if not is_overload_none(sample_weight):
            raise BodoError(
                "sklearn.naive_bayes.MultinomialNB.fit() : 'sample_weight' not supported."
            )
        func_text = "def _model_multinomial_nb_fit_impl(\n"
        func_text += "    m, X, y, sample_weight=None, _is_data_distributed=False\n"
        func_text += "):  # pragma: no cover\n"
        # Attempt to change data to numpy array. Any data that fails means, we don't support
        if y == bodo.boolean_array_type:
            # Explicitly call to_numpy() for boolean arrays because
            # coerce_to_ndarray() doesn't work for boolean array.
            func_text += "    y = y.to_numpy()\n"
        else:
            func_text += "    y = bodo.utils.conversion.coerce_to_ndarray(y)\n"
        if isinstance(X, DataFrameType) or X == bodo.boolean_array_type:
            # Explicitly call to_numpy() for boolean arrays because
            # coerce_to_ndarray() doesn't work for boolean array.
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
        func_text += '    with bodo.objmode(m="multinomial_nb_type"):\n'
        func_text += "        m = fit_multinomial_nb(\n"
        func_text += "            m, X_train, y_train, sample_weight, total_cols, _is_data_distributed\n"
        func_text += "        )\n"
        func_text += "    bodo.barrier()\n"
        func_text += "    return m\n"
        loc_vars = {}
        exec(
            func_text,
            globals(),
            loc_vars,
        )
        _model_multinomial_nb_fit_impl = loc_vars["_model_multinomial_nb_fit_impl"]
        return _model_multinomial_nb_fit_impl


def fit_multinomial_nb(
    m, X_train, y_train, sample_weight=None, total_cols=0, _is_data_distributed=False
):
    """Fit naive bayes Multinomial(parallel version)
    Since this model depends on having lots of columns, we do parallelization by columns
    """
    # 1. Compute class log probabilities
    # Taken as it's from sklearn https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/naive_bayes.py#L596
    m._check_X_y(X_train, y_train)
    _, n_features = X_train.shape
    m.n_features_in_ = n_features
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
    m.n_features_in_ = m.feature_log_prob_.shape[1]

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

# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoLogisticRegressionType = install_py_obj_class(
    types_name="logistic_regression_type",
    python_type=sklearn.linear_model.LogisticRegression,
    module=this_module,
    class_name="BodoLogisticRegressionType",
    model_name="BodoLogisticRegressionModel",
)


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
    check_sklearn_version()

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
        with bodo.objmode(m="logistic_regression_type"):
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


@register_jitable
def _raise_SGD_warning(sgd_name):
    """raise a BodoWarning for distributed training with SGD instead of user algorithm."""
    with bodo.no_warning_objmode:
        warnings.warn(
            f"Data is distributed so Bodo will fit model with SGD solver optimization ({sgd_name})",
            BodoWarning,
        )


@overload_method(BodoLogisticRegressionType, "fit", no_unliteral=True)
def overload_logistic_regression_fit(
    m,
    X,
    y,
    sample_weight=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """Logistic Regression fit overload"""
    # If data is replicated, run scikit-learn directly
    if is_overload_false(_is_data_distributed):

        def _logistic_regression_fit_impl(
            m, X, y, sample_weight=None, _is_data_distributed=False
        ):  # pragma: no cover
            with bodo.objmode():
                m.fit(X, y, sample_weight)
            return m

        return _logistic_regression_fit_impl
    else:
        if not is_overload_none(sample_weight):
            raise BodoError(
                "sklearn.linear_model.LogisticRegression.fit() : 'sample_weight' is not supported for distributed data."
            )

        # Create and run SGDClassifier(loss='log_loss')
        def _sgdc_logistic_regression_fit_impl(
            m, X, y, sample_weight=None, _is_data_distributed=False
        ):  # pragma: no cover
            if bodo.get_rank() == 0:
                _raise_SGD_warning("SGDClassifier")
            with bodo.objmode(clf="sgd_classifier_type"):
                # SGDClassifier doesn't allow l1_ratio to be None. default=0.15
                if m.l1_ratio is None:
                    l1_ratio = 0.15
                else:
                    l1_ratio = m.l1_ratio
                clf = sklearn.linear_model.SGDClassifier(
                    loss="log_loss",
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
            with bodo.objmode():
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


@overload_method(BodoLogisticRegressionType, "predict_proba", no_unliteral=True)
def overload_logistic_regression_predict_proba(m, X):
    """Overload Logistic Regression predict_proba. (Data parallelization)"""
    return parallel_predict_proba(m, X)


@overload_method(BodoLogisticRegressionType, "predict_log_proba", no_unliteral=True)
def overload_logistic_regression_predict_log_proba(m, X):
    """Overload Logistic Regression predict_log_proba. (Data parallelization)"""
    return parallel_predict_log_proba(m, X)


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
    """Overload coef_ attribute to be accessible inside bodo.jit"""

    def impl(m):  # pragma: no cover
        with bodo.objmode(result="float64[:,:]"):
            result = m.coef_
        return result

    return impl


# -------------------------------------Linear Regression--------------------
# Support sklearn.linear_model.LinearRegression object mode of Numba
# -----------------------------------------------------------------------------
# Typing and overloads to use LinearRegression inside Bodo functions
# directly via sklearn's API

# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoLinearRegressionType = install_py_obj_class(
    types_name="linear_regression_type",
    python_type=sklearn.linear_model.LinearRegression,
    module=this_module,
    class_name="BodoLinearRegressionType",
    model_name="BodoLinearRegressionModel",
)


# normalize was deprecated in version 1.0 and will be removed in 1.2.
@overload(sklearn.linear_model.LinearRegression, no_unliteral=True)
def sklearn_linear_model_linear_regression_overload(
    fit_intercept=True,
    copy_X=True,
    n_jobs=None,
    positive=False,
):
    check_sklearn_version()

    def _sklearn_linear_model_linear_regression_impl(
        fit_intercept=True,
        copy_X=True,
        n_jobs=None,
        positive=False,
    ):  # pragma: no cover
        with bodo.objmode(m="linear_regression_type"):
            m = sklearn.linear_model.LinearRegression(
                fit_intercept=fit_intercept,
                copy_X=copy_X,
                n_jobs=n_jobs,
                positive=positive,
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
    """Linear Regression fit overload"""
    # If data is replicated, run scikit-learn directly
    if is_overload_false(_is_data_distributed):

        def _linear_regression_fit_impl(
            m, X, y, sample_weight=None, _is_data_distributed=False
        ):  # pragma: no cover
            with bodo.objmode():
                m.fit(X, y, sample_weight)
            return m

        return _linear_regression_fit_impl
    else:
        if not is_overload_none(sample_weight):
            raise BodoError(
                "sklearn.linear_model.LinearRegression.fit() : 'sample_weight' is not supported for distributed data."
            )

        # Create and run SGDRegressor(loss="squared_error", penalty=None)
        def _sgdc_linear_regression_fit_impl(
            m, X, y, sample_weight=None, _is_data_distributed=False
        ):  # pragma: no cover
            if bodo.get_rank() == 0:
                _raise_SGD_warning("SGDRegressor")
            with bodo.objmode(clf="sgd_regressor_type"):
                clf = sklearn.linear_model.SGDRegressor(
                    loss="squared_error",
                    penalty=None,
                    fit_intercept=m.fit_intercept,
                )
            clf.fit(X, y, _is_data_distributed=True)
            with bodo.objmode():
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
    """Overload coef_ attribute to be accessible inside bodo.jit"""

    def impl(m):  # pragma: no cover
        with bodo.objmode(result="float64[:]"):
            result = m.coef_
        return result

    return impl


# -------------------------------------Lasso Regression--------------------
# Support sklearn.linear_model.Lasso object mode of Numba
# -----------------------------------------------------------------------------
# Typing and overloads to use Lasso inside Bodo functions
# directly via sklearn's API

# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoLassoType = install_py_obj_class(
    types_name="lasso_type",
    python_type=sklearn.linear_model.Lasso,
    module=this_module,
    class_name="BodoLassoType",
    model_name="BodoLassoModel",
)


@overload(sklearn.linear_model.Lasso, no_unliteral=True)
def sklearn_linear_model_lasso_overload(
    alpha=1.0,
    fit_intercept=True,
    precompute=False,
    copy_X=True,
    max_iter=1000,
    tol=0.0001,
    warm_start=False,
    positive=False,
    random_state=None,
    selection="cyclic",
):
    check_sklearn_version()

    def _sklearn_linear_model_lasso_impl(
        alpha=1.0,
        fit_intercept=True,
        precompute=False,
        copy_X=True,
        max_iter=1000,
        tol=0.0001,
        warm_start=False,
        positive=False,
        random_state=None,
        selection="cyclic",
    ):  # pragma: no cover
        with bodo.objmode(m="lasso_type"):
            m = sklearn.linear_model.Lasso(
                alpha=alpha,
                fit_intercept=fit_intercept,
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
    """Lasso fit overload"""
    # If data is replicated, run scikit-learn directly
    if is_overload_false(_is_data_distributed):

        def _lasso_fit_impl(
            m, X, y, sample_weight=None, check_input=True, _is_data_distributed=False
        ):  # pragma: no cover
            with bodo.objmode():
                m.fit(X, y, sample_weight, check_input)
            return m

        return _lasso_fit_impl
    else:
        if not is_overload_none(sample_weight):
            raise BodoError(
                "sklearn.linear_model.Lasso.fit() : 'sample_weight' is not supported for distributed data."
            )
        if not is_overload_true(check_input):
            raise BodoError(
                "sklearn.linear_model.Lasso.fit() : 'check_input' is not supported for distributed data."
            )

        # Create and run SGDRegressor(loss="squared_error", penalty='l1')
        def _sgdc_lasso_fit_impl(
            m, X, y, sample_weight=None, check_input=True, _is_data_distributed=False
        ):  # pragma: no cover
            if bodo.get_rank() == 0:
                _raise_SGD_warning("SGDRegressor")
            with bodo.objmode(clf="sgd_regressor_type"):
                clf = sklearn.linear_model.SGDRegressor(
                    loss="squared_error",
                    penalty="l1",
                    alpha=m.alpha,
                    fit_intercept=m.fit_intercept,
                    max_iter=m.max_iter,
                    tol=m.tol,
                    warm_start=m.warm_start,
                    random_state=m.random_state,
                )
            clf.fit(X, y, _is_data_distributed=True)
            with bodo.objmode():
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

# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoRidgeType = install_py_obj_class(
    types_name="ridge_type",
    python_type=sklearn.linear_model.Ridge,
    module=this_module,
    class_name="BodoRidgeType",
    model_name="BodoRidgeModel",
)


@overload(sklearn.linear_model.Ridge, no_unliteral=True)
def sklearn_linear_model_ridge_overload(
    alpha=1.0,
    fit_intercept=True,
    copy_X=True,
    max_iter=None,
    tol=0.001,
    solver="auto",
    positive=False,
    random_state=None,
):
    check_sklearn_version()

    def _sklearn_linear_model_ridge_impl(
        alpha=1.0,
        fit_intercept=True,
        copy_X=True,
        max_iter=None,
        tol=0.001,
        solver="auto",
        positive=False,
        random_state=None,
    ):  # pragma: no cover
        with bodo.objmode(m="ridge_type"):
            m = sklearn.linear_model.Ridge(
                alpha=alpha,
                fit_intercept=fit_intercept,
                copy_X=copy_X,
                max_iter=max_iter,
                tol=tol,
                solver=solver,
                positive=positive,
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
    """Ridge Regression fit overload"""
    # If data is replicated, run scikit-learn directly
    if is_overload_false(_is_data_distributed):

        def _ridge_fit_impl(
            m, X, y, sample_weight=None, _is_data_distributed=False
        ):  # pragma: no cover
            with bodo.objmode():
                m.fit(X, y, sample_weight)
            return m

        return _ridge_fit_impl
    else:
        if not is_overload_none(sample_weight):
            raise BodoError(
                "sklearn.linear_model.Ridge.fit() : 'sample_weight' is not supported for distributed data."
            )

        # Create and run SGDRegressor(loss="squared_error", penalty='l2')
        def _ridge_fit_impl(
            m, X, y, sample_weight=None, _is_data_distributed=False
        ):  # pragma: no cover
            if bodo.get_rank() == 0:
                _raise_SGD_warning("SGDRegressor")
            with bodo.objmode(clf="sgd_regressor_type"):
                if m.max_iter is None:
                    max_iter = 1000
                else:
                    max_iter = m.max_iter
                clf = sklearn.linear_model.SGDRegressor(
                    loss="squared_error",
                    penalty="l2",
                    alpha=0.001,
                    fit_intercept=m.fit_intercept,
                    max_iter=max_iter,
                    tol=m.tol,
                    random_state=m.random_state,
                )
            clf.fit(X, y, _is_data_distributed=True)
            with bodo.objmode():
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


@overload_attribute(BodoRidgeType, "coef_")
def get_ridge_coef(m):
    """Overload coef_ attribute to be accessible inside bodo.jit"""

    def impl(m):  # pragma: no cover
        with bodo.objmode(result="float64[:]"):
            result = m.coef_
        return result

    return impl


# ------------------------Linear Support Vector Classification-----------------
# Support sklearn.svm.LinearSVC object mode of Numba
# -----------------------------------------------------------------------------
# Typing and overloads to use LinearSVC inside Bodo functions
# directly via sklearn's API

# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoLinearSVCType = install_py_obj_class(
    types_name="linear_svc_type",
    python_type=sklearn.svm.LinearSVC,
    module=this_module,
    class_name="BodoLinearSVCType",
    model_name="BodoLinearSVCModel",
)


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
    check_sklearn_version()

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
        with bodo.objmode(m="linear_svc_type"):
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
    """Linear SVC fit overload"""
    # If data is replicated, run scikit-learn directly
    if is_overload_false(_is_data_distributed):

        def _svm_linear_svc_fit_impl(
            m, X, y, sample_weight=None, _is_data_distributed=False
        ):  # pragma: no cover
            with bodo.objmode():
                m.fit(X, y, sample_weight)
            return m

        return _svm_linear_svc_fit_impl
    else:
        if not is_overload_none(sample_weight):
            raise BodoError(
                "sklearn.svm.LinearSVC.fit() : 'sample_weight' is not supported for distributed data."
            )

        # Create and run SGDClassifier
        def _svm_linear_svc_fit_impl(
            m, X, y, sample_weight=None, _is_data_distributed=False
        ):  # pragma: no cover
            if bodo.get_rank() == 0:
                _raise_SGD_warning("SGDClassifier")
            with bodo.objmode(clf="sgd_classifier_type"):
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
            with bodo.objmode():
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
# ------------------------------------ OneHotEncoder -------------------------------------
# Support for sklearn.preprocessing.OneHotEncoder.
# Currently, only fit, transform, and get_feature_names_out are supported, as well as the
# categories_, drop_idx_, and n_features_in_ attributes.
# Support for inverse_transform is not yet added, since its output type can't be
# known at compile-time and depends on the most recent input to fit().
# We use sklearn's transform and get_feature_names_out directly in their Bodo
# implementation. For fit, we use a combination of sklearn's fit and a native
# implementation. We compute the categories seen on each rank using sklearn's
# fit implementation, then compute global values for these using MPI operations.
# ----------------------------------------------------------------------------------------


# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoPreprocessingOneHotEncoderType = install_py_obj_class(
    types_name="preprocessing_one_hot_encoder_type",
    python_type=sklearn.preprocessing.OneHotEncoder,
    module=this_module,
    class_name="BodoPreprocessingOneHotEncoderType",
    model_name="BodoPreprocessingOneHotEncoderModel",
)


# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoPreprocessingOneHotEncoderCategoriesType = install_py_obj_class(
    types_name="preprocessing_one_hot_encoder_categories_type",
    module=this_module,
    class_name="BodoPreprocessingOneHotEncoderCategoriesType",
    model_name="BodoPreprocessingOneHotEncoderCategoriesModel",
)


# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoPreprocessingOneHotEncoderDropIdxType = install_py_obj_class(
    types_name="preprocessing_one_hot_encoder_drop_idx_type",
    module=this_module,
    class_name="BodoPreprocessingOneHotEncoderDropIdxType",
    model_name="BodoPreprocessingOneHotEncoderDropIdxModel",
)


@overload_attribute(BodoPreprocessingOneHotEncoderType, "categories_")
def get_one_hot_encoder_categories_(m):
    """Overload OneHotEncoder's categories_ attribute to be accessible inside
    bodo.jit.

    Args:
        m (sklearn.preprocessing.OneHotEncoder): The OneHotEncoder to access

    Returns:
        result: The categories_ attribute of the given OneHotEncoder
    """

    def impl(m):  # pragma: no cover
        with bodo.objmode(result="preprocessing_one_hot_encoder_categories_type"):
            result = m.categories_
        return result

    return impl


@overload_attribute(BodoPreprocessingOneHotEncoderType, "drop_idx_")
def get_one_hot_encoder_drop_idx_(m):
    """Overload OneHotEncoder's drop_idx_ attribute to be accessible inside
    bodo.jit.

    Args:
        m (sklearn.preprocessing.OneHotEncoder): The OneHotEncoder to access

    Returns:
        result: The drop_idx_ attribute of the given OneHotEncoder
    """

    def impl(m):  # pragma: no cover
        with bodo.objmode(result="preprocessing_one_hot_encoder_drop_idx_type"):
            result = m.drop_idx_
        return result

    return impl


@overload_attribute(BodoPreprocessingOneHotEncoderType, "n_features_in_")
def get_one_hot_encoder_n_features_in_(m):
    """Overload OneHotEncoder's n_features_in_ attribute to be accessible inside
    bodo.jit.

    Args:
        m (sklearn.preprocessing.OneHotEncoder): The OneHotEncoder to access

    Returns:
        result: The n_features_in_ attribute of the given OneHotEncoder
    """

    def impl(m):  # pragma: no cover
        with bodo.objmode(result="int64"):
            result = m.n_features_in_
        return result

    return impl


@overload(sklearn.preprocessing.OneHotEncoder, no_unliteral=True)
def sklearn_preprocessing_one_hot_encoder_overload(
    categories="auto",
    drop=None,
    sparse_output=True,
    dtype=np.float64,
    handle_unknown="error",
    min_frequency=None,
    max_categories=None,
    feature_name_combiner="concat",
):
    """Provide implementation for __init__ function of OneHotEncoder.
    We simply call sklearn in objmode.

    Args:
        categories ('auto' or a list of array-like): Categories (unique values)
          per unique feature.
          - 'auto': Determine categories automatically from training data
          - list: categories[i] holes the categories expected in the i-th
            column. The passed categories should not mix strings and numeric
            values within a single feature, and should be sorted in case of
            numeric values.
        drop ('first', 'if_binary', or an array-like of shape (n_features,)):
          Specifies a methodology to use to drop one of the categories per
          feature. This is useful in situations where perfectly collinear
          features cause problems, such as when feeding the resulting data
          into an unregularized linear regression model. However, dropping one
          category breaks the symmetry of the original representation and can
          therefore induce a bias in downstream models, for instance penalized
          linear classification or regression models.
          - None: Retain all features (the default)
          - 'first': Drop the first category in each feature. If only one
            category is present, the feature will be dropped entirely.
          - 'if_binary': Drop the first category in each feature with two
            categories. Features with 1 or more than 2 categories are left
            intact.
          - array: drop[i] is the category in feature X[:, i] that should be
            dropped.
        sparse_output (bool): Only sparse_output=False is supported. Will return sparse
          if set True else will return an array.
        dtype (number type): Only dtype=np.float64 is supported. Desired
          datatype of output.
        handle_unknown ('error', 'ignore'): Specifies the way unknown
          categories are handled during transform.
          - 'error': Raise an error if an unknown category is present during
            transform.
          - 'ignore': When an unknown category is encountered during transform,
            the resulting one-hot-encoded columns for this feature will be all
            zeros. In the inverse transform, an unknown category will be
            denoted as None.
    """

    check_sklearn_version()

    # Because we only support dense float64 matrix output for now, check that
    # `sparse_output=False` and that `dtype=np.float64`. For compatibility with
    # check_unsupported_args, we convert `dtype` to string representation
    # since type classes aren't directly comparable.
    #
    # Adding support for additional output types would require more typing work
    # to determine the proper output type of transform().
    args_dict = {
        "sparse_output": sparse_output,
        "dtype": "float64" if "float64" in repr(dtype) else repr(dtype),
        "min_frequency": min_frequency,
        "max_categories": max_categories,
    }

    args_default_dict = {
        "sparse_output": False,
        "dtype": "float64",
        "min_frequency": None,
        "max_categories": None,
    }
    check_unsupported_args("OneHotEncoder", args_dict, args_default_dict, "ml")

    def _sklearn_preprocessing_one_hot_encoder_impl(
        categories="auto",
        drop=None,
        sparse_output=True,
        dtype=np.float64,
        handle_unknown="error",
        min_frequency=None,
        max_categories=None,
        feature_name_combiner="concat",
    ):  # pragma: no cover
        with bodo.objmode(m="preprocessing_one_hot_encoder_type"):
            m = sklearn.preprocessing.OneHotEncoder(
                categories=categories,
                drop=drop,
                sparse_output=sparse_output,
                dtype=dtype,
                handle_unknown=handle_unknown,
                min_frequency=min_frequency,
                max_categories=max_categories,
                feature_name_combiner=feature_name_combiner,
            )
        return m

    return _sklearn_preprocessing_one_hot_encoder_impl


def sklearn_preprocessing_one_hot_encoder_fit_dist_helper(m, X):
    """
    Distributed calculation of categories for one hot encoder.

    We follow sklearn's implementation of fit() and compute local fit outputs
    on each rank, before combining the results using allgatherv and reduction
    to get global outputs.

    Args:
        m (sklearn.preprocessing.OneHotEncoder): A OneHotEncoder object to fit
        X (array-like of shape (n_samples, n_features): The data to determine
          the categories of each feature.

    Returns:
        m: Fitted encoder
    """

    comm = MPI.COMM_WORLD

    # Compute local categories by using default sklearn implementation.
    # This updates m.categories_ on each local rank
    try:
        m._validate_params()
        fit_result_or_err = m._fit(
            X,
            handle_unknown=m.handle_unknown,
            force_all_finite="allow-nan",
        )
    except ValueError as e:  # pragma: no cover
        # Catch if any rank raises a ValueError for unknown categories,
        # so that we can broadcast and re-raise that error on all ranks.
        # Any other ValueErrors are re-raised
        if "Found unknown categories" in e.args[0]:
            fit_result_or_err = e
        else:
            raise e

    # If any rank raises a ValueError for unknown categories, re-raise that
    # error on all ranks to prevent deadlock on future MPI collective ops.
    # Instead of running allreduce with MPI.LOR, we use MPI.MAXLOC so that
    # the rank of the lowest failing process is also communicated. Then, we
    # broadcast the error message across all ranks.
    unknown_category_on_this_rank = int(isinstance(fit_result_or_err, ValueError))
    unknown_category_on_any_rank, failing_rank = comm.allreduce(
        (unknown_category_on_this_rank, comm.Get_rank()), op=MPI.MAXLOC
    )
    if unknown_category_on_any_rank:
        # If there's an error on any rank, broadcast the lowest erroring
        # rank's error to all ranks
        if comm.Get_rank() == failing_rank:
            err_msg = fit_result_or_err.args[0]
        else:
            err_msg = None
        err_msg = comm.bcast(err_msg, root=failing_rank)

        # Handle the case where multiple ranks raise an error. Each rank that
        # already has an error will re-raise their own error, and any rank
        # that does not have an error will re-raise the lowest rank's error.
        if unknown_category_on_this_rank:
            raise fit_result_or_err
        else:
            raise ValueError(err_msg)

    # If categories are given, aggregate local categories to global values
    # m.categories_ is a list of arrays where each array contains a list of
    # categories from the local X-data of a feature. To compute the global
    # categories for each feature, we want to allgather each rank's locally
    # computed categories for that feature and take the unique items.
    if m.categories == "auto":
        local_values_per_feat = m.categories_
        global_values_per_feat = []

        for local_values in local_values_per_feat:
            multi_local_values = bodo.allgatherv(local_values)
            global_values = _unique(multi_local_values)
            global_values_per_feat.append(global_values)

        m.categories_ = global_values_per_feat

    # Compute dropped indices. Since category info is now replicated,
    # we can just call sklearn
    m._set_drop_idx()
    m._n_features_outs = m._compute_n_features_outs()

    return m


@overload_method(BodoPreprocessingOneHotEncoderType, "fit", no_unliteral=True)
def overload_preprocessing_one_hot_encoder_fit(
    m,
    X,
    y=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """Provide implementations for OneHotEncoder's fit function.

    In case input is replicated, we simply call sklearn, else we use our native
    implementation.

    Args:
        m (sklearn.preprocessing.OneHotEncoder): A OneHotEncoder object to fit
        X (array-like of shape (n_samples, n_features): The data to determine
          the categories of each feature.
        y (ignored): Always ignored. Exists for compatibility with Pipeline
        _is_data_distributed (bool): Whether X is distributed or replicated

    Returns:
        m: Fitted encoder
    """

    func_text = "def _preprocessing_one_hot_encoder_fit_impl(\n"
    func_text += "    m, X, y=None, _is_data_distributed=False\n"
    func_text += "):\n"
    func_text += "    with bodo.objmode(m='preprocessing_one_hot_encoder_type'):\n"
    # sklearn.fit() expects a 2D array as input, but Bodo does not support
    # 2D string arrays - these are instead typed as 1D arrays of object
    # arrays. If X is provided like so, we coerce 1D array of arrays to 2D.
    func_text += "        if X.ndim == 1 and isinstance(X[0], (np.ndarray, pd.arrays.ArrowStringArray, pd.arrays.ArrowExtensionArray, list)):\n"
    func_text += "            X = np.vstack(X).astype(object)\n"

    if is_overload_true(_is_data_distributed):
        # If distributed, then use native implementation
        func_text += (
            "        m = sklearn_preprocessing_one_hot_encoder_fit_dist_helper(m, X)\n"
        )
    else:
        # If replicated, then just call sklearn
        func_text += "        m = m.fit(X, y)\n"

    func_text += "    return m\n"

    loc_vars = {}
    exec(func_text, globals(), loc_vars)
    _preprocessing_one_hot_encoder_fit_impl = loc_vars[
        "_preprocessing_one_hot_encoder_fit_impl"
    ]
    return _preprocessing_one_hot_encoder_fit_impl


@overload_method(BodoPreprocessingOneHotEncoderType, "transform", no_unliteral=True)
def overload_preprocessing_one_hot_encoder_transform(
    m,
    X,
):
    """
    Provide implementation for OneHotEncoder's transform function.
    We simply call sklearn's transform on each rank.

    Args:
        m (sklearn.preprocessing.OneHotEncoder): A OneHotEncoder object to use
        X (array-like of shape (n_samples, n_features)): The data to encode

    Returns:
        transformed_X (ndarray of shape (n_samples, n_encoded_features)):
          Transformed input.
    """

    def _preprocessing_one_hot_encoder_transform_impl(
        m,
        X,
    ):  # pragma: no cover
        with bodo.objmode(transformed_X="float64[:,:]"):
            # sklearn.fit() expects a 2D array as input, but Bodo does not support
            # 2D string arrays - these are instead typed as 1D arrays of object
            # arrays. If X is provided like so, we coerce 1D array of arrays to 2D.
            if X.ndim == 1 and isinstance(
                X[0],
                (
                    np.ndarray,
                    pd.arrays.ArrowStringArray,
                    pd.arrays.ArrowExtensionArray,
                    list,
                ),
            ):
                X = np.vstack(X).astype(object)

            transformed_X = m.transform(X)

        return transformed_X

    return _preprocessing_one_hot_encoder_transform_impl


@overload_method(
    BodoPreprocessingOneHotEncoderType, "get_feature_names_out", no_unliteral=True
)
def overload_preprocessing_one_hot_encoder_get_feature_names_out(
    m,
    input_features=None,
):
    """Provide implementation for the get_feature_names_out function.
    We simply call sklearn's get_feature_names_out on each rank.

    Args:
        m (sklearn.preprocessing.OneHotEncoder): A OneHotEncoder object to use
        input_features (array-like of string or None): Input features.
          If input_features is None, then feature_names_in_ is used as feature
          names in. If feature_names_in_ is not defined, then the following
          input feature names are generated:
          ["x0", "x1", ..., "x(n_features_in_ - 1)"].
          If input_features is an array-like, then input_features must match
          feature_names_in_ if feature_names_in_ is defined.
        X (array-like of shape (n_samples, n_features)): The data to encode

    Returns:
        transformed_X (ndarray of shape (n_samples, n_encoded_features)):
          Transformed input.
    """

    def _preprocessing_one_hot_encoder_get_feature_names_out_impl(
        m,
        input_features=None,
    ):  # pragma: no cover
        with bodo.objmode(out_features="string[:]"):
            out_features = get_feature_names_out(input_features)
        return out_features

    return _preprocessing_one_hot_encoder_get_feature_names_out_impl


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

# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoPreprocessingStandardScalerType = install_py_obj_class(
    types_name="preprocessing_standard_scaler_type",
    python_type=sklearn.preprocessing.StandardScaler,
    module=this_module,
    class_name="BodoPreprocessingStandardScalerType",
    model_name="BodoPreprocessingStandardScalerModel",
)


@overload(sklearn.preprocessing.StandardScaler, no_unliteral=True)
def sklearn_preprocessing_standard_scaler_overload(
    copy=True, with_mean=True, with_std=True
):
    """
    Provide implementation for __init__ functions of StandardScaler.
    We simply call sklearn in objmode.
    """

    check_sklearn_version()

    def _sklearn_preprocessing_standard_scaler_impl(
        copy=True, with_mean=True, with_std=True
    ):  # pragma: no cover
        with bodo.objmode(m="preprocessing_standard_scaler_type"):
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
    sample_weight=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position)
):
    """
    Provide implementations for the fit function.
    In case input is replicated, we simply call sklearn,
    else we use our native implementation.
    """
    if is_overload_true(_is_data_distributed):
        # If distributed, then use native implementation
        if not is_overload_none(sample_weight):
            raise BodoError(
                "sklearn.preprocessing.StandardScaler.fit(): "
                "'sample_weight' is not supported for distributed data."
            )

        def _preprocessing_standard_scaler_fit_impl(
            m, X, y=None, sample_weight=None, _is_data_distributed=False
        ):  # pragma: no cover
            with bodo.objmode(m="preprocessing_standard_scaler_type"):
                m = sklearn_preprocessing_standard_scaler_fit_dist_helper(m, X)

            return m

    else:
        # If replicated, then just call sklearn
        def _preprocessing_standard_scaler_fit_impl(
            m, X, y=None, sample_weight=None, _is_data_distributed=False
        ):  # pragma: no cover
            with bodo.objmode(m="preprocessing_standard_scaler_type"):
                m = m.fit(X, y, sample_weight)

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
    if isinstance(X, CSRMatrixType):
        types.csr_matrix_float64_int64 = CSRMatrixType(types.float64, types.int64)

        def _preprocessing_standard_scaler_transform_impl(
            m,
            X,
            copy=None,
        ):  # pragma: no cover
            with bodo.objmode(transformed_X="csr_matrix_float64_int64"):
                transformed_X = m.transform(X, copy=copy)
                transformed_X.indices = transformed_X.indices.astype(np.int64)
                transformed_X.indptr = transformed_X.indptr.astype(np.int64)
            return transformed_X

    else:

        def _preprocessing_standard_scaler_transform_impl(
            m,
            X,
            copy=None,
        ):  # pragma: no cover
            with bodo.objmode(transformed_X="float64[:,:]"):
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
    if isinstance(X, CSRMatrixType):
        types.csr_matrix_float64_int64 = CSRMatrixType(types.float64, types.int64)

        def _preprocessing_standard_scaler_inverse_transform_impl(
            m,
            X,
            copy=None,
        ):  # pragma: no cover
            with bodo.objmode(inverse_transformed_X="csr_matrix_float64_int64"):
                inverse_transformed_X = m.inverse_transform(X, copy=copy)
                inverse_transformed_X.indices = inverse_transformed_X.indices.astype(
                    np.int64
                )
                inverse_transformed_X.indptr = inverse_transformed_X.indptr.astype(
                    np.int64
                )
            return inverse_transformed_X

    else:

        def _preprocessing_standard_scaler_inverse_transform_impl(
            m,
            X,
            copy=None,
        ):  # pragma: no cover
            with bodo.objmode(inverse_transformed_X="float64[:,:]"):
                inverse_transformed_X = m.inverse_transform(X, copy=copy)
            return inverse_transformed_X

    return _preprocessing_standard_scaler_inverse_transform_impl


# ----------------------------------------------------------------------------------------
# ------------------------------------ Max-Abs-Scaler ------------------------------------
# Support for sklearn.preprocessing.MaxAbsScaler.
# Currently only fit, partial_fit, transform, and inverse_transform are supported.
# We use sklearn's transform and inverse_transform directly in their Bodo implementation.
# For fit and partial_fit, we use a combination of sklearn's fit function and a native
# implementation. We compute the max_abs and num_samples_seen on each rank using
# sklearn's fit implementation, then we compute the global values for these using MPI.
# ----------------------------------------------------------------------------------------


# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoPreprocessingMaxAbsScalerType = install_py_obj_class(
    types_name="preprocessing_max_abs_scaler_type",
    python_type=sklearn.preprocessing.MaxAbsScaler,
    module=this_module,
    class_name="BodoPreprocessingMaxAbsScalerType",
    model_name="BodoPreprocessingMaxAbsScalerModel",
)


@overload_attribute(BodoPreprocessingMaxAbsScalerType, "scale_")
def get_max_abs_scaler_scale_(m):
    """Overload scale_ attribute to be accessible inside bodo.jit"""

    def impl(m):  # pragma: no cover
        with bodo.objmode(result="float64[:]"):
            result = m.scale_
        return result

    return impl


@overload_attribute(BodoPreprocessingMaxAbsScalerType, "max_abs_")
def get_max_abs_scaler_max_abs_(m):
    """Overload max_abs_ attribute to be accessible inside bodo.jit"""

    def impl(m):  # pragma: no cover
        with bodo.objmode(result="float64[:]"):
            result = m.max_abs_
        return result

    return impl


@overload_attribute(BodoPreprocessingMaxAbsScalerType, "n_samples_seen_")
def get_max_abs_scaler_n_samples_seen_(m):
    """Overload n_samples_seen_ attribute to be accessible inside bodo.jit"""

    def impl(m):  # pragma: no cover
        with bodo.objmode(result="int64"):
            result = m.n_samples_seen_
        return result

    return impl


@overload(sklearn.preprocessing.MaxAbsScaler, no_unliteral=True)
def sklearn_preprocessing_max_abs_scaler_overload(copy=True):
    """
    Provide implementation for __init__ functions of MaxAbsScaler.
    We simply call sklearn in objmode.
    """

    check_sklearn_version()

    def _sklearn_preprocessing_max_abs_scaler_impl(copy=True):  # pragma: no cover
        with bodo.objmode(m="preprocessing_max_abs_scaler_type"):
            m = sklearn.preprocessing.MaxAbsScaler(copy=copy)
        return m

    return _sklearn_preprocessing_max_abs_scaler_impl


def sklearn_preprocessing_max_abs_scaler_fit_dist_helper(m, X, partial=False):
    """
    Distributed calculation of max_abs for max abs scaler.
    We use sklearn to calculate max_abs and n_samples_seen, then combine
    the results appropriately to get the global max_abs and n_samples_seen.
    """

    comm = MPI.COMM_WORLD
    num_pes = comm.Get_size()

    # Save old n_samples_seen_
    if hasattr(m, "n_samples_seen_"):
        old_n_samples_seen_ = m.n_samples_seen_
    else:
        old_n_samples_seen_ = 0

    # Call to get the max_abs and n_samples_seen
    if partial:
        m = m.partial_fit(X)
    else:
        m = m.fit(X)

    # Compute global n_samples_seen
    global_n_samples_seen = comm.allreduce(
        m.n_samples_seen_ - old_n_samples_seen_, op=MPI.SUM
    )
    m.n_samples_seen_ = global_n_samples_seen + old_n_samples_seen_

    # Compute global max_abs
    local_max_abs_by_rank = np.zeros(
        (num_pes, *m.max_abs_.shape), dtype=m.max_abs_.dtype
    )
    comm.Allgather(m.max_abs_, local_max_abs_by_rank)
    global_max_abs = np.nanmax(local_max_abs_by_rank, axis=0)

    # Re-compute the rest of the attributes
    m.scale_ = sklearn_handle_zeros_in_scale(global_max_abs)
    m.max_abs_ = global_max_abs

    return m


@overload_method(BodoPreprocessingMaxAbsScalerType, "fit", no_unliteral=True)
def overload_preprocessing_max_abs_scaler_fit(
    m,
    X,
    y=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """
    Provide implementations for the fit function.
    In case input is replicated, we simply call sklearn,
    else we use our native implementation.
    """
    if _is_data_distributed:
        # If distributed, then use native implementation
        def _preprocessing_max_abs_scaler_fit_impl(
            m, X, y=None, _is_data_distributed=False
        ):  # pragma: no cover
            with bodo.objmode(m="preprocessing_max_abs_scaler_type"):
                m = sklearn_preprocessing_max_abs_scaler_fit_dist_helper(
                    m, X, partial=False
                )
            return m

    else:
        # If replicated, then just call sklearn
        def _preprocessing_max_abs_scaler_fit_impl(
            m, X, y=None, _is_data_distributed=False
        ):  # pragma: no cover
            with bodo.objmode(m="preprocessing_max_abs_scaler_type"):
                m = m.fit(X, y)
            return m

    return _preprocessing_max_abs_scaler_fit_impl


@overload_method(BodoPreprocessingMaxAbsScalerType, "partial_fit", no_unliteral=True)
def overload_preprocessing_max_abs_scaler_partial_fit(
    m,
    X,
    y=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """
    Provide implementations for the partial_fit function.
    In case input is replicated, we simply call sklearn,
    else we use our native implementation.
    """
    if _is_data_distributed:
        # If distributed, then use native implementation
        def _preprocessing_max_abs_scaler_partial_fit_impl(
            m, X, y=None, _is_data_distributed=False
        ):  # pragma: no cover
            with bodo.objmode(m="preprocessing_max_abs_scaler_type"):
                m = sklearn_preprocessing_max_abs_scaler_fit_dist_helper(
                    m, X, partial=True
                )
            return m

    else:
        # If replicated, then just call sklearn
        def _preprocessing_max_abs_scaler_partial_fit_impl(
            m, X, y=None, _is_data_distributed=False
        ):  # pragma: no cover
            with bodo.objmode(m="preprocessing_max_abs_scaler_type"):
                m = m.partial_fit(X, y)
            return m

    return _preprocessing_max_abs_scaler_partial_fit_impl


@overload_method(BodoPreprocessingMaxAbsScalerType, "transform", no_unliteral=True)
def overload_preprocessing_max_abs_scaler_transform(
    m,
    X,
):
    """
    Provide implementation for the transform function.
    We simply call sklearn's transform on each rank.
    """
    if isinstance(X, CSRMatrixType):
        types.csr_matrix_float64_int64 = CSRMatrixType(types.float64, types.int64)

        def _preprocessing_max_abs_scaler_transform_impl(
            m,
            X,
        ):  # pragma: no cover
            with bodo.objmode(transformed_X="csr_matrix_float64_int64"):
                transformed_X = m.transform(X)
                transformed_X.indices = transformed_X.indices.astype(np.int64)
                transformed_X.indptr = transformed_X.indptr.astype(np.int64)
            return transformed_X

    else:

        def _preprocessing_max_abs_scaler_transform_impl(
            m,
            X,
        ):  # pragma: no cover
            with bodo.objmode(transformed_X="float64[:,:]"):
                transformed_X = m.transform(X)
            return transformed_X

    return _preprocessing_max_abs_scaler_transform_impl


@overload_method(
    BodoPreprocessingMaxAbsScalerType, "inverse_transform", no_unliteral=True
)
def overload_preprocessing_max_abs_scaler_inverse_transform(
    m,
    X,
):
    """
    Provide implementation for the inverse_transform function.
    We simply call sklearn's inverse_transform on each rank.
    """
    if isinstance(X, CSRMatrixType):
        types.csr_matrix_float64_int64 = CSRMatrixType(types.float64, types.int64)

        def _preprocessing_max_abs_scaler_inverse_transform_impl(
            m,
            X,
        ):  # pragma: no cover
            with bodo.objmode(inverse_transformed_X="csr_matrix_float64_int64"):
                inverse_transformed_X = m.inverse_transform(X)
                inverse_transformed_X.indices = inverse_transformed_X.indices.astype(
                    np.int64
                )
                inverse_transformed_X.indptr = inverse_transformed_X.indptr.astype(
                    np.int64
                )
            return inverse_transformed_X

    else:

        def _preprocessing_max_abs_scaler_inverse_transform_impl(
            m,
            X,
        ):  # pragma: no cover
            with bodo.objmode(inverse_transformed_X="float64[:,:]"):
                inverse_transformed_X = m.inverse_transform(X)
            return inverse_transformed_X

    return _preprocessing_max_abs_scaler_inverse_transform_impl


# ----------------------------------------------------------------------------------------
# ----------------------------------------LeavePOut---------------------------------------
# Support for sklearn.model_selection.LeavePOut.

# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoModelSelectionLeavePOutType = install_py_obj_class(
    types_name="model_selection_leave_p_out_type",
    python_type=sklearn.model_selection.LeavePOut,
    module=this_module,
    class_name="BodoModelSelectionLeavePOutType",
    model_name="BodoModelSelectionLeavePOutModel",
)


# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoModelSelectionLeavePOutGeneratorType = install_py_obj_class(
    types_name="model_selection_leave_p_out_generator_type",
    module=this_module,
    class_name="BodoModelSelectionLeavePOutGeneratorType",
    model_name="BodoModelSelectionLeavePOutGeneratorModel",
)


@overload(sklearn.model_selection.LeavePOut, no_unliteral=True)
def sklearn_model_selection_leave_p_out_overload(
    p,
):
    """
    Provide implementation for __init__ function of LeavePOut.
    We simply call sklearn in objmode.
    """

    check_sklearn_version()

    def _sklearn_model_selection_leave_p_out_impl(
        p,
    ):  # pragma: no cover
        with bodo.objmode(m="model_selection_leave_p_out_type"):
            m = sklearn.model_selection.LeavePOut(
                p=p,
            )
        return m

    return _sklearn_model_selection_leave_p_out_impl


def sklearn_model_selection_leave_p_out_generator_dist_helper(m, X):
    """
    Distributed calculation of train/test split indices for LeavePOut.
    We use sklearn on all the indices, then filter out the indices assigned
    to each individual rank.
    """
    # Compute index offset of each rank
    my_rank = bodo.get_rank()
    nranks = bodo.get_size()
    rank_data_len = np.empty(nranks, np.int64)
    bodo.libs.distributed_api.allgather(rank_data_len, len(X))
    if my_rank > 0:
        rank_start = np.sum(rank_data_len[:my_rank])
    else:
        rank_start = 0
    rank_end = rank_start + rank_data_len[my_rank]

    # Compute total data size
    global_data_size = np.sum(rank_data_len)

    # Raise error if dataset is too small
    if global_data_size <= m.p:
        raise ValueError(
            f"p={m.p} must be strictly less than the number of samples={global_data_size}"
        )

    # For all possible test set combinations, compute train and test indices
    # that belong to the current rank.
    # Since `combinations` returns deterministic and fixed-ordered output,
    # in lexicographic ordering according to the order of the input iterable,
    # this is safe to do in parallel on all ranks at once
    local_indices = np.arange(rank_start, rank_end)
    for combination in combinations(range(global_data_size), m.p):
        test_index = np.array(combination)
        test_index = test_index[test_index >= rank_start]
        test_index = test_index[test_index < rank_end]

        test_mask = np.zeros(len(X), dtype=bool)
        test_mask[test_index - rank_start] = True

        train_index = local_indices[np.logical_not(test_mask)]
        yield train_index, test_index


@overload_method(BodoModelSelectionLeavePOutType, "split", no_unliteral=True)
def overload_model_selection_leave_p_out_generator(
    m,
    X,
    y=None,
    groups=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """
    Provide implementations for the split function, which is a generator.
    In case input is replicated, we simply call sklearn,
    else we use our native implementation.
    """

    if is_overload_true(_is_data_distributed):
        # If distributed, then use native implementation
        def _model_selection_leave_p_out_generator_impl(
            m, X, y=None, groups=None, _is_data_distributed=False
        ):  # pragma: no cover
            with bodo.objmode(gen="model_selection_leave_p_out_generator_type"):
                gen = sklearn_model_selection_leave_p_out_generator_dist_helper(m, X)
            return gen

    else:
        # If replicated, then just call sklearn
        def _model_selection_leave_p_out_generator_impl(
            m, X, y=None, groups=None, _is_data_distributed=False
        ):  # pragma: no cover
            with bodo.objmode(gen="model_selection_leave_p_out_generator_type"):
                gen = m.split(X, y=y, groups=groups)
            return gen

    return _model_selection_leave_p_out_generator_impl


@overload_method(BodoModelSelectionLeavePOutType, "get_n_splits", no_unliteral=True)
def overload_model_selection_leave_p_out_get_n_splits(
    m,
    X,
    y=None,
    groups=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """
    Provide implementations for the get_n_splits function.
    """

    if is_overload_true(_is_data_distributed):
        # If distributed, then use native implementation
        def _model_selection_leave_p_out_get_n_splits_impl(
            m, X, y=None, groups=None, _is_data_distributed=False
        ):  # pragma: no cover
            with bodo.objmode(out="int64"):
                global_data_size = bodo.libs.distributed_api.dist_reduce(
                    len(X), np.int32(Reduce_Type.Sum.value)
                )
                out = int(comb(global_data_size, m.p, exact=True))
            return out

    else:
        # If replicated, then just call sklearn
        def _model_selection_leave_p_out_get_n_splits_impl(
            m, X, y=None, groups=None, _is_data_distributed=False
        ):  # pragma: no cover
            with bodo.objmode(out="int64"):
                out = m.get_n_splits(X)

            return out

    return _model_selection_leave_p_out_get_n_splits_impl


# ----------------------------------------------------------------------------------------
# ---------------------------------------- KFold -----------------------------------------
# Support for sklearn.model_selection.KFold.
# Both get_n_splits and split functions are supported.
# For split, if data is distributed and shuffle=False, use sklearn individually
# on each rank then add a rank offset. If data is distributed and shuffle=True,
# use sklearn on each rank individually, add a rank offset, then permute the output.
#
# Our implementation differs from sklearn's to ensure both train and test data are
# evenly distributed across ranks if possible. For example, if X=range(8), nprocs=2,
# and n_splits=4, then our first fold is test = [0,4] and train = [1,2,3,5,6,7],
# while sklearn's first fold is test = [0,1] and train = [2,3,4,5,6,7].
# ----------------------------------------------------------------------------------------


# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoModelSelectionKFoldType = install_py_obj_class(
    types_name="model_selection_kfold_type",
    python_type=sklearn.model_selection.KFold,
    module=this_module,
    class_name="BodoModelSelectionKFoldType",
    model_name="BodoModelSelectionKFoldModel",
)


@overload(sklearn.model_selection.KFold, no_unliteral=True)
def sklearn_model_selection_kfold_overload(
    n_splits=5,
    shuffle=False,
    random_state=None,
):
    """
    Provide implementation for __init__ function of KFold.
    We simply call sklearn in objmode.
    """

    check_sklearn_version()

    def _sklearn_model_selection_kfold_impl(
        n_splits=5,
        shuffle=False,
        random_state=None,
    ):  # pragma: no cover
        with bodo.objmode(m="model_selection_kfold_type"):
            m = sklearn.model_selection.KFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state,
            )
        return m

    return _sklearn_model_selection_kfold_impl


def sklearn_model_selection_kfold_generator_dist_helper(m, X, y=None, groups=None):
    """
    Distributed calculation of train/test split indices for KFold.
    We use sklearn on the indices assigned to each individual rank,
    then add the rank offset afterwards.
    """
    # Compute index offset of each rank
    my_rank = bodo.get_rank()
    nranks = bodo.get_size()
    rank_data_len = np.empty(nranks, np.int64)
    bodo.libs.distributed_api.allgather(rank_data_len, len(X))
    if my_rank > 0:
        rank_start = np.sum(rank_data_len[:my_rank])
    else:
        rank_start = 0
    rank_end = rank_start + len(X)

    # Compute total data size and global/local indices
    global_data_size = np.sum(rank_data_len)

    if global_data_size < m.n_splits:
        raise ValueError(
            f"number of splits n_splits={m.n_splits} greater than the number of samples {global_data_size}"
        )

    global_indices = np.arange(global_data_size)
    if m.shuffle:
        if m.random_state is None:
            seed = bodo.libs.distributed_api.bcast_scalar(np.random.randint(0, 2**31))
            np.random.seed(seed)
        else:
            np.random.seed(m.random_state)
        np.random.shuffle(global_indices)

    local_indices = global_indices[rank_start:rank_end]

    # Compute local fold sizes so that global fold sizes match sklearn's.
    # Suppose that m.n_splits = 3, global_data_size = 22 and nranks = 4, so
    # len(X) = [6, 6, 5, 5] on each rank. We want our global fold sizes to
    # be [8, 7, 7] to match sklearn, which yields these local fold sizes:
    #
    #       fold0  fold1  fold2
    # rank0   2      2      2
    # rank1   2      2      2
    # rank2   2      2      1
    # rank3   2      1      2
    #
    # Assuming that data is evenly distributed, each local fold has exactly
    # `global_data_size // (nranks * m.n_splits)` elements or maybe one more.
    # First, we compute the number of extra elements per fold, and further
    # subdivide into [4, 3, 3] extra elements for folds 0, 1, and 2. We use
    # np.repeat() to expand this into [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]. Now,
    # slicing this array by `my_rank::n_ranks` tells us which local folds get
    # an extra element in the current rank. Example: In rank 0, arr[0::4] gives
    # [arr[0], arr[4], arr[8]] which is [0, 1, 2]; while in rank 2, arr[2::4]
    # gives [arr[2], arr[6]] which is [0, 1].

    local_fold_sizes = np.full(
        m.n_splits, global_data_size // (nranks * m.n_splits), dtype=np.int32
    )

    n_extras = global_data_size % (nranks * m.n_splits)
    extras_per_local_fold = np.full(m.n_splits, n_extras // m.n_splits, dtype=int)
    extras_per_local_fold[: n_extras % m.n_splits] += 1

    global_extra_locs = np.repeat(np.arange(m.n_splits), extras_per_local_fold)
    local_extra_locs = global_extra_locs[my_rank::nranks]
    local_fold_sizes[local_extra_locs] += 1

    start = 0
    for fold_size in local_fold_sizes:
        stop = start + fold_size
        test_index = local_indices[start:stop]
        train_index = np.concatenate(
            (local_indices[:start], local_indices[stop:]), axis=0
        )
        yield train_index, test_index
        start = stop


@overload_method(BodoModelSelectionKFoldType, "split", no_unliteral=True)
def overload_model_selection_kfold_generator(
    m,
    X,
    y=None,
    groups=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """
    Provide implementations for the split function.
    In case input is replicated, we simply call sklearn,
    else we use our native implementation.

    Although the split function is a generator in sklearn, returning it as an
    opaque type to the user (as we do in LeavePOut) means we cannot iterate
    through folds in jitted code. As a quick hack to fix this, we return the
    result as a list of (train_idx, test_idx) tuples across all folds.
    This has O(nk) memory cost instead of O(n) for the generator case.

    Properly supporting split by returning an actual generator would require
    lowering the generator to numba and implementing `getiter` and `iternext`.
    """

    if is_overload_true(_is_data_distributed):
        # If distributed, then use native implementation

        def _model_selection_kfold_generator_impl(
            m, X, y=None, groups=None, _is_data_distributed=False
        ):  # pragma: no cover
            # Since we do not support iterating through generators directly,
            # as an unperformant hack, we convert the output to a list
            with bodo.objmode(gen="List(UniTuple(int64[:], 2))"):
                gen = list(
                    sklearn_model_selection_kfold_generator_dist_helper(
                        m, X, y=None, groups=None
                    )
                )

            return gen

    else:
        # If replicated, then just call sklearn

        def _model_selection_kfold_generator_impl(
            m, X, y=None, groups=None, _is_data_distributed=False
        ):  # pragma: no cover
            # Since we do not support iterating through generators directly,
            # as an unperformant hack, we convert the output to a list
            with bodo.objmode(gen="List(UniTuple(int64[:], 2))"):
                gen = list(m.split(X, y=y, groups=groups))

            return gen

    return _model_selection_kfold_generator_impl


@overload_method(BodoModelSelectionKFoldType, "get_n_splits", no_unliteral=True)
def overload_model_selection_kfold_get_n_splits(
    m,
    X=None,
    y=None,
    groups=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """
    Provide implementations for the get_n_splits function.
    We simply return the model's value of n_splits.
    """

    def _model_selection_kfold_get_n_splits_impl(
        m, X=None, y=None, groups=None, _is_data_distributed=False
    ):  # pragma: no cover
        with bodo.objmode(out="int64"):
            out = m.n_splits
        return out

    return _model_selection_kfold_get_n_splits_impl


# ---------------------------------------------------------------------------------------
# -----------------------------------train_test_split------------------------------------
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
    if train_size is None:
        train_size = -1.0
    if test_size is None:
        test_size = -1.0
    if train_size == -1.0 and test_size == -1.0:
        return 0.75, 0.25
    elif test_size == -1.0:
        return train_size, 1.0 - train_size
    elif train_size == -1.0:
        return 1.0 - test_size, test_size
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
    """Reset labels to its original type if changed"""
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
    check_sklearn_version()
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
    check_unsupported_args("train_test_split", args_dict, args_default_dict, "ml")
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
        func_text += "    with bodo.objmode(data_train='{}', data_test='{}', labels_train='{}', labels_test='{}'):\n".format(
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
        exec(func_text, globals(), loc_vars)
        _train_test_split_impl = loc_vars["_train_test_split_impl"]
        return _train_test_split_impl
    else:
        global get_data_slice_parallel
        if isinstance(get_data_slice_parallel, pytypes.FunctionType):
            get_data_slice_parallel = bodo.jit(
                get_data_slice_parallel,
                all_args_distributed_varlength=True,
                all_returns_distributed=True,
            )

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


# -----------------------------------------------------------------------------
# ----------------------------------shuffle------------------------------------


@overload(sklearn.utils.shuffle, no_unliteral=True)
def sklearn_utils_shuffle_overload(
    data,
    random_state=None,
    n_samples=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """
    Implement shuffle. If data is replicated, we simply call sklearn,
    else we use our native implementation.
    This simple implementation only supports one array for now.
    """
    if is_overload_false(_is_data_distributed):
        # If data is not distributed, then just call sklearn
        # Note: _is_data_distributed is set in the Distributed compiler pass
        #
        # Here, data is the underlying numba type of `data`. We need to set the
        # kwargs of objmode to be compile-time constants that represent the
        # output type of each PyObject defined under the bodo.objmode context.
        #
        # Following https://github.com/numba/numba/blob/main/numba/core/withcontexts.py#L182
        # and https://github.com/numba/numba/blob/main/numba/core/sigutils.py#L12,
        # bodo.objmode() will eval() the given type annotation string, with
        # the entries of numba.core.types as global variables, to determine the
        # type signature of each output.

        # Therefore, we need to define a unique entry for `data`'s type within
        # numba.core.types:
        data_type_name = f"utils_shuffle_type_{numba.core.ir_utils.next_label()}"
        if isinstance(data, (DataFrameType, SeriesType)):
            # Following train_test_split, make sure we use NumericIndexType
            # over other unsupported index types for pandas inputs
            data_typ = data.copy(index=NumericIndexType(types.int64))
            setattr(types, data_type_name, data_typ)
        else:
            setattr(types, data_type_name, data)
        func_text = "def _utils_shuffle_impl(\n"
        func_text += (
            "    data, random_state=None, n_samples=None, _is_data_distributed=False\n"
        )
        func_text += "):\n"
        func_text += f"    with bodo.objmode(out='{data_type_name}'):\n"
        func_text += "        out = sklearn.utils.shuffle(\n"
        func_text += (
            "            data, random_state=random_state, n_samples=n_samples\n"
        )
        func_text += "        )\n"
        func_text += "    return out\n"
        loc_vars = {}
        exec(func_text, globals(), loc_vars)
        _utils_shuffle_impl = loc_vars["_utils_shuffle_impl"]

    else:
        # If distributed, directly call bodo random_shuffle
        def _utils_shuffle_impl(
            data, random_state=None, n_samples=None, _is_data_distributed=False
        ):  # pragma: no cover
            m = bodo.random_shuffle(
                data, seed=random_state, n_samples=n_samples, parallel=True
            )
            return m

    return _utils_shuffle_impl


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


# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoPreprocessingMinMaxScalerType = install_py_obj_class(
    types_name="preprocessing_minmax_scaler_type",
    python_type=sklearn.preprocessing.MinMaxScaler,
    module=this_module,
    class_name="BodoPreprocessingMinMaxScalerType",
    model_name="BodoPreprocessingMinMaxScalerModel",
)


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

    check_sklearn_version()

    def _sklearn_preprocessing_minmax_scaler_impl(
        feature_range=(0, 1),
        copy=True,
        clip=False,
    ):  # pragma: no cover
        with bodo.objmode(m="preprocessing_minmax_scaler_type"):
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
        with bodo.objmode(m="preprocessing_minmax_scaler_type"):
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
        with bodo.objmode(transformed_X="float64[:,:]"):
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
        with bodo.objmode(inverse_transformed_X="float64[:,:]"):
            inverse_transformed_X = m.inverse_transform(X)
        return inverse_transformed_X

    return _preprocessing_minmax_scaler_inverse_transform_impl


# ----------------------------------------------------------------------------------------
# ----------------------------------- Robust-Scaler --------------------------------------
# Support for sklearn.preprocessing.RobustScaler.
# Currently only fit, transform and inverse_transform functions are supported.
# We use sklearn's transform and inverse_transform directly in their Bodo implementation.
# For distributed fit, we use a native implementation where we use our quantile_parallel
# and median array_kernels.
# ----------------------------------------------------------------------------------------


# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoPreprocessingRobustScalerType = install_py_obj_class(
    types_name="preprocessing_robust_scaler_type",
    python_type=sklearn.preprocessing.RobustScaler,
    module=this_module,
    class_name="BodoPreprocessingRobustScalerType",
    model_name="BodoPreprocessingRobustScalerModel",
)


@overload_attribute(BodoPreprocessingRobustScalerType, "with_centering")
def get_robust_scaler_with_centering(m):
    """Overload with_centering attribute to be accessible inside bodo.jit"""

    def impl(m):  # pragma: no cover
        with bodo.objmode(result="boolean"):
            result = m.with_centering
        return result

    return impl


@overload_attribute(BodoPreprocessingRobustScalerType, "with_scaling")
def get_robust_scaler_with_scaling(m):
    """Overload with_scaling attribute to be accessible inside bodo.jit"""

    def impl(m):  # pragma: no cover
        with bodo.objmode(result="boolean"):
            result = m.with_scaling
        return result

    return impl


@overload_attribute(BodoPreprocessingRobustScalerType, "quantile_range")
def get_robust_scaler_quantile_range(m):
    """Overload quantile_range attribute to be accessible inside bodo.jit"""

    typ = numba.typeof((25.0, 75.0))

    def impl(m):  # pragma: no cover
        with bodo.objmode(result=typ):
            result = m.quantile_range
        return result

    return impl


@overload_attribute(BodoPreprocessingRobustScalerType, "unit_variance")
def get_robust_scaler_unit_variance(m):
    """Overload unit_variance attribute to be accessible inside bodo.jit"""

    def impl(m):  # pragma: no cover
        with bodo.objmode(result="boolean"):
            result = m.unit_variance
        return result

    return impl


@overload_attribute(BodoPreprocessingRobustScalerType, "copy")
def get_robust_scaler_copy(m):
    """Overload copy attribute to be accessible inside bodo.jit"""

    def impl(m):  # pragma: no cover
        with bodo.objmode(result="boolean"):
            result = m.copy
        return result

    return impl


@overload_attribute(BodoPreprocessingRobustScalerType, "center_")
def get_robust_scaler_center_(m):
    """Overload center_ attribute to be accessible inside bodo.jit"""

    def impl(m):  # pragma: no cover
        with bodo.objmode(result="float64[:]"):
            result = m.center_
        return result

    return impl


@overload_attribute(BodoPreprocessingRobustScalerType, "scale_")
def get_robust_scaler_scale_(m):
    """Overload scale_ attribute to be accessible inside bodo.jit"""

    def impl(m):  # pragma: no cover
        with bodo.objmode(result="float64[:]"):
            result = m.scale_
        return result

    return impl


@overload(sklearn.preprocessing.RobustScaler, no_unliteral=True)
def sklearn_preprocessing_robust_scaler_overload(
    with_centering=True,
    with_scaling=True,
    quantile_range=(25.0, 75.0),
    copy=True,
    unit_variance=False,
):
    """
    Provide implementation for __init__ functions of RobustScaler.
    We simply call sklearn in objmode.
    """

    check_sklearn_version()

    def _sklearn_preprocessing_robust_scaler_impl(
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
        copy=True,
        unit_variance=False,
    ):  # pragma: no cover
        with bodo.objmode(m="preprocessing_robust_scaler_type"):
            m = sklearn.preprocessing.RobustScaler(
                with_centering=with_centering,
                with_scaling=with_scaling,
                quantile_range=quantile_range,
                copy=copy,
                unit_variance=unit_variance,
            )
        return m

    return _sklearn_preprocessing_robust_scaler_impl


@overload_method(BodoPreprocessingRobustScalerType, "fit", no_unliteral=True)
def overload_preprocessing_robust_scaler_fit(
    m,
    X,
    y=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position)
):
    """
    Provide implementations for the fit function.
    In case input is replicated, we simply call sklearn,
    else we use our native implementation.
    We only support numpy arrays and Pandas DataFrames at the moment.
    CSR matrices are not yet supported.
    """

    check_sklearn_version()

    # TODO Add general error-checking [BE-52]

    if is_overload_true(_is_data_distributed):
        # If distributed, then use native implementation

        func_text = f"def preprocessing_robust_scaler_fit_impl(\n"
        func_text += f"  m, X, y=None, _is_data_distributed=False\n"
        func_text += f"):\n"

        # In case a DataFrame was provided, convert it to a Numpy Array first.
        # This is required since we'll be looping over the columns, which
        # is not supported for DataFrames.
        # TODO Add a compile time check that all columns are numeric since
        # `to_numpy` will error out otherwise anyway. [BE-52]
        if isinstance(X, DataFrameType):
            func_text += f"  X = X.to_numpy()\n"

        func_text += f"  with bodo.objmode(qrange_l='float64', qrange_r='float64'):\n"
        func_text += f"    (qrange_l, qrange_r) = m.quantile_range\n"
        func_text += f"  if not 0 <= qrange_l <= qrange_r <= 100:\n"
        # scikit-learn throws the error: `"Invalid quantile range: %s" % str(self.quantile_range)`
        # but we cannot use format strings, so we use a slightly modified error message.
        func_text += f"    raise ValueError(\n"
        func_text += f"      'Invalid quantile range provided. Ensure that 0 <= quantile_range[0] <= quantile_range[1] <= 100.'\n"
        func_text += f"    )\n"
        func_text += f"  qrange_l, qrange_r = qrange_l / 100.0, qrange_r / 100.0\n"
        func_text += f"  X = bodo.utils.conversion.coerce_to_array(X)\n"
        func_text += f"  num_features = X.shape[1]\n"
        func_text += f"  if m.with_scaling:\n"
        func_text += f"    scales = np.zeros(num_features)\n"
        func_text += f"  else:\n"
        func_text += f"    scales = None\n"
        func_text += f"  if m.with_centering:\n"
        func_text += f"    centers = np.zeros(num_features)\n"
        func_text += f"  else:\n"
        func_text += f"    centers = None\n"
        func_text += f"  if m.with_scaling or m.with_centering:\n"
        ## XXX Not sure if prange is useful here
        func_text += f"    numba.parfors.parfor.init_prange()\n"
        func_text += f"    for feature_idx in numba.parfors.parfor.internal_prange(num_features):\n"
        func_text += f"      column_data = bodo.utils.conversion.ensure_contig_if_np(X[:, feature_idx])\n"
        func_text += f"      if m.with_scaling:\n"
        func_text += f"        q1 = bodo.libs.array_kernels.quantile_parallel(\n"
        func_text += f"          column_data, qrange_l, 0\n"
        func_text += f"        )\n"
        func_text += f"        q2 = bodo.libs.array_kernels.quantile_parallel(\n"
        func_text += f"          column_data, qrange_r, 0\n"
        func_text += f"        )\n"
        func_text += f"        scales[feature_idx] = q2 - q1\n"
        func_text += f"      if m.with_centering:\n"
        func_text += (
            f"        centers[feature_idx] = bodo.libs.array_ops.array_op_median(\n"
        )
        func_text += f"          column_data, True, True\n"
        func_text += f"        )\n"
        func_text += f"  if m.with_scaling:\n"
        # Handle zeros (See sklearn.preprocessing._data._handle_zeros_in_scale)
        # RobustScaler.fit calls
        # `self.scale_ = _handle_zeros_in_scale(self.scale_, copy=False)`
        # which translates to:
        func_text += f"    constant_mask = scales < 10 * np.finfo(scales.dtype).eps\n"
        func_text += f"    scales[constant_mask] = 1.0\n"
        func_text += f"    if m.unit_variance:\n"
        func_text += f"      with bodo.objmode(adjust='float64'):\n"
        func_text += (
            f"        adjust = stats.norm.ppf(qrange_r) - stats.norm.ppf(qrange_l)\n"
        )
        func_text += f"      scales = scales / adjust\n"
        func_text += f"  with bodo.objmode():\n"
        func_text += f"    m.center_ = centers\n"
        func_text += f"    m.scale_ = scales\n"
        func_text += f"  return m\n"

        loc_vars = {}
        exec(
            func_text,
            globals(),
            loc_vars,
        )
        _preprocessing_robust_scaler_fit_impl = loc_vars[
            "preprocessing_robust_scaler_fit_impl"
        ]
        return _preprocessing_robust_scaler_fit_impl
    else:
        # If replicated, then just use sklearn implementation

        def _preprocessing_robust_scaler_fit_impl(
            m, X, y=None, _is_data_distributed=False
        ):  # pragma: no cover
            with bodo.objmode(m="preprocessing_robust_scaler_type"):
                m = m.fit(X, y)
            return m

        return _preprocessing_robust_scaler_fit_impl


@overload_method(BodoPreprocessingRobustScalerType, "transform", no_unliteral=True)
def overload_preprocessing_robust_scaler_transform(
    m,
    X,
):
    """
    Provide implementation for the transform function.
    We simply call sklearn's transform on each rank.
    """

    check_sklearn_version()

    def _preprocessing_robust_scaler_transform_impl(
        m,
        X,
    ):  # pragma: no cover
        with bodo.objmode(transformed_X="float64[:,:]"):
            transformed_X = m.transform(X)
        return transformed_X

    return _preprocessing_robust_scaler_transform_impl


@overload_method(
    BodoPreprocessingRobustScalerType, "inverse_transform", no_unliteral=True
)
def overload_preprocessing_robust_scaler_inverse_transform(
    m,
    X,
):
    """
    Provide implementation for the inverse_transform function.
    We simply call sklearn's inverse_transform on each rank.
    """

    check_sklearn_version()

    def _preprocessing_robust_scaler_inverse_transform_impl(
        m,
        X,
    ):  # pragma: no cover
        with bodo.objmode(inverse_transformed_X="float64[:,:]"):
            inverse_transformed_X = m.inverse_transform(X)
        return inverse_transformed_X

    return _preprocessing_robust_scaler_inverse_transform_impl


# ----------------------------------------------------------------------------------------
# ------------------------------------- LabelEncoder--------------------------------------
# Support for sklearn.preprocessing.LabelEncoder.
# Currently only fit, fit_transform, transform and inverse_transform functions are supported.
# We use sklearn's transform and inverse_transform directly in their Bodo implementation.
# For fit, we use np.unique and then replicate its output to be classes_ attribute
# ----------------------------------------------------------------------------------------


def _pa_str_to_obj(a):
    """Convert string[pyarrow] arrays to object arrays to workaround Scikit-learn issues
    as of 1.4.0. See test_label_encoder.
    """
    if isinstance(a, pd.arrays.ArrowStringArray):
        return a.astype(object)
    return a


# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoPreprocessingLabelEncoderType = install_py_obj_class(
    types_name="preprocessing_label_encoder_type",
    python_type=sklearn.preprocessing.LabelEncoder,
    module=this_module,
    class_name="BodoPreprocessingLabelEncoderType",
    model_name="BodoPreprocessingLabelEncoderModel",
)


@overload(sklearn.preprocessing.LabelEncoder, no_unliteral=True)
def sklearn_preprocessing_label_encoder_overload():
    """
    Provide implementation for __init__ functions of LabelEncoder.
    We simply call sklearn in objmode.
    """

    check_sklearn_version()

    def _sklearn_preprocessing_label_encoder_impl():  # pragma: no cover
        with bodo.objmode(m="preprocessing_label_encoder_type"):
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
            y = bodo.utils.typing.decode_if_dict_array(y)
            y_classes = bodo.libs.array_kernels.unique(y, parallel=True)
            y_classes = bodo.allgatherv(y_classes, False)
            y_classes = bodo.libs.array_kernels.sort(
                y_classes, ascending=True, inplace=False
            )
            with bodo.objmode:
                y_classes_obj = _pa_str_to_obj(y_classes)
                m.classes_ = y_classes_obj

            return m

        return _sklearn_preprocessing_label_encoder_fit_impl

    else:
        # If replicated, then just call sklearn
        def _sklearn_preprocessing_label_encoder_fit_impl(
            m, y, _is_data_distributed=False
        ):  # pragma: no cover
            with bodo.objmode(m="preprocessing_label_encoder_type"):
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
        with bodo.objmode(transformed_y="int64[:]"):
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
            with bodo.objmode(transformed_y="int64[:]"):
                transformed_y = m.fit_transform(y)
            return transformed_y

        return _preprocessing_label_encoder_fit_transform_impl


# ----------------------------------------------------------------------------------------
# ----------------------------------- HashingVectorizer------------------------------------
# Support for sklearn.feature_extraction.text.HashingVectorizer
# Currently only fit_transform function is supported.
# We use sklearn's fit_transform directly in objmode on each rank.
# ----------------------------------------------------------------------------------------


# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoFExtractHashingVectorizerType = install_py_obj_class(
    types_name="f_extract_hashing_vectorizer_type",
    python_type=sklearn.feature_extraction.text.HashingVectorizer,
    module=this_module,
    class_name="BodoFExtractHashingVectorizerType",
    model_name="BodoFExtractHashingVectorizerModel",
)


@overload(sklearn.feature_extraction.text.HashingVectorizer, no_unliteral=True)
def sklearn_hashing_vectorizer_overload(
    input="content",
    encoding="utf-8",
    decode_error="strict",
    strip_accents=None,
    lowercase=True,
    preprocessor=None,
    tokenizer=None,
    stop_words=None,
    token_pattern=r"(?u)\b\w\w+\b",
    ngram_range=(1, 1),
    analyzer="word",
    n_features=(2**20),
    binary=False,
    norm="l2",
    alternate_sign=True,
    dtype=np.float64,
):
    """
    Provide implementation for __init__ functions of HashingVectorizer.
    We simply call sklearn in objmode.
    """

    check_sklearn_version()

    def _sklearn_hashing_vectorizer_impl(
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        analyzer="word",
        n_features=(2**20),
        binary=False,
        norm="l2",
        alternate_sign=True,
        dtype=np.float64,
    ):  # pragma: no cover
        with bodo.objmode(m="f_extract_hashing_vectorizer_type"):
            m = sklearn.feature_extraction.text.HashingVectorizer(
                input=input,
                encoding=encoding,
                decode_error=decode_error,
                strip_accents=strip_accents,
                lowercase=lowercase,
                preprocessor=preprocessor,
                tokenizer=tokenizer,
                stop_words=stop_words,
                token_pattern=token_pattern,
                ngram_range=ngram_range,
                analyzer=analyzer,
                n_features=n_features,
                binary=binary,
                norm=norm,
                alternate_sign=alternate_sign,
                dtype=dtype,
            )
        return m

    return _sklearn_hashing_vectorizer_impl


@overload_method(BodoFExtractHashingVectorizerType, "fit_transform", no_unliteral=True)
def overload_hashing_vectorizer_fit_transform(
    m,
    X,
    y=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """
    Provide implementation for the fit_transform function.
    We simply call sklearn's fit_transform on each rank.
    """
    types.csr_matrix_float64_int64 = CSRMatrixType(types.float64, types.int64)

    def _hashing_vectorizer_fit_transform_impl(
        m,
        X,
        y=None,
        _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
    ):  # pragma: no cover
        with bodo.objmode(transformed_X="csr_matrix_float64_int64"):
            transformed_X = m.fit_transform(X, y)
            transformed_X.indices = transformed_X.indices.astype(np.int64)
            transformed_X.indptr = transformed_X.indptr.astype(np.int64)
        return transformed_X

    return _hashing_vectorizer_fit_transform_impl


# -----------------------------------------------------------------------------
# Typing and overloads to use RandomForestRegressor inside Bodo functions
# directly via sklearn's API


# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoRandomForestRegressorType = install_py_obj_class(
    types_name="random_forest_regressor_type",
    python_type=sklearn.ensemble.RandomForestRegressor,
    module=this_module,
    class_name="BodoRandomForestRegressorType",
    model_name="BodoRandomForestRegressorModel",
)


@overload(sklearn.ensemble.RandomForestRegressor, no_unliteral=True)
def overload_sklearn_rf_regressor(
    n_estimators=100,
    criterion="squared_error",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=1.0,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=None,
    verbose=0,
    warm_start=False,
    ccp_alpha=0.0,
    max_samples=None,
    monotonic_cst=None,
):
    """
    Provide implementation for __init__ functions of RandomForestRegressor.
    We simply call sklearn in objmode.
    """

    # TODO n_jobs should be left unspecified so should probably throw an error if used

    check_sklearn_version()

    def _sklearn_ensemble_RandomForestRegressor_impl(
        n_estimators=100,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
    ):  # pragma: no cover
        with bodo.objmode(m="random_forest_regressor_type"):
            if random_state is not None and get_num_nodes() > 1:
                print("With multinode, fixed random_state seed values are ignored.\n")
                random_state = None
            m = sklearn.ensemble.RandomForestRegressor(
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=1,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                ccp_alpha=ccp_alpha,
                max_samples=max_samples,
                monotonic_cst=monotonic_cst,
            )
        return m

    return _sklearn_ensemble_RandomForestRegressor_impl


@overload_method(BodoRandomForestRegressorType, "predict", no_unliteral=True)
def overload_rf_regressor_predict(m, X):
    """Overload Random Forest Regressor predict. (Data parallelization)"""
    return parallel_predict_regression(m, X)


@overload_method(BodoRandomForestRegressorType, "score", no_unliteral=True)
def overload_rf_regressor_score(
    m,
    X,
    y,
    sample_weight=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """Overload Random Forest Regressor score."""
    return parallel_score(m, X, y, sample_weight, _is_data_distributed)


@overload_method(BodoRandomForestRegressorType, "fit", no_unliteral=True)
@overload_method(BodoRandomForestClassifierType, "fit", no_unliteral=True)
def overload_rf_classifier_model_fit(
    m,
    X,
    y,
    sample_weight=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """Distribute data to first rank in each node then call fit operation"""
    classname = "RandomForestClassifier"
    if isinstance(m, BodoRandomForestRegressorType):
        classname = "RandomForestRegressor"
    if not is_overload_none(sample_weight):
        raise BodoError(
            f"sklearn.ensemble.{classname}.fit() : 'sample_weight' is not supported for distributed data."
        )

    def _model_fit_impl(
        m, X, y, sample_weight=None, _is_data_distributed=False
    ):  # pragma: no cover
        # Get lowest rank in each node
        with bodo.objmode(first_rank_node="int32[:]"):
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

        with bodo.objmode:
            random_forest_model_fit(m, X, y)  # return value is m

        bodo.barrier()
        return m

    return _model_fit_impl


# ----------------------------------------------------------------------------------------
# ----------------------------------- CountVectorizer------------------------------------
# Support for sklearn.feature_extraction.text.CountVectorizer
# Currently fit_transform & get_feature_names_out functions are supported.
# ----------------------------------------------------------------------------------------


# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoFExtractCountVectorizerType = install_py_obj_class(
    types_name="f_extract_count_vectorizer_type",
    python_type=sklearn.feature_extraction.text.CountVectorizer,
    module=this_module,
    class_name="BodoFExtractCountVectorizerType",
    model_name="BodoFExtractCountVectorizerModel",
)


@overload(sklearn.feature_extraction.text.CountVectorizer, no_unliteral=True)
def sklearn_count_vectorizer_overload(
    input="content",
    encoding="utf-8",
    decode_error="strict",
    strip_accents=None,
    lowercase=True,
    preprocessor=None,
    tokenizer=None,
    stop_words=None,
    token_pattern=r"(?u)\b\w\w+\b",
    ngram_range=(1, 1),
    analyzer="word",
    max_df=1.0,
    min_df=1,
    max_features=None,
    vocabulary=None,
    binary=False,
    dtype=np.int64,
):
    """
    Provide implementation for __init__ functions of CountVectorizer.
    We simply call sklearn in objmode.
    """

    check_sklearn_version()

    # Per sklearn documentation, min_df: ignore terms that have a document
    # frequency strictly lower than the given threshold.

    if not is_overload_constant_number(min_df) or get_overload_const(min_df) != 1:
        raise BodoError(
            "sklearn.feature_extraction.text.CountVectorizer(): 'min_df' is not supported for distributed data.\n"
        )

    # Per sklearn documentation, max_df: ignore terms that have a document
    # frequency strictly higher than the given threshold.
    if not is_overload_constant_number(max_df) or get_overload_const(min_df) != 1:
        raise BodoError(
            "sklearn.feature_extraction.text.CountVectorizer(): 'max_df' is not supported for distributed data.\n"
        )

    def _sklearn_count_vectorizer_impl(
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        analyzer="word",
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.int64,
    ):  # pragma: no cover
        with bodo.objmode(m="f_extract_count_vectorizer_type"):
            m = sklearn.feature_extraction.text.CountVectorizer(
                input=input,
                encoding=encoding,
                decode_error=decode_error,
                strip_accents=strip_accents,
                lowercase=lowercase,
                preprocessor=preprocessor,
                tokenizer=tokenizer,
                stop_words=stop_words,
                token_pattern=token_pattern,
                ngram_range=ngram_range,
                analyzer=analyzer,
                max_df=max_df,
                min_df=min_df,
                max_features=max_features,
                vocabulary=vocabulary,
                binary=binary,
                dtype=dtype,
            )
        return m

    return _sklearn_count_vectorizer_impl


@overload_attribute(BodoFExtractCountVectorizerType, "vocabulary_")
def get_cv_vocabulary_(m):
    """Overload vocabulary_ attribute to be accessible inside bodo.jit"""

    types.dict_string_int = types.DictType(types.unicode_type, types.int64)

    def impl(m):  # pragma: no cover
        with bodo.objmode(result="dict_string_int"):
            result = m.vocabulary_
        return result

    return impl


def _cv_fit_transform_helper(m, X):
    """Initial fit computation to get vocabulary if user didn't provide it"""
    change_voc = False
    local_vocabulary = m.vocabulary
    if m.vocabulary is None:
        m.fit(X)
        local_vocabulary = m.vocabulary_
        change_voc = True
    return change_voc, local_vocabulary


@overload_method(BodoFExtractCountVectorizerType, "fit_transform", no_unliteral=True)
def overload_count_vectorizer_fit_transform(
    m,
    X,
    y=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """
    Provide implementation for the fit_transform function.
    If distributed, run fit to get vocabulary on each rank locally and gather it.
    Then, run fit_transform with combined vocabulary
    If replicated, simply call fit_transform on each rank.
    """
    check_sklearn_version()
    types.csr_matrix_int64_int64 = CSRMatrixType(types.int64, types.int64)
    if is_overload_true(_is_data_distributed):
        types.dict_str_int = types.DictType(types.unicode_type, types.int64)

        def _count_vectorizer_fit_transform_impl(
            m,
            X,
            y=None,
            _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
        ):  # pragma: no cover
            with bodo.objmode(local_vocabulary="dict_str_int", changeVoc="bool_"):
                changeVoc, local_vocabulary = _cv_fit_transform_helper(m, X)
            # Gather vocabulary from each rank and generate its integer indices (alphabetical order)
            if changeVoc:
                local_vocabulary = bodo.utils.conversion.coerce_to_array(
                    list(local_vocabulary.keys())
                )
                all_vocabulary = bodo.libs.array_kernels.unique(
                    local_vocabulary, parallel=True
                )
                all_vocabulary = bodo.allgatherv(all_vocabulary, False)
                all_vocabulary = bodo.libs.array_kernels.sort(
                    all_vocabulary, ascending=True, inplace=True
                )
                new_data = {}
                for i in range(len(all_vocabulary)):
                    new_data[all_vocabulary[i]] = i
            else:
                new_data = local_vocabulary
            # Run fit_transform with generated vocabulary_
            with bodo.objmode(transformed_X="csr_matrix_int64_int64"):
                if changeVoc:
                    m.vocabulary = new_data
                transformed_X = m.fit_transform(X, y)
                transformed_X.indices = transformed_X.indices.astype(np.int64)
                transformed_X.indptr = transformed_X.indptr.astype(np.int64)
            return transformed_X

        return _count_vectorizer_fit_transform_impl
    else:
        # If replicated, then just call sklearn
        def _count_vectorizer_fit_transform_impl(
            m,
            X,
            y=None,
            _is_data_distributed=False,
        ):  # pragma: no cover
            with bodo.objmode(transformed_X="csr_matrix_int64_int64"):
                transformed_X = m.fit_transform(X, y)
                transformed_X.indices = transformed_X.indices.astype(np.int64)
                transformed_X.indptr = transformed_X.indptr.astype(np.int64)
            return transformed_X

        return _count_vectorizer_fit_transform_impl


# NOTE: changed get_feature_names as it will be removed in 1.2
# and will be replaced by get_feature_names_out
@overload_method(
    BodoFExtractCountVectorizerType, "get_feature_names_out", no_unliteral=True
)
def overload_count_vectorizer_get_feature_names_out(m):
    """Array mapping from feature integer indices to feature name."""

    check_sklearn_version()

    def impl(m):  # pragma: no cover
        with bodo.objmode(result=bodo.string_array_type):
            result = m.get_feature_names_out()
        return result

    return impl
