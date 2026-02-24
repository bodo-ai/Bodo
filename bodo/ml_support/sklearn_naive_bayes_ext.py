"""Support scikit-learn Naive Bayes algorithms."""

import sys

import numba
import numpy as np
import sklearn.naive_bayes
from mpi4py import MPI
from numba.extending import (
    overload,
    overload_method,
)
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import _check_sample_weight

import bodo
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.ml_support.sklearn_ext import (
    parallel_predict,
    parallel_score,
)
from bodo.utils.py_objs import install_py_obj_class
from bodo.utils.typing import (
    BodoError,
    is_overload_false,
    is_overload_none,
)

this_module = sys.modules[__name__]

# -------------------------------------MultinomialNB----------------------------------------
# Support sklearn.naive_bayes.MultinomialNB using object mode of Numba
# -----------------------------------------------------------------------------
# Typing and overloads to use MultinomialNB inside Bodo functions
# directly via sklearn's API

# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoMultinomialNBType, _ = install_py_obj_class(
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
):  # pragma: no cover
    """MultinomialNB fit overload"""
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
        if not is_overload_none(sample_weight):
            raise BodoError(
                "sklearn.naive_bayes.MultinomialNB.fit() : 'sample_weight' not supported."
            )
        func_text = "def _model_multinomial_nb_fit_impl(\n"
        func_text += "    m, X, y, sample_weight=None, _is_data_distributed=False\n"
        func_text += "):  # pragma: no cover\n"
        # Attempt to change data to numpy array. Any data that fails means, we don't support
        if y == bodo.types.boolean_array_type:
            # Explicitly call to_numpy() for boolean arrays because
            # coerce_to_ndarray() doesn't work for boolean array.
            func_text += "    y = y.to_numpy()\n"
        else:
            func_text += "    y = bodo.utils.conversion.coerce_to_ndarray(y)\n"
        if isinstance(X, DataFrameType) or X == bodo.types.boolean_array_type:
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
        func_text += '    with numba.objmode(m="multinomial_nb_type"):\n'
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
    sizes = np.ones(nranks, dtype=np.int64) * (total_cols // nranks)
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
