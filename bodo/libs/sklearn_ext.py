"""Support sklearn.ensemble.RandomForestClassifier using object mode of Numba
"""
import numpy as np
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

from bodo.utils.typing import is_overload_constant_str, get_overload_const_str

import bodo
from bodo.libs.distributed_api import get_node_portion
from mpi4py import MPI
import itertools
import sklearn.ensemble


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
