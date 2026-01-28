"""Support scikit-learn Ensemble-based methods for
classification, regression and anomaly detection.
"""

import itertools
import sys

import numba
import numpy as np
import sklearn.ensemble
from mpi4py import MPI
from numba.extending import (
    overload,
    overload_method,
)

import bodo
from bodo.libs.distributed_api import (
    create_subcomm_mpi4py,
    get_host_ranks,
    get_nodes_first_ranks,
    get_num_nodes,
)
from bodo.ml_support.sklearn_ext import (
    parallel_predict,
    parallel_predict_log_proba,
    parallel_predict_proba,
    parallel_predict_regression,
    parallel_score,
)
from bodo.utils.py_objs import install_py_obj_class
from bodo.utils.typing import (
    BodoError,
    is_overload_none,
)

this_module = sys.modules[__name__]
# -----------------------------------------------------------------------------
# Typing and overloads to use RandomForestClassifier inside Bodo functions
# directly via sklearn's API

# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoRandomForestClassifierType, _ = install_py_obj_class(
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
        with numba.objmode(m="random_forest_classifier_type"):
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


@overload_method(BodoRandomForestClassifierType, "predict", no_unliteral=True)
def overload_model_predict(m, X):
    """Overload Random Forest Classifier predict. (Data parallelization)"""
    return parallel_predict(m, X)


@overload_method(BodoRandomForestClassifierType, "predict_proba", no_unliteral=True)
def overload_rf_predict_proba(m, X):
    """Overload Random Forest Classifier predict_proba. (Data parallelization)"""
    return parallel_predict_proba(m, X)


@overload_method(BodoRandomForestClassifierType, "predict_log_proba", no_unliteral=True)
def overload_rf_predict_log_proba(m, X):
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


# -----------------------------------------------------------------------------
# Typing and overloads to use RandomForestRegressor inside Bodo functions
# directly via sklearn's API


# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoRandomForestRegressorType, random_forest_regressor_type = install_py_obj_class(
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
        with numba.objmode(m="random_forest_regressor_type"):
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

        from joblib import parallel_backend

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
        with numba.objmode(first_rank_node="int32[:]"):
            first_rank_node = get_nodes_first_ranks()
        if _is_data_distributed:
            nnodes = len(first_rank_node)
            X = bodo.gatherv(X)
            y = bodo.gatherv(y)
            # Broadcast X, y to first rank in each node
            if nnodes > 1:
                X = bodo.libs.distributed_api.bcast(X, comm_ranks=first_rank_node)
                y = bodo.libs.distributed_api.bcast(y, comm_ranks=first_rank_node)

        with numba.objmode:
            random_forest_model_fit(m, X, y)  # return value is m

        bodo.barrier()
        return m

    return _model_fit_impl
