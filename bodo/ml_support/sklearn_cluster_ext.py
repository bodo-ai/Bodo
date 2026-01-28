"""Support scikit-learn Clustering Algorithms"""

import sys

import numba
import numpy as np
import sklearn.cluster
from mpi4py import MPI
from numba.extending import (
    overload,
    overload_method,
)

import bodo
from bodo.libs.distributed_api import get_host_ranks
from bodo.utils.py_objs import install_py_obj_class

this_module = sys.modules[__name__]
# --------------------------------------------------------------------------------------------------#
# --------------------------------------- K-Means --------------------------------------------------#
# Support for sklearn.cluster.KMeans using objmode. We implement a basic wrapper around sklearn's
# implementation of KMeans.
# --------------------------------------------------------------------------------------------------#

# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoKMeansClusteringType, _ = install_py_obj_class(
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
        with numba.objmode(m="kmeans_clustering_type"):
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

        with numba.objmode(m="kmeans_clustering_type"):
            m = kmeans_fit_helper(
                m, len(X), all_X, all_sample_weight, _is_data_distributed
            )

        return m

    return _cluster_kmeans_fit_impl


def kmeans_predict_helper(m, X):
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
        preds = m.predict(X).astype(np.int64).flatten()

    # Restore
    m._n_threads = orig_nthreads
    return preds


@overload_method(BodoKMeansClusteringType, "predict", no_unliteral=True)
def overload_kmeans_clustering_predict(
    m,
    X,
):
    def _cluster_kmeans_predict(m, X):  # pragma: no cover
        with numba.objmode(preds="int64[:]"):
            # TODO: Set _n_threads to 1, even though it shouldn't be necessary
            preds = kmeans_predict_helper(m, X)
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
        with numba.objmode(X_new="float64[:,:]"):
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
