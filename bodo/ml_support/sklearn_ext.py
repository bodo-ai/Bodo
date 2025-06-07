"""Support scikit-learn using object mode of Numba"""

import numpy as np
from scipy import stats  # noqa

import bodo


def parallel_predict_regression(m, X):
    """
    Implement the regression prediction operation in parallel.
    Each rank has its own copy of the model and predicts for its
    own set of data.
    """

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
