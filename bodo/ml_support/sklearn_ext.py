"""Support scikit-learn using object mode of Numba"""

import sys

import numpy as np
import sklearn.linear_model
import sklearn.utils
from scipy import stats  # noqa

import bodo
from bodo.utils.typing import (
    BodoError,
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
