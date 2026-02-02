"""Support scikit-learn linear models."""

import sys
import warnings

import numba
import numpy as np
import pandas as pd
import sklearn.feature_extraction
from mpi4py import MPI
from numba.extending import (
    overload,
    overload_attribute,
    overload_method,
    register_jitable,
)
from sklearn.metrics import hinge_loss, log_loss, mean_squared_error

import bodo
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
    BodoWarning,
    is_overload_false,
    is_overload_none,
    is_overload_true,
)

this_module = sys.modules[__name__]
# -------------------------------------SGDClassifier----------------------------------------
# Support sklearn.linear_model.SGDClassifier using object mode of Numba
# The model it fits can be controlled with the loss parameter; by default, it fits a linear support vector machine (SVM).
# Logistic regression (loss='log_loss')
# -----------------------------------------------------------------------------

# Typing and overloads to use SGDClassifier inside Bodo functions
# directly via sklearn's API

# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoSGDClassifierType, _ = install_py_obj_class(
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
    elif m.loss == "log_loss":
        loss_func = log_loss
    elif m.loss == "squared_error":
        loss_func = mean_squared_error
    else:
        raise ValueError(f"loss {m.loss} not supported")

    if isinstance(y_classes, pd.arrays.ArrowExtensionArray):
        y_classes = y_classes.to_numpy()

    if not (regC := isinstance(m, sklearn.linear_model.SGDRegressor)):
        # Function used to produce input for loss function
        predict_func = m.predict_proba if m.loss == "log_loss" else m.decision_function

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
            y_pred = predict_func(X)
            cur_loss = loss_func(y, y_pred, labels=y_classes)
        cur_loss_sum = comm.allreduce(cur_loss, op=MPI.SUM)
        cur_loss = cur_loss_sum / nranks
        # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/linear_model/_sgd_fast.pyx#L620
        if m.tol > -np.inf and cur_loss > best_loss - m.tol * total_datasize:
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

            with numba.objmode(m="sgd_classifier_type"):
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
            with numba.objmode(m="sgd_classifier_type"):
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
        with numba.objmode(result="float64[:,:]"):
            result = m.coef_
        return result

    return impl


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
BodoSGDRegressorType, _ = install_py_obj_class(
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
    coef_init=None,
    intercept_init=None,
    sample_weight=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
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
            with numba.objmode(m="sgd_regressor_type"):
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
            with numba.objmode(m="sgd_regressor_type"):
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


# -------------------------------------Logisitic Regression--------------------
# Support sklearn.linear_model.LogisticRegression object mode of Numba
# -----------------------------------------------------------------------------
# Typing and overloads to use LogisticRegression inside Bodo functions
# directly via sklearn's API

# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoLogisticRegressionType, _ = install_py_obj_class(
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


@register_jitable
def _raise_SGD_warning(sgd_name):
    """raise a BodoWarning for distributed training with SGD instead of user algorithm."""
    with bodo.ir.object_mode.no_warning_objmode:
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
            with numba.objmode():
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
            with numba.objmode(clf="sgd_classifier_type"):
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
        with numba.objmode(result="float64[:,:]"):
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
BodoLinearRegressionType, _ = install_py_obj_class(
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
    def _sklearn_linear_model_linear_regression_impl(
        fit_intercept=True,
        copy_X=True,
        n_jobs=None,
        positive=False,
    ):  # pragma: no cover
        with numba.objmode(m="linear_regression_type"):
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
            with numba.objmode():
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
            with numba.objmode(clf="sgd_regressor_type"):
                clf = sklearn.linear_model.SGDRegressor(
                    loss="squared_error",
                    penalty=None,
                    fit_intercept=m.fit_intercept,
                )
            clf.fit(X, y, _is_data_distributed=True)
            with numba.objmode():
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
        with numba.objmode(result="float64[:]"):
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
BodoLassoType, _ = install_py_obj_class(
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
        with numba.objmode(m="lasso_type"):
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
            with numba.objmode():
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
            with numba.objmode(clf="sgd_regressor_type"):
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
            with numba.objmode():
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
BodoRidgeType, _ = install_py_obj_class(
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
        with numba.objmode(m="ridge_type"):
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
            with numba.objmode():
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
            with numba.objmode(clf="sgd_regressor_type"):
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
            with numba.objmode():
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
        with numba.objmode(result="float64[:]"):
            result = m.coef_
        return result

    return impl
