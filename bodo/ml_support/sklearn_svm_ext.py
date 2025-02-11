"""Support scikit-learn SVM algorithms."""

import sys

import sklearn.svm
from numba.extending import (
    overload,
    overload_method,
)

import bodo
from bodo.ml_support.sklearn_ext import (
    check_sklearn_version,
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

# ------------------------Linear Support Vector Classification-----------------
# Support sklearn.svm.LinearSVC object mode of Numba
# -----------------------------------------------------------------------------
# Typing and overloads to use LinearSVC inside Bodo functions
# directly via sklearn's API

# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoLinearSVCType, _ = install_py_obj_class(
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
                bodo.ml_support.sklearn_linear_model_ext._raise_SGD_warning(
                    "SGDClassifier"
                )
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
