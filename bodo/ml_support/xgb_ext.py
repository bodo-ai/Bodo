"""Support XGBoost (Scikit-Learn Wrapper interface for XGBoost) using object mode of Numba"""
# Tests are in https://github.com/bodo-ai/engine-e2e-tests/tree/xgb-tests/xgboost

import sys

import numpy as np
import xgboost
from numba.extending import overload, overload_attribute, overload_method

import bodo
from bodo.utils.py_objs import install_py_obj_class

this_module = sys.modules[__name__]
# ------------------------XGBClassifier-----------------
# Support xgboost.XGBClassifier object mode of Numba
# -----------------------------------------------------------------------------
# Typing and overloads to use XGBClassifier inside Bodo functions
# directly via sklearn's API


BodoXGBClassifierType, _ = install_py_obj_class(
    types_name="xgbclassifier_type",
    python_type=xgboost.XGBClassifier,
    module=this_module,
    class_name="BodoXGBClassifierType",
    model_name="BodoXGBClassifierModel",
)


@overload(xgboost.XGBClassifier, no_unliteral=True)
def sklearn_xgbclassifier_overload(
    objective="binary:logistic",
    use_label_encoder=True,
    n_estimators=100,
    n_jobs=None,
    random_state=None,
    max_depth=None,
    learning_rate=None,
    verbosity=None,
    booster=None,
    tree_method=None,
    gamma=None,
    min_child_weight=None,
    max_delta_step=None,
    subsample=None,
    colsample_bytree=None,
    colsample_bylevel=None,
    colsample_bynode=None,
    reg_alpha=None,
    reg_lambda=None,
    scale_pos_weight=None,
    base_score=None,
    missing=np.nan,
    num_parallel_tree=None,
    monotone_constraints=None,
    interaction_constraints=None,
    importance_type="gain",
    gpu_id=None,
    validate_parameters=None,
):  # pragma: no cover
    def _xgbclassifier_impl(
        objective="binary:logistic",
        use_label_encoder=True,
        n_estimators=100,
        n_jobs=None,
        random_state=None,
        max_depth=None,
        learning_rate=None,
        verbosity=None,
        booster=None,
        tree_method=None,
        gamma=None,
        min_child_weight=None,
        max_delta_step=None,
        subsample=None,
        colsample_bytree=None,
        colsample_bylevel=None,
        colsample_bynode=None,
        reg_alpha=None,
        reg_lambda=None,
        scale_pos_weight=None,
        base_score=None,
        missing=np.nan,
        num_parallel_tree=None,
        monotone_constraints=None,
        interaction_constraints=None,
        importance_type="gain",
        gpu_id=None,
        validate_parameters=None,
    ):  # pragma: no cover
        with bodo.objmode(m="xgbclassifier_type"):
            xgboost.rabit.init()
            m = xgboost.XGBClassifier(
                objective=objective,
                use_label_encoder=use_label_encoder,
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                random_state=random_state,
                max_depth=max_depth,
                learning_rate=learning_rate,
                verbosity=verbosity,
                booster=booster,
                tree_method=tree_method,
                gamma=gamma,
                min_child_weight=min_child_weight,
                max_delta_step=max_delta_step,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                colsample_bylevel=colsample_bylevel,
                colsample_bynode=colsample_bynode,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                scale_pos_weight=scale_pos_weight,
                base_score=base_score,
                missing=missing,
                num_parallel_tree=num_parallel_tree,
                monotone_constraints=monotone_constraints,
                interaction_constraints=interaction_constraints,
                importance_type=importance_type,
                gpu_id=gpu_id,
                validate_parameters=validate_parameters,
            )
        return m

    return _xgbclassifier_impl


@overload_method(BodoXGBClassifierType, "fit", no_unliteral=True)
def overload_xgbclassifier_fit(
    m,
    X,
    y,
    sample_weight=None,
    base_margin=None,
    eval_set=None,
    eval_metric=None,
    early_stopping_rounds=None,
    verbose=True,
    xgb_model=None,
    sample_weight_eval_set=None,
    feature_weights=None,
    callbacks=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):  # pragma: no cover
    """XGBClassifier fit overload"""

    def _xgbclassifier_fit_impl(
        m,
        X,
        y,
        sample_weight=None,
        base_margin=None,
        eval_set=None,
        eval_metric=None,
        early_stopping_rounds=None,
        verbose=True,
        xgb_model=None,
        sample_weight_eval_set=None,
        feature_weights=None,
        callbacks=None,
        _is_data_distributed=False,
    ):  # pragma: no cover
        with bodo.objmode():
            m.n_jobs = 1
            m.fit(
                X,
                y,
                sample_weight,
                base_margin,
                eval_set,
                eval_metric,
                early_stopping_rounds,
                verbose,
                xgb_model,
                sample_weight_eval_set,
                feature_weights,
                callbacks,
            )
        return m

    return _xgbclassifier_fit_impl


@overload_method(BodoXGBClassifierType, "predict", no_unliteral=True)
def overload_xgbclassifier_predict(
    m,
    X,
    output_margin=False,
    ntree_limit=None,
    validate_features=True,
    base_margin=None,
):  # pragma: no cover
    """Overload XGBClassifier predict."""

    def _model_predict_impl(
        m,
        X,
        output_margin=False,
        ntree_limit=None,
        validate_features=True,
        base_margin=None,
    ):  # pragma: no cover
        # TODO: return data could be of different type or shape (2D)
        with bodo.objmode(result="int64[:]"):
            m.n_jobs = 1
            result = (
                m.predict(X, output_margin, ntree_limit, validate_features, base_margin)
                .astype(np.int64)
                .flatten()
            )
        return result

    return _model_predict_impl


@overload_method(BodoXGBClassifierType, "predict_proba", no_unliteral=True)
def overload_xgbclassifier_predict_proba(
    m,
    X,
    ntree_limit=None,
    validate_features=True,
    base_margin=None,
):  # pragma: no cover
    """Overload XGBClassifier predict."""

    def _model_predict_proba_impl(
        m,
        X,
        ntree_limit=None,
        validate_features=True,
        base_margin=None,
    ):  # pragma: no cover
        with bodo.objmode(result="float64[:,:]"):
            m.n_jobs = 1
            result = m.predict_proba(
                X, ntree_limit, validate_features, base_margin
            ).astype(np.float64)
        return result

    return _model_predict_proba_impl


# ------------------------XGBRegressor-----------------
# Support xgboost.XGBRegressor object mode of Numba
# -----------------------------------------------------------------------------
# Typing and overloads to use XGBRegressor inside Bodo functions
# directly via sklearn's API


# Tests are in engine-e2e-tests, so added #pragma: no cover

BodoXGBRegressorType, _ = install_py_obj_class(
    types_name="xgbregressor_type",
    python_type=xgboost.XGBRegressor,
    module=this_module,
    class_name="BodoXGBRegressorType",
    model_name="BodoXGBRegressorModel",
)  # pragma: no cover


@overload(xgboost.XGBRegressor, no_unliteral=True)
def sklearn_xgbregressor_overload(
    n_estimators=100,
    max_depth=None,
    learning_rate=None,
    verbosity=None,
    objective=None,
    booster=None,
    tree_method=None,
    n_jobs=None,
    gamma=None,
    min_child_weight=None,
    max_delta_step=None,
    subsample=None,
    colsample_bytree=None,
    colsample_bylevel=None,
    colsample_bynode=None,
    reg_alpha=None,
    reg_lambda=None,
    scale_pos_weight=None,
    base_score=None,
    random_state=None,
    missing=np.nan,
    num_parallel_tree=None,
    monotone_constraints=None,
    interaction_constraints=None,
    importance_type="gain",
    gpu_id=None,
    validate_parameters=None,
):  # pragma: no cover
    def _xgbregressor_impl(
        n_estimators=100,
        max_depth=None,
        learning_rate=None,
        verbosity=None,
        objective=None,
        booster=None,
        tree_method=None,
        n_jobs=None,
        gamma=None,
        min_child_weight=None,
        max_delta_step=None,
        subsample=None,
        colsample_bytree=None,
        colsample_bylevel=None,
        colsample_bynode=None,
        reg_alpha=None,
        reg_lambda=None,
        scale_pos_weight=None,
        base_score=None,
        random_state=None,
        missing=np.nan,
        num_parallel_tree=None,
        monotone_constraints=None,
        interaction_constraints=None,
        importance_type="gain",
        gpu_id=None,
        validate_parameters=None,
    ):  # pragma: no cover
        with bodo.objmode(m="xgbregressor_type"):
            xgboost.rabit.init()
            m = xgboost.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                verbosity=verbosity,
                objective=objective,
                booster=booster,
                tree_method=tree_method,
                n_jobs=n_jobs,
                gamma=gamma,
                min_child_weight=min_child_weight,
                max_delta_step=max_delta_step,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                colsample_bylevel=colsample_bylevel,
                colsample_bynode=colsample_bynode,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                scale_pos_weight=scale_pos_weight,
                base_score=base_score,
                random_state=random_state,
                missing=missing,
                num_parallel_tree=num_parallel_tree,
                monotone_constraints=monotone_constraints,
                interaction_constraints=interaction_constraints,
                importance_type=importance_type,
                gpu_id=gpu_id,
                validate_parameters=validate_parameters,
            )
        return m

    return _xgbregressor_impl


@overload_method(BodoXGBRegressorType, "fit", no_unliteral=True)
def overload_xgbregressor_fit(
    m,
    X,
    y,
    sample_weight=None,
    base_margin=None,
    eval_set=None,
    eval_metric=None,
    early_stopping_rounds=None,
    verbose=True,
    xgb_model=None,
    sample_weight_eval_set=None,
    feature_weights=None,
    callbacks=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):  # pragma: no cover
    """XGBRegressor fit overload"""

    def _xgbregressor_fit_impl(
        m,
        X,
        y,
        sample_weight=None,
        base_margin=None,
        eval_set=None,
        eval_metric=None,
        early_stopping_rounds=None,
        verbose=True,
        xgb_model=None,
        sample_weight_eval_set=None,
        feature_weights=None,
        callbacks=None,
        _is_data_distributed=False,
    ):  # pragma: no cover
        with bodo.objmode():
            m.n_jobs = 1
            m.fit(
                X,
                y,
                sample_weight,
                base_margin,
                eval_set,
                eval_metric,
                early_stopping_rounds,
                verbose,
                xgb_model,
                sample_weight_eval_set,
                feature_weights,
                callbacks,
            )
        return m

    return _xgbregressor_fit_impl


@overload_method(BodoXGBRegressorType, "predict", no_unliteral=True)
def overload_xgbregressor_predict(
    m,
    X,
    output_margin=False,
    ntree_limit=None,
    validate_features=True,
    base_margin=None,
):  # pragma: no cover
    """Overload XGBRegressor predict."""

    def _model_predict_impl(
        m,
        X,
        output_margin=False,
        ntree_limit=None,
        validate_features=True,
        base_margin=None,
    ):  # pragma: no cover
        with bodo.objmode(result="float64[:]"):
            m.n_jobs = 1
            result = (
                m.predict(X, output_margin, ntree_limit, validate_features, base_margin)
                .astype(np.float64)
                .flatten()
            )
        return result

    return _model_predict_impl


@overload_attribute(BodoXGBClassifierType, "feature_importances_")
@overload_attribute(BodoXGBRegressorType, "feature_importances_")
def get_xgb_feature_importances(m):
    """Overload feature_importances_ attribute to be accessible inside bodo.jit"""

    def impl(m):  # pragma: no cover
        with bodo.objmode(result="float32[:]"):
            result = m.feature_importances_
        return result

    return impl
