"""Support scikit-learn preprocessing methods."""

import numbers
import sys

import numba
import numpy as np
import pandas as pd
import sklearn.metrics
from mpi4py import MPI
from numba.core import types
from numba.extending import (
    overload,
    overload_attribute,
    overload_method,
)
from scipy import stats  # noqa
from sklearn.preprocessing._data import (
    _handle_zeros_in_scale as sklearn_handle_zeros_in_scale,
)
from sklearn.utils._encode import _unique
from sklearn.utils.extmath import (
    _safe_accumulator_op as sklearn_safe_accumulator_op,
)

import bodo
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.libs.csr_matrix_ext import CSRMatrixType
from bodo.utils.py_objs import install_py_obj_class
from bodo.utils.typing import (
    BodoError,
    check_unsupported_args,
    is_overload_none,
    is_overload_true,
)

this_module = sys.modules[__name__]
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
BodoPreprocessingOneHotEncoderType, _ = install_py_obj_class(
    types_name="preprocessing_one_hot_encoder_type",
    python_type=sklearn.preprocessing.OneHotEncoder,
    module=this_module,
    class_name="BodoPreprocessingOneHotEncoderType",
    model_name="BodoPreprocessingOneHotEncoderModel",
)


# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoPreprocessingOneHotEncoderCategoriesType, _ = install_py_obj_class(
    types_name="preprocessing_one_hot_encoder_categories_type",
    module=this_module,
    class_name="BodoPreprocessingOneHotEncoderCategoriesType",
    model_name="BodoPreprocessingOneHotEncoderCategoriesModel",
)


# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoPreprocessingOneHotEncoderDropIdxType, _ = install_py_obj_class(
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
        with numba.objmode(result="preprocessing_one_hot_encoder_categories_type"):
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
        with numba.objmode(result="preprocessing_one_hot_encoder_drop_idx_type"):
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
        with numba.objmode(result="int64"):
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
        with numba.objmode(m="preprocessing_one_hot_encoder_type"):
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
            ensure_all_finite="allow-nan",
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
    func_text += "    with numba.objmode(m='preprocessing_one_hot_encoder_type'):\n"
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
        with numba.objmode(transformed_X="float64[:,:]"):
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
        with numba.objmode(out_features="string[:]"):
            out_features = m.get_feature_names_out(input_features)
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
BodoPreprocessingStandardScalerType, _ = install_py_obj_class(
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

    def _sklearn_preprocessing_standard_scaler_impl(
        copy=True, with_mean=True, with_std=True
    ):  # pragma: no cover
        with numba.objmode(m="preprocessing_standard_scaler_type"):
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
            with numba.objmode(m="preprocessing_standard_scaler_type"):
                m = sklearn_preprocessing_standard_scaler_fit_dist_helper(m, X)

            return m

    else:
        # If replicated, then just call sklearn
        def _preprocessing_standard_scaler_fit_impl(
            m, X, y=None, sample_weight=None, _is_data_distributed=False
        ):  # pragma: no cover
            with numba.objmode(m="preprocessing_standard_scaler_type"):
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
            with numba.objmode(transformed_X="csr_matrix_float64_int64"):
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
            with numba.objmode(transformed_X="float64[:,:]"):
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
            with numba.objmode(inverse_transformed_X="csr_matrix_float64_int64"):
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
            with numba.objmode(inverse_transformed_X="float64[:,:]"):
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
BodoPreprocessingMaxAbsScalerType, _ = install_py_obj_class(
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
        with numba.objmode(result="float64[:]"):
            result = m.scale_
        return result

    return impl


@overload_attribute(BodoPreprocessingMaxAbsScalerType, "max_abs_")
def get_max_abs_scaler_max_abs_(m):
    """Overload max_abs_ attribute to be accessible inside bodo.jit"""

    def impl(m):  # pragma: no cover
        with numba.objmode(result="float64[:]"):
            result = m.max_abs_
        return result

    return impl


@overload_attribute(BodoPreprocessingMaxAbsScalerType, "n_samples_seen_")
def get_max_abs_scaler_n_samples_seen_(m):
    """Overload n_samples_seen_ attribute to be accessible inside bodo.jit"""

    def impl(m):  # pragma: no cover
        with numba.objmode(result="int64"):
            result = m.n_samples_seen_
        return result

    return impl


@overload(sklearn.preprocessing.MaxAbsScaler, no_unliteral=True)
def sklearn_preprocessing_max_abs_scaler_overload(copy=True):
    """
    Provide implementation for __init__ functions of MaxAbsScaler.
    We simply call sklearn in objmode.
    """

    def _sklearn_preprocessing_max_abs_scaler_impl(copy=True):  # pragma: no cover
        with numba.objmode(m="preprocessing_max_abs_scaler_type"):
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
            with numba.objmode(m="preprocessing_max_abs_scaler_type"):
                m = sklearn_preprocessing_max_abs_scaler_fit_dist_helper(
                    m, X, partial=False
                )
            return m

    else:
        # If replicated, then just call sklearn
        def _preprocessing_max_abs_scaler_fit_impl(
            m, X, y=None, _is_data_distributed=False
        ):  # pragma: no cover
            with numba.objmode(m="preprocessing_max_abs_scaler_type"):
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
            with numba.objmode(m="preprocessing_max_abs_scaler_type"):
                m = sklearn_preprocessing_max_abs_scaler_fit_dist_helper(
                    m, X, partial=True
                )
            return m

    else:
        # If replicated, then just call sklearn
        def _preprocessing_max_abs_scaler_partial_fit_impl(
            m, X, y=None, _is_data_distributed=False
        ):  # pragma: no cover
            with numba.objmode(m="preprocessing_max_abs_scaler_type"):
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
            with numba.objmode(transformed_X="csr_matrix_float64_int64"):
                transformed_X = m.transform(X)
                transformed_X.indices = transformed_X.indices.astype(np.int64)
                transformed_X.indptr = transformed_X.indptr.astype(np.int64)
            return transformed_X

    else:

        def _preprocessing_max_abs_scaler_transform_impl(
            m,
            X,
        ):  # pragma: no cover
            with numba.objmode(transformed_X="float64[:,:]"):
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
            with numba.objmode(inverse_transformed_X="csr_matrix_float64_int64"):
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
            with numba.objmode(inverse_transformed_X="float64[:,:]"):
                inverse_transformed_X = m.inverse_transform(X)
            return inverse_transformed_X

    return _preprocessing_max_abs_scaler_inverse_transform_impl


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
BodoPreprocessingMinMaxScalerType, _ = install_py_obj_class(
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

    def _sklearn_preprocessing_minmax_scaler_impl(
        feature_range=(0, 1),
        copy=True,
        clip=False,
    ):  # pragma: no cover
        with numba.objmode(m="preprocessing_minmax_scaler_type"):
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
        with numba.objmode(m="preprocessing_minmax_scaler_type"):
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
        with numba.objmode(transformed_X="float64[:,:]"):
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
        with numba.objmode(inverse_transformed_X="float64[:,:]"):
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
BodoPreprocessingRobustScalerType, _ = install_py_obj_class(
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
        with numba.objmode(result="boolean"):
            result = m.with_centering
        return result

    return impl


@overload_attribute(BodoPreprocessingRobustScalerType, "with_scaling")
def get_robust_scaler_with_scaling(m):
    """Overload with_scaling attribute to be accessible inside bodo.jit"""

    def impl(m):  # pragma: no cover
        with numba.objmode(result="boolean"):
            result = m.with_scaling
        return result

    return impl


@overload_attribute(BodoPreprocessingRobustScalerType, "quantile_range")
def get_robust_scaler_quantile_range(m):
    """Overload quantile_range attribute to be accessible inside bodo.jit"""

    typ = numba.typeof((25.0, 75.0))

    def impl(m):  # pragma: no cover
        with numba.objmode(result=typ):
            result = m.quantile_range
        return result

    return impl


@overload_attribute(BodoPreprocessingRobustScalerType, "unit_variance")
def get_robust_scaler_unit_variance(m):
    """Overload unit_variance attribute to be accessible inside bodo.jit"""

    def impl(m):  # pragma: no cover
        with numba.objmode(result="boolean"):
            result = m.unit_variance
        return result

    return impl


@overload_attribute(BodoPreprocessingRobustScalerType, "copy")
def get_robust_scaler_copy(m):
    """Overload copy attribute to be accessible inside bodo.jit"""

    def impl(m):  # pragma: no cover
        with numba.objmode(result="boolean"):
            result = m.copy
        return result

    return impl


@overload_attribute(BodoPreprocessingRobustScalerType, "center_")
def get_robust_scaler_center_(m):
    """Overload center_ attribute to be accessible inside bodo.jit"""

    def impl(m):  # pragma: no cover
        with numba.objmode(result="float64[:]"):
            result = m.center_
        return result

    return impl


@overload_attribute(BodoPreprocessingRobustScalerType, "scale_")
def get_robust_scaler_scale_(m):
    """Overload scale_ attribute to be accessible inside bodo.jit"""

    def impl(m):  # pragma: no cover
        with numba.objmode(result="float64[:]"):
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

    def _sklearn_preprocessing_robust_scaler_impl(
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
        copy=True,
        unit_variance=False,
    ):  # pragma: no cover
        with numba.objmode(m="preprocessing_robust_scaler_type"):
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

    # TODO Add general error-checking [BE-52]

    if is_overload_true(_is_data_distributed):
        # If distributed, then use native implementation

        func_text = "def preprocessing_robust_scaler_fit_impl(\n"
        func_text += "  m, X, y=None, _is_data_distributed=False\n"
        func_text += "):\n"

        # In case a DataFrame was provided, convert it to a Numpy Array first.
        # This is required since we'll be looping over the columns, which
        # is not supported for DataFrames.
        # TODO Add a compile time check that all columns are numeric since
        # `to_numpy` will error out otherwise anyway. [BE-52]
        if isinstance(X, DataFrameType):
            func_text += "  X = X.to_numpy()\n"

        func_text += "  with numba.objmode(qrange_l='float64', qrange_r='float64'):\n"
        func_text += "    (qrange_l, qrange_r) = m.quantile_range\n"
        func_text += "  if not 0 <= qrange_l <= qrange_r <= 100:\n"
        # scikit-learn throws the error: `"Invalid quantile range: %s" % str(self.quantile_range)`
        # but we cannot use format strings, so we use a slightly modified error message.
        func_text += "    raise ValueError(\n"
        func_text += "      'Invalid quantile range provided. Ensure that 0 <= quantile_range[0] <= quantile_range[1] <= 100.'\n"
        func_text += "    )\n"
        func_text += "  qrange_l, qrange_r = qrange_l / 100.0, qrange_r / 100.0\n"
        func_text += "  X = bodo.utils.conversion.coerce_to_array(X)\n"
        func_text += "  num_features = X.shape[1]\n"
        func_text += "  if m.with_scaling:\n"
        func_text += "    scales = np.zeros(num_features)\n"
        func_text += "  else:\n"
        func_text += "    scales = None\n"
        func_text += "  if m.with_centering:\n"
        func_text += "    centers = np.zeros(num_features)\n"
        func_text += "  else:\n"
        func_text += "    centers = None\n"
        func_text += "  if m.with_scaling or m.with_centering:\n"
        ## XXX Not sure if prange is useful here
        func_text += "    numba.parfors.parfor.init_prange()\n"
        func_text += "    for feature_idx in numba.parfors.parfor.internal_prange(num_features):\n"
        func_text += "      column_data = bodo.utils.conversion.ensure_contig_if_np(X[:, feature_idx])\n"
        func_text += "      if m.with_scaling:\n"
        func_text += "        q1 = bodo.libs.array_kernels.quantile_parallel(\n"
        func_text += "          column_data, qrange_l, 0\n"
        func_text += "        )\n"
        func_text += "        q2 = bodo.libs.array_kernels.quantile_parallel(\n"
        func_text += "          column_data, qrange_r, 0\n"
        func_text += "        )\n"
        func_text += "        scales[feature_idx] = q2 - q1\n"
        func_text += "      if m.with_centering:\n"
        func_text += (
            "        centers[feature_idx] = bodo.libs.array_ops.array_op_median(\n"
        )
        func_text += "          column_data, True, True\n"
        func_text += "        )\n"
        func_text += "  if m.with_scaling:\n"
        # Handle zeros (See sklearn.preprocessing._data._handle_zeros_in_scale)
        # RobustScaler.fit calls
        # `self.scale_ = _handle_zeros_in_scale(self.scale_, copy=False)`
        # which translates to:
        func_text += "    constant_mask = scales < 10 * np.finfo(scales.dtype).eps\n"
        func_text += "    scales[constant_mask] = 1.0\n"
        func_text += "    if m.unit_variance:\n"
        func_text += "      with numba.objmode(adjust='float64'):\n"
        func_text += (
            "        adjust = stats.norm.ppf(qrange_r) - stats.norm.ppf(qrange_l)\n"
        )
        func_text += "      scales = scales / adjust\n"
        func_text += "  with numba.objmode():\n"
        func_text += "    m.center_ = centers\n"
        func_text += "    m.scale_ = scales\n"
        func_text += "  return m\n"

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
            with numba.objmode(m="preprocessing_robust_scaler_type"):
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

    def _preprocessing_robust_scaler_transform_impl(
        m,
        X,
    ):  # pragma: no cover
        with numba.objmode(transformed_X="float64[:,:]"):
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

    def _preprocessing_robust_scaler_inverse_transform_impl(
        m,
        X,
    ):  # pragma: no cover
        with numba.objmode(inverse_transformed_X="float64[:,:]"):
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


def _pa_arr_to_numpy(a):
    """Convert Arrow arrays to Numpy arrays to workaround Scikit-learn issues
    as of 1.7.0. See test_label_encoder.
    """
    if isinstance(a, (pd.arrays.ArrowStringArray, pd.arrays.ArrowExtensionArray)):
        return a.to_numpy()
    return a


# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoPreprocessingLabelEncoderType, _ = install_py_obj_class(
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

    def _sklearn_preprocessing_label_encoder_impl():  # pragma: no cover
        with numba.objmode(m="preprocessing_label_encoder_type"):
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
            with numba.objmode:
                y_classes_obj = _pa_arr_to_numpy(y_classes)
                m.classes_ = y_classes_obj

            return m

        return _sklearn_preprocessing_label_encoder_fit_impl

    else:
        # If replicated, then just call sklearn
        def _sklearn_preprocessing_label_encoder_fit_impl(
            m, y, _is_data_distributed=False
        ):  # pragma: no cover
            with numba.objmode(m="preprocessing_label_encoder_type"):
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
        with numba.objmode(transformed_y="int64[:]"):
            transformed_y = m.transform(y).astype(np.int64)
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
            with numba.objmode(transformed_y="int64[:]"):
                transformed_y = m.fit_transform(y).astype(np.int64)
            return transformed_y

        return _preprocessing_label_encoder_fit_transform_impl
