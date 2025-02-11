"""Support scikit-learn Feature extraction methods"""

import sys

import numpy as np
import sklearn.feature_extraction
from numba.core import types
from numba.extending import (
    overload,
    overload_attribute,
    overload_method,
)

import bodo
from bodo.libs.csr_matrix_ext import CSRMatrixType
from bodo.ml_support.sklearn_ext import check_sklearn_version
from bodo.utils.py_objs import install_py_obj_class
from bodo.utils.typing import (
    BodoError,
    get_overload_const,
    is_overload_constant_number,
    is_overload_true,
)

this_module = sys.modules[__name__]
# ----------------------------------------------------------------------------------------
# ----------------------------------- HashingVectorizer------------------------------------
# Support for sklearn.feature_extraction.text.HashingVectorizer
# Currently only fit_transform function is supported.
# We use sklearn's fit_transform directly in objmode on each rank.
# ----------------------------------------------------------------------------------------


# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoFExtractHashingVectorizerType, f_extract_hashing_vectorizer_type = (
    install_py_obj_class(
        types_name="f_extract_hashing_vectorizer_type",
        python_type=sklearn.feature_extraction.text.HashingVectorizer,
        module=this_module,
        class_name="BodoFExtractHashingVectorizerType",
        model_name="BodoFExtractHashingVectorizerModel",
    )
)


@overload(sklearn.feature_extraction.text.HashingVectorizer, no_unliteral=True)
def sklearn_hashing_vectorizer_overload(
    input="content",
    encoding="utf-8",
    decode_error="strict",
    strip_accents=None,
    lowercase=True,
    preprocessor=None,
    tokenizer=None,
    stop_words=None,
    token_pattern=r"(?u)\b\w\w+\b",
    ngram_range=(1, 1),
    analyzer="word",
    n_features=(2**20),
    binary=False,
    norm="l2",
    alternate_sign=True,
    dtype=np.float64,
):
    """
    Provide implementation for __init__ functions of HashingVectorizer.
    We simply call sklearn in objmode.
    """

    check_sklearn_version()

    def _sklearn_hashing_vectorizer_impl(
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        analyzer="word",
        n_features=(2**20),
        binary=False,
        norm="l2",
        alternate_sign=True,
        dtype=np.float64,
    ):  # pragma: no cover
        with bodo.objmode(m="f_extract_hashing_vectorizer_type"):
            m = sklearn.feature_extraction.text.HashingVectorizer(
                input=input,
                encoding=encoding,
                decode_error=decode_error,
                strip_accents=strip_accents,
                lowercase=lowercase,
                preprocessor=preprocessor,
                tokenizer=tokenizer,
                stop_words=stop_words,
                token_pattern=token_pattern,
                ngram_range=ngram_range,
                analyzer=analyzer,
                n_features=n_features,
                binary=binary,
                norm=norm,
                alternate_sign=alternate_sign,
                dtype=dtype,
            )
        return m

    return _sklearn_hashing_vectorizer_impl


@overload_method(BodoFExtractHashingVectorizerType, "fit_transform", no_unliteral=True)
def overload_hashing_vectorizer_fit_transform(
    m,
    X,
    y=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """
    Provide implementation for the fit_transform function.
    We simply call sklearn's fit_transform on each rank.
    """
    types.csr_matrix_float64_int64 = CSRMatrixType(types.float64, types.int64)

    def _hashing_vectorizer_fit_transform_impl(
        m,
        X,
        y=None,
        _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
    ):  # pragma: no cover
        with bodo.objmode(transformed_X="csr_matrix_float64_int64"):
            transformed_X = m.fit_transform(X, y)
            transformed_X.indices = transformed_X.indices.astype(np.int64)
            transformed_X.indptr = transformed_X.indptr.astype(np.int64)
        return transformed_X

    return _hashing_vectorizer_fit_transform_impl


# ----------------------------------------------------------------------------------------
# ----------------------------------- CountVectorizer------------------------------------
# Support for sklearn.feature_extraction.text.CountVectorizer
# Currently fit_transform & get_feature_names_out functions are supported.
# ----------------------------------------------------------------------------------------


# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoFExtractCountVectorizerType, _ = install_py_obj_class(
    types_name="f_extract_count_vectorizer_type",
    python_type=sklearn.feature_extraction.text.CountVectorizer,
    module=this_module,
    class_name="BodoFExtractCountVectorizerType",
    model_name="BodoFExtractCountVectorizerModel",
)


@overload(sklearn.feature_extraction.text.CountVectorizer, no_unliteral=True)
def sklearn_count_vectorizer_overload(
    input="content",
    encoding="utf-8",
    decode_error="strict",
    strip_accents=None,
    lowercase=True,
    preprocessor=None,
    tokenizer=None,
    stop_words=None,
    token_pattern=r"(?u)\b\w\w+\b",
    ngram_range=(1, 1),
    analyzer="word",
    max_df=1.0,
    min_df=1,
    max_features=None,
    vocabulary=None,
    binary=False,
    dtype=np.int64,
):
    """
    Provide implementation for __init__ functions of CountVectorizer.
    We simply call sklearn in objmode.
    """

    check_sklearn_version()

    # Per sklearn documentation, min_df: ignore terms that have a document
    # frequency strictly lower than the given threshold.

    if not is_overload_constant_number(min_df) or get_overload_const(min_df) != 1:
        raise BodoError(
            "sklearn.feature_extraction.text.CountVectorizer(): 'min_df' is not supported for distributed data.\n"
        )

    # Per sklearn documentation, max_df: ignore terms that have a document
    # frequency strictly higher than the given threshold.
    if not is_overload_constant_number(max_df) or get_overload_const(min_df) != 1:
        raise BodoError(
            "sklearn.feature_extraction.text.CountVectorizer(): 'max_df' is not supported for distributed data.\n"
        )

    def _sklearn_count_vectorizer_impl(
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        analyzer="word",
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.int64,
    ):  # pragma: no cover
        with bodo.objmode(m="f_extract_count_vectorizer_type"):
            m = sklearn.feature_extraction.text.CountVectorizer(
                input=input,
                encoding=encoding,
                decode_error=decode_error,
                strip_accents=strip_accents,
                lowercase=lowercase,
                preprocessor=preprocessor,
                tokenizer=tokenizer,
                stop_words=stop_words,
                token_pattern=token_pattern,
                ngram_range=ngram_range,
                analyzer=analyzer,
                max_df=max_df,
                min_df=min_df,
                max_features=max_features,
                vocabulary=vocabulary,
                binary=binary,
                dtype=dtype,
            )
        return m

    return _sklearn_count_vectorizer_impl


@overload_attribute(BodoFExtractCountVectorizerType, "vocabulary_")
def get_cv_vocabulary_(m):
    """Overload vocabulary_ attribute to be accessible inside bodo.jit"""

    types.dict_string_int = types.DictType(types.unicode_type, types.int64)

    def impl(m):  # pragma: no cover
        with bodo.objmode(result="dict_string_int"):
            result = m.vocabulary_
        return result

    return impl


def _cv_fit_transform_helper(m, X):
    """Initial fit computation to get vocabulary if user didn't provide it"""
    change_voc = False
    local_vocabulary = m.vocabulary
    if m.vocabulary is None:
        m.fit(X)
        local_vocabulary = m.vocabulary_
        change_voc = True
    return change_voc, local_vocabulary


@overload_method(BodoFExtractCountVectorizerType, "fit_transform", no_unliteral=True)
def overload_count_vectorizer_fit_transform(
    m,
    X,
    y=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """
    Provide implementation for the fit_transform function.
    If distributed, run fit to get vocabulary on each rank locally and gather it.
    Then, run fit_transform with combined vocabulary
    If replicated, simply call fit_transform on each rank.
    """
    check_sklearn_version()
    types.csr_matrix_int64_int64 = CSRMatrixType(types.int64, types.int64)
    if is_overload_true(_is_data_distributed):
        types.dict_str_int = types.DictType(types.unicode_type, types.int64)

        def _count_vectorizer_fit_transform_impl(
            m,
            X,
            y=None,
            _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
        ):  # pragma: no cover
            with bodo.objmode(local_vocabulary="dict_str_int", changeVoc="bool_"):
                changeVoc, local_vocabulary = _cv_fit_transform_helper(m, X)
            # Gather vocabulary from each rank and generate its integer indices (alphabetical order)
            if changeVoc:
                local_vocabulary = bodo.utils.conversion.coerce_to_array(
                    list(local_vocabulary.keys())
                )
                all_vocabulary = bodo.libs.array_kernels.unique(
                    local_vocabulary, parallel=True
                )
                all_vocabulary = bodo.allgatherv(all_vocabulary, False)
                all_vocabulary = bodo.libs.array_kernels.sort(
                    all_vocabulary, ascending=True, inplace=True
                )
                new_data = {}
                for i in range(len(all_vocabulary)):
                    new_data[all_vocabulary[i]] = i
            else:
                new_data = local_vocabulary
            # Run fit_transform with generated vocabulary_
            with bodo.objmode(transformed_X="csr_matrix_int64_int64"):
                if changeVoc:
                    m.vocabulary = new_data
                transformed_X = m.fit_transform(X, y)
                transformed_X.indices = transformed_X.indices.astype(np.int64)
                transformed_X.indptr = transformed_X.indptr.astype(np.int64)
            return transformed_X

        return _count_vectorizer_fit_transform_impl
    else:
        # If replicated, then just call sklearn
        def _count_vectorizer_fit_transform_impl(
            m,
            X,
            y=None,
            _is_data_distributed=False,
        ):  # pragma: no cover
            with bodo.objmode(transformed_X="csr_matrix_int64_int64"):
                transformed_X = m.fit_transform(X, y)
                transformed_X.indices = transformed_X.indices.astype(np.int64)
                transformed_X.indptr = transformed_X.indptr.astype(np.int64)
            return transformed_X

        return _count_vectorizer_fit_transform_impl


# NOTE: changed get_feature_names as it will be removed in 1.2
# and will be replaced by get_feature_names_out
@overload_method(
    BodoFExtractCountVectorizerType, "get_feature_names_out", no_unliteral=True
)
def overload_count_vectorizer_get_feature_names_out(m):
    """Array mapping from feature integer indices to feature name."""

    check_sklearn_version()

    def impl(m):  # pragma: no cover
        with bodo.objmode(result=bodo.string_array_type):
            result = m.get_feature_names_out()
        return result

    return impl
