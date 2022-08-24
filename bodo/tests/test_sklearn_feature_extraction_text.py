# Copyright (C) 2022 Bodo Inc. All rights reserved.
""" Test supported sklearn.feature_extraction.text models"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.utils._testing import assert_almost_equal

import bodo
from bodo.tests.utils import _get_dist_arg, check_func

# ---------------------- CountVectorizer tests ----------------------


def test_count_vectorizer(memory_leak_check):
    """Test CountVectorizer's vocabulary and fit_transform"""

    cat_in_the_hat_docs = [
        "One Cent, Two Cents, Old Cent, New Cent: All About Money (Cat in the Hat's Learning Library",
        "Inside Your Outside: All About the Human Body (Cat in the Hat's Learning Library)",
        "Oh, The Things You Can Do That Are Good for You: All About Staying Healthy (Cat in the Hat's Learning Library)",
        "On Beyond Bugs: All About Insects (Cat in the Hat's Learning Library)",
        "There's No Place Like Space: All About Our Solar System (Cat in the Hat's Learning Library)",
    ]
    df = pd.DataFrame({"A": cat_in_the_hat_docs})

    def impl(df):
        v = CountVectorizer()
        v.fit_transform(df["A"])
        ans = v.get_feature_names_out()
        return ans

    check_func(impl, (df,), is_out_distributed=False)

    # Test vocabulary_ and stop_words
    def impl2(df):
        v = CountVectorizer(stop_words="english")
        v.fit_transform(df["A"])
        return sorted(v.vocabulary_)

    check_func(impl2, (df,), is_out_distributed=False)

    # Test user-defined vocabulary
    # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/feature_extraction/tests/test_text.py#L315
    def impl3(docs, vocab):
        v = CountVectorizer(vocabulary=vocab)
        ans = v.fit_transform(docs)
        return ans

    JUNK_FOOD_DOCS = (
        "the pizza pizza beer copyright",
        "the pizza burger beer copyright",
        "the the pizza beer beer copyright",
        "the burger beer beer copyright",
        "the coke burger coke copyright",
        "the coke burger burger",
    )
    vocab = {"pizza": 0, "beer": 1}
    result = bodo.jit(
        impl3,
        all_args_distributed_block=True,
        all_returns_distributed=True,
    )(_get_dist_arg(np.array(JUNK_FOOD_DOCS), False), vocab)
    X = bodo.allgatherv(result)
    terms = set(vocab.keys())
    assert X.shape[1] == len(terms)
    # assert same values in both sklearn and Bodo
    X_sk = impl3(JUNK_FOOD_DOCS, vocab)
    assert np.array_equal(X.todense(), X_sk.todense())

    # Test replicated
    check_func(impl3, (JUNK_FOOD_DOCS, vocab), only_seq=True)


def test_hashing_vectorizer(memory_leak_check):
    """Test HashingVectorizer's fit_transform method.
    Taken from here (https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/feature_extraction/tests/test_text.py#L573)
    """

    JUNK_FOOD_DOCS = (
        "the pizza pizza beer copyright",
        "the pizza burger beer copyright",
        "the the pizza beer beer copyright",
        "the burger beer beer copyright",
        "the coke burger coke copyright",
        "the coke burger burger",
    )

    NOTJUNK_FOOD_DOCS = (
        "the salad celeri copyright",
        "the salad salad sparkling water copyright",
        "the the celeri celeri copyright",
        "the tomato tomato salad water",
        "the tomato salad water copyright",
    )

    ALL_FOOD_DOCS = JUNK_FOOD_DOCS + NOTJUNK_FOOD_DOCS

    def test_fit_transform(X):
        v = HashingVectorizer()
        X_transformed = v.fit_transform(X)
        return X_transformed

    result = bodo.jit(
        test_fit_transform,
        all_args_distributed_block=True,
        all_returns_distributed=True,
    )(_get_dist_arg(np.array(ALL_FOOD_DOCS), False))
    result = bodo.allgatherv(result)
    token_nnz = result.nnz
    assert result.shape == (len(ALL_FOOD_DOCS), (2**20))
    # By default the hashed values receive a random sign and l2 normalization
    # makes the feature values bounded
    assert np.min(result.data) > -1
    assert np.min(result.data) < 0
    assert np.max(result.data) > 0
    assert np.max(result.data) < 1
    # Check that the rows are normalized
    for i in range(result.shape[0]):
        assert_almost_equal(np.linalg.norm(result[0].data, 2), 1.0)

    check_func(test_fit_transform, (np.array(ALL_FOOD_DOCS),))

    # Check vectorization with some non-default parameters
    def test_fit_transform_args(X):
        v = HashingVectorizer(ngram_range=(1, 2), norm="l1")
        ans = v.fit_transform(X)
        return ans

    X = bodo.jit(distributed=["X", "ans"])(test_fit_transform_args)(
        _get_dist_arg(np.array(ALL_FOOD_DOCS))
    )
    X = bodo.allgatherv(X)
    assert X.shape == (len(ALL_FOOD_DOCS), (2**20))

    # ngrams generate more non zeros
    ngrams_nnz = X.nnz
    assert ngrams_nnz > token_nnz
    assert ngrams_nnz < 2 * token_nnz

    # makes the feature values bounded
    assert np.min(X.data) > -1
    assert np.max(X.data) < 1

    # Check that the rows are normalized
    for i in range(X.shape[0]):
        assert_almost_equal(np.linalg.norm(X[0].data, 1), 1.0)
