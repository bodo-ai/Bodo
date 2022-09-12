# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Tests for pd.CategoricalDtype/pd.Categorical  functionality
"""
import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func, gen_nonascii_list
from bodo.utils.typing import BodoError


@pytest.mark.slow
@pytest.mark.parametrize(
    "dtype",
    [
        # using "string[pyarrow]" type to match Bodo output type for string arrays
        pd.CategoricalDtype(pd.array(["AA", "B", "CC"], "string[pyarrow]")),
        pytest.param(
            pd.CategoricalDtype(pd.array(["CC", "AA", "B"], "string[pyarrow]"), True),
            marks=pytest.mark.slow,
        ),
        pytest.param(pd.CategoricalDtype([3, 2, 1, 4]), marks=pytest.mark.slow),
        pytest.param(
            pd.CategoricalDtype(pd.array(gen_nonascii_list(4), "string[pyarrow]")),
            marks=pytest.mark.slow,
        ),
    ],
)
def test_unbox_dtype(dtype, memory_leak_check):
    # just unbox
    def impl(dtype):
        return True

    check_func(impl, (dtype,))

    # unbox and box
    def impl2(dtype):
        return dtype

    check_func(impl2, (dtype,))

    # make sure proper Index type is used for categories and get_loc is supported
    def impl3(dtype):
        return dtype.categories.get_loc(dtype.categories[0])

    check_func(impl3, (dtype,))


@pytest.fixture(
    params=[
        pd.Categorical(["CC", "AA", "B", "D", "AA", None, "B", "CC"]),
        pytest.param(
            pd.Categorical(["CC", "AA", None, "B", "D", "AA", "B", "CC"], ordered=True),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Categorical(gen_nonascii_list(8)),
            marks=pytest.mark.slow,
        ),
        pytest.param(pd.Categorical([3, 1, 2, -1, 4, 1, 3, 2]), marks=pytest.mark.slow),
        pytest.param(
            pd.Categorical([3, 1, 2, -1, 4, 1, 3, 2], ordered=True),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Categorical(
                np.array(
                    ["2020-01-14", "2020-01-15", "2020-01-16", "2020-01-17", "NAT"],
                    dtype="datetime64[ns]",
                )
            )
        ),
        pytest.param(
            pd.Categorical(
                np.append(
                    pd.timedelta_range(start="1 day", periods=4),
                    [np.timedelta64("NaT")],
                )
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Categorical([3, 1.321, 0.0122, -1.321, 0.0, 1, 3, 2]),
            marks=pytest.mark.slow,
        ),
    ]
)
def cat_arr_value(request):
    return request.param


@pytest.mark.slow
def test_unbox_cat_arr(cat_arr_value, memory_leak_check):
    # just unbox
    def impl(arr):
        return True

    check_func(impl, (cat_arr_value,))

    # unbox and box
    def impl2(arr):
        return arr

    check_func(impl2, (cat_arr_value,))


@pytest.mark.smoke
def test_setitem_scalar(cat_arr_value, memory_leak_check):
    """
    Tests setitem for Categorical Arrays on scalar values.
    """

    def test_impl(A, i, val):
        A[i] = val
        return A

    val = cat_arr_value[0]
    np.random.seed(1)
    for ind in [1, [0, 1, 4], np.random.ranf(len(cat_arr_value)) < 0.2, slice(1, 4)]:
        check_func(
            test_impl, (cat_arr_value, ind, val), dist_test=False, copy_input=True
        )


def test_setitem_cat_array_compile_time(cat_arr_value, memory_leak_check):
    """
    Tests setitem for Categorical Arrays with Categorical
    Array.
    """

    def test_impl(A, i, val):
        A[i] = val
        return A

    val = cat_arr_value[:3]
    for idx in [
        [0, 1, 4],
        np.array([True, True] + ([False] * (len(cat_arr_value) - 3)) + [True]),
        slice(1, 4),
    ]:
        check_func(
            test_impl, (cat_arr_value, idx, val), dist_test=False, copy_input=True
        )


# TODO: Add memory leak check when constant lowering memory leak is fixed.
@pytest.mark.slow
def test_setitem_cat_array_compile_time_err(cat_arr_value):
    """
    Tests setitem err for Categorical Arrays with Categorical
    Array known at compile time.
    """

    def test_impl(A, i, val):
        A[i] = val
        return A

    val = pd.Series(cat_arr_value.categories)[:3].astype("category").values
    for idx in [
        [0, 1, 4],
        np.array([True, True] + ([False] * (len(cat_arr_value) - 3)) + [True]),
        slice(1, 4),
    ]:
        with pytest.raises(
            BodoError,
            match="Cannot set a Categorical with another, without identical categories",
        ):
            bodo.jit(test_impl)(cat_arr_value, idx, val)


# TODO: Add memory leak check when constant lowering memory leak is fixed.
@pytest.mark.slow
def test_setitem_cat_array_runtime(cat_arr_value):
    """
    Tests setitem for Categorical Arrays with Categorical
    Array not known at compile time.
    """

    def test_impl(A, i, val):
        A[i] = val.astype("category").values[:3]
        return A

    # Set ordered to false so the dtypes at compile/runtime match
    cat_arr_value = cat_arr_value.astype(
        pd.CategoricalDtype(cat_arr_value.dtype.categories, ordered=False)
    )

    val = pd.Series(cat_arr_value.categories)
    for idx in [
        [0, 1, 4],
        np.array([True, True] + ([False] * (len(cat_arr_value) - 3)) + [True]),
        slice(1, 4),
    ]:
        check_func(
            test_impl, (cat_arr_value, idx, val), dist_test=False, copy_input=True
        )


# TODO: Add memory leak check when constant lowering memory leak is fixed.
@pytest.mark.slow
def test_setitem_cat_array_runtime_err(cat_arr_value):
    """
    Tests setitem err for Categorical Arrays with Categorical
    Array not known at compile time.
    """

    def test_impl(A, i, val):
        A[i] = val.astype("category").values
        return A

    # Set ordered to false so the dtypes at compile time match
    cat_arr_value = cat_arr_value.astype(
        pd.CategoricalDtype(cat_arr_value.dtype.categories, ordered=False)
    )

    val = pd.Series(cat_arr_value.categories)[:3]
    for idx in [
        [0, 1, 4],
        np.array([True, True] + ([False] * (len(cat_arr_value) - 3)) + [True]),
        slice(1, 4),
    ]:
        with pytest.raises(
            ValueError,
            match="Cannot set a Categorical with another, without identical categories",
        ):
            bodo.jit(test_impl)(cat_arr_value, idx, val)


@pytest.mark.slow
def test_setitem_categories(cat_arr_value, memory_leak_check):
    """
    Tests setitem for Categorical Arrays with a list/array
    of values that match the categories.
    """

    def test_impl(A, i, val):
        A[i] = val
        return A

    val = cat_arr_value.categories.values[:3]
    for idx in [
        [0, 1, 4],
        np.array([True, True] + ([False] * (len(cat_arr_value) - 3)) + [True]),
        slice(1, 4),
    ]:
        check_func(
            test_impl, (cat_arr_value, idx, val), dist_test=False, copy_input=True
        )


@pytest.mark.smoke
def test_getitem_int(cat_arr_value, memory_leak_check):
    def test_impl(A, i):
        return A[i]

    i = 1
    check_func(test_impl, (cat_arr_value, i), dist_test=False)


def test_getitem_bool(cat_arr_value, memory_leak_check):
    def test_impl(A, ind):
        return A[ind]

    np.random.seed(1)
    ind = np.random.ranf(len(cat_arr_value)) < 0.2
    check_func(test_impl, (cat_arr_value, ind), dist_test=False)


@pytest.mark.slow
def test_getitem_slice(cat_arr_value, memory_leak_check):
    def test_impl(A, ind):
        return A[ind]

    ind = slice(1, 4)
    check_func(test_impl, (cat_arr_value, ind), dist_test=False)


@pytest.mark.slow
def test_cmp(memory_leak_check):
    """test eq/ne comparison of Categorical array and value"""
    # literal value
    def impl1(A):
        return A == 1

    # non-literal value
    def impl2(A, a):
        return A == a

    # non-literal value
    def impl3(A, a):
        return A != a

    A = pd.Categorical([3, 1, 2, -1, 4, 1, 3, 2, None, 7, 8, 12] * 10)
    check_func(impl1, (A,))
    check_func(impl2, (A, 2))
    check_func(impl3, (A, 2))


@pytest.mark.slow
def test_pd_categorical(memory_leak_check):
    """test pd.Categorical() constructor"""

    # dtype provided
    def impl1(A, dtype):
        return pd.Categorical(A, dtype=dtype)

    # categories provided, unordered
    def impl2(A, cats):
        return pd.Categorical(A, categories=cats)

    # categories provided, ordered
    def impl3(A, cats):
        return pd.Categorical(A, categories=cats, ordered=True)

    # no extra argument
    def impl4(A):
        return pd.Categorical(A)

    A = np.array([3, 1, 2, -1, 4, 1, 3, 2, 3, 7, 8, 12] * 10)
    check_func(impl1, (A, pd.CategoricalDtype([3, 1, 2, -1, 4, 12])))
    check_func(impl2, (A, [3, 1, 2, -1, 4, 12]))
    check_func(impl3, (A, [3, 1, 2, -1, 4, 12]))
    check_func(impl4, (A,))
    check_func(impl4, (pd.array(A, "Int64"),), check_dtype=False)
    check_func(impl4, (pd.array(np.abs(A), "UInt64"),), check_dtype=False)


@pytest.mark.slow
def test_astype(memory_leak_check):
    """test astype for categorical array, which allows going back to original values"""

    # int value
    def impl1(A):
        return A.astype(np.int64)

    # string value
    def impl2(A):
        return A.astype(str)

    A = pd.Categorical([3, 1, 2, -1, 4, 1, 3, 2, 3, 7, 8, 12] * 10)
    check_func(impl1, (A,))
    A = pd.Categorical(["CC", "AA", "B", "D", "AA", "B", "CC"])
    check_func(impl2, (A,))
    A = pd.Categorical(gen_nonascii_list(7))
    check_func(impl2, (A,))


@pytest.mark.slow
def test_pd_get_dummies(cat_arr_value, memory_leak_check):
    def test_impl(A):
        return pd.get_dummies(A)

    check_func(
        test_impl,
        (cat_arr_value,),
        check_categorical=False,
    )


@pytest.mark.slow
def test_pd_get_dummies_series(cat_arr_value, memory_leak_check):
    def test_impl(S):
        return pd.get_dummies(S)

    S = pd.Series(cat_arr_value)
    check_func(test_impl, (S,), check_categorical=False)


# TODO(ehsan): add memory_leak_check when leaks in the literal case are resolved
@pytest.mark.slow
def test_replace():
    def test_impl(A, to_replace, value):
        return A.replace(to_replace, value)

    A = pd.Categorical(["CC", "AA", "B", "D", "AA", None, "B", "CC"])
    to_replace = "CC"
    value = "ZZZZ"
    check_func(test_impl, (A, to_replace, value))
    A = pd.Categorical([3, 1, 2, -1, 4, 1, 3, 2, 7, 8, 12] * 10, ordered=True)
    to_replace = 2
    value = 5
    check_func(test_impl, (A, to_replace, value))
    A = pd.Categorical(["CC", "AA", "≥", "≡", "≠", None, "B", "≤"])
    to_replace = "≤"
    value = "µ"
    check_func(test_impl, (A, to_replace, value))


@pytest.mark.slow
def test_replace_list(memory_leak_check):
    def test_impl(A, to_replace, value):
        return A.replace(to_replace, value)

    A = pd.Categorical(["CC", "AA", "B", "D", "AA", None, "B", "CC"])
    to_replace = ["CC", "AA"]
    value = "ZZZZ"
    check_func(test_impl, (A, to_replace, value))
    A = pd.Categorical([3, 1, 2, -1, 4, 1, 3, 2, 7, 8, 12] * 10, ordered=True)
    to_replace = [2, 3, 7]
    value = 5
    check_func(test_impl, (A, to_replace, value), dist_test=False)


# Readd memory_leak_check when lowering memory leak is handled
@pytest.mark.slow
def test_replace_const_string():
    def test_impl(A):
        return A.replace("CC", "ZZZZ")

    A = pd.Categorical(["CC", "AA", "B", "D", "AA", None, "B", "CC"])
    check_func(test_impl, (A,))


# Readd memory_leak_check when lowering memory leak is handled
def test_replace_const():
    def test_impl(A):
        return A.replace(2, 5)

    A = pd.Categorical([3, 1, 2, -1, 4, 1, 3, 2, 7, 8, 12], ordered=True)
    check_func(test_impl, (A,))


# Readd memory_leak_check when lowering memory leak is handled
@pytest.mark.slow
def test_replace_const_list():
    def test_impl(A):
        return A.replace([2, 3, 7], 5)

    A = pd.Categorical([3, 1, 2, -1, 4, 1, 3, 2, 7, 8, 12], ordered=True)
    check_func(test_impl, (A,))


# TODO(ehsan): add memory_leak_check when leaks in the literal case are resolved
@pytest.mark.slow
def test_replace_delete():
    def test_impl(A, to_replace, value):
        return A.replace(to_replace, value)

    A = pd.Categorical(["CC", "AA", "B", "D", "AA", None, "B", "CC"])
    to_replace = "CC"
    value = "D"
    check_func(test_impl, (A, to_replace, value))
    A = pd.Categorical([3, 1, 2, -1, 4, 1, 3, 2, 7, 8, 12], ordered=True)
    to_replace = 2
    value = 1
    check_func(test_impl, (A, to_replace, value))
    A = pd.Categorical(["CC", "AA", None, "≥", "≡", "≠", None, "B", "≤", None])
    to_replace = "≤"
    value = "µ"
    check_func(test_impl, (A, to_replace, value))


@pytest.mark.slow
def test_replace_delete_list(memory_leak_check):
    def test_impl(A, to_replace, value):
        return A.replace(to_replace, value)

    A = pd.Categorical(["CC", "AA", "B", "D", "AA", None, "B", "CC"])
    to_replace = ["CC", "AA"]
    value = "D"
    check_func(test_impl, (A, to_replace, value))
    A = pd.Categorical([3, 1, 2, -1, 4, 1, 3, 2, 7, 8, 12], ordered=True)
    to_replace = [2, 3]
    value = 1
    check_func(test_impl, (A, to_replace, value))


# Readd memory_leak_check when lowering memory leak is handled
@pytest.mark.slow
def test_replace_delete_const():
    def test_impl(A):
        return A.replace(2, 1)

    A = pd.Categorical([3, 1, 2, -1, 4, 1, 3, 2, 7, 8, 12], ordered=True)
    check_func(test_impl, (A,))


@pytest.mark.slow
def test_replace_same(memory_leak_check):
    def test_impl(A, to_replace, value):
        return A.replace(to_replace, value)

    A = pd.Categorical(["CC", "AA", "B", "D", "AA", None, "B", "CC"])
    to_replace = "CC"
    value = "CC"
    check_func(test_impl, (A, to_replace, value))
    A = pd.Categorical([3, 1, 2, -1, 4, 1, 3, 2, 7, 8, 12] * 10, ordered=True)
    to_replace = 2
    value = 2
    check_func(test_impl, (A, to_replace, value))


# Readd memory_leak_check when lowering memory leak is handled
@pytest.mark.slow
def test_replace_same_const():
    def test_impl(A):
        return A.replace("CC", "CC")

    A = pd.Categorical(["CC", "AA", "B", "D", "AA", None, "B", "CC"])
    check_func(test_impl, (A,))


@pytest.mark.slow
def test_replace_missing(memory_leak_check):
    def test_impl(A, to_replace, value):
        return A.replace(to_replace, value)

    A = pd.Categorical(["CC", "AA", "B", "D", "AA", None, "B", "CC"])
    to_replace = "ZZ"
    value = "CC"
    check_func(test_impl, (A, to_replace, value))
    A = pd.Categorical([3, 1, 2, -1, 4, 1, 3, 2, 7, 8, 12] * 10, ordered=True)
    to_replace = 5
    value = 2
    check_func(test_impl, (A, to_replace, value))


# Readd memory_leak_check when lowering memory leak is handled
@pytest.mark.slow
def test_replace_missing_const():
    def test_impl(A):
        return A.replace("ZZ", "CC")

    A = pd.Categorical(["CC", "AA", "B", "D", "AA", None, "B", "CC"])
    check_func(test_impl, (A,))


def test_pd_categorical_compile_time():
    """Checks that pd.Categorical exposes categories at compile time
    when the categories are known."""

    def test_impl1(S):
        cat_S = pd.Categorical(S, categories=["A", "B", "C"], ordered=False)
        return pd.get_dummies(cat_S)

    def test_impl2(S):
        cat_S = pd.Categorical(S, categories=["A", "B", "C"], ordered=True)
        return pd.get_dummies(cat_S)

    def test_impl3(S):
        cat_S = pd.Categorical(S, categories=["A", "B", "C"])
        return pd.get_dummies(cat_S)

    S = pd.Series(["A", None, "B", "C", "B", "C", "A"])
    check_func(test_impl1, (S,), check_categorical=False)
    check_func(test_impl2, (S,), check_categorical=False)
    check_func(test_impl3, (S,), check_categorical=False)


@pytest.mark.slow
def test_constant_lowering(memory_leak_check):
    arr = pd.Categorical(["A", "B", "A", "ABC", "B", None, "ABC", "D"])

    def impl():
        return arr

    check_func(impl, (), check_categorical=False, only_seq=True)


@pytest.mark.slow
def test_categorical_nbytes(memory_leak_check):
    """Test CategoricalArrayType nbytes"""

    def impl(A):
        return A.nbytes

    A = pd.Categorical([1, 2, 5, None, 2] * 2, ordered=True)
    py_out = 10 + 25 * bodo.get_size()  # A.dtype is replicated on ranks
    check_func(impl, (A,), py_output=py_out, only_1D=True)
    check_func(impl, (A,), py_output=35, only_seq=True)
