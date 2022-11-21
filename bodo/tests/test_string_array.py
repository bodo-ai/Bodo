# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Test Bodo's string array data type
"""
import numba
import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import (
    SeqTestPipeline,
    check_func,
    dist_IR_contains,
    gen_nonascii_list,
)
from bodo.utils.typing import BodoError


@pytest.fixture(
    params=[
        # unicode
        pytest.param(
            pd.array(
                [
                    "¿abc¡Y tú, quién te crees?",
                    "ÕÕÕú¡úú,úũ¿ééé",
                    "россия очень, холодная страна",
                    pd.NA,
                    "مرحبا, العالم ، هذا هو بودو",
                    "Γειά σου ,Κόσμε",
                    "Español es agra,dable escuchar",
                    "한국,가,고싶다ㅠ",
                    "🢇🄐,🏈𠆶💑😅",
                ],
                dtype="string[pyarrow]",
            ),
            marks=pytest.mark.slow,
        ),
        # ASCII array
        pd.array(
            ["AB", "", "ABC", pd.NA, "C", "D", "abcd", "ABCD"], dtype="string[pyarrow]"
        ),
    ]
)
def str_arr_value(request):
    return request.param


@pytest.mark.slow
def test_np_sort(memory_leak_check):
    def impl(arr):
        return np.sort(arr)

    A = pd.array((["AB", "", "ABC", "abcd", "PQ", "DDE"] + gen_nonascii_list(2)) * 8)

    check_func(impl, (A,))


@pytest.mark.slow
def test_hash(str_arr_value, memory_leak_check):
    def impl(S):
        return S.map(lambda x: None if pd.isna(x) else hash(x))

    # check_dtype=False because None converts the output to Float in Pandas
    # dist_test = False because the randomness causes different inputs on each core.
    check_func(impl, (pd.Series(str_arr_value),), check_dtype=False, dist_test=False)


@pytest.mark.slow
def test_np_repeat(str_arr_value, memory_leak_check):
    def impl(arr):
        return np.repeat(arr, 2)

    check_func(impl, (str_arr_value,))


@pytest.mark.slow
def test_np_unique(memory_leak_check):
    def impl(arr):
        return np.unique(arr)

    # Create an array here because np.unique fails on NA in pandas
    arr = pd.array((["AB", "", "ABC", "abcd", "ab", "AB"] + gen_nonascii_list(2)))

    check_func(impl, (arr,), sort_output=True, is_out_distributed=False)


@pytest.mark.slow
def test_unbox(str_arr_value, memory_leak_check):
    # just unbox
    def impl(arr_arg):
        return True

    check_func(impl, (str_arr_value,))

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl2, (str_arr_value,))


@pytest.mark.slow
def test_constant_lowering(str_arr_value, memory_leak_check):
    def impl():
        return str_arr_value

    pd.testing.assert_series_equal(
        pd.Series(bodo.jit(impl)()), pd.Series(str_arr_value), check_dtype=False
    )


@pytest.mark.slow
def test_constant_lowering_refcount(memory_leak_check):
    """make sure refcount handling works for constant globals and destructor is not
    called leading to a segfault.
    """
    arr = np.array(
        (["AB", "", "ABC", None, "C", "D", "abcd", "ABCD"] + gen_nonascii_list(2))
    )

    @bodo.jit(distributed=False)
    def g(A):
        if len(A) > 30:
            print(len(A[0]))

    @bodo.jit(distributed=False)
    def f():
        g(arr)

    f()


@pytest.mark.slow
def test_string_dtype(memory_leak_check):
    # unbox and box
    def impl(d):
        return d

    check_func(impl, (pd.StringDtype(),))

    # constructor
    def impl2():
        return pd.StringDtype()

    check_func(impl2, ())


@pytest.mark.smoke
def test_getitem_int(str_arr_value, memory_leak_check):
    """
    Test operator.getitem on String array with a integer ind
    """

    def test_impl(A, i):
        return A[i]

    check_func(test_impl, (str_arr_value, 4))
    check_func(test_impl, (str_arr_value, -1))


@pytest.mark.slow
def test_getitem_int_arr(str_arr_value, memory_leak_check):
    """
    Test operator.getitem on String array with an integer list ind
    """

    def test_impl(A, ind):
        return A[ind]

    ind = np.array([0, 2, 3])
    # Pandas outputs differs (object vs string)
    # TODO [BE-483]: Fix distributed
    check_func(test_impl, (str_arr_value, ind), check_dtype=False, dist_test=False)
    check_func(
        test_impl,
        (str_arr_value, list(ind)),
        check_dtype=False,
        dist_test=False,
    )
    with pytest.raises(
        ValueError, match="Cannot index with an integer indexer containing NA values"
    ):
        ind = pd.array([0, pd.NA], "Int64")
        bodo.jit(test_impl)(str_arr_value, ind)


@pytest.mark.slow
def test_getitem_bool(str_arr_value, memory_leak_check):
    """
    Test operator.getitem on String array with a boolean ind
    """

    def test_impl(A, ind):
        return A[ind]

    np.random.seed(0)
    ind = np.random.ranf(len(str_arr_value)) < 0.5
    # Pandas outputs differs (object vs string)
    # TODO [BE-483]: Fix distributed
    check_func(test_impl, (str_arr_value, ind), check_dtype=False, dist_test=False)
    check_func(
        test_impl,
        (str_arr_value, list(ind)),
        check_dtype=False,
        dist_test=False,
    )
    # Check nullable with pd.array and insert NA values
    ind = pd.array(ind)
    ind[1] = None
    ind[3] = None
    check_func(test_impl, (str_arr_value, ind), check_dtype=False, dist_test=False)


@pytest.mark.slow
def test_getitem_slice(str_arr_value, memory_leak_check):
    """
    Test operator.getitem on String array with a slice ind
    """

    def test_impl(A, ind):
        return A[ind]

    ind = slice(1, 4)
    # Pandas outputs differs (object vs string)
    # TODO [BE-483]: Fix distributed
    check_func(test_impl, (str_arr_value, ind), check_dtype=False, dist_test=False)
    # Test with a slice with non step > 1
    # TODO [BE-483]: Fix distributed
    ind = slice(1, 4, 2)
    check_func(test_impl, (str_arr_value, ind), check_dtype=False, dist_test=False)


@pytest.mark.smoke
def test_setitem_int(memory_leak_check):
    def test_impl(A, idx, val):
        A[idx] = val
        return A

    A = pd.array(["AB", "", "한국", pd.NA, "abcd"])
    idx = 2
    val = "국한"  # same size as element 2 but different value
    check_func(test_impl, (A, idx, val), copy_input=True)


@pytest.mark.slow
def test_setitem_none_int(memory_leak_check):
    def test_impl(n, idx):
        A = bodo.libs.str_arr_ext.pre_alloc_string_array(n, n - 1)
        for i in range(n):
            if i == idx:
                A[i] = None
                continue
            A[i] = "A"
        return A

    py_output = pd.array(["A", None] + ["A"] * 6, "string")
    check_func(test_impl, (8, 1), copy_input=True, dist_test=False, py_output=py_output)


@pytest.mark.slow
def test_setitem_optional_int(memory_leak_check):
    def test_impl(n, idx):
        A = bodo.libs.str_arr_ext.pre_alloc_string_array(n, n - 1)
        for i in range(n):
            if i == idx:
                value = None
            else:
                value = "A"
            A[i] = value
        return A

    py_output = pd.array(["A", None] + ["A"] * 6, "string")
    check_func(test_impl, (8, 1), copy_input=True, dist_test=False, py_output=py_output)


@pytest.mark.slow
def test_setitem_slice(memory_leak_check):
    """
    Test operator.setitem with a slice index. String arrays
    should only have setitem used during initialization, so
    we create a new string array in the test.
    """

    nonascii_val = gen_nonascii_list(1)[0]

    def test_impl(val):
        A = bodo.libs.str_arr_ext.pre_alloc_string_array(8, -1)
        A[0] = nonascii_val
        A[1] = "CD"
        A[2:7] = val
        A[7] = "GH"
        return A

    values = (pd.array(["IJ"] * 5), ["IJ"] * 5, "IJ")
    py_output = pd.array([nonascii_val, "CD"] + ["IJ"] * 5 + ["GH"], "string")
    for val in values:
        check_func(test_impl, (val,), dist_test=False, py_output=py_output)


@pytest.mark.slow
def test_setitem_slice_optional(memory_leak_check):
    """
    Test operator.setitem with a slice index and an optional type.
    String arrays should only have setitem used during
    initialization, so we create a new string array in the test.
    """

    def test_impl(val, flag):
        A = bodo.libs.str_arr_ext.pre_alloc_string_array(8, -1)
        A[0] = "AB"
        A[1] = "CD"
        if flag:
            A[2:7] = val
        else:
            A[2:7] = None
        A[7] = "GH"
        return A

    values = (pd.array(["IJ"] * 5), ["IJ"] * 5, "IJ")
    py_output_flag = pd.array(["AB", "CD"] + ["IJ"] * 5 + ["GH"], "string")
    py_output_no_flag = pd.array(["AB", "CD"] + [None] * 5 + ["GH"], "string")
    for val in values:
        check_func(test_impl, (val, True), dist_test=False, py_output=py_output_flag)
        check_func(
            test_impl, (val, False), dist_test=False, py_output=py_output_no_flag
        )


@pytest.mark.slow
def test_setitem_slice_none(memory_leak_check):
    """
    Test operator.setitem with a slice index and None.
    String arrays should only have setitem used during
    initialization, so we create a new string array in the test.
    """

    def test_impl():
        A = bodo.libs.str_arr_ext.pre_alloc_string_array(8, -1)
        A[0] = "AB"
        A[1] = "CD"
        A[2:7] = None
        A[7] = "GH"
        return A

    py_output = pd.array(["AB", "CD"] + [None] * 5 + ["GH"], "string")
    check_func(test_impl, (), dist_test=False, py_output=py_output)


def test_setitem_bool(memory_leak_check):
    """
    Test operator.setitem with a bool index.
    The bool setitem index is used with Series.loc/Series.iloc, so
    it modifies the array in place. However, the size of the elements
    should be the same.
    """

    def test_impl(val, idx):
        A = bodo.libs.str_arr_ext.pre_alloc_string_array(8, -1)
        for i in range(8):
            A[i] = "AB"
        A[idx] = val
        return A

    values = (pd.array(["IJ"] * 5), np.array(["IJ"] * 5), "IJ")
    py_output = pd.array(["AB"] * 2 + ["IJ"] * 5 + ["AB"], "string")
    idx = [False, False, True, True, True, True, True, False]
    array_idx = np.array(idx)
    for val in values:
        check_func(test_impl, (val, idx), dist_test=False, py_output=py_output)
        check_func(test_impl, (val, array_idx), dist_test=False, py_output=py_output)


@pytest.mark.slow
def test_setitem_bool_optional(memory_leak_check):
    """
    Test operator.setitem with a bool index and an optional type.
    The bool setitem index is used with Series.loc/Series.iloc, so
    it modifies the array in place. However, the size of the elements
    should be the same.
    """

    def test_impl(val, idx, flag):
        A = bodo.libs.str_arr_ext.pre_alloc_string_array(8, -1)
        for i in range(8):
            A[i] = "AB"
        if flag:
            A[idx] = val
        else:
            A[idx] = None
        return A

    values = (pd.array(["IJ"] * 5), np.array(["IJ"] * 5), "IJ")
    py_output_flag = pd.array(["AB"] * 2 + ["IJ"] * 5 + ["AB"], "string")
    py_output_no_flag = pd.array(["AB"] * 2 + [None] * 5 + ["AB"], "string")
    idx = [False, False, True, True, True, True, True, False]
    array_idx = np.array(idx)
    for val in values:
        check_func(
            test_impl, (val, idx, False), dist_test=False, py_output=py_output_no_flag
        )
        check_func(
            test_impl,
            (val, array_idx, False),
            dist_test=False,
            py_output=py_output_no_flag,
        )
        check_func(
            test_impl, (val, idx, True), dist_test=False, py_output=py_output_flag
        )
        check_func(
            test_impl, (val, array_idx, True), dist_test=False, py_output=py_output_flag
        )


@pytest.mark.slow
def test_setitem_bool_none(memory_leak_check):
    """
    Test operator.setitem with a bool index and None.
    The bool setitem index is used with Series.loc/Series.iloc, so
    it modifies the array in place. However, the size of the elements
    should be the same.
    """

    def test_impl(idx):
        A = bodo.libs.str_arr_ext.pre_alloc_string_array(8, -1)
        for i in range(8):
            A[i] = "AB"
        A[idx] = None
        return A

    py_output = pd.array(["AB"] * 2 + [None] * 5 + ["AB"], "string")
    idx = [False, False, True, True, True, True, True, False]
    array_idx = np.array(idx)
    check_func(test_impl, (idx,), dist_test=False, py_output=py_output)
    check_func(test_impl, (array_idx,), dist_test=False, py_output=py_output)


@pytest.mark.slow
def test_dtype(memory_leak_check):
    def test_impl(A):
        return A.dtype

    check_func(test_impl, (pd.array((["AA", "B"] + gen_nonascii_list(2)) * 4),))


@pytest.mark.slow
def test_nbytes(memory_leak_check):
    """Test nbytes for string arrays"""

    def impl(arr):
        return arr.nbytes

    A = np.array(["AA", "B"] * 4, object)
    py_output = 8 * (8 + bodo.get_size()) + 12 + bodo.get_size()
    if bodo.hiframes.boxing._use_dict_str_type:
        # Each rank has a dictionary with 3 8-byte offsets, 3 characters and a null byte
        # There is also an index array with 8 4-byte offsets
        py_output = (8 * 3 + 3 + 1) * bodo.get_size() + 8 * 4 + bodo.get_size()

    # set use_dict_encoded_strings to avoid automatic testing of dict-encoded strings
    # since py_output will be different leading to errors
    check_func(
        impl,
        (A,),
        py_output=py_output,
        use_dict_encoded_strings=bodo.hiframes.boxing._use_dict_str_type,
    )


@pytest.mark.slow
def test_ndim(memory_leak_check):
    def test_impl(A):
        return A.ndim

    check_func(test_impl, (pd.array((["AA", "B"] + gen_nonascii_list(2)) * 4),))


@pytest.mark.slow
def test_tolist(memory_leak_check):
    def impl(A):
        return A.tolist()

    # NOTE: only Numpy array has tolist(), not Pandas arrays
    check_func(
        impl, (np.array((["A", "BCD"] + gen_nonascii_list(2)) * 4),), only_seq=True
    )
    check_func(impl, (np.arange(11),), only_seq=True)


@pytest.mark.slow
def test_astype_str(memory_leak_check):
    def test_impl(A):
        return A.astype(str)

    check_func(test_impl, (pd.array((["AA", "B"] + gen_nonascii_list(2)) * 4),))


@pytest.mark.skipif(
    bodo.hiframes.boxing._use_dict_str_type, reason="not supported for dict string type"
)
def test_str_copy_inplace(memory_leak_check):
    """Test inplace string copy optimization across arrays in series pass"""

    # scalar case
    def impl1(A):
        B = bodo.libs.str_arr_ext.pre_alloc_string_array(1, -1)
        B[0] = A[1]
        return B

    A = np.array(["AA", "B"] * 4, object)
    j_func = numba.njit(pipeline_class=SeqTestPipeline, parallel=True)(impl1)
    assert j_func(A) == A[1]
    fir = j_func.overloads[j_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(fir, "get_str_arr_item_copy")

    # both parallel case
    def impl2(A):
        n = len(A)
        B = bodo.libs.str_arr_ext.pre_alloc_string_array(n, -1)
        for i in bodo.prange(n):
            B[i] = A[i]
        return B

    check_func(impl2, (A,), py_output=A, only_1D=True)

    # output parallel case
    def impl3(s, n):
        A = bodo.libs.str_arr_ext.str_arr_from_sequence([s])
        B = bodo.libs.str_arr_ext.pre_alloc_string_array(n, -1)
        for i in bodo.prange(n):
            B[i] = A[0]

        return B

    s = "ABC"
    n = 6
    py_out = np.array([s] * n, object)
    check_func(impl3, (s, n), py_output=py_out, only_1D=True)

    # TODO[BE-2275]: handle sequential access to distributed arrays through internal
    # functuons properly
    # # input parallel case
    # def impl4(A):
    #     B = bodo.libs.str_arr_ext.pre_alloc_string_array(1, -1)
    #     B[0] = A[0]
    #     return B

    # A = np.array([f"A{bodo.get_rank()}", "B"] * 4, object)
    # assert bodo.jit(distributed=["A"])(impl4)(A)[0] == "A0"


def _check_str_item_length(impl):
    """make sure 'impl' is optimized to use str_item_length"""
    A = np.array(["AA", "B"] * 4, object)
    j_func = numba.njit(pipeline_class=SeqTestPipeline, parallel=True)(impl)
    assert j_func(A) == impl(A)
    fir = j_func.overloads[j_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(fir, "get_str_arr_str_length")


@pytest.mark.skipif(
    bodo.hiframes.boxing._use_dict_str_type, reason="not supported for dict string type"
)
def test_str_length_inplace(memory_leak_check):
    """Test optimizing len(A[i]) with inplace item length in series pass"""

    def impl1(A):
        return len(A[0])

    def impl2(A):
        s = 0
        for i in range(len(A)):
            s += len(A[i])
        return s

    def impl3(A):
        return pd.Series(A).str.len().sum()

    _check_str_item_length(impl1)
    _check_str_item_length(impl2)
    _check_str_item_length(impl3)


@pytest.mark.slow
def test_str_array_setitem_unsupported(memory_leak_check):
    """
    Checks that string array setitem with unsupported index, value
    pairs throw BodoErrors. String Array setitem shouldn't occur
    after initialization, but since these tests should error at compile
    time, this shouldn't be an issue.
    """

    def impl(arr, idx, val):
        arr[idx] = val

    err_msg = "StringArray setitem with index .* and value .* not supported yet."

    arr = pd.array(["AB", "", "ABC", pd.NA, "abcd"])
    # Check integer with a non-string
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl)(arr, 2, ["erw", "qwewq"])
    # Check slice with a numpy str array
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl)(arr, slice(0, 2), np.array(["erw", "qwe"]))

    bool_list = np.array([True, False, True, False, True])
    # Check boolean array with list
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl)(arr, bool_list, ["erw", "qwewq"])
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl)(arr, np.array(bool_list), ["erw", "qwewq"])

    # Check integer array index (not supported yet)
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl)(arr, np.array([1, 2]), ["erw", "qwewq"])
