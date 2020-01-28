# Copyright (C) 2019 Bodo Inc. All rights reserved.
import pandas as pd
import numpy as np
import pytest

import numba
import bodo
from bodo.utils.typing import BodoWarning, BodoError
from bodo.tests.utils import (
    count_array_REPs,
    count_parfor_REPs,
    count_parfor_OneDs,
    count_array_OneDs,
    dist_IR_contains,
    get_start_end,
    count_array_OneD_Vars,
    _get_dist_arg,
    _test_equal,
    check_func,
)


@pytest.mark.parametrize("A", [np.arange(11), np.arange(33).reshape(11, 3)])
def test_array_shape1(A):
    # get first dimention size using array.shape for distributed arrays
    def impl1(A):
        return A.shape[0]

    bodo_func = bodo.jit(distributed={"A"})(impl1)
    start, end = get_start_end(len(A))
    assert bodo_func(A[start:end]) == impl1(A)
    assert count_array_REPs() == 0
    assert dist_IR_contains("dist_reduce")


def test_array_shape2():
    # get first dimention size using array.shape for distributed arrays
    # transposed array case
    def impl1(A):
        B = A.T
        return B.shape[1]

    bodo_func = bodo.jit(distributed={"A"})(impl1)
    n = 11
    A = np.arange(n * 3).reshape(n, 3)
    start, end = get_start_end(n)
    assert bodo_func(A[start:end]) == impl1(A)
    assert count_array_REPs() == 0
    assert dist_IR_contains("dist_reduce")
    # TODO: test Array.ctypes.shape[0] cases


@pytest.mark.parametrize("A", [np.arange(11), np.arange(33).reshape(11, 3)])
def test_array_shape3(A):
    # get first dimention size using array.shape for distributed arrays
    def impl1(A):
        return A.shape

    bodo_func = bodo.jit(distributed={"A"})(impl1)
    start, end = get_start_end(len(A))
    assert bodo_func(A[start:end]) == impl1(A)
    assert count_array_REPs() == 0
    assert dist_IR_contains("dist_reduce")


def test_array_shape4():
    # transposed array case
    def impl1(A):
        B = A.T
        return B.shape

    bodo_func = bodo.jit(distributed={"A"})(impl1)
    n = 11
    A = np.arange(n * 3).reshape(n, 3)
    start, end = get_start_end(n)
    assert bodo_func(A[start:end]) == impl1(A)
    assert count_array_REPs() == 0
    assert dist_IR_contains("dist_reduce")


def test_array_len1():
    # get first dimention size using array.shape for distributed arrays
    def impl1(A):
        return len(A)

    bodo_func = bodo.jit(distributed={"A"})(impl1)
    n = 11
    A = np.arange(n * 3).reshape(n, 3)
    start, end = get_start_end(n)
    assert bodo_func(A[start:end]) == impl1(A)
    assert count_array_REPs() == 0
    assert dist_IR_contains("dist_reduce")
    # TODO: tests with array created inside the function


@pytest.mark.parametrize("A", [np.arange(11), np.arange(33).reshape(11, 3)])
def test_array_size1(A):
    def impl1(A):
        return A.size

    bodo_func = bodo.jit(distributed={"A"})(impl1)
    start, end = get_start_end(len(A))
    assert bodo_func(A[start:end]) == impl1(A)
    assert count_array_REPs() == 0
    assert dist_IR_contains("dist_reduce")
    # TODO: tests with array created inside the function


def test_1D_Var_parfor1():
    # 1D_Var parfor where index is used in computation
    def impl1(A, B):
        C = A[B != 0]
        s = 0
        for i in bodo.prange(len(C)):
            s += i + C[i]
        return s

    bodo_func = bodo.jit(distributed={"A", "B"})(impl1)
    A = np.arange(11)
    start, end = get_start_end(len(A))
    B = np.arange(len(A)) % 2
    assert bodo_func(A[start:end], B[start:end]) == impl1(A, B)
    assert count_array_REPs() == 0


def test_1D_Var_parfor2():
    # 1D_Var parfor where index is used in computation
    def impl1(A, B):
        C = A[B != 0]
        s = 0
        for i in bodo.prange(len(C)):
            s += i + C[i, 0]
        return s

    bodo_func = bodo.jit(distributed={"A", "B"})(impl1)
    A = np.arange(33).reshape(11, 3)
    start, end = get_start_end(len(A))
    B = np.arange(len(A)) % 2
    assert bodo_func(A[start:end], B[start:end]) == impl1(A, B)
    assert count_array_REPs() == 0


def test_print1():
    # no vararg
    # TODO: capture stdout and make sure there is only one print
    def impl1(a, b):
        print(a, b)

    bodo_func = bodo.jit()(impl1)
    bodo_func(1, 2)
    bodo_func(np.ones(3), 3)
    bodo_func((3, 4), 2)


def test_print2():
    # vararg
    # TODO: capture stdout and make sure there is only one print
    def impl1(a):
        print(*a)

    bodo_func = bodo.jit()(impl1)
    bodo_func((3, 4))
    bodo_func((3, np.ones(3)))


def test_print3():
    # arg and vararg
    # TODO: capture stdout and make sure there is only one print
    def impl1(a, b):
        print(a, *b)

    bodo_func = bodo.jit()(impl1)
    bodo_func(1, (3, 4))
    bodo_func(np.ones(3), (3, np.ones(3)))


@pytest.mark.parametrize("A", [np.arange(11), np.arange(33).reshape(11, 3)])
def test_1D_Var_alloc_simple(A):
    # make sure 1D_Var alloc and parfor handling works for 1D/2D arrays
    def impl1(A, B):
        C = A[B]
        return C.sum()

    bodo_func = bodo.jit(distributed={"A", "B"})(impl1)
    start, end = get_start_end(len(A))
    B = np.arange(len(A)) % 2 != 0
    assert bodo_func(A[start:end], B[start:end]) == impl1(A, B)
    assert count_array_REPs() == 0


def test_1D_Var_alloc1():
    # XXX: test with different PYTHONHASHSEED values
    def impl1(A, B):
        C = A[B]
        n = len(C)
        if n < 1:
            D = C + 1.0
        else:
            # using prange instead of an operator to avoid empty being inside the
            # parfor init block, with parfor stop variable already transformed
            D = np.empty(n)
            for i in bodo.prange(n):
                D[i] = C[i] + 1.0
        return D

    bodo_func = bodo.jit(distributed={"A", "B", "D"})(impl1)
    A = np.arange(11)
    start, end = get_start_end(len(A))
    B = np.arange(len(A)) % 2 != 0
    res = bodo_func(A[start:end], B[start:end]).sum()
    dist_sum = bodo.jit(
        lambda a: bodo.libs.distributed_api.dist_reduce(
            a, np.int32(bodo.libs.distributed_api.Reduce_Type.Sum.value)
        )
    )
    assert dist_sum(res) == impl1(A, B).sum()
    assert count_parfor_REPs() == 0


def test_1D_Var_alloc2():
    # XXX: test with different PYTHONHASHSEED values
    # 2D case
    def impl1(A, B):
        m = A.shape[1]
        C = A[B]
        n = C.shape[0]  # len(C), TODO: fix array analysis
        if n < 1:
            D = C + 1.0
        else:
            # using prange instead of an operator to avoid empty being inside the
            # parfor init block, with parfor stop variable already transformed
            D = np.empty((n, m))
            for i in bodo.prange(n):
                D[i] = C[i] + 1.0
        return D

    bodo_func = bodo.jit(distributed={"A", "B", "D"})(impl1)
    A = np.arange(33).reshape(11, 3)
    start, end = get_start_end(len(A))
    B = np.arange(len(A)) % 2 != 0
    res = bodo_func(A[start:end], B[start:end]).sum()
    dist_sum = bodo.jit(
        lambda a: bodo.libs.distributed_api.dist_reduce(
            a, np.int32(bodo.libs.distributed_api.Reduce_Type.Sum.value)
        )
    )
    assert dist_sum(res) == impl1(A, B).sum()
    assert count_parfor_REPs() == 0


def test_1D_Var_alloc3():
    # XXX: test with different PYTHONHASHSEED values
    # Series case
    def impl1(A, B):
        C = A[B]
        n = len(C)
        if n < 1:
            D = C + 1.0
        else:
            # using prange instead of an operator to avoid empty being inside the
            # parfor init block, with parfor stop variable already transformed
            D = pd.Series(np.empty(n))
            for i in bodo.prange(n):
                D.values[i] = C.values[i] + 1.0
        return D

    bodo_func = bodo.jit(distributed={"A", "B", "D"})(impl1)
    A = pd.Series(np.arange(11))
    start, end = get_start_end(len(A))
    B = np.arange(len(A)) % 2 != 0
    res = bodo_func(A[start:end], B[start:end]).sum()
    dist_sum = bodo.jit(
        lambda a: bodo.libs.distributed_api.dist_reduce(
            a, np.int32(bodo.libs.distributed_api.Reduce_Type.Sum.value)
        )
    )
    assert dist_sum(res) == impl1(A, B).sum()
    assert count_parfor_REPs() == 0


def test_str_alloc_equiv1():
    def impl(n):
        C = bodo.libs.str_arr_ext.pre_alloc_string_array(n, 10)
        return len(C)

    bodo_func = bodo.jit()(impl)
    n = 11
    assert bodo_func(n) == n
    assert count_array_REPs() == 0
    assert not dist_IR_contains("dist_reduce")


def test_series_alloc_equiv1():
    def impl(n):
        if n < 10:
            S = pd.Series(np.ones(n))
        else:
            S = pd.Series(np.zeros(n))
        # B = np.full(len(S), 2)  # TODO: np.full dist handling
        B = np.empty(len(S))
        return B

    bodo_func = bodo.jit(distributed={"B"})(impl)
    n = 11
    bodo_func(n)
    assert count_parfor_REPs() == 0
    assert not dist_IR_contains("dist_reduce")


# TODO: test other array types
@pytest.mark.parametrize(
    "A", [np.arange(11), np.arange(33).reshape(11, 3), pd.Series(["aa", "bb", "c"] * 4)]
)
@pytest.mark.parametrize(
    "s", [slice(3), slice(1, 9), slice(7, None), slice(4, 6), slice(-3, None)]
)
def test_getitem_slice_1D(A, s):
    # get a slice of 1D array
    def impl1(A, s):
        return A[s]

    bodo_func = bodo.jit(distributed={"A"})(impl1)
    start, end = get_start_end(len(A))
    np.testing.assert_array_equal(bodo_func(A[start:end], s), impl1(A, s))
    assert count_array_OneDs() > 0


@pytest.mark.parametrize(
    "A", [np.arange(11), np.arange(33).reshape(11, 3), pd.Series(["aa", "bb", "c"] * 4)]
)
@pytest.mark.parametrize("s", [slice(3), slice(1, 9), slice(7, None), slice(4, 6)])
def test_getitem_slice_1D_Var(A, s):
    # get a slice of 1D array
    def impl1(A, B, s):
        C = A[B]
        return C[s]

    bodo_func = bodo.jit(distributed={"A", "B"})(impl1)
    start, end = get_start_end(len(A))
    B = np.arange(len(A)) % 2 != 0
    np.testing.assert_array_equal(
        bodo_func(A[start:end], B[start:end], s), impl1(A, B, s)
    )
    assert count_array_OneD_Vars() > 0


# TODO: np.arange(33).reshape(11, 3)
@pytest.mark.parametrize(
    "A", [pd.Series(np.arange(11)), pd.Series(["aafa", "bbac", "cff"] * 4)]
)
@pytest.mark.parametrize("s", [0, 1, 3, 7, 10, -1, -2])
def test_getitem_int_1D(A, s):
    # get a single value of 1D_Block array
    def impl1(A, s):
        return A.values[s]

    bodo_func = bodo.jit(distributed={"A"})(impl1)
    start, end = get_start_end(len(A))
    if A.ndim == 1:
        assert bodo_func(A[start:end], s) == impl1(A, s)
    else:
        np.testing.assert_array_equal(bodo_func(A[start:end], s), impl1(A, s))
    assert count_array_OneDs() > 0


# TODO: np.arange(33).reshape(11, 3)
@pytest.mark.parametrize("A", [pd.Series(np.arange(11))])
#    pd.Series(['aafa', 'bbac', 'cff']*4)])
@pytest.mark.parametrize("s", [0, 1, 3, -1, -2])
def test_getitem_int_1D_Var(A, s):
    # get a single value of 1D_Block array
    def impl1(A, B, s):
        C = A.values[B]
        return C[s]

    bodo_func = bodo.jit(distributed={"A", "B"})(impl1)
    start, end = get_start_end(len(A))
    B = np.arange(len(A)) % 2 != 0
    if A.ndim == 1:
        assert bodo_func(A[start:end], B[start:end], s) == impl1(A, B, s)
    else:
        np.testing.assert_array_equal(
            bodo_func(A[start:end], B[start:end], s), impl1(A, B, s)
        )
    assert count_array_OneD_Vars() > 0


def test_getitem_const_slice_multidim():
    """test getitem of multi-dim distributed array with a constant slice in first
    dimension.
    """
    def impl(A):
        return A[1:3,0,1:]

    bodo_func = bodo.jit(distributed={"A"})(impl)
    n = 5
    A = np.arange(n*n*n).reshape(n, n, n)
    start, end = get_start_end(len(A))
    np.testing.assert_array_equal(bodo_func(A[start:end]), impl(A))


def test_dist_tuple1():
    def impl1(A):
        B1, B2 = A
        return (B1 + B2).sum()

    n = 11
    A = (np.arange(n), np.ones(n))
    start, end = get_start_end(n)
    A_par = (A[0][start:end], A[1][start:end])
    bodo_func = bodo.jit(distributed={"A"})(impl1)
    assert bodo_func(A_par) == impl1(A)
    assert count_array_OneDs() > 0


def test_dist_tuple2():
    # TODO: tuple getitem with variable index
    def impl1(A, B):
        C = (A, B)
        return C

    n = 11
    A = np.arange(n)
    B = np.ones(n)
    start, end = get_start_end(n)
    bodo_func = bodo.jit(distributed={"A", "B", "C"})(impl1)

    py_out = impl1(A, B)
    bodo_out = bodo_func(A[start:end], B[start:end])
    np.testing.assert_array_equal(bodo_out[0], py_out[0][start:end])
    np.testing.assert_array_equal(bodo_out[1], py_out[1][start:end])
    assert count_array_OneDs() > 0


def test_dist_tuple3():
    """Make sure passing a dist tuple with non-dist elements doesn't cause REP
    """

    def impl1(v):
        (_, df) = v
        return df

    n = 11
    df = pd.DataFrame({"A": np.arange(n)})
    v = (n, df)
    bodo.jit(distributed={"v", "df"})(impl1)(v)
    assert count_array_OneDs() > 0


def test_dist_list1():
    """Test support for build_list of dist data
    """

    def impl1(df):
        v = [(1, df)]
        return v

    n = 11
    df = pd.DataFrame({"A": np.arange(n)})
    bodo.jit(distributed={"v", "df"})(impl1)(df)
    assert count_array_OneDs() > 0


def test_dist_list_append1():
    """Test support for list.append of dist tuple
    """

    def impl1(df):
        v = [(1, df)]
        v.append((1, df))
        return v

    n = 11
    df = pd.DataFrame({"A": np.arange(n)})
    bodo.jit(distributed={"v", "df"})(impl1)(df)
    assert count_array_OneDs() > 0


def test_dist_list_append2():
    """Test support for list.append of dist data
    """

    def impl1(df):
        v = [df]
        v.append(df)
        return v

    n = 11
    df = pd.DataFrame({"A": np.arange(n)})
    bodo.jit(distributed={"v", "df"})(impl1)(df)
    assert count_array_OneDs() > 0


def test_dist_list_getitem1():
    """Test support for getitem of distributed list
    """

    def impl1(v):
        df = v[1]
        return df

    n = 11
    df = pd.DataFrame({"A": np.arange(n)})
    v = [df, df]
    bodo.jit(distributed={"v", "df"})(impl1)(v)
    assert count_array_OneDs() > 0


def test_dist_dict1():
    """Test support for build_map of dist data
    """

    def impl1(df):
        v = {1: df}
        return v

    n = 11
    df = pd.DataFrame({"A": np.arange(n)})
    bodo.jit(distributed={"v", "df"})(impl1)(df)
    assert count_array_OneDs() > 0


def test_dist_dict_getitem1():
    """Test support for getitem of dist dictionary
    """

    def impl1(v):
        df = v[1]
        return df

    n = 11
    df = pd.DataFrame({"A": np.arange(n)})
    v = bodo.typed.Dict.empty(bodo.int64, bodo.typeof(df))
    v[0] = df
    v[1] = df
    bodo.jit(distributed={"v", "df"})(impl1)(v)
    assert count_array_OneDs() > 0


def test_dist_warning1():
    """Make sure BodoWarning is thrown when there is no parallelism discovered due
    to unsupported function
    """

    def impl(n):
        A = np.ones((n, n))
        # using a function we are not likely to support for warning test
        # should be changed when/if we support slogdet()
        return np.linalg.slogdet(A)

    if bodo.get_rank() == 0:  # warning is thrown only on rank 0
        with pytest.warns(BodoWarning, match="No parallelism found for function"):
            bodo.jit(impl)(10)
    else:
        bodo.jit(impl)(10)


def test_dist_warning2():
    """Make sure BodoWarning is thrown when there is no parallelism discovered due
    to return of dataframe
    """

    def impl(n):
        return pd.DataFrame({"A": np.ones(n)})

    if bodo.get_rank() == 0:  # warning is thrown only on rank 0
        with pytest.warns(BodoWarning, match="No parallelism found for function"):
            bodo.jit(impl)(10)
    else:
        bodo.jit(impl)(10)


def test_dist_warning3():
    """Make sure BodoWarning is thrown when a tuple variable with both distributable
    and non-distributable elemets is returned
    """

    def impl(n):
        df = pd.DataFrame({"A": np.ones(n)})
        return (n, df)

    if bodo.get_rank() == 0:  # warning is thrown only on rank 0
        with pytest.warns(BodoWarning, match="No parallelism found for function"):
            bodo.jit(impl)(10)
    else:
        bodo.jit(impl)(10)


def test_empty_object_array_warning():
    """Make sure BodoWarning is thrown when there is an empty object array in input
    """

    def impl(A):
        return A

    with pytest.warns(BodoWarning, match="Empty object array passed to Bodo"):
        bodo.jit(impl)(np.array([], dtype=np.object))
    with pytest.warns(BodoWarning, match="Empty object array passed to Bodo"):
        bodo.jit(impl)(pd.Series(np.array([], dtype=np.object)))


def test_dist_flags():
    """Make sure Bodo flags are preserved when the same Dispatcher that has distributed
    flags is called with different data types, triggering multiple compilations.
    See #357
    """

    def impl(A):
        return A.sum()

    n = 50
    A = np.arange(n)
    bodo_func = bodo.jit(all_args_distributed=True)(impl)
    result_bodo = bodo_func(_get_dist_arg(A, False))
    result_python = impl(A)
    if bodo.get_rank() == 0:
        _test_equal(result_bodo, result_python)

    A = np.arange(n, dtype=np.float64)  # change dtype to trigger compilation again
    result_bodo = bodo_func(_get_dist_arg(A, False))
    result_python = impl(A)
    if bodo.get_rank() == 0:
        _test_equal(result_bodo, result_python)


def test_dist_objmode():
    """Test use of objmode inside prange including a reduction.
    Tests a previous issue where deepcopy in get_parfor_reductions failed for
    ObjModeLiftedWith const.
    """
    import scipy.special as sc

    def objmode_test(n):
        A = np.arange(n)
        s = 0
        for i in bodo.prange(len(A)):
            x = A[i]
            with bodo.objmode(y="float64"):
                y = sc.entr(x)  # call entropy function on each data element
            s += y
        return s

    assert bodo.jit(objmode_test)(10) == objmode_test(10)


def test_diagnostics_not_compiled_error():
    """make sure error is thrown when calling diagnostics for a function that is not
    compiled yet
    """

    def test_impl():
        return np.arange(10).sum()

    with pytest.raises(BodoError, match="Distributed diagnostics not available for"):
        bodo.jit(test_impl).distributed_diagnostics()


def test_sort_output_1D_Var_size():
    """Test using size variable of an output 1D_Var array of a Sort node
    """
    # RangeIndex of output Series needs size of Sort output array
    def impl(S):
        res = pd.Series(S.sort_values().values)
        return res

    S = pd.Series([3, 4, 1, 2, 5])
    check_func(impl, (S,))
