# Copyright (C) 2019 Bodo Inc. All rights reserved.
import pandas as pd
import numpy as np
import itertools
import numba
import bodo
import random
from bodo.tests.utils import (
    count_array_REPs,
    count_parfor_REPs,
    count_parfor_OneDs,
    count_array_OneDs,
    count_array_OneD_Vars,
    dist_IR_contains,
    get_rank,
    get_start_end,
    DeadcodeTestPipeline,
)
from bodo.utils.utils import is_expr, is_assign
import pytest
from bodo.tests.utils import check_func


def test_membership():
    d = numba.typed.Dict.empty(
        key_type=numba.core.types.unicode_type, value_type=numba.core.types.int64
    )
    d["A"] = 0
    d["B"] = 0

    def test_impl(d):
        test = "A" in d
        return test

    check_func(test_impl, (d,))


def test_getitem():
    def test_impl(N):
        A = np.ones(N)
        B = np.ones(N) > 0.5
        C = A[B]
        return C.sum()

    n = 128
    check_func(test_impl, (n,))


def test_setitem1():
    def test_impl(N):
        A = np.arange(10) + 1.0
        A[0] = 30
        return A.sum()

    n = 128
    check_func(test_impl, (n,))


def test_setitem2():
    def test_impl(N):
        A = np.arange(10) + 1.0
        A[0:4] = 30
        return A.sum()

    n = 128
    check_func(test_impl, (n,))


def test_astype():
    def test_impl(N):
        return np.ones(N).astype(np.int32).sum()

    n = 128
    check_func(test_impl, (n,))


def test_shape():
    def test_impl(N):
        return np.ones(N).shape[0]

    n = 128
    check_func(test_impl, (n,))


def test_inplace_binop():
    def test_impl(N):
        A = np.ones(N)
        B = np.ones(N)
        B += A
        return B.sum()

    n = 128
    check_func(test_impl, (n,))


def test_getitem_multidim():
    def test_impl(N):
        A = np.ones((N, 3))
        B = np.ones(N) > 0.5
        C = A[B, 2]
        return C.sum()

    n = 128
    check_func(test_impl, (n,))


def test_whole_slice():
    def test_impl(N):
        X = np.ones((N, 4))
        X[:, 3] = (X[:, 3]) / (np.max(X[:, 3]) - np.min(X[:, 3]))
        return X.sum()

    n = 128
    check_func(test_impl, (n,))


def test_strided_getitem():
    def test_impl(N):
        A = np.ones(N)
        B = A[::7]
        return B.sum()

    n = 128
    check_func(test_impl, (n,))


def test_array_sum_axis():
    """test array.sum() with axis argument
    """
    def test_impl1(A):
        return A.sum(0)

    def test_impl2(A):
        return A.sum(axis=0)

    def test_impl3(A):
        return A.sum(axis=1)

    A = np.arange(33).reshape(11, 3)
    check_func(test_impl1, (A,), is_out_distributed=False)
    check_func(test_impl2, (A,), is_out_distributed=False)
    check_func(test_impl3, (A,))


@pytest.mark.skip(reason="TODO: replace since to_numeric() doesn't need locals anymore")
def test_inline_locals():
    # make sure locals in inlined function works
    @bodo.jit(locals={"B": bodo.float64[:]})
    def g(S):
        B = pd.to_numeric(S, errors="coerce")
        return B

    def f():
        return g(pd.Series(["1.2"]))

    pd.testing.assert_series_equal(bodo.jit(f)(), f())


@pytest.fixture(params=["float32", "float64", "int32", "int64"])
def test_dtypes_input(request):
    return request.param


@pytest.fixture(params=["sum", "prod", "min", "max", "argmin", "argmax"])
def test_funcs_input(request):
    return request.param


def test_reduce(test_dtypes_input, test_funcs_input):
    import sys

    # loc allreduce doesn't support int64 on windows
    dtype = test_dtypes_input
    func = test_funcs_input
    if not (
        sys.platform.startswith("win")
        and dtype == "int64"
        and func in ["argmin", "argmax"]
    ):

        func_text = """def f(n):
            A = np.arange(0, n, 1, np.{})
            return A.{}()
        """.format(
            dtype, func
        )
        loc_vars = {}
        exec(func_text, {"np": np, "bodo": bodo}, loc_vars)
        test_impl = loc_vars["f"]
        n = 21  # XXX arange() on float32 has overflow issues on large n
        check_func(test_impl, (n,))


def test_reduce2(test_dtypes_input, test_funcs_input):
    import sys

    dtype = test_dtypes_input
    func = test_funcs_input

    # loc allreduce doesn't support int64 on windows
    if not (
        sys.platform.startswith("win")
        and dtype == "int64"
        and func in ["argmin", "argmax"]
    ):

        func_text = """def f(A):
            return A.{}()
        """.format(
            func
        )
        loc_vars = {}
        exec(func_text, {"np": np}, loc_vars)
        test_impl = loc_vars["f"]

        n = 21
        np.random.seed(0)
        A = np.random.randint(0, 10, n).astype(dtype)
        check_func(test_impl, (A,))


def test_reduce_filter1(test_dtypes_input, test_funcs_input):
    import sys

    dtype = test_dtypes_input
    func = test_funcs_input
    # loc allreduce doesn't support int64 on windows
    if not (
        sys.platform.startswith("win")
        and dtype == "int64"
        and func in ["argmin", "argmax"]
    ):

        func_text = """def f(A):
            A = A[A>5]
            return A.{}()
        """.format(
            func
        )
        loc_vars = {}
        exec(func_text, {"np": np}, loc_vars)
        test_impl = loc_vars["f"]
        n = 21
        np.random.seed(0)
        A = np.random.randint(0, 10, n).astype(dtype)
        check_func(test_impl, (A,))


def test_array_reduce():
    binops = ["+=", "*=", "+=", "*=", "|=", "|="]
    dtypes = [
        "np.float32",
        "np.float32",
        "np.float64",
        "np.float64",
        "np.int32",
        "np.int64",
    ]
    for (op, typ) in zip(binops, dtypes):
        func_text = """def f(n):
                A = np.arange(0, 10, 1, {})
                B = np.arange(0 +  3, 10 + 3, 1, {})
                for i in numba.prange(n):
                    A {} B
                return A
        """.format(
            typ, typ, op
        )
        loc_vars = {}
        exec(func_text, {"np": np, "numba": numba, "bodo": bodo}, loc_vars)
        test_impl = loc_vars["f"]

        bodo_func = bodo.jit(test_impl)

        n = 128
        np.testing.assert_allclose(bodo_func(n), test_impl(n))
        assert count_array_OneDs() == 0
        assert count_parfor_OneDs() == 1


def _check_IR_no_getitem(test_impl, args):
    """makes sure there is no getitem/static_getitem left in the IR after optimization
    """
    bodo_func = numba.njit(pipeline_class=DeadcodeTestPipeline, parallel=True)(
        test_impl
    )
    bodo_func(*args)  # calling the function to get function IR
    fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert len(fir.blocks) == 1
    # make sure there is no getitem in IR
    for stmt in fir.blocks[0].body:
        assert not (
            is_assign(stmt)
            and (
                is_expr(stmt.value, "getitem") or is_expr(stmt.value, "static_getitem")
            )
        )


def test_trivial_slice_getitem_opt():
    """Make sure trivial slice getitem is optimized out, e.g. B = A[:]
    """

    def test_impl1(df):
        return df.iloc[:, 0]

    def test_impl2(A):
        return A[:]

    df = pd.DataFrame({"A": [1, 2, 5]})
    _check_IR_no_getitem(test_impl1, (df,))
    _check_IR_no_getitem(test_impl2, (np.arange(10),))


def _check_IR_single_label(test_impl, args):
    """makes sure the IR has a single label
    """
    bodo_func = numba.njit(pipeline_class=DeadcodeTestPipeline, parallel=True)(
        test_impl
    )
    bodo_func(*args)  # calling the function to get function IR
    fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert len(fir.blocks) == 1


# global flag used for testing below
g_flag = True


def test_dead_branch_remove():
    """Make sure dead branches are removed
    """

    def test_impl1():
        if g_flag:
            return 3
        return 2

    def test_impl2():
        f = False
        if f:
            return 3
        return 2


    _check_IR_single_label(test_impl1, ())
    _check_IR_single_label(test_impl2, ())


def test_return():
    def test_impl(N):
        A = np.arange(N)
        return A

    n = 128
    check_func(test_impl, (n,))


def test_return_tuple():
    def test_impl(N):
        A = np.arange(N)
        B = np.arange(N) + 1.5
        return A, B

    n = 128
    check_func(test_impl, (n,))


def test_input():
    def test_impl(A):
        return len(A)

    n = 128
    arr = np.ones(n)
    check_func(test_impl, (arr,))


def test_rebalance():
    def test_impl(N):
        A = np.arange(n)
        B = A[A > 10]
        C = bodo.libs.distributed_api.rebalance_array(B)
        return C.sum()

    n = 128
    check_func(test_impl, (n,))


def test_rebalance_loop():
    def test_impl(N):
        A = np.arange(n)
        B = A[A > 10]
        s = 0
        for i in range(3):
            s += B.sum()
        return s

    n = 128
    try:
        bodo.transforms.distributed_analysis.auto_rebalance = True
        check_func(test_impl, (n,))
    finally:
        bodo.transforms.distributed_analysis.auto_rebalance = False


def test_transpose():
    def test_impl(n):
        A = np.ones((30, 40, 50))
        B = A.transpose((0, 2, 1))
        C = A.transpose(0, 2, 1)
        return B.sum() + C.sum()

    n = 128
    check_func(test_impl, (n,))


def test_np_dot():
    def test_impl(n, k):
        A = np.ones((n, k))
        g = np.arange(k).astype(np.float64)
        B = np.dot(A, g)
        return B.sum()

    n = 128
    k = 3
    check_func(test_impl, (n, k))


def test_np_dot_empty_vm():
    """test for np.dot() called on empty vector and matrix (for Numba #5539)
    """
    X = np.array([]).reshape(0, 2)
    Y = np.array([])
    nb_res = numba.njit(lambda X, Y: np.dot(Y, X))(X, Y)
    py_res = np.dot(Y, X)
    np.testing.assert_array_equal(py_res, nb_res)


@pytest.mark.skip(reason="Numba's perfmute generation needs to use np seed properly")
def test_permuted_array_indexing():
    def get_np_state_ptr():
        return numba._helperlib.rnd_get_np_state_ptr()

    def _copy_py_state(r, ptr):
        """
        Copy state of Python random *r* to Numba state *ptr*.
        """
        mt = r.getstate()[1]
        ints, index = mt[:-1], mt[-1]
        numba._helperlib.rnd_set_state(ptr, (index, list(ints)))
        return ints, index

    def _rank_begin(arr_len):
        f = bodo.jit(
            lambda arr_len, num_ranks, rank: bodo.libs.distributed_api.get_start(
                arr_len, np.int32(num_ranks), np.int32(rank)
            )
        )
        num_ranks = bodo.libs.distributed_api.get_size()
        rank = bodo.libs.distributed_api.get_rank()
        return f(arr_len, num_ranks, rank)

    def _rank_end(arr_len):
        f = bodo.jit(
            lambda arr_len, num_ranks, rank: bodo.libs.distributed_api.get_end(
                arr_len, np.int32(num_ranks), np.int32(rank)
            )
        )
        num_ranks = bodo.libs.distributed_api.get_size()
        rank = bodo.libs.distributed_api.get_rank()
        return f(arr_len, num_ranks, rank)

    def _rank_bounds(arr_len):
        return _rank_begin(arr_len), _rank_end(arr_len)

    def _follow_cpython(ptr, seed=2):
        r = random.Random(seed)
        _copy_py_state(r, ptr)
        return r

    # Since Numba uses Python's PRNG for producing random numbers in NumPy,
    # we cannot compare against NumPy.  Therefore, we implement permutation
    # in Python.
    def python_permutation(n, r):
        arr = np.arange(n)
        r.shuffle(arr)
        return arr

    def test_one_dim(arr_len):
        A = np.arange(arr_len)
        B = np.copy(A)
        P = np.random.permutation(arr_len)
        A, B = A[P], B[P]
        return A, B

    # Implementation that uses Python's PRNG for producing a permutation.
    # We test against this function.
    def python_one_dim(arr_len, r):
        A = np.arange(arr_len)
        B = np.copy(A)
        P = python_permutation(arr_len, r)
        A, B = A[P], B[P]
        return A, B

    # Ideally, in above *_impl functions we should just call
    # np.random.seed() and they should produce the same sequence of random
    # numbers.  However, since Numba's PRNG uses NumPy's initialization
    # method for initializing PRNG, we cannot just set seed.  Instead, we
    # resort to this hack that generates a Python Random object with a fixed
    # seed and copies the state to Numba's internal NumPy PRNG state.  For
    # details please see https://github.com/numba/numba/issues/2782.
    r = _follow_cpython(get_np_state_ptr())

    hpat_func1 = bodo.jit(
        locals={"A:return": "distributed", "B:return": "distributed"}
    )(test_one_dim)

    # Test one-dimensional array indexing.
    for arr_len in [11, 111, 128, 120]:
        hpat_A, hpat_B = hpat_func1(arr_len)
        python_A, python_B = python_one_dim(arr_len, r)
        rank_bounds = self._rank_bounds(arr_len)
        np.testing.assert_allclose(hpat_A, python_A[slice(*rank_bounds)])
        np.testing.assert_allclose(hpat_B, python_B[slice(*rank_bounds)])

    # Test two-dimensional array indexing.  Like in one-dimensional case
    # above, in addition to NumPy version that is compiled by Numba, we
    # implement a Python version.
    def test_two_dim(arr_len):
        first_dim = arr_len // 2
        A = np.arange(arr_len).reshape(first_dim, 2)
        B = np.copy(A)
        P = np.random.permutation(first_dim)
        A, B = A[P], B[P]
        return A, B

    def python_two_dim(arr_len, r):
        first_dim = arr_len // 2
        A = np.arange(arr_len).reshape(first_dim, 2)
        B = np.copy(A)
        P = python_permutation(first_dim, r)
        A, B = A[P], B[P]
        return A, B

    hpat_func2 = bodo.jit(
        locals={"A:return": "distributed", "B:return": "distributed"}
    )(test_two_dim)

    for arr_len in [18, 66, 128]:
        hpat_A, hpat_B = hpat_func2(arr_len)
        python_A, python_B = python_two_dim(arr_len, r)
        rank_bounds = _rank_bounds(arr_len // 2)
        np.testing.assert_allclose(hpat_A, python_A[slice(*rank_bounds)])
        np.testing.assert_allclose(hpat_B, python_B[slice(*rank_bounds)])

    # Test that the indexed array is not modified if it is not being
    # assigned to.
    def test_rhs(arr_len):
        A = np.arange(arr_len)
        B = np.copy(A)
        P = np.random.permutation(arr_len)
        C = A[P]
        return A, B, C

    hpat_func3 = bodo.jit(
        locals={
            "A:return": "distributed",
            "B:return": "distributed",
            "C:return": "distributed",
        }
    )(test_rhs)

    for arr_len in [15, 23, 26]:
        A, B, _ = hpat_func3(arr_len)
        np.testing.assert_allclose(A, B)
