# Copyright (C) 2022 Bodo Inc. All rights reserved.
import fractions
import random

import numba
import numpy as np
import pandas as pd
import pytest
from numba.core import types

import bodo
from bodo.tests.utils import (
    DeadcodeTestPipeline,
    DistTestPipeline,
    check_func,
    count_array_OneDs,
    count_parfor_OneDs,
    dist_IR_contains,
)
from bodo.utils.typing import BodoError
from bodo.utils.utils import is_assign, is_expr


@pytest.mark.slow
def test_membership(memory_leak_check):
    d = numba.typed.Dict.empty(
        key_type=numba.core.types.unicode_type, value_type=numba.core.types.int64
    )
    d["A"] = 0
    d["B"] = 0

    def test_impl(d):
        test = "A" in d
        return test

    check_func(test_impl, (d,))


@pytest.mark.slow
def test_dict_unbox(memory_leak_check):
    """test unboxing a regular dictionary"""
    d = {"A": 1, "B": 2}

    def test_impl(d):
        return d

    check_func(test_impl, (d,))

    # Save default developer mode value
    default_mode = numba.core.config.DEVELOPER_MODE

    # Test as a developer
    numba.core.config.DEVELOPER_MODE = 1

    # test an unsupported data type
    with pytest.raises(
        numba.errors.TypingError,
        match="Cannot type dict element type",
    ):
        bodo.jit(test_impl)({"A": fractions.Fraction(2, 3)})

    # Reset back to original setting
    numba.core.config.DEVELOPER_MODE = default_mode


# using a global instead of freevar to test the raise_on_unsupported_feature patch
d1 = {"A": 1, "B": 2}


# TODO(ehsan): add memory_leak_check when memory leak is resolved [BE-2114]
def test_dict_constant_lowering():
    """test constant lowering for dictionaries"""

    # fast path, keys/values can be arrays
    def impl1():
        return d1["B"]

    # slow path, keys/values are lowered individually
    d2 = {"A": pd.Series([1, 2, 4]), "B": pd.Series([5, 4])}

    def impl2():
        return d2["B"]

    check_func(impl1, (), only_seq=True)
    check_func(impl2, (), only_seq=True)


l1 = ["A", "ABC", "D2", "E4", "F53"]


def test_list_constant_lowering(memory_leak_check):
    """test constant lowering for lists"""

    def impl():
        return l1[-2]

    check_func(impl, (), only_seq=True)


l2 = [1, "A", "B"]


def test_list_constant_lowering_error_checking(memory_leak_check):
    """make sure value types are checked to be the same in list constant lowering"""

    def impl():
        return l2[-1]

    with pytest.raises(BodoError, match="Values in list must have the same data type"):
        bodo.jit(impl)()


s1 = {1, 4, 5, 11, 3}


# TODO [BE-2140]: add memory_leak_check when memory leak is resolved
def test_set_constant_lowering():
    """test constant lowering for lists"""

    def impl():
        return s1

    check_func(impl, (), only_seq=True)


s2 = {1, "A", "B"}


def test_set_constant_lowering_error_checking(memory_leak_check):
    """make sure value types are checked to be the same in set constant lowering"""

    def impl():
        return s2

    with pytest.raises(BodoError, match="Values in set must have the same data type"):
        bodo.jit(impl)()


@pytest.mark.smoke
def test_getitem(memory_leak_check):
    def test_impl(N):
        A = np.ones(N)
        B = np.ones(N) > 0.5
        C = A[B]
        return C.sum()

    n = 128
    check_func(test_impl, (n,))


@pytest.mark.smoke
def test_setitem1(memory_leak_check):
    def test_impl(N):
        A = np.arange(10) + 1.0
        A[0] = 30
        return A.sum()

    n = 128
    check_func(test_impl, (n,))


def test_setitem2(memory_leak_check):
    def test_impl(N):
        A = np.arange(10) + 1.0
        A[0:4] = 30
        return A.sum()

    n = 128
    check_func(test_impl, (n,))


@pytest.mark.slow
def test_astype(memory_leak_check):
    def test_impl(N):
        return np.ones(N).astype(np.int32).sum()

    n = 128
    check_func(test_impl, (n,))


@pytest.mark.slow
def test_shape(memory_leak_check):
    def test_impl(N):
        return np.ones(N).shape[0]

    n = 128
    check_func(test_impl, (n,))


@pytest.mark.smoke
def test_inplace_binop(memory_leak_check):
    def test_impl(N):
        A = np.ones(N)
        B = np.ones(N)
        B += A
        return B.sum()

    n = 128
    check_func(test_impl, (n,))


def test_getitem_multidim(memory_leak_check):
    def test_impl(N):
        A = np.ones((N, 3))
        B = np.ones(N) > 0.5
        C = A[B, 2]
        return C.sum()

    n = 128
    check_func(test_impl, (n,))


def test_whole_slice(memory_leak_check):
    def test_impl(N):
        X = np.ones((N, 4))
        X[:, 3] = (X[:, 3]) / (np.max(X[:, 3]) - np.min(X[:, 3]))
        return X.sum()

    n = 128
    check_func(test_impl, (n,))


def test_strided_getitem(memory_leak_check):
    def test_impl(N):
        A = np.ones(N)
        B = A[::7]
        return B.sum()

    n = 128
    check_func(test_impl, (n,))


def test_array_sum_axis(memory_leak_check):
    """test array.sum() with axis argument"""

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
def test_inline_locals(memory_leak_check):
    # make sure locals in inlined function works
    @bodo.jit(locals={"B": bodo.float64[:]})
    def g(S):
        B = pd.to_numeric(S, errors="coerce")
        return B

    def f():
        return g(pd.Series(["1.2"]))

    pd.testing.assert_series_equal(bodo.jit(f)(), f())


@pytest.fixture(params=["float32", "float64", "int32", "int64"])
def test_dtypes_input(request, memory_leak_check):
    return request.param


@pytest.fixture(params=["sum", "prod", "min", "max", "argmin", "argmax"])
def test_funcs_input(request, memory_leak_check):
    return request.param


@pytest.mark.smoke
def test_reduce(test_dtypes_input, test_funcs_input, memory_leak_check):
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


@pytest.mark.slow
def test_reduce2(test_dtypes_input, test_funcs_input, memory_leak_check):
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


@pytest.mark.slow
def test_reduce_init_val(memory_leak_check):
    """make sure _root_rank_select is not generated for common reductions with neutral
    init value.
    """

    def impl(n):
        return np.ones(n).sum()

    bodo_func = bodo.jit(pipeline_class=DistTestPipeline)(impl)
    assert bodo_func(11) == 11.0
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert not dist_IR_contains(f_ir, "_root_rank_select")


@pytest.mark.smoke
def test_reduce_filter1(test_dtypes_input, test_funcs_input, memory_leak_check):
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


def test_array_reduce(memory_leak_check):
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
    """makes sure there is no getitem/static_getitem left in the IR after optimization"""
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


@pytest.mark.slow
def test_trivial_slice_getitem_opt(memory_leak_check):
    """Make sure trivial slice getitem is optimized out, e.g. B = A[:]"""

    def test_impl1(df):
        return df.iloc[:, 0]

    def test_impl2(A):
        return A[:]

    df = pd.DataFrame({"A": [1, 2, 5]})
    _check_IR_no_getitem(test_impl1, (df,))
    _check_IR_no_getitem(test_impl2, (np.arange(10),))


def _check_IR_single_label(test_impl, args):
    """makes sure the IR has a single label"""
    bodo_func = numba.njit(pipeline_class=DeadcodeTestPipeline, parallel=True)(
        test_impl
    )
    bodo_func(*args)  # calling the function to get function IR
    fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert len(fir.blocks) == 1


# global flag used for testing below
g_flag = True


@pytest.mark.slow
def test_dead_branch_remove(memory_leak_check):
    """Make sure dead branches are removed"""

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


@pytest.mark.slow
def test_is_jit_execution(memory_leak_check):
    """make sure bodo.is_jit_execution() returns True inside a jit function, but False
    otherwise.
    """
    assert bodo.is_jit_execution() == False
    assert bodo.jit(lambda: bodo.is_jit_execution())() == True


@pytest.mark.slow
def test_return(memory_leak_check):
    def test_impl(N):
        A = np.arange(N)
        return A

    n = 128
    check_func(test_impl, (n,))


@pytest.mark.slow
def test_return_tuple(memory_leak_check):
    def test_impl(N):
        A = np.arange(N)
        B = np.arange(N) + 1.5
        return A, B

    n = 128
    check_func(test_impl, (n,))


@pytest.mark.slow
def test_input(memory_leak_check):
    def test_impl(A):
        return len(A)

    n = 128
    arr = np.ones(n)
    check_func(test_impl, (arr,))


@pytest.mark.slow
def test_transpose(memory_leak_check):
    def test_impl(n):
        A = np.ones((30, 40, 50))
        B = A.transpose((0, 2, 1))
        C = A.transpose(0, 2, 1)
        return B.sum() + C.sum()

    n = 128
    check_func(test_impl, (n,))


@pytest.mark.slow
def test_np_dot(memory_leak_check):
    def test_impl(n, k):
        A = np.ones((n, k))
        g = np.arange(k).astype(np.float64)
        B = np.dot(A, g)
        return B.sum()

    n = 128
    k = 3
    check_func(test_impl, (n, k))


@pytest.mark.smoke
def test_np_array(memory_leak_check):
    """test distribution of np.array() and np.asarray().
    array input can be distributed but not list input.
    """

    def test_impl1(A):
        return np.array(A)

    def test_impl2(A):
        return np.asarray(A)

    # TODO: enable when supported by Numba
    # check_func(test_impl1, (np.ones(11),))
    check_func(test_impl1, ([1, 2, 5, 1, 2, 3],), is_out_distributed=False)
    check_func(test_impl2, (np.ones(11),))
    check_func(test_impl2, ([1, 2, 5, 1, 2, 3],), is_out_distributed=False)


@pytest.mark.slow
def test_np_dot_empty_vm(memory_leak_check):
    """test for np.dot() called on empty vector and matrix (for Numba #5539)"""
    X = np.array([]).reshape(0, 2)
    Y = np.array([])
    nb_res = numba.njit(lambda X, Y: np.dot(Y, X))(X, Y)
    py_res = np.dot(Y, X)
    np.testing.assert_array_equal(py_res, nb_res)


@pytest.mark.slow
def test_dist_false_cache(memory_leak_check):
    """test (distributed=False, cache=True) jit combination (issue #2534)"""

    @bodo.jit(distributed=False, cache=True)
    def test(data):
        data = data.values

    df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 4, 5]})
    test(df)


@pytest.mark.slow
def test_np_full(memory_leak_check):
    """Test np.full() support (currently in Series pass)"""

    def impl1(shape, fill_value, dtype):
        return np.full(shape, fill_value, dtype)

    def impl2(shape, fill_value, dtype):
        return np.full(shape=shape, fill_value=fill_value, dtype=dtype)

    def impl3(n, m, fill_value, dtype):
        return np.full(shape=(n, m), fill_value=fill_value, dtype=dtype)

    check_func(impl1, (11, 1.0, np.float64))
    check_func(impl1, ((111,), 1, np.float32))
    check_func(impl2, (11, 1.3, np.float32))
    # FIXME: using a separate function for multi-dim case since passing tuple shape to
    # pndindex() fails in Numba
    check_func(impl3, (111, 4, 1, np.float32))


@pytest.mark.skip(reason="Numba's perfmute generation needs to use np seed properly")
def test_permuted_array_indexing(memory_leak_check):
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

    hpat_func1 = bodo.jit(distributed=["A", "B"])(test_one_dim)

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

    hpat_func2 = bodo.jit(distributed=["A", "B"])(test_two_dim)

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

    hpat_func3 = bodo.jit(distributed=["A", "B", "C"])(test_rhs)

    for arr_len in [15, 23, 26]:
        A, B, _ = hpat_func3(arr_len)
        np.testing.assert_allclose(A, B)


def test_func_non_jit_error(memory_leak_check):
    """make sure proper error is thrown when calling a non-JIT function"""

    def f():
        return 1

    def test_impl():
        return f()

    with pytest.raises(
        BodoError, match="Cannot call non-JIT function 'f' from JIT function"
    ):
        bodo.jit(test_impl)()


def test_func_nested_jit_error(memory_leak_check):
    """make sure proper error is thrown when calling a JIT function with errors"""

    @bodo.jit
    def f(df):
        return df["C"].min()

    @bodo.jit
    def g():
        df = pd.DataFrame({"A": [1, 2, 3]})
        return f(df)

    with pytest.raises(BodoError, match=r"does not include column C"):
        g()


# TODO: fix leak and add memory_leak_check
def test_udf_nest_jit_convert():
    """make sure nested JIT calls inside UDFs are converted to sequential properly.
    See [BE-2225].
    """

    @bodo.jit
    def f2(df):
        return df.last("2D")

    @bodo.jit
    def f1(df):
        df = df.set_index("B")
        return f2(df)

    @bodo.jit(distributed=["df", "df2"])
    def g(df):
        df2 = df.groupby("A").apply(f1)
        return df2

    df = pd.DataFrame(
        {
            "A": [1, 1, 3, 4, 1],
            "B": pd.date_range("2020-01-01", periods=5),
            "C": [1, 2, 3, 4, 5],
        }
    )
    g(df)


def test_updated_container_binop(memory_leak_check):
    """make sure binop of updated containers is detected and raises proper error"""

    def impl(p):
        a = []
        if p:
            a.append("C")
        return pd.DataFrame(np.ones((4, 3)), columns=["A", "B"] + a)

    @bodo.jit
    def f(a):
        a.append("A")

    # container used in loop before update call
    def impl2(n):
        a = []
        for i in range(n):
            f(a)
            a.append(str(i))
        return pd.DataFrame(np.ones((4, 5)), columns=["A", "B"] + a)

    # container used in loop after update call
    def impl3(n):
        a = []
        for i in range(n):
            a.append(str(i))
            f(a)
        return pd.DataFrame(np.ones((3, 3)), columns=a)

    # test unroll for set
    def impl4(n):
        a = set()
        for i in range(n):
            a.add(str(i))
        return pd.DataFrame(np.ones((5, 5)), columns=list({"A", "B"} | a))

    # update value is non-const
    def impl5(n, p):
        a = []
        for i in range(n):
            if p:
                c = 2
            else:
                c = str(i)
            a.append(c)
        return pd.DataFrame(np.ones((3, 3)), columns=a)

    # avoid unroll if loop is too large
    def impl6(n):
        a = []
        for i in range(n):
            a.append(str(i))
        return pd.DataFrame(np.ones((3, 3)), columns=a)

    # avoid unroll if loop is too large (general const loop case)
    def impl7(n):
        out = []
        for i in range(n):
            out.append([i, "A"])
        return out

    with pytest.raises(
        BodoError,
        match="argument 'columns' requires a constant value but variable '.*' is updated inplace using 'append'",
    ):
        bodo.jit(impl)(True)
    with pytest.raises(
        BodoError,
        match="argument 'columns' requires a constant value but variable '.*' is updated inplace using 'append'",
    ):
        bodo.jit(impl2)(3)
    with pytest.raises(
        BodoError,
        match="argument 'columns' requires a constant value but variable '.*' is updated inplace using 'append'",
    ):
        bodo.jit(impl3)(3)
    assert set(bodo.jit(impl4)(3).columns) == {"A", "B", "0", "1", "2"}
    with pytest.raises(
        BodoError,
        match="argument 'columns' requires a constant value but variable '.*' is updated inplace using 'append'",
    ):
        bodo.jit(impl5)(3, True)
    with pytest.raises(
        BodoError,
        match="argument 'columns' requires a constant value but variable '.*' is updated inplace using 'append'",
    ):
        bodo.jit(impl6)(30000)

    with pytest.raises(numba.errors.TypingError):
        bodo.jit(impl7)(30000)


def test_updated_container_df_rename():
    """Test updated container in df.rename() input for [BE-287]"""

    def impl():
        cols = ["1", "24", "124"]
        d = {"A": "B"}
        for c in cols:
            d[c] = "CE"
        df = pd.DataFrame({"A": np.random.randn(10)})
        new_df = df.rename(d, axis=1)
        return new_df

    with pytest.raises(
        BodoError,
        match=r"rename\(\): argument 'mapper' requires a constant value",
    ):
        bodo.jit(impl)()


def test_unroll_label_offset_bug():
    """Test for a bug in loop unrolling where block labels would clash"""

    def impl(df):
        df.columns = [
            f"quality_metrics_{c}" if c not in ["hex_id"] else c for c in df.columns
        ]
        return df

    df = pd.DataFrame({"A": [1, 2, 3], "hex_id": [4, 5, 6], "C": ["a", "b", "c"]})
    with pytest.raises(
        BodoError,
        match=r"dataframe columns requires constant names",
    ):
        bodo.jit(impl)(df)


def test_const_func_eval(memory_leak_check):
    """test evaluating jit functions in compile time for extracting constants"""

    @bodo.jit
    def g(df_cols):
        return [f"quality_metrics_{c}" if c not in ["hex_id"] else c for c in df_cols]

    def impl(df):
        df.columns = g(df.columns)
        return df

    df = pd.DataFrame({"A": [1, 2, 3], "hex_id": [4, 5, 6], "C": ["a", "b", "c"]})
    check_func(impl, (df,), copy_input=True, only_seq=True)


def test_pure_func(datapath):
    """Test determining pure functions.
    Tests a representative sample of various cases and code paths.
    """
    import time

    import h5py

    from bodo.utils.transform import _func_is_pure

    fname_pq = datapath("example.parquet")
    fname_h5 = datapath("lr.hdf5")
    fname_np = datapath("np_file1.dat")
    fname_csv = datapath("csv_data1.csv")

    # print
    def impl1():
        print("abc")

    # yield
    def impl2():
        for a in [1, 2, 3]:
            yield a

    # objmode
    def impl3():
        with bodo.objmode():
            impl1()

    # pq read
    def impl4():
        return pd.read_parquet(fname_pq)

    # np read
    def impl5():
        return np.fromfile(fname_np, np.float64)

    # np write
    def impl6():
        np.ones(3).tofile("example.dat")

    # pq write
    def impl7():
        df = pd.DataFrame({"A": [1, 2, 3]})
        df.to_parquet("example.pq")

    # h5py
    def impl8():
        f = h5py.File(fname_h5)
        X = f["points"][:, :]
        f.close()
        return X

    # random
    def impl9():
        return np.random.ranf(10)

    # time
    def impl10():
        return time.time()

    # csv read
    def impl11():
        return pd.read_csv(fname_csv)

    # csv write
    def impl12():
        df = pd.DataFrame({"A": [1, 2, 3]})
        df.to_csv("example.pq")

    # input list setitem
    def impl13(l):
        l[1] = 3

    # input dict update in place
    def impl14(d):
        d.pop(3)

    assert not _func_is_pure(impl1, (), {})
    assert not _func_is_pure(impl2, (), {})
    assert not _func_is_pure(impl3, (), {})
    assert not _func_is_pure(impl4, (), {})
    assert not _func_is_pure(impl5, (), {})
    assert not _func_is_pure(impl6, (), {})
    assert not _func_is_pure(impl7, (), {})
    assert not _func_is_pure(impl8, (), {})
    assert not _func_is_pure(impl9, (), {})
    assert not _func_is_pure(impl10, (), {})
    assert not _func_is_pure(impl11, (), {})
    assert not _func_is_pure(impl12, (), {})
    assert not _func_is_pure(impl13, (types.List(types.int64),), {})
    assert not _func_is_pure(impl14, (types.DictType(types.int64, types.int64),), {})


# default string type changes with _use_dict_str_type making type annotation invalid
@pytest.mark.skipif(
    bodo.hiframes.boxing._use_dict_str_type, reason="cannot test with dict string type"
)
def test_objmode_types():
    """
    Test creating types in JIT code and passing to objmode
    """

    def impl(A):
        with bodo.objmode(B=bodo.int64[::1]):
            B = 2 * A
        return B

    # complex type
    def impl2():
        with bodo.objmode(
            df=bodo.DataFrameType(
                (bodo.float64[::1], bodo.string_array_type),
                bodo.RangeIndexType(bodo.none),
                ("A", "B"),
            )
        ):
            df = pd.DataFrame({"A": [1.1, 2.2, 3.3], "B": ["A1", "A22", "A33"]})
        return df

    f = lambda A: (A.view("datetime64[ns]"), A.view("timedelta64[ns]"))

    def impl3(A):
        with bodo.objmode(B=bodo.datetime64ns[::1], C=bodo.timedelta64ns[::1]):
            B, C = f(A)
        return B, C

    check_func(impl, (np.arange(11),))
    check_func(impl2, (), only_seq=True)
    check_func(impl3, (np.arange(11),))


def test_enumerate_unituple(memory_leak_check):
    """Ensure the enumrate parallel code doesn't break on an enumerate
    that should produce a unituple."""

    def test_impl(iterable):
        val_sum = 0
        count_sum = 0
        for (
            i,
            val,
        ) in enumerate(iterable):
            val_sum += val
            count_sum += i
        return (val_sum, count_sum)

    t = tuple(np.arange(40))
    check_func(test_impl, (t,))


def test_dict_list(memory_leak_check):
    """make sure there is no unnecessary error when using lists inside dicts
    (Apple issue on email)
    """

    @bodo.jit
    def has_key(d, k):
        return k in d

    @bodo.jit
    def bang(d):
        if has_key(d, "numbers"):
            print("{}".format(d["numbers"]))
        for i in list(d["numbers"]):
            print(i)
        return 0

    bang({"numbers": list(range(0, 10))})


# TODO: Add memory_leak_check after memory leak is solved.
def test_literal_list_cast():
    """test when literal list needs to be cast to regular list"""

    # groupby forces 'a' to become a literal list, for loop forces cast to regular list
    def impl(df, a):
        df2 = df.groupby(a).sum()
        new_a = []
        for v in a:
            new_a.append(f"{v}_2")
        return df2.B.sum(), new_a

    df = pd.DataFrame({"A": [1, 2, 3], "B": [1, 2, 3]})
    check_func(impl, (df, ["A"]))


# TODO: Add memory_leak_check after memory leak is solved.
@pytest.mark.slow
def test_reversed():
    """
    test reversed on a list of floats
    """

    def test_impl(l):
        out = []
        for i in reversed(l):
            out.append(i)
        return out

    check_func(test_impl, ([0.1, 0.2, 0.3, 0.4],))


@pytest.mark.smoke
def test_jitclass(memory_leak_check):
    """test @bodo.jitclass decorator with various attribute/method cases"""

    @bodo.jitclass(
        {
            "df": bodo.hiframes.pd_dataframe_ext.DataFrameType(
                (bodo.int64[::1], bodo.float64[::1]),
                bodo.hiframes.pd_index_ext.RangeIndexType(numba.core.types.none),
                ("A", "B"),
            )
        },
        distributed=["df"],
    )
    class MyClass:
        def __init__(self, n):
            self.df = pd.DataFrame({"A": np.arange(n), "B": np.ones(n)})

        def sum1(self):
            return self.df.A.sum()

        @property
        def sum_vals(self):
            return self.df.sum().sum()

    n = 21
    # test using jitclass out of JIT context
    c = MyClass(n)
    res1 = np.arange(n).sum()
    res2 = np.arange(n).sum() + np.ones(n).sum()
    assert c.sum1() == res1
    assert c.sum_vals == res2

    # test using jitclass inside JIT context
    @bodo.jit
    def f1(n):
        c = MyClass(n)
        return c.sum1()

    @bodo.jit
    def f2(n):
        c = MyClass(n)
        return c.sum_vals

    assert f1(n) == res1
    assert f2(n) == res2

    # test jitclass errors
    @bodo.jitclass(
        {
            "df": bodo.hiframes.pd_dataframe_ext.DataFrameType(
                (bodo.float64[::1],),
                bodo.hiframes.pd_index_ext.RangeIndexType(numba.core.types.none),
                ("A",),
            )
        },
        distributed=["df"],
    )
    class MyClass1:
        def __init__(self):
            self.df = pd.DataFrame({"A": [1.2, 3.3]})

    with pytest.raises(BodoError, match="distribution of value is not compatible with"):
        MyClass1()

    @bodo.jitclass(
        {
            "df": bodo.hiframes.pd_dataframe_ext.DataFrameType(
                (bodo.float64[::1],),
                bodo.hiframes.pd_index_ext.RangeIndexType(numba.core.types.none),
                ("A",),
            )
        },
        distributed=["df"],
        returns_maybe_distributed=False,
    )
    class MyClass2:
        def __init__(self):
            self.df = pd.DataFrame({"A": np.ones(10)})

        def f1(self):
            return self.df.A

    with pytest.raises(BodoError, match="distribution of value is not compatible with"):
        c = MyClass2()
        c.f1()


@pytest.mark.slow
def test_missing_arg_msg():
    """
    test error raise when user didn't provide all required arguments
    in bodo.jit user-defined method
    """
    import numba

    @bodo.jit
    def test2(arg1):
        return arg1 + 2

    def test():
        return test2()

    # Save default developer mode value
    default_mode = numba.core.config.DEVELOPER_MODE

    # Test as a user
    numba.core.config.DEVELOPER_MODE = 0

    with pytest.raises(
        numba.core.errors.TypingError,
        match=r"missing a required argument",
    ):
        bodo.jit(test)()

    # Reset back to original setting
    numba.core.config.DEVELOPER_MODE = default_mode


@pytest.mark.skipif(bodo.get_size() < 2, reason="requires multiple ranks")
def test_print_empty_slice_multi_arg(capsys):
    """
    Test avoiding print of empty slices of distributed data,
    when the print() call has non-slice arguments also.
    """

    @bodo.jit(distributed=["df"])
    def test_impl(df):
        print("head: ", df.head())

    df = pd.DataFrame({"A": [1, 2, 3], "B": [1, 2, 3], "C": [1, 2, 3]})

    test_impl(df)
    out, _ = capsys.readouterr()

    assert "Empty DataFrame" not in out


def test_dict_scalar_to_array(memory_leak_check):
    """
    Tests the BodoSQL kernel scalar to array works as expected
    with various inputs.
    """
    arr_type = bodo.dict_str_arr_type

    def impl1(arg, len):
        return bodo.utils.conversion.coerce_scalar_to_array(arg, len, arr_type)

    def impl2(arg, len, flag):
        # Test optional type
        optional_arg = arg if flag else None
        return bodo.utils.conversion.coerce_scalar_to_array(optional_arg, len, arr_type)

    n = 50
    scalar_str = "testing"
    full_output = np.array(["testing"] * 50, dtype=object)
    null_output = np.array([None] * 50)

    check_func(impl1, (scalar_str, n), py_output=full_output)
    check_func(impl1, (None, n), py_output=null_output)
    check_func(impl2, (scalar_str, n, True), py_output=full_output)
    check_func(impl2, (scalar_str, n, False), py_output=null_output)


def test_int_scalar_to_array(memory_leak_check):
    """
    Tests that coerce_scalar_to_array keeps integers as non-nullable.
    """
    arr_type = bodo.IntegerArrayType(types.int64)

    def impl(arg, len):
        return bodo.utils.conversion.coerce_scalar_to_array(arg, len, arr_type)

    arg = 1
    n = 100
    py_output = np.array([arg] * n, dtype=np.int64)
    check_func(impl, (arg, n), py_output=py_output)


@pytest.mark.slow
def test_parfor_empty_entry_block(memory_leak_check):
    """make sure CFG simplification can handle empty entry block corner case properly.
    See BodoSQL/bodosql/tests/test_named_param_df_apply.py::test_case
    """
    out_arr_type = bodo.boolean_array

    @bodo.jit
    def impl(arrs, n, b, c, d, e, f, g):
        table1_A = arrs[0]
        numba.parfors.parfor.init_prange()
        out_arr = bodo.utils.utils.alloc_type(n, out_arr_type, (-1,))
        for i in numba.parfors.parfor.internal_prange(n):
            out_arr[i] = (
                b
                if b
                else (
                    False
                    if ((c > 12))
                    else (
                        False
                        if ((d == "hello"))
                        else (
                            False
                            if ((e == "hello"))
                            else (
                                False
                                if ((f > pd.Timestamp("2021-10-14 00:00:00")))
                                else (
                                    False
                                    if ((g > pd.Timestamp("2021-10-14 00:00:00")))
                                    else (
                                        (
                                            None
                                            if (pd.isna(table1_A[i]))
                                            else ((table1_A[i] > 1))
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        return out_arr

    A = np.arange(3)
    b = False
    c = 12
    d = "hello"
    e = "hello2"
    f = pd.Timestamp("2021-10-14 15:32:28.400163")
    g = pd.Timestamp("2019-03-04 11:12:28")
    np.testing.assert_array_equal(
        impl((A,), len(A), b, c, d, e, f, g), np.array([False, False, False])
    )
