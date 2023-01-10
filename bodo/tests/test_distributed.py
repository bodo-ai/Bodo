# Copyright (C) 2022 Bodo Inc. All rights reserved.
import datetime
import random
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from numba.core.ir_utils import find_callname, guard

import bodo
from bodo.tests.dataframe_common import df_value  # noqa
from bodo.tests.test_numpy_array import arr_tuple_val  # noqa
from bodo.tests.utils import (
    DistTestPipeline,
    SeqTestPipeline,
    _get_dist_arg,
    _test_equal,
    _test_equal_guard,
    check_func,
    count_array_OneD_Vars,
    count_array_OneDs,
    count_array_REPs,
    count_parfor_REPs,
    dist_IR_contains,
    gen_random_string_binary_array,
    get_start_end,
    reduce_sum,
)
from bodo.transforms.distributed_analysis import Distribution, is_REP
from bodo.utils.typing import BodoError, BodoWarning
from bodo.utils.utils import is_call_assign

random.seed(4)
np.random.seed(1)


@pytest.mark.slow
@pytest.mark.parametrize("A", [np.arange(11), np.arange(33).reshape(11, 3)])
def test_array_shape1(A, memory_leak_check):
    # get first dimention size using array.shape for distributed arrays
    def impl1(A):
        return A.shape[0]

    bodo_func = bodo.jit(distributed_block={"A"}, pipeline_class=DistTestPipeline)(
        impl1
    )
    start, end = get_start_end(len(A))
    assert bodo_func(A[start:end]) == impl1(A)
    assert count_array_REPs() == 0
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "dist_reduce")


@pytest.mark.slow
def test_array_shape2(memory_leak_check):
    # get first dimention size using array.shape for distributed arrays
    # transposed array case
    def impl1(A):
        B = A.T
        return B.shape[1]

    bodo_func = bodo.jit(distributed_block={"A"}, pipeline_class=DistTestPipeline)(
        impl1
    )
    n = 11
    A = np.arange(n * 3).reshape(n, 3)
    start, end = get_start_end(n)
    assert bodo_func(A[start:end]) == impl1(A)
    assert count_array_REPs() == 0
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "dist_reduce")
    # TODO: test Array.ctypes.shape[0] cases


@pytest.mark.slow
@pytest.mark.parametrize("A", [np.arange(11), np.arange(33).reshape(11, 3)])
def test_array_shape3(A, memory_leak_check):
    # get first dimention size using array.shape for distributed arrays
    def impl1(A):
        return A.shape

    bodo_func = bodo.jit(distributed_block={"A"}, pipeline_class=DistTestPipeline)(
        impl1
    )
    start, end = get_start_end(len(A))
    assert bodo_func(A[start:end]) == impl1(A)
    assert count_array_REPs() == 0
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "dist_reduce")


@pytest.mark.slow
def test_array_shape4(memory_leak_check):
    # transposed array case
    def impl1(A):
        B = A.T
        return B.shape

    bodo_func = bodo.jit(distributed_block={"A"}, pipeline_class=DistTestPipeline)(
        impl1
    )
    n = 11
    A = np.arange(n * 3).reshape(n, 3)
    start, end = get_start_end(n)
    assert bodo_func(A[start:end]) == impl1(A)
    assert count_array_REPs() == 0
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "dist_reduce")


@pytest.mark.slow
def test_array_len1(memory_leak_check):
    # get first dimention size using array.shape for distributed arrays
    def impl1(A):
        return len(A)

    bodo_func = bodo.jit(distributed_block={"A"}, pipeline_class=DistTestPipeline)(
        impl1
    )
    n = 11
    A = np.arange(n * 3).reshape(n, 3)
    start, end = get_start_end(n)
    assert bodo_func(A[start:end]) == impl1(A)
    assert count_array_REPs() == 0
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "dist_reduce")
    # TODO: tests with array created inside the function


@pytest.mark.slow
@pytest.mark.parametrize("A", [np.arange(11), np.arange(33).reshape(11, 3)])
def test_array_size1(A, memory_leak_check):
    def impl1(A):
        return A.size

    bodo_func = bodo.jit(distributed_block={"A"}, pipeline_class=DistTestPipeline)(
        impl1
    )
    start, end = get_start_end(len(A))
    assert bodo_func(A[start:end]) == impl1(A)
    assert count_array_REPs() == 0
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "dist_reduce")
    # TODO: tests with array created inside the function


@pytest.mark.parametrize("A", [np.ones(11), np.arange(33, dtype="int8").reshape(11, 3)])
def test_array_nbytes(A, memory_leak_check):
    def impl(A):
        return A.nbytes

    check_func(impl, (A,))


def test_concat_axis_1(memory_leak_check):
    """make sure concatenate with axis=1 is supported properly in distributed analysis"""

    def impl(A, B):
        C = np.concatenate((A, B), axis=1)
        return C

    bodo_func = bodo.jit(distributed_block={"A", "B", "C"})(impl)
    bodo_func(np.ones((3, 1)), np.zeros((3, 1)))
    # all arrays should be 1D, not 1D_Var
    assert count_array_REPs() == 0
    assert count_array_OneDs() > 0
    assert count_array_OneD_Vars() == 0


def test_1D_Var_parfor1(memory_leak_check):
    # 1D_Var parfor where index is used in computation
    def impl1(A, B):
        C = A[B != 0]
        s = 0
        for i in bodo.prange(len(C)):
            s += i + C[i]
        return s

    bodo_func = bodo.jit(distributed_block={"A", "B"})(impl1)
    A = np.arange(11)
    start, end = get_start_end(len(A))
    B = np.arange(len(A)) % 2
    assert bodo_func(A[start:end], B[start:end]) == impl1(A, B)
    assert count_array_REPs() == 0


def test_1D_Var_parfor2(memory_leak_check):
    # 1D_Var parfor where index is used in computation
    def impl1(A, B):
        C = A[B != 0]
        s = 0
        for i in bodo.prange(len(C)):
            s += i + C[i, 0]
        return s

    bodo_func = bodo.jit(distributed_block={"A", "B"})(impl1)
    A = np.arange(33).reshape(11, 3)
    start, end = get_start_end(len(A))
    B = np.arange(len(A)) % 2
    assert bodo_func(A[start:end], B[start:end]) == impl1(A, B)
    assert count_array_REPs() == 0


def test_1D_Var_parfor3(memory_leak_check):
    """test 1D parfor on length of an array that is assigned in an if/else block.
    Array analysis may not generate 'size_var = C.shape[0]' (keep 'len(C)').
    """

    def impl1(A, B, flag):
        if flag:
            C = A[B]
        else:
            C = A[~B]
        s = 0
        for j in range(3):
            for i in bodo.prange(len(C)):
                s += i + C[i, 0] + j
        return s

    bodo_func = bodo.jit(distributed_block={"A", "B"})(impl1)
    A = np.arange(33).reshape(11, 3)
    start, end = get_start_end(len(A))
    B = (np.arange(len(A)) % 2) != 0
    assert bodo_func(A[start:end], B[start:end], True) == impl1(A, B, True)
    assert count_array_REPs() == 0


def test_1D_Var_parfor4(memory_leak_check):
    """test 1D parfor inside a sequential loop"""

    def impl1(A, B):
        C = A[B]
        s = 0
        for j in range(3):
            for i in bodo.prange(len(C)):
                s += i + C[i, 0] + j
        return s

    bodo_func = bodo.jit(distributed_block={"A", "B"})(impl1)
    A = np.arange(33).reshape(11, 3)
    start, end = get_start_end(len(A))
    B = (np.arange(len(A)) % 2) != 0
    assert bodo_func(A[start:end], B[start:end]) == impl1(A, B)
    assert count_array_REPs() == 0


def test_jit_inside_prange(memory_leak_check):
    """test calling jit functions inside a prange loop"""

    @bodo.jit(distributed=False)
    def f(df):
        return df.sort_values("A")

    def impl(df, n):
        s = 3
        for i in bodo.prange(n):
            s += f(df).A.iloc[-1] + i
        return s

    df = pd.DataFrame({"A": [3, 1, 11, -3, 9, 1, 6]})
    n = 11
    assert bodo.jit(impl)(df, n) == impl(df, n)


def test_print1(memory_leak_check):
    # no vararg
    # TODO: capture stdout and make sure there is only one print
    def impl1(a, b):
        print(a, b)

    bodo_func = bodo.jit()(impl1)
    bodo_func(1, 2)
    bodo_func(np.ones(3), 3)
    bodo_func((3, 4), 2)


@pytest.mark.slow
def test_print2(memory_leak_check):
    # vararg
    # TODO: capture stdout and make sure there is only one print
    def impl1(a):
        print(*a)

    bodo_func = bodo.jit()(impl1)
    bodo_func((3, 4))
    bodo_func((3, np.ones(3)))


@pytest.mark.slow
def test_print3(memory_leak_check):
    # arg and vararg
    # TODO: capture stdout and make sure there is only one print
    def impl1(a, b):
        print(a, *b)

    bodo_func = bodo.jit()(impl1)
    bodo_func(1, (3, 4))
    bodo_func(np.ones(3), (3, np.ones(3)))


def test_print_empty_groupby(memory_leak_check, capsys):
    """make sure empty groups are not printed except possibly
    on rank 0."""

    def impl(df):
        df2 = df.groupby(["site_name"], as_index=False, dropna=False).size()
        print(df2)

    df = pd.DataFrame({"site_name": "only_group", "B": np.arange(1000)})
    bodo.jit(distributed=["A"])(impl)(df)
    captured = capsys.readouterr()
    if bodo.get_rank() != 0:
        assert "Empty" not in captured.out


def test_print_dist_slice(memory_leak_check, capsys):
    """make sure empty distributed slices are not printed"""

    def impl1(A, n):
        print(A[:n])

    def impl2(df):
        print(df.head())

    def impl3(A):
        print(A[::2])

    # array case
    bodo.jit(distributed=["A"])(impl1)(np.arange(111), 4)
    captured = capsys.readouterr()
    assert "[]" not in captured.out

    # dataframe case
    bodo.jit(distributed=["df"])(impl2)(pd.DataFrame({"A": np.arange(111)}))
    captured = capsys.readouterr()
    assert "Empty" not in captured.out

    # dataframe case, non-trivial Index
    bodo.jit(distributed=["df"])(impl2)(
        pd.DataFrame({"A": np.arange(111)}, index=np.arange(111) + 2)
    )
    captured = capsys.readouterr()
    assert "Empty" not in captured.out

    # series case
    bodo.jit(distributed=["df"])(impl2)(pd.Series(np.arange(111)))
    captured = capsys.readouterr()
    assert "[]" not in captured.out

    # series case, non-trival Index
    bodo.jit(distributed=["df"])(impl2)(
        pd.Series(np.arange(111), index=np.arange(111) + 2)
    )
    captured = capsys.readouterr()
    assert "[]" not in captured.out


def test_print_dist_get_rank(memory_leak_check, capsys):
    """make sure print calls with rank are not avoided on non-zero ranks
    (since REP input is not printed)
    """

    def impl():
        print("rank", bodo.get_rank())

    bodo.jit(impl)()
    captured = capsys.readouterr()
    assert captured.out


def test_range_index_1D_Var(memory_leak_check):
    """Test 1D_Var handling of RangeIndex. See [BE-2569]."""

    def impl(df):
        df = df.groupby("A", as_index=False).count()
        print(df.head())

    df = pd.DataFrame({"A": [3, 3, 3, 3, 3], "B": [1, 2, 3, 4, 5]})
    check_func(impl, (df,), only_1DVar=True)


def test_bodo_func_dist_call1(memory_leak_check):
    """make sure calling other bodo functions with their distributed flags set works as
    expected (dist info is propagated across functions).
    """

    @bodo.jit(distributed=["A", "C", "B"])
    def g(A, C, b=3):  # test default value
        B = 2 * A + b + C
        return B

    @bodo.jit(distributed=["Y"])
    def impl1(n):
        X = np.arange(n)
        Y = g(X, C=X + 1)  # call with both positional and kw args
        return Y

    # pass another bodo jit function as argument
    @bodo.jit(distributed=["Y"])
    def impl2(n, h):
        X = np.arange(n)
        Y = h(X, C=X + 1)  # call with both positional and kw args
        return Y

    impl1(11)
    assert count_array_REPs() == 0
    impl2(11, g)
    assert count_array_REPs() == 0


def test_bodo_func_dist_call_star_arg(memory_leak_check):
    """test calling other bodo functions with star arg set as distributed"""

    @bodo.jit(distributed=["A", "B"])
    def g(*A):
        B = A[0]
        return B

    @bodo.jit(distributed=["Y"])
    def impl1(n):
        X = np.arange(n)
        Y = g(X, X + 1)
        return Y

    impl1(11)
    assert count_array_REPs() == 0


def test_bodo_func_dist_call_tup(memory_leak_check):
    """make sure calling other bodo functions with their distributed flags set works
    when they return tuples.
    """

    @bodo.jit(distributed=["A", "B"])
    def f1(n):
        A = np.arange(n)
        B = np.ones(n)
        S = A, B
        return S

    @bodo.jit(distributed=["B"])
    def impl1(n):
        A, B = f1(n)
        return B

    impl1(11)
    assert count_array_REPs() == 0


def test_bodo_func_dist_call_tup2(memory_leak_check):
    """make sure calling other bodo functions with their distributed flags set works
    when they return tuples with only one distributed data structure.
    """

    # two return values are distributable, but one is distributed
    @bodo.jit(distributed=["B"], returns_maybe_distributed=False)
    def f1(n):
        A = np.arange(3)
        B = np.ones(n)
        S = A, B, 3
        return S

    @bodo.jit(distributed=["C"], returns_maybe_distributed=False)
    def impl1(n):
        A, B, a = f1(n)
        C = B + A[0] + a
        return C

    impl1(11)
    assert count_array_OneD_Vars() > 0

    # error checking case, caller's value can't be distributed
    @bodo.jit(distributed=["B", "A"], returns_maybe_distributed=False)
    def f2(n):
        A = np.arange(n)
        B = np.ones(n)
        S = A, B, 3
        return S

    @bodo.jit(returns_maybe_distributed=False)
    def impl2(n):
        A, B, a = f2(n)
        C = B + a
        return C

    with pytest.raises(
        BodoError,
        match=(
            r"is marked as distributed by 'f2' but not possible to distribute in caller function 'impl2'.\n"
            r"Distributed diagnostics:\nSetting distribution of variable"
        ),
    ):
        impl2(11)


def test_diag_for_return_error(memory_leak_check):
    """make sure the proper error is raised when calling other bodo functions with
    replicated args that conflict with distributed flag
    """

    @bodo.jit(distributed=["A"], returns_maybe_distributed=False)
    def f2(n):
        A = np.arange(n)
        return A

    @bodo.jit(returns_maybe_distributed=False)
    def impl(n):
        A = f2(n)
        return A

    with pytest.raises(
        BodoError,
        match=(
            r"is marked as distributed by 'f2' but not possible to distribute in caller function 'impl'.\n"
            r"Distributed diagnostics:\nSetting distribution of variable"
        ),
    ):
        impl(11)


def test_diag_code_loc(memory_leak_check):
    """make sure code location info is printed along with diagnostics info"""

    @bodo.jit(distributed=["S2"])
    def impl():
        S2 = pd.Series([1, 2, 3])
        return S2

    with pytest.raises(
        BodoError,
        match=(
            r"Tuples and lists are not distributed by default(?s:.*)pd.Series\(\[1, 2, 3\]\)"
        ),
    ):
        impl()


def test_dist_flag_warn1(memory_leak_check):
    """raise a warning when distributed flag is used for variables other than arguments
    and return values.
    """

    @bodo.jit(distributed=["A", "C", "B", "D"])
    def impl1(A, flag):
        B = 2 * A
        if flag:
            C = B + 1
            return C
        else:
            D = B + 2
            return D

    # same for replicated
    @bodo.jit(replicated=["A", "C", "B", "D"])
    def impl2(A, flag):
        B = 2 * A
        if flag:
            C = B + 1
            return C
        else:
            D = B + 2
            return D

    if bodo.get_rank() == 0:  # warning is thrown only on rank 0
        with pytest.warns(BodoWarning, match="Only function arguments and return"):
            impl1(np.arange(11), True)
    else:
        impl1(np.arange(11), True)
    assert count_array_REPs() == 0

    if bodo.get_rank() == 0:  # warning is thrown only on rank 0
        with pytest.warns(BodoWarning, match="Only function arguments and return"):
            impl2(np.arange(11), True)
    else:
        impl2(np.arange(11), True)


@pytest.mark.slow
@pytest.mark.filterwarnings("error:No parallelism")
def test_dist_flag_no_warn(memory_leak_check):
    """make sure there is no parallelism warning when there is no array or parfor"""

    def impl():
        return 0

    bodo.jit(impl)()


def test_dist_arg_diag1(memory_leak_check):
    """Make sure parallelism warning includes diagnostics info about replicated
    arguments
    """

    @bodo.jit
    def impl1(df):
        return df.A.sum()

    df = pd.DataFrame({"A": [1, 2, 3]})

    if bodo.get_rank() == 0:  # warning is thrown only on rank 0
        with pytest.warns(
            BodoWarning, match="Distributed analysis replicated argument 'df'"
        ):
            impl1(df)
    else:
        impl1(df)
    assert count_array_REPs() > 0


def test_dist_global_diag1(memory_leak_check):
    """Make sure parallelism warning includes diagnostics info about replicated
    global values
    """

    df = pd.DataFrame({"A": [1, 2, 3]})

    @bodo.jit
    def impl1():
        return df.A.sum()

    if bodo.get_rank() == 0:  # warning is thrown only on rank 0
        with pytest.warns(
            BodoWarning, match="Distributed analysis replicated global value 'df'"
        ):
            impl1()
    else:
        impl1()
    assert count_array_REPs() > 0


def test_dist_global_meta1(memory_leak_check):
    """Make sure Bodo parallelizes global value with Bodo distributed metadata"""

    @bodo.jit
    def f(n):
        return pd.DataFrame({"A": np.arange(n)})

    n = 12
    df = f(n)

    @bodo.jit
    def impl1():
        return df.A.sum()

    assert impl1() == np.arange(n).sum()
    assert count_array_REPs() == 0


def test_dist_global_flag1(memory_leak_check):
    """Make sure Bodo parallelizes global value with distributed flag"""

    n = 12
    df = pd.DataFrame({"A": np.arange(n)})
    df = bodo.scatterv(df if bodo.get_rank() == 0 else None)
    del df._bodo_meta

    @bodo.jit(distributed=["df"])
    def impl1():
        return df.A.sum()

    assert impl1() == np.arange(n).sum()
    assert count_array_REPs() == 0


def test_bodo_func_rep(memory_leak_check):
    """test calling other bodo functions without distributed flag"""

    @bodo.jit
    def g(A):
        return A

    @bodo.jit
    def impl1(n):
        X = np.arange(n)
        Y = g(X)
        return Y.sum()

    impl1(11)
    assert count_array_REPs() > 0


@pytest.mark.smoke
@pytest.mark.parametrize("A", [np.arange(11), np.arange(33).reshape(11, 3)])
def test_1D_Var_alloc_simple(A, memory_leak_check):
    # make sure 1D_Var alloc and parfor handling works for 1D/2D arrays
    def impl1(A, B):
        C = A[B]
        return C.sum()

    bodo_func = bodo.jit(distributed_block={"A", "B"})(impl1)
    start, end = get_start_end(len(A))
    B = np.arange(len(A)) % 2 != 0
    assert bodo_func(A[start:end], B[start:end]) == impl1(A, B)
    assert count_array_REPs() == 0


def test_1D_Var_alloc1(memory_leak_check):
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

    bodo_func = bodo.jit(distributed_block={"A", "B", "D"})(impl1)
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


def test_1D_Var_alloc2(memory_leak_check):
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

    bodo_func = bodo.jit(distributed_block={"A", "B", "D"})(impl1)
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
    # Memory leak occurs with test_1D_Var_alloc3 and Python 3.10.
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

    bodo_func = bodo.jit(distributed_block={"A", "B", "D"})(impl1)
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


def test_1D_Var_alloc4(memory_leak_check):
    """make sure allocation can match output's distribution with other arrays with same
    size. The arrays of "CC" columns should be assigned 1D_Var even though they don't
    interact directly with other arrays of their dataframes.
    """

    @bodo.jit(distributed=["df1", "df2", "df3"])
    def f(df1, df2):
        df1 = df1.rename(columns={"A": "B"})
        df1["CC"] = 11
        df2 = df2.rename(columns={"A": "B"})
        df2["CC"] = 1
        df3 = pd.concat([df1, df2])
        return df3

    df1 = pd.DataFrame({"A": [3, 4, 8]})
    df2 = pd.DataFrame({"A": [3, 4, 8]})
    f(df1, df2)
    assert count_array_REPs() == 0
    assert count_array_OneDs() == 0
    assert count_array_OneD_Vars() > 0


def test_str_alloc_equiv1(memory_leak_check):
    def impl(n):
        C = bodo.libs.str_arr_ext.pre_alloc_string_array(n, 10)
        return len(C)

    bodo_func = bodo.jit(pipeline_class=DistTestPipeline)(impl)
    n = 11
    assert bodo_func(n) == n
    assert count_array_REPs() == 0
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert not dist_IR_contains(f_ir, "dist_reduce")


def test_dist_list_mul_concat(memory_leak_check):
    """
    Test that [df] * n can return a distributed output via pd.concat
    """

    def impl1(df, n):
        return pd.concat([df] * n)

    def impl2(df, n):
        return pd.concat(n * [df])

    df = pd.DataFrame(
        {
            "A": np.arange(1000),
            "B": np.arange(1000, 2000),
        }
    )
    # pd.concat concatenates locally on each rank, so reset the index
    # and sort.
    check_func(impl1, (df, 5), sort_output=True, reset_index=True)
    check_func(impl2, (df, 5), sort_output=True, reset_index=True)


def test_series_alloc_equiv1(memory_leak_check):
    def impl(n):
        if n < 10:
            S = pd.Series(np.ones(n))
        else:
            S = pd.Series(np.zeros(n))
        # B = np.full(len(S), 2)  # TODO: np.full dist handling
        B = np.empty(len(S))
        return B

    bodo_func = bodo.jit(distributed_block={"B"}, pipeline_class=DistTestPipeline)(impl)
    n = 11
    bodo_func(n)
    assert count_parfor_REPs() == 0
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert not dist_IR_contains(f_ir, "dist_reduce")


def test_arange_equiv():
    """make sure array size equivalence is set properly for arange() call"""

    def test(idx):
        ans = pd.Series(np.arange(20), index=idx)
        return ans

    arr = np.array([1, 4] * 10)
    check_func(test, (arr,), only_1DVar=True)


# TODO: test other array types
@pytest.mark.parametrize(
    "A",
    [
        np.arange(11),
        np.arange(33).reshape(11, 3),
        pd.Series(["aa", "bb", "c"] * 4),
        pd.RangeIndex(11, 19, 3),
        pd.Series(1, pd.RangeIndex(11, 19, 3)),
    ],
)
@pytest.mark.parametrize(
    "s", [slice(3), slice(1, 9), slice(7, None), slice(4, 6), slice(-3, None)]
)
def test_getitem_slice(A, s, memory_leak_check):
    # get a slice of 1D/1D_Var array
    def impl1(A, s):
        return A[s]

    check_func(impl1, (A, s), check_typing_issues=False)


# TODO: np.arange(33).reshape(11, 3)
@pytest.mark.parametrize(
    "A", [pd.Series(np.arange(11)), pd.Series(["aafa", "bbac", "cff"] * 4)]
)
@pytest.mark.parametrize("s", [0, 1, 3, 7, 10, -1, -2])
def test_getitem_int_1D(A, s, memory_leak_check):
    # get a single value of 1D_Block array
    def impl1(A, s):
        return A.values[s]

    bodo_func = bodo.jit(distributed_block={"A"})(impl1)
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
def test_getitem_int_1D_Var(A, s, memory_leak_check):
    # get a single value of 1D_Block array
    def impl1(A, B, s):
        C = A.values[B]
        return C[s]

    bodo_func = bodo.jit(distributed_block={"A", "B"})(impl1)
    start, end = get_start_end(len(A))
    B = np.arange(len(A)) % 2 != 0
    if A.ndim == 1:
        assert bodo_func(A[start:end], B[start:end], s) == impl1(A, B, s)
    else:
        np.testing.assert_array_equal(
            bodo_func(A[start:end], B[start:end], s), impl1(A, B, s)
        )
    assert count_array_OneD_Vars() > 0


def test_getitem_const_slice_multidim(memory_leak_check):
    """test getitem of multi-dim distributed array with a constant slice in first
    dimension.
    """

    def impl(A):
        return A[1:3, 0, 1:]

    n = 5
    A = np.arange(n * n * n).reshape(n, n, n)
    check_func(impl, (A,))


def test_getitem_slice_const_size(memory_leak_check):
    """test getitem of multi-dim distributed array with a constant slice in first
    dimension.
    """
    # setitem without stride
    def impl1():
        N = 10
        X = np.ones((N, 3))
        X[:, 1] = 3
        return X

    # getitem without stride
    def impl2():
        N = 10
        X = np.ones((N, 3))
        A = X[:, 1]
        return A

    # TODO: support
    # setitem with stride
    # def impl3():
    #     N = 10
    #     X = np.ones((N, 3))
    #     X[::2,1] = 3
    #     return X.sum()

    # getitem with stride
    # def impl4():
    #     N = 10
    #     X = np.ones((N, 3))
    #     A = X[::2,1]
    #     return A.sum()

    check_func(impl1, ())
    check_func(impl2, ())
    # check_func(impl3, ())
    # check_func(impl4, ())


def test_setitem_slice_scalar(memory_leak_check):
    """test setitem of distributed array with a scalar or lower dimention array value"""

    def impl(A, val):
        A[4:-3:2] = val
        return A

    # scalar value
    A = np.arange(11)
    val = -1
    check_func(impl, (A, val))

    # multi-dim array with lower dimension array value
    # using a new implementation since Numba doesn't support lists in array setitem
    def impl2(A, val):
        A[::2] = np.array(val)
        return A

    A = np.arange(33).reshape(11, 3)
    val = [-1, -3, -2]
    check_func(impl2, (A, val))


def test_setitem_bool_index_scalar(memory_leak_check):
    """test setting a scalar or lower dimension array value to distributed array
    positions selected by a boolean index
    """

    def impl(A, I, val):
        A[I] = val
        return A

    # scalar value
    A = np.arange(11)
    I = A % 4 == 0
    val = -1
    check_func(impl, (A, I, val))

    # multi-dim array with scalar value
    # TODO: support 2D bool indexing in Numba
    # A = np.arange(33).reshape(11, 3)
    # I = A % 4 == 0
    # val = -1
    # check_func(impl, (A, I, val))

    # multi-dim array with lower dimension array value
    # using a new implementation since Numba doesn't support lists in array setitem
    def impl2(A, I, val):
        A[I] = np.array(val)
        return A

    A = np.arange(33).reshape(11, 3)
    I = A[:, 0] % 4 == 0
    val = [-1, -3, -2]
    check_func(impl2, (A, I, val))


@pytest.mark.smoke
def test_setitem_scalar(memory_leak_check):
    """test setitem of distributed array with a scalar"""

    def impl(A, val):
        A[1] = val
        return A

    # scalar value
    A = np.arange(11)
    val = -1
    check_func(impl, (A, val))

    # multi-dim array with lower dimension array value
    # using a new implementation since Numba doesn't support lists in array setitem
    def impl2(A, val, i):
        A[i] = np.array(val)
        return A

    A = np.arange(33).reshape(11, 3)
    val = [-1, -3, -2]
    check_func(impl2, (A, val, -1))


@pytest.mark.parametrize("dtype", [np.float32, np.uint8, np.int64])
def test_arr_reshape(dtype, memory_leak_check):
    """test reshape of multi-dim distributed arrays"""
    # reshape to more dimensions
    def impl1(A, n):
        return A.reshape(3, n // 3)

    # reshape to more dimensions with tuple input
    def impl2(A, n):
        return A.reshape((3, n // 3))

    # reshape to more dimensions np call
    def impl3(A, n):
        return np.reshape(A, (3, n // 3))

    # reshape to fewer dimensions
    def impl4(A, n):
        return A.reshape(3, n // 3)

    # reshape to 1 dimension
    def impl5(A, n):
        return A.reshape(n)

    # reshape to same dimensions (no effect)
    def impl6(A, n):
        return A.reshape(3, n // 3)

    # reshape to same dimensions (no effect)
    def impl7(A, n):
        return A.reshape(3, 2, n // 6)

    # reshape to add 1 extra dimension to 1-dim array
    def impl8(A, n):
        return A.reshape(n, 1)

    A = np.arange(12, dtype=dtype)
    check_func(impl1, (A, 12))
    check_func(impl2, (A, 12))
    check_func(impl3, (A, 12))
    A = np.arange(12, dtype=dtype).reshape(2, 3, 2)
    check_func(impl4, (A, 12))
    check_func(impl5, (A, 12))
    A = np.arange(12, dtype=dtype).reshape(3, 4)
    check_func(impl6, (A, 12))
    check_func(impl7, (A, 12))
    check_func(impl8, (np.arange(12, dtype=dtype), 12))


def test_np_dot(is_slow_run, memory_leak_check):
    """test np.dot() distribute transform"""

    # reduction across rows, input: (1D dist array, 2D dist array)
    def impl1(X, Y):
        return np.dot(Y, X)

    # reduction across rows, input: (2D dist array, 1D REP array)
    def impl2(X, d):
        w = np.arange(0, d, 1, np.float64)
        return np.dot(X, w)

    # using the @ operator
    def impl3(X, d):
        w = np.arange(0, d, 1, np.float64)
        return X @ w

    # using the @ operator
    def impl4(Y):
        w = np.arange(0, len(Y), 1, np.float64)
        return w @ Y

    n = 11
    d = 3
    np.random.seed(1)
    X = np.random.ranf((n, d))
    Y = np.arange(n, dtype=np.float64)
    check_func(impl1, (X, Y), is_out_distributed=False)
    check_func(impl2, (X, d))
    check_func(impl3, (X, d))
    if is_slow_run:
        check_func(impl4, (Y,))


def test_dist_tuple1(memory_leak_check):
    def impl1(A):
        B1, B2 = A
        return (B1 + B2).sum()

    n = 11
    A = (np.arange(n), np.ones(n))
    start, end = get_start_end(n)
    A_par = (A[0][start:end], A[1][start:end])
    bodo_func = bodo.jit(distributed_block={"A"})(impl1)
    assert bodo_func(A_par) == impl1(A)
    assert count_array_OneDs() > 0


def test_dist_tuple2(memory_leak_check):
    # TODO: tuple getitem with variable index
    def impl1(A, B):
        C = (A, B)
        return C

    n = 11
    A = np.arange(n)
    B = np.ones(n)
    start, end = get_start_end(n)
    bodo_func = bodo.jit(distributed_block={"A", "B", "C"})(impl1)

    py_out = impl1(A, B)
    bodo_out = bodo_func(A[start:end], B[start:end])
    np.testing.assert_array_equal(bodo_out[0], py_out[0][start:end])
    np.testing.assert_array_equal(bodo_out[1], py_out[1][start:end])
    assert count_array_OneDs() > 0


def test_dist_tuple3(memory_leak_check):
    """Make sure passing a dist tuple with non-dist elements doesn't cause REP"""

    def impl1(v):
        (_, df) = v
        return df

    n = 11
    df = pd.DataFrame({"A": np.arange(n)})
    v = (n, df)
    bodo.jit(distributed_block={"v", "df"})(impl1)(v)
    assert count_array_OneDs() > 0


def test_dist_list1(memory_leak_check):
    """Test support for build_list of dist data"""

    def impl1(df):
        v = [(1, df)]
        return v

    n = 11
    df = pd.DataFrame({"A": np.arange(n)})
    bodo.jit(distributed_block={"v", "df"})(impl1)(df)
    assert count_array_OneDs() > 0


def test_dist_list_append1(memory_leak_check):
    """Test support for list.append of dist tuple"""

    def impl1(df):
        v = [(1, df)]
        v.append((1, df))
        return v

    n = 11
    df = pd.DataFrame({"A": np.arange(n)})
    bodo.jit(distributed_block={"v", "df"})(impl1)(df)
    assert count_array_OneDs() > 0


def test_dist_list_append2(memory_leak_check):
    """Test support for list.append of dist data"""

    def impl1(df):
        v = [df]
        v.append(df)
        return v

    n = 11
    df = pd.DataFrame({"A": np.arange(n)})
    bodo.jit(distributed_block={"v", "df"})(impl1)(df)
    assert count_array_OneDs() > 0


def test_dist_list_getitem1(memory_leak_check):
    """Test support for getitem of distributed list"""

    def impl1(v):
        df = v[1]
        return df

    n = 11
    df = pd.DataFrame({"A": np.arange(n)})
    v = [df, df]
    bodo.jit(distributed_block={"v", "df"})(impl1)(v)
    assert count_array_OneDs() > 0


def test_dist_list_loop(memory_leak_check):
    """Test support for loop over distributed list"""

    def impl1(v):
        s = 0
        for df in v:
            s += df.A.sum()
        return s

    n = 11
    df = pd.DataFrame({"A": np.arange(n)})
    df_chunk = _get_dist_arg(df)
    v = [df, df]
    v_chunks = [df_chunk, df_chunk]
    assert bodo.jit(distributed={"v", "df"})(impl1)(v_chunks) == impl1(v)
    assert count_array_OneD_Vars() > 0


def test_dist_list_setitem1(memory_leak_check):
    """Test support for setitem of distributed list"""

    def impl1(v, df):
        v[1] = df

    n = 11
    df = pd.DataFrame({"A": np.arange(n)})
    v = [df, df]
    bodo.jit(distributed_block={"v", "df"})(impl1)(v, df)
    assert count_array_OneDs() >= 2


def test_dist_list_loop_concat(memory_leak_check):
    """Test support for list of dist data used with a loop and concat"""

    def impl(df):
        dfs = []
        for _ in range(3):
            dfs.append(df)
        output = pd.concat(dfs)
        return output

    n = 11
    df = pd.DataFrame({"A": np.arange(n)})
    check_func(impl, (df,), sort_output=True)


def test_dist_dict1(memory_leak_check):
    """Test support for build_map of dist data"""

    def impl1(df):
        v = {1: df}
        return v

    n = 11
    df = pd.DataFrame({"A": np.arange(n)})
    bodo.jit(distributed_block={"v", "df"})(impl1)(df)
    assert count_array_OneDs() > 0


def test_dist_dict_getitem1(memory_leak_check):
    """Test support for getitem of dist dictionary"""

    def impl1(v):
        df = v[1]
        return df

    n = 11
    df = pd.DataFrame({"A": np.arange(n)})
    v = bodo.typed.Dict.empty(bodo.int64, bodo.typeof(df))
    v[0] = df
    v[1] = df
    bodo.jit(distributed_block={"v", "df"})(impl1)(v)
    assert count_array_OneDs() > 0


def test_dist_dict_setitem1(memory_leak_check):
    """Test support for setitem of dist dictionary"""

    def impl1(v, df):
        v[1] = df

    n = 11
    df = pd.DataFrame({"A": np.arange(n)})
    v = bodo.typed.Dict.empty(bodo.int64, bodo.typeof(df))
    v[0] = df
    v[1] = df
    bodo.jit(distributed_block={"v", "df"})(impl1)(v, df)
    assert count_array_OneDs() >= 2


def test_return_maybe_dist(memory_leak_check):
    """test returns_maybe_distributed jit flag"""
    # tuple return
    def impl1(n):
        df = pd.DataFrame({"A": np.arange(n), "B": np.ones(n)})
        return df, df.sum()

    # non-tuple return
    def impl2(n):
        df = pd.DataFrame({"A": np.arange(n), "B": np.ones(n)})
        return df

    # call another function with returns_maybe_distributed
    f = bodo.jit(returns_maybe_distributed=True)(impl1)

    def impl3(n):
        df = f(n)[0]
        return df

    n = 11
    bodo_func = bodo.jit(returns_maybe_distributed=True)(impl1)
    bodo_ret = bodo_func(n)
    pd_ret = impl1(n)
    assert _test_equal_guard(bodo.allgatherv(bodo_ret[0]), pd_ret[0])
    assert _test_equal_guard(bodo_ret[1], pd_ret[1])
    assert bodo_func.overloads[bodo_func.signatures[0]].metadata[
        "is_return_distributed"
    ] == [True, False]

    n = 11
    bodo_func = bodo.jit(returns_maybe_distributed=True)(impl2)
    assert _test_equal_guard(bodo.allgatherv(bodo_func(n)), impl2(n))
    assert (
        bodo_func.overloads[bodo_func.signatures[0]].metadata["is_return_distributed"]
        == True
    )

    n = 11
    bodo_func = bodo.jit(distributed=["df"])(impl3)
    pd_ret = pd.DataFrame({"A": np.arange(n), "B": np.ones(n)})
    assert _test_equal_guard(bodo.allgatherv(bodo_func(n)), pd_ret)
    assert (
        bodo_func.overloads[bodo_func.signatures[0]].metadata["is_return_distributed"]
        == True
    )


# TODO: Add memory_leak_check when bug is solved
def test_concat_reduction():
    """test dataframe concat reduction, which produces distributed output"""

    def impl(n):
        df = pd.DataFrame()
        for i in bodo.prange(n):
            df = df.append(pd.DataFrame({"A": np.arange(i)}))

        return df

    check_func(
        impl, (11,), reset_index=True, check_dtype=False, is_out_distributed=True
    )


def test_series_concat_reduction():
    """test Series concat reduction, which produces distributed output"""

    def impl(n):
        S = pd.Series(np.empty(0, np.int64))
        for i in bodo.prange(n):
            S = S.append(pd.Series(np.arange(i)))

        return S

    check_func(
        impl, (11,), reset_index=True, check_dtype=False, is_out_distributed=True
    )


def test_dist_warning1(memory_leak_check):
    """Make sure BodoWarning is thrown when there is no parallelism discovered due
    to unsupported function
    """

    @bodo.jit
    def f(n):
        A = np.ones((n, n))
        # using a function we are not likely to support for warning test
        # should be changed when/if we support slogdet()
        return np.linalg.slogdet(A)

    if bodo.get_rank() == 0:  # warning is thrown only on rank 0
        with pytest.warns(
            BodoWarning,
            match=r"No parallelism found for function(?s:.*)np.linalg.slogdet",
        ):
            f(10)
    else:
        f(10)

    # make sure diag info doesn't have repetition [BE-1493]
    assert (
        len(f.overloads[f.signatures[0]].metadata["distributed_diagnostics"].diag_info)
        == 1
    )


@pytest.mark.slow
def test_dist_warning2(memory_leak_check):
    """Make sure BodoWarning is thrown when there is no parallelism discovered due
    to return of dataframe
    """

    def impl(n):
        return pd.DataFrame({"A": np.ones(n)})

    if bodo.get_rank() == 0:  # warning is thrown only on rank 0
        with pytest.warns(BodoWarning, match="No parallelism found for function"):
            bodo.jit(returns_maybe_distributed=False)(impl)(10)
    else:
        bodo.jit(returns_maybe_distributed=False)(impl)(10)


@pytest.mark.slow
def test_dist_warning3(memory_leak_check):
    """Make sure BodoWarning is thrown when a tuple variable with both distributable
    and non-distributable elemets is returned
    """

    def impl(n):
        df = pd.DataFrame({"A": np.ones(n)})
        return (n, df)

    if bodo.get_rank() == 0:  # warning is thrown only on rank 0
        with pytest.warns(BodoWarning, match="No parallelism found for function"):
            bodo.jit(returns_maybe_distributed=False)(impl)(10)
    else:
        bodo.jit(returns_maybe_distributed=False)(impl)(10)


def test_getitem_bool_REP(memory_leak_check):
    """make sure output of array getitem with bool index can make its inputs REP"""

    def test_impl(n):
        df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) + 3})
        df = df[df.A != 0]
        return df

    n = 11
    if bodo.get_rank() == 0:  # warning is thrown only on rank 0
        with pytest.warns(BodoWarning, match="No parallelism found for function"):
            bodo.jit(returns_maybe_distributed=False)(test_impl)(n)
    else:
        bodo.jit(returns_maybe_distributed=False)(test_impl)(n)


def test_df_filter_branch(memory_leak_check):
    """branches can cause array analysis to remove size equivalences for some array
    definitions since the array analysis pass is not proper data flow yet. However,
    1D_Var size adjustment needs to find the array to get the local size so it tries
    pattern matching for definition of the size. This test (from customer code)
    exercises this case.
    """

    def test_impl(df, flag):
        df2 = df[df.A == 1]
        if flag:
            todelete = np.zeros(len(df2), np.bool_)
            todelete = np.where(df2.A != 2, True, todelete)
            df2 = df2[~todelete]

        df2 = df2[df2.A == 3]
        return df2

    df = pd.DataFrame({"A": [1, 11, 2, 0, 3]})
    check_func(test_impl, (df, True))


def test_empty_object_array_warning(memory_leak_check):
    """Make sure BodoWarning is thrown when there is an empty object array in input"""

    def impl(A):
        return A

    with pytest.warns(BodoWarning, match="Empty object array passed to Bodo"):
        bodo.jit(impl)(np.array([], dtype=object))
    with pytest.warns(BodoWarning, match="Empty object array passed to Bodo"):
        bodo.jit(impl)(pd.Series(np.array([], dtype=object)))
    with pytest.warns(BodoWarning, match="Field value in struct array is NA"):
        bodo.jit(impl)(
            pd.Series(np.array([{"A": None, "B": 2.2}, {"A": "CC", "B": 1.2}]))
        )


def test_dist_flags(memory_leak_check):
    """Make sure Bodo flags are preserved when the same Dispatcher that has distributed
    flags is called with different data types, triggering multiple compilations.
    See #357
    """

    def impl(A):
        return A.sum()

    n = 50
    A = np.arange(n)
    bodo_func = bodo.jit(all_args_distributed_block=True)(impl)
    result_bodo = bodo_func(_get_dist_arg(A, False))
    result_python = impl(A)
    if bodo.get_rank() == 0:
        _test_equal(result_bodo, result_python)

    A = np.arange(n, dtype=np.float64)  # change dtype to trigger compilation again
    result_bodo = bodo_func(_get_dist_arg(A, False))
    result_python = impl(A)
    if bodo.get_rank() == 0:
        _test_equal(result_bodo, result_python)


def test_user_distributed_rep(memory_leak_check):
    """Ensures that if user species an argument as distributed
    but it must be replicated then it throws an error.
    See [BE-508]
    """

    def impl(arr):
        return list(arr)

    arr = np.arange(50)
    arr_chunk = _get_dist_arg(arr, False)
    err_msg = "Variable 'arr' has distributed flag in function 'impl', but it's not possible to distribute it."
    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl, distributed=["arr"])(arr_chunk)
    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl, distributed_block=["arr"])(arr_chunk)
    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl, all_args_distributed_varlength=True)(arr_chunk)
    with pytest.raises(
        BodoError,
        match=err_msg,
    ):
        bodo.jit(impl, all_args_distributed_block=True)(arr_chunk)


def test_replicated_flag(memory_leak_check):
    """test replicated flag in jit decorator"""

    # mix of dist/rep inputs
    @bodo.jit(distributed=["df1"], replicated=["df2"])
    def impl1(df1, df2):
        return df1.A.sum() + df2.A.sum()

    # replicated=True for whole function
    @bodo.jit(replicated=True)
    def impl2():
        return np.arange(50)

    # mix of dist/rep outputs
    @bodo.jit(distributed=["df1"], replicated=["df2"])
    def impl3():
        df1 = pd.DataFrame({"A": np.arange(50)})
        df2 = pd.DataFrame({"A": np.arange(30)})
        return df1, df2

    # single rep output
    @bodo.jit(replicated=["df1"])
    def impl4():
        df1 = pd.DataFrame({"A": np.arange(50)})
        return df1

    # nested JIT calls
    @bodo.jit(replicated=["df1"])
    def f(df1):
        return df1.A.sum()

    @bodo.jit
    def g():
        df = pd.DataFrame({"A": np.arange(50)})
        return f(df)

    df1 = pd.DataFrame({"A": np.arange(50)})
    # use scatterv so that bodo meta is set to distributed
    df1_chunk = bodo.scatterv(df1)
    df2 = pd.DataFrame({"A": np.arange(30) ** 2})
    impl1(df1_chunk, df2)
    arr_dists = (
        impl1.overloads[impl1.signatures[0]]
        .metadata["distributed_diagnostics"]
        .array_dists
    )
    assert (
        arr_dists["df1"] == Distribution.OneD_Var
        and arr_dists["df2"] == Distribution.REP
    ), "df1 should be distributed and df2 replicated based on user flags"
    impl1._reset_overloads()
    impl1(df2, df1_chunk)
    arr_dists = (
        impl1.overloads[impl1.signatures[0]]
        .metadata["distributed_diagnostics"]
        .array_dists
    )
    assert (
        arr_dists["df1"] == Distribution.OneD_Var
        and arr_dists["df2"] == Distribution.REP
    ), "df1 should be distributed and df2 replicated based on user flags"

    impl2()
    assert not impl2.overloads[impl2.signatures[0]].metadata[
        "is_return_distributed"
    ], "output of impl2 should be replicated since replicated=True is set"

    impl3()
    assert impl3.overloads[impl3.signatures[0]].metadata["is_return_distributed"] == [
        True,
        False,
    ], "df1 should be distributed and df2 replicated based on user flags"
    impl4()
    assert (
        impl4.overloads[impl4.signatures[0]].metadata["is_return_distributed"] == False
    ), "output of impl4 should be replicated based on user flags"

    g()
    arr_dists = (
        g.overloads[g.signatures[0]].metadata["distributed_diagnostics"].array_dists
    )
    for v in arr_dists.values():
        assert is_REP(v), "variables in g() should be REP based on user flags for f()"


def test_replicated_error_check(memory_leak_check):
    """make sure Bodo raises an error if a variable is marked as both distributed and replicated"""

    @bodo.jit(distributed=["S"], replicated=["S"])
    def impl(S):
        return S.sum()

    with pytest.raises(
        BodoError,
        match=(r"marked as both 'distributed' and 'replicated'"),
    ):
        impl(pd.Series([1, 2, 3]))


@pytest.mark.slow
def test_dist_objmode(memory_leak_check):
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

    np.testing.assert_allclose(bodo.jit(objmode_test)(10), objmode_test(10))


@pytest.mark.slow
def test_unique_allgatherv(memory_leak_check):
    """make sure allgatherv of unique is removed in sequential compiler pipline"""

    def impl(S):
        return S.unique()

    bodo_func = bodo.jit(pipeline_class=SeqTestPipeline)(impl)
    bodo_func(pd.Series([1, 2, 3, 1]))
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    # check for gatherv call in the IR
    block = next(iter(f_ir.blocks.values()))
    for stmt in block.body:
        assert not (
            is_call_assign(stmt)
            and guard(find_callname, f_ir, stmt.value)
            in (("gatherv", "bodo"), ("allgatherv", "bodo"))
        )


def test_dist_objmode_dist(memory_leak_check):
    """make sure output chunks from objmode are assigned 1D_Var distribution"""

    def impl(n):
        A = np.arange(n)
        with bodo.objmode(B="int64[:]"):
            B = A[:3]
        return B

    n = 111
    res = bodo.jit(all_returns_distributed=True)(impl)(n)
    assert len(res) == 3
    assert count_array_OneD_Vars() > 0


@pytest.mark.slow
def test_diagnostics_not_compiled_error(memory_leak_check):
    """make sure error is thrown when calling diagnostics for a function that is not
    compiled yet
    """

    def test_impl():
        return np.arange(10).sum()

    with pytest.raises(BodoError, match="Distributed diagnostics not available for"):
        bodo.jit(test_impl).distributed_diagnostics()


@pytest.mark.slow
def test_diagnostics_trace(capsys, memory_leak_check):
    """make sure distributed diagnostics trace info is printed in diagnostics dump"""

    @bodo.jit(args_maybe_distributed=False)
    def f(A):
        return A.sum()

    @bodo.jit
    def g():
        return f(np.arange(10))

    g()
    g.distributed_diagnostics()
    if bodo.get_rank() == 0:
        assert (
            "input of another Bodo call without distributed flag"
            in capsys.readouterr().out
        )


def test_diagnostics_list(capsys, memory_leak_check):
    """make sure distributed diagnostics prints lists of distributions properly"""

    @bodo.jit(distributed=["A", "B"])
    def f(A, B):
        return (A, B)

    f(np.ones(3), np.arange(3))
    f.distributed_diagnostics()
    if bodo.get_rank() == 0:
        assert "[1D_Block_Var, 1D_Block_Var]" in capsys.readouterr().out


def test_sort_output_1D_Var_size(memory_leak_check):
    """Test using size variable of an output 1D_Var array of a Sort node"""
    # RangeIndex of output Series needs size of Sort output array
    def impl(S):
        res = pd.Series(S.sort_values().values)
        return res

    S = pd.Series([3, 4, 1, 2, 5])
    check_func(impl, (S,))


def test_df_1D_Var_col_set_string(memory_leak_check):
    """Test setting a new column of a 1D_Var dataframe to a string value, making sure
    the number of characters for the local string array is correct
    """

    def impl(df):
        df["B"] = "A"
        # Scalar strings are dictionary encoded so we check the number of
        # indices.
        return len(df.B.values._data), len(df.B.values._indices)

    n = 11
    df = pd.DataFrame({"A": np.arange(n)})
    df_chunk = _get_dist_arg(df, var_length=True)
    dict_len, indices_len = bodo.jit(distributed={"df"})(impl)(df_chunk)
    assert dict_len == 1
    assert reduce_sum(indices_len) == n
    assert count_array_OneD_Vars() > 0


def check_dist_meta(df, dist):
    """return True if 'df' has Bodo metadata with same distribution as 'dist'"""
    return (
        hasattr(df, "_bodo_meta")
        and "dist" in df._bodo_meta
        and df._bodo_meta["dist"] == dist.value
    )


def test_bodo_meta(memory_leak_check, datapath):
    """Test Bodo metadata on data structures returned from JIT functions"""
    fname = datapath("example.parquet")

    # df created inside JIT function
    @bodo.jit
    def impl1(fname):
        df = pd.read_parquet(fname)
        return df

    # df passed into JIT and returned
    @bodo.jit(distributed=["df"])
    def impl2(df):
        return df

    # df passed into JIT with dist meta
    @bodo.jit
    def impl3(df):
        return df

    # Series created inside JIT function
    @bodo.jit
    def impl4(fname):
        df = pd.read_parquet(fname)
        return df.one

    out_df1 = impl1(fname)
    assert count_array_OneDs() > 0
    check_dist_meta(out_df1, Distribution.OneD)

    out_df2 = impl2(pd.DataFrame({"A": np.arange(11)}))
    assert count_array_OneD_Vars() > 0
    check_dist_meta(out_df2, Distribution.OneD_Var)

    out_df3 = impl3(out_df1)
    assert count_array_OneDs() > 0
    check_dist_meta(out_df3, Distribution.OneD)

    # series input/output
    out_S1 = impl4(fname)
    assert count_array_OneDs() > 0
    check_dist_meta(out_S1, Distribution.OneD)

    out_S2 = impl2(pd.Series(np.arange(11)))
    assert count_array_OneD_Vars() > 0
    check_dist_meta(out_S2, Distribution.OneD_Var)

    out_S3 = impl3(out_S1)
    assert count_array_OneDs() > 0
    check_dist_meta(out_S3, Distribution.OneD)


def test_bodo_meta_jit_calls(memory_leak_check):
    """Test automatic distribution detection across JIT calls"""

    # dist argument is passed but output is replicated
    @bodo.jit
    def g(df):
        return df.sum()

    @bodo.jit
    def impl1(n):
        df = pd.DataFrame({"A": np.arange(n)})
        Y = g(df)
        return Y

    # replicated argument is passed
    @bodo.jit
    def g2(df):
        return df + 1

    @bodo.jit
    def impl2(df):
        Y = g2(df)
        return Y

    # change argument distribution (recompile)
    @bodo.jit
    def g3(df):
        return df + 2

    @bodo.jit
    def impl3(n):
        df = pd.DataFrame({"A": np.arange(n)})
        Y = g3(df)
        df["B"] = [1, 2, 3]  # forces REP after g3 call
        return Y, df

    @bodo.jit
    def g4(S):
        return np.asarray([S.sum()])

    # Series creation
    @bodo.jit
    def impl4(n):
        S = pd.Series(np.arange(n))
        Y = g4(S)
        return Y

    @bodo.jit
    def impl5(n):
        S = pd.Series(np.arange(n))
        Y = g3(S)
        S.index = [1, 2, 3]  # forces REP after g3 call
        return Y, S

    # dataframe tests
    impl1(11)
    assert count_array_REPs() > 0
    assert count_array_OneDs() > 0
    impl2(pd.DataFrame({"A": [1, 3, 5] * 2}))
    assert count_array_REPs() > 0
    assert count_array_OneDs() == 0
    assert count_array_OneD_Vars() == 0
    impl3(3)
    assert count_array_REPs() > 0
    assert count_array_OneDs() == 0
    assert count_array_OneD_Vars() == 0

    # Series tests
    impl4(11)
    assert count_array_REPs() > 0
    assert count_array_OneDs() > 0
    impl2(pd.Series([1, 3, 5] * 2))
    assert count_array_REPs() > 0
    assert count_array_OneDs() == 0
    assert count_array_OneD_Vars() == 0
    impl5(3)
    assert count_array_REPs() > 0
    assert count_array_OneDs() == 0
    assert count_array_OneD_Vars() == 0


@pytest.mark.slow
def test_partial_dist_flag(memory_leak_check):
    """make sure specifying distributions partially works as expected"""

    @bodo.jit(distributed=["df"])
    def f(df):
        df2 = df.groupby("A").sum()
        return df2

    df = pd.DataFrame(
        {"A": [1, 1, 1, 2, 2, 2, 3, 3, 3], "B": [1, 2, 3, 1, 2, 3, 1, 2, 3]}
    )
    f(df)
    assert count_array_REPs() == 0


def test_dist_type_change_multi_func1(memory_leak_check):
    """test a corner case in multi-function case where distribution of output of a
    dispatcher call is not set properly [BE-1328]
    """

    # distribution of df should be set to 1D and q1/q2 calls shouldn't cause errors
    @bodo.jit
    def load_df():
        df = pd.DataFrame({"A": np.arange(10)})
        return df

    @bodo.jit
    def q1(df):
        pass

    @bodo.jit
    def q2(df):
        pass

    @bodo.jit
    def test():
        df = load_df()
        q1(df)
        q2(df)

    test()


def test_dist_type_change_multi_func2(memory_leak_check):
    """test a corner case in multi-function case where distribution of a dispatcher call
    input variable changes (forced by another dispatcher), leading to data type change
    which requires dispatcher recompilation
    """

    @bodo.jit(distributed=False)
    def q1(df):
        pass

    @bodo.jit
    def q2(df):
        pass

    @bodo.jit
    def test():
        df = pd.DataFrame({"A": np.arange(10)})
        q1(df)
        q2(df)

    test()


def test_dist_type_change_multi_func3(memory_leak_check):
    """same as test_dist_type_change_multi_func1 but checks for tuple return in load_df"""

    @bodo.jit
    def load_df():
        df = pd.DataFrame({"A": np.arange(10)})
        return (df,)

    @bodo.jit
    def q1(df):
        pass

    @bodo.jit
    def q2(df):
        pass

    @bodo.jit
    def test():
        (df,) = load_df()
        q1(df)
        q2(df)

    test()


def _check_scatterv_gatherv_allgatherv(orig_data, n):
    """check the output of scatterv() on 'data'. Confirms that calling gatherv and allgatherv on the distributed data
    succeeds"""
    if bodo.get_rank() != 0:
        data_to_scatter = None
    else:
        data_to_scatter = orig_data
    recv_data = bodo.scatterv(data_to_scatter)
    rank = bodo.get_rank()
    n_pes = bodo.get_size()

    if isinstance(recv_data, tuple):
        recv_len = len(recv_data[0])
    else:
        recv_len = len(recv_data)
    # check length
    # checking on all PEs that all PEs passed avoids hangs
    passed = _test_equal_guard(
        recv_len, bodo.libs.distributed_api.get_node_portion(n, n_pes, rank)
    )
    n_passed = reduce_sum(passed)
    assert n_passed == n_pes

    # check data with gatherv
    gathered_data = bodo.gatherv(recv_data)
    if rank == 0:
        passed = _test_equal_guard(gathered_data, orig_data)
    else:
        passed = 1

    n_passed = reduce_sum(passed)
    assert n_passed == n_pes

    # check data with gatherv
    allgathered_data = bodo.allgatherv(recv_data)
    passed = _test_equal_guard(allgathered_data, orig_data)
    n_passed = reduce_sum(passed)
    assert n_passed == n_pes


n = 22
n_col = 3


def get_random_integerarray(n):
    np.random.seed(5)
    return pd.arrays.IntegerArray(
        np.random.randint(0, 10, n, np.int32), np.random.ranf(n) < 0.30
    )


def get_random_floatingarray(n):
    np.random.seed(5)
    return pd.arrays.FloatingArray(np.random.ranf(n), np.random.ranf(n) < 0.30)


def get_random_booleanarray(n):
    np.random.seed(5)
    return pd.arrays.BooleanArray(np.random.ranf(n) < 0.50, np.random.ranf(n) < 0.30)


def get_random_decimalarray(n):
    np.random.seed(5)
    return np.array([None if a < 0.3 else Decimal(str(a)) for a in np.random.ranf(n)])


def get_random_int64index(n):
    np.random.seed(5)
    return pd.Index(np.random.randint(0, 10, n), dtype="Int64")


@pytest.mark.parametrize(
    "data",
    [
        np.arange(n, dtype=np.float32),  # 1D np array
        pytest.param(
            np.arange(n * n_col).reshape(n, n_col), marks=pytest.mark.slow
        ),  # 2D np array
        pytest.param(
            gen_random_string_binary_array(n), marks=pytest.mark.slow
        ),  # string array
        pytest.param(
            gen_random_string_binary_array(n, is_binary=True), marks=pytest.mark.slow
        ),  # binary array
        pytest.param(get_random_integerarray(n), marks=pytest.mark.slow),
        pytest.param(get_random_floatingarray(n), marks=pytest.mark.slow),
        pytest.param(get_random_booleanarray(n), marks=pytest.mark.slow),
        pytest.param(get_random_decimalarray(n), marks=pytest.mark.slow),
        pytest.param(
            pd.date_range("2017-01-13", periods=n).date, marks=pytest.mark.slow
        ),  # date array
        pytest.param(
            pd.RangeIndex(n), marks=pytest.mark.slow
        ),  # RangeIndex, TODO: test non-trivial start/step when gatherv() supports them
        pytest.param(
            pd.RangeIndex(n, name="A"), marks=pytest.mark.slow
        ),  # RangeIndex with name
        pytest.param(get_random_int64index(n), marks=pytest.mark.slow),
        pytest.param(
            pd.Index(gen_random_string_binary_array(n), name="A"),
            marks=pytest.mark.slow,
        ),  # String Index
        pytest.param(
            pd.DatetimeIndex(pd.date_range("1983-10-15", periods=n)),
            marks=pytest.mark.slow,
        ),  # DatetimeIndex
        pytest.param(
            pd.timedelta_range(start="1D", periods=n, name="A"), marks=pytest.mark.slow
        ),  # TimedeltaIndex
        pytest.param(
            pd.MultiIndex.from_arrays(
                [
                    gen_random_string_binary_array(n),
                    np.arange(n),
                    pd.date_range("2001-10-15", periods=n),
                ],
                names=["AA", "B", None],
            ),
            marks=pytest.mark.slow,
        ),
        pd.Series(gen_random_string_binary_array(n), np.arange(n) + 1, name="A"),
        pd.Series(
            gen_random_string_binary_array(n, is_binary=True),
            np.arange(n) + 1,
            name="A",
        ),
        pd.DataFrame(
            {
                "A": gen_random_string_binary_array(n),
                "AB": np.arange(n),
                "CCC": pd.date_range("2001-10-15", periods=n),
            },
            np.arange(n) + 2,
        ),
        pytest.param(
            pd.Series(["BB", "CC"] + (["AA"] * (n - 2)), dtype="category"),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Index(["BB", "CC"] + (["AA"] * (n - 2)), dtype="category"),
            marks=pytest.mark.slow,
        ),
        pytest.param(pd.interval_range(start=0, end=5), id="interval_idx"),
        # list(str) array
        # unboxing crashes for case below (issue #812)
        # pd.Series(gen_random_string_binary_array(n)).map(lambda a: None if pd.isna(a) else [a, "A"]).values
        pytest.param(
            pd.Series(["A"] * n).map(lambda a: None if pd.isna(a) else [a, "A"]).values,
            marks=pytest.mark.slow,
        ),
        pytest.param(
            np.array(
                [
                    [1, 3],
                    [2],
                    np.nan,
                    [4, 5, 6],
                    [],
                    [1, 1753],
                    [],
                    [-10],
                    [4, 10],
                    np.nan,
                    [42],
                ]
                * 2,
                dtype=object,
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            np.array(
                [
                    [2.0, -3.2],
                    [2.2, 1.3],
                    np.nan,
                    [4.1, 5.2, 6.3],
                    [],
                    [1.1, 1.2],
                    [],
                    [-42.0],
                    [3.14],
                    [2.0, 3.0],
                    np.nan,
                ]
                * 2,
                dtype=object,
            ),
            marks=pytest.mark.slow,
        ),
    ],
)
# TODO: Add memory_leak_check when bug is resolved (failed on data13)
def test_scatterv_gatherv_allgatherv_python(data):
    """Test bodo.scatterv(), gatherv(), and allgatherv() for Bodo distributed data types"""
    n = len(data)

    _check_scatterv_gatherv_allgatherv(data, n)
    # check it works in a tuple aswell
    _check_scatterv_gatherv_allgatherv((data,), n)


def test_scatterv_gatherv_allgatherv_df_python(df_value):
    """Test bodo.scatterv(), gatherv(), and allgatherv() for all supported dataframe types"""
    n = len(df_value)

    _check_scatterv_gatherv_allgatherv(df_value, n)
    # check it works in a tuple aswell
    _check_scatterv_gatherv_allgatherv((df_value,), n)


def test_scatterv_gatherv_allgatherv_df_jit(df_value, memory_leak_check):
    """test using scatterv for all supported dataframe types inside jit functions"""

    def impl(df):
        return bodo.scatterv(df)

    df_scattered = bodo.jit(all_returns_distributed=True)(impl)(df_value)
    # We have some minor dtype differences from pandas
    _test_equal_guard(df_value, bodo.allgatherv(df_scattered), check_dtype=False)

    passed = 1
    gathered_val = bodo.gatherv(df_scattered)
    if bodo.get_rank() == 0:
        passed = _test_equal_guard(df_value, gathered_val, check_dtype=False)

    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size()


def test_scatterv_None_warning(df_value):
    """Test that scatterv returns warning if value is not None on ranks != 0"""

    if bodo.get_rank() == 0:
        df_proper = df_value
    else:
        df_proper = None
    df_proper = bodo.scatterv(df_proper)

    df_improper = df_value
    if bodo.get_rank() == 0:
        df_improper = bodo.scatterv(df_improper)
    else:
        with pytest.warns(BodoWarning) as warn:
            df_improper = bodo.scatterv(df_improper)

        assert len(warn) == 1
        assert (
            str(warn[0].message)
            == "bodo.scatterv(): A non-None value for 'data' was found on a rank other than the root. "
            "This data won't be sent to any other ranks and will be overwritten with data from rank 0."
        )

    pd.testing.assert_frame_equal(df_proper, df_improper, check_column_type=False)


def test_gatherv_empty_df(memory_leak_check):
    """test using gatherv inside jit functions"""

    def impl(df):
        return bodo.gatherv(df)

    df = pd.DataFrame()
    df_gathered = bodo.jit()(impl)(df)
    pd.testing.assert_frame_equal(df, df_gathered, check_column_type=False)


def test_gatherv_str(memory_leak_check):
    """make sure gatherv() works for string arrays with over-allocated data arrays"""

    def impl(S):
        # replace implementation uses automatic string data expansion (over-allocation)
        S2 = S.str.replace("A", "ABC")
        return bodo.gatherv(S2)

    S = pd.Series(["A", "B", "C"])
    S2 = bodo.jit(distributed=["S"])(impl)(S)
    if bodo.get_rank() == 0:
        pd.testing.assert_series_equal(
            pd.concat([S] * bodo.get_size()).str.replace("A", "ABC"),
            S2,
            check_index=False,
            check_dtype=False,
        )


def test_get_chunk_bounds(memory_leak_check):
    """make sure get_chunk_bounds() works properly"""

    @bodo.jit(distributed=["A"])
    def impl(A):
        return bodo.libs.distributed_api.get_chunk_bounds(A)

    # create data chunks on different processes and check expected output
    rank = bodo.get_rank()
    n_pes = bodo.get_size()
    if n_pes > 3:
        return
    if n_pes == 1:
        A = np.array([4, 6, 8, 11], np.int64)
        out = np.array([11], np.int64)
    elif n_pes == 2:
        if rank == 0:
            A = np.array([3, 4, 7, 11, 16], np.int64)
        else:
            A = np.array([17, 19], np.int64)
        out = np.array([16, 19], np.int64)
    elif n_pes == 3:
        if rank == 0:
            A = np.array([3, 4], np.int64)
        elif rank == 1:
            A = np.array([3, 4, 7, 11, 16], np.int64)
        else:
            A = np.array([17, 19], np.int64)
        out = np.array([4, 16, 19], np.int64)

    np.testing.assert_array_equal(impl(A), out)

    # test empty chunk corner cases
    int64_min = np.iinfo(np.int64).min
    if n_pes == 1:
        A = np.array([], np.int64)
        out = np.array([int64_min], np.int64)
    elif n_pes == 2:
        if rank == 0:
            A = np.array([3], np.int64)
        else:
            A = np.array([], np.int64)
        out = np.array([3, 3], np.int64)
    elif n_pes == 3:
        if rank == 0:
            A = np.array([], np.int64)
        elif rank == 1:
            A = np.array([3, 4, 7, 11, 16], np.int64)
        else:
            A = np.array([], np.int64)
        out = np.array([int64_min, 16, 16], np.int64)

    np.testing.assert_array_equal(impl(A), out)

    # error checking unsupported array type
    with pytest.raises(BodoError, match=(r"only supports Numpy int input")):
        impl(np.array([1.1]))


@pytest.mark.slow
@pytest.mark.parametrize(
    "val1, val2",
    [
        ("hello from rank0", "other"),
        (1, 0),
        (
            np.datetime64("2005-02-25").astype("datetime64[ns]"),
            np.datetime64("2000-01-01").astype("datetime64[ns]"),
        ),
        (
            np.timedelta64(10).astype("timedelta64[ns]"),
            np.timedelta64(-1).astype("timedelta64[ns]"),
        ),
        (True, False),
        (3.14159, 123.123123),
    ],
)
def test_bcast_scalar(val1, val2):
    def impl():
        if bodo.get_rank() == 0:
            scalar_val = val1
        else:
            scalar_val = val2
        n = bodo.libs.distributed_api.bcast_scalar(scalar_val)
        return n

    check_func(impl, (), py_output=val1)


@pytest.mark.slow
def test_bcast_tuple():

    dt_val1, dt_val2 = (
        np.datetime64("2005-02-25").astype("datetime64[ns]"),
        np.datetime64("2000-01-01").astype("datetime64[ns]"),
    )
    td_val1, td_val2 = (
        np.timedelta64(10).astype("timedelta64[ns]"),
        np.timedelta64(-1).astype("timedelta64[ns]"),
    )

    def impl():
        if bodo.get_rank() == 0:
            tuple_val = ("Hi from rank 0", 1, 3.14159, dt_val1, True, td_val1)
        else:
            tuple_val = ("foo", -1, -1.0, dt_val2, False, td_val2)
        n = bodo.libs.distributed_api.bcast_tuple(tuple_val)
        return n

    check_func(
        impl, (), py_output=("Hi from rank 0", 1, 3.14159, dt_val1, True, td_val1)
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "val",
    [
        1000,
        np.datetime64("2015-05-12").astype("datetime64[ns]"),
        np.timedelta64(100).astype("timedelta64[ns]"),
        False,
        0.0091823,
    ],
)
def test_send_recv(val):
    from bodo.libs.distributed_api import recv, send

    if bodo.get_size() == 1:
        return
    np.random.seed(np.uint32(hash(val)))
    send_rank = np.random.randint(bodo.get_size())
    recv_rank = np.random.randint(bodo.get_size())
    # make sure send_rank != recv_rank
    if send_rank == recv_rank:
        if send_rank == 0:
            recv_rank = 1
        else:
            recv_rank = send_rank - 1
    assert send_rank != recv_rank
    print(f"send_rank: {send_rank}")
    print(f"recv_rank: {recv_rank}")

    send_type = bodo.typeof(val)

    def impl(send_val, recv_rank, send_rank):
        tag1 = 101
        if bodo.get_rank() == send_rank:
            send(send_val, recv_rank, tag1)

        if bodo.get_rank() == recv_rank:
            output_1 = recv(send_type, send_rank, tag1)

            return output_1

        return None

    check_func(impl, (val, recv_rank, send_rank))


@pytest.mark.slow
def test_bcast(arr_tuple_val):
    import datetime

    if not isinstance(arr_tuple_val[0], np.ndarray) or isinstance(
        arr_tuple_val[0][0], datetime.date
    ):
        return

    def impl():
        A = arr_tuple_val[0]
        if bodo.get_rank() == 0:
            arr_val = A
        else:
            arr_val = np.empty(len(A), A.dtype)
        bodo.libs.distributed_api.bcast(arr_val)
        return arr_val

    check_func(
        impl,
        (),
        py_output=arr_tuple_val[0],
        is_out_distributed=False,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "val0, val1, val2, val3",
    [
        (0, 1, 2, 3),
        (
            np.datetime64("2015-05-12").astype("datetime64[ns]"),
            np.datetime64("2011-10-11").astype("datetime64[ns]"),
            np.datetime64("2005-01-02").astype("datetime64[ns]"),
            np.datetime64("2001-03-11").astype("datetime64[ns]"),
        ),
        (
            np.timedelta64(100).astype("timedelta64[ns]"),
            np.timedelta64(-25).astype("timedelta64[ns]"),
            np.timedelta64(10).astype("timedelta64[ns]"),
            np.timedelta64(1000).astype("timedelta64[ns]"),
        ),
        (False, True, False, True),
        (0.0091823, 12.14523, -1231.12398, 11.0),
    ],
)
def test_all_to_all(val0, val1, val2, val3):

    if bodo.get_size() > 4:
        return

    from bodo.libs.distributed_api import alltoall

    def impl(n):
        values_to_send_to_rank_0 = [val0] * n
        values_to_send_to_rank_1 = [val1] * n
        values_to_send_to_rank_2 = [val2] * n
        values_to_send_to_rank_3 = [val3] * n
        if bodo.get_size() == 1:
            L = values_to_send_to_rank_0
        if bodo.get_size() >= 2:
            L = values_to_send_to_rank_0 + values_to_send_to_rank_1
        if bodo.get_size() >= 3:
            L = (
                values_to_send_to_rank_0
                + values_to_send_to_rank_1
                + values_to_send_to_rank_2
            )
        if bodo.get_size() >= 4:
            L = (
                values_to_send_to_rank_0
                + values_to_send_to_rank_1
                + values_to_send_to_rank_2
                + values_to_send_to_rank_3
            )

        A = np.array(L)
        recv_buf = np.empty(len(A), A.dtype)
        alltoall(A, recv_buf, n)

        return recv_buf

    check_func(impl, (1,), is_out_distributed=False)
    check_func(impl, (10,), is_out_distributed=False)


@pytest.mark.slow
def test_barrier_error():
    import numba

    def f():
        bodo.barrier("foo")

    with pytest.raises(
        numba.core.errors.TypingError,
        match=r"too many positional arguments",
    ):
        bodo.jit(f)()


@pytest.mark.slow
@pytest.mark.parametrize(
    "val1, val2",
    [
        ("hello from rank0", "other"),
        (1, 0),
        (
            np.datetime64("2005-02-25").astype("datetime64[ns]"),
            np.datetime64("2000-01-01").astype("datetime64[ns]"),
        ),
        (
            np.timedelta64(10).astype("timedelta64[ns]"),
            np.timedelta64(-1).astype("timedelta64[ns]"),
        ),
        (True, False),
        (3.14159, 123.123123),
    ],
)
@pytest.mark.parametrize("root", [0, 1, 2])
def test_bcast_scalar_root(val1, val2, root):
    # Run if root is in range of num. of processes
    if root >= bodo.get_size():
        return

    def impl():
        if bodo.get_rank() == root:
            scalar_val = val1
        else:
            scalar_val = val2
        n = bodo.libs.distributed_api.bcast_scalar(scalar_val, root)
        return n

    check_func(impl, (), py_output=val1)


# @pytest.mark.slow
@pytest.mark.parametrize(
    "val1, val2",
    [
        (
            np.datetime64("2005-02-25").astype("datetime64[ns]"),
            np.datetime64("2000-01-01").astype("datetime64[ns]"),
        ),
        (
            np.timedelta64(10).astype("timedelta64[ns]"),
            np.timedelta64(-1).astype("timedelta64[ns]"),
        ),
    ],
)
@pytest.mark.parametrize("root", [0, 1, 2])
def test_bcast_tuple_root(val1, val2, root):
    # Run if root is in range of num. of processes
    if root >= bodo.get_size():
        return

    def impl():
        if bodo.get_rank() == root:
            tuple_val = ("Hi from rank 0", 1, 3.14159, val1, True, None)
        else:
            tuple_val = ("foo", -1, -1.0, val2, False, None)
        n = bodo.libs.distributed_api.bcast_tuple(tuple_val, root)
        return n

    check_func(impl, (), py_output=("Hi from rank 0", 1, 3.14159, val1, True, None))


@pytest.mark.parametrize(
    "val1, val2",
    [
        (
            # StringArray
            pd.array(
                [
                    "True",
                    "False",
                    "go",
                    "bears",
                    "u",
                    "who",
                    "power",
                    "trip",
                ]
            ),
            pd.array(
                [
                    "hihi",
                    "gogo1",
                    "to",
                    "yours",
                    "w",
                    "hii",
                    "trips",
                    "powr",
                ]
            ),
        ),
        # Numpy array
        (
            np.array([1.121, 0.0, 35.13431, -2414.4242, 23211.22], dtype=np.float32),
            np.array(
                [1.121, 0.0, 5.1, -2.42, 32.2],
                dtype=np.float32,
            ),
        ),
        # Nullable float
        (
            pd.array(
                [
                    1.121,
                    0.0,
                    35.13431,
                    -2414.4242,
                    23211.22,
                    None,
                ],
                dtype=pd.Float64Dtype(),
            ),
            pd.array(
                [
                    1.121,
                    0.0,
                    5.1,
                    -2.42,
                    32.2,
                    None,
                ],
                dtype=pd.Float64Dtype(),
            ),
        ),
        # Nullable boolean
        (
            np.array(
                [
                    True,
                    False,
                    True,
                    True,
                    False,
                    False,
                    True,
                    None,
                ]
            ),
            np.array(
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    None,
                ]
            ),
        ),
        # date
        (
            np.append(
                pd.date_range("2017-07-03", "2017-07-17").date,
                [datetime.date(2016, 3, 3)],
            ),
            np.append(
                pd.date_range("2017-07-15", "2017-07-29").date,
                [datetime.date(2018, 6, 7)],
            ),
        ),
        # Decimal
        (
            np.array(
                [
                    Decimal("1.6"),
                    Decimal("-0.222"),
                    Decimal("1111.316"),
                    Decimal("1234.00046"),
                    Decimal("5.1"),
                    Decimal("-11131.0056"),
                    Decimal("0.0"),
                ]
            ),
            np.array(
                [
                    Decimal("0.0"),
                    Decimal("-0.222"),
                    Decimal("1111.316"),
                    Decimal("1"),
                    Decimal("5.1"),
                    Decimal("-1"),
                    Decimal("-1"),
                ]
            ),
        ),
    ],
)
@pytest.mark.parametrize("root", [0, 1, 2])
@pytest.mark.skip(reason="[BE-2375] Seg fault with Linux")
def test_bcast(val1, val2, root):
    """
    NOTE: per current implementation. Array is pre-allocated (i.e. same array size).
    """
    # Run if root is in range of num. of processes
    if root >= bodo.get_size():
        return

    def impl():
        if bodo.get_rank() == root:
            val = val1
        else:
            val = val2
        bodo.libs.distributed_api.bcast(val, root)
        return val

    check_func(impl, (), py_output=val1, is_out_distributed=False)


def test_gatherv_global(memory_leak_check):
    """
    Test that gatherv is properly optimized out when run with
    a global array.
    """
    arr = np.arange(100)

    @bodo.jit
    def impl1():
        return bodo.gatherv(arr)

    # Test the sequential pipeline
    @bodo.jit(distributed=False)
    def impl2():
        return bodo.gatherv(arr)

    np.testing.assert_array_equal(impl1(), arr)
    np.testing.assert_array_equal(impl2(), arr)


def test_dist_flag_info_propogation(memory_leak_check):
    """
    Tests that distribution information is properly propogated when calling nested Bodo functions with
    flags specifying distribution information
    """
    df = pd.DataFrame({"A": np.arange(12)})

    @bodo.jit()
    def inner_fn(argument_df):
        return argument_df

    def outer_fn(argument_df):
        return inner_fn(argument_df)

    check_func(outer_fn, (df,))
