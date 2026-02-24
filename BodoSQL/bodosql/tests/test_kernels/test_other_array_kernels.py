"""Test Bodo's array kernel utilities for BodoSQL miscellaneous functions"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from numba.core import types
from numba.extending import overload

import bodo
import bodosql
from bodo.tests.utils import check_func, pytest_slow_unless_codegen
from bodosql.kernels.array_kernel_utils import vectorized_sol

# Skip unless any library or BodoSQL codegen or files were changed
pytestmark = pytest_slow_unless_codegen


@pytest.fixture(
    params=[
        pytest.param(
            (
                pd.Series([0, 42, None], dtype=pd.Int32Dtype()).repeat(3),
                pd.Series([0, 42, None] * 3, dtype=pd.Int32Dtype()),
            ),
            id="vector_vector",
        ),
        pytest.param(
            (pd.Series([0, 1, -1, None, 2, -2, 0, None], dtype=pd.Int32Dtype()), 0),
            id="vector_scalar_zero",
        ),
        pytest.param(
            (0, pd.Series([0, 1, -1, None, 2, -2, 0, None], dtype=pd.Int32Dtype())),
            id="scalar_vector_zero",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (3, pd.Series([0, 1, -1, None, 2, -2, 0, None], dtype=pd.Int32Dtype())),
            id="scalar_vector_nonzero",
        ),
        pytest.param(
            (
                pd.Series([0, 1, -1, None, 2, -2, 0, None], dtype=pd.Int16Dtype()),
                np.uint8(255),
            ),
            id="vector_scalar_nonzero",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (pd.Series([0, 1, -1, None, 2, -2, 0, None], dtype=pd.Int64Dtype()), None),
            id="vector_null",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (None, pd.Series([0, 1, -1, None, 2, -2, 0, None], dtype=pd.Int64Dtype())),
            id="null_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (15, -42),
            id="scalar_scalar_nonzero",
        ),
        pytest.param((0, 13), id="scalar_scalar_mixed", marks=pytest.mark.slow),
        pytest.param((0, 0), id="scalar_scalar_zero_zero", marks=pytest.mark.slow),
        pytest.param((0, None), id="scalar_scalar_zero_null", marks=pytest.mark.slow),
        pytest.param(
            (64, None), id="scalar_scalar_nonzero_null", marks=pytest.mark.slow
        ),
        pytest.param((None, 0), id="scalar_scalar_null_zero", marks=pytest.mark.slow),
        pytest.param(
            (None, -15), id="scalar_scalar_null_nonzero", marks=pytest.mark.slow
        ),
        pytest.param(
            (None, None), id="scalar_scalar_null_null", marks=pytest.mark.slow
        ),
        pytest.param(
            (
                pd.Series([0, 1, 127, 128, 255, None] * 5, dtype=pd.UInt8Dtype()),
                pd.Series([0, 1, 127, -128, -1, None], dtype=pd.Int8Dtype()).repeat(5),
            ),
            id="mixed_int_vector_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series([True, False, None], dtype=pd.BooleanDtype()).repeat(3),
                pd.Series([True, False, None] * 3, dtype=pd.BooleanDtype()),
            ),
            id="boolean_vector_vector",
            marks=pytest.mark.slow,
        ),
    ],
)
def boolean_numerical_scalar_vector(request):
    return request.param


def test_booland(boolean_numerical_scalar_vector, memory_leak_check):
    def impl(A, B):
        return pd.Series(bodosql.kernels.booland(A, B))

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in boolean_numerical_scalar_vector):
        impl = lambda A, B: bodosql.kernels.booland(A, B)

    def booland_scalar_fn(A, B):
        if pd.notna(A) and pd.notna(B) and A != 0 and B != 0:
            return True
        elif (pd.notna(A) and A == 0) or (pd.notna(B) and B == 0):
            return False
        else:
            return None

    booland_answer = vectorized_sol(
        boolean_numerical_scalar_vector, booland_scalar_fn, pd.BooleanDtype()
    )

    check_func(
        impl,
        boolean_numerical_scalar_vector,
        py_output=booland_answer,
        check_dtype=False,
        reset_index=True,
    )


def test_boolor(boolean_numerical_scalar_vector, memory_leak_check):
    def impl(A, B):
        return pd.Series(bodosql.kernels.boolor(A, B))

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in boolean_numerical_scalar_vector):
        impl = lambda A, B: bodosql.kernels.boolor(A, B)

    def boolor_scalar_fn(A, B):
        if (pd.notna(A) and A != 0) or (pd.notna(B) and B != 0):
            return True
        elif pd.notna(A) and A == 0 and pd.notna(B) and B == 0:
            return False
        else:
            return None

    boolor_answer = vectorized_sol(
        boolean_numerical_scalar_vector, boolor_scalar_fn, pd.BooleanDtype()
    )

    check_func(
        impl,
        boolean_numerical_scalar_vector,
        py_output=boolor_answer,
        check_dtype=False,
        reset_index=True,
        sort_output=False,
    )


def test_boolxor(boolean_numerical_scalar_vector, memory_leak_check):
    def impl(A, B):
        return pd.Series(bodosql.kernels.boolxor(A, B))

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in boolean_numerical_scalar_vector):
        impl = lambda A, B: bodosql.kernels.boolxor(A, B)

    def boolxor_scalar_fn(A, B):
        if pd.isna(A) or pd.isna(B):
            return None
        else:
            return (A == 0) != (B == 0)

    boolxor_answer = vectorized_sol(
        boolean_numerical_scalar_vector, boolxor_scalar_fn, pd.BooleanDtype()
    )

    check_func(
        impl,
        boolean_numerical_scalar_vector,
        py_output=boolxor_answer,
        check_dtype=False,
        reset_index=True,
    )


def test_boolnot(boolean_numerical_scalar_vector, memory_leak_check):
    def impl(A):
        return pd.Series(bodosql.kernels.boolnot(A))

    impl_scalar = lambda A: bodosql.kernels.boolnot(A)

    def boolnot_scalar_fn(A):
        if pd.isna(A):
            return None
        if A == 0:
            return True
        else:
            return False

    boolxor_answer_0 = vectorized_sol(
        (boolean_numerical_scalar_vector[0],), boolnot_scalar_fn, pd.BooleanDtype()
    )
    boolxor_answer_1 = vectorized_sol(
        (boolean_numerical_scalar_vector[1],), boolnot_scalar_fn, pd.BooleanDtype()
    )

    check_func(
        impl
        if isinstance(boolean_numerical_scalar_vector[0], pd.Series)
        else impl_scalar,
        (boolean_numerical_scalar_vector[0],),
        py_output=boolxor_answer_0,
        check_dtype=False,
        reset_index=True,
    )
    check_func(
        impl
        if isinstance(boolean_numerical_scalar_vector[1], pd.Series)
        else impl_scalar,
        (boolean_numerical_scalar_vector[1],),
        py_output=boolxor_answer_1,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series(pd.array([True, False, True, False, True, None])),
                pd.Series(pd.array([None, None, 2, 3, 4, -1])),
                pd.Series(pd.array([5, 6, None, None, 9, -1])),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                pd.Series(pd.array([True, True, True, False, False])),
                pd.Series(pd.array(["A", "B", "C", "D", "E"])),
                "-",
            ),
            id="vector_vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (pd.Series(pd.array([False, True, False, True, False])), 1.0, -1.0),
            id="vector_scalar_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(pd.array([True, True, False, False, True])),
                pd.Series(pd.array(["A", "B", "C", "D", "E"])),
                None,
            ),
            id="vector_vector_null",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (True, 42, 16),
            id="all_scalar_no_null",
        ),
        pytest.param(
            (None, 42, 16),
            id="all_scalar_with_null_cond",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (True, None, 16),
            id="all_scalar_with_null_branch",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (True, 13, None),
            id="all_scalar_with_unused_null",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (False, None, None),
            id="all_scalar_both_null_branch",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (None, None, None),
            id="all_scalar_all_null",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    [True, False, None, True, False, None, True, False],
                    dtype=pd.BooleanDtype(),
                ),
                np.array(
                    [
                        bodo.types.TimestampTZ.fromLocal("2024-01-01 12:00:00", 0),
                        None,
                        bodo.types.TimestampTZ.fromLocal("2024-01-01", 60),
                        None,
                        bodo.types.TimestampTZ.fromLocal("2024-01-01 12:00:00", -150),
                        None,
                        None,
                        bodo.types.TimestampTZ.fromLocal("2024-01-01", 300),
                    ]
                ),
                bodo.types.TimestampTZ.fromLocal("2024-01-01 06:45:00", 0),
            ),
            id="timestamp_tz",
        ),
    ],
)
def test_cond(args, memory_leak_check):
    def impl(arr, ifbranch, elsebranch):
        return pd.Series(bodosql.kernels.cond(arr, ifbranch, elsebranch))

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in args):
        impl = lambda arr, ifbranch, elsebranch: bodosql.kernels.cond(
            arr, ifbranch, elsebranch
        )

    # Simulates COND on a single row
    def cond_scalar_fn(arr, ifbranch, elsebranch):
        return ifbranch if ((not pd.isna(arr)) and arr) else elsebranch

    cond_answer = vectorized_sol(args, cond_scalar_fn, None)
    check_func(
        impl,
        args,
        py_output=cond_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series([1, None, 3, None, 5, None, 7], dtype=pd.Int32Dtype()),
                pd.Series([1, 2, 3, 4, None, None, None], dtype=pd.Int8Dtype()),
                pd.Series([None, None, None, 50, 60, 70, 80], dtype=pd.UInt32Dtype()),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                pd.Series([1, None, 3, None, 5, None, 7], dtype=pd.Int32Dtype()),
                pd.Series(list("ABCDEFG")),
                "",
            ),
            id="vector_vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series([0.5, None, 1.0, None, 2.0, None, 3.0]),
                False,
                pd.Series(
                    [True, False, None, True, False, None, True],
                    dtype=pd.BooleanDtype(),
                ),
            ),
            id="vector_scalar_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (pd.Series(["", None, "AB", None, "alphabet"]), 1.0, -1.0),
            id="vector_scalar_scalar",
        ),
        pytest.param(
            (
                pd.Series([1, 2, 3, 4, None, None, None], dtype=pd.Int32Dtype()),
                pd.Series([1, None, 3, None, 5, None, 7], dtype=pd.Int32Dtype()),
                None,
            ),
            id="vector_vector_null",
        ),
        pytest.param(
            (
                pd.Series(["A", None, "B", None, "C", None, "D", None, "E"]),
                None,
                128,
            ),
            id="vector_null_scalar",
        ),
        pytest.param(
            (0.5, "A", "B"),
            id="all_scalar",
        ),
        pytest.param(
            (10, None, 16),
            id="scalar_null_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            ("foo", 13, None),
            id="scalar_scalar_null",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (True, None, None),
            id="scalar_null_null",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (None, 10, 20),
            id="null_scalar_scalar",
        ),
        pytest.param(
            (None, None, 0.5),
            id="null_null_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (None, False, None),
            id="null_scalar_null",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (None, None, None),
            id="all_null",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series([1, None, 3, None, 5, None, 7, None], dtype=pd.Int8Dtype()),
                np.array(
                    [
                        bodo.types.TimestampTZ.fromLocal("2024-01-01 12:00:00", 0),
                        bodo.types.TimestampTZ.fromLocal("2024-01-01", 60),
                        None,
                        None,
                        bodo.types.TimestampTZ.fromLocal("2024-01-01 12:00:00", -150),
                        bodo.types.TimestampTZ.fromLocal("2024-01-01", 300),
                        None,
                        None,
                    ]
                ),
                bodo.types.TimestampTZ.fromLocal("2024-01-01 06:45:00", 0),
            ),
            id="timestamp_tz",
        ),
    ],
)
def test_nvl2(args, memory_leak_check):
    def impl(arr, not_null_branch, null_branch):
        return pd.Series(bodosql.kernels.nvl2(arr, not_null_branch, null_branch))

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in args):
        impl = lambda arr, not_null_branch, null_branch: bodosql.kernels.nvl2(
            arr, not_null_branch, null_branch
        )

    # Simulates NVL2 on a single row
    def nvl2_scalar_fn(arr, not_null_branch, null_branch):
        return not_null_branch if (not pd.isna(arr)) else null_branch

    nvl2_answer = vectorized_sol(args, nvl2_scalar_fn, None)
    check_func(
        impl,
        args,
        py_output=nvl2_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args, is_scalar_a, is_scalar_b, answer",
    [
        pytest.param(
            (
                pd.Series([0, 42, None] * 3, dtype=pd.Int32Dtype()),
                pd.Series(
                    [0, 0, 0, 42, 42, 42, None, None, None], dtype=pd.Int32Dtype()
                ),
            ),
            False,
            False,
            pd.Series(
                [True, False, False] + [False, True, False] + [False, False, True]
            ),
            id="int32-vector-vector",
        ),
        pytest.param(
            (pd.Series([0, 36, 42, None, -42, 1], dtype=pd.Int32Dtype()), 42),
            False,
            True,
            pd.Series([False, False, True, False, False, False]),
            id="int32-vector-scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                0,
                pd.Series([0, 36, 42, None, -42, 1], dtype=pd.Int32Dtype()),
            ),
            True,
            False,
            pd.Series([True, False, False, False, False, False]),
            id="int32-scalar-vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (pd.Series([0, 36, 42, None, -42, 1], dtype=pd.Int32Dtype()), None),
            False,
            True,
            pd.Series([False, False, False, True, False, False]),
            id="int32-vector-null",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                None,
                pd.Series([0, 36, 42, None, -42, 1], dtype=pd.Int32Dtype()),
            ),
            True,
            False,
            pd.Series([False, False, False, True, False, False]),
            id="int32-null-vector",
            marks=pytest.mark.slow,
        ),
        pytest.param((42, 42), True, True, True, id="int64-scalar-scalar"),
        pytest.param((39, None), True, True, False, id="int64-scalar-null"),
        pytest.param((None, 42), True, True, False, id="int64-null-scalar"),
        pytest.param((None, None), True, True, True, id="int64-null-null"),
        pytest.param(
            (
                pd.Series(["A", "B", "a", "AAA", None] * 4),
                pd.Series(["A", "B", "a", "AAA", None]).repeat(4),
            ),
            False,
            False,
            pd.Series(
                [True, False, False, False]
                + [False, False, True, False]
                + [False, False, False, False]
                + [False, True, False, False]
                + [False, False, False, True]
            ),
            id="string-vector-vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series([True, False, None, True] * 4),
                pd.Series([True, False, None, False]).repeat(4),
            ),
            False,
            False,
            pd.Series(
                [True, False, False, True]
                + [False, True, False, False]
                + [False, False, True, False]
                + [False, True, False, False]
            ),
            id="boolean-vector-vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    (
                        list(pd.date_range("2018", "2019", periods=3, unit="ns").date)
                        + [None]
                    )
                    * 4
                ),
                pd.Series(
                    list(pd.date_range("2018", "2019", periods=3, unit="ns").date)
                    + [None]
                ).repeat(4),
            ),
            False,
            False,
            pd.Series(
                [True, False, False, False]
                + [False, True, False, False]
                + [False, False, True, False]
                + [False, False, False, True]
            ),
            id="date-vector-vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        bodo.types.TimestampTZ.fromUTC("2024-01-02 03:04:05", 0),
                        bodo.types.TimestampTZ.fromUTC("2024-01-02 03:04:05", 60),
                        None,
                        bodo.types.TimestampTZ.fromUTC("2024-01-02 03:04:05", 0),
                        None,
                    ]
                ),
                pd.Series(
                    [
                        bodo.types.TimestampTZ.fromUTC("2024-01-02 03:04:05", 10),
                        bodo.types.TimestampTZ.fromUTC("2024-01-02 02:04:05", 0),
                        bodo.types.TimestampTZ.fromUTC("2024-01-02 03:04:05", 0),
                        None,
                        None,
                    ]
                ),
            ),
            False,
            False,
            pd.Series([True, False, False, False, True]),
            id="timestamptz-vector-vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series([[], [1], [1, 2], [1, 2, 3], [None, 2, 3], None] * 6),
                pd.Series([[], [1], [1, 2], [1, 2, 3], [None, 2, 3], None]).repeat(6),
            ),
            False,
            False,
            pd.Series(
                [True, False, False, False, False, False]
                + [False, True, False, False, False, False]
                + [False, False, True, False, False, False]
                + [False, False, False, True, False, False]
                + [False, False, False, False, True, False]
                + [False, False, False, False, False, True]
            ),
            id="int_array-vector-vector",
        ),
        pytest.param(
            (
                pd.Series([1, 2]),
                pd.Series(
                    [[1], [], [1, 2], [1, 2, 3], [None, 2, 3], None, [2, 1], [2]]
                ),
            ),
            True,
            False,
            pd.Series([False, False, True, False, False, False, False, False]),
            id="int_array-scalar-vector",
        ),
        pytest.param(
            (
                np.array([1, 2, 3]),
                np.array([1, 2, 3]),
            ),
            True,
            True,
            True,
            id="int_array-scalar-scalar",
        ),
        pytest.param(
            (
                pd.Series(
                    [{"A": 0, "B": "A"}, {"A": 1, "B": "B"}, None, {"A": 0, "B": "C"}]
                    * 4,
                    dtype=pd.ArrowDtype(
                        pa.struct(
                            [pa.field("A", pa.int32()), pa.field("B", pa.string())]
                        )
                    ),
                ).values,
                pd.Series(
                    [{"A": 0, "B": "A"}, {"A": 1, "B": "B"}, None, {"A": 0, "B": "C"}],
                    dtype=pd.ArrowDtype(
                        pa.struct(
                            [pa.field("A", pa.int32()), pa.field("B", pa.string())]
                        )
                    ),
                )
                .repeat(4)
                .values,
            ),
            False,
            False,
            pd.Series(
                [True, False, False, False]
                + [False, True, False, False]
                + [False, False, True, False]
                + [False, False, False, True]
            ),
            id="struct-vector-vector",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        {
                            "A": 0,
                            "B": [0, 1],
                            "C": [{"D": 0, "E": "A"}, {"D": 0, "E": "A"}],
                        },
                        {
                            "A": 0,
                            "B": [],
                            "C": [{"D": 0, "E": "A"}, {"D": 0, "E": "A"}],
                        },
                        {
                            "A": 0,
                            "B": [0, 1],
                            "C": [{"D": 0, "E": "A"}, {"D": 0, "E": "C"}],
                        },
                    ]
                    * 9,
                    dtype=pd.ArrowDtype(
                        pa.struct(
                            [
                                pa.field("A", pa.int64()),
                                pa.field("B", pa.large_list(pa.int32())),
                                pa.field(
                                    "C",
                                    pa.large_list(
                                        pa.struct(
                                            [
                                                pa.field("D", pa.int32()),
                                                pa.field("E", pa.string()),
                                            ]
                                        )
                                    ),
                                ),
                            ]
                        )
                    ),
                ).values,
                pd.Series(
                    [
                        {
                            "A": 0,
                            "B": [0, 1],
                            "C": [{"D": 0, "E": "A"}, {"D": 0, "E": "A"}],
                        },
                        {
                            "A": 0,
                            "B": [],
                            "C": [{"D": 0, "E": "A"}, {"D": 0, "E": "A"}],
                        },
                        {
                            "A": 0,
                            "B": [0, 1],
                            "C": [{"D": 0, "E": "A"}, {"D": 0, "E": "C"}],
                        },
                    ],
                    dtype=pd.ArrowDtype(
                        pa.struct(
                            [
                                pa.field("A", pa.int32()),
                                pa.field("B", pa.large_list(pa.int32())),
                                pa.field(
                                    "C",
                                    pa.large_list(
                                        pa.struct(
                                            [
                                                pa.field("D", pa.int32()),
                                                pa.field("E", pa.string()),
                                            ]
                                        )
                                    ),
                                ),
                            ]
                        )
                    ),
                )
                .repeat(9)
                .values,
            ),
            False,
            False,
            pd.Series(
                [True, False, False] * 3
                + [False, True, False] * 3
                + [False, False, True] * 3
            ),
            id="struct_nested-vector-vector",
        ),
        pytest.param(
            (
                {"A": 0, "B": 1},
                {"A": 0, "B": 1},
            ),
            True,
            True,
            True,
            id="map-scalar-scalar-match_exact",
        ),
        pytest.param(
            (
                {"A": 0, "B": 1},
                {"A": 0, "B": 0},
            ),
            True,
            True,
            False,
            id="map-scalar-scalar-mismatch_value",
        ),
        pytest.param(
            (
                {"A": 0, "B": 1},
                {"A": 0, "C": 1},
            ),
            True,
            True,
            False,
            id="map-scalar-scalar-mismatch_key",
        ),
        pytest.param(
            (
                {"A": 0, "B": 1},
                pd.Series(
                    [
                        {"A": 0, "B": 1},  # Exact match
                        {"B": 1, "A": 0},  # Wrong order
                        {"A": 0, "B": 2},  # Wrong value
                        {"A": 0},  # Missing pair
                        None,
                        {"A": 0, "B": 1, "C": 2},  # Extra pair
                    ],
                    dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int32())),
                ).values,
            ),
            True,
            False,
            pd.Series([True] * 2 + [False] * 4),
            id="map-scalar-vector",
        ),
        pytest.param(
            (1, "1"),
            True,
            True,
            False,
            id="mismatched-types",
        ),
    ],
)
def test_bool_equal_null(args, is_scalar_a, is_scalar_b, answer, memory_leak_check):
    # Avoid distributed testing on mixed scalar-vector cases due
    # to complications with nested arrays.
    distributed = not (is_scalar_a or is_scalar_b)
    is_out_distributed = distributed
    only_seq = not distributed
    dist_test = only_seq

    # avoid Series conversion for scalar output
    if (not is_scalar_a) or (not is_scalar_b):

        def impl1(A, B):
            return pd.Series(bodosql.kernels.equal_null(A, B, is_scalar_a, is_scalar_b))

        def impl2(A, B):
            return pd.Series(
                bodosql.kernels.not_equal_null(A, B, is_scalar_a, is_scalar_b)
            )

        equal_null_answer = answer
        not_equal_null_answer = ~answer
    else:
        impl1 = lambda A, B: bodosql.kernels.equal_null(A, B, is_scalar_a, is_scalar_b)
        impl2 = lambda A, B: bodosql.kernels.not_equal_null(
            A, B, is_scalar_a, is_scalar_b
        )

        equal_null_answer = answer
        not_equal_null_answer = not answer

    check_func(
        impl1,
        args,
        py_output=equal_null_answer,
        check_dtype=False,
        reset_index=True,
        distributed=distributed,
        is_out_distributed=is_out_distributed,
        only_seq=only_seq,
        dist_test=dist_test,
    )
    check_func(
        impl2,
        args,
        py_output=not_equal_null_answer,
        check_dtype=False,
        reset_index=True,
        is_out_distributed=is_out_distributed,
        only_seq=only_seq,
        dist_test=dist_test,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series(
                    [
                        b"sxcsdasdfdf",
                        None,
                        b"",
                        b"asadf1234524asdfa",
                        b"\0\0\0\0",
                        None,
                        b"hello world",
                    ]
                    * 2
                ),
                pd.Series(
                    [
                        b"sxcsdasdfdf",
                        b"239i1u8yighjbfdnsma4",
                        b"i12u3gewqds",
                        None,
                        b"1203-94euwidsfhjk",
                        None,
                        b"hello world",
                    ]
                    * 2
                ),
                None,
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                12345678.123456789,
                pd.Series(
                    [
                        12345678.123456789,
                        None,
                        1,
                        2,
                        3,
                        None,
                        4,
                        12345678.123456789,
                        5,
                    ]
                    * 2
                ),
                None,
            ),
            id="scalar_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    pd.array(
                        [
                            pd.Timestamp("2022-01-02 00:00:00"),
                            None,
                            pd.Timestamp("2002-01-02 00:00:00"),
                            pd.Timestamp("2022"),
                            None,
                            pd.Timestamp("2122-01-12 00:00:00"),
                            pd.Timestamp("2022"),
                            pd.Timestamp("2022-01-02 00:01:00"),
                            pd.Timestamp("2022-11-02 00:00:00"),
                        ]
                        * 2
                    ),
                    dtype="datetime64[ns]",
                ),
                pd.Timestamp("2022"),
                None,
            ),
            id="vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                None,
                pd.Series(
                    pd.array(
                        [
                            b"12345678.123456789",
                            None,
                            b"a",
                            b"b",
                            b"c",
                            b"d",
                            b"e",
                            b"12345678.123456789",
                            b"g",
                        ]
                        * 2
                    )
                ),
                pd.StringDtype(),
            ),
            id="null_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    pd.array(
                        [
                            pd.Timedelta(minutes=40),
                            pd.Timedelta(hours=2),
                            pd.Timedelta(5),
                            pd.Timedelta(days=3),
                            pd.Timedelta(days=13),
                            pd.Timedelta(weeks=3),
                            pd.Timedelta(seconds=3),
                            None,
                            None,
                        ]
                        * 2
                    )
                ),
                None,
                None,
            ),
            id="vector_null",
            marks=pytest.mark.slow,
        ),
        pytest.param((-426472, 2, pd.Int64Dtype()), id="all_scalar_not_null"),
        pytest.param(
            ("hello world", None, pd.StringDtype()),
            id="all_scalar_null_arg1",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (None, b"0923u8hejrknsd", None),
            id="all_scalar_null_arg0",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (None, None, None),
            id="all_null",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_nullif(args, memory_leak_check):
    def impl(arg0, arg1):
        return pd.Series(bodosql.kernels.nullif(arg0, arg1))

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in args):
        impl = lambda arg0, arg1: bodosql.kernels.nullif(arg0, arg1)

    # Simulates NULLIF on a single row
    def nullif_scalar_fn(arg0, arg1):
        if pd.isna(arg0) or arg0 == arg1:
            return None
        else:
            return arg0

    arg0, arg1, out_dtype = args

    nullif_answer = vectorized_sol((arg0, arg1), nullif_scalar_fn, out_dtype)

    check_func(
        impl,
        (arg0, arg1),
        py_output=nullif_answer,
        check_dtype=False,
        reset_index=True,
    )


def test_random_seedless_vector(memory_leak_check):
    """Tests that the random_seedless kernel is generating an array of one
    million random 64 bit integers with the following checks:


     - The input length matches the output length
     - The smallest randomly chosen value is within 0.1% of the
       smallest possible int64
     - The largest randomly chosen value is within 0.1% of the
       largest possible int64
     - At most 5 duplicate values are generated
    """

    def min_max_test(A):
        R = pd.Series(bodosql.kernels.random_seedless(A))
        min_val = R.min()
        max_val = R.max()
        return min_val, max_val, len(R)

    def unique_test(A):
        R = pd.Series(bodosql.kernels.random_seedless(A))
        total_vals = len(R)
        distinct_vals = R.nunique()
        return total_vals - distinct_vals

    n = 10**6
    A = pd.DataFrame({0: pd.Series(np.arange(n))})
    min_max_result = (np.float64(-(2**63)), np.float64((2**63) - 1), np.uint64(n))
    check_func(min_max_test, (A,), py_output=min_max_result, rtol=0.001)
    unique_target = np.float64(5)
    check_func(unique_test, (A,), py_output=unique_target, atol=5)


def test_random_seedless_scalar(memory_leak_check):
    """Tests that the random_seedless kernel is generating one million random 64
    bit integers (one at a time) with the following checks:

      - The input length matches the output length
      - The smallest randomly chosen value is within 0.1% of the
        smallest possible int64
      - The largest randomly chosen value is within 0.1% of the
        largest possible int64
      - At most 5 duplicate values are generated
    """

    def min_max_test(n):
        L = []
        for _ in range(n):
            L.append(bodosql.kernels.random_seedless(None))
        R = pd.Series(L)
        min_val = R.min()
        max_val = R.max()
        return min_val, max_val, len(R)

    def unique_test(n):
        L = []
        for _ in range(n):
            L.append(bodosql.kernels.random_seedless(None))
        R = pd.Series(L)
        total_vals = len(R)
        distinct_vals = R.nunique()
        return total_vals - distinct_vals

    n = 10**6
    min_max_result = (np.float64(-(2**63)), np.float64((2**63) - 1), np.uint64(n))
    check_func(min_max_test, (n,), py_output=min_max_result, rtol=0.001, only_seq=True)
    unique_target = np.float64(5)
    check_func(unique_test, (n,), py_output=unique_target, atol=5, only_seq=True)


def test_uniform_min_max(memory_leak_check):
    """Tests the uniform kernel with the following checks:
    - None of the outputs are less than the lower bound
    - None of the outputs are more than the upper bound
    """

    def impl():
        gen = np.arange(10**6)
        R1 = pd.Series(bodosql.kernels.uniform(0, 9, gen))
        R2 = pd.Series(bodosql.kernels.uniform(-2048, 2048, gen))
        R3 = pd.Series(bodosql.kernels.uniform(-1.0, 1.0, gen))
        R4 = pd.Series(bodosql.kernels.uniform(0, 12345678.9, gen))
        min1, max1 = R1.min(), R1.max()
        min2, max2 = R2.min(), R2.max()
        min3, max3 = R3.min(), R3.max()
        min4, max4 = R4.min(), R4.max()
        return (
            min1,
            max1,
            min2,
            max2,
            min3 >= -1.0,
            max3 <= 1.0,
            min4 >= 0,
            max4 <= 12345678.9,
        )

    min_max_result = (0, 9, -2048, 2048, True, True, True, True)
    check_func(
        impl,
        (),
        py_output=min_max_result,
        is_out_distributed=False,
    )


def test_uniform_distribution(memory_leak_check):
    """Tests the uniform kernel with the following checks:
    - The mean is approximately (lower + upper) / 12
    - The variance is approximately ((upper - lower) ** 2) / 12
    - The skew is approximately 0.0
    """

    def impl():
        gen = np.arange(10**6)
        R1 = pd.Series(bodosql.kernels.uniform(-2048, 2048, gen))
        R2 = pd.Series(bodosql.kernels.uniform(0.0, 100.0, gen))
        avg1, var1, skew1 = R1.mean(), R1.var(), R1.skew()
        avg2, var2, skew2 = R2.mean(), R2.var(), R2.skew()
        return avg1, var1, skew1, avg2, var2, skew2

    distribution_result = (0.0, 1398101.3, 0.0, 50.0, 833.3, 0.0)
    check_func(
        impl,
        (),
        py_output=distribution_result,
        is_out_distributed=False,
        atol=0.1,
        rtol=0.01,
    )


def test_uniform_count(memory_leak_check):
    """Tests the uniform kernel with the following checks:
    - Each value in the domain (for integers) appears approximately the
      same number of times.
    """

    def impl():
        gen = np.arange(10**6)
        R = pd.Series(bodosql.kernels.uniform(0, 99, gen))
        C = R.value_counts().sort_index().astype("Float64")
        return C

    count_result = pd.Series([10000.0] * 100, name="count")
    check_func(
        impl,
        (),
        py_output=count_result,
        rtol=0.1,
    )


def test_uniform_generation(memory_leak_check):
    """Tests the uniform kernel with the following checks:
    - Duplicate gen values result in duplicate outputs
    """

    def impl(gen):
        R1 = pd.Series(bodosql.kernels.uniform(2.718281828, 3.1415926, gen))
        R2 = pd.Series(bodosql.kernels.uniform(2.718281828, 3.1415926, gen))
        return R1 == R2

    np.random.seed(42)
    gen = np.random.randint(0, 1000, 10**6)
    generation_result = pd.Series([True] * 10**6)
    check_func(
        impl,
        (gen,),
        py_output=generation_result,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series(
                    np.array(
                        [1.0, None, 3.0, 4.0, 5.0, 6.0, None, 8.0], dtype=np.float64
                    )
                ),
                pd.Series(
                    np.array(
                        [None, 4.0, 9.0, 16.0, 25.0, 36.0, None, 64.0], dtype=np.float64
                    )
                ),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                pd.Series(
                    np.array(
                        [1.0, None, 3.0, 4.0, 5.0, 6.0, None, 8.0], dtype=np.float64
                    )
                ),
                -42.16,
            ),
            id="vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                10875.3115512,
                pd.Series(
                    np.array(
                        [None, 4.0, 9.0, 16.0, 25.0, 36.0, None, 64.0], dtype=np.float64
                    )
                ),
            ),
            id="scalar_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    np.array(
                        [1.0, None, 3.0, 4.0, 5.0, 6.0, None, 8.0], dtype=np.float64
                    )
                ),
                None,
            ),
            id="vector_null",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                None,
                pd.Series(
                    np.array(
                        [None, 4.0, 9.0, 16.0, 25.0, 36.0, None, 64.0], dtype=np.float64
                    )
                ),
            ),
            id="null_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param((100.0, 95.2), id="all_scalar_no_null"),
        pytest.param((10.03, None), id="all_scalar_with_null"),
    ],
)
def test_regr_valxy(args, memory_leak_check):
    def impl1(y, x):
        return pd.Series(bodosql.kernels.regr_valx(y, x))

    def impl2(y, x):
        return pd.Series(bodosql.kernels.regr_valy(y, x))

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in args):
        impl1 = lambda y, x: bodosql.kernels.regr_valx(y, x)
        impl2 = lambda y, x: bodosql.kernels.regr_valy(y, x)

    # Simulates REGR_VALX on a single row
    def regr_valx_scalar_fn(y, x):
        if pd.isna(y) or pd.isna(x):
            return None
        else:
            return x

    # Simulates REGR_VALY on a single row
    def regr_valy_scalar_fn(y, x):
        if pd.isna(y) or pd.isna(x):
            return None
        else:
            return y

    regr_valx_answer = vectorized_sol(args, regr_valx_scalar_fn, None)
    regr_valy_answer = vectorized_sol(args, regr_valy_scalar_fn, None)
    check_func(
        impl1,
        args,
        py_output=regr_valx_answer,
        check_dtype=False,
        reset_index=True,
    )
    check_func(
        impl2,
        args,
        py_output=regr_valy_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "arg",
    [
        pytest.param(pd.array([False, True, True, None, False] * 6), id="vector"),
        pytest.param(False, id="scalar"),
    ],
)
def test_is_functions(arg, memory_leak_check):
    """
    Tests for the array kernels is_false, is_true,
    is_not_false, and is_not_true.
    """

    def impl1(arr):
        return pd.Series(bodosql.kernels.is_false(arr))

    def impl2(arr):
        return pd.Series(bodosql.kernels.is_true(arr))

    def impl3(arr):
        return pd.Series(bodosql.kernels.is_not_false(arr))

    def impl4(arr):
        return pd.Series(bodosql.kernels.is_not_true(arr))

    if isinstance(arg, bool):
        impl1 = lambda arr: bodosql.kernels.is_false(arr)
        impl2 = lambda arr: bodosql.kernels.is_true(arr)
        impl3 = lambda arr: bodosql.kernels.is_not_false(arr)
        impl4 = lambda arr: bodosql.kernels.is_not_true(arr)

    args = (arg,)

    is_false_scalar_fn = lambda val: False if pd.isna(val) else val == False
    is_true_scalar_fn = lambda val: False if pd.isna(val) else val == True
    is_not_false_scalar_fn = lambda val: True if pd.isna(val) else val != False
    is_not_true_scalar_fn = lambda val: True if pd.isna(val) else val != True

    answer1 = vectorized_sol(args, is_false_scalar_fn, None)
    answer2 = vectorized_sol(args, is_true_scalar_fn, None)
    answer3 = vectorized_sol(args, is_not_false_scalar_fn, None)
    answer4 = vectorized_sol(args, is_not_true_scalar_fn, None)

    check_func(
        impl1,
        args,
        py_output=answer1,
        check_dtype=False,
        reset_index=True,
    )
    check_func(
        impl2,
        args,
        py_output=answer2,
        check_dtype=False,
        reset_index=True,
    )
    check_func(
        impl3,
        args,
        py_output=answer3,
        check_dtype=False,
        reset_index=True,
    )
    check_func(
        impl4,
        args,
        py_output=answer4,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.slow
def test_option_bool_fns(memory_leak_check):
    def impl(A, B, flag0, flag1):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        return (
            bodosql.kernels.booland(arg0, arg1),
            bodosql.kernels.boolor(arg0, arg1),
            bodosql.kernels.boolxor(arg0, arg1),
            bodosql.kernels.boolnot(arg0),
            bodosql.kernels.equal_null(arg0, arg1, is_scalar_a=True, is_scalar_b=True),
            bodosql.kernels.not_equal_null(
                arg0, arg1, is_scalar_a=True, is_scalar_b=True
            ),
        )

    for A in [0, 16]:
        for B in [0, 16]:
            for flag0 in [True, False]:
                for flag1 in [True, False]:
                    a = A if flag0 else None
                    b = B if flag1 else None
                    A0 = (
                        True
                        if a == 16 and b == 16
                        else (False if a == 0 or b == 0 else None)
                    )
                    A1 = (
                        True
                        if a == 16 or b == 16
                        else (False if a == 0 and b == 0 else None)
                    )
                    A2 = None if a == None or b == None else (a != b)
                    A3 = None if a == None else not a
                    A4 = a == b
                    A5 = a != b
                    check_func(
                        impl, (A, B, flag0, flag1), py_output=(A0, A1, A2, A3, A4, A5)
                    )


@pytest.mark.slow
def test_cond_option(memory_leak_check):
    def impl(A, B, C, flag0, flag1, flag2):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        arg2 = C if flag2 else None
        return bodosql.kernels.cond(arg0, arg1, arg2)

    for flag0 in [True, False]:
        for flag1 in [True, False]:
            for flag2 in [True, False]:
                if flag0:
                    answer = "A" if flag1 else None
                else:
                    answer = "B" if flag2 else None
                check_func(
                    impl, (True, "A", "B", flag0, flag1, flag2), py_output=answer
                )


@pytest.mark.slow
def test_option_nullif(memory_leak_check):
    def impl(A, B, flag0, flag1):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        return bodosql.kernels.nullif(arg0, arg1)

    A, B = 0.1, 0.5
    for flag0 in [True, False]:
        for flag1 in [True, False]:
            answer = None if not flag0 else 0.1
            check_func(impl, (A, B, flag0, flag1), py_output=answer)


@pytest.mark.slow
def test_option_regr_valxy(memory_leak_check):
    def impl(A, B, flag0, flag1):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        return (
            bodosql.kernels.regr_valx(arg0, arg1),
            bodosql.kernels.regr_valy(arg0, arg1),
        )

    A, B = 0.1, 0.5
    for flag0 in [True, False]:
        for flag1 in [True, False]:
            answer = (0.5, 0.1) if flag0 and flag1 else (None, None)
            check_func(impl, (A, B, flag0, flag1), py_output=answer)


@pytest.mark.slow
def test_option_is_functions(memory_leak_check):
    """
    Tests for the array kernels is_false, is_true,
    is_not_false, and is_not_true on optional data.
    """

    def impl(A, flag):
        arg = A if flag else None
        return (
            bodosql.kernels.is_false(arg),
            bodosql.kernels.is_true(arg),
            bodosql.kernels.is_not_false(arg),
            bodosql.kernels.is_not_true(arg),
        )

    A = True
    for flag in [True, False]:
        answer = (False, True, True, False) if flag else (False, False, True, True)
        check_func(impl, (A, flag), py_output=answer)


@pytest.mark.parametrize(
    "arr, ind, is_scalar_arr, expected",
    [
        pytest.param(
            np.array([1, 4, 9, 16, 25]),
            2,
            True,
            9,
            id="int_array-scalar-scalar",
        ),
        pytest.param(
            np.array([1, 2, 3, None, 4]),
            pd.Series([-1, 0, 1, 2, 3, 4, 5]),
            True,
            pd.Series([None, 1, 2, 3, None, 4, None], dtype=pd.Int32Dtype()),
            id="int_array-scalar-vector",
        ),
        pytest.param(
            pd.Series([[1, 2, 3, 4], [5, 6, None]] * 3),
            0,
            False,
            pd.Series([1, 5] * 3),
            id="int_array-vector-scalar",
        ),
        pytest.param(
            pd.Series([[1, 2, 3, None, 4]] * 7),
            pd.Series([-1, 0, 1, 2, 3, 4, 5]),
            False,
            pd.Series([None, 1, 2, 3, None, 4, None]),
            marks=pytest.mark.slow,
            id="int_array-vector-vector",
        ),
        pytest.param(
            pd.Series([[1.5, 2.333, -3.1, None, 4.0]] * 7),
            pd.Series([-1, 0, 1, 2, 3, 4, 5]),
            False,
            pd.Series([None, 1.5, 2.333, -3.1, None, 4.0, None]),
            marks=pytest.mark.slow,
            id="float_array-vector-vector",
        ),
        pytest.param(
            np.array(["abc", "def", "ghi", None] * 2),
            pd.Series([-1, 0, 1, 2, 3, 4, 8]),
            True,
            pd.Series(
                [None, "abc", "def", "ghi", None, "abc", None], dtype=pd.StringDtype()
            ),
            id="string_array-scalar-vector",
        ),
        pytest.param(
            pd.Series([[True, False, None]] * 5),
            pd.Series([-1, 0, 1, 2, 3]),
            False,
            pd.Series([None, True, False, None, None]),
            marks=pytest.mark.slow,
            id="bool_array-vector-vector",
        ),
        pytest.param(
            pd.Series(
                [
                    [
                        pd.Timestamp("2023-11-27", tz="UTC"),
                        pd.Timestamp("2023-11-27 21:11:54.764555+0000", tz="UTC"),
                        pd.Timestamp("2022-01-01 13:01:59", tz="UTC"),
                        pd.Timestamp("1999-09-09 09:09:09", tz="UTC"),
                        None,
                    ]
                ]
                * 7,
            ),
            pd.Series([-1, 0, 1, 2, 3, 4, 5]),
            False,
            pd.Series(
                [
                    None,
                    pd.Timestamp("2023-11-27", tz="UTC"),
                    pd.Timestamp("2023-11-27 21:11:54.764555+0000", tz="UTC"),
                    pd.Timestamp("2022-01-01 13:01:59", tz="UTC"),
                    pd.Timestamp("1999-09-09 09:09:09", tz="UTC"),
                    None,
                    None,
                ],
                dtype="datetime64[ns, UTC]",
            ),
            marks=pytest.mark.slow,
            id="timestamp_tz_array-vector-vector",
        ),
        pytest.param(
            pd.Series(
                [
                    [
                        pd.Timestamp("2023-11-27"),
                        pd.Timestamp("2023-11-27 21:11:54.764555"),
                        pd.Timestamp("2022-01-01 13:01:59"),
                    ]
                ]
                * 5,
            ),
            pd.Series([-1, 0, 1, 2, -1]),
            False,
            pd.Series(
                [
                    None,
                    pd.Timestamp("2023-11-27"),
                    pd.Timestamp("2023-11-27 21:11:54.764555"),
                    pd.Timestamp("2022-01-01 13:01:59"),
                    None,
                ],
                dtype="datetime64[ns]",
            ),
            marks=pytest.mark.slow,
            id="timestamp_ntz_array-vector-vector",
        ),
        pytest.param(
            pd.Series(
                [[1, 2], [3, 4], [5]] * 2,
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            ).values,
            1,
            True,
            pd.array([3, 4]),
            id="array_int_array-scalar-scalar",
        ),
        pytest.param(
            pd.Series(
                [[[1, 2], [3, 4], [5]] * 2] * 6,
                dtype=pd.ArrowDtype(pa.large_list(pa.large_list(pa.int32()))),
            ),
            pd.Series([1, 2] * 3),
            False,
            pd.Series([[3, 4], [5]] * 3),
            id="array_int_array-vector-vector",
        ),
        pytest.param(
            pd.array(
                [
                    {
                        "W": {"A": 1, "B": "A"},
                        "X": "AB",
                        "Y": [1.1, 2.2],
                        "Z": [[1], None, [3, None]],
                    },
                    {
                        "W": {"A": 1, "B": "ABC"},
                        "X": "C",
                        "Y": [1.1],
                        "Z": [[11], None],
                    },
                    None,
                    {
                        "W": {"A": 1, "B": ""},
                        "X": "D",
                        "Y": [4.0, np.nan],
                        "Z": [[1], None],
                    },
                    {
                        "W": {"A": 1, "B": "AA"},
                        "X": "VFD",
                        "Y": [1.2],
                        "Z": [[], [3, 1]],
                    },
                    {
                        "W": {"A": 1, "B": "DFG"},
                        "X": "LMMM",
                        "Y": [9.0, 1.2, 3.1],
                        "Z": [[10, 11], [11, 0, -3, -5]],
                    },
                ]
                * 2,
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [
                            pa.field(
                                "W",
                                pa.struct(
                                    pa.struct(
                                        [
                                            pa.field(
                                                "A",
                                                pa.int32(),
                                                pa.field("B", pa.string()),
                                            )
                                        ]
                                    )
                                ),
                            ),
                            pa.field("X", pa.string()),
                            pa.field("Y", pa.large_list(pa.float32())),
                            pa.field("Z", pa.large_list(pa.large_list(pa.int64()))),
                        ]
                    )
                ),
            ),
            0,
            True,
            {
                "W": {"A": 1, "B": "A"},
                "X": "AB",
                "Y": [1.1, 2.2],
                "Z": [[1], None, [3, None]],
            },
            id="struct_array-scalar-scalar",
        ),
        pytest.param(
            pd.array(
                [
                    {
                        "W": {"A": 1, "B": "A"},
                        "X": "AB",
                        "Y": [1.1, 2.2],
                        "Z": [[1], None, [3, None]],
                    },
                    {
                        "W": {"A": 1, "B": "ABC"},
                        "X": "C",
                        "Y": [1.1],
                        "Z": [[11], None],
                    },
                    None,
                    {
                        "W": {"A": 1, "B": ""},
                        "X": "D",
                        "Y": [4.0, np.nan],
                        "Z": [[1], None],
                    },
                    {
                        "W": {"A": 1, "B": "AA"},
                        "X": "VFD",
                        "Y": [1.2],
                        "Z": [[], [3, 1]],
                    },
                    {
                        "W": {"A": 1, "B": "DFG"},
                        "X": "LMMM",
                        "Y": [9.0, 1.2, 3.1],
                        "Z": [[10, 11], [11, 0, -3, -5]],
                    },
                ]
                * 2,
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [
                            pa.field(
                                "W",
                                pa.struct(
                                    [
                                        pa.field(
                                            "A", pa.int32(), pa.field("B", pa.string())
                                        )
                                    ]
                                ),
                            ),
                            pa.field("X", pa.string()),
                            pa.field("Y", pa.large_list(pa.float32())),
                            pa.field("Z", pa.large_list(pa.large_list(pa.int64()))),
                        ]
                    )
                ),
            ),
            pd.Series([-1, 0, 1, 2, 3, 4]),
            True,
            pd.Series(
                [
                    None,
                    {
                        "W": {"A": 1, "B": "A"},
                        "X": "AB",
                        "Y": [1.1, 2.2],
                        "Z": [[1], None, [3, None]],
                    },
                    {
                        "W": {"A": 1, "B": "ABC"},
                        "X": "C",
                        "Y": [1.1],
                        "Z": [[11], None],
                    },
                    None,
                    {
                        "W": {"A": 1, "B": ""},
                        "X": "D",
                        "Y": [4.0, np.nan],
                        "Z": [[1], None],
                    },
                    {
                        "W": {"A": 1, "B": "AA"},
                        "X": "VFD",
                        "Y": [1.2],
                        "Z": [[], [3, 1]],
                    },
                ],
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [
                            pa.field(
                                "W",
                                pa.struct(
                                    [
                                        pa.field(
                                            "A", pa.int32(), pa.field("B", pa.string())
                                        )
                                    ]
                                ),
                            ),
                            pa.field("X", pa.string()),
                            pa.field("Y", pa.large_list(pa.float32())),
                            pa.field("Z", pa.large_list(pa.large_list(pa.int64()))),
                        ]
                    )
                ),
            ),
            marks=pytest.mark.slow,
            id="struct_array-scalar-vector",
        ),
        pytest.param(
            pd.Series(
                [
                    {"A": 0, "B": 1},
                    {"A": 2, "C": 3},
                ]
                * 3,
                dtype=pd.ArrowDtype(pa.map_(pa.large_string(), pa.int8())),
            ).values,
            0,
            True,
            {"A": 0, "B": 1},
            id="map_array-scalar-scalar",
        ),
        pytest.param(
            pd.Series(
                [
                    {"A": 0, "B": 1},
                    {"A": 2, "C": 3},
                ],
                dtype=pd.ArrowDtype(pa.map_(pa.large_string(), pa.int8())),
            ).values,
            pd.Series([0, 1] * 3),
            True,
            pd.Series(
                [
                    {"A": 0, "B": 1},
                    {"A": 2, "C": 3},
                ]
                * 3,
                dtype=pd.ArrowDtype(pa.map_(pa.large_string(), pa.int8())),
            ),
            marks=pytest.mark.slow,
            id="map_array-scalar-vector",
        ),
        pytest.param(
            pd.Series(
                [
                    [
                        {"A": 0, "B": 1},
                        {"A": 2, "C": 3},
                    ]
                ]
                * 6,
                dtype=pd.ArrowDtype(
                    pa.large_list(pa.map_(pa.large_string(), pa.int8()))
                ),
            ),
            pd.Series([0, 1] * 3),
            False,
            pd.Series(
                [
                    {"A": 0, "B": 1},
                    {"A": 2, "C": 3},
                ]
                * 3,
                dtype=pd.ArrowDtype(pa.map_(pa.large_string(), pa.int8())),
            ),
            id="map_array-vector-vector",
        ),
    ],
)
def test_arr_get(arr, ind, is_scalar_arr, expected, memory_leak_check):
    is_scalar_ind = not isinstance(ind, pd.Series)
    both_scalar = is_scalar_arr and is_scalar_ind
    no_scalar = not is_scalar_arr and not is_scalar_ind

    if both_scalar:

        def impl(arr, ind):
            return bodosql.kernels.arr_get(arr, ind, is_scalar_arr, is_scalar_ind)

    else:

        def impl(arr, ind):
            return pd.Series(
                bodosql.kernels.arr_get(arr, ind, is_scalar_arr, is_scalar_ind)
            )

    check_func(
        impl,
        (arr, ind),
        py_output=expected,
        check_dtype=False,
        distributed=no_scalar,
        is_out_distributed=no_scalar,
        dist_test=no_scalar,
    )


@pytest.mark.slow
def test_option_arr_get(memory_leak_check):
    def impl(A, B, flag0, flag1):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        return bodosql.kernels.arr_get(arg0, arg1, True)

    for flag0 in [True]:
        for flag1 in [True, False]:
            answer = 9 if flag0 and flag1 else None
            check_func(
                impl,
                (np.array([1, 4, 9, 16]), 2, flag0, flag1),
                py_output=answer,
                dist_test=False,
            )


@pytest.mark.parametrize(
    "arr, ind, is_scalar_arr, is_scalar_idx, expected",
    [
        pytest.param(
            np.array([1, 4, 9, 16, 25]),
            -1,
            True,
            True,
            True,
            id="scalar-scalar-negative-idx",
        ),
        pytest.param(
            np.array([1, 4, 9, 16, 25]),
            100,
            True,
            True,
            True,
            id="scalar-scalar-out-of-bounds-idx",
        ),
        pytest.param(
            np.array([1, 4, 9, 16, 25]),
            "2",
            True,
            True,
            True,
            id="scalar-scalar",
        ),
        pytest.param(
            pd.Series([[1, 2, 3, 4], [5, 6, None]] * 6),
            True,
            False,
            True,
            pa.nulls(12),
            id="vector-scalar",
        ),
        pytest.param(
            pd.Series([[1, 2, 3, 4], [5, 6, None]] * 6),
            pd.Series(
                [
                    pd.Timestamp("2023-11-27 21:11:54.764555+0000", tz="UTC"),
                    pd.Timestamp("2001-01-01", tz="UTC"),
                ]
                * 6,
                dtype="datetime64[ns, UTC]",
            ),
            False,
            False,
            pa.nulls(12),
            id="vector-vector",
        ),
        pytest.param(
            pd.Series([[1, 2, 3, 4], [5, 6, None]] * 6),
            pd.Series([1, 2] * 6),
            False,
            True,
            pa.nulls(12),
            id="vector-scalar_array",
        ),
        pytest.param(
            np.array([1, 4, 9, 16, 25]),
            pd.Series([1, 2] * 3),
            True,
            True,
            True,
            id="scalar-scalar_array",
        ),
    ],
)
def test_arr_get_invalid(
    arr, ind, is_scalar_arr, is_scalar_idx, expected, memory_leak_check
):
    """
    Tests that the behavior of get on arrays with invalid/out of bound indices is to always return null.
    """
    both_scalar = is_scalar_arr and is_scalar_idx
    no_scalar = not is_scalar_arr and not is_scalar_idx
    if both_scalar:

        def impl(arr, ind):
            return (
                bodosql.kernels.arr_get(arr, ind, is_scalar_arr, is_scalar_arr) is None
            )

    else:

        def impl(arr, ind):
            return bodosql.kernels.arr_get(arr, ind, is_scalar_arr, is_scalar_idx)

    check_func(
        impl,
        (arr, ind),
        py_output=expected,
        check_dtype=False,
        distributed=no_scalar,
        is_out_distributed=no_scalar,
        dist_test=no_scalar,
        reset_index=True,
    )


# Used in the test below
internal_struct_type = pa.struct(
    [
        pa.field(
            "W",
            pa.struct(
                pa.struct(
                    [
                        pa.field(
                            "A",
                            pa.int32(),
                            pa.field("B", pa.string()),
                        )
                    ]
                )
            ),
        ),
        pa.field("X", pa.string()),
        pa.field("Y", pa.large_list(pa.float32())),
        pa.field("Z", pa.large_list(pa.large_list(pa.int64()))),
    ]
)


@pytest.fixture(
    params=[
        pytest.param(
            (
                {"A": 0, "B": 1},
                "A",
                True,
                0,
            ),
            id="int_map-scalar-scalar",
        ),
        pytest.param(
            (
                {"A": "B", "B": "C"},
                "B",
                True,
                "C",
            ),
            id="string_map-scalar-scalar",
        ),
        pytest.param(
            (
                {"A": 1, "B": "2", "C": 1.23},
                "C",
                True,
                1.23,
            ),
            id="mixed_map-scalar-scalar",
            marks=pytest.mark.skip(
                "Needs mixed value type scalar map type: https://bodo.atlassian.net/browse/BSE-2320"
            ),
        ),
        pytest.param(
            (
                {"A": 0, "B": 1, "C": None},
                pd.Series(["A", "B", "C", "D"]),
                True,
                pd.Series([0, 1, None, None], dtype=pd.Int32Dtype()),
            ),
            id="int_map-scalar-vector",
            marks=pytest.mark.skip(
                "Causes a memory leak: https://bodo.atlassian.net/browse/BSE-2440"
            ),
        ),
        pytest.param(
            (
                pd.array(
                    [
                        {"A": 0, "B": 1, "C": 2},
                        {"A": 0, "B": 1, "C": 2},
                        None,
                        {"D": 0, "C": 1, "B": 2},
                    ],
                    dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int64())),
                ),
                pd.Series(["A", "D", "C", "D"]),
                False,
                pd.Series([0, None, None, 0], dtype=pd.Int32Dtype()),
            ),
            id="int_map-vector-vector",
        ),
        pytest.param(
            (
                pd.array(
                    [
                        {"A": "A", "B": "B", "C": "C"},
                        {"A": "A", "B": "B", "C": "C"},
                        None,
                        {"D": "D", "C": "C", "B": "B"},
                    ],
                    dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.string())),
                ),
                pd.Series(["D", "B", "C", "D"]),
                False,
                pd.Series([None, "B", None, "D"]),
            ),
            id="string_map-vector-vector",
        ),
        pytest.param(
            (
                pd.array(
                    [
                        {
                            "A": 1,
                            "B": "2",
                            "C": 1.23,
                            "D": -1,
                            "E": 123,
                            "F": True,
                            "G": pd.Timestamp("2023-11-27"),
                        },
                        {
                            "A": -2,
                            "B": "3",
                            "C": 12.3,
                            "D": 0,
                            "E": -321,
                            "F": False,
                            "G": pd.Timestamp("2000-01-01"),
                        },
                        {
                            "A": 1,
                            "B": "-100",
                            "C": -123.45,
                            "D": 100,
                            "E": -101,
                            "F": False,
                            "G": pd.Timestamp("2111-11-11"),
                        },
                    ]
                    * 3,
                    dtype=pd.ArrowDtype(
                        pa.struct(
                            [
                                pa.field("A", pa.int64()),
                                pa.field("B", pa.string()),
                                pa.field("C", pa.float32()),
                                pa.field("D", pa.int64()),
                                pa.field("E", pa.int64()),
                                pa.field("F", pa.bool_()),
                                pa.field("G", pa.timestamp("ns")),
                            ]
                        )
                    ),
                ),
                "B",
                False,
                pd.Series(["2", "3", "-100"] * 3),
            ),
            id="mixed_map-vector-scalar-string-output",
        ),
        pytest.param(
            (
                pd.array(
                    [
                        {
                            "A": 1,
                            "B": "2",
                            "C": 1.23,
                            "D": -1,
                            "E": 123,
                            "F": True,
                            "G": pd.Timestamp("2023-11-27"),
                        },
                        {
                            "A": -2,
                            "B": "3",
                            "C": 12.3,
                            "D": 0,
                            "E": -321,
                            "F": False,
                            "G": pd.Timestamp("2000-01-01"),
                        },
                        {
                            "A": 1,
                            "B": "-100",
                            "C": -123.45,
                            "D": 100,
                            "E": -101,
                            "F": False,
                            "G": pd.Timestamp("2111-11-11"),
                        },
                    ]
                    * 3,
                    dtype=pd.ArrowDtype(
                        pa.struct(
                            [
                                pa.field("A", pa.int64()),
                                pa.field("B", pa.string()),
                                pa.field("C", pa.float32()),
                                pa.field("D", pa.int64()),
                                pa.field("E", pa.int64()),
                                pa.field("F", pa.bool_()),
                                pa.field("G", pa.timestamp("ns")),
                            ]
                        )
                    ),
                ),
                "C",
                False,
                pd.Series([1.23, 12.3, -123.45] * 3),
            ),
            id="mixed_map-vector-scalar-float-output",
        ),
        pytest.param(
            (
                pd.array(
                    [
                        {
                            "A": 1,
                            "B": "2",
                            "C": 1.23,
                            "D": -1,
                            "E": 123,
                            "F": True,
                            "G": pd.Timestamp("2023-11-27"),
                        },
                        {
                            "A": -2,
                            "B": "3",
                            "C": 12.3,
                            "D": 0,
                            "E": -321,
                            "F": False,
                            "G": pd.Timestamp("2000-01-01"),
                        },
                        {
                            "A": 1,
                            "B": "-100",
                            "C": -123.45,
                            "D": 100,
                            "E": -101,
                            "F": False,
                            "G": pd.Timestamp("2111-11-11"),
                        },
                    ]
                    * 3,
                    dtype=pd.ArrowDtype(
                        pa.struct(
                            [
                                pa.field("A", pa.int64()),
                                pa.field("B", pa.string()),
                                pa.field("C", pa.float32()),
                                pa.field("D", pa.int64()),
                                pa.field("E", pa.int64()),
                                pa.field("F", pa.bool_()),
                                pa.field("G", pa.timestamp("ns")),
                            ]
                        )
                    ),
                ),
                pd.Series(["A", "E", "D"] * 3),
                False,
                pd.Series([1, -321, -101] * 3),
            ),
            marks=pytest.mark.slow,
            id="mixed_map-vector-vector-int-output",
        ),
        pytest.param(
            (
                pd.array(
                    [
                        {
                            "A": 1,
                            "B": "2",
                            "C": 1.23,
                            "D": -1,
                            "E": 123,
                            "F": True,
                            "G": pd.Timestamp("2023-11-27"),
                        },
                        {
                            "A": -2,
                            "B": "3",
                            "C": 12.3,
                            "D": 0,
                            "E": -321,
                            "F": False,
                            "G": pd.Timestamp("2000-01-01"),
                        },
                        {
                            "A": 1,
                            "B": "-100",
                            "C": -123.45,
                            "D": 100,
                            "E": -101,
                            "F": False,
                            "G": pd.Timestamp("2111-11-11"),
                        },
                    ]
                    * 3,
                    dtype=pd.ArrowDtype(
                        pa.struct(
                            [
                                pa.field("A", pa.int64()),
                                pa.field("B", pa.string()),
                                pa.field("C", pa.float32()),
                                pa.field("D", pa.int64()),
                                pa.field("E", pa.int64()),
                                pa.field("F", pa.bool_()),
                                pa.field("G", pa.timestamp("ns")),
                            ]
                        )
                    ),
                ),
                pd.Series(["G"] * 9),
                False,
                pd.Series(
                    [
                        pd.Timestamp("2023-11-27"),
                        pd.Timestamp("2000-01-01"),
                        pd.Timestamp("2111-11-11"),
                    ]
                    * 9,
                    dtype="datetime64[ns]",
                ),
            ),
            marks=pytest.mark.slow,
            id="mixed_map-vector-vector-ts-output",
        ),
        pytest.param(
            (
                {
                    "A": {
                        "W": {"A": 1, "B": "A"},
                        "X": "AB",
                        "Y": [1.1, 2.2],
                        "Z": [[1], None, [3, None]],
                    },
                    "B": {
                        "W": {"A": 1, "B": "ABC"},
                        "X": "C",
                        "Y": [1.1],
                        "Z": [[11], None],
                    },
                    "C": {
                        "W": {"A": 1, "B": ""},
                        "X": "D",
                        "Y": [4.0, np.nan],
                        "Z": [[1], None],
                    },
                    "D": {
                        "W": {"A": 1, "B": "AA"},
                        "X": "VFD",
                        "Y": [1.2],
                        "Z": [[], [3, 1]],
                    },
                    "E": {
                        "W": {"A": 1, "B": "DFG"},
                        "X": "LMMM",
                        "Y": [9.0, 1.2, 3.1],
                        "Z": [[10, 11], [11, 0, -3, -5]],
                    },
                },
                "D",
                True,
                {
                    "W": {"A": 1, "B": "AA"},
                    "X": "VFD",
                    "Y": [1.2],
                    "Z": [[], [3, 1]],
                },
            ),
            id="nested_map-scalar-scalar",
            marks=pytest.mark.skip(
                "Needs mixed value type scalar map type: https://bodo.atlassian.net/browse/BSE-2320"
            ),
        ),
        pytest.param(
            (
                {
                    "A": {
                        "A": {
                            "W": {"A": 1, "B": "A"},
                            "X": "AB",
                            "Y": [1.1, 2.2],
                            "Z": [[1], None, [3, None]],
                        },
                        "B": {
                            "W": {"A": 1, "B": "ABC"},
                            "X": "C",
                            "Y": [1.1],
                            "Z": [[11], None],
                        },
                    },
                    "C": {
                        "C": {
                            "W": {"A": 1, "B": ""},
                            "X": "D",
                            "Y": [4.0, np.nan],
                            "Z": [[1], None],
                        },
                        "D": {
                            "W": {"A": 1, "B": "AA"},
                            "X": "VFD",
                            "Y": [1.2],
                            "Z": [[], [3, 1]],
                        },
                    },
                },
                "C",
                True,
                {
                    "C": {
                        "W": {"A": 1, "B": ""},
                        "X": "D",
                        "Y": [4.0, np.nan],
                        "Z": [[1], None],
                    },
                    "D": {
                        "W": {"A": 1, "B": "AA"},
                        "X": "VFD",
                        "Y": [1.2],
                        "Z": [[], [3, 1]],
                    },
                },
            ),
            id="nested_nested_map-scalar-scalar",
            marks=pytest.mark.skip(
                "Needs mixed value type scalar map type: https://bodo.atlassian.net/browse/BSE-2320"
            ),
        ),
        pytest.param(
            (
                pd.array(
                    [
                        {
                            "A": {
                                "W": {"A": 1, "B": "A"},
                                "X": "AB",
                                "Y": [1.1, 2.2],
                                "Z": [[1], None, [3, None]],
                            },
                            "B": {
                                "W": {"A": 1, "B": "ABC"},
                                "X": "C",
                                "Y": [1.1],
                                "Z": [[11], None],
                            },
                        },
                        {
                            "A": {
                                "W": {"A": 1, "B": ""},
                                "X": "D",
                                "Y": [4.0, np.nan],
                                "Z": [[1], None],
                            },
                            "B": {
                                "W": {"A": 1, "B": "AA"},
                                "X": "VFD",
                                "Y": [1.2],
                                "Z": [[], [3, 1]],
                            },
                        },
                        None,
                    ]
                    * 2,
                    dtype=pd.ArrowDtype(
                        pa.struct(
                            [
                                pa.field("A", internal_struct_type),
                                pa.field("B", internal_struct_type),
                            ]
                        )
                    ),
                ),
                "B",
                False,
                pd.Series(
                    pd.array(
                        [
                            {
                                "W": {"A": 1, "B": "ABC"},
                                "X": "C",
                                "Y": [1.1],
                                "Z": [[11], None],
                            },
                            {
                                "W": {"A": 1, "B": "AA"},
                                "X": "VFD",
                                "Y": [1.2],
                                "Z": [[], [3, 1]],
                            },
                            None,
                        ]
                        * 2,
                        dtype=pd.ArrowDtype(internal_struct_type),
                    )
                ),
            ),
            marks=pytest.mark.slow,
            id="nested_map-vector-scalar",
        ),
        pytest.param(
            (
                pd.array(
                    [
                        {
                            "A": {
                                "W": {"A": 1, "B": "A"},
                                "X": "AB",
                                "Y": [1.1, 2.2],
                                "Z": [[1], None, [3, None]],
                            },
                            "B": {
                                "W": {"A": 1, "B": "ABC"},
                                "X": "C",
                                "Y": [1.1],
                                "Z": [[11], None],
                            },
                        },
                        {
                            "A": {
                                "W": {"A": 1, "B": ""},
                                "X": "D",
                                "Y": [4.0, np.nan],
                                "Z": [[1], None],
                            },
                            "B": {
                                "W": {"A": 1, "B": "AA"},
                                "X": "VFD",
                                "Y": [1.2],
                                "Z": [[], [3, 1]],
                            },
                        },
                        None,
                    ]
                    * 2,
                    dtype=pd.ArrowDtype(
                        pa.struct(
                            [
                                pa.field("A", internal_struct_type),
                                pa.field("B", internal_struct_type),
                            ]
                        )
                    ),
                ),
                pd.Series(
                    ["B", "A", "A"] + [None, None, None],
                    dtype=pd.ArrowDtype(pa.string()),
                ),
                False,
                [
                    {
                        "W": {"A": 1, "B": "ABC"},
                        "X": "C",
                        "Y": [1.1],
                        "Z": [[11], None],
                    },
                    {
                        "W": {"A": 1, "B": ""},
                        "X": "D",
                        "Y": [4.0, np.nan],
                        "Z": [[1], None],
                    },
                    None,
                ]
                + [None, None, None],
            ),
            marks=pytest.mark.slow,
            id="nested_map-vector-vector",
        ),
    ],
)
def map_get_values(request):
    return request.param


def test_map_get(map_get_values, memory_leak_check):
    """Tests the functionality of the GET kernel on map values, for valid maps/indices."""

    map, ind, is_scalar_map, expected = map_get_values

    is_scalar_ind = not isinstance(ind, pd.Series)
    if (not is_scalar_ind) and (
        is_scalar_map or isinstance(map.dtype.pyarrow_dtype, pa.StructType)
    ):
        pytest.skip(
            "TODO: Support non-constant indices for map_get with pyarrow struct types"
        )

    both_scalar = is_scalar_map and is_scalar_ind
    no_scalar = not is_scalar_map and not is_scalar_ind

    def convertMapScalarHelper(map):
        pass

    @overload(convertMapScalarHelper)
    def convertMapScalarHelperUtil(map):
        if isinstance(map, types.DictType):

            def impl(map):
                # return map
                return pd.Series([map])[0]

        else:

            def impl(map):
                return map

        return impl

    if both_scalar:
        ind_lowered_as_global = ind

        def impl(map, ind):
            map = convertMapScalarHelper(map)
            return bodosql.kernels.arr_get(
                map, ind_lowered_as_global, is_scalar_map, is_scalar_ind
            )

    elif is_scalar_ind:
        ind_lowered_as_global = ind

        def impl(map, ind):
            return pd.Series(
                bodosql.kernels.arr_get(
                    map, ind_lowered_as_global, is_scalar_map, is_scalar_ind
                )
            )

    elif is_scalar_map:

        def impl(map, ind):
            map = convertMapScalarHelper(map)
            return pd.Series(
                bodosql.kernels.arr_get(map, ind, is_scalar_map, is_scalar_ind)
            )

    else:

        def impl(map, ind):
            return pd.Series(
                bodosql.kernels.arr_get(map, ind, is_scalar_map, is_scalar_ind)
            )

    check_func(
        impl,
        (map, ind),
        py_output=expected,
        check_dtype=False,
        distributed=no_scalar,
        is_out_distributed=no_scalar,
        dist_test=no_scalar,
    )


def test_get_ignore_case(map_get_values, memory_leak_check):
    """Tests the functionality of the GET kernel on map values, for valid maps/indices."""

    map, ind, is_scalar_map, expected = map_get_values
    is_scalar_ind = not isinstance(ind, pd.Series)

    is_scalar_ind = not isinstance(ind, pd.Series)

    if (not is_scalar_ind) and (
        is_scalar_map or isinstance(map.dtype.pyarrow_dtype, pa.StructType)
    ):
        pytest.skip(
            "TODO: Support non-constant indices for map_get with pyarrow struct types"
        )
    else:
        # Reverse the idx's capitalization
        def reverseCapitalization(s):
            newStr = ""
            for c in s:
                if c.isupper():
                    newStr += c.lower()
                else:
                    newStr += c.upper()
            return newStr

        if is_scalar_ind:
            ind = reverseCapitalization(ind)
        else:
            ind = ind.apply(lambda s: reverseCapitalization(s))

    both_scalar = is_scalar_map and is_scalar_ind
    no_scalar = not is_scalar_map and not is_scalar_ind

    def convertMapScalarHelper(map):
        pass

    @overload(convertMapScalarHelper)
    def convertMapScalarHelperUtil(map):
        if isinstance(map, types.DictType):

            def impl(map):
                return pd.Series([map])[0]

        else:

            def impl(map):
                return map

        return impl

    if both_scalar:
        ind_lowered_as_global = ind

        def impl(map, ind):
            map = convertMapScalarHelper(map)
            return bodosql.kernels.get_ignore_case(
                map, ind_lowered_as_global, is_scalar_map, is_scalar_ind
            )

    elif is_scalar_ind:
        ind_lowered_as_global = ind

        def impl(map, ind):
            return pd.Series(
                bodosql.kernels.get_ignore_case(
                    map, ind_lowered_as_global, is_scalar_map, is_scalar_ind
                )
            )

    elif is_scalar_map:

        def impl(map, ind):
            map = convertMapScalarHelper(map)
            return pd.Series(
                bodosql.kernels.get_ignore_case(map, ind, is_scalar_map, is_scalar_ind)
            )

    else:

        def impl(map, ind):
            return pd.Series(
                bodosql.kernels.get_ignore_case(map, ind, is_scalar_map, is_scalar_ind)
            )

    check_func(
        impl,
        (map, ind),
        py_output=expected,
        check_dtype=False,
        distributed=no_scalar,
        is_out_distributed=no_scalar,
        dist_test=no_scalar,
    )
