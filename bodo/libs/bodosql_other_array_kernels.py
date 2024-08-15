# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Implements miscellaneous array kernels that are specific to BodoSQL
"""


import numba
from llvmlite import ir
from numba.core import cgutils, types
from numba.core.typing import signature
from numba.cpython.randomimpl import get_next_int32, get_state_ptr
from numba.extending import intrinsic, overload

import bodo
from bodo.libs.bodosql_array_kernel_utils import *
from bodo.libs.bodosql_array_kernels import get_field
from bodo.utils.typing import (
    get_overload_const_bool,
    is_overload_none,
    raise_bodo_error,
)
from bodo.utils.utils import is_array_typ


@numba.generated_jit(nopython=True)
def booland(A, B):
    """Handles cases where BOOLAND receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [A, B]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.booland", ["A", "B"], i
            )

    def impl(A, B):  # pragma: no cover
        return booland_util(A, B)

    return impl


@numba.generated_jit(nopython=True)
def boolor(A, B):
    """Handles cases where BOOLOR receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [A, B]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.boolor", ["A", "B"], i
            )

    def impl(A, B):  # pragma: no cover
        return boolor_util(A, B)

    return impl


@numba.generated_jit(nopython=True)
def boolxor(A, B):
    """Handles cases where BOOLXOR receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [A, B]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.boolxor", ["A", "B"], i
            )

    def impl(A, B):  # pragma: no cover
        return boolxor_util(A, B)

    return impl


@numba.generated_jit(nopython=True)
def boolnot(A):
    """Handles cases where BOOLNOT receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(A, types.optional):  # pragma: no cover
        return unopt_argument("bodo.libs.bodosql_array_kernels.boolnot_util", ["A"], 0)

    def impl(A):  # pragma: no cover
        return boolnot_util(A)

    return impl


def cond(arr, ifbranch, elsebranch):  # pragma: no cover
    pass


@overload(cond)
def overload_cond(arr, ifbranch, elsebranch):
    """Handles cases where IF receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [arr, ifbranch, elsebranch]
    for i in range(3):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.cond",
                ["arr", "ifbranch", "elsebranch"],
                i,
            )

    def impl(arr, ifbranch, elsebranch):  # pragma: no cover
        return cond_util(arr, ifbranch, elsebranch)

    return impl


@numba.generated_jit(nopython=True)
def nvl2(arr, not_null_branch, null_branch):
    """Handles cases where NVL2 receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [arr, not_null_branch, null_branch]
    for i in range(3):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.nvl2",
                ["arr", "not_null_branch", "null_branch"],
                i,
            )

    def impl(arr, not_null_branch, null_branch):  # pragma: no cover
        return nvl2_util(arr, not_null_branch, null_branch)

    return impl


@numba.generated_jit(nopython=True)
def equal_null(
    A, B, is_scalar_a=False, is_scalar_b=False, dict_encoding_state=None, func_id=-1
):
    """Handles cases where EQUAL_NULL receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [A, B]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.equal_null",
                [
                    "A",
                    "B",
                    "is_scalar_a",
                    "is_scalar_b",
                    "dict_encoding_state",
                    "func_id",
                ],
                i,
                default_map={
                    "is_scalar_a": False,
                    "is_scalar_b": False,
                    "dict_encoding_state": None,
                    "func_id": -1,
                },
            )

    def impl(
        A, B, is_scalar_a=False, is_scalar_b=False, dict_encoding_state=None, func_id=-1
    ):  # pragma: no cover
        return equal_null_util(
            A, B, is_scalar_a, is_scalar_b, dict_encoding_state, func_id
        )

    return impl


@numba.generated_jit(nopython=True)
def not_equal_null(
    A, B, is_scalar_a=False, is_scalar_b=False, dict_encoding_state=None, func_id=-1
):
    """Handles cases where NOT_EQUAL_NULL receives optional arguments and forwards
    to the appropriate version of the real implementation. This is implemented
    by calling NOT on EQUAL_NULL"""
    args = [A, B]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.not_equal_null",
                [
                    "A",
                    "B",
                    "is_scalar_a",
                    "is_scalar_b",
                    "dict_encoding_state",
                    "func_id",
                ],
                i,
                default_map={
                    "is_scalar_a": False,
                    "is_scalar_b": False,
                    "dict_encoding_state": None,
                    "func_id": -1,
                },
            )

    def impl(
        A, B, is_scalar_a=False, is_scalar_b=False, dict_encoding_state=None, func_id=-1
    ):  # pragma: no cover
        return bodo.libs.bodosql_array_kernels.boolnot(
            equal_null_util(
                A, B, is_scalar_a, is_scalar_b, dict_encoding_state, func_id
            )
        )

    return impl


@numba.generated_jit(nopython=True)
def booland_util(A, B):
    """A dedicated kernel for the SQL function BOOLAND which takes in two numbers
    (or columns) and returns True if they are both not zero and not null,
    False if one of them is zero, and NULL otherwise.


    Args:
        A (numerical array/series/scalar): the first number(s) being operated on
        B (numerical array/series/scalar): the second number(s) being operated on

    Returns:
        boolean series/scalar: the AND of the number(s) with the specified null
        handling rules
    """

    verify_int_float_arg(A, "BOOLAND", "A")
    verify_int_float_arg(B, "BOOLAND", "B")

    arg_names = ["A", "B"]
    arg_types = [A, B]
    propagate_null = [False] * 2

    # A = scalar null, B = anything
    if A == bodo.none:
        propagate_null = [False, True]
        scalar_text = "if arg1 != 0:\n"
        scalar_text += "   bodo.libs.array_kernels.setna(res, i)\n"
        scalar_text += "else:\n"
        scalar_text += "   res[i] = False\n"

    # B = scalar null, A = anything
    elif B == bodo.none:
        propagate_null = [True, False]
        scalar_text = "if arg0 != 0:\n"
        scalar_text += "   bodo.libs.array_kernels.setna(res, i)\n"
        scalar_text += "else:\n"
        scalar_text += "   res[i] = False\n"

    elif bodo.utils.utils.is_array_typ(A, True):
        # A & B are both vectors
        if bodo.utils.utils.is_array_typ(B, True):
            scalar_text = "if bodo.libs.array_kernels.isna(A, i) and bodo.libs.array_kernels.isna(B, i):\n"
            scalar_text += "   bodo.libs.array_kernels.setna(res, i)\n"
            scalar_text += "elif bodo.libs.array_kernels.isna(A, i) and arg1 != 0:\n"
            scalar_text += "   bodo.libs.array_kernels.setna(res, i)\n"
            scalar_text += "elif bodo.libs.array_kernels.isna(B, i) and arg0 != 0:\n"
            scalar_text += "   bodo.libs.array_kernels.setna(res, i)\n"
            # This case is only triggered if A[i] and B[i] are both not null
            scalar_text += "else:\n"
            scalar_text += "   res[i] = (arg0 != 0) and (arg1 != 0)"

        # A is a vector, B is a non-null scalar
        else:
            scalar_text = "if bodo.libs.array_kernels.isna(A, i) and arg1 != 0:\n"
            scalar_text += "   bodo.libs.array_kernels.setna(res, i)\n"
            scalar_text += "else:\n"
            scalar_text += "   res[i] = (arg0 != 0) and (arg1 != 0)"

    # B is a vector, A is a non-null scalar
    elif bodo.utils.utils.is_array_typ(B, True):
        scalar_text = "if bodo.libs.array_kernels.isna(B, i) and arg0 != 0:\n"
        scalar_text += "   bodo.libs.array_kernels.setna(res, i)\n"
        scalar_text += "else:\n"
        scalar_text += "   res[i] = (arg0 != 0) and (arg1 != 0)"

    # A and B are both non-null scalars
    else:
        scalar_text = "res[i] = (arg0 != 0) and (arg1 != 0)"

    out_dtype = bodo.libs.bool_arr_ext.boolean_array_type

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def boolor_util(A, B):
    """A dedicated kernel for the SQL function BOOLOR which takes in two numbers
    (or columns) and returns True if at least one of them is not zero or null,
    False if both of them are equal to zero, and null otherwise


    Args:
        A (numerical array/series/scalar): the first number(s) being operated on
        B (numerical array/series/scalar): the second number(s) being operated on

    Returns:
        boolean series/scalar: the OR of the number(s) with the specified null
        handling rules
    """

    verify_int_float_arg(A, "BOOLOR", "A")
    verify_int_float_arg(B, "BOOLOR", "B")

    arg_names = ["A", "B"]
    arg_types = [A, B]
    propagate_null = [False] * 2

    # A = scalar null, B = anything
    if A == bodo.none:
        propagate_null = [False, True]
        scalar_text = "if arg1 == 0:\n"
        scalar_text += "   bodo.libs.array_kernels.setna(res, i)\n"
        scalar_text += "else:\n"
        scalar_text += "   res[i] = True\n"

    # B = scalar null, A = anything
    elif B == bodo.none:
        propagate_null = [True, False]
        scalar_text = "if arg0 == 0:\n"
        scalar_text += "   bodo.libs.array_kernels.setna(res, i)\n"
        scalar_text += "else:\n"
        scalar_text += "   res[i] = True\n"

    elif bodo.utils.utils.is_array_typ(A, True):
        # A & B are both vectors
        if bodo.utils.utils.is_array_typ(B, True):
            scalar_text = "if bodo.libs.array_kernels.isna(A, i) and bodo.libs.array_kernels.isna(B, i):\n"
            scalar_text += "   bodo.libs.array_kernels.setna(res, i)\n"
            scalar_text += "elif bodo.libs.array_kernels.isna(A, i) and arg1 != 0:\n"
            scalar_text += "   res[i] = True\n"
            scalar_text += "elif bodo.libs.array_kernels.isna(A, i) and arg1 == 0:\n"
            scalar_text += "   bodo.libs.array_kernels.setna(res, i)\n"
            scalar_text += "elif bodo.libs.array_kernels.isna(B, i) and arg0 != 0:\n"
            scalar_text += "   res[i] = True\n"
            scalar_text += "elif bodo.libs.array_kernels.isna(B, i) and arg0 == 0:\n"
            scalar_text += "   bodo.libs.array_kernels.setna(res, i)\n"
            # This case is only triggered if A[i] and B[i] are both not null
            scalar_text += "else:\n"
            scalar_text += "   res[i] = (arg0 != 0) or (arg1 != 0)"

        # A is a vector, B is a non-null scalar
        else:
            scalar_text = "if bodo.libs.array_kernels.isna(A, i) and arg1 != 0:\n"
            scalar_text += "   res[i] = True\n"
            scalar_text += "elif bodo.libs.array_kernels.isna(A, i) and arg1 == 0:\n"
            scalar_text += "   bodo.libs.array_kernels.setna(res, i)\n"
            scalar_text += "else:\n"
            scalar_text += "   res[i] = (arg0 != 0) or (arg1 != 0)"

    # B is a vector, A is a non-null scalar
    elif bodo.utils.utils.is_array_typ(B, True):
        scalar_text = "if bodo.libs.array_kernels.isna(B, i) and arg0 != 0:\n"
        scalar_text += "   res[i] = True\n"
        scalar_text += "elif bodo.libs.array_kernels.isna(B, i) and arg0 == 0:\n"
        scalar_text += "   bodo.libs.array_kernels.setna(res, i)\n"
        scalar_text += "else:\n"
        scalar_text += "   res[i] = (arg0 != 0) or (arg1 != 0)"

    # A and B are both non-null scalars
    else:
        scalar_text = "res[i] = (arg0 != 0) or (arg1 != 0)"

    out_dtype = bodo.libs.bool_arr_ext.boolean_array_type

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def boolxor_util(A, B):
    """A dedicated kernel for the SQL function BOOLXOR which takes in two numbers
    (or columns) and returns True if one of them is zero and the other is nonzero,
    NULL if either input is NULL, and False otherwise


    Args:
        A (numerical array/series/scalar): the first number(s) being operated on
        B (numerical array/series/scalar): the second number(s) being operated on

    Returns:
        boolean series/scalar: the XOR of the number(s) with the specified null
        handling rules
    """

    verify_int_float_arg(A, "BOOLXOR", "A")
    verify_int_float_arg(B, "BOOLXOR", "B")

    arg_names = ["A", "B"]
    arg_types = [A, B]
    propagate_null = [True] * 2

    scalar_text = "res[i] = (arg0 == 0) != (arg1 == 0)"

    out_dtype = bodo.libs.bool_arr_ext.boolean_array_type

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def boolnot_util(A):
    """A dedicated kernel for the SQL function BOOLNOT which takes in a number
    (or column) and returns True if it is zero, False if it is nonzero, and
    NULL if it is NULL.


    Args:
        A (numerical array/series/scalar): the number(s) being operated on

    Returns:
        boolean series/scalar: the NOT of the number(s) with the specified null
        handling rules
    """

    verify_int_float_arg(A, "BOOLNOT", "A")

    arg_names = ["A"]
    arg_types = [A]
    propagate_null = [True]

    scalar_text = "res[i] = arg0 == 0"

    out_dtype = bodo.libs.bool_arr_ext.boolean_array_type

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def nullif(arr0, arr1, dict_encoding_state=None, func_id=-1):
    """Handles cases where NULLIF receives optional arguments and forwards
    to args appropriate version of the real implementation"""
    args = [arr0, arr1]
    for i, arg in enumerate(args):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.nullif",
                ["arr0", "arr1", "dict_encoding_state=None", "func_id=-1"],
                i,
            )

    def impl(arr0, arr1, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return nullif_util(arr0, arr1, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True)
def regr_valx(y, x):
    """Handles cases where REGR_VALX receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [y, x]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.regr_valx",
                ["y", "x"],
                i,
            )

    def impl(y, x):  # pragma: no cover
        return regr_valx_util(y, x)

    return impl


@numba.generated_jit(nopython=True)
def regr_valy(y, x):
    """Handles cases where REGR_VALY receives optional arguments and forwards
    to the appropriate version of the real implementation (recycles regr_valx
    by swapping the order of the arguments)"""
    args = [y, x]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.regr_valy",
                ["y", "x"],
                i,
            )

    def impl(y, x):  # pragma: no cover
        return regr_valx(x, y)

    return impl


def cond_util(arr, ifbranch, elsebranch):  # pragma: no cover
    pass


@overload(cond_util)
def overload_cond_util(arr, ifbranch, elsebranch):
    """A dedicated kernel for the SQL function IF which takes in 3 values:
    a boolean (or boolean column) and two values (or columns) with the same
    type and returns the first or second value depending on whether the boolean
    is true or false


    Args:
        arr (boolean array/series/scalar): the T/F values
        ifbranch (any array/series/scalar): the value(s) to return when true
        elsebranch (any array/series/scalar): the value(s) to return when false

    Returns:
        int series/scalar: the difference in months between the two dates
    """

    verify_boolean_arg(arr, "cond", "arr")

    # Both branches cannot be scalar nulls if the output is an array
    # (causes a typing ambiguity). This shouldn't occur in practice.
    # See testIFF in SimplificationTest
    # TODO: Replace with null array for robustness.
    if (
        bodo.utils.utils.is_array_typ(arr, True)
        and ifbranch == bodo.none
        and elsebranch == bodo.none
    ):
        raise_bodo_error("Both branches of IF() cannot be scalar NULL")

    arg_names = ["arr", "ifbranch", "elsebranch"]
    arg_types = [arr, ifbranch, elsebranch]
    propagate_null = [False] * 3
    # If the conditional is an array, add a null check (null = False)
    if bodo.utils.utils.is_array_typ(arr, True):
        scalar_text = "if (not bodo.libs.array_kernels.isna(arr, i)) and arg0:\n"
    # If the conditional is a non-null scalar, case on its truthiness
    elif arr != bodo.none:
        scalar_text = "if arg0:\n"
    # Skip the ifbranch if the conditional is a scalar None (since we know that
    # the condition is always false)
    else:
        scalar_text = ""
    if arr != bodo.none:
        # If the ifbranch is an array, add a null check
        if bodo.utils.utils.is_array_typ(ifbranch, True):
            scalar_text += "   if bodo.libs.array_kernels.isna(ifbranch, i):\n"
            scalar_text += "      bodo.libs.array_kernels.setna(res, i)\n"
            scalar_text += "   else:\n"
            scalar_text += "      res[i] = arg1\n"
        # If the ifbranch is a scalar null, just set to null
        elif ifbranch == bodo.none:
            scalar_text += "   bodo.libs.array_kernels.setna(res, i)\n"
        # If the ifbranch is a non-null scalar, then no null check is required
        else:
            scalar_text += "   res[i] = arg1\n"
        scalar_text += "else:\n"
    # If the elsebranch is an array, add a null check
    if bodo.utils.utils.is_array_typ(elsebranch, True):
        scalar_text += "   if bodo.libs.array_kernels.isna(elsebranch, i):\n"
        scalar_text += "      bodo.libs.array_kernels.setna(res, i)\n"
        scalar_text += "   else:\n"
        scalar_text += "      res[i] = arg2\n"
    # If the elsebranch is a scalar null, just set to null
    elif elsebranch == bodo.none:
        scalar_text += "   bodo.libs.array_kernels.setna(res, i)\n"
    # If the elsebranch is a non-null scalar, then no null check is required
    else:
        scalar_text += "   res[i] = arg2\n"

    # Get the common dtype from the two branches
    out_dtype = get_common_broadcasted_type([ifbranch, elsebranch], "IF")

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
    )


@numba.generated_jit(nopython=True)
def nvl2_util(arr, not_null_branch, null_branch):
    """A dedicated kernel for the SQL function NVL2 which is equivalent
    cases on whether the first argument is null to return the second
    argument (if not null) or the third argument (if null)


    Args:
        arr (boolean array/series/scalar): the value(s) having nulls checked
        not_null_branch (any array/series/scalar): the value(s) to return when arr is not null
        null_branch (any array/series/scalar): the value(s) to return when arr is null

    Returns:
        any series/scalar: the selected values from the branch arguments
    """

    arg_names = ["arr", "not_null_branch", "null_branch"]
    arg_types = [arr, not_null_branch, null_branch]
    propagate_null = [False] * 3

    if bodo.utils.utils.is_array_typ(arr, True):
        # Both branches cannot be scalar nulls if the output is an array
        # (causes a typing ambiguity)
        if (
            bodo.utils.utils.is_array_typ(arr, True)
            and not_null_branch == bodo.none
            and null_branch == bodo.none
        ):  # pragma: no cover
            raise_bodo_error("Both branches of NVL2() cannot be scalar NULL")

        scalar_text = "if not bodo.libs.array_kernels.isna(arr, i):\n"
        if bodo.utils.utils.is_array_typ(not_null_branch, True):
            scalar_text += "  if bodo.libs.array_kernels.isna(not_null_branch, i):\n"
            scalar_text += "    bodo.libs.array_kernels.setna(res, i)\n"
            scalar_text += "  else:\n"
            scalar_text += "    res[i] = arg1\n"
        elif not_null_branch == bodo.none:
            scalar_text += "    bodo.libs.array_kernels.setna(res, i)\n"
        else:
            scalar_text += "    res[i] = arg1\n"
        scalar_text += "else:\n"
        if bodo.utils.utils.is_array_typ(null_branch, True):
            scalar_text += "  if bodo.libs.array_kernels.isna(null_branch, i):\n"
            scalar_text += "    bodo.libs.array_kernels.setna(res, i)\n"
            scalar_text += "  else:\n"
            scalar_text += "    res[i] = arg2\n"
        elif null_branch == bodo.none:
            scalar_text += "    bodo.libs.array_kernels.setna(res, i)\n"
        else:
            scalar_text += "    res[i] = arg2\n"
    else:
        # If the first argument is a scalar, either always return
        # the second argument or always return the third argument.
        if arr == bodo.none:
            scalar_text = "res[i] = arg2"
        else:
            scalar_text = "res[i] = arg1"

    # Get the common dtype from the two branches
    out_dtype = get_common_broadcasted_type([not_null_branch, null_branch], "NVL2")

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
    )


@numba.generated_jit(nopython=True, no_unliteral=True)
def equal_null_util(A, B, is_scalar_a, is_scalar_b, dict_encoding_state, func_id):
    """A dedicated kernel for the SQL function EQUAL_NULL which takes in two values
    (or columns) and returns True if they are equal (where NULL is treated as
    a known value)

    Args:
        A (any array/series/scalar): the first value(s) being compared
        B (any array/series/scalar): the second value(s) being compared
        is_scalar_a (boolean): True if A should be treated as a scalar
        is_scalar_b (boolean): True if B should be treated as a scalar

    Returns:
        boolean series/scalar: whether the number(s) are equal, or both null
    """
    is_scalar_a = get_overload_const_bool(is_scalar_a, "equal_null", "is_scalar_a")
    is_scalar_b = get_overload_const_bool(is_scalar_b, "equal_null", "is_scalar_b")

    are_arrays = [not is_scalar_a, not is_scalar_b, False, False, False, False]
    typ_a = A.data if bodo.hiframes.pd_series_ext.is_series_type(A) else A
    typ_b = B.data if bodo.hiframes.pd_series_ext.is_series_type(B) else B
    dtype_a = typ_a if is_scalar_a else typ_a.dtype
    dtype_b = typ_b if is_scalar_b else typ_b.dtype

    arg_names = [
        "A",
        "B",
        "is_scalar_a",
        "is_scalar_b",
        "dict_encoding_state",
        "func_id",
    ]
    arg_types = [A, B, is_scalar_a, is_scalar_b, dict_encoding_state, func_id]
    propagate_null = [False] * 6

    # If there is no unified type for A and B, then we should return false for
    # every row.
    unified_typ, _ = bodo.utils.typing.get_common_scalar_dtype(
        [dtype_a, dtype_b], allow_downcast=True
    )

    if A == bodo.none:
        # A = scalar null, B = scalar null
        if B == bodo.none:
            scalar_text = "res[i] = True"

        # A = scalar null, B is a vector
        elif are_arrays[1]:
            scalar_text = "res[i] = bodo.libs.array_kernels.isna(B, i)"

        # A = scalar null, B = non-null scalar
        else:
            scalar_text = "res[i] = False"

    elif B == bodo.none:
        # A is a vector, B = null
        if are_arrays[0]:
            scalar_text = "res[i] = bodo.libs.array_kernels.isna(A, i)"

        # A = non-null scalar, B = null
        else:
            scalar_text = "res[i] = False"
    # A & B are both vectors
    elif are_arrays[0] and are_arrays[1]:
        if unified_typ == None:
            # Incompatible inner types
            scalar_text = "res[i] = False"
        else:
            scalar_text = "if bodo.libs.array_kernels.isna(A, i) and bodo.libs.array_kernels.isna(B, i):\n"
            scalar_text += "   res[i] = True\n"
            scalar_text += "elif bodo.libs.array_kernels.isna(A, i) or bodo.libs.array_kernels.isna(B, i):\n"
            scalar_text += "   res[i] = False\n"
            scalar_text += "else:\n"
            scalar_text += "   res[i] = semi_safe_equals(arg0, arg1)"
    # A is a vector, B is a non-null scalar
    elif are_arrays[0]:
        scalar_text = "res[i] = (not bodo.libs.array_kernels.isna(A, i)) and semi_safe_equals(arg0, arg1)"
    # B is a vector, A is a non-null scalar
    elif are_arrays[1]:
        scalar_text = "res[i] = (not bodo.libs.array_kernels.isna(B, i)) and semi_safe_equals(arg0, arg1)"
    # A and B are both non-null scalars
    else:
        scalar_text = "res[i] = semi_safe_equals(arg0, arg1)"

    out_dtype = bodo.libs.bool_arr_ext.boolean_array_type
    use_dict_caching = not is_overload_none(dict_encoding_state)
    extra_globals = {
        "semi_safe_equals": bodo.libs.bodosql_array_kernels.semi_safe_equals
    }
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
        extra_globals=extra_globals,
        are_arrays=are_arrays,
    )


@numba.generated_jit(nopython=True)
def nullif_util(arr0, arr1, dict_encoding_state, func_id):
    """A dedicated kernel for the SQL function NULLIF which takes in two
    scalars (or columns), which returns NULL if the two values are equal, and
    arg0 otherwise.


    Args:
        arg0 (array/series/scalar): The 0-th argument. This value is returned if
            the two arguments are equal.
        arg1 (array/series/scalar): The 1st argument.

    Returns:
        string series/scalar: the string/column of formatted numbers
    """

    arg_names = ["arr0", "arr1", "dict_encoding_state", "func_id"]
    arg_types = [arr0, arr1, dict_encoding_state, func_id]
    # If the first argument is NULL, the output is always NULL
    propagate_null = [True, False, False, False]
    # NA check needs to come first here, otherwise the equality check misbehaves

    if arr1 == bodo.none:
        scalar_text = "res[i] = arg0\n"
    elif bodo.utils.utils.is_array_typ(arr1, True):
        scalar_text = "if bodo.libs.array_kernels.isna(arr1, i) or arg0 != arg1:\n"
        scalar_text += "   res[i] = arg0\n"
        scalar_text += "else:\n"
        scalar_text += "   bodo.libs.array_kernels.setna(res, i)"
    else:
        scalar_text = "if arg0 != arg1:\n"
        scalar_text += "   res[i] = arg0\n"
        scalar_text += "else:\n"
        scalar_text += "   bodo.libs.array_kernels.setna(res, i)"

    out_dtype = get_common_broadcasted_type([arr0, arr1], "NULLIF", supress_error=True)
    if out_dtype is None:
        # If the types are incompatible but that wasn't caught by the type
        # checker in BodoSQL, then we must have VARIANT inputs. In that case,
        # we know that the two values will never be equal, so we just return
        # the first argument.
        def impl(arr0, arr1, dict_encoding_state, func_id):
            return arr0

        return impl

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        # We need to remove NAs because we treat them as duplicates.
        # TODO: Avoid this in the future.
        may_cause_duplicate_dict_array_values=True,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True)
def regr_valx_util(y, x):
    """A dedicated kernel for the SQL function REGR_VALX which takes in two numbers
    (or columns) and returns NULL if the first argument is NULL, otherwise the
    second argument

    Args:
        y (float array/series/scalar): the number(s) whose null-ness is preserved
        x (float array/series/scalar): the number(s) whose output is copied if no-null

    Returns:
        float series/scalar: a copy of x, but where nulls from y are propagated
    """
    verify_int_float_arg(y, "regr_valx", "y")
    verify_int_float_arg(x, "regr_valx", "x")

    arg_names = ["y", "x"]
    arg_types = [y, x]
    propogate_null = [True] * 2
    scalar_text = "res[i] = arg1"

    out_dtype = bodo.libs.float_arr_ext.FloatingArrayType(bodo.float64)

    return gen_vectorized(arg_names, arg_types, propogate_null, scalar_text, out_dtype)


def is_true(arr):  # pragma: no cover
    pass


def is_false(arr):  # pragma: no cover
    pass


def is_not_true(arr):  # pragma: no cover
    pass


def is_not_false(arr):  # pragma: no cover
    pass


def is_true_util(arr):  # pragma: no cover
    pass


def is_false_util(arr):  # pragma: no cover
    pass


def is_not_true_util(arr):  # pragma: no cover
    pass


def is_not_false_util(arr):  # pragma: no cover
    pass


def create_is_func_overload(fn_name):  # pragma: no cover
    def overload_func(arr):
        """Handles functions for x is [NOT] TRUE/FALSE. These have special
        NA handling that requires custom kernels."""
        if isinstance(arr, types.optional):
            return unopt_argument(
                f"bodo.libs.bodosql_array_kernels.{fn_name}_util",
                ["arr"],
                0,
            )

        func_text = "def impl(arr):\n"
        func_text += f"  return bodo.libs.bodosql_array_kernels.{fn_name}_util(arr)"
        loc_vars = {}
        exec(func_text, {"bodo": bodo}, loc_vars)

        return loc_vars["impl"]

    return overload_func


def create_is_func_util_overload(fn_name):  # pragma: no cover
    """Create overload functions to handle for x is [NOT] TRUE/FALSE. These have special
        NA handling that requires custom kernels.

    Args:
        fn_name: the function being implemented

    Returns:
        (function): a utility that takes in a boolean value and returns the appropriate
        value.
    """

    def overload_is_func(arr):
        verify_boolean_arg(arr, fn_name, "arr")
        arg_names = ["arr"]
        arg_types = [arr]
        # NA has custom handling
        propagate_null = [False]
        if "not" in fn_name:
            # e.g. NULL IS NOT FALSE == TRUE
            na_output = True
            operator_str = "!="
        else:
            # e.g. NULL IS FALSE == False
            na_output = False
            operator_str = "=="

        if "true" in fn_name:
            target_val = True
        else:
            target_val = False

        scalar_text = ""
        if bodo.utils.utils.is_array_typ(arr, True):
            scalar_text += "if bodo.libs.array_kernels.isna(arr, i):\n"
        else:
            scalar_text += "if arr is None:\n"
        scalar_text += f"  res[i] = {na_output}\n"
        scalar_text += "else:\n"
        scalar_text += f"  res[i] = arg0 {operator_str} {target_val}\n"

        # These functions can't output null so we switch to a non-nullable
        # array.
        out_dtype = bodo.boolean_array_type
        return gen_vectorized(
            arg_names, arg_types, propagate_null, scalar_text, out_dtype
        )

    return overload_is_func


def _install_is_overload():
    """Creates and installs the overloads for is functions.
    These are functions with custom null handling that map
    NA -> Booleans"""
    funcs_utils_names = [
        ("is_true", is_true, is_true_util),
        ("is_false", is_false, is_false_util),
        ("is_not_true", is_not_true, is_not_true_util),
        ("is_not_false", is_not_false, is_not_false_util),
    ]
    for fn_name, func, util in funcs_utils_names:
        func_overload_impl = create_is_func_overload(fn_name)
        overload(func)(func_overload_impl)
        util_overload_impl = create_is_func_util_overload(fn_name)
        overload(util)(util_overload_impl)


_install_is_overload()


@numba.njit(no_cpython_wrapper=True)
def ensure_single_value(A):  # pragma: no cover
    """Implements Calcite's SINGLE_VALUE, which returns input if it has only one value.
    Otherwise raises an error.
    https://github.com/apache/calcite/blob/f14cf4c32b9079984a988bbad40230aa6a59b127/core/src/main/java/org/apache/calcite/sql/fun/SqlSingleValueAggFunction.java#L36

    Args:
        A (Series | array): input column with single value

    Raises:
        ValueError: error if input has more than one value

    Returns:
        Series | array: same as input
    """
    if len(A) != 1:
        raise ValueError("Expected single value in column")

    return A


@intrinsic
def gen_random_int64(typingcontext):
    """A subset of the numba implementation of random.randrange.
    Designed to always output 1 random 64-bit integer, with the
    start/stop/step being assumed as int_min/int_max/1 and all of the
    checks that are required for arbitrary values removed.

    The implementation obtains a pointer to the randomization state for
    Python's random module (as opposed to the seperate pointer for numpy)
    and uses it to generate two more random 32-bit integers which are
    then concatenated.

    The code was derived from the following:
    https://github.com/numba/numba/blob/e4ff3cf5fcb59e91fc7d46c340f2e7eb664dd0c0/numba/cpython/randomimpl.py

    This monkeypatch was required because the standard numba implementation
    of random.randrange was not conducive to producing random integers
    from the entire domain of 64 bit integers due to several arithmetic
    checks that were in place to verify the inputs. These checks are useful
    for arbitrary inputs which could be invalid, but in this case are unecessary
    (because the constants are known) and harmful (because they cause an
    overflow whcih in turn leads to invalid results)
    """
    int_ty = types.Integer.from_bitwidth(64, True)

    def codegen(context, builder, sig, args):
        int64_t = ir.IntType(64)
        c32 = ir.Constant(int64_t, 32)
        state_ptr = get_state_ptr(context, builder, "py")
        ret = cgutils.alloca_once_value(builder, ir.Constant(int64_t, 0))
        low = get_next_int32(context, builder, state_ptr)
        high = get_next_int32(context, builder, state_ptr)
        total = builder.add(
            builder.zext(low, int64_t), builder.shl(builder.zext(high, int64_t), c32)
        )
        builder.store(total, ret)
        return builder.load(ret)

    return signature(int_ty), codegen


@numba.generated_jit(nopython=True)
def random_seedless(A):
    """Kernel for the BodoSQL function RANDOM() when no seed is provided.
       No wrapper function is required


    Args:
        A (any series/array/scalar): either a null input to indicate that the
        output is scalar, or a vector input whose length should be matched by
        the output.

    Returns:
        int64 array/scalar: either one random value (if the input was a scalar)
        or an array of random values (if the input was a vector) where the
        length matches the input. The values are random 64 bit integers.

    """
    if A == bodo.none:

        def impl(A):  # pragma: no cover
            return np.int64(gen_random_int64())

    else:

        def impl(A):  # pragma: no cover
            n = len(A)
            res = bodo.libs.int_arr_ext.alloc_int_array(n, np.int64)
            numba.parfors.parfor.init_prange()
            for i in numba.parfors.parfor.internal_prange(n):
                res[i] = gen_random_int64()
            return res

    return impl


@numba.generated_jit(nopython=True)
def uniform(lo, hi, gen):
    """Handles cases where UNIFORM receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(gen, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.uniform_util", ["lo", "hi", "gen"], 2
        )

    def impl(lo, hi, gen):  # pragma: no cover
        return uniform_util(lo, hi, gen)

    return impl


@numba.generated_jit(nopython=True)
def uniform_util(lo, hi, gen):
    """Kernel for the BodoSQL function UNIFORM()

    Args:
        lo (int/float scalar): the lower bound of the distribution
        hi (int/float scalar): the upper bound of the distribution
        gen (int64 scalar/array): the generating sequence for the randomness.
        If a scalar, then one random value is produced. If a vector, then one
        value is produced for each value in the array. The generating sequence
        works such that two rows with the same value for gen will produce the
        same random output, but if the rows are distinct then the outputs are
        random (e.g. gen is related to the seed for the randomness).

    Returns:
        int/float array/scalar: a uniform distribution of integers from the domain
        [lo,hi]. If both lo and hi are integers then the output is an integer,
        otherwise it is a float. Duplicate inputs for the gen input will result
        in duplicate outputs.

    Important note: this function uses the NumPy random module instead of the
    random module because if the same module were used as the random kernel,
    then any seed-setting done with this kernel could affect the random kernel.
    """
    lo_int = isinstance(lo, types.Integer)
    hi_int = isinstance(hi, types.Integer)
    lo_float = isinstance(lo, types.Float)
    hi_float = isinstance(hi, types.Float)
    assert lo_int or lo_float, "Input 'lo' to UNIFORM must be a scalar number"
    assert hi_int or hi_float, "Input 'hi' to UNIFORM must be a scalar number"

    verify_int_arg(gen, "UNIFORM", "gen")

    arg_names = ["lo", "hi", "gen"]
    arg_types = [lo, hi, gen]
    propagate_null = [False] * 3

    scalar_text = "np.random.seed(arg2)\n"

    if lo_int and hi_int:
        scalar_text += f"res[i] = np.random.randint(arg0, arg1+1)"
        out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int64)
    else:
        scalar_text += f"res[i] = np.random.uniform(arg0, arg1)"
        out_dtype = bodo.libs.float_arr_ext.FloatingArrayType(bodo.float64)

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
    )


def arr_get(arr, ind, is_scalar_arr=False, is_scalar_idx=True):  # pragma: no cover
    # Dummy function used for overload
    pass


@overload(arr_get, no_unliteral=True)
def overload_arr_get(arr, ind, is_scalar_arr=False, is_scalar_idx=True):
    """Handles cases where GET receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [arr, ind]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.arr_get",
                ["arr", "ind", "is_scalar_arr", "is_scalar_idx"],
                i,
                default_map={"is_scalar_arr": False, "is_scalar_idx": True},
            )

    def impl(arr, ind, is_scalar_arr=False, is_scalar_idx=True):  # pragma: no cover
        return arr_get_util(arr, ind, is_scalar_arr, is_scalar_idx)

    return impl


def arr_get_util(arr, ind, is_scalar_arr, is_scalar_idx):  # pragma: no cover
    # Dummy function used for overload
    pass


@overload(arr_get_util, no_unliteral=True)
def overload_arr_get_util(arr, ind, is_scalar_arr, is_scalar_idx):
    """
    A dedicated kernel for the SQL function GET which takes in an array/map
    and an index, and returns the elements at that index/key of the array/map.
    For invalid inputs, null/null array is returned.

    Args:
        arr (array/column of arrays): the data array(s)
        ind (integer/column of integers): the index/indices. Null is returned if ind is invalid or out of bounds.
        is_scalar_arr: if true, treats the inputs as scalar arrays i.e. a single element of an array of arrays

    Returns:
        arr's inner type/column of inner type: the element at ind of array arr
    """

    arg_names = ["arr", "ind", "is_scalar_arr", "is_scalar_idx"]
    arg_types = [arr, ind, is_scalar_arr, is_scalar_idx]
    propagate_null = [True, True, False, False]

    is_scalar_arr_bool = get_overload_const_bool(
        is_scalar_arr, "arr_get", "is_scalar_arr"
    )

    is_scalar_idx_bool = get_overload_const_bool(
        is_scalar_idx, "arr_get", "is_scalar_idx"
    )

    if is_valid_array_get(arr, ind, is_scalar_arr_bool, is_scalar_idx_bool):
        # Handle array[int_idx] case
        dtype = (
            arr.data.dtype
            if bodo.hiframes.pd_series_ext.is_series_type(arr)
            else arr.dtype
        )
        arr_type = (
            bodo.utils.typing.dtype_to_array_type(dtype)
            if is_scalar_arr_bool
            else dtype
        )
        out_dtype = bodo.utils.typing.to_nullable_type(arr_type)
        scalar_text = "if arg1 < 0 or arg1 >= len(arg0) or bodo.libs.array_kernels.isna(arg0, arg1):\n"
        scalar_text += "   bodo.libs.array_kernels.setna(res, i)\n"
        scalar_text += "else:\n"
        scalar_text += "   res[i] = arg0[arg1]"
    elif is_valid_object_get(arr, ind, is_scalar_arr_bool, is_scalar_idx_bool):

        def impl(arr, ind, is_scalar_arr, is_scalar_idx):
            return get_field(arr, ind, is_scalar_arr, False)

        return impl

    else:
        # In all other cases, return null
        out_dtype = bodo.null_array_type
        scalar_text = "bodo.libs.array_kernels.setna(res, i)\n"

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        are_arrays=[not is_scalar_arr_bool, not is_scalar_idx_bool, False, False],
    )


def get_ignore_case(
    arr, ind, is_scalar_arr=False, is_scalar_idx=True
):  # pragma: no cover
    # Dummy function used for overload
    pass


@overload(get_ignore_case, no_unliteral=True)
def overload_get_ignore_case(arr, ind, is_scalar_arr=False, is_scalar_idx=True):
    """Handles cases where GET_IGNORE_CASE receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [arr, ind]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.get_ignore_case",
                ["arr", "ind", "is_scalar_arr", "is_scalar_idx"],
                i,
                default_map={"is_scalar_arr": False, "is_scalar_idx": True},
            )

    def impl(arr, ind, is_scalar_arr=False, is_scalar_idx=True):  # pragma: no cover
        return get_ignore_case_util(arr, ind, is_scalar_arr, is_scalar_idx)

    return impl


def get_ignore_case_util(arr, ind, is_scalar_arr, is_scalar_idx):  # pragma: no cover
    # Dummy function used for overload
    pass


@overload(get_ignore_case_util, no_unliteral=True)
def overload_get_ignore_case_util(arr, ind, is_scalar_arr, is_scalar_idx):
    """
    A dedicated kernel for the SQL function GET_IGNORE_CASE which takes in an map
    and an key, and returns the element for that key (case insensitive).
    For invalid inputs, null/null array is returned.

    Args:
        map (object/column of objects): the data array
        ind (string/variant column): the index/indices. Null is returned if ind is not a valid key.
        is_scalar_arr: if true, treats the array input as a scalar arrays i.e. a single element of an array of arrays
        is_scalar_idx: if true, treats the index as a scalar index i.e. a single element of an array of arrays said values

    Returns:
        the value(s) of the map for the specified key(s) of the data arr
    """

    arg_names = ["arr", "ind", "is_scalar_arr", "is_scalar_idx"]
    arg_types = [arr, ind, is_scalar_arr, is_scalar_idx]
    propagate_null = [True, True, False, False]

    is_scalar_arr_bool = get_overload_const_bool(
        is_scalar_arr, "arr_get", "is_scalar_arr"
    )

    is_scalar_idx_bool = get_overload_const_bool(
        is_scalar_idx, "arr_get", "is_scalar_arr"
    )

    # GET_IGNORE_CASE only works for object type
    if is_valid_object_get(arr, ind, is_scalar_arr_bool, is_scalar_idx_bool):

        def impl(arr, ind, is_scalar_arr, is_scalar_idx):
            return get_field(arr, ind, is_scalar_arr, True)

        return impl
    else:
        # In all other cases, return null
        out_dtype = bodo.null_array_type
        scalar_text = "bodo.libs.array_kernels.setna(res, i)\n"
        return gen_vectorized(
            arg_names,
            arg_types,
            propagate_null,
            scalar_text,
            out_dtype,
            are_arrays=[not is_scalar_arr_bool, not is_scalar_idx_bool, False, False],
        )


def is_valid_array_get(arg0_type, arg1_type, arg0_scalar, arg1_scalar):
    """Returns true if the input is a valid GET operation on an array type
    That is, arg0 is an array type, and arg1 is a int type.

    Note that we don't check for scalar nulls here, since that is implicitly handled by the
    null array case.
    """

    # Check if arg0 is an array type
    if not (
        is_array_item_array(arg0_type)
        or (arg0_scalar and is_array_typ(arg0_type, True))
    ):
        return False

    # Check if arg1 is an int type
    if not is_valid_int_arg(arg1_type):
        return False

    # is_valid_int_arg will return true if it's an int or an int array.
    # The input is scalar, we need to disallow int array, because indexing
    # an array by an array in GET should return null.
    if arg1_scalar and bodo.utils.utils.is_array_typ(arg1_type, True):
        return False

    return True


def is_valid_object_get(arg0_type, arg1_type, arg0_scalar, arg1_scalar):
    """
    Returns true if the input is a valid GET operation on an object type.
    That is, arg0 is an object type, and arg1 is a string type.

    Note that we don't check for scalar nulls here, since that is implicitly handled by the
    null array case.
    """

    # Check if arg0 is an object type
    if not is_valid_SQL_object_arg(arg0_type):
        return False

    # Check if arg1 is an string type
    if not is_valid_string_arg(arg1_type):
        return False

    # is_valid_int_arg will return true if it's an int or an int array.
    # The input is scalar, we need to disallow string array, because indexing
    # an object by an array in GET should return null.
    if arg1_scalar and bodo.utils.utils.is_array_typ(arg1_type, True):
        return False

    return True
