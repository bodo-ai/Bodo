# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Implements miscellaneous array kernels that are specific to BodoSQL
"""

import numba
from numba.core import types

import bodo
from bodo.libs.bodosql_array_kernel_utils import *
from bodo.utils.typing import raise_bodo_error


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


@numba.generated_jit(nopython=True)
def cond(arr, ifbranch, elsebranch):
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
def equal_null(A, B):
    """Handles cases where EQUAL_NULL receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [A, B]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.equal_null", ["A", "B"], i
            )

    def impl(A, B):  # pragma: no cover
        return equal_null_util(A, B)

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

    out_dtype = bodo.libs.bool_arr_ext.boolean_array

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def boolor_util(A, B):
    """A dedicated kernel for the SQL function BOOLOR which takes in two numbers
    (or columns) and returns True if at least one of them is not zero or null,
    False if both of them are equal to zero, and null otheriwse


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

    out_dtype = bodo.libs.bool_arr_ext.boolean_array

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

    out_dtype = bodo.libs.bool_arr_ext.boolean_array

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

    out_dtype = bodo.libs.bool_arr_ext.boolean_array

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def nullif(arr0, arr1):
    """Handles cases where NULLIF recieves optional arguments and forwards
    to args appropriate version of the real implementation"""
    args = [arr0, arr1]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.nullif", ["arr0", "arr1"], i
            )

    def impl(arr0, arr1):  # pragma: no cover
        return nullif_util(arr0, arr1)

    return impl


@numba.generated_jit(nopython=True)
def regr_valx(y, x):
    """Handles cases where REGR_VALX receives optional arguments and forwards
    to the apropriate version of the real implementaiton"""
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
    to the apropriate version of the real implementaiton (recycles regr_valx
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


@numba.generated_jit(nopython=True)
def cond_util(arr, ifbranch, elsebranch):
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
    # (causes a typing ambiguity)
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
def equal_null_util(A, B):
    """A dedicated kernel for the SQL function EQUAL_NULL which takes in two values
    (or columns) and returns True if they are equal (where NULL is treated as
    a known value)

    Args:
        A (any array/series/scalar): the first value(s) being compared
        B (any array/series/scalar): the second value(s) being compared

    Returns:
        boolean series/scalar: whether the number(s) are equal, or both null
    """

    get_common_broadcasted_type([A, B], "EQUAL_NULL")

    arg_names = ["A", "B"]
    arg_types = [A, B]
    propagate_null = [False] * 2

    if A == bodo.none:
        # A = scalar null, B = scalar null
        if B == bodo.none:
            scalar_text = "res[i] = True"

        # A = scalar null, B is a vector
        elif bodo.utils.utils.is_array_typ(B, True):
            scalar_text = "res[i] = bodo.libs.array_kernels.isna(B, i)"

        # A = scalar null, B = non-null scalar
        else:
            scalar_text = "res[i] = False"

    elif B == bodo.none:

        # A is a vector, B = null
        if bodo.utils.utils.is_array_typ(A, True):
            scalar_text = "res[i] = bodo.libs.array_kernels.isna(A, i)"

        # A = non-null scalar, B = null
        else:
            scalar_text = "res[i] = False"

    elif bodo.utils.utils.is_array_typ(A, True):

        # A & B are both vectors
        if bodo.utils.utils.is_array_typ(B, True):
            scalar_text = "if bodo.libs.array_kernels.isna(A, i) and bodo.libs.array_kernels.isna(B, i):\n"
            scalar_text += "   res[i] = True\n"
            scalar_text += "elif bodo.libs.array_kernels.isna(A, i) or bodo.libs.array_kernels.isna(B, i):\n"
            scalar_text += "   res[i] = False\n"
            scalar_text += "else:\n"
            scalar_text += "   res[i] = arg0 == arg1"

        # A is a vector, B is a non-null scalar
        else:
            scalar_text = (
                "res[i] = (not bodo.libs.array_kernels.isna(A, i)) and arg0 == arg1"
            )

    # B is a vector, A is a non-null scalar
    elif bodo.utils.utils.is_array_typ(B, True):
        scalar_text = (
            "res[i] = (not bodo.libs.array_kernels.isna(B, i)) and arg0 == arg1"
        )

    # A and B are both non-null scalars
    else:
        scalar_text = "res[i] = arg0 == arg1"

    out_dtype = bodo.libs.bool_arr_ext.boolean_array

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def nullif_util(arr0, arr1):
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

    arg_names = ["arr0", "arr1"]
    arg_types = [arr0, arr1]
    # If the first argument is NULL, the output is always NULL
    propagate_null = [True, False]
    # NA check needs to come first here, otherwise the equalify check misbehaves

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

    out_dtype = get_common_broadcasted_type([arr0, arr1], "NULLIF")

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


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

    out_dtype = types.Array(bodo.float64, 1, "C")

    return gen_vectorized(arg_names, arg_types, propogate_null, scalar_text, out_dtype)
