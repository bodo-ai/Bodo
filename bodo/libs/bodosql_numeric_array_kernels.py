# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Implements numerical array kernels that are specific to BodoSQL
"""


import numba
from numba.core import types
from numba.extending import overload

import bodo
from bodo.libs.bodosql_array_kernel_utils import *
from bodo.utils.utils import is_array_typ


def cbrt(arr):  # pragma: no cover
    return


def factorial(arr):  # pragma: no cover
    return


def mod(arr0, arr1):  # pragma: no cover
    return


def sign(arr):  # pragma: no cover
    return


def sqrt(arr):  # pragma: no cover
    return


def round(arr0, arr1):  # pragma: no cover
    return


def trunc(arr0, arr1):  # pragma: no cover
    return


def abs(arr):  # pragma: no cover
    return


def ln(arr):  # pragma: no cover
    return


def log2(arr):  # pragma: no cover
    return


def log10(arr):  # pragma: no cover
    return


def exp(arr):  # pragma: no cover
    return


def power(arr0, arr1):  # pragma: no cover
    return


def sqrt_util(arr):  # pragma: no cover
    return


def square(arr):  # pragma: no cover
    return


def cbrt_util(arr):  # pragma: no cover
    return


def factorial_util(arr):  # pragma: no cover
    return


def mod_util(arr0, arr1):  # pragma: no cover
    return


def sign_util(arr):  # pragma: no cover
    return


def round_util(arr0, arr1):  # pragma: no cover
    return


def trunc_util(arr0, arr1):  # pragma: no cover
    return


def abs_util(arr):  # pragma: no cover
    return


def ln_util(arr):  # pragma: no cover
    return


def log2_util(arr):  # pragma: no cover
    return


def log10_util(arr):  # pragma: no cover
    return


def exp_util(arr):  # pragma: no cover
    return


def power_util(arr0, arr1):  # pragma: no cover
    return


def square_util(arr):  # pragma: no cover
    return


funcs_utils_names = (
    (abs, abs_util, "ABS"),
    (cbrt, cbrt_util, "CBRT"),
    (factorial, factorial_util, "FACTORIAL"),
    (ln, ln_util, "LN"),
    (log2, log2_util, "LOG2"),
    (log10, log10_util, "LOG10"),
    (mod, mod_util, "MOD"),
    (sign, sign_util, "SIGN"),
    (round, round_util, "ROUND"),
    (trunc, trunc_util, "TRUNC"),
    (exp, exp_util, "EXP"),
    (power, power_util, "POWER"),
    (sqrt, sqrt_util, "SQRT"),
    (square, square_util, "SQUARE"),
)
double_arg_funcs = (
    "MOD",
    "TRUNC",
    "POWER",
    "ROUND",
)

single_arg_funcs = set(a[2] for a in funcs_utils_names if a[2] not in double_arg_funcs)

_float = {
    16: types.float16,
    32: types.float32,
    64: types.float64,
}
_int = {
    8: types.int8,
    16: types.int16,
    32: types.int32,
    64: types.int64,
}
_uint = {
    8: types.uint8,
    16: types.uint16,
    32: types.uint32,
    64: types.uint64,
}


def _get_numeric_output_dtype(func_name, arr0, arr1=None):
    """
    Helper function that returns the expected output_dtype for given input
    dtype(s) func_name.
    """
    arr0_dtype = arr0.dtype if is_array_typ(arr0) else arr0
    arr1_dtype = arr1.dtype if is_array_typ(arr1) else arr1
    # default to float64 without further information
    out_dtype = bodo.float64
    if (arr0 is None or arr0_dtype == bodo.none) or (
        func_name in double_arg_funcs and (arr1 is None or arr1_dtype == bodo.none)
    ):
        if isinstance(out_dtype, types.Float):
            return bodo.libs.float_arr_ext.FloatingArrayType(out_dtype)
        else:
            return types.Array(out_dtype, 1, "C")

    # if input is float32 rather than float64, switch the default output dtype to float32
    if isinstance(arr0_dtype, types.Float):
        if isinstance(arr1_dtype, types.Float):
            out_dtype = _float[max(arr0_dtype.bitwidth, arr1_dtype.bitwidth)]
        else:
            out_dtype = arr0_dtype
    if func_name == "SIGN":
        # we match the bitwidth of the input if we are using an integer
        # (matching float bitwidth is handled above)
        if isinstance(arr0_dtype, types.Integer):
            out_dtype = arr0_dtype
    elif func_name == "MOD":
        if isinstance(arr0_dtype, types.Integer) and isinstance(
            arr1_dtype, types.Integer
        ):
            if arr0_dtype.signed:
                if arr1_dtype.signed:
                    out_dtype = arr1_dtype
                else:
                    # If arr0 is signed and arr1 is unsigned, our output may be signed
                    # and may must support a bitwidth of double arr1.
                    # e.g. say arr0_dtype = bodo.int64, arr1_dtype = bodo.uint16,
                    # we know 0 <= arr1 <= 2^(15) - 1, however the output is based off
                    # the  sign of arr0 and thus we need to support signed ints
                    # of _double_ the bitwidth, -2^(15) <= arr <= 2^(15) - 1, so
                    # we use out_dtype = bodo.int32.
                    out_dtype = _int[min(64, arr1_dtype.bitwidth * 2)]
            else:
                # if arr0 is unsigned, we will use the dtype of arr1
                out_dtype = arr1_dtype
    elif func_name == "ABS":
        # if arr0 is a signed integer, we will use and unsigned integer of double the bitwidth,
        # following the same reasoning as noted in the above comment for MOD.
        if isinstance(arr0_dtype, types.Integer):
            if arr0_dtype.signed:
                out_dtype = _uint[min(64, arr0_dtype.bitwidth * 2)]
            else:
                out_dtype = arr0_dtype
    elif func_name in ("ROUND", "FLOOR", "CEIL"):
        #
        # can use types.Number, but this would include types.Complex
        if isinstance(arr0_dtype, (types.Float, types.Integer)):
            out_dtype = arr0_dtype
    elif func_name == "FACTORIAL":
        # the output of factorial is always a 64-bit integer
        # TODO: support 128-bit to match Snowflake
        out_dtype = bodo.int64

    if isinstance(out_dtype, types.Integer):
        return bodo.libs.int_arr_ext.IntegerArrayType(out_dtype)
    elif isinstance(out_dtype, types.Float):
        return bodo.libs.float_arr_ext.FloatingArrayType(out_dtype)
    else:
        return types.Array(out_dtype, 1, "C")


def create_numeric_func_overload(func_name):
    """
    Returns the appropriate numeric function that will overload the given function name.
    """

    if func_name not in double_arg_funcs:
        func_name = func_name.lower()

        def overload_func(arr):
            """Handles cases where func_name receives an optional argument and forwards
            to the appropriate version of the real implementation"""
            if isinstance(arr, types.optional):
                return unopt_argument(
                    f"bodo.libs.bodosql_array_kernels.{func_name}", ["arr"], 0
                )

            func_text = "def impl(arr):\n"
            func_text += (
                f"  return bodo.libs.bodosql_array_kernels.{func_name}_util(arr)"
            )
            loc_vars = {}
            exec(func_text, {"bodo": bodo}, loc_vars)

            return loc_vars["impl"]

    else:
        func_name = func_name.lower()

        def overload_func(arr0, arr1):
            """Handles cases where func_name receives an optional argument and forwards
            to the appropriate version of the real implementation"""
            args = [arr0, arr1]
            for i in range(2):
                if isinstance(args[i], types.optional):
                    return unopt_argument(
                        f"bodo.libs.bodosql_array_kernels.{func_name}",
                        ["arr0", "arr1"],
                        i,
                    )

            func_text = "def impl(arr0, arr1):\n"
            func_text += (
                f"  return bodo.libs.bodosql_array_kernels.{func_name}_util(arr0, arr1)"
            )
            loc_vars = {}
            exec(func_text, {"bodo": bodo}, loc_vars)

            return loc_vars["impl"]

    return overload_func


def create_numeric_util_overload(func_name):  # pragma: no cover
    """Creates an overload function to support trig functions on
       a string array representing a column of a SQL table

    Args:
        func_name: which trig function is being called (e.g. "ACOS")

    Returns:
        (function): a utility that takes in one argument and returns
        the appropriate trig function applied to the argument, where the
        argument could be an array/scalar/null.
    """

    if func_name not in double_arg_funcs:

        def overload_numeric_util(arr):
            verify_int_float_arg(arr, func_name, "arr")

            arg_names = [
                "arr",
            ]
            arg_types = [arr]
            propagate_null = [True]
            scalar_text = ""
            if func_name in single_arg_funcs:
                if func_name == "FACTORIAL":
                    scalar_text += "if arg0 > 20 or np.abs(np.int64(arg0)) != arg0:\n"
                    scalar_text += "  bodo.libs.array_kernels.setna(res, i)\n"
                    scalar_text += "else:\n"
                    scalar_text += f"  res[i] = np.math.factorial(np.int64(arg0))"
                elif func_name == "LN":
                    scalar_text += f"res[i] = np.log(arg0)"
                else:
                    scalar_text += f"res[i] = np.{func_name.lower()}(arg0)"
            else:
                ValueError(f"Unknown function name: {func_name}")

            out_dtype = _get_numeric_output_dtype(func_name, arr)

            return gen_vectorized(
                arg_names, arg_types, propagate_null, scalar_text, out_dtype
            )

    else:

        def overload_numeric_util(arr0, arr1):
            verify_int_float_arg(arr0, func_name, "arr0")
            verify_int_float_arg(arr1, func_name, "arr1")

            arg_names = [
                "arr0",
                "arr1",
            ]
            arg_types = [arr0, arr1]
            propagate_null = [True, True]
            # we calculate out_dtype beforehand for determining if we can use
            # a more efficient MOD implementation
            out_dtype = _get_numeric_output_dtype(func_name, arr0, arr1)
            scalar_text = ""
            # we select the appropriate scalar text based on the function name
            if func_name == "MOD":
                # There is a discrepancy between numpy and SQL mod, whereby SQL mod returns the sign
                # of the divisor, whereas numpy mod and Python's returns the sign of the dividend,
                # so we need to use the equivalent of np.fmod / C equivalent to match SQL's behavior.
                # np.fmod is currently broken in numba [BE-3184] so we use an equivalent implementation.
                scalar_text += "if arg1 == 0:\n"
                scalar_text += "  bodo.libs.array_kernels.setna(res, i)\n"
                scalar_text += "else:\n"
                scalar_text += (
                    "  res[i] = np.sign(arg0) * np.mod(np.abs(arg0), np.abs(arg1))"
                )
            elif func_name == "POWER":
                scalar_text += "res[i] = np.power(np.float64(arg0), arg1)"
            elif func_name == "ROUND":
                scalar_text += "res[i] = round_half_always_up(arg0, arg1)"
            elif func_name == "TRUNC":
                scalar_text += "if int(arg1) == arg1:\n"
                # numpy truncates to the integer nearest to zero, so we shift by the number of decimals as appropriate
                # to get the desired result. (multiplication is used to maintain precision)
                scalar_text += (
                    "  res[i] = np.trunc(arg0 * (10.0 ** arg1)) * (10.0 ** -arg1)\n"
                )
                scalar_text += "else:\n"
                scalar_text += "  bodo.libs.array_kernels.setna(res, i)"
            else:
                raise ValueError(f"Unknown function name: {func_name}")

            extra_globals = {
                "round_half_always_up": round_half_always_up,
                "ceil": math.ceil,
                "floor": math.floor,
            }

            return gen_vectorized(
                arg_names,
                arg_types,
                propagate_null,
                scalar_text,
                out_dtype,
                extra_globals=extra_globals,
            )

    return overload_numeric_util


def _install_numeric_overload(funcs_utils_names):
    """Creates and installs the overloads for trig functions"""
    for func, util, func_name in funcs_utils_names:
        func_overload_impl = create_numeric_func_overload(func_name)
        overload(func)(func_overload_impl)
        util_overload_impl = create_numeric_util_overload(func_name)
        overload(util)(util_overload_impl)


_install_numeric_overload(funcs_utils_names)


@numba.generated_jit(nopython=True)
def round_half_always_up(x, places):
    """Variant of builtin round() that avoids Python's tie-breaking behavior:

                             x: 0.5, 1.5, 2.5, 3.5
                   round(x, 0): 0.0, 2.0, 2.0, 4.0
    round_half_always_up(x, 0): 0.0, 1.0, 2.0, 3.0

    Args:
        x (integer/float): the number to be rounded
        places (integer): how many places to round to (can be positive or negative)

    Returns:
        integer/float: x rounded to the specified number of places (same type as x)
    """
    func_text = "def impl(x, places):\n"
    func_text += "  sign = -1 if x < 0 else 1\n"
    func_text += "  x = abs(x)\n"
    func_text += "  shift = 10 ** abs(places)\n"
    func_text += "  if places < 0:\n"
    func_text += "    x /= shift\n"
    func_text += "    r = x % 1\n"
    func_text += "    if r < 0.5: x = floor(x)\n"
    func_text += "    else: x = ceil(x)\n"
    func_text += "    x *= shift\n"
    func_text += "  else:\n"
    func_text += "    x *= shift\n"
    func_text += "    r = x % 1\n"
    func_text += "    if r < 0.5: x = floor(x)\n"
    func_text += "    else: x = ceil(x)\n"
    func_text += "    x /= shift\n"
    if is_valid_int_arg(x):
        func_text += "  x = np.int64(x)\n"
    func_text += "  return x * sign\n"
    loc_vars = {}
    exec(
        func_text,
        {"bodo": bodo, "floor": math.floor, "ceil": math.ceil, "np": np},
        loc_vars,
    )
    impl = loc_vars["impl"]
    return impl


@numba.generated_jit(nopython=True)
def floor(data, precision):  # pragma: no cover
    """Handles cases where FLOOR receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [data, precision]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.floor",
                ["data", "precision"],
                i,
            )

    def impl(data, precision):  # pragma: no cover
        return floor_util(data, precision)

    return impl


@numba.generated_jit(nopython=True)
def ceil(data, precision):  # pragma: no cover
    """Handles cases where CEIL receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [data, precision]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.ceil",
                ["data", "precision"],
                i,
            )

    def impl(data, precision):  # pragma: no cover
        return ceil_util(data, precision)

    return impl


@numba.generated_jit(nopython=True)
def floor_util(data, precision):  # pragma: no cover
    """A dedicated kernel for the SQL function FLOOR which takes in
    a numerical scalar/column and a number of places and rounds the
    number to that number of places, always rounding down.
    Args:
        data (numerical array/series/scalar): the data to round.
        precision (integer array/series/scalar): the number of places to round to.
    Returns:
        numerical series/scalar: the data rounded down.
    """
    verify_int_float_arg(data, "floor", "data")
    verify_int_arg(precision, "floor", "precision")

    arg_names = ["data", "precision"]
    arg_types = [data, precision]
    propagate_null = [True] * 2

    if is_valid_int_arg(data):
        data_dtype = data.dtype if is_array_typ(data) else data
        cast_expr = ""
        if data_dtype.signed:
            cast_expr = f"np.int{data_dtype.bitwidth}"
        else:
            cast_expr = f"np.uint{data_dtype.bitwidth}"
        scalar_text = (
            f"res[i] = {cast_expr}(floor(arg0 * (10.0 ** arg1)) / (10.0 ** arg1))"
        )
    else:
        scalar_text = "res[i] = floor(arg0 * (10.0 ** arg1)) / (10.0 ** arg1)"

    out_dtype = _get_numeric_output_dtype("FLOOR", data)

    extra_globals = {"floor": math.floor}

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        extra_globals=extra_globals,
    )


@numba.generated_jit(nopython=True)
def ceil_util(data, precision):  # pragma: no cover
    """A dedicated kernel for the SQL function CEIL which takes in
    a numerical scalar/column and a number of places and rounds the
    number to that number of places, always rounding up.
    Args:
        data (numerical array/series/scalar): the data to round.
        precision (integer array/series/scalar): the number of places to round to.
    Returns:
        numerical series/scalar: the data rounded up.
    """
    verify_int_float_arg(data, "ceil", "data")
    verify_int_arg(precision, "ceil", "precision")

    arg_names = ["data", "precision"]
    arg_types = [data, precision]
    propagate_null = [True] * 2

    if is_valid_int_arg(data):
        scalar_text = "res[i] = np.int64(ceil(arg0 * (10.0 ** arg1)) / (10.0 ** arg1))"
    else:
        scalar_text = "res[i] = ceil(arg0 * (10.0 ** arg1)) / (10.0 ** arg1)"

    out_dtype = _get_numeric_output_dtype("CEIL", data)

    extra_globals = {"ceil": math.ceil}

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        extra_globals=extra_globals,
    )


@numba.generated_jit(nopython=True)
def bitand(A, B):
    """Handles cases where BITAND receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [A, B]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.bitand",
                ["A", "B"],
                i,
            )

    def impl(A, B):  # pragma: no cover
        return bitand_util(A, B)

    return impl


@numba.generated_jit(nopython=True)
def bitshiftleft(A, B):
    """Handles cases where BITSHIFTLEFT receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [A, B]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.bitshiftleft",
                ["A", "B"],
                i,
            )

    def impl(A, B):  # pragma: no cover
        return bitshiftleft_util(A, B)

    return impl


@numba.generated_jit(nopython=True)
def bitnot(A):
    """Handles cases where BITNOT receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(A, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.bitnot_util",
            ["A"],
            0,
        )

    def impl(A):  # pragma: no cover
        return bitnot_util(A)

    return impl


@numba.generated_jit(nopython=True)
def bitor(A, B):
    """Handles cases where BITOR receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [A, B]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.bitor",
                ["A", "B"],
                i,
            )

    def impl(A, B):  # pragma: no cover
        return bitor_util(A, B)

    return impl


@numba.generated_jit(nopython=True)
def bitshiftright(A, B):
    """Handles cases where BITSHIFTRIGHT receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [A, B]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.bitshiftright",
                ["A", "B"],
                i,
            )

    def impl(A, B):  # pragma: no cover
        return bitshiftright_util(A, B)

    return impl


@numba.generated_jit(nopython=True)
def bitxor(A, B):
    """Handles cases where BITXOR receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [A, B]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.bitxor",
                ["A", "B"],
                i,
            )

    def impl(A, B):  # pragma: no cover
        return bitxor_util(A, B)

    return impl


@numba.generated_jit(nopython=True)
def conv(arr, old_base, new_base):
    """Handles cases where CONV receives optional arguments and forwards
    to args appropriate version of the real implementation"""
    args = [arr, old_base, new_base]
    for i in range(3):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.conv",
                ["arr", "old_base", "new_base"],
                i,
            )

    def impl(arr, old_base, new_base):  # pragma: no cover
        return conv_util(arr, old_base, new_base)

    return impl


@numba.generated_jit(nopython=True)
def getbit(A, B):
    """Handles cases where GETBIT receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [A, B]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.getbit",
                ["A", "B"],
                i,
            )

    def impl(A, B):  # pragma: no cover
        return getbit_util(A, B)

    return impl


@numba.generated_jit(nopython=True)
def haversine(lat1, lon1, lat2, lon2):
    """
    Handles cases where HAVERSINE receives optional arguments and forwards
    to the appropriate version of the real implementaiton.
    """
    args = [lat1, lon1, lat2, lon2]
    for i in range(4):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.haversine",
                ["lat1", "lon1", "lat2", "lon2"],
                i,
            )

    def impl(lat1, lon1, lat2, lon2):  # pragma: no cover
        return haversine_util(lat1, lon1, lat2, lon2)

    return impl


@numba.generated_jit(nopython=True)
def div0(arr, divisor):
    """
    Handles cases where DIV0 receives optional arguments and forwards
    to the appropriate version of the real implementaiton.
    """
    args = [arr, divisor]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.div0", ["arr", "divisor"], i
            )

    def impl(arr, divisor):  # pragma: no cover
        return div0_util(arr, divisor)

    return impl


@numba.generated_jit(nopython=True)
def log(arr, base):
    """Handles cases where LOG receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [arr, base]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.log",
                ["arr", "base"],
                i,
            )

    def impl(arr, base):  # pragma: no cover
        return log_util(arr, base)

    return impl


@numba.generated_jit(nopython=True)
def negate(arr):
    """Handles cases where -X receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.negate_util",
            ["arr"],
            0,
        )

    def impl(arr):  # pragma: no cover
        return negate_util(arr)

    return impl


@numba.generated_jit(nopython=True)
def width_bucket(arr, min_val, max_val, num_buckets):
    """
    Handles cases where WIDTH_BUCKET receives optional arguments and forwards
    the arguments to appropriate version of the real implementation.
    """
    args = [arr, min_val, max_val, num_buckets]
    for i in range(4):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.width_bucket",
                ["arr", "min_val", "max_val", "num_buckets"],
                i,
            )

    def impl(arr, min_val, max_val, num_buckets):  # pragma: no cover
        return width_bucket_util(arr, min_val, max_val, num_buckets)

    return impl


@numba.generated_jit(nopython=True)
def bitand_util(A, B):
    """A dedicated kernel for the SQL function BITAND which takes in two numbers
    (or columns) and takes the bitwise-AND of them.


    Args:
        A (integer array/series/scalar): the first number(s) in the AND
        B (integer array/series/scalar): the second number(s) in the AND

    Returns:
        integer series/scalar: the output of the bitwise-AND
    """

    verify_int_arg(A, "bitand", "A")
    verify_int_arg(B, "bitand", "B")

    arg_names = ["A", "B"]
    arg_types = [A, B]
    propagate_null = [True] * 2
    scalar_text = "res[i] = arg0 & arg1"

    out_dtype = get_common_broadcasted_type([A, B], "bitand")

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def bitshiftleft_util(A, B):
    """A dedicated kernel for the SQL function BITSHIFTLEFT which takes in two numbers
    (or columns) and takes the bitwise-leftshift of them.


    Args:
        A (integer array/series/scalar): the number(s) being leftshifted
        B (integer array/series/scalar): the number(s) of bits to leftshift by

    Returns:
        integer series/scalar: the output of the bitwise-leftshift
    """

    verify_int_arg(A, "bitshiftleft", "A")
    verify_int_arg(B, "bitshiftleft", "B")

    arg_names = ["A", "B"]
    arg_types = [A, B]
    propagate_null = [True] * 2
    scalar_text = "res[i] = arg0 << arg1"

    out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int64)

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def bitnot_util(A):
    """A dedicated kernel for the SQL function BITNOT which takes in a number
    (or column) and takes the bitwise-not of it.


    Args:
        A (integer array/series/scalar): the number(s) being inverted

    Returns:
        integer series/scalar: the output of the bitwise-not
    """

    verify_int_arg(A, "bitnot", "A")

    arg_names = ["A"]
    arg_types = [A]
    propagate_null = [True]
    scalar_text = "res[i] = ~arg0"

    if A == bodo.none:
        out_dtype = bodo.none
    else:
        if is_array_typ(A, True):
            scalar_type = A.dtype
        else:
            scalar_type = A
        out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(scalar_type)

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def bitor_util(A, B):
    """A dedicated kernel for the SQL function BITOR which takes in two numbers
    (or columns) and takes the bitwise-OR of them.


    Args:
        A (integer array/series/scalar): the first number(s) in the OR
        B (integer array/series/scalar): the second number(s) in the OR

    Returns:
        integer series/scalar: the output of the bitwise-OR
    """

    verify_int_arg(A, "bitor", "A")
    verify_int_arg(B, "bitor", "B")

    arg_names = ["A", "B"]
    arg_types = [A, B]
    propagate_null = [True] * 2
    scalar_text = "res[i] = arg0 | arg1"

    out_dtype = get_common_broadcasted_type([A, B], "bitor")

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def bitshiftright_util(A, B):
    """A dedicated kernel for the SQL function BITSHIFTRIGHT which takes in two numbers
    (or columns) and takes the bitwise-rightshift of them.


    Args:
        A (integer array/series/scalar): the number(s) being rightshifted
        B (integer array/series/scalar): the number(s) of bits to rightshift by

    Returns:
        integer series/scalar: the output of the bitwise-rightshift
    """

    verify_int_arg(A, "bitshiftright", "A")
    verify_int_arg(B, "bitshiftright", "B")

    arg_names = ["A", "B"]
    arg_types = [A, B]
    propagate_null = [True] * 2

    if A == bodo.none:
        scalar_type = out_dtype = bodo.none
    else:
        if is_array_typ(A, True):
            scalar_type = A.dtype
        else:
            scalar_type = A
        out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(scalar_type)

    scalar_text = f"res[i] = arg0 >> arg1\n"

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def bitxor_util(A, B):
    """A dedicated kernel for the SQL function BITXOR which takes in two numbers
    (or columns) and takes the bitwise-XOR of them.


    Args:
        A (integer array/series/scalar): the first number(s) in the XOR
        B (integer array/series/scalar): the second number(s) in the XOR

    Returns:
        integer series/scalar: the output of the bitwise-XOR
    """

    verify_int_arg(A, "bitxor", "A")
    verify_int_arg(B, "bitxor", "B")

    arg_names = ["A", "B"]
    arg_types = [A, B]
    propagate_null = [True] * 2
    scalar_text = "res[i] = arg0 ^ arg1"

    out_dtype = get_common_broadcasted_type([A, B], "bitxor")

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def conv_util(arr, old_base, new_base):
    """A dedicated kernel for the CONV function REVERSE which takes in three
    integers (or integer columns) and converts the first column from the base
    indicated in the first second column to the base indicated by the third
    column.


    Args:
        arr (string array/series/scalar): the number(s) to be re-based
        old_base (int array/series/scalar): the original numerical base(s).
        Currently only supports numbers between 2 and 36 (inclusive).
        new_base (int array/series/scalar): the new numerical base(s). Currently
        only supports 2, 8, 10 and 16.

    Returns:
        string series/scalar: the converted numbers
    """

    verify_string_arg(arr, "CONV", "arr")
    verify_int_arg(old_base, "CONV", "old_base")
    verify_int_arg(new_base, "CONV", "new_base")

    arg_names = ["arr", "old_base", "new_base"]
    arg_types = [arr, old_base, new_base]
    propagate_null = [True] * 3
    scalar_text = "old_val = int(arg0, arg1)\n"
    scalar_text += "if arg2 == 2:\n"
    scalar_text += "   res[i] = format(old_val, 'b')\n"
    scalar_text += "elif arg2 == 8:\n"
    scalar_text += "   res[i] = format(old_val, 'o')\n"
    scalar_text += "elif arg2 == 10:\n"
    scalar_text += "   res[i] = format(old_val, 'd')\n"
    scalar_text += "elif arg2 == 16:\n"
    scalar_text += "   res[i] = format(old_val, 'x')\n"
    scalar_text += "else:\n"
    scalar_text += "   bodo.libs.array_kernels.setna(res, i)\n"

    out_dtype = bodo.string_array_type

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def getbit_util(A, B):
    """A dedicated kernel for the SQL function GETBIT which takes in two numbers
    (or columns) and returns the bit from the first one corresponding to the
    value of the second one


    Args:
        A (integer array/series/scalar): the number(s) whose bits are extracted
        B (integer array/series/scalar): the location(s) of the bits to extract

    Returns:
        boolean series/scalar: B'th bit of A
    """

    verify_int_arg(A, "bitshiftright", "A")
    verify_int_arg(B, "bitshiftright", "B")

    arg_names = ["A", "B"]
    arg_types = [A, B]
    propagate_null = [True] * 2
    scalar_text = "res[i] = (arg0 >> arg1) & 1"

    out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.uint8)

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def haversine_util(lat1, lon1, lat2, lon2):
    """A dedicated kernel for the SQL function HAVERSINE which takes in
    four floats representing two latitude and longitude coordinates and
    returns the haversine or great circle distance between the points
    (on the Earth).
    Args:
        arr (string array/series/scalar): the string(s) being repeated
        repeats (integer array/series/scalar): the number(s) of repeats
    Returns:
        string series/scalar: the repeated string(s)
    """
    verify_int_float_arg(lat1, "HAVERSINE", "lat1")
    verify_int_float_arg(lon1, "HAVERSINE", "lon1")
    verify_int_float_arg(lat2, "HAVERSINE", "lat2")
    verify_int_float_arg(lon2, "HAVERSINE", "lon2")

    arg_names = ["lat1", "lon1", "lat2", "lon2"]
    arg_types = [lat1, lon1, lat2, lon2]
    propogate_null = [True] * 4
    scalar_text = "arg0, arg1, arg2, arg3 = map(np.radians, (arg0, arg1, arg2, arg3))\n"
    dlat = "(arg2 - arg0) * 0.5"
    dlon = "(arg3 - arg1) * 0.5"
    h = f"np.square(np.sin({dlat})) + (np.cos(arg0) * np.cos(arg2) * np.square(np.sin({dlon})))"
    # r = 6731 is used for the radius of Earth (2r below)
    scalar_text += f"res[i] = 12742.0 * np.arcsin(np.sqrt({h}))\n"

    out_dtype = bodo.libs.float_arr_ext.FloatingArrayType(bodo.float64)

    return gen_vectorized(arg_names, arg_types, propogate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def div0_util(arr, divisor):
    """
    Kernel for div0.
    """
    verify_int_float_arg(arr, "DIV0", "arr")
    verify_int_float_arg(divisor, "DIV0", "divisor")

    arg_names = ["arr", "divisor"]
    arg_types = [arr, divisor]
    propogate_null = [True] * 2
    scalar_text = "res[i] = arg0 / arg1 if arg1 else 0\n"

    out_dtype = bodo.libs.float_arr_ext.FloatingArrayType(bodo.float64)

    return gen_vectorized(arg_names, arg_types, propogate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def log_util(arr, base):
    """A dedicated kernel for the SQL function LOG which takes in two numbers
    (or columns) and takes the log of the first one with the second as the base.


    Args:
        arr (float array/series/scalar): the number(s) whose logarithm is being taken
        target (float array/series/scalar): the base(s) of the logarithm

    Returns:
        float series/scalar: the output of the logarithm
    """

    verify_int_float_arg(arr, "log", "arr")
    verify_int_float_arg(base, "log", "base")

    arg_names = ["arr", "base"]
    arg_types = [arr, base]
    propagate_null = [True] * 2
    scalar_text = "res[i] = np.log(arg0) / np.log(arg1)"

    out_dtype = bodo.libs.float_arr_ext.FloatingArrayType(bodo.float64)

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def negate_util(arr):
    """A dedicated kernel for unary negation in SQL.

    Note: This kernel is shared by the - operator and the negate
    function and any "negate" operations we generate. As a result,
    we don't do a type check.


    Args:
        arr (numeric array/series/scalar): the number(s) whose sign is being flipped

    Returns:
        numeric series/scalar: the opposite of the input array
    """
    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]

    # Extract the underlying scalar dtype for casting integers
    if is_array_typ(arr, False):
        scalar_type = arr.dtype
    elif is_array_typ(arr, True):
        scalar_type = arr.data.dtype
    else:
        scalar_type = arr

    # If we have an unsigned integer, manually upcast then make it signed before negating
    scalar_text = {
        types.uint8: "res[i] = -np.int16(arg0)",
        types.uint16: "res[i] = -np.int32(arg0)",
        types.uint32: "res[i] = -np.int64(arg0)",
    }.get(scalar_type, "res[i] = -arg0")

    # If the dtype is unsigned, make the output dtype signed
    scalar_type = {
        types.uint8: types.int16,
        types.uint16: types.int32,
        types.uint32: types.int64,
        types.uint64: types.int64,
    }.get(scalar_type, scalar_type)

    if isinstance(scalar_type, types.Integer):
        # Only integers have a changed dtype. This path is avoided
        # in case we are negating a scalar without an array.
        out_dtype = bodo.utils.typing.dtype_to_array_type(scalar_type)
    else:
        # If we don't modify the code just return the same type.
        out_dtype = arr

    # Ensure the output is nullable.
    out_dtype = bodo.utils.typing.to_nullable_type(out_dtype)
    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def width_bucket_util(arr, min_val, max_val, num_buckets):
    verify_int_float_arg(arr, "WIDTH_BUCKET", "arr")
    verify_int_float_arg(min_val, "WIDTH_BUCKET", "min_val")
    verify_int_float_arg(max_val, "WIDTH_BUCKET", "max_val")
    verify_int_arg(num_buckets, "WIDTH_BUCKET", "num_buckets")

    arg_names = ["arr", "min_val", "max_val", "num_buckets"]
    arg_types = [arr, min_val, max_val, num_buckets]
    propagate_null = [True] * 4
    scalar_text = (
        "if arg1 >= arg2: raise ValueError('min_val must be less than max_val')\n"
    )
    scalar_text += (
        "if arg3 <= 0: raise ValueError('num_buckets must be a positive integer')\n"
    )
    scalar_text += "res[i] = min(max(-1.0, math.floor((arg0 - arg1) / ((arg2 - arg1) / arg3))), arg3) + 1.0"

    out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int64)

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


def add_numeric(arr0, arr1):  # pragma: no cover
    pass


def subtract_numeric(arr0, arr1):  # pragma: no cover
    pass


def multiply_numeric(arr0, arr1):  # pragma: no cover
    pass


def divide_numeric(arr0, arr1):  # pragma: no cover
    pass


def add_numeric_util(arr0, arr1):  # pragma: no cover
    pass


def subtract_numeric_util(arr0, arr1):  # pragma: no cover
    pass


def multiply_numeric_util(arr0, arr1):  # pragma: no cover
    pass


def divide_numeric_util(arr0, arr1):  # pragma: no cover
    pass


def create_numeric_operators_func_overload(func_name):
    """
    Returns the appropriate numeric operator function to support numeric operator functions
    with Snowflake SQL semantics. These SQL numeric operators are special because the
    output type conforms to SQL typing rules. Based on Snowflake,
    these have the following rules:
    https://docs.snowflake.com/en/sql-reference/operators-arithmetic.html#arithmetic-operators

    Summarizing and translating the rules from snowflake's generic decimal types to our types,
    we get the following approximate results:
        - Division always outputs Float64
        - Multiplication/Subtraction/Addition always promotes to the next largest type.
          This is because we do not have information like the number of leading digits and
          therefore must assume the worst case.

    TODO([BE-4191]): FIX Calcite to match this behavior.


    This is the ordering of types from largest to smallest:

        - Float64
        - Float32
        - Int64/UInt64
        - Int32/UInt32
        - Int16/UInt16
        - Int8/UInt8

    Note that SQL doesn't have defined unsigned integers. As a result we keep
    all outputs signed. We may add proper unsigned types in the future.
    """

    def overload_func(arr0, arr1):
        """Handles cases where func_name receives an optional argument and forwards
        to the appropriate version of the real implementation"""
        args = [arr0, arr1]
        for i in range(2):
            if isinstance(args[i], types.optional):
                return unopt_argument(
                    f"bodo.libs.bodosql_array_kernels.{func_name}",
                    ["arr0", "arr1"],
                    i,
                )

        func_text = "def impl(arr0, arr1):\n"
        func_text += (
            f"  return bodo.libs.bodosql_array_kernels.{func_name}_util(arr0, arr1)"
        )
        loc_vars = {}
        exec(func_text, {"bodo": bodo}, loc_vars)

        return loc_vars["impl"]

    return overload_func


def create_numeric_operators_util_func_overload(func_name):  # pragma: no cover
    """Creates an overload function to support numeric operator functions
    with Snowflake SQL semantics. These SQL numeric operators are special because the
    output type conforms to SQL typing rules. Based on Snowflake,
    these have the following rules:
    https://docs.snowflake.com/en/sql-reference/operators-arithmetic.html#arithmetic-operators

    Summarizing and translating the rules from snowflake's generic decimal types to our types,
    we get the following approximate results:
        - Division always outputs Float64
        - Multiplication/Subtraction/Addition always promotes to the next largest type.
          This is because we do not have information like the number of leading digits and
          therefore must assume the worst case.

    TODO([BE-4191]): FIX Calcite to match this behavior.


    This is the ordering of types from largest to smallest:

        - Float64
        - Float32
    ----------------------------- 2 integers won't promote to floats
        - Int64/UInt64
        - Int32/UInt32
        - Int16/UInt16
        - Int8/UInt8

    Note that SQL doesn't have defined unsigned integers. As a result we keep
    all outputs signed. We may add proper unsigned types in the future.


    Args:
        func_name: which operator function is being called (e.g. "add_numeric")

    Returns:
        (function): a utility that returns an overload with the operator functionality.
    """

    def overload_func_util(arr0, arr1):
        def determine_output_arr_type(func_name, dtype1, dtype2):
            """Helper function to derive the output array type.

            Args:
                func_name (str): Name of the function being performed.
                dtype1 (types.Type): element dtype of the first array.
                dtype2 (types.Type): element dtype of the second array.

            Returns:
                types.Type: The Array type to return. Note we always.

            """
            if func_name == "divide_numeric":
                # TODO: Fix the expected output type inside calcite.
                out_dtype = types.float64
            elif dtype1 == types.none and dtype2 == types.none:
                # If both values are None we will output None.
                # As a result, none of this code matters and
                # we can just choose to output any array.
                out_dtype = types.int64
            elif dtype1 in (types.float64, types.float32) or dtype2 in (
                types.float64,
                types.float32,
            ):
                # Always promote floats if possible.
                out_dtype = types.float64
            else:
                # None may just set NA. Still BodoSQL should use type
                # promotion when it sees 2 different integer types.
                # We may be forced to cast if this type is optional.
                if dtype1 == types.none:
                    dtype1 = dtype2
                elif dtype2 == types.none:
                    dtype2 = dtype1
                max_bitwidth = max(dtype1.bitwidth, dtype2.bitwidth)
                if max_bitwidth == 64:
                    out_dtype = types.int64
                elif max_bitwidth == 32:
                    out_dtype = types.int64
                elif max_bitwidth == 16:
                    out_dtype = types.int32
                else:
                    out_dtype = types.int16

            # Always make the output nullable in case the input is optional. In the
            # future we can grab this information from SQL or the original function
            return bodo.utils.typing.to_nullable_type(types.Array(out_dtype, 1, "C"))

        verify_int_float_arg(arr0, func_name, "arr0")
        verify_int_float_arg(arr1, func_name, "arr1")
        arg_names = ["arr0", "arr1"]
        arg_types = [arr0, arr1]
        propagate_null = [True] * 2

        if is_array_typ(arr0):
            elem_dtype0 = arr0.dtype
        else:
            elem_dtype0 = arr0
        if is_array_typ(arr1):
            elem_dtype1 = arr1.dtype
        else:
            elem_dtype1 = arr1

        out_dtype = determine_output_arr_type(func_name, elem_dtype0, elem_dtype1)
        out_elem_dtype = out_dtype.dtype
        # determine if we need to cast each input.
        cast_arr0 = elem_dtype0 != out_elem_dtype
        cast_arr1 = elem_dtype1 != out_elem_dtype
        if func_name == "add_numeric":
            operator_str = "+"
        elif func_name == "subtract_numeric":
            operator_str = "-"
        elif func_name == "multiply_numeric":
            operator_str = "*"
        elif func_name == "divide_numeric":
            operator_str = "/"

        cast_name = f"np.{out_elem_dtype.name}"
        if cast_arr0:
            arg0_str = f"{cast_name}(arg0)"
        else:
            arg0_str = f"arg0"
        if cast_arr1:
            arg1_str = f"{cast_name}(arg1)"
        else:
            arg1_str = f"arg1"
        # cast the output in case the operation causes type promotion.
        scalar_text = f"res[i] = {cast_name}({arg0_str} {operator_str} {arg1_str})"
        return gen_vectorized(
            arg_names, arg_types, propagate_null, scalar_text, out_dtype
        )

    return overload_func_util


def _install_numeric_operators_overload():
    """Creates and installs the overloads for numeric operator
    functions."""
    for func, util, func_name in (
        (add_numeric, add_numeric_util, "add_numeric"),
        (subtract_numeric, subtract_numeric_util, "subtract_numeric"),
        (multiply_numeric, multiply_numeric_util, "multiply_numeric"),
        (divide_numeric, divide_numeric_util, "divide_numeric"),
    ):
        func_overload_impl = create_numeric_operators_func_overload(func_name)
        overload(func)(func_overload_impl)
        util_overload_impl = create_numeric_operators_util_func_overload(func_name)
        overload(util)(util_overload_impl)


_install_numeric_operators_overload()
