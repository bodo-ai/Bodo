"""
Implements comparison operation array kernels that are specific to BodoSQL
"""

from numba.core import types
from numba.extending import overload

import bodo
import bodosql
from bodo.utils.typing import is_overload_none
from bodosql.kernels.array_kernel_utils import gen_vectorized, unopt_argument


def equal(arr0, arr1, dict_encoding_state=None, func_id=-1):  # pragma: no cover
    pass


def not_equal(arr0, arr1, dict_encoding_state=None, func_id=-1):  # pragma: no cover
    pass


def less_than(arr0, arr1, dict_encoding_state=None, func_id=-1):  # pragma: no cover
    pass


def greater_than(arr0, arr1, dict_encoding_state=None, func_id=-1):  # pragma: no cover
    pass


def less_than_or_equal(
    arr0, arr1, dict_encoding_state=None, func_id=-1
):  # pragma: no cover
    pass


def greater_than_or_equal(
    arr0, arr1, dict_encoding_state=None, func_id=-1
):  # pragma: no cover
    pass


def equal_util(arr0, arr1, dict_encoding_state, func_id):  # pragma: no cover
    pass


def not_equal_util(arr0, arr1, dict_encoding_state, func_id):  # pragma: no cover
    pass


def less_than_util(arr0, arr1, dict_encoding_state, func_id):  # pragma: no cover
    pass


def greater_than_util(arr0, arr1, dict_encoding_state, func_id):  # pragma: no cover
    pass


def less_than_or_equal_util(
    arr0, arr1, dict_encoding_state, func_id
):  # pragma: no cover
    pass


def greater_than_or_equal_util(
    arr0, arr1, dict_encoding_state, func_id
):  # pragma: no cover
    pass


def create_comparison_operators_func_overload(func_name):
    """Creates an overload function to support comparison operator functions
    with Snowflake SQL semantics. These SQL operators treat NULL as unknown, so if
    either input is null the output is null.

    Note: Several different types can be compared so we don't do any type checking.

    Returns:
        (function): a utility that returns an overload with the operator functionality.
    """

    def overload_func(arr0, arr1, dict_encoding_state=None, func_id=-1):
        """Handles cases where func_name receives an optional argument and forwards
        to the appropriate version of the real implementation"""
        args = [arr0, arr1]
        for i in range(2):
            if isinstance(args[i], types.optional):
                return unopt_argument(
                    f"bodosql.kernels.{func_name}",
                    ["arr0", "arr1", "dict_encoding_state", "func_id"],
                    i,
                    default_map={"dict_encoding_state": None, "func_id": -1},
                )

        func_text = (
            "def impl_cmp_kernel(arr0, arr1, dict_encoding_state=None, func_id=-1):\n"
        )
        func_text += f"  return bodosql.kernels.comparison_array_kernels.{func_name}_util(arr0, arr1, dict_encoding_state, func_id)"
        loc_vars = {}
        exec(func_text, {"bodo": bodo, "bodosql": bodosql}, loc_vars)

        return loc_vars["impl_cmp_kernel"]

    return overload_func


def create_comparison_operators_util_func_overload(func_name):  # pragma: no cover
    """Creates an overload function to support comparison operator functions
    with Snowflake SQL semantics. These SQL operators treat NULL as unknown, so if
    either input is null the output is null.

    Note: Several different types can be compared so we don't do any type checking.

    Returns:
        (function): a utility that returns an overload with the operator functionality.
    """

    def overload_func_util(arr0, arr1, dict_encoding_state, func_id):
        arg_names = ["arr0", "arr1", "dict_encoding_state", "func_id"]
        arg_types = [arr0, arr1, dict_encoding_state, func_id]
        propagate_null = [True, True, False, False]
        out_dtype = bodo.boolean_array_type
        if func_name == "equal":
            operator_str = "=="
        elif func_name == "not_equal":
            operator_str = "!="
        elif func_name == "less_than":
            operator_str = "<"
        elif func_name == "greater_than":
            operator_str = ">"
        elif func_name == "less_than_or_equal":
            operator_str = "<="
        else:
            operator_str = ">="

        # decimal array comparison is done in Arrow to avoid function call overhead for
        # each row
        if isinstance(arr0, bodo.DecimalArrayType) or isinstance(
            arr1, bodo.DecimalArrayType
        ):
            return eval(
                f"lambda arr0, arr1, dict_encoding_state, func_id: arr0 {operator_str} arr1"
            )

        # Always unbox in case of Timestamp to avoid issues
        scalar_text = f"res[i] = bodo.utils.conversion.unbox_if_tz_naive_timestamp(arg0) {operator_str} bodo.utils.conversion.unbox_if_tz_naive_timestamp(arg1)"
        use_dict_caching = not is_overload_none(dict_encoding_state)
        return gen_vectorized(
            arg_names,
            arg_types,
            propagate_null,
            scalar_text,
            out_dtype,
            # Add support for dict encoding caching with streaming.
            dict_encoding_state_name="dict_encoding_state"
            if use_dict_caching
            else None,
            func_id_name="func_id" if use_dict_caching else None,
        )

    return overload_func_util


def _install_comparison_operators_overload():
    """Creates and installs the overloads for comparison operator
    functions."""
    for func, util, func_name in (
        (equal, equal_util, "equal"),
        (not_equal, not_equal_util, "not_equal"),
        (less_than, less_than_util, "less_than"),
        (greater_than, greater_than_util, "greater_than"),
        (less_than_or_equal, less_than_or_equal_util, "less_than_or_equal"),
        (greater_than_or_equal, greater_than_or_equal_util, "greater_than_or_equal"),
    ):
        func_overload_impl = create_comparison_operators_func_overload(func_name)
        overload(func)(func_overload_impl)
        util_overload_impl = create_comparison_operators_util_func_overload(func_name)
        overload(util)(util_overload_impl)


_install_comparison_operators_overload()
