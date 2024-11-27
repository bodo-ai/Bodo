"""
Implements BodoSQL array kernels related to ARRAY utilities
"""

from numba.core import types
from numba.extending import overload

import bodo
from bodo.utils.typing import raise_bodo_error
from bodo.utils.utils import is_array_typ


def semi_safe_equals(arg0, arg1):  # pragma: no cover
    # Dummy function used for overload
    pass


@overload(semi_safe_equals)
def overload_semi_safe_equals(arg0, arg1):
    """
    Takes in two values and returns True if they are equivalent. Designed to
    also work on nulls and semi-structured data. For arrays, returns True or False
    if the entire arrays are equivalent (not an element-wise comparison). Some
    examples (sse = semi_safe_equals):

    sse(None, None) -> True
    sse(None, 42) -> False
    sse(42, 42) -> True
    sse([1, 2, None], [1, 2, None]) -> True
    sse([1, None, 2], [1, 2, None]) -> False
    sse({"A": 0, "B": [3.0, 4.0]}, {"A": 0, "B": [3.0, 4.0]}) -> True
    sse({"A": 0, "B": [3.0, 4.0]}, {"A": 1, "B": [3.0, 4.0]}) -> False
    sse({"A": 0, "B": [3.0, 4.0]}, {"A": 0, "B": [3.0, 5.0]}) -> False
    sse({"A": 0, "B": 1, "C": 2}, {"A": 0, "C": 2, "B": 1}) -> True
    sse({"A": 0, "B": 1, "C": 2}, {"A": 0, "B": 1}) -> False
    sse({"A": 0, "B": 1, "C": 2}, {"A": 0, "B": 1, "C": 3}) -> False
    sse({"A": 0, "B": 1, "C": 2}, {"A": 0, "B": 1, "C": 2, "D": 3}) -> False
    sse({"A": 0, "B": 1, "C": 2}, {"A": 0, "B": 1, "D": 2}) -> False

    Arguments:
        arg0 (any): The first element to be compared.
        arg1 (any): The second element to be compared.

    Returns:
        (boolean): Whether the two elements are equal.
    """
    # Recursive implementation for arrays: check that all inner elements are equal.
    if is_array_typ(arg0) and is_array_typ(arg1):

        def impl(arg0, arg1):  # pragma: no cover
            # Verify that both arrays have the same length.
            if len(arg0) != len(arg1):
                return False
            # Verify that every entry in both arrays are either both null,
            # or both the same non-nul value.
            for i in range(len(arg0)):
                null_a = bodo.libs.array_kernels.isna(arg0, i)
                null_b = bodo.libs.array_kernels.isna(arg1, i)
                both_null = null_a and null_b
                neither_null = (not null_a) and (not null_b)
                if not (
                    both_null or (neither_null and semi_safe_equals(arg0[i], arg1[i]))
                ):
                    return False
            return True

        return impl

    # Handling cases where the inputs came from a struct array.
    if isinstance(arg0, bodo.libs.struct_arr_ext.StructType) or isinstance(
        arg1, bodo.libs.struct_arr_ext.StructType
    ):
        if isinstance(arg0, bodo.libs.struct_arr_ext.StructType) and isinstance(
            arg1, bodo.libs.struct_arr_ext.StructType
        ):
            # If the struct field names are different, the two structs are
            # automatically considered not-equal.
            if arg0.names != arg1.names:
                return lambda arg0, arg1: False  # pragma: no cover
            # Otherwise, we need to check for equality of values.
            func_text = "def impl(arg0, arg1):\n"
            for name in arg0.names:
                func_text += f"   if not semi_safe_equals(arg0['{name}'], arg1['{name}']): return False\n"
            func_text += "   return True"
            loc_vars = {}
            extra_globals = {"semi_safe_equals": semi_safe_equals}
            exec(func_text, extra_globals, loc_vars)
            return loc_vars["impl"]
        else:
            raise_bodo_error(
                f"semi_safe_equals: not currently supported between types {arg0} and {arg1}"
            )

    # If both inputs are null, they are equal.
    if arg0 == bodo.none and arg1 == bodo.none:
        return lambda arg0, arg1: True  # pragma: no cover

    # If exactly one input is null, they are not equal.
    if arg0 == bodo.none or arg1 == bodo.none:
        return lambda arg0, arg1: False  # pragma: no cover

    # Two entries from a MapArray are equal if they have the same key-value pairs.
    # This is checked by sorting the entries by key (which should always be a comparable type).
    if isinstance(arg0, types.DictType) and isinstance(arg1, types.DictType):

        def impl(arg0, arg1):  # pragma: no cover
            if len(arg0) != len(arg1):
                return False
            items_0 = sorted(arg0.items(), key=lambda x: x[0])
            items_1 = sorted(arg1.items(), key=lambda x: x[0])
            for i in range(len(items_0)):
                k0, v0 = items_0[i]
                k1, v1 = items_1[i]
                if k0 != k1:
                    return False
                if not semi_safe_equals(v0, v1):
                    return False
            return True

        return impl

    # Map array scalar values
    if isinstance(arg0, bodo.libs.map_arr_ext.MapScalarType) or isinstance(
        arg1, bodo.libs.map_arr_ext.MapScalarType
    ):
        # Reuse dict implementation above
        def impl(arg0, arg1):  # pragma: no cover
            return semi_safe_equals(dict(arg0), dict(arg1))

        return impl

    # Fallback implementation for most scalars.
    def impl(arg0, arg1):  # pragma: no cover
        return arg0 == arg1

    return impl
