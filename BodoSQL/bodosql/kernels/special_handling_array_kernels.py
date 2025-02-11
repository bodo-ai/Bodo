"""
Implements array kernels that are specific to BodoSQL. These kernels require special codegen
that cannot be done through the the normal gen_vectorized path
"""

from numba.core import types
from numba.extending import overload

import bodo
from bodo.utils.typing import (
    dtype_to_array_type,
    get_common_scalar_dtype,
    is_nullable,
    is_scalar_type,
    raise_bodo_error,
)
from bodo.utils.utils import is_array_typ
from bodosql.kernels.array_kernel_utils import unopt_argument


def is_in_set_null(arr_to_check, out_arr, null_as):  # pragma: no cover
    pass


@overload(is_in_set_null)
def is_in_set_null_overload(arr_to_check, out_arr, null_as):
    """
    Sets values in out_arr based on null values in arr_to_check and null_as
    """
    # BSE-4544
    # TODO: only do this null setting in the case that arr_to_check is nullable
    # TODO: directly copy/clone the whole bit mask
    if null_as == types.none:

        def impl(arr_to_check, out_arr, null_as):
            for i in range(len(arr_to_check)):
                if bodo.libs.array_kernels.isna(arr_to_check, i):
                    bodo.libs.array_kernels.setna(out_arr, i)

        return impl
    else:
        assert is_scalar_type(null_as)

        def impl(arr_to_check, out_arr, null_as):
            n = len(arr_to_check)
            for i in range(n):
                if bodo.libs.array_kernels.isna(arr_to_check, i):
                    out_arr[i] = null_as

        return impl


def is_in(
    arr_to_check, arr_search_vals, null_as=None, is_parallel=False
):  # pragma: no cover
    pass


def is_in_util(
    arr_to_check, arr_search_vals, null_as=None, is_parallel=False
):  # pragma: no cover
    pass


@overload(is_in)
def is_in_overload(arr_to_check, arr_search_vals, null_as=None, is_parallel=False):
    """
    Handles cases where IS_IN receives optional arguments and forwards
    the arguments to appropriate version of the real implementation.

    Note that this is not included in the broadcasted_fixed_arg_functions,
    as the arrays do not need to meet each others distribution, or be the same length.

    Currently, we enforce that the second argument is a replicated array,
    (and is_parallel is therefore always false) but we may
    remove this restriction in the future.

    Args:
        arr_to_check (pandas Array): The values array. For each element, we check if it is present in
            arr_search_vals, and set True/False in the output boolean array accordingly.
        arr_search_vals (pandas Array): The values to search for in arr_to_check. Currently, is always
            replicated
        null_as (scalar, optional): The value to set in the output array if the corresponding value in
            arr_to_check is null. If null_as is None, then the output array will have nulls where
            arr_to_check has nulls. Defaults to None.
        is_parallel (bool, optional): Indicates if we should perform a distributed is_in check.
            Set in distributed pass depending on the distribution of arr_search_vals. Defaults to False.

    Returns:
        Pandas Array of boolean values
    """
    args = [arr_to_check, arr_search_vals]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.is_in",
                ["arr_to_check", "arr_search_vals", "null_as", "is_parallel"],
                i,
                default_map={"null_as": None, "is_parallel": False},
            )

    def impl(
        arr_to_check, arr_search_vals, null_as=None, is_parallel=False
    ):  # pragma: no cover
        return is_in_util(arr_to_check, arr_search_vals, null_as, is_parallel)

    return impl


@overload(is_in_util)
def is_in_util_overload(arr_to_check, arr_search_vals, null_as=None, is_parallel=False):
    """
    Helper function for is_in. See is_in for information on arguments
    """

    assert is_array_typ(arr_search_vals), (
        f"expected argument 'arr_search_vals' to be array type. Found: {arr_search_vals}"
    )

    if arr_to_check == types.none:

        def impl(
            arr_to_check, arr_search_vals, null_as=None, is_parallel=False
        ):  # pragma: no cover
            return None

        return impl

    if arr_to_check == arr_search_vals:
        """If the types match, we don't have to do any casting, we can just use the array isin kernel"""

        def impl(
            arr_to_check, arr_search_vals, null_as=None, is_parallel=False
        ):  # pragma: no cover
            # code modified from overload_series_isin
            n = len(arr_to_check)
            out_arr = bodo.libs.bool_arr_ext.alloc_false_bool_array(n)

            bodo.libs.array.array_isin(
                out_arr, arr_to_check, arr_search_vals, is_parallel
            )
            is_in_set_null(arr_to_check, out_arr, null_as)

            return out_arr

        return impl
    elif arr_to_check == bodo.dict_str_arr_type:
        """
        Special implementation to handle dict encoded arrays.
        In this case, instead of converting arr_to_check to regular string array, we convert
        arr_search_vals to dict encoded.
        This allows us to do a specialized implementation for
        array_isin c++.

        Test for this path can be found here:
        bodo/tests/bodosql_array_kernel_tests/test_bodosql_special_handling_array_kernels.py::test_is_in_dict_enc_string
        """

        # Check the types match
        assert arr_search_vals.dtype == bodo.string_type, (
            "Internal error: arr_to_check is dict encoded, but arr_search_vals does not have string dtype"
        )

        def impl(
            arr_to_check, arr_search_vals, null_as=None, is_parallel=False
        ):  # pragma: no cover
            # code modified from overload_series_isin
            n = len(arr_to_check)
            out_arr = bodo.libs.bool_arr_ext.alloc_false_bool_array(n)

            arr_search_vals = bodo.libs.str_arr_ext.str_arr_to_dict_str_arr(
                arr_search_vals
            )

            bodo.libs.array.array_isin(
                out_arr, arr_to_check, arr_search_vals, is_parallel
            )
            is_in_set_null(arr_to_check, out_arr, null_as)

            return out_arr

        return impl

    # In all other cases, we must attempt to unify the two types to determine the output array type

    # check that the provided array types are comparable
    lhs_scalar_type = arr_to_check.dtype if is_array_typ(arr_to_check) else arr_to_check
    rhs_scalar_type = arr_search_vals.dtype
    common_scalar_typ, _ = get_common_scalar_dtype([lhs_scalar_type, rhs_scalar_type])
    assert common_scalar_typ is not None, (
        "Internal error in is_in_util: arguments do not have a common scalar dtype"
    )

    needs_nullable_conversion = is_nullable(arr_to_check) or is_nullable(
        arr_search_vals
    )

    # Explicitly cast to nullable integer dtype, as get_common_scalar_dtype returns non-null int dtype
    # TODO: this will likely need to be extended when we support nullable/non-null floating arrays
    if isinstance(common_scalar_typ, types.Integer) and needs_nullable_conversion:
        common_scalar_typ = bodo.libs.int_arr_ext.IntDtype(common_scalar_typ)

    unified_array_type = dtype_to_array_type(
        common_scalar_typ, needs_nullable_conversion
    )
    if needs_nullable_conversion:
        assert is_nullable(unified_array_type), (
            "Internal error in is_in_util: unified_array_type is not nullable, but is required to be"
        )

    if is_array_typ(arr_to_check):

        def impl(
            arr_to_check, arr_search_vals, null_as=None, is_parallel=False
        ):  # pragma: no cover
            # code modified from overload_series_isin
            n = len(arr_to_check)
            out_arr = bodo.libs.bool_arr_ext.alloc_false_bool_array(n)

            # NOTE: array_isin requires that the array_infos are equal, which means that we have to
            # convert both arrays to the same type if one is not nullable, or we need to do up casting
            arr_to_check = bodo.utils.conversion.fix_arr_dtype(
                arr_to_check, common_scalar_typ, nan_to_str=False
            )
            arr_search_vals = bodo.utils.conversion.fix_arr_dtype(
                arr_search_vals, common_scalar_typ, nan_to_str=False
            )

            bodo.libs.array.array_isin(
                out_arr, arr_to_check, arr_search_vals, is_parallel
            )
            is_in_set_null(arr_to_check, out_arr, null_as)

            return out_arr

        return impl
    elif is_scalar_type(arr_to_check):

        def impl(
            arr_to_check, arr_search_vals, null_as=None, is_parallel=False
        ):  # pragma: no cover
            # convert scalar to array, do the operation, and then return the scalar value
            arr_to_check = bodo.utils.conversion.fix_arr_dtype(
                bodo.utils.conversion.coerce_to_array(
                    arr_to_check,
                    scalar_to_arr_len=1,
                    use_nullable_array=needs_nullable_conversion,
                ),
                common_scalar_typ,
            )

            out_arr = bodo.libs.bool_arr_ext.alloc_false_bool_array(1)
            bodo.libs.array.array_isin(
                out_arr, arr_to_check, arr_search_vals, is_parallel
            )
            # No need to check for nullability, since the input is a scalar
            return out_arr[0]

        return impl
    else:
        raise_bodo_error(
            f"is_in_util expects array or scalar input for arg0. Found {arr_to_check}"
        )
