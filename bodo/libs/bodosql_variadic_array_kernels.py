# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Implements array kernels that are specific to BodoSQL which have a variable
number of arguments
"""

from numba.core import types
from numba.extending import overload

import bodo
from bodo.libs.bodosql_array_kernel_utils import *
from bodo.utils.typing import raise_bodo_error


def coalesce(A):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(coalesce)
def overload_coalesce(A):
    """Handles cases where COALESCE recieves optional arguments and forwards
    to the appropriate version of the real implementation"""
    if not isinstance(A, (types.Tuple, types.UniTuple)):
        raise_bodo_error("Coalesce argument must be a tuple")
    for i in range(len(A)):
        if isinstance(A[i], types.optional):
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.coalesce",
                ["A"],
                i,
                container_length=len(A),
            )

    def impl(A):  # pragma: no cover
        return coalesce_util(A)

    return impl


def coalesce_util(A):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(coalesce_util, no_unliteral=True)
def overload_coalesce_util(A):
    """A dedicated kernel for the SQL function COALESCE which takes in array of
       1+ columns/scalars and returns the first value from each row that is
       not NULL.

    Args:
        A (any array/scalar tuple): the array of values that are coalesced
        into a single column by choosing the first non-NULL value

    Raises:
        BodoError: if there are 0 columns, or the types don't match

    Returns:
        pd.Series: a Series containing the coalesce values of the input array
    """
    if len(A) == 0:
        raise_bodo_error("Cannot coalesce 0 columns")

    # Figure out which columns can be ignored (NULLS or after a scalar)
    array_override = None
    dead_cols = []
    for i in range(len(A)):
        if A[i] == bodo.none:
            dead_cols.append(i)
        elif not bodo.utils.utils.is_array_typ(A[i]):
            for j in range(i + 1, len(A)):
                dead_cols.append(j)
                if bodo.utils.utils.is_array_typ(A[j]):
                    array_override = f"A[{j}]"
            break

    arg_names = [f"A{i}" for i in range(len(A)) if i not in dead_cols]
    arg_types = [A[i] for i in range(len(A)) if i not in dead_cols]
    propagate_null = [False] * (len(A) - len(dead_cols))
    scalar_text = ""
    first = True
    found_scalar = False
    dead_offset = 0

    for i in range(len(A)):

        # If A[i] is NULL or comes after a scalar, it can be skipped
        if i in dead_cols:
            dead_offset += 1
            continue

        # If A[i] is an array, its value is the answer if it is not NULL
        elif bodo.utils.utils.is_array_typ(A[i]):
            cond = "if" if first else "elif"
            scalar_text += f"{cond} not bodo.libs.array_kernels.isna(A{i}, i):\n"
            scalar_text += f"   res[i] = arg{i-dead_offset}\n"
            first = False

        # If A[i] is a non-NULL scalar, then it is the answer and stop searching
        else:
            assert (
                not found_scalar
            ), "should not encounter more than one scalar due to dead column pruning"
            if first:
                scalar_text += f"res[i] = arg{i-dead_offset}\n"
            else:
                scalar_text += "else:\n"
                scalar_text += f"   res[i] = arg{i-dead_offset}\n"
            found_scalar = True
            break

    # If no other conditions were entered, and we did not encounter a scalar,
    # set to NULL
    if not found_scalar:
        if not first:
            scalar_text += "else:\n"
            scalar_text += "   bodo.libs.array_kernels.setna(res, i)"
        else:
            scalar_text += "bodo.libs.array_kernels.setna(res, i)"

    # Create the mapping from each local variable to the corresponding element in the array
    # of columns/scalars
    arg_string = "A"
    arg_sources = {f"A{i}": f"A[{i}]" for i in range(len(A)) if i not in dead_cols}

    out_dtype = get_common_broadcasted_type(arg_types, "COALESCE")

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        arg_string,
        arg_sources,
        array_override,
        support_dict_encoding=False,
    )
