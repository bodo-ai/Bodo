# Copyright (C) 2022 Bodo Inc. All rights reserved.
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


@numba.generated_jit(nopython=True)
def decode(A):
    """Handles cases where DECODE receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if not isinstance(A, (types.Tuple, types.UniTuple)):
        raise_bodo_error("Decode argument must be a tuple")
    for i in range(len(A)):
        if isinstance(A[i], types.optional):
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.decode",
                ["A"],
                i,
                container_length=len(A),
            )

    def impl(A):  # pragma: no cover
        return decode_util(A)

    return impl


@numba.generated_jit(nopython=True)
def decode_util(A):
    """A dedicated kernel for the SQL function decode which takes in an input
    scalar/column a variable number of arguments in pairs (with an
    optional default argument at the end) with the following behavior:

    DECODE(A, 0, 'a', 1, 'b', '_')
        - if A = 0 -> output 'a'
        - if A = 1 -> output 'b'
        - if A = anything else -> output '_'


    Args:
        A: (any tuple): the variadic arguments which must obey the following
        rules:
            - Length >= 3
            - First argument and every first argument in a pair must be the
              same underlying scalar type
            - Every first argument in a pair (plus the last argument if there are
              an even number) must be the same underlying scalar type

    Returns:
        any series/scalar: the mapped values
    """
    if len(A) < 3:
        raise_bodo_error("Need at least 3 arguments to DECODE")

    arg_names = [f"A{i}" for i in range(len(A))]
    arg_types = [A[i] for i in range(len(A))]
    propagate_null = [False] * len(A)
    scalar_text = ""

    # Loop over every argument that is being compared with the first argument
    # to see if they match. A[i+1] is the corresponding output argument.
    for i in range(1, len(A) - 1, 2):

        # The start of each conditional
        cond = "if" if len(scalar_text) == 0 else "elif"

        # The code that is outputted inside of a conditional once a match is found:
        if A[i + 1] == bodo.none:
            match_code = "   bodo.libs.array_kernels.setna(res, i)\n"
        elif bodo.utils.utils.is_array_typ(A[i + 1]):
            match_code = f"   if bodo.libs.array_kernels.isna({arg_names[i+1]}, i):\n"
            match_code += f"      bodo.libs.array_kernels.setna(res, i)\n"
            match_code += f"   else:\n"
            match_code += f"      res[i] = arg{i+1}\n"
        else:
            match_code = f"   res[i] = arg{i+1}\n"

        # Match if the first column is a SCALAR null and this column is a scalar null or
        # a column with a null in it
        if A[0] == bodo.none and (
            bodo.utils.utils.is_array_typ(A[i]) or A[i] == bodo.none
        ):
            if A[i] == bodo.none:
                scalar_text += f"{cond} True:\n"
                scalar_text += match_code
                break
            else:
                scalar_text += (
                    f"{cond} bodo.libs.array_kernels.isna({arg_names[i]}, i):\n"
                )
                scalar_text += match_code

        # Otherwise, if the first column is a NULL, skip this column
        elif A[0] == bodo.none:
            pass

        elif bodo.utils.utils.is_array_typ(A[0]):
            # If A[0] is an array, A[i] is an array, and they are equal or both
            # null, then A[i+1] is the answer
            if bodo.utils.utils.is_array_typ(A[i]):
                scalar_text += f"{cond} (bodo.libs.array_kernels.isna({arg_names[0]}, i) and bodo.libs.array_kernels.isna({arg_names[i]}, i)) or (not bodo.libs.array_kernels.isna({arg_names[0]}, i) and not bodo.libs.array_kernels.isna({arg_names[i]}, i) and arg0 == arg{i}):\n"
                scalar_text += match_code

            # If A[0] is an array, A[i] is null, and A[0] is null in the
            # current row, then A[i+1] is the answer
            elif A[i] == bodo.none:
                scalar_text += (
                    f"{cond} bodo.libs.array_kernels.isna({arg_names[0]}, i):\n"
                )
                scalar_text += match_code

            # If A[0] is an array, A[i] is a scalar, and A[0] is not null
            # in the current row and equals the A[i], then A[i+1] is the answer
            else:
                scalar_text += f"{cond} (not bodo.libs.array_kernels.isna({arg_names[0]}, i)) and arg0 == arg{i}:\n"
                scalar_text += match_code

        # If A[0] is a scalar and A[i] is NULL, skip this pair
        elif A[i] == bodo.none:
            pass

        # If A[0] is a scalar and A[i] is an array, and the current row of
        # A[i] is not null and equal to A[0], then A[i+1] is the answer
        elif bodo.utils.utils.is_array_typ(A[i]):
            scalar_text += f"{cond} (not bodo.libs.array_kernels.isna({arg_names[i]}, i)) and arg0 == arg{i}:\n"
            scalar_text += match_code

        # If A[0] is a scalar and A[0] is a scalar and they are equal, then A[i+1] is the answer
        else:
            scalar_text += f"{cond} arg0 == arg{i}:\n"
            scalar_text += match_code

    # If the optional default was provided, set the answer to it if nothing
    # else matched, otherwise set to null
    if len(scalar_text) > 0:
        scalar_text += "else:\n"
    if len(A) % 2 == 0 and A[-1] != bodo.none:
        if bodo.utils.utils.is_array_typ(A[-1]):
            scalar_text += f"   if bodo.libs.array_kernels.isna({arg_names[-1]}, i):\n"
            scalar_text += "      bodo.libs.array_kernels.setna(res, i)\n"
            scalar_text += "   else:\n"
        scalar_text += f"      res[i] = arg{len(A)-1}"
    else:
        scalar_text += "   bodo.libs.array_kernels.setna(res, i)"

    # Create the mapping from each local variable to the corresponding element in the array
    # of columns/scalars
    arg_string = "A"
    arg_sources = {f"A{i}": f"A[{i}]" for i in range(len(A))}

    # Extract all of the arguments that correspond to inputs vs outputs
    if len(arg_types) % 2 == 0:
        input_types = [arg_types[0]] + arg_types[1:-1:2]
        output_types = arg_types[2::2] + [arg_types[-1]]
    else:
        input_types = [arg_types[0]] + arg_types[1::2]
        output_types = arg_types[2::2]

    # Verify that all the inputs have a common type, and all the outputs
    # have a common type
    in_dtype = get_common_broadcasted_type(input_types, "DECODE")
    out_dtype = get_common_broadcasted_type(output_types, "DECODE")

    # If all of the outputs are NULLs, just use the same array type as the input
    if out_dtype == bodo.none:
        out_dtype = in_dtype

    # Only allow the output to be dictionary encoded under the following
    # circumstances:
    #   1. The first argument is the array
    #   2. None of the inputs are bodo.none
    #   3. There is no default argument
    support_dict_encoding = (
        bodo.utils.utils.is_array_typ(A[0])
        and bodo.none not in input_types
        and len(arg_types) % 2 == 1
    )

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        arg_string,
        arg_sources,
        support_dict_encoding=support_dict_encoding,
    )
