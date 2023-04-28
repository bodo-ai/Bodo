# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Implements array kernels that are specific to BodoSQL which have a variable
number of arguments
"""

from numba.core import types
from numba.extending import overload

import bodo
from bodo.libs.bodosql_array_kernel_utils import *
from bodo.utils.typing import (
    get_common_scalar_dtype,
    is_str_arr_type,
    raise_bodo_error,
)
from bodo.utils.utils import is_array_typ


def coalesce(A):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(coalesce)
def overload_coalesce(A):
    """Handles cases where COALESCE receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if not isinstance(A, (types.Tuple, types.UniTuple)):
        raise_bodo_error("Coalesce argument must be a tuple")
    for i in range(len(A)):
        if isinstance(A[i], types.optional):
            # Note: If we have an optional scalar and its not the last argument,
            # then the NULL vs non-NULL case can lead to different decisions
            # about dictionary encoding in the output. This will lead to a memory
            # leak as the dict-encoding result will be cast to a regular string array.
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.coalesce",
                ["A"],
                0,
                container_arg=i,
                container_length=len(A),
            )

    def impl(A):  # pragma: no cover
        return coalesce_util(A)

    return impl


def coalesce_util(A):  # pragma: no cover
    # Dummy function used for overload
    return


def detect_coalesce_casting(arg_types, arg_names):
    """Takes in the list of dtypes and argument names for a call to coalesce.
    If the combination is one of the allowed special cases, returns a tuple
    of True, the corresponding output dtype, and the casting instructions
    required to transform some of the arguments to be compatible with the
    new dtype.

    The current list of allowed special cases:
    - Mix of tz-naive timestamp and date -> cast dates to tz-naive timestamp
    - Mix of tz-aware timestamp and date -> cast dates to tz-aware timestamp

    Note: this function can be expanded in future (with great caution) to allow
    more implicit casting cases.

    Args:
        arg_types (List[dtypes]): the types of the inputs to COALESCE
        arg_names (List[string]): the names of the inputs to COALESCE

    Returns:
        Tuple[boolean, optional dtype, optional string]: a boolean indicating
        whether the list of types matches one of the special cases described above,
        the dtype htat the resulting array should have, and a multiline string
        containing the prefix code required to cast all of the arguments
        that need to be upcasted for the COALESCE to work.
    """
    default_result = (False, None, [])
    time_zone = None
    n = len(arg_types)
    # Scan through the arrays and mark which ones belong to which of the
    # dtypes of interest, aborting early if multiple different timezones
    # are found.
    tz_naive = np.array([False] * n)
    tz_aware = np.array([False] * n)
    date = np.array([False] * n)
    for i in range(len(arg_types)):
        if is_valid_date_arg(arg_types[i]):
            date[i] = True
        elif is_valid_tz_naive_datetime_arg(arg_types[i]):
            tz_naive[i] = True
        # [BE-4699] Investigate more timezone cases and if need be
        elif is_valid_tz_aware_datetime_arg(arg_types[i]):
            tz_aware[i] = True
            tz = get_tz_if_exists(arg_types[i])
            if time_zone is None:
                time_zone = tz
            elif tz != time_zone:
                return default_result
    # If all of all of teh arguments are the same underlying type, skip this
    # subroutine as it is no longer necessary
    if np.all(tz_naive) or np.all(tz_aware) or np.all(date):
        return default_result
    # Case 1: mix of tz-naive and date
    if np.all(tz_naive | date):
        out_dtype = types.Array(bodo.datetime64ns, 1, "C")
        casts = [
            f"{arg_names[i]} = bodo.libs.bodosql_array_kernels.to_timestamp({arg_names[i]}, None, None, 0)\n"
            for i in range(n)
            if date[i]
        ]
        return (True, out_dtype, "".join(casts))
    # Case 2: mix of tz-aware and date
    if np.all(tz_aware | date):
        out_dtype = bodo.DatetimeArrayType(time_zone)
        casts = [
            f"{arg_names[i]} = bodo.libs.bodosql_array_kernels.to_timestamp({arg_names[i]}, None, {repr(time_zone)}, 0)\n"
            for i in range(n)
            if date[i]
        ]
        return (True, out_dtype, "".join(casts))
    return default_result


@overload(coalesce_util, no_unliteral=True)
def overload_coalesce_util(A):
    """A dedicated kernel for the SQL function COALESCE which takes in array of
       1+ columns/scalars and returns the first value from each row that is
       not NULL.

       This kernel has optimized implementations for handling strings. First, if dealing
        with normal string arrays we avoid any intermediate allocation by using get_str_arr_item_copy.

        Next, we also keep the output dictionary encoded if all inputs are a dictionary encoded
        array followed by possibly one scalar value.

    Args:
        A (any array/scalar tuple): the array of values that are coalesced
        into a single column by choosing the first non-NULL value

    Raises:
        BodoError: if there are 0 columns, or the types don't match

    Returns:
        an array containing the coalesce values of the input array
    """
    if len(A) == 0:
        raise_bodo_error("Cannot coalesce 0 columns")

    # Figure out which columns can be ignored (NULLS or after a scalar)
    array_override = None
    dead_cols = []
    has_array_output = False
    for i in range(len(A)):
        if A[i] == bodo.none:
            dead_cols.append(i)
        elif not bodo.utils.utils.is_array_typ(A[i]):
            for j in range(i + 1, len(A)):
                dead_cols.append(j)
                if bodo.utils.utils.is_array_typ(A[j]):
                    # Indicate if the output should be an array. This is for the
                    # rare edge case where a scalar comes before an array so the
                    # length of the column needs to be determined from a later array.
                    array_override = f"A[{j}]"
                    has_array_output = True
            break
        else:
            has_array_output = True

    arg_names = [f"A{i}" for i in range(len(A)) if i not in dead_cols]
    arg_types = [A[i] for i in range(len(A)) if i not in dead_cols]
    # Special case: detect if the type combinations correspond to one of the
    # special combinations that are allowed to be coalesced, and if so
    # return the combined type and the code required to handle the implicit casts
    is_coalesce_casting_case, out_dtype, coalesce_casts = detect_coalesce_casting(
        arg_types, arg_names
    )
    if not is_coalesce_casting_case:
        # Normal case: determine the output dtype by combining all of the input types
        out_dtype = get_common_broadcasted_type(arg_types, "COALESCE")
        prefix_code = ""
    else:
        prefix_code = coalesce_casts
    # Determine if we have string data with an array output
    is_string_data = has_array_output and is_str_arr_type(out_dtype)
    propagate_null = [False] * (len(A) - len(dead_cols))

    dict_encode_data = False
    # If we have string data determine if we should do dictionary encoding
    if is_string_data:
        dict_encode_data = True
        for j, typ in enumerate(arg_types):
            # all arrays must be dictionaries or a scalar
            dict_encode_data = dict_encode_data and (
                typ == bodo.string_type
                or typ == bodo.dict_str_arr_type
                or (
                    isinstance(typ, bodo.SeriesType)
                    and typ.data == bodo.dict_str_arr_type
                )
            )

    # Track if each individual column is dictionary encoded.
    # This is only used if the output is dictionary encoded and
    # is garbage otherwise.
    scalar_text = ""
    first = True
    found_scalar = False
    dead_offset = 0
    # If we use dictionary encoding we will generate a prefix
    # to allocate for our custom implementation
    if dict_encode_data:
        # If we are dictionary encoding data then we will generate prefix code to compute new indices
        # and generate an original dictionary.
        prefix_code += "num_strings = 0\n"
        prefix_code += "num_chars = 0\n"
        prefix_code += "is_dict_global = True\n"
        for i in range(len(A)):
            if i in dead_cols:
                dead_offset += 1
                continue
            elif arg_types[i - dead_offset] != bodo.string_type:
                # Dictionary encoding will directly access the indices and data arrays.
                prefix_code += f"old_indices{i - dead_offset} = A{i}._indices\n"
                prefix_code += f"old_data{i - dead_offset} = A{i}._data\n"
                # Set if the output dict is global based on each dictionary.
                prefix_code += (
                    f"is_dict_global = is_dict_global and A{i}._has_global_dictionary\n"
                )
                # Determine the offset to add to the index in this array.
                prefix_code += f"index_offset{i - dead_offset} = num_strings\n"
                # Update the total number of strings and characters.
                prefix_code += f"num_strings += len(old_data{i - dead_offset})\n"
                prefix_code += f"num_chars += bodo.libs.str_arr_ext.num_total_chars(old_data{i - dead_offset})\n"
            else:
                prefix_code += f"num_strings += 1\n"
                # Scalar needs to be utf8 encoded for the number of characters
                prefix_code += (
                    f"num_chars += bodo.libs.str_ext.unicode_to_utf8_len(A{i})\n"
                )

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
            if dict_encode_data:
                # If data is dictionary encoded just copy the indices
                scalar_text += f"   res[i] = old_indices{i-dead_offset}[i] + index_offset{i - dead_offset}\n"
            elif is_string_data:
                # If we have string data directly copy from one array to another without an intermediate
                # allocation.
                scalar_text += (
                    f"   bodo.libs.str_arr_ext.get_str_arr_item_copy(res, i, A{i}, i)\n"
                )
            else:
                scalar_text += f"   res[i] = bodo.utils.conversion.unbox_if_tz_naive_timestamp(arg{i-dead_offset})\n"
            first = False

        # If A[i] is a non-NULL scalar, then it is the answer and stop searching
        else:
            assert (
                not found_scalar
            ), "should not encounter more than one scalar due to dead column pruning"
            indent = ""
            if not first:
                scalar_text += "else:\n"
                indent = "   "
            if dict_encode_data:
                # If the data is dictionary encoded just copy the index that was allocated in the
                # dictionary. A scalar must only be the last element so its always index num_strings - 1
                scalar_text += f"{indent}res[i] = num_strings - 1\n"
            else:
                scalar_text += f"{indent}res[i] = bodo.utils.conversion.unbox_if_tz_naive_timestamp(arg{i-dead_offset})\n"
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

    # If we have dictionary encoding we need to allocate a suffix to process the dictionary encoded array.
    # We allocate the dictionary at the end for cache locality.
    suffix_code = None
    if dict_encode_data:
        dead_offset = 0
        suffix_code = "dict_data = bodo.libs.str_arr_ext.pre_alloc_string_array(num_strings, num_chars)\n"
        suffix_code += "curr_index = 0\n"
        # Track if the output dictionary is global. Even though it is not unique it may
        # still be the same on all ranks if the component dictionaries were all global.
        # Note: If there are any scalars they will be the same on all ranks so that is
        # still global.
        for i in range(len(A)):
            if i in dead_cols:
                dead_offset += 1
            elif arg_types[i - dead_offset] != bodo.string_type:
                # Copy the old dictionary into the new dictionary
                suffix_code += f"section_len = len(old_data{i - dead_offset})\n"
                # TODO: Add a kernel to copy everything at once?
                suffix_code += f"for l in range(section_len):\n"
                suffix_code += f"    bodo.libs.str_arr_ext.get_str_arr_item_copy(dict_data, curr_index + l, old_data{i - dead_offset}, l)\n"
                suffix_code += f"curr_index += section_len\n"
            else:
                # Just store the scalar.
                suffix_code += f"dict_data[curr_index] = A{i}\n"
                # This should be unnecessary but update the index
                suffix_code += f"curr_index += 1\n"
        # Wrap the output into an actual dictionary encoded array.
        # Note: We cannot assume it is unique even if each component were unique.
        suffix_code += "duplicated_res = bodo.libs.dict_arr_ext.init_dict_arr(dict_data, res, is_dict_global, False)\n"
        # Drop any duplicates and update the dictionary
        suffix_code += "res = bodo.libs.array.drop_duplicates_local_dictionary(duplicated_res, False)\n"

    # Create the mapping from each local variable to the corresponding element in the array
    # of columns/scalars
    arg_string = "A"
    arg_sources = {f"A{i}": f"A[{i}]" for i in range(len(A)) if i not in dead_cols}

    if dict_encode_data:
        # If have we a dictionary encoded output then the main loop is used to compute
        # the indices.
        out_dtype = bodo.libs.dict_arr_ext.dict_indices_arr_type

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
        prefix_code=prefix_code,
        suffix_code=suffix_code,
        # If we have a string array avoid any intermediate allocations
        alloc_array_scalars=not is_string_data,
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
                0,
                container_arg=i,
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


def concat_ws(A, sep):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(concat_ws)
def overload_concat_ws(A, sep):
    """Handles cases where concat_ws receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if not isinstance(A, (types.Tuple, types.UniTuple)):
        raise_bodo_error("concat_ws argument must be a tuple")
    for i in range(len(A)):
        if isinstance(A[i], types.optional):
            # Note: If we have an optional scalar and its not the last argument,
            # then the NULL vs non-NULL case can lead to different decisions
            # about dictionary encoding in the output. This will lead to a memory
            # leak as the dict-encoding result will be cast to a regular string array.
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.concat_ws",
                ["A", "sep"],
                0,
                container_arg=i,
                container_length=len(A),
            )
    if isinstance(sep, types.optional):
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.concat_ws",
            ["A", "sep"],
            1,
        )

    def impl(A, sep):  # pragma: no cover
        return concat_ws_util(A, sep)

    return impl


def concat_ws_util(A, sep):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(concat_ws_util, no_unliteral=True)
def overload_concat_ws_util(A, sep):
    """A dedicated kernel for the SQL function CONCAT_WS which takes in array of
       1+ columns/scalars and a separator and returns the result of concatenating
       together all of the values.

    Args:
        A (any array/scalar tuple): the array of values that are concatenated
        into a single column.

    Raises:
        BodoError: if there are 0 columns, or the types don't match

    Returns:
        an array containing the concatenated values of the input array
    """
    if len(A) == 0:
        raise_bodo_error("Cannot concatenate 0 columns")

    arg_names = []
    arg_types = []
    # Verify that every argument is a string array.
    for i, arr_typ in enumerate(A):
        arg_name = f"A{i}"
        # TODO: Allow all binary data as well.
        verify_string_arg(arr_typ, "CONCAT_WS", arg_name)
        arg_names.append(arg_name)
        arg_types.append(arr_typ)
    # Verify sep
    arg_names.append("sep")
    verify_string_arg(sep, "CONCAT_WS", "sep")
    arg_types.append(sep)
    propagate_null = [True] * len(arg_names)
    # Determine the output dtype. Note: we don't keep data dictionary
    # encoded because there are too many possible combinations.
    out_dtype = bodo.string_array_type

    # Create the mapping from the tuple to the local variable.
    arg_string = "A, sep"
    arg_sources = {f"A{i}": f"A[{i}]" for i in range(len(A))}

    concat_args = ",".join([f"arg{i}" for i in range(len(A))])
    scalar_text = f"  res[i] = arg{len(A)}.join([{concat_args}])\n"
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        arg_string,
        arg_sources,
    )


def least_greatest_codegen(A, is_greatest):
    """
    A codegen function for SQL functions LEAST and GREATEST,
    which takes in an array of 1+ columns/scalars. Depending on
    the value of is_greatest, a flag which indicates whether
    the function is LEAST or GREATEST, this function will return
    the smallest/largest value.

    Args:
        A (any array/scalar tuple): the array of values that are compared
        to find the smallest value.

    Raises:
        BodoError: if there are 0 columns, or the types don't match

    Returns:
        an array containing the smallest/largest value of the input array
    """

    if len(A) == 0:
        raise_bodo_error("Cannot compare 0 columns")

    arg_names = []
    arg_types = []
    has_array_typ = False

    for i, arr_typ in enumerate(A):
        arg_name = f"A{i}"
        arg_names.append(arg_name)
        arg_types.append(arr_typ)
        if is_array_typ(arr_typ):
            has_array_typ = True

    propagate_null = [True] * len(arg_names)

    func = "GREATEST" if is_greatest else "LEAST"
    if has_array_typ:
        out_dtype = get_common_broadcasted_type(arg_types, func)
    else:
        out_dtype = get_common_scalar_dtype(arg_types)[0]

    # Create the mapping from the tuple to the local variable.
    arg_string = "A"
    arg_sources = {f"A{i}": f"A[{i}]" for i in range(len(A))}

    # When returning a scalar we return a pd.Timestamp type.
    unbox_str = "unbox_if_tz_naive_timestamp" if is_array_typ(out_dtype) else ""
    valid_arg_typ = out_dtype.dtype if is_array_typ(out_dtype) else out_dtype

    if is_valid_datetime_or_date_arg(valid_arg_typ):
        func_args = ", ".join(f"{unbox_str}(arg{i})" for i in range(len(arg_names)))
    else:
        func_args = ", ".join(f"arg{i}" for i in range(len(arg_names)))

    func = "max" if is_greatest else "min"
    scalar_text = f"  res[i] = {func}(({func_args}))\n"

    extra_globals = {
        "unbox_if_tz_naive_timestamp": bodo.utils.conversion.unbox_if_tz_naive_timestamp,
    }
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        arg_string,
        arg_sources,
        extra_globals=extra_globals,
    )


def least(A):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(least)
def overload_least(A):
    """Handles cases where LEAST receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if not isinstance(A, (types.Tuple, types.UniTuple)):
        raise_bodo_error("Least argument must be a tuple")
    for i in range(len(A)):
        if isinstance(A[i], types.optional):
            # Note: If we have an optional scalar and its not the last argument,
            # then the NULL vs non-NULL case can lead to different decisions
            # about dictionary encoding in the output. This will lead to a memory
            # leak as the dict-encoding result will be cast to a regular string array.
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.least",
                ["A"],
                0,
                container_arg=i,
                container_length=len(A),
            )

    def impl(A):  # pragma: no cover
        return least_util(A)

    return impl


def least_util(A):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(least_util, no_unliteral=True)
def overload_least_util(A):
    """A dedicated kernel for the SQL function LEAST which takes in array of
       1+ columns/scalars and returns the smallest value.

    Args:
        A (any array/scalar tuple): the array of values that are compared
        to find the smallest value.

    Returns:
        an array containing the smallest value of the input array
    """

    return least_greatest_codegen(A, is_greatest=False)


def greatest(A):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(greatest)
def overload_greatest(A):
    """Handles cases where GREATEST receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if not isinstance(A, (types.Tuple, types.UniTuple)):
        raise_bodo_error("Greatest argument must be a tuple")
    for i in range(len(A)):
        if isinstance(A[i], types.optional):
            # Note: If we have an optional scalar and its not the last argument,
            # then the NULL vs non-NULL case can lead to different decisions
            # about dictionary encoding in the output. This will lead to a memory
            # leak as the dict-encoding result will be cast to a regular string array.
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.greatest",
                ["A"],
                0,
                container_arg=i,
                container_length=len(A),
            )

    def impl(A):  # pragma: no cover
        return greatest_util(A)

    return impl


def greatest_util(A):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(greatest_util, no_unliteral=True)
def overload_greatest_util(A):
    """A dedicated kernel for the SQL function GREATEST which takes in array of
       1+ columns/scalars and returns the largest value.

    Args:
        A (any array/scalar tuple): the array of values that are compared
        to find the largest value.

    Returns:
        an array containing the largest value of the input array
    """
    return least_greatest_codegen(A, is_greatest=True)
