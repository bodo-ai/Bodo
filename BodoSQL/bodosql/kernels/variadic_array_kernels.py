"""
Implements array kernels that are specific to BodoSQL which have a variable
number of arguments
"""

import numba
import numpy as np
import pandas as pd
from numba.core import types
from numba.extending import overload

import bodo
import bodosql
from bodo.utils.typing import (
    get_common_scalar_dtype,
    get_overload_const_bool,
    get_overload_const_list,
    get_overload_const_str,
    get_overload_const_tuple,
    is_bin_arr_type,
    is_overload_constant_bool,
    is_overload_constant_list,
    is_overload_constant_str,
    is_overload_none,
    is_str_arr_type,
    raise_bodo_error,
)
from bodo.utils.utils import is_array_typ
from bodosql.kernels.array_kernel_utils import (
    gen_vectorized,
    get_common_broadcasted_type,
    get_tz_if_exists,
    is_valid_date_arg,
    is_valid_datetime_or_date_arg,
    is_valid_tz_aware_datetime_arg,
    is_valid_tz_naive_datetime_arg,
    unopt_argument,
    verify_binary_arg,
    verify_string_arg,
    verify_string_binary_arg,
)


def object_construct_keep_null(values, names, scalars):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(object_construct_keep_null)
def overload_object_construct_keep_null(values, names, scalars):
    """A dedicated kernel for the SQL function OBJECT_CONSTRUCT_KEEP_NULL which
       takes in a variable number of key-value pairs as arguments and turns them
       into JSON data.

    Args:
        values (any tuple): the values for each key-value pair
        names (string tuple): the names of the JSON fields for each key-value pair
        scalars (boolean tuple): a boolean for each value indicating if it is a scalar

    Returns:
        the inputs combined into a JSON value
    """
    names = bodo.utils.typing.unwrap_typeref(names).meta
    scalars = bodo.utils.typing.unwrap_typeref(scalars).meta
    if len(values) != len(names) or len(values) != len(scalars) or len(values) == 0:
        raise_bodo_error("object_construct_keep_null: invalid argument lengths")

    arg_names = []
    arg_types = []
    scalar_dtypes = []
    are_arrays = []
    optionals = []
    # Extract the underlying types of each element so that the corresponding
    # struct type can be built.
    for i, arr_typ in enumerate(values):
        arg_name = f"v{i}"
        arg_names.append(arg_name)
        arg_types.append(arr_typ)
        is_optional = False
        if scalars[i]:
            # Represent null as a dummy type in the struct
            if arr_typ == bodo.none:
                arr_typ = bodo.null_array_type
            if isinstance(arr_typ, types.optional):
                arr_typ = arr_typ.type
                is_optional = True
            scalar_dtypes.append(arr_typ)
            are_arrays.append(False)
        else:
            if bodo.hiframes.pd_series_ext.is_series_type(arr_typ):
                scalar_dtypes.append(arr_typ.data.dtype)
            else:
                scalar_dtypes.append(arr_typ.dtype)
            are_arrays.append(True)
        optionals.append(is_optional)
    out_dtype = bodo.utils.typing.dtype_to_array_type(
        bodo.libs.struct_arr_ext.StructType(tuple(scalar_dtypes), names), True
    )

    propagate_null = [False] * len(arg_names)

    # Create the mapping from the tuple to the local variable.
    arg_string = "values, names, scalars"
    arg_sources = {f"v{i}": f"values[{i}]" for i in range(len(values))}

    data = ", ".join(
        f"bodo.utils.conversion.unbox_if_tz_naive_timestamp(arg{i})"
        for i in range(len(values))
    )
    nulls = []
    for i in range(len(values)):
        if values[i] == bodo.none:
            nulls.append("True")
        elif are_arrays[i]:
            nulls.append(f"bodo.libs.array_kernels.isna(v{i}, i)")
        elif optionals[i]:
            nulls.append(f"v{i} is None")
        else:
            nulls.append("False")
    scalar_text = f"null_vector = np.array([{', '.join(nulls)}], dtype=np.bool_)\n"
    scalar_text += f"res[i] = bodo.libs.struct_arr_ext.init_struct_with_nulls(({data},), null_vector, names)"
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        arg_string,
        arg_sources,
        are_arrays=are_arrays,
    )


def object_construct(values, names, scalars):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(object_construct)
def overload_object_construct(values, names, scalars):
    """A dedicated kernel for the SQL function OBJECT_CONSTRUCT which
       takes in a variable number of key-value pairs as arguments and turns them
       into JSON data, ignoring any key-value pairs when the value is null.

    Args:
        values (any tuple): the values for each key-value pair
        names (string tuple): the names of the JSON fields for each key-value pair
        scalars (boolean tuple): a boolean for each value indicating if it is a scalar

    Returns:
        the inputs combined into a JSON value
    """
    names = bodo.utils.typing.unwrap_typeref(names).meta
    scalars = bodo.utils.typing.unwrap_typeref(scalars).meta
    if len(values) != len(names) or len(values) != len(scalars) or len(values) == 0:
        raise_bodo_error("object_construct_keep_null: invalid argument lengths")

    arg_names = []
    arg_types = []
    arr_types = []
    are_arrays = []
    optionals = []
    # Extract the underlying types of each element so that the corresponding
    # map type can be derived.
    for i, arr_typ in enumerate(values):
        arg_name = f"v{i}"
        arg_names.append(arg_name)
        arg_types.append(arr_typ)
        is_optional = False
        if scalars[i]:
            # Represent null as a dummy type in the struct
            if arr_typ == bodo.none:
                arr_typ = bodo.null_array_type
            if isinstance(arr_typ, types.optional):
                arr_typ = arr_typ.type
                is_optional = True
            arr_types.append(bodo.utils.typing.dtype_to_array_type(arr_typ))
            are_arrays.append(False)
        else:
            if bodo.hiframes.pd_series_ext.is_series_type(arr_typ):
                arr_types.append(arr_typ.data)
            else:
                arr_types.append(arr_typ)
            are_arrays.append(True)
        optionals.append(is_optional)

    combined_dtype = bodosql.kernels.array_kernel_utils.get_combined_type(
        arr_types, "object_construct"
    )

    # TODO: optimize so the keys are dictionary encoded
    out_dtype = bodo.MapArrayType(bodo.string_array_type, combined_dtype)

    propagate_null = [False] * len(arg_names)

    # Create the mapping from the tuple to the local variable.
    arg_string = "values, names, scalars"
    arg_sources = {f"v{i}": f"values[{i}]" for i in range(len(values))}

    # For each of the value arguments, generate an expression to determine
    # if that particular key-value pair should be kept or dropped.
    nulls = []
    for i in range(len(values)):
        if values[i] == bodo.none:
            nulls.append("True")
        elif are_arrays[i]:
            nulls.append(f"bodo.libs.array_kernels.isna(v{i}, i)")
        elif optionals[i]:
            nulls.append(f"v{i} is None")
        else:
            nulls.append("False")

    # At this point, the implementations diverge depending on if the result
    # is a scalar or an array.
    extra_globals = {}
    key_type = out_dtype.key_arr_type
    val_type = out_dtype.value_arr_type
    prefix_code = ""

    # build a struct array to insert into the final map array.
    extra_globals["struct_typ_tuple"] = (key_type, val_type)
    extra_globals["map_struct_names"] = bodo.utils.typing.ColNamesMetaType(
        ("key", "value")
    )
    # Build an array of booleans to determine which pairs from the current
    # row should be kept.
    scalar_text = f"pairs_to_keep = np.zeros({len(names)}, dtype=np.bool_)\n"
    for i in range(len(values)):
        scalar_text += f"if not ({nulls[i]}):\n"
        scalar_text += f"  pairs_to_keep[{i}] = True\n"
    # Based on the number to be kept, allocate a struct array to store that many
    # key-value pairs.
    scalar_text += "n_keep = pairs_to_keep.sum()\n"
    scalar_text += "struct_arr = bodo.libs.struct_arr_ext.pre_alloc_struct_array(n_keep, (-1,), struct_typ_tuple, ('key', 'value'), None)\n"
    # For each pair, if it is to be kept, write a struct containing that pair
    # into he struct array.
    scalar_text += "write_idx = 0\n"
    for i in range(len(values)):
        scalar_text += f"if pairs_to_keep[{i}]:\n"
        if isinstance(values[i], bodo.MapArrayType):
            prefix_code += f"  in_offsets_{i} = bodo.libs.array_item_arr_ext.get_offsets(v{i}._data)\n"
            prefix_code += f"  in_struct_arr_{i} = bodo.libs.array_item_arr_ext.get_data(v{i}._data)\n"
            scalar_text += f"  start_offset_{i} = in_offsets_{i}[np.int64(i)]\n"
            scalar_text += f"  end_offset_{i} = in_offsets_{i}[np.int64(i+1)]\n"
            scalar_text += f"  val_arg_{i} = in_struct_arr_{i}[start_offset_{i} : end_offset_{i}]\n"
            val_arg = f"val_arg_{i}"
        else:
            val_arg = f"arg{i}"
        scalar_text += f"  struct_arr[write_idx] = bodo.libs.struct_arr_ext.init_struct_with_nulls(({repr(names[i])}, {val_arg}), (False, False), map_struct_names)\n"
        scalar_text += "  write_idx += 1\n"
    if any(are_arrays):
        scalar_text += "res[i] = struct_arr\n"
    else:
        # If our output should be scalar, then construct the map scalar we need
        scalar_text += (
            "key_data, value_data = bodo.libs.struct_arr_ext.get_data(struct_arr)\n"
        )
        scalar_text += "nulls = bodo.libs.struct_arr_ext.get_null_bitmap(struct_arr)\n"
        scalar_text += "res[i] = bodo.libs.map_arr_ext.init_map_value(key_data, value_data, nulls)\n"

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        arg_string,
        arg_sources,
        are_arrays=are_arrays,
        extra_globals=extra_globals,
        prefix_code=prefix_code,
    )


def coalesce(A, dict_encoding_state=None, func_id=-1):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(coalesce)
def overload_coalesce(A, dict_encoding_state=None, func_id=-1):
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
                "bodosql.kernels.coalesce",
                ["A", "dict_encoding_state", "func_id"],
                0,
                container_arg=i,
                container_length=len(A),
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(A, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return coalesce_util(A, dict_encoding_state, func_id)

    return impl


def coalesce_util(A, dict_encoding_state=None, func_id=-1):  # pragma: no cover
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
        the dtype that the resulting array should have, and a multiline string
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
            f"{arg_names[i]} = bodosql.kernels.to_timestamp({arg_names[i]}, None, None, 0)\n"
            for i in range(n)
            if date[i]
        ]
        return (True, out_dtype, "".join(casts))
    # Case 2: mix of tz-aware and date
    if np.all(tz_aware | date):
        out_dtype = bodo.DatetimeArrayType(time_zone)
        casts = [
            f"{arg_names[i]} = bodosql.kernels.to_timestamp({arg_names[i]}, None, {repr(time_zone)}, 0)\n"
            for i in range(n)
            if date[i]
        ]
        return (True, out_dtype, "".join(casts))
    return default_result


@overload(coalesce_util, no_unliteral=True)
def overload_coalesce_util(A, dict_encoding_state=None, func_id=-1):
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
                prefix_code += "num_strings += 1\n"
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
                scalar_text += f"   res[i] = old_indices{i - dead_offset}[i] + index_offset{i - dead_offset}\n"
            elif is_string_data:
                # If we have string data directly copy from one array to another without an intermediate
                # allocation.
                scalar_text += (
                    f"   bodo.libs.str_arr_ext.get_str_arr_item_copy(res, i, A{i}, i)\n"
                )
            else:
                scalar_text += f"   res[i] = bodo.utils.conversion.unbox_if_tz_naive_timestamp(arg{i - dead_offset})\n"
            first = False

        # If A[i] is a non-NULL scalar, then it is the answer and stop searching
        else:
            assert not found_scalar, (
                "should not encounter more than one scalar due to dead column pruning"
            )
            indent = ""
            if not first:
                scalar_text += "else:\n"
                indent = "   "
            if dict_encode_data:
                # If the data is dictionary encoded just copy the index that was allocated in the
                # dictionary. A scalar must only be the last element so its always index num_strings - 1
                scalar_text += f"{indent}res[i] = num_strings - 1\n"
            else:
                scalar_text += f"{indent}res[i] = bodo.utils.conversion.unbox_if_tz_naive_timestamp(arg{i - dead_offset})\n"
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
        suffix_code = ""
        if not (is_overload_none(dict_encoding_state)):
            # If we have a valid dict_encoding_state we can avoid computing the new dictionary.
            arr_ids = ", ".join([f"A{i}._dict_id" for i in range(len(A))])
            suffix_code += f"dict_ids = [{arr_ids}]\n"
            arr_lens = ", ".join([f"len(A{i}._data)" for i in range(len(A))])
            suffix_code += f"dict_lens = [{arr_lens}]\n"
            suffix_code += "if bodo.libs.streaming.dict_encoding.state_contains_multi_input_dict_array(dict_encoding_state, func_id, dict_ids, dict_lens):\n"
            suffix_code += "  dict_data, new_dict_id = bodo.libs.streaming.dict_encoding.get_array_multi_input(\n"
            suffix_code += "    dict_encoding_state, func_id, dict_ids, dict_lens, bodo.string_array_type\n"
            suffix_code += "  )\n"
            suffix_code += "else:\n"
            indent = "  "
        else:
            indent = ""
        suffix_code += f"{indent}dict_data = bodo.libs.str_arr_ext.pre_alloc_string_array(num_strings, num_chars)\n"
        suffix_code += f"{indent}curr_index = 0\n"
        # Track if the output dictionary is global. Even though it is not unique it may
        # still be the same on all ranks if the component dictionaries were all global.
        # Note: If there are any scalars they will be the same on all ranks so that is
        # still global.
        for i in range(len(A)):
            if i in dead_cols:
                dead_offset += 1
            elif arg_types[i - dead_offset] != bodo.string_type:
                # Copy the old dictionary into the new dictionary
                suffix_code += f"{indent}section_len = len(old_data{i - dead_offset})\n"
                # TODO: Add a kernel to copy everything at once?
                suffix_code += f"{indent}for l in range(section_len):\n"
                suffix_code += f"{indent}    bodo.libs.str_arr_ext.get_str_arr_item_copy(dict_data, curr_index + l, old_data{i - dead_offset}, l)\n"
                suffix_code += f"{indent}curr_index += section_len\n"
            else:
                # Just store the scalar.
                suffix_code += f"{indent}dict_data[curr_index] = A{i}\n"
                # This should be unnecessary but update the index
                suffix_code += f"{indent}curr_index += 1\n"
        suffix_code += f"{indent}new_dict_id = bodo.libs.dict_arr_ext.generate_dict_id(num_strings)\n"
        # Update the cache.
        if not (is_overload_none(dict_encoding_state)):
            suffix_code += (
                "  bodo.libs.streaming.dict_encoding.set_array_multi_input(\n"
            )
            suffix_code += "    dict_encoding_state, func_id, dict_ids, dict_lens, dict_data, new_dict_id\n"
            suffix_code += "  )\n"

        # Wrap the output into an actual dictionary encoded array.
        # Note: We cannot assume it is unique even if each component were unique.
        suffix_code += "duplicated_res = bodo.libs.dict_arr_ext.init_dict_arr(dict_data, res, is_dict_global, False, new_dict_id)\n"
        # Drop any duplicates and update the dictionary if and only if we aren't caching
        if not (is_overload_none(dict_encoding_state)):
            suffix_code += "res = duplicated_res\n"
        else:
            suffix_code += "res = bodo.libs.array.drop_duplicates_local_dictionary(duplicated_res, False)\n"

    # Create the mapping from each local variable to the corresponding element in the array
    # of columns/scalars
    arg_string = "A, dict_encoding_state=None, func_id=-1"
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
def decode(A, dict_encoding_state=None, func_id=-1):
    """Handles cases where DECODE receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if not isinstance(A, (types.Tuple, types.UniTuple)):
        raise_bodo_error("Decode argument must be a tuple")
    for i, val in enumerate(A):
        if isinstance(val, types.optional):
            return unopt_argument(
                "bodosql.kernels.decode",
                ["A", "dict_encoding_state", "func_id"],
                0,
                container_arg=i,
                container_length=len(A),
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(A, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return decode_util(A, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True)
def decode_util(A, dict_encoding_state, func_id):
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
            match_code = f"   if bodo.libs.array_kernels.isna({arg_names[i + 1]}, i):\n"
            match_code += "      bodo.libs.array_kernels.setna(res, i)\n"
            match_code += "   else:\n"
            match_code += f"      res[i] = arg{i + 1}\n"
        else:
            match_code = f"   res[i] = arg{i + 1}\n"

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
        scalar_text += f"      res[i] = arg{len(A) - 1}"
    else:
        scalar_text += "   bodo.libs.array_kernels.setna(res, i)"

    # Create the mapping from each local variable to the corresponding element in the array
    # of columns/scalars
    arg_string = "A, dict_encoding_state, func_id"
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

    # Only allow the output to be dictionary encoded if the first argument is
    # the array
    support_dict_encoding = bodo.utils.utils.is_array_typ(A[0])

    use_dict_caching = support_dict_encoding and not is_overload_none(
        dict_encoding_state
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
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


def object_filter_keys(A, keep_keys, scalars):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(object_filter_keys, no_unliteral=True)
def overload_object_filter_keys(A, keep_keys, scalars):
    """Handles cases where OBJECT_PICK/OBJECT_DELETE receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if not isinstance(A, (types.Tuple, types.UniTuple)):
        raise_bodo_error("OBJECT_PICK/OBJECT_DELETE argument must be a tuple")
    for i, val in enumerate(A):
        if isinstance(val, types.optional):
            return unopt_argument(
                "bodosql.kernels.object_filter_keys",
                ["A", "keep_keys", "scalars"],
                0,
                container_arg=i,
                container_length=len(A),
            )

    def impl(A, keep_keys, scalars):  # pragma: no cover
        return object_filter_keys_util(A, keep_keys, scalars)

    return impl


def object_filter_keys_util(A, keep_keys, scalars):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(object_filter_keys_util, no_unliteral=True)
def overload_object_filter_keys_util(A, keep_keys, scalars):
    """BodoSQL kernel for the SQL functions OBJECT_PICK/OBJECT_DELETE which
       takes in a variable number of strings and removes the entries
       whose keys match one of those strings from the data.

    Args:
        A (tuple): a tuple where the first element is the JSON (or column
        of JSON values) to modify, and the remaining elements are the
        key strings to drop.
        keep_keys (boolean): if true, keeps only the keys in A, otherwise
        drops the keys in A.
        scalars (tuple): a tuple indicating which arguments are scalars.
    Returns:
        (json column/scalar) the json data with the specified keys dropped
    """

    if not is_overload_constant_bool(keep_keys):
        raise_bodo_error(
            "OBJECT_PICK/OBJECT_DELETE keep_keys argument must be a const bool"
        )

    scalar_meta = bodo.utils.typing.unwrap_typeref(scalars).meta
    are_arrays = [not scalar_meta[i] for i in range(len(scalar_meta))]

    keep_mode = get_overload_const_bool(keep_keys)
    func_name = "object_pick" if keep_mode else "object_filter_keys"

    args_tup = get_overload_const_tuple(A)
    n_keys_to_drop = len(args_tup) - 1

    json_data = A[0]

    # If the input is null, just return it.
    if json_data == bodo.none or json_data == bodo.null_array_type:
        return lambda A, keep_keys, scalars: A[0]  # pragma: no cover

    json_type = json_data
    if bodo.hiframes.pd_series_ext.is_series_type(json_type):
        json_type = json_type.data

    key_names = [f"k{i}" for i in range(n_keys_to_drop)]
    key_arg_names = [f"arg{i + 1}" for i in range(n_keys_to_drop)]
    arg_names = ["json_data"] + key_names
    arg_types = list(A)

    # Create the mapping from the tuple to the local variable.
    arg_string = "A, keep_keys, scalars"
    arg_sources = {f"k{i}": f"A[{i + 1}]" for i in range(n_keys_to_drop)}
    arg_sources["json_data"] = "A[0]"

    propagate_null = [True] * (n_keys_to_drop + 1)
    extra_globals = {}

    if isinstance(
        json_type, (bodo.StructArrayType, bodo.libs.struct_arr_ext.StructType)
    ):
        # Generated code for when the json data is a struct array or struct scalar.

        # 1. extract all of the key names to drop at compile time since this affects
        # the type of the returned struct. This means all the remaining arguments must
        # be string literals.
        keys_to_drop = list(A)[1:]
        string_names_to_filter = []
        for k in keys_to_drop:
            if not is_overload_constant_str(k):
                raise_bodo_error(
                    f"{func_name} unsupported on struct arrays with non-constant keys"
                )
            string_names_to_filter.append(get_overload_const_str(k))
        if keep_mode:
            keep_condition = lambda k: k in string_names_to_filter
        else:
            keep_condition = lambda k: k not in string_names_to_filter
        # 2. For each field in the struct, determine if it is kept or dropped at compile
        # time based on whether the field name matches one of the names to drop.
        data = []
        nulls = []
        names = []
        dtypes = []
        n_fields = len(json_type.data)
        for i in range(n_fields):
            name = json_type.names[i]
            if keep_condition(name):
                null_check = (
                    f"bodo.libs.struct_arr_ext.is_field_value_null(arg0, '{name}')"
                )
                data.append(f"None if {null_check} else arg0['{name}']")
                nulls.append(null_check)
                names.append(name)
                dtypes.append(json_type.data[i])

        # 3. Use the information from the remaining keys to re-construct the input struct
        # with the desired subset of fields.
        if len(nulls) > 0:
            scalar_text = (
                f"null_vector = np.array([{', '.join(nulls)}], dtype=np.bool_)\n"
            )
        else:
            scalar_text = "null_vector = np.empty(0, dtype=np.bool_)\n"
        scalar_text += f"res[i] = bodo.libs.struct_arr_ext.init_struct_with_nulls(({', '.join(data)}{',' if len(data) else ''}), null_vector, names)"
        out_dtype = bodo.StructArrayType(tuple(dtypes), tuple(names))
        extra_globals["names"] = bodo.utils.typing.ColNamesMetaType(tuple(names))

    elif isinstance(json_type, (bodo.MapArrayType, bodo.MapScalarType)):
        keep_condition = "in" if keep_mode else "not in"

        # Generated code for when the json data comes from a MapArray or a scalar dictionary.

        # 1. Generate an array of booleans per key in the dictionary determining whether each
        # key should be kept or dropped, and sum it to count the total number to be kept.
        scalar_text = "n_pairs = len(arg0._keys)\n"
        scalar_text += "pairs_to_keep = np.empty(n_pairs, dtype=np.bool_)\n"
        scalar_text += "for idx in range(n_pairs):\n"
        scalar_text += f"   pairs_to_keep[idx] = arg0._keys[idx] {keep_condition} [{', '.join(key_arg_names)}]\n"
        scalar_text += "n_keep = pairs_to_keep.sum()\n"
        # The rest of the path but specifically for MapArrays:

        # 2. Allocate a struct array to represent the key-value pairs
        # for each key where pairs_to_keep is True
        key_type = json_type.key_arr_type
        val_type = json_type.value_arr_type
        extra_globals["struct_typ_tuple"] = (key_type, val_type)
        extra_globals["map_struct_names"] = bodo.utils.typing.ColNamesMetaType(
            ("key", "value")
        )
        scalar_text += "struct_arr = bodo.libs.struct_arr_ext.pre_alloc_struct_array(n_keep, (-1,), struct_typ_tuple, ('key', 'value'), None)\n"

        # 3. For each key-value pair that is to be kept, determine if the
        # value is null by doing a lookup in the original map array, then
        # construct a (key, value) struct to place in the array.
        scalar_text += "write_idx = 0\n"
        scalar_text += "for read_idx in range(len(pairs_to_keep)):\n"
        scalar_text += "  if pairs_to_keep[read_idx]:\n"
        scalar_text += "    key = arg0._keys[read_idx]\n"
        scalar_text += "    value = arg0._values[read_idx]\n"
        scalar_text += (
            "    value_is_null = bodo.libs.array_kernels.isna(arg0._values, read_idx)\n"
        )
        scalar_text += "    struct_arr[write_idx] = bodo.libs.struct_arr_ext.init_struct_with_nulls((key, value), (False, value_is_null), map_struct_names)\n"
        scalar_text += "    write_idx += 1\n"

        # 4. Store the struct array in the current row of the map array
        if any(are_arrays):
            scalar_text += "res[i] = struct_arr\n"
        else:
            # If our output should be scalar, then construct the map scalar we need
            scalar_text += (
                "key_data, value_data = bodo.libs.struct_arr_ext.get_data(struct_arr)\n"
            )
            scalar_text += (
                "nulls = bodo.libs.struct_arr_ext.get_null_bitmap(struct_arr)\n"
            )
            scalar_text += "res[i] = bodo.libs.map_arr_ext.init_map_value(key_data, value_data, nulls)\n"
        out_dtype = json_type
    else:  # pragma: no cover
        raise_bodo_error(
            f"object_filter_keys: unsupported type for json data '{json_data}'"
        )

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        arg_string,
        arg_sources,
        extra_globals=extra_globals,
        are_arrays=are_arrays,
    )


def concat_ws(A, sep, dict_encoding_state=None, func_id=-1):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(concat_ws)
def overload_concat_ws(A, sep, dict_encoding_state=None, func_id=-1):
    """Handles cases where concat_ws receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if not isinstance(A, (types.Tuple, types.UniTuple)):
        raise_bodo_error("concat_ws argument must be a tuple")
    arg_names = ["A", "sep", "dict_encoding_state", "func_id"]
    default_map = {"dict_encoding_state": None, "func_id": -1}
    for i, val in enumerate(A):
        if isinstance(val, types.optional):
            # Note: If we have an optional scalar and its not the last argument,
            # then the NULL vs non-NULL case can lead to different decisions
            # about dictionary encoding in the output. This will lead to a memory
            # leak as the dict-encoding result will be cast to a regular string array.
            return unopt_argument(
                "bodosql.kernels.concat_ws",
                arg_names,
                0,
                container_arg=i,
                container_length=len(A),
                default_map=default_map,
            )
    if isinstance(sep, types.optional):
        return unopt_argument(
            "bodosql.kernels.concat_ws",
            arg_names,
            1,
            default_map=default_map,
        )

    def impl(A, sep, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return concat_ws_util(A, sep, dict_encoding_state, func_id)

    return impl


def concat_ws_util(A, sep, dict_encoding_state, func_id):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(concat_ws_util, no_unliteral=True)
def overload_concat_ws_util(A, sep, dict_encoding_state, func_id):
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

    # Determine the output dtype. Note: we don't keep data dictionary encoded because there are too many possible combinations.
    out_dtype = verify_string_binary_arg(sep, "CONCAT_WS", "sep")

    arg_names = []
    arg_types = []
    # Verify that all arguments are string or binary arrays and the same type as sep.
    for i, arr_typ in enumerate(A):
        arg_name = f"A{i}"
        if out_dtype is None:
            out_dtype = verify_string_binary_arg(sep, "CONCAT_WS", "sep")
        elif out_dtype:
            verify_string_arg(arr_typ, "CONCAT_WS", arg_name)
        else:
            verify_binary_arg(arr_typ, "CONCAT_WS", arg_name)
        arg_names.append(arg_name)
        arg_types.append(arr_typ)

    arg_names.append("sep")
    arg_types.append(sep)
    propagate_null = [True] * len(arg_names)
    out_dtype = (
        bodo.string_array_type
        if out_dtype is None or out_dtype is True
        else bodo.binary_array_type
    )

    # Create the mapping from the tuple to the local variable.
    arg_string = "A, sep, dict_encoding_state, func_id"
    arg_sources = {f"A{i}": f"A[{i}]" for i in range(len(A))}

    concat_args = ",".join([f"arg{i}" for i in range(len(A))])
    scalar_text = f"  res[i] = arg{len(A)}.join([{concat_args}])\n"

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        arg_string,
        arg_sources,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


def least_greatest_codegen(A, is_greatest, dict_encoding_state, func_id):
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
    arg_string = "A, dict_encoding_state, func_id"
    arg_sources = {f"A{i}": f"A[{i}]" for i in range(len(A))}

    # When returning a scalar we return a pd.Timestamp type.
    unbox_str = "unbox_if_tz_naive_timestamp" if is_array_typ(out_dtype) else ""
    valid_arg_typ = out_dtype.dtype if is_array_typ(out_dtype) else out_dtype

    if is_valid_datetime_or_date_arg(valid_arg_typ):
        func_args = ", ".join(f"{unbox_str}(arg{i})" for i in range(len(arg_names)))
    else:
        func_args = ", ".join(f"arg{i}" for i in range(len(arg_names)))

    func = "max" if is_greatest else "min"

    # If only 1 column is passed, set result to be the same as input.
    if len(A) == 1:
        scalar_text = "  res[i] = A0[i]\n"
    else:
        scalar_text = f"  res[i] = {func}(({func_args}))\n"

    extra_globals = {
        "unbox_if_tz_naive_timestamp": bodo.utils.conversion.unbox_if_tz_naive_timestamp,
    }
    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        arg_string,
        arg_sources,
        extra_globals=extra_globals,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


def least(A, dict_encoding_state=None, func_id=-1):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(least)
def overload_least(A, dict_encoding_state=None, func_id=-1):
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
                "bodosql.kernels.least",
                ["A", "dict_encoding_state", "func_id"],
                0,
                container_arg=i,
                container_length=len(A),
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(A, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return least_util(A, dict_encoding_state, func_id)

    return impl


def least_util(A, dict_encoding_state, func_id):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(least_util, no_unliteral=True)
def overload_least_util(A, dict_encoding_state, func_id):
    """A dedicated kernel for the SQL function LEAST which takes in array of
       1+ columns/scalars and returns the smallest value.

    Args:
        A (any array/scalar tuple): the array of values that are compared
        to find the smallest value.

    Returns:
        an array containing the smallest value of the input array
    """

    return least_greatest_codegen(
        A, is_greatest=False, dict_encoding_state=dict_encoding_state, func_id=func_id
    )


def greatest(A, dict_encoding_state=None, func_id=-1):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(greatest)
def overload_greatest(A, dict_encoding_state=None, func_id=-1):
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
                "bodosql.kernels.greatest",
                ["A", "dict_encoding_state", "func_id"],
                0,
                container_arg=i,
                container_length=len(A),
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(A, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return greatest_util(A, dict_encoding_state, func_id)

    return impl


def greatest_util(A, dict_encoding_state, func_id):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(greatest_util, no_unliteral=True)
def overload_greatest_util(A, dict_encoding_state, func_id):
    """A dedicated kernel for the SQL function GREATEST which takes in array of
       1+ columns/scalars and returns the largest value.

    Args:
        A (any array/scalar tuple): the array of values that are compared
        to find the largest value.

    Returns:
        an array containing the largest value of the input array
    """
    return least_greatest_codegen(
        A, is_greatest=True, dict_encoding_state=dict_encoding_state, func_id=func_id
    )


def row_number(df, by_cols, ascending, na_position):  # pragma: no cover
    return


@overload(row_number, no_unliteral=True, inline="always")
def overload_row_number(df, by, ascending, na_position):
    """Performs the ROW_NUMBER operation on a DataFrame based on the sorting
       parameters provided. The result is returned as a DataFrame containing
       a single column called ROW_NUMBER due to constraints in how the code
       generation for window functions is currently handled.

    Args:
        df (pd.DataFrame): the DataFrame whose row ordinals are being sought
        by (constant List[str]): list of column names to sort by
        ascending (constant List[bool]): list indicating which columns to sort
        in ascending versus descending order
        na_position (constant List[str]): list of "first" or "last" values
        indicating which of the columns should have nulls placed first or last

    Returns:
        pd.DataFrame: a DataFrame with a single column ROW_NUMBER indicating
        what row number each row of the original DataFrame would be located
        in (1-indexed) if it were sorted bby the parameters provided.
    """
    if (
        not is_overload_constant_list(by)
        or not is_overload_constant_list(ascending)
        or not is_overload_constant_list(na_position)
    ):  # pragma: no cover
        raise_bodo_error(
            "row_number by, ascending and na_position arguments must be constant lists"
        )
    by_list = get_overload_const_list(by)
    asc_list = get_overload_const_list(ascending)
    na_list = get_overload_const_list(na_position)
    func_text = "def impl(df, by, ascending, na_position):\n"
    cols = ", ".join([f"df['{col}'].values" for col in by_list])
    func_text += "   n = len(df)\n"
    func_text += (
        "   index_2 = bodo.hiframes.pd_index_ext.init_range_index(0, n, 1, None)\n"
    )
    func_text += f"   df2 = bodo.hiframes.pd_dataframe_ext.init_dataframe(({cols},), index_2, __col_name_meta_value_1)\n"
    # Calculate the "bounds" for each ranks. It's really just a cumulative sum of
    # the length of the chunks on all the ranks. We will later use this to shuffle
    # data back as part of 'sort_index'.
    func_text += "   index_bounds = bodo.libs.distributed_api.get_chunk_bounds(bodo.utils.conversion.coerce_to_array(df2.index))\n"
    func_text += f"   df3 = df2.sort_values(by={by_list}, ascending={asc_list}, na_position={na_list})\n"
    func_text += "   rows = np.arange(1, n+1)\n"
    func_text += (
        "   index_3 = bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df3)\n"
    )
    func_text += "   df4 = bodo.hiframes.pd_dataframe_ext.init_dataframe((rows,), index_3, __col_name_meta_value_2)\n"
    # Sort by index. Use the "bounds" calculated earlier to make sure that
    # the ranks receive the right data (corresponding to the lengths in the
    # input dataframe).
    func_text += "   return df4.sort_index(_bodo_chunk_bounds=index_bounds)\n"

    __col_name_meta_value_1 = bodo.utils.typing.ColNamesMetaType(tuple(by_list))
    __col_name_meta_value_2 = bodo.utils.typing.ColNamesMetaType(("ROW_NUMBER",))

    loc_vars = {}
    exec(
        func_text,
        {
            "bodo": bodo,
            "bodosql": bodosql,
            "np": np,
            "pd": pd,
            "__col_name_meta_value_1": __col_name_meta_value_1,
            "__col_name_meta_value_2": __col_name_meta_value_2,
        },
        loc_vars,
    )
    impl = loc_vars["impl"]

    return impl


def array_construct(A, scalar_tup):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(array_construct, no_unliteral=True)
def overload_array_construct(A, scalar_tup):
    """A dedicated kernel for the SQL function ARRAY_CONSTRUCT which takes in a variable
       number of arguments and turns them into a single array. The arguments should ideally
       be of compatible types. This kernel has no un-optionalizing wrapper due to the way
       it handles optional types.

    Note: ARRAY_CONSTRUCT(x) is not the same as TO_ARRAY(x) because of different null rules.
    ARRAY_CONSTRUCT will construct a singleton array containing null. TO_ARRAY will return null.

    Args:
        A (any array/scalar tuple): the tuple of values to be placed into an array together
        scalar_tup (boolean tuple): tuple indicating which arguments correspond to scalars

    Returns:
        (any array array/scalar): the values combined into an array
    """

    # Currently cannot support the zero-length case since its type is unknown
    if len(A) == 0:
        raise_bodo_error("ARRAY_CONSTRUCT with no arguments not currently supported")

    arg_names = []
    arg_types = []
    optionals = []

    are_scalars = get_overload_const_tuple(scalar_tup)

    for i, arr_typ in enumerate(A):
        arg_name = f"A{i}"
        arg_names.append(arg_name)
        if isinstance(arr_typ, types.Optional):
            arg_types.append(arr_typ.type)
            optionals.append(True)
        else:
            arg_types.append(arr_typ)
            optionals.append(False)

    # Create the mapping from the tuple to the local variable.
    arg_string = "A, scalar_tup"
    arg_sources = {f"A{i}": f"A[{i}]" for i in range(len(A))}

    propagate_null = [False] * len(arg_names)

    inner_arr_type = get_common_broadcasted_type(arg_types, "ARRAY_CONSTRUCT")

    # Currently not able to support writing a dictionary encoded inner array
    # [BSE-1831] TODO: see if we can optimize ARRAY_CONSTRUCT for dictionary encoding
    if inner_arr_type == bodo.dict_str_arr_type:
        inner_arr_type = bodo.string_array_type
    if inner_arr_type == bodo.none:
        inner_arr_type = bodo.null_array_type
    out_dtype = bodo.libs.array_item_arr_ext.ArrayItemArrayType(inner_arr_type)

    if all(are_scalars) and any(
        is_array_typ(t, include_index_series=True) for t in arg_types
    ):  # pragma: no cover
        inner_arr_type = bodo.libs.array_item_arr_ext.ArrayItemArrayType(inner_arr_type)

    extra_globals = {"inner_arr_type": inner_arr_type}

    if is_str_arr_type(inner_arr_type):
        scalar_text = (
            f"inner_arr = bodo.libs.str_arr_ext.pre_alloc_string_array({len(A)}, -1)\n"
        )
    elif is_bin_arr_type(inner_arr_type):
        scalar_text = f"inner_arr = bodo.libs.binary_arr_ext.pre_alloc_binary_array({len(A)}, -1)\n"
    else:
        scalar_text = f"inner_arr = bodo.utils.utils.alloc_type({len(A)}, inner_arr_type, (-1,))\n"
    for i, typ in enumerate(arg_types):
        if bodo.hiframes.pd_series_ext.is_series_type(typ):
            typ = typ.data
        if not are_scalars[i]:
            scalar_text += f"if bodo.libs.array_kernels.isna(A{i}, i):\n"
            scalar_text += f"   bodo.libs.array_kernels.setna(inner_arr, {i})\n"
            scalar_text += "else:\n"
            scalar_text += f"   inner_arr[{i}] = arg{i}\n"
        elif typ == bodo.none:
            scalar_text += f"bodo.libs.array_kernels.setna(inner_arr, {i})\n"
        elif optionals[i]:
            scalar_text += f"if arg{i} is None:\n"
            scalar_text += f"   bodo.libs.array_kernels.setna(inner_arr, {i})\n"
            scalar_text += "else:\n"
            scalar_text += f"   inner_arr[{i}] = arg{i}\n"
        else:
            scalar_text += f"inner_arr[{i}] = arg{i}\n"
    scalar_text += "res[i] = inner_arr"

    are_arrays = [not is_scalar for is_scalar in are_scalars]

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        arg_string,
        arg_sources,
        extra_globals=extra_globals,
        are_arrays=are_arrays,
    )
