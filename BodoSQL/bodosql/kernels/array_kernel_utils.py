"""
Common utilities for all BodoSQL array kernels
"""

import datetime
import math
import re

import numba
import numpy as np
import pandas as pd
import pyarrow as pa
from numba.core import types
from numba.extending import overload

import bodo
import bodosql
from bodo.hiframes.datetime_timedelta_ext import PDTimeDeltaType
from bodo.hiframes.pd_offsets_ext import DateOffsetType
from bodo.hiframes.pd_series_ext import (
    SeriesType,
    is_timedelta64_series_typ,
    pd_timedelta_type,
)
from bodo.libs.struct_arr_ext import StructArrayType, StructType
from bodo.utils.typing import (
    is_overload_bool,
    is_overload_constant_bool,
    is_overload_constant_bytes,
    is_overload_constant_number,
    is_overload_constant_str,
    is_overload_none,
    is_valid_float_arg,
    is_valid_int_arg,
    raise_bodo_error,
)


def is_valid_SQL_object_arg(arg):
    """
    Returns true if the given argument is a valid SQL object (scalar or column). This function is
    used to check if an argument is valid for SQL functions that accept SQL objects as arguments.
    """
    return (
        isinstance(arg, StructArrayType)
        or isinstance(arg, StructType)
        or (
            isinstance(arg, bodo.libs.map_arr_ext.MapArrayType)
            and (
                arg.key_arr_type == bodo.types.string_array_type
                or arg.key_arr_type == bodo.types.dict_str_arr_type
            )
        )
        or (
            isinstance(arg, bodo.libs.map_arr_ext.MapScalarType)
            and (
                arg.key_arr_type == bodo.types.string_array_type
                or arg.key_arr_type == bodo.types.dict_str_arr_type
            )
        )
    )


def is_array_item_array(typ):
    is_series = isinstance(typ, bodo.hiframes.pd_series_ext.SeriesType)
    if is_series:
        return isinstance(typ.data, bodo.libs.array_item_arr_ext.ArrayItemArrayType)
    return isinstance(typ, bodo.libs.array_item_arr_ext.ArrayItemArrayType)


def indent_block(text, indentation):
    """Adjusts the indentation of a multiline string so that it can be injected
       another multiline string at a specified indentation level.

    Args:
        text (string): the (potentially multiline) string that needs to have
        its indentation adjusted
        indentation (integer): the amount of spaces that should occur before
        the smallest level of indentation in the block of code

    Returns:
        string: the same multiline string with the indentation of all lines adjusted
        so that the first line has the indentation specified, with a trailing
        newline character. If the input string is None, an empty string is returned instead.
    """
    if not text:
        return ""
    first_line = text.splitlines()[0]
    i = len(first_line) - len(first_line.lstrip())
    return (
        "\n".join([" " * indentation + line[i:] for line in text.splitlines()]) + "\n"
    )


def gen_vectorized(
    arg_names,
    arg_types,
    propagate_null,
    scalar_text,
    out_dtype,
    arg_string=None,
    arg_sources=None,
    array_override=None,
    support_dict_encoding=True,
    may_cause_duplicate_dict_array_values=False,
    prefix_code=None,
    suffix_code=None,
    res_list=False,
    extra_globals=None,
    alloc_array_scalars=True,
    synthesize_dict_if_vector=None,
    synthesize_dict_setup_text=None,
    synthesize_dict_scalar_text=None,
    synthesize_dict_global=False,
    synthesize_dict_unique=False,
    dict_encoding_state_name=None,
    func_id_name=None,
    are_arrays=None,
):
    """Creates an impl for a column compute function that has several inputs
       that could all be scalars, nulls, or arrays by broadcasting appropriately.

    Args:
        arg_names (string list): the names of each argument
        arg_types (dtype list): the types of each argument
        propagate_null (bool list): a mask indicating which arguments produce
            an output of null when the input is null
        scalar_text (string): the func_text of the core operation done on each
            set of scalar values after all broadcasting is handled. The string should
            refer to the scalar values as arg0, arg1, arg2, etc. (where arg0
            corresponds to the current value from arg_names[0]), and store the final
            answer of the calculation in res[i]
        out_dtype (dtype): the dtype of the output array
        arg_string (optional string): the string that goes in the def line to
            describe the parameters. If not provided, is inferred from arg_names
        arg_sources (optional dict): key-value pairings describing how to
            obtain the arg_names from the arguments described in arg_string
        array_override (optional string): a string representing how to obtain
            the length of the final array. If not provided, inferred from arg_types.
            If provided, ensures that the returned answer is always an array,
            even if all of the arg_types are scalars.
        support_dict_encoding (optional boolean): if true, allows dictionary
            encoded outputs under certain conditions
        may_cause_duplicate_dict_array_values (optional boolean): Indicates that the
            given operation may cause duplicate values in the ._data field of a dictionary
            encoded output (slicing, for example). Only has effect if support_dict_encoding
            is also true.
        prefix_code (optional string): if provided, embeds the code string
            right before the loop begins.
        suffix_code (optional string): if provided, embeds the code string
            after the loop before the function ends (not used if there is no loop)
        res_list (optional boolean): if provided, sets up res as a list instead
            of an array, and does not use a prange. Not compatible with
            support_dict_encoding.
        alloc_array_scalars (boolean): When generating the func_text should array values
            be unpacked into local variables. This is an optimization that should only be
            done when the allocation can be expensive (e.g. strings) and there is an optimized
            way to compute the result without the intermediate allocate (e.g. copying
            values with get_str_arr_item_copy). If this is False the scalar text should never reference
            array values using the local variable names and is responsible for directly using the
            optimized implementation.
        synthesize_dict_if_vector (optional string list): if provided, dictates that dictionary
            encoded synthesis should be done if the arguments in 'V' locations are vectors,
            'S' locations are scalars, and '?' locations are either. For example,
            if ['V', '?', 'S'], dictionary encoding synthesis would be enabled
            if the first argument was a vector and the third argument was a scalar.
        synthesize_dict_setup_text (optional string): if provided, specifies the string to
            embed to initialize the dictionary encoded array's dictionary when performing
            dictionary synthesis. The dictionary should be named dict_res.
        synthesize_dict_scalar_text (optional string): if provided, specifies the string to
            embed to fill in the index array for each row of the inputs when performing
            dictionary synthesis.
        synthesize_dict_global (bool): if dictionary synthesis is used, is the dictionary
            global? Default is False.
        synthesize_dict_unique (bool): if dictionary synthesis is used, is the dictionary
            unique? Default is False.
        dict_encoding_state_name (Optional[str]): Variable name to use for kernels that
            support dictionary encoding caching in streaming. If none then no caching
            is available.
        func_id_name (Optional[str]): Variable name to use for a func_id with dictionary
            encoding caching in streaming.
        are_arrays (Optional[list[bool]]): List of bools of length args, when passed it is used
            to specify whether to treat the corresponding arg as an array

    Returns:
        function: a broadcasted version of the calculation described by
        scalar_text, which can be used as an overload.

    Internal Doc explaining more about this utility:
    https://bodo.atlassian.net/wiki/spaces/B/pages/1080492110/BodoSQL+Array+Kernel+Parametrization

    Below is an example that would vectorize the sum operation, where if either
    element is NULL the output is NULL. In this case, it is being constructed
    for when both arguments are arrays.

    arg_names = ['left', 'right']
    arg_types = [series(int64, ...), series(int64, ...)]
    propagate_null = [True, True]
    out_dtype = types.int64
    scalar_text = "res[i] = arg0 + arg1"

    This would result in an impl constructed from the following func_text:

    def impl(left, right):
        n = len(left)
        res = bodo.utils.utils.alloc_type(n, out_dtype, -1)
        left = bodo.utils.conversion.coerce_to_array(left)
        right = bodo.utils.conversion.coerce_to_array(right)
        numba.parfors.parfor.init_prange()
        for i in numba.parfors.parfor.internal_prange(n):
            if bodo.libs.array_kernels.isna(left, i):
                bodo.libs.array_kernels.setna(res, i)
                continue
            if bodo.libs.array_kernels.isna(left, i):
                bodo.libs.array_kernels.setna(res, i)
                continue
            arg0 = left[i]
            arg1 = right[i]
            res[i] = arg0 + arg1
        return res

    (Where out_dtype is mapped to types.int64)

    NOTE: dictionary encoded outputs operate under the following assumptions:
    - The output will only be dictionary encoded if exactly one of the inputs
        is dictionary encoded, and the rest are all scalars, and support_dict_encoding
        is True.
    - The indices do not change, except for some of them becoming null if the
        string they refer to is also transformed into null
    - The size of the dictionary will never change, even if several of its
        values become unused, duplicates, or nulls.
    - This function cannot be inlined as inlining the dictionary allocation
        is unsafe.
    - All functions invoked in scalar_text must be deterministic (no randomness
        involved).
    - Nulls are never converted to non-null values.
    """
    assert not (res_list and support_dict_encoding), (
        "Cannot use res_list with support_dict_encoding"
    )

    if are_arrays is None:
        are_arrays = [bodo.utils.utils.is_array_typ(typ) for typ in arg_types]
    all_scalar = not any(are_arrays)
    out_null = any(
        propagate_null[i]
        for i in range(len(arg_types))
        if arg_types[i] == bodo.types.none
    )
    # Construct a dictionary encoded output from a non-dictionary encoded input
    # if the 'V'/'?' arguments in synthesize_dict_if_vector are arrays and the
    # 'S'/'?' arguments are scalars.
    use_dict_synthesis = False
    if synthesize_dict_if_vector is not None:
        assert synthesize_dict_setup_text is not None, (
            "synthesize_dict_setup_text must be provided if synthesize_dict_if_vector is provided"
        )
        assert synthesize_dict_scalar_text is not None, (
            "synthesize_dict_scalar_text must be provided if synthesize_dict_if_vector is provided"
        )
        use_dict_synthesis = True
        for i in range(len(arg_types)):
            if are_arrays[i] and synthesize_dict_if_vector[i] == "S":
                use_dict_synthesis = False
            if not are_arrays[i] and synthesize_dict_if_vector[i] == "V":
                use_dict_synthesis = False

    # The output is dictionary-encoded if exactly one of the inputs is
    # dictionary encoded, the rest are all scalars, and the output dtype
    # is a string array
    vector_args = 0
    dict_encoded_arg = -1
    for i in range(len(arg_types)):
        if bodo.utils.utils.is_array_typ(arg_types[i], False):
            vector_args += 1
            if arg_types[i] == bodo.types.dict_str_arr_type:
                dict_encoded_arg = i
        elif bodo.utils.utils.is_array_typ(arg_types[i], True):
            vector_args += 1
            if arg_types[i].data == bodo.types.dict_str_arr_type:
                dict_encoded_arg = i
    use_dict_encoding = (
        support_dict_encoding and vector_args == 1 and dict_encoded_arg >= 0
    )
    # Flushing nulls from the new dictionary array back to the new index array
    # is only required if the scalar_text contains a setna call, or if one of
    # the arguments with null propagation is a scalar NULL
    use_null_flushing = (
        use_dict_encoding
        and out_dtype == bodo.types.string_array_type
        and (
            any(
                arg_types[i] == bodo.types.none and propagate_null[i]
                for i in range(len(arg_types))
            )
            or "bodo.libs.array_kernels.setna" in scalar_text
        )
    )

    if arg_string is None:
        arg_string = ", ".join(arg_names)

    func_text = f"def impl_gen_vectorized({arg_string}):\n"

    # Extract each argument name from the arg_string. Currently this is used for
    # a tuple input for variadic functions, but this use case may expand in the
    # future, at which point this comment will be updated
    if arg_sources is not None:
        for argument, source in arg_sources.items():
            func_text += f"   {argument} = {source}\n"

    # If all the inputs are scalar, either output None immediately or
    # compute a single scalar computation without the loop
    if all_scalar and array_override is None:
        if out_null:
            func_text += "   return None"
        else:
            func_text += indent_block(prefix_code, 3)

            for i in range(len(arg_names)):
                func_text += f"   arg{i} = {arg_names[i]}\n"
            scalar_version = (
                scalar_text
                # res[i] is now stored as answer, since there is no res array
                .replace("res[i] =", "answer =")
                # Calls to setna mean that the answer is NULL, so they are
                # replaced with "return None".
                .replace("bodo.libs.array_kernels.setna(res, i)", "return None")
                # NOTE: scalar_text should not contain any isna calls in
                # the case where all of the inputs are scalar.
            )
            func_text += indent_block(scalar_version, 3)
            func_text += "   return answer"

    else:
        # Convert all Series to arrays
        for i in range(len(arg_names)):
            if bodo.hiframes.pd_series_ext.is_series_type(arg_types[i]):
                func_text += f"   {arg_names[i]} = bodo.hiframes.pd_series_ext.get_series_data({arg_names[i]})\n"

        # If an array override is provided, use it to obtain the length
        if array_override != None:
            size_text = f"len({array_override})"

        # Otherwise, determine the size of the final output array from the
        # first argument that is an array
        else:
            for i in range(len(arg_names)):
                if are_arrays[i]:
                    size_text = f"len({arg_names[i]})"
                    break

        # If using dictionary encoding, ensure that the argument name refers
        # to the dictionary, and also extract the indices
        if use_dict_encoding:
            # In this path the output is still dictionary encoded, so the indices
            # and other attributes need to be copied
            func_text += f"   cache_dict_id = {arg_names[dict_encoded_arg]}._dict_id\n"
            # If we are outputting a dictionary we need a copy of the indices for the new
            # array. Alternatively if we have not propagate_null[dict_encoded_arg] then we
            # will be modifying the indices in place so we must make a copy.
            if (
                out_dtype == bodo.types.string_array_type
                or not propagate_null[dict_encoded_arg]
            ):
                func_text += (
                    f"   indices = {arg_names[dict_encoded_arg]}._indices.copy()\n"
                )
            else:
                func_text += f"   indices = {arg_names[dict_encoded_arg]}._indices\n"
            if out_dtype == bodo.types.string_array_type:
                # In Bodo, if has _has_unique_local_dictionary is True, there are no duplicate values in the
                # dictionary. Therefore, if we're performing an operation that may create duplicate values,
                # we need to set the values appropriately.
                func_text += f"   has_global = {arg_names[dict_encoded_arg]}._has_global_dictionary\n"
                if may_cause_duplicate_dict_array_values:
                    func_text += "   is_dict_unique = False\n"
                else:
                    func_text += f"   is_dict_unique = {arg_names[dict_encoded_arg]}._has_unique_local_dictionary\n"

                func_text += (
                    f"   {arg_names[i]} = {arg_names[dict_encoded_arg]}._data\n"
                )
            # In this path the output is not dictionary encoded, so the data
            # is needed but won't be copied.
            else:
                func_text += (
                    f"   {arg_names[i]} = {arg_names[dict_encoded_arg]}._data\n"
                )

        # If dictionary encoded outputs are not being used, then the output is
        # still bodo.types.string_array_type, the number of loop iterations is still the
        # length of the indices, and scalar_text/propagate_null should work the
        # same because isna checks the data & indices, and scalar_text uses the
        # arguments extracted by getitem.
        func_text += f"   n = {size_text}\n"

        # If prefix_code was provided, embed it before the loop
        # (unless the output is all-null)
        if prefix_code and not out_null:
            func_text += indent_block(prefix_code, 3)

        # cache_dict_arrays dictates if we are in a streaming context where
        # it may beneficial to cache dictionary encoded arrays. This is only supported
        # with regular use_dict_encoding (without use_dict_synthesis) and a function
        # only supports this if it has provided a variable for both dict_encoding_state_name
        # and func_id_name.
        cache_dict_arrays = (
            not use_dict_synthesis
            and use_dict_encoding
            and dict_encoding_state_name is not None
            and func_id_name is not None
        )
        # Additional indentation in case we are caching dictionaries.
        dict_caching_indent = 6 if cache_dict_arrays else 3
        # Code that can be taken in the "cache miss" (e.g. regular computation)
        # needs to be indented as a block, so we write to an intermediate variable.
        non_cached_text = ""

        # If creating a dictionary encoded output from scratch, embed the text
        # to create the dictionary itself before the main loop
        if use_dict_synthesis:
            non_cached_text += indent_block(synthesize_dict_setup_text, 3)
            out_dtype = bodo.libs.dict_arr_ext.dict_indices_arr_type
            non_cached_text += (
                "   res = bodo.utils.utils.alloc_type(n, out_dtype, (-1,))\n"
            )
            non_cached_text += "   numba.parfors.parfor.init_prange()\n"
            non_cached_text += "   for i in numba.parfors.parfor.internal_prange(n):\n"

        # If dictionary encoded outputs are not being used, then the output is
        # still bodo.types.string_array_type, the number of loop iterations is still the
        # length of the indices, and scalar_text/propagate_null should work the
        # same because isna checks the data & indices, and scalar_text uses the
        # arguments extracted by getitem.
        elif use_dict_encoding:
            # Add a null value at the end of the dictionary and compute the scalar
            # kernel output for null values if we are not just propagating input nulls
            # to output array.
            # See test_bool.py::test_equal_null[vector_scalar_string]
            dict_len = "n" if propagate_null[dict_encoded_arg] else "(n + 1)"
            arr_name = arg_names[dict_encoded_arg]
            func_text += "   vec_iter_start = 0\n"
            func_text += f"   vec_iter_end = {dict_len}\n"
            if cache_dict_arrays:
                # Insert code to check the dictionary encoding state for possible caching.
                # If a kernel supports cache_dict_arrays then the same dictionary may be reused
                # for multiple iterations (based on cache_dict_id). If so, we check the state to
                # see if we can skip the computation. Otherwise we generate the result as normal
                # and update the dictionary encoding state.
                func_text += "   use_cached_value = False\n"
                func_text += f"   cached_dict_length = bodo.libs.streaming.dict_encoding.state_contains_dict_array({dict_encoding_state_name}, {func_id_name}, cache_dict_id)\n"
                func_text += "   if cached_dict_length == n:\n"
                func_text += "      res, new_dict_id, _ = bodo.libs.streaming.dict_encoding.get_array(\n"
                func_text += f"         {dict_encoding_state_name},\n"
                func_text += f"         {func_id_name},\n"
                func_text += "         cache_dict_id,\n"
                func_text += "         out_dtype,\n"
                func_text += "      )\n"
                func_text += "   else:\n"
                # If the cached length is non 0, it means we have some prefix of
                # the input data already computed
                non_cached_text += "   vec_iter_start = max(cached_dict_length, 0)\n"
                # How many elements do we iterate over? If we're not propogating
                # nulls, we need to account for the fact that the cached length is
                # off by 1 since we prepended a NULL the first time.
                non_cached_text += "   vec_iter_end = vec_iter_end - vec_iter_start\n"
                if not propagate_null[dict_encoded_arg]:
                    non_cached_text += "   if cached_dict_length >= 0:\n"
                    non_cached_text += "     vec_iter_end -= 1\n"
                    non_cached_text += "   if cached_dict_length < 0:\n"
                    non_cached_text += f"     {arr_name} = bodo.libs.array_kernels.concat([bodo.libs.array_kernels.gen_na_array(1, {arr_name}), {arr_name}])\n"
                # We're only computing over the suffix here
                non_cached_text += "   if cached_dict_length > 0:\n"
                non_cached_text += f"     {arr_name} = {arr_name}[vec_iter_start:]\n"
            else:
                if not propagate_null[dict_encoded_arg]:
                    non_cached_text += f"   {arr_name} = bodo.libs.array_kernels.concat([bodo.libs.array_kernels.gen_na_array(1, {arr_name}), {arr_name}])\n"
            # adding one extra element in dictionary for null output if necessary
            if out_dtype == bodo.types.string_array_type:
                non_cached_text += "   res = bodo.libs.str_arr_ext.pre_alloc_string_array(vec_iter_end, -1)\n"
            else:
                non_cached_text += "   res = bodo.utils.utils.alloc_type(vec_iter_end, out_dtype, (-1,))\n"

            non_cached_text += "   for i in range(vec_iter_end):\n"
        else:
            if res_list:
                non_cached_text += "   res = []\n"
                non_cached_text += "   for i in range(n):\n"
            else:
                non_cached_text += (
                    "   res = bodo.utils.utils.alloc_type(n, out_dtype, (-1,))\n"
                )
                non_cached_text += "   numba.parfors.parfor.init_prange()\n"
                non_cached_text += (
                    "   for i in numba.parfors.parfor.internal_prange(n):\n"
                )

        # If the argument types imply that every row is null, then just set each
        # row of the output array to null
        if out_null:
            non_cached_text += "      bodo.libs.array_kernels.setna(res, i)\n"

        else:
            # For each column that propagates nulls, add an isna check (and
            # also convert Series to arrays)
            for i in range(len(arg_names)):
                if are_arrays[i]:
                    if propagate_null[i]:
                        non_cached_text += f"      if bodo.libs.array_kernels.isna({arg_names[i]}, i):\n"
                        if res_list:
                            non_cached_text += "         res.append(None)\n"
                        else:
                            non_cached_text += (
                                "         bodo.libs.array_kernels.setna(res, i)\n"
                            )
                        non_cached_text += "         continue\n"

            # Add the local variables that the scalar computation will use
            for i in range(len(arg_names)):
                if are_arrays[i]:
                    if alloc_array_scalars:
                        non_cached_text += f"      arg{i} = {arg_names[i]}[i]\n"
                else:
                    non_cached_text += f"      arg{i} = {arg_names[i]}\n"

            if not use_dict_synthesis:
                # Add the scalar computation. The text must use the argument variables
                # in the form arg0, arg1, etc. and store its final answer in res[i].
                non_cached_text += indent_block(scalar_text, 6)

            else:
                # If using dictionary encoded synthesis, do the same but where res[i]
                # will store the index in the dictionary calculated earlier
                non_cached_text += indent_block(synthesize_dict_scalar_text, 6)

        if cache_dict_arrays:
            # Add suffix to precomputed prefix (if the prefix exists)
            non_cached_text += "   if cached_dict_length > 0:\n"
            non_cached_text += "      old_res, new_dict_id, _ = bodo.libs.streaming.dict_encoding.get_array(\n"
            non_cached_text += f"         {dict_encoding_state_name},\n"
            non_cached_text += f"         {func_id_name},\n"
            non_cached_text += "         cache_dict_id,\n"
            non_cached_text += "         out_dtype,\n"
            non_cached_text += "      )\n"
            non_cached_text += (
                "      res = bodo.libs.array_kernels.concat([old_res, res])\n"
            )

            # If we're not extending an old dictionary, generate the new dict id
            # and insert the set_array call. This populates the cache with the
            # result for the next time this kernel is called.
            non_cached_text += "   else:\n"
            if out_dtype == bodo.types.string_array_type:
                non_cached_text += (
                    "      new_dict_id = bodo.libs.dict_arr_ext.generate_dict_id(n)\n"
                )
            else:
                non_cached_text += "      new_dict_id = -1\n"
            # Cache the newly computed results
            non_cached_text += "   bodo.libs.streaming.dict_encoding.set_array(\n"
            non_cached_text += f"      {dict_encoding_state_name},\n"
            non_cached_text += f"      {func_id_name},\n"
            non_cached_text += "      cache_dict_id,\n"
            non_cached_text += "      n,\n"
            non_cached_text += "      res,\n"
            non_cached_text += "      new_dict_id,\n"
            non_cached_text += "   )\n"

        # Insert the caching else path into the func_text.
        func_text += indent_block(non_cached_text, dict_caching_indent)

        # If using dictionary encoding, construct the output from the
        # new dictionary + the indices
        if use_dict_encoding:
            if not propagate_null[dict_encoded_arg]:
                # If NULL was transformed to a non-NULL value, then we need to
                # update `indices` (which is a copy for bodo.types.string_array_type
                # and therefore safe to modify) so that NULL entries map to the
                # newly added transformed NULL value.
                # TODO(aneesh) in the case where NULL was mapped to NULL, it
                # would be faster if we could just "pop" the first element off of
                # the dictionary instead.
                func_text += "   numba.parfors.parfor.init_prange()\n"
                func_text += (
                    "   for i in numba.parfors.parfor.internal_prange(len(indices)):\n"
                )
                func_text += "     if bodo.libs.array_kernels.isna(indices, i):\n"
                func_text += "       indices[i] = 0\n"
                func_text += "     else:\n"
                func_text += "       indices[i] += 1\n"
            # Flush the nulls back to the index array, if necessary
            if use_null_flushing:
                func_text += "   numba.parfors.parfor.init_prange()\n"
                func_text += (
                    "   for i in numba.parfors.parfor.internal_prange(len(indices)):\n"
                )
                func_text += "      if not bodo.libs.array_kernels.isna(indices, i):\n"
                func_text += "         loc = indices[i]\n"
                func_text += "         if bodo.libs.array_kernels.isna(res, loc):\n"
                func_text += "            bodo.libs.array_kernels.setna(indices, i)\n"
            # If the output dtype is a string array, create the new dictionary encoded array
            if out_dtype == bodo.types.string_array_type:
                dict_id = "new_dict_id" if cache_dict_arrays else "None"
                func_text += f"   res = bodo.libs.dict_arr_ext.init_dict_arr(res, indices, has_global, is_dict_unique, {dict_id})\n"
            # Otherwise, use the indices to copy the values from the smaller array
            # into a larger one (flushing nulls along the way)
            else:
                func_text += "   res2 = bodo.utils.utils.alloc_type(len(indices), out_dtype, (-1,))\n"
                func_text += "   numba.parfors.parfor.init_prange()\n"
                func_text += (
                    "   for i in numba.parfors.parfor.internal_prange(len(indices)):\n"
                )
                # Copy nulls from the old index array to the output array
                if propagate_null[dict_encoded_arg]:
                    func_text += "      if bodo.libs.array_kernels.isna(indices, i):\n"
                    func_text += "         bodo.libs.array_kernels.setna(res2, i)\n"
                    func_text += "         continue\n"
                # For not propagate_null[dict_encoded_arg] we have maps nulls to 0
                func_text += "      loc = indices[i]\n"
                # Copy nulls from the smaller array to the output array
                func_text += "      if bodo.libs.array_kernels.isna(res, loc):\n"
                func_text += "         bodo.libs.array_kernels.setna(res2, i)\n"
                # Copy values from the smaller array to the larger array
                func_text += "      else:\n"
                func_text += "         res2[i] = res[loc]\n"
                func_text += "   res = res2\n"

        # If prefix_text was provided, embed it after the loop
        func_text += indent_block(suffix_code, 3)

        # If using dictionary encoded synthesis, construct the final dict-encoded array
        if use_dict_synthesis:
            func_text += f"   return bodo.libs.dict_arr_ext.init_dict_arr(dict_res, res, {synthesize_dict_global}, {synthesize_dict_unique}, None)\n"
        else:
            func_text += "   return res"
    loc_vars = {}
    exec_globals = {
        "bodosql": bodosql,
        "bodo": bodo,
        "math": math,
        "numba": numba,
        "re": re,
        "np": np,
        "out_dtype": out_dtype,
        "pd": pd,
        "datetime": datetime,
    }
    if extra_globals is not None:
        exec_globals.update(extra_globals)
    exec(
        func_text,
        exec_globals,
        loc_vars,
    )
    impl_gen_vectorized = loc_vars["impl_gen_vectorized"]

    return impl_gen_vectorized


def gen_coerced(func_name, cast_func_name, arg_names, index_to_cast):
    """Generates a function that coerces the argument at the given index
    to the correct type for the function.

    Args:
        func_name (str): the name of the function that will be called
        cast_func_name (str): the name of the function that will be called to cast the argument, with a format
            string for the argument
        arg_names (list of str): the names of the arguments to the function
        index_to_cast (int): the index of the argument to coerce

    Returns:
        function: the generated function
    """

    arg_strings = []
    for i in range(len(arg_names)):
        if i == index_to_cast:
            arg_strings.append(cast_func_name.format(arg_names[i]))
        else:
            arg_strings.append(arg_names[i])
    func_text = f"def impl({', '.join(arg_names)}):\n"
    func_text += f"  return {func_name}({', '.join(arg_strings)})"

    loc_vars = {}
    exec(func_text, {"bodo": bodo, "bodosql": bodosql}, loc_vars)

    return loc_vars["impl"]


def convert_numeric_to_int(func_name, arg_names, args, numeric_args_to_cast):
    func_text = f"def impl({', '.join(arg_names)}):\n"
    for arg_name, arg in zip(arg_names, args):
        # This conversion is only done for float arguments
        if (
            arg_name in numeric_args_to_cast
            and is_valid_float_arg(arg)
            or (isinstance(arg, types.optional) and is_valid_float_arg(arg.type))
        ):
            func_text += f"  {arg_name} = bodosql.kernels.casting_array_kernels.round_to_int64({arg_name})\n"
    func_text += f"  return {func_name}({', '.join(arg_names)})\n"

    loc_vars = {}
    exec(func_text, {"bodo": bodo, "bodosql": bodosql}, loc_vars)
    impl = loc_vars["impl"]

    return impl


def unopt_argument(
    func_name, arg_names, i, container_arg=0, container_length=None, default_map=None
):
    """Creates an impl that cases on whether or not a certain argument to a function
       is None in order to un-optionalize that argument

    Args:
        func_name (string): the name of the function with the optional arguments
        arg_names (string list): the name of each argument to the function
        i (integer): the index of the argument from arg_names being unoptionalized
        container_arg (optional int): Which argument in the container are we checking?
            Used alongside container_length.
        container_length (optional int): if provided, treat the arg_names[i] as
        a container of this many arguments. Used so we can pass in arbitrary sized
        containers or arguments to handle SQL functions with variadic arguments,
        such as coalesce
        default_map (Optional[Dict[str, any]]): A map from argument name to default value
            for the header.

    Returns:
        function: the impl that re-calls func_name with arg_names[i] no longer optional
    """
    # Fill in the default argument.
    if default_map is None:
        header_list = arg_names
    else:
        header_list = [
            f"{arg}={default_map[arg]!r}" if arg in default_map else arg
            for arg in arg_names
        ]
    if container_length != None:
        # If this path the one of the arguments is a tuple of arrays and the optional
        # value is a member of the tuple. In this path we execute the follow steps.
        # Step 1: Replace the tuple with a new tuple.
        # Step 2: Generate the new function calls.

        # Here is an example:
        #   call(arg0, arg1, arg2)
        #   i = 1
        #   container_arg = 2
        #   container_length = 3
        #
        # Here are the two different tuple options:
        # args1_str = (arg1[0], arg1[1], None)
        # args2_str = (arg1[0], arg1[1], bodo.utils.indexing.unoptional(arg1[2]))
        #
        # Then the total call becomes
        # total_args1 = arg0, (arg1[0], arg1[1], None), arg2
        # total_args2 = arg0, (arg1[0], arg1[1], bodo.utils.indexing.unoptional(arg1[2])), arg2
        args1 = [
            f"{arg_names[i]}{[j]}" if j != container_arg else "None"
            for j in range(container_length)
        ]
        # Note: (,) is not valid code.
        extra_comma = "," if container_length != 0 else ""
        args1_str = f"({', '.join(args1)}{extra_comma})"
        args2 = [
            f"{arg_names[i]}{[j]}"
            if j != container_arg
            else f"bodo.utils.indexing.unoptional({arg_names[i]}[{j}])"
            for j in range(container_length)
        ]
        args2_str = f"({', '.join(args2)}{extra_comma})"
        total_args1 = [
            arg_names[j] if j != i else args1_str for j in range(len(arg_names))
        ]
        total_args2 = [
            arg_names[j] if j != i else args2_str for j in range(len(arg_names))
        ]
        func_text = f"def impl({', '.join(header_list)}):\n"
        func_text += f"   if {arg_names[i]}[{container_arg}] is None:\n"
        func_text += f"      return {func_name}({', '.join(total_args1)})\n"
        func_text += "   else:\n"
        func_text += f"      return {func_name}({', '.join(total_args2)})\n"
    else:
        # In this path we just replace individual arguments.
        #   call(arg0, arg1, arg2)
        #   i = 1
        #
        # args1 = (arg0, None, arg2)
        # args2 = (arg0, bodo.utils.indexing.unoptional(arg1), arg2)
        #
        args1 = [arg_names[j] if j != i else "None" for j in range(len(arg_names))]
        args2 = [
            arg_names[j]
            if j != i
            else f"bodo.utils.indexing.unoptional({arg_names[j]})"
            for j in range(len(arg_names))
        ]
        func_text = f"def impl({', '.join(header_list)}):\n"
        func_text += f"   if {arg_names[i]} is None:\n"
        func_text += f"      return {func_name}({', '.join(args1)})\n"
        func_text += "   else:\n"
        func_text += f"      return {func_name}({', '.join(args2)})\n"

    loc_vars = {}
    exec(
        func_text,
        {
            "bodo": bodo,
            "bodosql": bodosql,
            "numba": numba,
        },
        loc_vars,
    )
    impl = loc_vars["impl"]

    return impl


def is_decimal_float_pair(arg1, arg2):
    """
    Verifies that the pair of arguments to a SQL function is a decimal and a float
    (scalar or vector)

    Args:
        arg1 (dtype): the dtype of the first argument being checked
        arg2 (dtype): the dtype of the second argument being checked

    returns: True if the arguments are a decimal and a float, False otherwise
    """

    return (is_valid_float_arg(arg1) and is_valid_decimal_arg(arg2)) or (
        is_valid_float_arg(arg2) and is_valid_decimal_arg(arg1)
    )


def is_valid_numeric_bool(arg):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is a numeric or boolean
        (scalar or vector)

    Args:
        arg (dtype): the dtype of the argument being checked

    returns: True if the argument is a numeric or boolean, False otherwise
    """
    return not (
        arg != types.none
        and not isinstance(arg, (types.Integer, types.Float, types.Boolean))
        and not (
            bodo.utils.utils.is_array_typ(arg, True)
            and isinstance(arg.dtype, (types.Integer, types.Float, types.Boolean))
        )
        and not is_overload_constant_number(arg)
        and not is_overload_constant_bool(arg)
    )


def verify_int_arg(arg, f_name, a_name):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is an integer
       (scalar or vector)

    Args:
        arg (dtype): the dtype of the argument being checked
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being checked

    raises: BodoError if the argument is not an integer, integer column, or NULL
    """
    if not is_valid_int_arg(arg):
        raise_bodo_error(
            f"{f_name} {a_name} argument must be an integer, integer column, or null, but was {arg}"
        )


def is_numeric_without_decimal(arg):
    return (
        is_overload_none(arg)
        or isinstance(arg, (types.Integer, types.Boolean, types.Float))
        or (
            bodo.utils.utils.is_array_typ(arg, True)
            and isinstance(arg.dtype, (types.Integer, types.Boolean, types.Float))
        )
    )


def verify_int_float_arg(arg, f_name, a_name):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is an integer, float,
       or boolean (scalar or vector)

    Args:
        arg (dtype): the dtype of the argument being checked
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being checked

    raises: BodoError if the argument is not an integer/float/bool scalar/column, or NULL
    """
    if not is_numeric_without_decimal(arg):
        raise_bodo_error(
            f"{f_name} {a_name} argument must be a numeric, numeric column, or null, but was {arg}"
        )


def verify_numeric_arg(arg, f_name, a_name):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is an integer, float,
    boolean, or decimal (scalar or vector)

    Args:
        arg (dtype): the dtype of the argument being checked
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being checked

    raises: BodoError if the argument is not an integer/float/bool scalar/column, or NULL
    """
    if not (
        is_numeric_without_decimal(arg)
        or isinstance(arg, bodo.types.Decimal128Type)
        or (
            bodo.utils.utils.is_array_typ(arg, True)
            and isinstance(arg.dtype, bodo.types.Decimal128Type)
        )
    ):
        raise_bodo_error(
            f"{f_name} {a_name} argument must be a numeric, numeric column, or null, but was {arg}"
        )


def is_valid_string_arg(arg):  # pragma: no cover
    """
    Args:
        arg (dtype): the dtype of the argument being checked
    returns: False if the argument is not a string or string column
    """
    arg = types.unliteral(arg)
    return not (
        arg not in (types.none, types.unicode_type)
        and not (
            bodo.utils.utils.is_array_typ(arg, True)
            and (arg.dtype == types.unicode_type)
        )
        and not is_overload_constant_str(arg)
    )


def is_valid_binary_arg(arg):  # pragma: no cover
    """
    Args:
        arg (dtype): the dtype of the argument being checked
    returns: False if the argument is not binary data
    """
    return not (
        arg not in (types.none, bodo.types.bytes_type)
        and not (
            bodo.utils.utils.is_array_typ(arg, True)
            and arg.dtype == bodo.types.bytes_type
        )
        and not is_overload_constant_bytes(arg)
        and not isinstance(arg, types.Bytes)
    )


def is_valid_datetime_or_date_arg(arg):
    """
    Args:
        arg (dtype): the dtype of the argument being checked
    returns: True if the input argument is a scalar or vector date, datetime64,
    or timestamp type (with or without a timezone).

    Note; If the presence or absence of a timezone is important, specifically use
    is_valid_tz_aware_datetime_arg to check if the dtype has a timezone.
    """

    return (
        is_valid_date_arg(arg)
        or is_valid_tz_naive_datetime_arg(arg)
        or is_valid_tz_aware_datetime_arg(arg)
    )


def is_valid_timedelta_arg(arg):
    """
    Args:
        arg (dtype): the dtype of the argument being checked
    returns: False if the argument is not timedelta data

    Note: In BodoSQL, scalar timedelta types are both timedelta64ns,
    and the columnar timedelta types are both .
    """

    return (
        arg == pd_timedelta_type
        or arg == types.NPTimedelta("ns")
        or isinstance(arg, DateOffsetType)
        or (
            bodo.utils.utils.is_array_typ(arg, True)
            and (
                is_timedelta64_series_typ(arg)
                or isinstance(arg, PDTimeDeltaType)
                or arg.dtype == bodo.types.timedelta64ns
                or arg == bodo.types.timedelta_array_type
            )
        )
    )


def is_valid_boolean_arg(arg):  # pragma: no cover
    """
    Args:
        arg (dtype): the dtype of the argument being checked
    returns: False if the argument is not a boolean or boolean column
    """
    return not (
        arg != types.boolean
        and not (
            bodo.utils.utils.is_array_typ(arg, True) and arg.dtype == types.boolean
        )
        and not is_overload_constant_bool(arg)
    )


def is_valid_decimal_arg(arg):  # pragma: no cover
    """
    Args:
        arg (dtype): the dtype of the argument being checked
    returns: False if the argument is not a decimal or decimal column
    """
    return isinstance(arg, (bodo.types.Decimal128Type, bodo.types.DecimalArrayType))


def verify_array_arg(arg, is_scalar, f_name, a_name):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is an array
       (scalar or vector)
    Args:
        arg (dtype): the dtype of the argument being checked
        is_scalar (boolean): if True, checks if arg is any array,
        if False checks if it is an array item array.
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being checked
    raises: BodoError if the argument is not a array scalar, array column, or null
    """
    if arg == bodo.types.none or arg == bodo.types.null_array_type:
        return
    if is_scalar:
        if not bodo.utils.utils.is_array_typ(arg, False):
            raise_bodo_error(
                f"{f_name} {a_name} argument must be an array scalar, array column, or null, but was {arg}"
            )
    else:
        if not (
            isinstance(arg, bodo.types.ArrayItemArrayType)
            or (
                bodo.hiframes.pd_series_ext.is_series_type(arg)
                and isinstance(arg.data, bodo.types.ArrayItemArrayType)
            )
        ):
            raise_bodo_error(
                f"{f_name} {a_name} argument must be an array scalar, array column, or null, but was {arg}"
            )


def verify_string_arg(arg, f_name, a_name):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is a string
       (scalar or vector)
    Args:
        arg (dtype): the dtype of the argument being checked
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being checked
    raises: BodoError if the argument is not a string, string column, or null
    """
    if not is_valid_string_arg(arg):
        raise_bodo_error(
            f"{f_name} {a_name} argument must be a string, string column, or null, but was {arg}"
        )


def verify_scalar_string_arg(arg, f_name, a_name):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is a scalar string
    Args:
        arg (dtype): the dtype of the argument being checked
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being checked
    raises: BodoError if the argument is not a scalar string
    """
    if arg not in (types.unicode_type, bodo.types.none) and not isinstance(
        arg, types.StringLiteral
    ):
        raise_bodo_error(
            f"{f_name} {a_name} argument must be a scalar string, but was {arg}"
        )


def verify_binary_arg(arg, f_name, a_name):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is binary data
       (scalar or vector)
    Args:
        arg (dtype): the dtype of the argument being checked
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being checked
    raises: BodoError if the argument is not binary data or null
    """
    if not is_valid_binary_arg(arg):
        raise_bodo_error(
            f"{f_name} {a_name} argument must be binary data or null, but was {arg}"
        )


def verify_string_binary_arg(arg, f_name, a_name):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is binary data, string, or string column
       (scalar or vector)
    Args:
        arg (dtype): the dtype of the argument being checked
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being checked
    raises: BodoError if the argument is not binary data, string, string column or null
    returns: True if the argument is a string, False if the argument is binary data, and None if the argument is both string and binary i.e. NoneType
    """
    is_string = is_valid_string_arg(arg)
    is_binary = is_valid_binary_arg(arg)

    if is_string and is_binary:
        return None
    elif is_string:
        return True
    elif is_binary:
        return False
    else:
        raise_bodo_error(
            f"{f_name} {a_name} argument must be a binary data, string, string column, or null, but was {arg}"
        )


def verify_string_numeric_arg(
    arg, f_name, a_name, include_decimal=False
):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is a string, integer, float, or boolean
        (scalar or vector)
    Args:
        arg (dtype): the dtype of the argument being checked
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being checked
    raises: BodoError if the argument is not a string, integer, float, boolean, string column,
            integer column, float column, or boolean column
    """
    if not is_valid_string_arg(arg) and not is_valid_numeric_bool(arg):
        if not include_decimal or not is_valid_decimal_arg(arg):
            raise_bodo_error(
                f"{f_name} {a_name} argument must be a string, integer, float, boolean, string column, integer column, float column, or boolean column, but was {arg}"
            )


def verify_boolean_arg(arg, f_name, a_name):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is a boolean
       (scalar or vector)

    Args:
        arg (dtype): the dtype of the argument being checked
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being checked

    raises: BodoError if the argument is not an boolean, boolean column, or NULL
    """
    if (
        arg not in (types.none, types.boolean)
        and not (
            bodo.utils.utils.is_array_typ(arg, True) and arg.dtype == types.boolean
        )
        and not is_overload_bool(arg)
    ):
        raise_bodo_error(
            f"{f_name} {a_name} argument must be a boolean, boolean column, or null, but was {arg}"
        )


def is_valid_timestamptz_arg(arg):
    return arg == bodo.types.timestamptz_type or (
        bodo.utils.utils.is_array_typ(arg, True)
        and arg.dtype == bodo.types.timestamptz_type
    )


def is_valid_date_arg(arg):
    """
    Is the type an acceptable date argument for a BodoSQL array
    kernel. This is a date scalar, array, or Series value.

    Args:
        arg (types.Type): A Bodo type.

    Returns:
        bool: Is this type one of the date types.
    """
    return arg == bodo.types.datetime_date_type or (
        bodo.utils.utils.is_array_typ(arg, True)
        and arg.dtype == bodo.types.datetime_date_type
    )


def is_valid_tz_naive_datetime_arg(arg):
    """
    Is the type an acceptable tz naive datetime argument for a BodoSQL array
    kernel. This is a Timestamp scalar where tz == None, dt64 array, or
    dt64 Series.

    Args:
        arg (types.Type): A Bodo type.

    Returns:
        bool: Is this type one of the tz-naive datetime types.
    """
    return (
        arg
        in (
            bodo.types.datetime64ns,
            bodo.types.pd_timestamp_tz_naive_type,
            bodo.types.pd_datetime_tz_naive_type,
        )
        or (
            bodo.utils.utils.is_array_typ(arg, True)
            and (
                arg.dtype == bodo.types.datetime64ns
                or (
                    isinstance(
                        arg.dtype, bodo.libs.pd_datetime_arr_ext.PandasDatetimeTZDtype
                    )
                    and arg.dtype.tz is None
                )
            )
        )
        or (isinstance(arg, bodo.types.DatetimeArrayType) and arg.tz is None)
    )


def is_valid_tz_aware_datetime_arg(arg):
    """
    Is the type an acceptable tz aware datetime argument for a BodoSQL array
    kernel. This is a Timestamp scalar where tz != None, DatetimeArray, or
    DatetimeArray Series.

    Args:
        arg (types.Type): A Bodo type.

    Returns:
        bool: Is this type one of the tz-aware datetime types.
    """
    return (isinstance(arg, bodo.types.PandasTimestampType) and arg.tz is not None) or (
        bodo.utils.utils.is_array_typ(arg, True)
        and isinstance(arg.dtype, bodo.libs.pd_datetime_arr_ext.PandasDatetimeTZDtype)
        and arg.dtype.tz is not None
    )


def is_valid_timestamptz_arg(arg):
    return arg == bodo.types.timestamptz_type or (
        bodo.utils.utils.is_array_typ(arg, True)
        and arg.dtype == bodo.types.timestamptz_type
    )


def verify_datetime_arg(arg, f_name, a_name):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is a datetime
       (scalar or vector)

    Args:
        arg (dtype): the dtype of the argument being checked
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being checked

    raises: BodoError if the argument is not a datetime, datetime column, or NULL
    """
    if not (
        is_overload_none(arg)
        or is_valid_date_arg(arg)
        or is_valid_tz_naive_datetime_arg(arg)
    ):
        raise_bodo_error(
            f"{f_name} {a_name} argument must be a datetime, datetime column, or null without a tz, but was {arg}"
        )


def verify_date_arg(arg, f_name, a_name):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is a datetime.date
       (scalar or vector)

    Args:
        arg (dtype): the dtype of the argument being checked
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being checked

    raises: BodoError if the argument is not a datetime, datetime column, or NULL
    """
    if not (is_overload_none(arg) or is_valid_date_arg(arg)):
        raise_bodo_error(
            f"{f_name} {a_name} argument must be a date, date column, or null object, but was {arg}"
        )


def verify_datetime_arg_allow_tz(
    arg, f_name, a_name, allow_timestamp_tz=False
):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is a datetime
       (scalar or vector) that allows timezones.

    Args:
        arg (dtype): the dtype of the argument being checked
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being checked
        allow_timestamp_tz (bool): True if TIMESTAMP_TZ arguments are allowed

    raises: BodoError if the argument is not a datetime, datetime column, or NULL
    """
    if not (
        is_overload_none(arg)
        or is_valid_date_arg(arg)
        or is_valid_tz_naive_datetime_arg(arg)
        or is_valid_tz_aware_datetime_arg(arg)
        or (allow_timestamp_tz and is_valid_timestamptz_arg(arg))
    ):
        raise_bodo_error(
            f"{f_name} {a_name} argument must be a datetime, datetime column, or null, but was {arg}"
        )


def verify_timestamp_tz_arg(
    arg, f_name, a_name, allow_timestamp_tz=False
):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is a timestamp_tz
       (scalar or vector).

    Args:
        arg (dtype): the dtype of the argument being checked
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being checked

    raises: BodoError if the argument is not a datetime, datetime column, or NULL
    """
    if not is_valid_timestamptz_arg(arg):
        raise_bodo_error(
            f"{f_name} {a_name} argument must be a timestamp_tz scalar/column or null, but was {arg}"
        )


def verify_timestamp_arg_allow_tz(arg, f_name, a_name):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is a timestamp
       (scalar or vector) that allows timezones. This is different from
       verify_datetime_arg_allow_tz because it doesn't allow date.

    Args:
        arg (dtype): the dtype of the argument being checked
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being checked

    raises: BodoError if the argument is not a timestamp, timestamp column, or NULL
    """
    if not (
        is_overload_none(arg)
        or is_valid_tz_naive_datetime_arg(arg)
        or is_valid_tz_aware_datetime_arg(arg)
        or is_valid_timestamptz_arg(arg)
    ):
        raise_bodo_error(
            f"{f_name} {a_name} argument must be a timestamp, timestamp column, or null, but was {arg}"
        )


def verify_datetime_arg_require_tz(arg, f_name, a_name):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is a datetime
    (scalar or vector) that has a timezone.
    Args:
        arg (dtype): the dtype of the argument being checked
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being checked
    raises: BodoError if the argument is not a datetime, datetime column, or NULL
    """
    if not (is_overload_none(arg) or is_valid_tz_aware_datetime_arg(arg)):
        raise_bodo_error(
            f"{f_name} {a_name} argument must be a tz-aware datetime, datetime column, or null, but was {arg}"
        )


def verify_sql_interval(arg, f_name, a_name):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is an acceptable
    interval. This is either a valid Timedelta scalar or array, None, or
    pd.DateOffset.
    Args:
        arg (dtype): the dtype of the argument being checked
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being checked
    raises: BodoError if the argument is a invalid interval type
    """
    if not (
        is_overload_none(arg)
        or is_valid_timedelta_arg(arg)
        or arg == bodo.types.date_offset_type
    ):
        raise_bodo_error(
            f"{f_name} {a_name} argument must be a Timedelta scalar/column, DateOffset, or null, but was {arg}"
        )


def verify_td_arg(arg, f_name, a_name):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is a timedelta
    array or scalar.
    Args:
        arg (dtype): the dtype of the argument being checked
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being checked
    raises: BodoError if the argument is an invalid timedelta type
    """
    if not (is_overload_none(arg) or is_valid_timedelta_arg(arg)):
        raise_bodo_error(
            f"{f_name} {a_name} argument must be a Timedelta scalar/column or null, but was {arg}"
        )


def get_tz_if_exists(arg):  # pragma: no cover
    """Returns the timezone from a scalar or vector datetime/timestamp argument,
        or None if it has no timestamp.

    Args:
        arg (dtype): the dtype of the argument whose timezone is being extracted.

    Returns: the timezone (if the argument has one) or None (if it does not)
    """
    if is_valid_tz_aware_datetime_arg(arg):
        if bodo.utils.utils.is_array_typ(arg, True):
            return arg.dtype.tz
        else:
            return arg.tz
    return None


def is_valid_time_arg(arg):
    """
    Is the type an acceptable time argument for a BodoSQL array
    kernel. This is a date time, array, or Series value.

    Args:
        arg (types.Type): A Bodo type.

    Returns:
        bool: Is this type a time type
    """
    return isinstance(arg, bodo.types.TimeType) or (
        bodo.utils.utils.is_array_typ(arg, True)
        and isinstance(arg.dtype, bodo.bodo.types.TimeType)
    )


def verify_time_or_datetime_arg_allow_tz(
    arg, f_name, a_name, allow_timestamp_tz=False
):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is a time/datetime
       (scalar or vector) that allows timezones.

    Args:
        arg (dtype): the dtype of the argument being checked
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being checked
        allow_timestamp_tz (bool): True if TIMESTAMP_TZ arguments are allowed

    raises: BodoError if the argument is not a datetime, datetime column, or NULL
    """
    if not (
        is_overload_none(arg)
        or is_valid_date_arg(arg)
        or is_valid_time_arg(arg)
        or is_valid_tz_naive_datetime_arg(arg)
        or is_valid_tz_aware_datetime_arg(arg)
        or (allow_timestamp_tz and is_valid_timestamptz_arg(arg))
    ):
        raise_bodo_error(
            f"{f_name} {a_name} argument must be a time/datetime, time/datetime column, or null without a tz, but was {arg}"
        )


def verify_date_or_datetime_arg_forbid_tz(arg, f_name, a_name):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is a date/datetime
       (scalar or vector) that does not allow timezones.

    Args:
        arg (dtype): the dtype of the argument being checked
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being checked

    raises: BodoError if the argument is not a datetime, datetime column, or NULL
    """
    if not (
        is_overload_none(arg)
        or is_valid_date_arg(arg)
        or is_valid_tz_naive_datetime_arg(arg)
    ):
        raise_bodo_error(
            f"{f_name} {a_name} argument must be a date/datetime, date/datetime column, or null without a tz, but was {arg}"
        )


def verify_time_or_datetime_arg_forbid_tz(arg, f_name, a_name):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is a time/datetime
       (scalar or vector) that does not allow timezones.

    Args:
        arg (dtype): the dtype of the argument being checked
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being checked

    raises: BodoError if the argument is not a datetime, datetime column, or NULL
    """
    if not (
        is_overload_none(arg)
        or is_valid_date_arg(arg)
        or is_valid_time_arg(arg)
        or is_valid_tz_naive_datetime_arg(arg)
    ):
        raise_bodo_error(
            f"{f_name} {a_name} argument must be a time/datetime, time/datetime column, or null without a tz, but was {arg}"
        )


def get_common_broadcasted_type(arg_types, func_name, suppress_error=False):
    """Takes in a list of types from arrays/Series/scalars, verifies that they
    have a common underlying scalar type, and if so returns the corresponding
    array type (+ ensures that it is nullable). Assumes scalar Nones coerce to any
    type.  In all other cases, throws an error.

    Args:
        arg_types (dtype list/tuple): the types of the arrays/Series/scalars being checked
        func_name (string): the name of the function being compiled
        suppress_error (boolean): if True, don't return an error if the types are incompatible and return None instead

    Returns:
        dtype: the common underlying dtype of the inputted types. If all inputs are
            Nonetype, returns nonetype, as all inputs are scalar, and there is no need
            to find a common array type.

    raises: BodoError if the underlying types are not compatible
    """
    # Extract the underlying type of each scalar/vector
    elem_types = []
    for i in range(len(arg_types)):
        # Array
        if bodo.utils.utils.is_array_typ(arg_types[i], False):
            elem_types.append(arg_types[i])
        # Series
        elif bodo.utils.utils.is_array_typ(arg_types[i], True):
            elem_types.append(arg_types[i].data)
        # Scalar
        else:
            elem_types.append(arg_types[i])
    if len(elem_types) == 0:
        return bodo.types.none
    elif len(elem_types) == 1:
        if bodo.utils.utils.is_array_typ(elem_types[0]):
            return bodo.utils.typing.to_nullable_type(elem_types[0])
        elif elem_types[0] == bodo.types.none:
            return bodo.types.none
        else:
            return bodo.utils.typing.to_nullable_type(
                bodo.utils.typing.dtype_to_array_type(elem_types[0])
            )
    else:
        # Verify that the underlying scalar types are common before extracting
        # the corresponding output_dtype
        scalar_dtypes = []
        for i in range(len(arg_types)):
            if bodo.utils.utils.is_array_typ(arg_types[i]):
                scalar_dtypes.append(elem_types[i].dtype)
            # Avoid appending nonetypes to elem_types,
            # as scalar NULL coerces to any type.
            elif elem_types[i] == bodo.types.none:
                pass
            else:
                scalar_dtypes.append(elem_types[i])

        # All arguments were None scalars, return none
        if len(scalar_dtypes) == 0:
            return bodo.types.none

        common_dtype, _ = bodo.utils.typing.get_common_scalar_dtype(
            scalar_dtypes, allow_downcast=True
        )
        if common_dtype is None:
            if suppress_error:
                return None
            raise_bodo_error(
                f"Cannot call {func_name} on columns with different dtypes: {scalar_dtypes}"
            )
        return bodo.utils.typing.to_nullable_type(
            bodo.utils.typing.dtype_to_array_type(common_dtype)
        )


def vectorized_sol(args, scalar_fn, dtype, manual_coercion=False):
    """Creates a py_output for a vectorized function using its arguments and the
       a function that is applied to the scalar values

    Args:
        args (any list): a list of arguments, each of which is either a scalar
        or vector (vectors must be the same size)
        scalar_fn (function): the function that is applied to scalar values
        corresponding to each row
        dtype (dtype): the dtype of the final output array
        manual_coercion (boolean, optional): whether to manually coerce the
        non-null elements of the output array to the dtype

    Returns:
        scalar or Series: the result of applying scalar_fn to each row of the
        vectors with scalar args broadcasted (or just the scalar output if
        all of the arguments are scalar)
    """
    length = -1
    for arg in args:
        if isinstance(
            arg, (pd.core.arrays.base.ExtensionArray, pd.Series, np.ndarray, pa.Array)
        ):
            length = len(arg)
            break
    if length == -1:
        return dtype(scalar_fn(*args)) if manual_coercion else scalar_fn(*args)
    arglist = []
    for arg in args:
        if isinstance(
            arg, (pd.core.arrays.base.ExtensionArray, pd.Series, np.ndarray, pa.Array)
        ):
            arglist.append(arg)
        else:
            arglist.append([arg] * length)
    if manual_coercion:
        return pd.Series([dtype(scalar_fn(*params)) for params in zip(*arglist)])
    else:
        return pd.Series([scalar_fn(*params) for params in zip(*arglist)], dtype=dtype)


def gen_windowed(
    calculate_block,
    out_dtype,
    constant_block=None,
    setup_block=None,
    enter_block=None,
    exit_block=None,
    empty_block=None,
    num_args=1,
    propagate_nan=True,
    extra_globals=None,
    min_elements=1,
):
    """Creates an impl for a window frame function that accumulates some value
    as elements enter and exit a window that starts/ends some number of indices
    before/after each element of the array. Unbounded preceding/following
    can be implemented by providing lower/upper bounds that are larger than
    the size of the input.

    Note: the implementations are only designed to work sequentially, BodoSQL
    window functions currently only work on partitioned data.

    Args:
        calculate_block (string): How should the current array value be
        calculated in terms of the up-to-date accumulators

        out_dtype (dtype): what is the dtype of the output data.

        constant_block (optional): What should happen if the output value will
        always be the same due to the window size being so big. If None,
        this means that there is no such case.

        setup_block (string, optional): What should happen to initialize the
        accumulators. If not provided, does nothing.

        enter_block (string, optional): what should happen when a non-null
        element enters the window. If not provided, does nothing.

        exit_block (string, optional): what should happen when a non-null
        element exits the window. If not provided, does nothing.

        empty_block (string, optional): What should happen if the window frame
        only contains null/empty. If not provided, calls setna.

        num_args (integer): how many arguments the function takes in.
        The default is 1. Only 1 or 2 arguments supported. If 1, the input is
        called S. If 2, the inputs are called (Y, X).

        propagate_nan (boolean): If True, the output is NaN if any element
        in the current window frame is NaN. Note: if a NaN is encountered, the
        enter/exit blocks are not invoked. The default value is True.

        min_elements [integer]: the minimum number of non-null elements required
        to produce a non-null output.

    Returns:
        function: a window function that takes in a Series, lower bound,
        and upper bound, and outputs an array where each value corresponds
        to the desired aggregation of the values from the specified bounds
        starting from each index.

    Usage notes:
        - When writing enter_block/exit_block/calculate_block, the variable "elemi" is used to
          denote the current value entering/exiting the ith array. You may assume
          that this value is not null. E.g. if there are 2 inputs, the values
          are called elem0 and elem1.
        - The variable "in_window "is used to denote the length of the window
          frame that is not out of bounds, excluding nulls.
        - When writing calculate_block, store the final answer as "res[i] = ..."
        - When writing constant_block, store the value answer as "constant_value = ..."
        - The original variables can be acccessed as S (if 1 argument), or Y and X
          (if 2 arguments).
        - After being coerced to arrays, they can be accessed as arr0, arr1, etc.

    Below is an example of the generated code when used to calculate sum on an
    array of floats:

    calculate_block = "res[i] = total"
    constant_block = "constant_value = S.sum()"
    setup_block = "total = 0"
    enter_block = "total += elem0"
    exit_block = "total -= elem0"
    empty_block = None
    num_args = 1
    propagate_nan = True
    min_elements = 1

    def impl(S, lower_bound, upper_bound):
        n = len(S)
        arr0 = bodo.utils.conversion.coerce_to_array(S)
        res = bodo.utils.utils.alloc_type(n, out_dtype, (-1,))
        if upper_bound < lower_bound:
            for i in range(n):
                bodo.libs.array_kernels.setna(res, i)
        elif lower_bound <= -n+1 and n-1 <= upper_bound:
            non_null_count = False
            has_nan = False

            for i in range(n):
                if not (bodo.libs.array_kernels.isna(arr0, i)):
                    non_null_count += 1
                if np.isnan(arr0[i]):
                    has_nan = True

            if has_nan:
                for i in range(n):
                    res[i] = np.nan

            elif non_null_count < 1:
                for i in range(n):
                    bodo.libs.array_kernels.setna(res, i)
            else:
                constant_value = S.sum()
                res[:] = constant_value
        else:
            exiting = lower_bound
            entering = upper_bound
            in_window = 0
            nan_counter = 0
            total = 0
            for i in range(min(max(0, exiting), n), min(max(0, entering + 1), n)):
                if not (bodo.libs.array_kernels.isna(arr0, i)):
                    in_window += 1
                    elem0 = arr0[i]
                    if np.isnan(arr0[i]):
                        nan_counter += 1
                    else:
                        total += elem0
            for i in range(n):
                if in_window < 1:
                    bodo.libs.array_kernels.setna(res, i)
                else:
                    if nan_counter > 0:
                        res[i] = np.nan
                    else:
                        res[i] = total
                if 0 <= exiting < n:
                    if not (bodo.libs.array_kernels.isna(arr0, exiting)):
                        in_window -= 1
                        elem0 = arr0[exiting]
                        if np.isnan(arr0[exiting]):
                            nan_counter -= 1
                        else:
                            total -= elem0
                exiting += 1
                entering += 1
                if 0 <= entering < n:
                    if not (bodo.libs.array_kernels.isna(arr0, entering)):
                        in_window += 1
                        elem0 = arr0[entering]
                        if np.isnan(arr0[entering]):
                            nan_counter += 1
                        else:
                            total += elem0
        return res
    """
    if empty_block is None:
        empty_block = "bodo.libs.array_kernels.setna(res, i)"

    nan_block = "res[i] = np.nan"

    if num_args not in (1, 2):
        raise_bodo_error(
            f"Unsupported number of arguments for sliding window kernel: {num_args}"
        )
    if num_args == 1:
        var_names = ["S"]
    else:
        var_names = ["Y", "X"]

    # [BE-4401] Optimize any_arr_is_null and any_arr_is_nan.

    # Define a function that takes in an index name and generates code to detect
    # whether any of the input arrays at that index is null
    def any_arr_is_null(idx):
        null_arg_terms = [
            f"bodo.libs.array_kernels.isna(arr{i}, {idx})" for i in range(num_args)
        ]
        return " or ".join(null_arg_terms)

    # Same as any_arr_is_null but for NaN
    def any_arr_is_nan(idx):
        nan_arg_terms = [f"np.isnan(arr{i}[{idx}])" for i in range(num_args)]
        return " or ".join(nan_arg_terms)

    # Modify enter_block so that if any of the inputs are NaN, this row is skipped
    # and the NaN counter is incremented
    if propagate_nan and enter_block is not None:
        new_enter_block = f"if {any_arr_is_nan('entering')}:\n"
        new_enter_block += "   nan_counter += 1\n"
        new_enter_block += "else:\n"
        new_enter_block += indent_block(enter_block, 3)
        enter_block = new_enter_block

    # Modify exit_block so that if any of the inputs are NaN, this row is skipped
    # and the NaN counter is decremented
    if propagate_nan and exit_block is not None:
        new_exit_block = f"if {any_arr_is_nan('exiting')}:\n"
        new_exit_block += "   nan_counter -= 1\n"
        new_exit_block += "else:\n"
        new_exit_block += indent_block(exit_block, 3)
        exit_block = new_exit_block

    # Modify calculate_block so that if any of the inputs are NaN, this row is
    # automatically NaN
    if propagate_nan:
        new_calculate_block = "if nan_counter > 0:\n"
        new_calculate_block += "   res[i] = np.nan\n"
        new_calculate_block += "else:\n"
        new_calculate_block += indent_block(calculate_block, 3)
        calculate_block = new_calculate_block

    # Declare the function and set up the variables based on how many arguments there are
    func_text = f"def impl({', '.join(var_names)}, lower_bound, upper_bound):\n"
    func_text += f"   n = len({var_names[0]})\n"
    for i in range(num_args):
        func_text += (
            f"   arr{i} = bodo.utils.conversion.coerce_to_array({var_names[i]})\n"
        )

    # Initialize the output array
    func_text += "   res = bodo.utils.utils.alloc_type(n, out_dtype, (-1,))\n"

    # If the bounds are invalid, fill everything with null
    func_text += "   if upper_bound < lower_bound:\n"
    func_text += "      for i in range(n):\n"
    func_text += "         bodo.libs.array_kernels.setna(res, i)\n"

    # If a constant block is provided, check to see if the bounds will the entire
    # window to have the same value
    if constant_block is not None:
        func_text += "   elif lower_bound <= -n+1 and n-1 <= upper_bound:\n"

        # Check to see if there are any non-null entries. If not, set the entire
        # output output array to NULL
        func_text += "      non_null_count = 0\n"
        if propagate_nan:
            func_text += "      has_nan = False\n"
        func_text += "      for i in range(n):\n"
        if propagate_nan:
            func_text += f"         if non_null_count >= {min_elements} and has_nan:\n"
            func_text += "            break\n"
        func_text += f"         if not ({any_arr_is_null('i')}):\n"
        func_text += "            non_null_count += 1\n"
        if propagate_nan:
            func_text += f"         if ({any_arr_is_nan('i')}):\n"
            func_text += "            has_nan = True\n"
            func_text += "      if has_nan:\n"
            func_text += "         for i in range(n):\n"
            func_text += indent_block(nan_block, 12)
            func_text += f"      elif non_null_count < {min_elements}:\n"
        else:
            func_text += f"      if non_null_count < {min_elements}:\n"
        func_text += "         for i in range(n):\n"
        func_text += indent_block(empty_block, 12)
        func_text += "      else:\n"
        func_text += indent_block(constant_block, 9)
        func_text += "         res[:] = constant_value\n"

    # Otherwise, set up the sliding window calculation
    func_text += "   else:\n"
    func_text += "      exiting = lower_bound\n"
    func_text += "      entering = upper_bound\n"
    func_text += "      in_window = 0\n"
    if propagate_nan:
        func_text += "      nan_counter = 0\n"
    func_text += indent_block(setup_block, 6)

    # Loop over all entries that have entered the window frame by the time that
    # we need to calculate the value for the first row and invoke the enter block
    func_text += (
        "      for i in range(min(max(0, exiting), n), min(max(0, entering + 1), n)):\n"
    )
    func_text += f"         if not ({any_arr_is_null('i')}):\n"
    func_text += "            in_window += 1\n"
    if enter_block is not None:
        if "elem" in enter_block:
            for i in range(num_args):
                func_text += f"            elem{i} = arr{i}[i]\n"
        func_text += indent_block(enter_block.replace("entering", "i"), 12)

    # Loop over the entire array. Anytime there are zero non-null entries, invoke
    # the empty block
    func_text += "      for i in range(n):\n"
    func_text += f"         if in_window < {min_elements}:\n"
    func_text += indent_block(empty_block, 12)

    # Otherwise, calculate the value of the current row based on the accumulated
    # values that have entered/exited up to this point
    func_text += "         else:\n"
    if "elem" in calculate_block:
        for i in range(num_args):
            func_text += f"            elem{i} = arr{i}[i]\n"
    func_text += indent_block(calculate_block, 12)

    # If the exiting index is in bounds and none of the input arrays are
    # null in the current row, invoke the exit block
    func_text += "         if 0 <= exiting < n:\n"
    func_text += f"            if not ({any_arr_is_null('exiting')}):\n"
    func_text += "               in_window -= 1\n"
    if exit_block is not None:
        if "elem" in exit_block:
            for i in range(num_args):
                func_text += f"               elem{i} = arr{i}[exiting]\n"
        func_text += indent_block(exit_block, 15)

    # Increment the entering and exiting indices
    func_text += "         exiting += 1\n"
    func_text += "         entering += 1\n"

    # If the entering index is in bounds and none of the input arrays are
    # null in the current row, invoke the enter block
    func_text += "         if 0 <= entering < n:\n"
    func_text += f"            if not ({any_arr_is_null('entering')}):\n"
    func_text += "               in_window += 1\n"
    if enter_block is not None:
        if "elem" in enter_block:
            for i in range(num_args):
                func_text += f"               elem{i} = arr{i}[entering]\n"
        func_text += indent_block(enter_block, 15)
    func_text += "   return res"
    loc_vars = {}
    if extra_globals is None:
        extra_globals = {}
    exec(
        func_text,
        {
            "bodo": bodo,
            "bodosql": bodosql,
            "numba": numba,
            "np": np,
            "out_dtype": out_dtype,
            "pd": pd,
            **extra_globals,
        },
        loc_vars,
    )
    impl = loc_vars["impl"]

    return impl


def make_slice_window_agg(
    out_dtype_fn, agg_func, min_elements=1, propagate_nan=True, extra_globals=None
):
    """
    Generates a kernel for a window function based on calculating a slice
    of the input for each window frame and then calling an aggregation on
    the slice.

    Args:
        out_dtype_fn [function]: a function that takes in the input type of
                                 the column and returns the datatype of the
                                 output array
        agg_func [function]: a function that takes in a string representing a
                             series of values and produces the corresponding
                             aggregate output

    The following arguments are also available for use to pass on to
    gen_windowed. For their full meaning, see the doscstring of gen_windowed:

        propagate_nan [boolean]
        extra_globals [dictionary, optional]
        min_elements [integer]
    """

    def impl(S, lower_bound, upper_bound):  # pragma: no cover
        calculate_block = (
            "slice = S.iloc[min(max(0, exiting), n):min(max(0, entering + 1), n)]\n"
        )
        calculate_block += f"res[i] = {agg_func('slice')}\n"
        constant_block = f"constant_value = {agg_func('S')}\n"
        out_dtype = out_dtype_fn(S)
        return gen_windowed(
            calculate_block,
            out_dtype,
            constant_block=constant_block,
            propagate_nan=propagate_nan,
            extra_globals=extra_globals,
            min_elements=min_elements,
        )

    return impl


def bit_agg_type_inference(arr):
    """Takes in the input to a bitXXX_agg window function and returns the corresponding
    output dtype. If the input is an integer type, the output has the same integer
    dtype. Otherwise, the output uses int64."""
    if isinstance(arr, SeriesType):
        arr = arr.data
    if isinstance(arr.dtype, types.Integer):
        out_dtype = arr.dtype
    else:
        out_dtype = types.int64
    return bodo.utils.typing.to_nullable_type(
        bodo.utils.typing.dtype_to_array_type(out_dtype)
    )


def check_insert_args(pos, len):  # pragma: no cover
    pass


@overload(check_insert_args)
def overload_check_insert_args(pos, len):
    def impl(pos, len):  # pragma: no cover
        assert pos >= 1, "<pos> argument must be at least 1!"
        assert len >= 0, "<len> argument must be at least 0!"

    return impl


def get_combined_type(in_types, calling_func):
    """
    Takes in a tuple of types representing various field types from a struct array
    and returns the common array type used when a lateral flatten operation
    combines all of the fields into one column.

    Args:
        in_types (tuple): the array types of the struct fields
        calling_func (string): what function is this utility being used for

    Returns:
        dtype: the array type used to store the combined values.

    Raise:
        BodoError: if the types are not able to be harmonized
    """
    if len(in_types) == 0:
        raise_bodo_error(f"{calling_func}: must have non-empty types tuple")

    seed_type = in_types[0]

    # If the first type is a string array or dictionary encoded array,
    # verify that all of the inputs are one of those two.
    if bodo.utils.typing.is_str_arr_type(seed_type):
        if not all(
            bodo.utils.typing.is_str_arr_type(typ) for typ in in_types
        ):  # pragma: no cover
            raise_bodo_error(f"{calling_func}: unsupported mix of types {in_types}")
        return bodo.types.string_array_type

    # If the first type is is an array item array, verify that all of
    # the other types are also array item arrays and then recursively
    # repeat the procedure on all of the inner types.
    if isinstance(seed_type, bodo.types.ArrayItemArrayType):
        if not all(
            isinstance(typ, bodo.types.ArrayItemArrayType) for typ in in_types
        ):  # pragma: no cover
            raise_bodo_error(f"{calling_func}: unsupported mix of types {in_types}")
        inner_dtypes = [typ.dtype for typ in in_types]
        combined_inner_dtype = get_combined_type(inner_dtypes, calling_func)
        return bodo.types.ArrayItemArrayType(combined_inner_dtype)

    # For map arrays, recursively ensure the two child arrays match
    if isinstance(seed_type, (bodo.types.MapArrayType, types.DictType)):
        if not all(
            isinstance(typ, (bodo.types.MapArrayType, types.DictType))
            for typ in in_types
        ):  # pragma: no cover
            raise_bodo_error(f"{calling_func}: unsupported mix of types {in_types}")
        key_types = []
        val_types = []
        for typ in in_types:
            if isinstance(typ, bodo.types.MapArrayType):
                key_types.append(typ.key_arr_type)
                val_types.append(typ.value_arr_type)
            else:
                key_types.append(bodo.utils.typing.dtype_to_array_type(typ.key_type))
                val_types.append(bodo.utils.typing.dtype_to_array_type(typ.value_type))
        combined_key_type = get_combined_type(key_types, calling_func)
        combined_val_type = get_combined_type(val_types, calling_func)
        return bodo.types.MapArrayType(combined_key_type, combined_val_type)

    # For struct arrays, match the names recursively ensure the all child arrays match
    if isinstance(
        seed_type, (bodo.types.StructArrayType, bodo.libs.struct_arr_ext.StructType)
    ):
        if not all(
            isinstance(
                typ, (bodo.types.StructArrayType, bodo.libs.struct_arr_ext.StructType)
            )
            and typ.names == seed_type.names
            for typ in in_types
        ):  # pragma: no cover
            raise_bodo_error(f"{calling_func}: unsupported mix of types {in_types}")
        data_types = []
        for i in range(len(seed_type.names)):
            field_types = []
            for typ in in_types:
                if isinstance(typ, bodo.types.StructArrayType):
                    field_types.append(typ.data[i])
                else:
                    field_types.append(
                        bodo.utils.typing.dtype_to_array_type(typ.data[i])
                    )
            data_types.append(get_combined_type(field_types, calling_func))
        return bodo.types.StructArrayType(tuple(data_types), seed_type.names)

    # For all other array types, get the common scalar type if possible
    dtypes = [typ.dtype for typ in in_types]
    common_dtype, _ = bodo.utils.typing.get_common_scalar_dtype(dtypes)
    if common_dtype is None:  # pragma: no cover
        raise_bodo_error(f"{calling_func}: unsupported mix of types {in_types}")

    # If any of the inputs are nullable, the output is also nullable
    common_dtype = bodo.utils.typing.dtype_to_array_type(common_dtype)
    if any(bodo.utils.typing.is_nullable(typ) for typ in in_types):
        common_dtype = bodo.utils.typing.to_nullable_type(common_dtype)

    return common_dtype
