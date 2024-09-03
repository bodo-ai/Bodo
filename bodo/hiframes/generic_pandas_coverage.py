# Copyright (C) 2024 Bodo Inc. All rights reserved.

import numba
import numpy as np
import pandas as pd

import bodo
from bodo.utils.typing import (
    raise_bodo_error,
)


def generate_simple_series_impl(
    arg_names,
    arg_types,
    out_type,
    scalar_text,
    keep_name=True,
    keep_index=True,
    iterate_over_dict=True,
    maintain_dict=True,
    modifies_nulls=False,
    may_create_duplicates=True,
):
    """A helper utility used to generate generic implementations of pandas APIs. This utility
       is specifically for pandas APIs where the input is Series-like data, the output is
       the same, and the output can be calculated at a row-by-row level. This utility also
       automatically contends with dictionary encoded optimizations

    Args:
        arg_names (Tuple[str]): the names of all of the inputs to the function being implemented. It
        is assumed that the first argument is the name of the Series-like input.
        arg_types (Tuple[types]): the types of all of the inputs to the function being implemented. It
        is assumed that the first input is Series-like data, and none of the others are as well, since
        that should be handled by a different code generation utility.
        out_type (type): the type that is to be returned by the function.
        scalar_text (string): the func_text for the computations at a row-by-row level.
        keep_name (bool): if returning a Series, indicates that it should use the same name as the
        original input (if False, uses None).
        keep_index (bool): if returning a Series, indicates that it should use the same index as the
        original input (if False, creates a new RangeIndex).
        iterate_over_dict (bool): indicates that the implementation should try to optimize when the
        first argument is a dictionary encoded array by looping over the dictionary instead of the
        entire array.
        maintain_dict (bool): indicates that if the optimization by iterate_over_dict is taken, the
        result should be returned as a dictionary encoded array using the same indices as the input.
        modifies_nulls (bool): indicates that the output could contain nulls in rows where the input
        did not have nulls (or vice versa).
        may_create_duplicates (bool): indicates that the output could contain duplicate strings even
        if the input did not have any.

    When writing scalar_text, assume that the data being iterated over is in an array called
    "data" (already extracted from the Series), that the iterator variable is called "i", and
    that the answer should be written to an already allocated array called "result".
    """

    series_arg_name = arg_names[0]
    series_arg = arg_types[0]

    # Create the function definition line
    func_text = "def impl(" + ", ".join(arg_names) + "):\n"

    # Extract the underlying array of the series as a variable called "data"
    if isinstance(series_arg, bodo.hiframes.pd_series_ext.SeriesType):
        func_text += (
            f" data = bodo.hiframes.pd_series_ext.get_series_data({series_arg_name})\n"
        )
        name_text = f"bodo.hiframes.pd_series_ext.get_series_name({series_arg_name})"
        index_text = f"bodo.hiframes.pd_series_ext.get_series_index({series_arg_name})"
    elif isinstance(series_arg, bodo.hiframes.series_str_impl.SeriesStrMethodType):
        func_text += f" data = bodo.hiframes.pd_series_ext.get_series_data({series_arg_name}._obj)\n"
        name_text = (
            f"bodo.hiframes.pd_series_ext.get_series_name({series_arg_name}._obj)"
        )
        index_text = (
            f"bodo.hiframes.pd_series_ext.get_series_index({series_arg_name}._obj)"
        )
    else:
        raise_bodo_error(
            f"generate_simple_series_impl: unsupported input type {series_arg}"
        )

    is_dict_input = (
        series_arg == bodo.dict_str_arr_type
        or (
            isinstance(series_arg, bodo.hiframes.pd_series_ext.SeriesType)
            and series_arg.data == bodo.dict_str_arr_type
        )
        or (
            isinstance(series_arg, bodo.hiframes.series_str_impl.SeriesStrMethodType)
            and series_arg.stype.data == bodo.dict_str_arr_type
        )
    )
    out_arr_type = (
        out_type.data
        if isinstance(out_type, bodo.hiframes.pd_series_ext.SeriesType)
        else out_type
    )
    out_dict = out_arr_type == bodo.dict_str_arr_type and maintain_dict
    dict_loop = (
        is_dict_input and iterate_over_dict and not (out_dict and modifies_nulls)
    )

    if dict_loop:
        if may_create_duplicates:
            func_text += " is_dict_unique = False\n"
        else:
            func_text += " is_dict_unique = data.is_dict_unique\n"
        func_text += " has_global = data._has_global_dictionary\n"
        func_text += " indices = data._indices\n"
        func_text += " data = data._data\n"

    # Allocate the output array and set up a loop that will write to it
    func_text += " result = bodo.utils.utils.alloc_type(len(data), out_dtype, (-1,))\n"
    func_text += " numba.parfors.parfor.init_prange()\n"
    func_text += " for i in numba.parfors.parfor.internal_prange(len(data)):\n"

    # Embed the scalar_text inside the loop
    for line in scalar_text.splitlines():
        func_text += f"  {line}\n"

    if dict_loop:
        if out_dict:
            # If the output is also a dictionary encoded array, create the answer by
            # taking the result array and combining it with the original indices
            func_text += " result =  bodo.libs.dict_arr_ext.init_dict_arr(result, indices, has_global, is_dict_unique, None)\n"

        else:
            # Otherwise, create the answer array by copying the values from the smaller
            # answer array based on the indices
            func_text += " expanded_result = bodo.utils.utils.alloc_type(len(indices), out_dtype, (-1,))\n"
            func_text += " numba.parfors.parfor.init_prange()\n"
            func_text += " for i in numba.parfors.parfor.internal_prange(len(data)):\n"
            func_text += "  idx = indices[i]\n"
            func_text += "  if bodo.libs.array_kernels.isna(result, idx):\n"
            func_text += "   bodo.libs.array_kernels.setna(expanded_result, i)\n"
            func_text += "  else:\n"
            func_text += "   expanded_result[i] = result[idx]\n"
            func_text += " result = expanded_result\n"

    # Create the logic that returns the final result based on the allocated result array.
    if bodo.utils.utils.is_array_typ(out_type, False):
        # If returning a regular array, then result is the answer.
        func_text += " return result\n"

    elif isinstance(out_type, bodo.hiframes.pd_series_ext.SeriesType):
        # If returning a Series, then wrap the result array to create the Series.
        if keep_name:
            func_text += f" name = {name_text}\n"
        else:
            func_text += " name = None\n"
        if keep_index:
            func_text += f" index = {index_text}\n"
        else:
            func_text += " index = bodo.hiframes.pd_index_ext.init_range_index(0, len(result), 1, None)\n"
        func_text += (
            " return bodo.hiframes.pd_series_ext.init_series(result, index, name)\n"
        )

    else:
        raise_bodo_error(
            f"generate_simple_series_impl: unsupported output type {out_type}"
        )

    loc_vars = {}
    exec(
        func_text,
        {
            "bodo": bodo,
            "numba": numba,
            "pandas": pd,
            "np": np,
            "out_dtype": out_arr_type,
        },
        loc_vars,
    )
    impl = loc_vars["impl"]
    return impl
