import numba
import numpy as np
import pandas as pd

import bodo
from bodo.utils.typing import (
    ColNamesMetaType,
    raise_bodo_error,
)


def generate_simple_series_impl(
    arg_names,
    arg_types,
    out_type,
    scalar_text,
    preprocess_text=None,
    arg_defaults=None,
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
        preprocess_text (string): the func_text for additional logic before row-by-row computation.
        arg_defaults (dict): a mapping of argument names to default values if applicable.
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
    if arg_defaults is None:
        func_text = "def bodo_generate_simple_series(" + ", ".join(arg_names) + "):\n"
    else:
        arg_def_strings = [
            name if name not in arg_defaults else f"{name}={arg_defaults.get(name)}"
            for name in arg_names
        ]
        func_text = (
            "def bodo_generate_simple_series(" + ", ".join(arg_def_strings) + "):\n"
        )

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

    # Embed preprocess_text
    if preprocess_text is not None:
        for line in preprocess_text.splitlines():
            func_text += f" {line}\n"

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
            func_text += (
                " for i in numba.parfors.parfor.internal_prange(len(indices)):\n"
            )
            func_text += "  idx = indices[i]\n"
            func_text += "  if bodo.libs.array_kernels.isna(indices, i) or bodo.libs.array_kernels.isna(result, idx):\n"
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
    return bodo.utils.utils.bodo_exec(
        func_text,
        {
            "bodo": bodo,
            "numba": numba,
            "pandas": pd,
            "np": np,
            "out_dtype": out_arr_type,
        },
        {},
        __name__,
    )


def generate_series_to_df_impl(
    arg_names,
    arg_defaults,
    arg_types,
    out_names,
    out_types,
    scalar_text,
    keep_name=True,
    keep_index=True,
    iterate_over_dict=True,
    maintain_dict=True,
    modifies_nulls=False,
    may_create_duplicates=True,
):
    """A helper utility used to generate generic implementations of pandas APIs. This utility
       is specifically for pandas APIs where the input is Series-like data, the output is a
       DataFrame, and the output can be calculated at a row-by-row level. This utility also
       automatically contends with dictionary encoded optimizations

    Args:
        arg_names (Tuple[str]): the names of all of the inputs to the function being implemented. It
        is assumed that the first argument is the name of the Series-like input.
        arg_defaults(Tuple[str|None]): the default value strings for each of the arguments.
        arg_types (Tuple[type]): the types of all of the inputs to the function being implemented. It
        is assumed that the first input is Series-like data, and none of the others are as well, since
        that should be handled by a different code generation utility.
        out_names (Tuple[int|str]): the names of the returned DataFrame's columns.
        out_types (Tuple[type]): the types of each of the returned DataFrame's columns.
        scalar_text (string): the func_text for the computations at a row-by-row level.
        keep_name (bool): indicates that it should use the same name as the
        original input (if False, uses None).
        keep_index (bool): indicates that it should use the same index as the
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
    that the answer for columns 0, 1, 2... should be written to already allocated arrays named
    res0, res1, res2...
    """

    series_arg_name = arg_names[0]
    series_arg = arg_types[0]
    n_out = len(out_types)

    # Create the function definition line
    arg_strings = [
        name if default is None else f"{name}={default}"
        for name, default in zip(arg_names, arg_defaults)
    ]
    func_text = "def bodo_generate_series_to_df(" + ", ".join(arg_strings) + "):\n"

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
            f"generate_series_to_df_impl: unsupported input type {series_arg}"
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
    out_dicts = [typ == bodo.dict_str_arr_type for typ in out_types]
    out_any_dict = any(out_dicts) and maintain_dict
    dict_loop = (
        is_dict_input and iterate_over_dict and not (out_any_dict and modifies_nulls)
    )

    if dict_loop:
        if may_create_duplicates:
            func_text += " is_dict_unique = False\n"
        else:
            func_text += " is_dict_unique = data.is_dict_unique\n"
        func_text += " has_global = data._has_global_dictionary\n"
        func_text += " indices = data._indices\n"
        func_text += " data = data._data\n"

    # Allocate the output arrays and set up a loop that will write to them
    for i in range(n_out):
        func_text += (
            f" res{i} = bodo.utils.utils.alloc_type(len(data), out_dtype{i}, (-1,))\n"
        )
    func_text += " numba.parfors.parfor.init_prange()\n"
    func_text += " for i in numba.parfors.parfor.internal_prange(len(data)):\n"

    # Embed the scalar_text inside the loop
    for line in scalar_text.splitlines():
        func_text += f"  {line}\n"

    if dict_loop:
        for i in range(n_out):
            if out_dicts[i]:
                # If the output is also a dictionary encoded array, create the answer by
                # taking the result array and combining it with the original indices
                func_text += f" res{i} =  bodo.libs.dict_arr_ext.init_dict_arr(res{i}, indices, has_global, is_dict_unique, None)\n"

            else:
                # Otherwise, create the answer array by copying the values from the smaller
                # answer array based on the indices
                func_text += f" expanded_result{i} = bodo.utils.utils.alloc_type(len(indices), out_dtype{i}, (-1,))\n"
                func_text += " numba.parfors.parfor.init_prange()\n"
                func_text += (
                    " for i in numba.parfors.parfor.internal_prange(len(indices)):\n"
                )
                func_text += "  idx = indices[i]\n"
                func_text += f"  if bodo.libs.array_kernels.isna(res{i}, idx):\n"
                func_text += (
                    f"   bodo.libs.array_kernels.setna(expanded_result{i}, i)\n"
                )
                func_text += "  else:\n"
                func_text += f"   expanded_result{i}[i] = res{i}[idx]\n"
                func_text += f" res{i} = expanded_result{i}\n"

    # Wrap the collection of result arrays to create the DataFrame.
    if keep_name:
        func_text += f" name = {name_text}\n"
    else:
        func_text += " name = None\n"
    if keep_index:
        func_text += f" index = {index_text}\n"
    else:
        func_text += " index = bodo.hiframes.pd_index_ext.init_range_index(0, len(res0), 1, None)\n"
    func_text += f" return bodo.hiframes.pd_dataframe_ext.init_dataframe(({', '.join(f'res{i}' for i in range(n_out))},), index, __col_name_meta_value)\n"

    glbls = {
        "bodo": bodo,
        "numba": numba,
        "pandas": pd,
        "np": np,
        "__col_name_meta_value": ColNamesMetaType(out_names),
    }
    for i in range(n_out):
        glbls[f"out_dtype{i}"] = out_types[i]

    return bodo.utils.utils.bodo_exec(
        func_text,
        glbls,
        {},
        __name__,
    )
