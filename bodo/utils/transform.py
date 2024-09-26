# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Helper functions for transformations.
"""
import itertools
import math
import operator
import typing as pt
from collections import namedtuple

import numba
import numpy as np
import pandas as pd
from numba.core import event, ir, ir_utils, types
from numba.core.ir_utils import (
    GuardException,
    build_definitions,
    compile_to_numba_ir,
    compute_cfg_from_blocks,
    find_callname,
    find_const,
    get_definition,
    guard,
    is_setitem,
    mk_unique_var,
    replace_arg_nodes,
    require,
)
from numba.core.registry import CPUDispatcher
from numba.core.typing.templates import fold_arguments

import bodo
from bodo.libs.array_item_arr_ext import ArrayItemArrayType
from bodo.libs.map_arr_ext import MapArrayType
from bodo.libs.str_arr_ext import string_array_type
from bodo.libs.struct_arr_ext import StructArrayType, StructType
from bodo.libs.tuple_arr_ext import TupleArrayType
from bodo.utils.typing import (
    BodoConstUpdatedError,
    BodoError,
    can_literalize_type,
    get_literal_value,
    get_overload_const_bool,
    get_overload_const_list,
    is_literal_type,
    is_overload_constant_bool,
    raise_bodo_error,
)
from bodo.utils.utils import is_array_typ, is_assign, is_call, is_expr

ReplaceFunc = namedtuple(
    "ReplaceFunc",
    [
        "func",
        "arg_types",
        "args",
        "glbls",
        "inline_bodo_calls",
        "run_full_pipeline",
        "pre_nodes",
    ],
)


# Bodo types that have parameters and need to be instantiated from class
bodo_types_with_params = {
    "ArrayItemArrayType",
    "CSRMatrixType",
    "CategoricalArrayType",
    "CategoricalIndexType",
    "DataFrameType",
    "DatetimeIndexType",
    "Decimal128Type",
    "DecimalArrayType",
    "IntegerArrayType",
    "FloatingArrayType",
    "IntervalArrayType",
    "IntervalIndexType",
    "List",
    "MapArrayType",
    "NumericIndexType",
    "PDCategoricalDtype",
    "PeriodIndexType",
    "RangeIndexType",
    "SeriesType",
    "StringIndexType",
    "BinaryIndexType",
    "StructArrayType",
    "TimedeltaIndexType",
    "TupleArrayType",
}


# list of list/set/dict function names that update the container inplace
container_update_method_names = (
    # dict
    "clear",
    "pop",
    "popitem",
    "update",
    # set
    "add",
    "difference_update",
    "discard",
    "intersection_update",
    "remove",
    "symmetric_difference_update",
    # list
    "append",
    "extend",
    "insert",
    "reverse",
    "sort",
)


no_side_effect_call_tuples = {
    # general python functions
    (int,),
    (list,),
    (set,),
    (dict,),
    (min,),
    (max,),
    (abs,),
    (len,),
    (bool,),
    (str,),
    ("ceil", math),
    # Pandas APIs
    ("Int32Dtype", pd),
    ("Int64Dtype", pd),
    ("Timestamp", pd),
    ("Week", "offsets", "tseries", pd),
    # Series
    ("init_series", "pd_series_ext", "hiframes", bodo),
    ("get_series_data", "pd_series_ext", "hiframes", bodo),
    ("get_series_index", "pd_series_ext", "hiframes", bodo),
    ("get_series_name", "pd_series_ext", "hiframes", bodo),
    ("get_index_data", "pd_index_ext", "hiframes", bodo),
    ("get_index_name", "pd_index_ext", "hiframes", bodo),
    # Index
    ("init_binary_str_index", "pd_index_ext", "hiframes", bodo),
    ("init_numeric_index", "pd_index_ext", "hiframes", bodo),
    ("init_categorical_index", "pd_index_ext", "hiframes", bodo),
    ("_dti_val_finalize", "pd_index_ext", "hiframes", bodo),
    ("init_datetime_index", "pd_index_ext", "hiframes", bodo),
    ("init_timedelta_index", "pd_index_ext", "hiframes", bodo),
    ("init_range_index", "pd_index_ext", "hiframes", bodo),
    ("init_heter_index", "pd_index_ext", "hiframes", bodo),
    # Int array
    ("get_int_arr_data", "int_arr_ext", "libs", bodo),
    ("get_int_arr_bitmap", "int_arr_ext", "libs", bodo),
    ("init_integer_array", "int_arr_ext", "libs", bodo),
    ("alloc_int_array", "int_arr_ext", "libs", bodo),
    # Float array
    ("init_float_array", "float_arr_ext", "libs", bodo),
    ("alloc_float_array", "float_arr_ext", "libs", bodo),
    # str array
    ("inplace_eq", "str_arr_ext", "libs", bodo),
    # bool array
    ("get_bool_arr_data", "bool_arr_ext", "libs", bodo),
    ("get_bool_arr_bitmap", "bool_arr_ext", "libs", bodo),
    ("init_bool_array", "bool_arr_ext", "libs", bodo),
    ("alloc_bool_array", "bool_arr_ext", "libs", bodo),
    ("alloc_false_bool_array", "bool_arr_ext", "libs", bodo),
    ("alloc_true_bool_array", "bool_arr_ext", "libs", bodo),
    # Datetime array
    ("datetime_date_arr_to_dt64_arr", "pd_timestamp_ext", "hiframes", bodo),
    # tz-aware array
    ("alloc_pd_datetime_array", "pd_datetime_arr_ext", "libs", bodo),
    ("init_datetime_array", "pd_datetime_arr_ext", "libs", bodo),
    # Both of these functions are set as global imports.
    (bodo.libs.bool_arr_ext.compute_or_body,),
    (bodo.libs.bool_arr_ext.compute_and_body,),
    (
        "alloc_datetime_date_array",
        "datetime_date_ext",
        "hiframes",
        bodo,
    ),
    (
        "alloc_datetime_timedelta_array",
        "datetime_timedelta_ext",
        "hiframes",
        bodo,
    ),
    ("cat_replace", "pd_categorical_ext", "hiframes", bodo),
    ("init_categorical_array", "pd_categorical_ext", "hiframes", bodo),
    ("alloc_categorical_array", "pd_categorical_ext", "hiframes", bodo),
    ("get_categorical_arr_codes", "pd_categorical_ext", "hiframes", bodo),
    ("_sum_handle_nan", "series_kernels", "hiframes", bodo),
    ("_box_cat_val", "series_kernels", "hiframes", bodo),
    ("_mean_handle_nan", "series_kernels", "hiframes", bodo),
    ("_var_handle_mincount", "series_kernels", "hiframes", bodo),
    ("_compute_var_nan_count_ddof", "series_kernels", "hiframes", bodo),
    ("_sem_handle_nan", "series_kernels", "hiframes", bodo),
    ("dist_return", "distributed_api", "libs", bodo),
    ("rep_return", "distributed_api", "libs", bodo),
    ("distributed_transpose", "distributed_api", "libs", bodo),
    # DataFrame
    ("init_dataframe", "pd_dataframe_ext", "hiframes", bodo),
    ("get_dataframe_data", "pd_dataframe_ext", "hiframes", bodo),
    ("get_dataframe_all_data", "pd_dataframe_ext", "hiframes", bodo),
    ("get_dataframe_table", "pd_dataframe_ext", "hiframes", bodo),
    ("get_dataframe_column_names", "pd_dataframe_ext", "hiframes", bodo),
    ("get_table_data", "table", "hiframes", bodo),
    ("get_dataframe_index", "pd_dataframe_ext", "hiframes", bodo),
    ("init_rolling", "pd_rolling_ext", "hiframes", bodo),
    ("init_groupby", "pd_groupby_ext", "hiframes", bodo),
    # array kernels
    ("calc_nitems", "array_kernels", "libs", bodo),
    ("concat", "array_kernels", "libs", bodo),
    ("unique", "array_kernels", "libs", bodo),
    ("nunique", "array_kernels", "libs", bodo),
    ("quantile", "array_kernels", "libs", bodo),
    ("explode", "array_kernels", "libs", bodo),
    ("explode_no_index", "array_kernels", "libs", bodo),
    ("get_arr_lens", "array_kernels", "libs", bodo),
    ("str_arr_from_sequence", "str_arr_ext", "libs", bodo),
    ("get_str_arr_str_length", "str_arr_ext", "libs", bodo),
    ("parse_datetime_str", "pd_timestamp_ext", "hiframes", bodo),
    ("integer_to_dt64", "pd_timestamp_ext", "hiframes", bodo),
    ("dt64_to_integer", "pd_timestamp_ext", "hiframes", bodo),
    ("timedelta64_to_integer", "pd_timestamp_ext", "hiframes", bodo),
    ("integer_to_timedelta64", "pd_timestamp_ext", "hiframes", bodo),
    ("npy_datetimestruct_to_datetime", "pd_timestamp_ext", "hiframes", bodo),
    ("isna", "array_kernels", "libs", bodo),
    (bodo.libs.str_arr_ext.num_total_chars,),
    ("num_total_chars", "str_arr_ext", "libs", bodo),
    # TODO: handle copy properly, copy of some types can have side effects?
    ("copy",),
    ("from_iterable_impl", "typing", "utils", bodo),
    ("chain", itertools),
    ("groupby",),
    ("rolling",),
    (pd.CategoricalDtype,),
    (bodo.hiframes.pd_categorical_ext.get_code_for_value,),
    # Numpy
    ("asarray", np),
    ("int32", np),
    ("int64", np),
    ("float64", np),
    ("float32", np),
    ("bool_", np),
    ("full", np),
    ("round", np),
    ("isnan", np),
    ("isnat", np),
    ("arange", np),
    # Numba
    ("internal_prange", "parfor", numba),
    ("internal_prange", "parfor", "parfors", numba),
    ("empty_inferred", "ndarray", "unsafe", numba),
    ("_slice_span", "unicode", numba),
    ("_normalize_slice", "unicode", numba),
    # pyspark
    ("init_session_builder", "pyspark_ext", "libs", bodo),
    ("init_session", "pyspark_ext", "libs", bodo),
    ("init_spark_df", "pyspark_ext", "libs", bodo),
    # hdf5
    ("h5size", "h5_api", "io", bodo),
    ("pre_alloc_struct_array", "struct_arr_ext", "libs", bodo),
    (bodo.libs.struct_arr_ext.pre_alloc_struct_array,),
    ("pre_alloc_tuple_array", "tuple_arr_ext", "libs", bodo),
    (bodo.libs.tuple_arr_ext.pre_alloc_tuple_array,),
    ("pre_alloc_array_item_array", "array_item_arr_ext", "libs", bodo),
    (bodo.libs.array_item_arr_ext.pre_alloc_array_item_array,),
    ("dist_reduce", "distributed_api", "libs", bodo),
    (bodo.libs.distributed_api.dist_reduce,),
    ("get_chunk_bounds", "distributed_api", "libs", bodo),
    (bodo.libs.distributed_api.get_chunk_bounds,),
    ("pre_alloc_string_array", "str_arr_ext", "libs", bodo),
    (bodo.libs.str_arr_ext.pre_alloc_string_array,),
    ("pre_alloc_binary_array", "binary_arr_ext", "libs", bodo),
    (bodo.libs.binary_arr_ext.pre_alloc_binary_array,),
    ("pre_alloc_map_array", "map_arr_ext", "libs", bodo),
    (bodo.libs.map_arr_ext.pre_alloc_map_array,),
    # dict array
    ("convert_dict_arr_to_int", "dict_arr_ext", "libs", bodo),
    ("cat_dict_str", "dict_arr_ext", "libs", bodo),
    ("str_replace", "dict_arr_ext", "libs", bodo),
    ("dict_arr_to_numeric", "dict_arr_ext", "libs", bodo),
    ("dict_arr_eq", "dict_arr_ext", "libs", bodo),
    ("dict_arr_ne", "dict_arr_ext", "libs", bodo),
    ("str_startswith", "dict_arr_ext", "libs", bodo),
    ("str_endswith", "dict_arr_ext", "libs", bodo),
    ("str_contains_non_regex", "dict_arr_ext", "libs", bodo),
    ("str_series_contains_regex", "dict_arr_ext", "libs", bodo),
    ("str_capitalize", "dict_arr_ext", "libs", bodo),
    ("str_lower", "dict_arr_ext", "libs", bodo),
    ("str_swapcase", "dict_arr_ext", "libs", bodo),
    ("str_title", "dict_arr_ext", "libs", bodo),
    ("str_upper", "dict_arr_ext", "libs", bodo),
    ("str_center", "dict_arr_ext", "libs", bodo),
    ("str_get", "dict_arr_ext", "libs", bodo),
    ("str_repeat_int", "dict_arr_ext", "libs", bodo),
    ("str_lstrip", "dict_arr_ext", "libs", bodo),
    ("str_rstrip", "dict_arr_ext", "libs", bodo),
    ("str_strip", "dict_arr_ext", "libs", bodo),
    ("str_zfill", "dict_arr_ext", "libs", bodo),
    ("str_ljust", "dict_arr_ext", "libs", bodo),
    ("str_rjust", "dict_arr_ext", "libs", bodo),
    ("str_find", "dict_arr_ext", "libs", bodo),
    ("str_rfind", "dict_arr_ext", "libs", bodo),
    ("str_index", "dict_arr_ext", "libs", bodo),
    ("str_rindex", "dict_arr_ext", "libs", bodo),
    ("str_slice", "dict_arr_ext", "libs", bodo),
    ("str_extract", "dict_arr_ext", "libs", bodo),
    ("str_extractall", "dict_arr_ext", "libs", bodo),
    ("str_extractall_multi", "dict_arr_ext", "libs", bodo),
    ("str_len", "dict_arr_ext", "libs", bodo),
    ("str_count", "dict_arr_ext", "libs", bodo),
    ("str_isalnum", "dict_arr_ext", "libs", bodo),
    ("str_isalpha", "dict_arr_ext", "libs", bodo),
    ("str_isdigit", "dict_arr_ext", "libs", bodo),
    ("str_isspace", "dict_arr_ext", "libs", bodo),
    ("str_islower", "dict_arr_ext", "libs", bodo),
    ("str_isupper", "dict_arr_ext", "libs", bodo),
    ("str_istitle", "dict_arr_ext", "libs", bodo),
    ("str_isnumeric", "dict_arr_ext", "libs", bodo),
    ("str_isdecimal", "dict_arr_ext", "libs", bodo),
    ("str_match", "dict_arr_ext", "libs", bodo),
    ("prange", bodo),
    (bodo.prange,),
    ("objmode", bodo),
    (bodo.objmode,),
    ("no_warning_objmode", bodo),
    (bodo.no_warning_objmode,),
    # Helper functions, inlined in astype
    ("get_label_dict_from_categories", "pd_categorial_ext", "hiframes", bodo),
    (
        "get_label_dict_from_categories_no_duplicates",
        "pd_categorial_ext",
        "hiframes",
        bodo,
    ),
    # Nullable Tuple
    ("build_nullable_tuple", "nullable_tuple_ext", "libs", bodo),
    # Table
    ("generate_mappable_table_func", "table_utils", "utils", bodo),
    ("table_astype", "table_utils", "utils", bodo),
    ("table_concat", "table_utils", "utils", bodo),
    ("concat_tables", "table_utils", "utils", bodo),
    ("table_filter", "table", "hiframes", bodo),
    ("table_local_filter", "table", "hiframes", bodo),
    ("table_subset", "table", "hiframes", bodo),
    ("local_len", "table", "hiframes", bodo),
    ("logical_table_to_table", "table", "hiframes", bodo),
    ("set_table_data", "table", "hiframes", bodo),
    ("set_table_null", "table", "hiframes", bodo),
    ("create_empty_table", "table", "hiframes", bodo),
    # Series.str/string
    ("startswith",),
    ("endswith",),
    ("upper",),
    ("lower",),
    # BodoSQL
    ("__bodosql_replace_columns_dummy", "dataframe_impl", "hiframes", bodo),
    # Indexing
    ("scalar_optional_getitem", "indexing", "utils", bodo),
    ("bitmap_size", "indexing", "utils", bodo),
    ("get_dt64_bitmap_fill", "indexing", "utils", bodo),
    # Streaming state init functions
    ("init_join_state", "stream_join", "libs", bodo),
    ("init_groupby_state", "stream_groupby", "libs", bodo),
    ("init_grouping_sets_state", "stream_groupby", "libs", bodo),
    ("init_union_state", "stream_union", "libs", bodo),
    ("init_window_state", "stream_window", "libs", bodo),
    ("init_stream_sort_state", "stream_sort", "libs", bodo),
    ("init_dict_encoding_state", "stream_dict_encoding", "libs", bodo),
    ("init_table_builder_state", "table_builder", "libs", bodo),
    ("iceberg_writer_init", "stream_iceberg_write", "io", bodo),
    ("snowflake_writer_init", "snowflake_write", "io", bodo),
    # Datetime utils
    # TODO(njriasan): Move all "pure" datetime_date_ext functions
    # to the same file so can have file level DCE.
    ("now_date", "datetime_date_ext", "hiframes", bodo),
    ("now_date_wrapper", "datetime_date_ext", "hiframes", bodo),
    ("now_date_wrapper_consistent", "datetime_date_ext", "hiframes", bodo),
    ("today_rank_consistent", "datetime_date_ext", "hiframes", bodo),
    ("now_impl_consistent", "pd_timestamp_ext", "hiframes", bodo),
    # Filter Expression
    ("make_scalar", "filter", "ir", bodo),
    ("make_ref", "filter", "ir", bodo),
    ("make_op", "filter", "ir", bodo),
}


# Numpy type names with associated constructor calls like np.int32()
_np_type_names = {
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
    "bool_",
}


def remove_hiframes(rhs, lives, call_list):
    call_tuple = tuple(call_list)
    if call_tuple in no_side_effect_call_tuples:
        return True
    # Note we cannot import bodo.hiframes.pd_index_ext globally,
    # which is why this isn't in no_side_effect_call_tuples.
    # This needs to be refactored to have 1 single representation.
    if call_tuple == (bodo.hiframes.pd_index_ext.init_range_index,):
        return True

    if len(call_list) == 4 and call_list[1:] == [
        "conversion",
        "utils",
        bodo,
    ]:  # pragma: no cover
        # all conversion functions are side effect-free
        return True

    # Check for all bodosql array kernels.
    if call_list[1:] == ["bodosql_array_kernels", "libs", bodo]:  # pragma: no cover
        # all bodosql array kernels are side effect-free
        return True

    # TODO: handle copy() of the relevant types properly
    if len(call_list) == 2 and call_list[0] == "copy":
        return True

    # the call is dead if the read array is dead
    # TODO: return array from call to avoid using lives
    if call_list == ["h5read", "h5_api", "io", bodo] and rhs.args[5].name not in lives:
        return True

    if (
        call_list == ["move_str_binary_arr_payload", "str_arr_ext", "libs", bodo]
        and rhs.args[0].name not in lives
    ):
        return True

    # the call is dead if the updated array is dead
    if (
        call_list
        in (
            ["setna", "array_kernels", "libs", bodo],
            ["copy_array_element", "array_kernels", "libs", bodo],
            ["get_str_arr_item_copy", "str_arr_ext", "libs", bodo],
        )
        and rhs.args[0].name not in lives
    ):
        return True

    if (
        call_list == ["ensure_column_unboxed", "table", "hiframes", bodo]
        and rhs.args[0].name not in lives
        and rhs.args[1].name not in lives
    ):
        return True

    if (
        call_list == ["generate_table_nbytes", "table_utils", "utils", bodo]
        and rhs.args[1].name not in lives
    ):
        # Arg1 is the output.
        return True

    # constructor calls of tuple subclasses like namedtuple don't have side-effect
    # e.g. Row(a, b) in UDFs
    if len(call_tuple) == 1 and tuple in getattr(call_tuple[0], "__mro__", ()):
        return True

    # Note: Numba will try the other call handlers if this returns False
    return False


# Note: To register additional handling for removing unused functions
# you can append to numba.core.ir_utils.remove_call_handlers.
numba.core.ir_utils.remove_call_handlers.append(remove_hiframes)


def compile_func_single_block(
    func: pt.Callable,
    args,
    ret_var: ir.Var,
    typing_info=None,
    extra_globals: pt.Optional[dict[str, pt.Any]] = None,
    infer_types: bool = True,
    run_untyped_pass: bool = False,
    flags=None,
    replace_globals: bool = False,
    add_default_globals: bool = True,
) -> list[ir.Stmt]:
    """compiles functions that are just a single basic block.
    Does not handle defaults, freevars etc.
    typing_info is a structure that has typingctx, typemap, calltypes
    (could be the pass itself since not mutated).
    """
    # TODO: support recursive processing of compile function if necessary
    if replace_globals:
        glbls = {"numba": numba, "np": np, "bodo": bodo, "pd": pd, "math": math}
    else:
        glbls = func.__globals__
    if extra_globals is not None:
        glbls.update(extra_globals)
    if add_default_globals:
        glbls.update(
            {
                "numba": numba,
                "np": np,
                "bodo": bodo,
                "pd": pd,
                "math": math,
            }
        )
    loc = ir.Loc("", 0)
    if ret_var:
        loc = ret_var.loc
    if typing_info and infer_types:
        loc = typing_info.curr_loc
        f_ir = compile_to_numba_ir(
            func,
            glbls,
            typingctx=typing_info.typingctx,
            targetctx=typing_info.targetctx,
            arg_typs=tuple(typing_info.typemap[arg.name] for arg in args),
            typemap=typing_info.typemap,
            calltypes=typing_info.calltypes,
        )
    else:
        f_ir = compile_to_numba_ir(func, glbls)
    assert (
        len(f_ir.blocks) == 1
    ), "only single block functions supported in compile_func_single_block()"
    if run_untyped_pass:
        arg_typs = tuple(typing_info.typemap[arg.name] for arg in args)
        untyped_pass = bodo.transforms.untyped_pass.UntypedPass(
            f_ir, typing_info.typingctx, arg_typs, {}, {}, flags
        )
        untyped_pass.run()
    f_block = f_ir.blocks.popitem()[1]
    replace_arg_nodes(f_block, args)
    nodes = f_block.body[:-2]

    # update Loc objects, avoid changing input arg vars
    update_locs(nodes[len(args) :], loc)
    for stmt in nodes[: len(args)]:
        stmt.target.loc = loc

    if ret_var is not None:
        cast_assign = f_block.body[-2]
        assert is_assign(cast_assign) and is_expr(cast_assign.value, "cast")
        func_ret = cast_assign.value.value
        nodes.append(ir.Assign(func_ret, ret_var, loc))

    return nodes


def update_locs(node_list, loc):
    """Update Loc objects for list of generated statements"""
    for stmt in node_list:
        stmt.loc = loc
        for v in stmt.list_vars():
            v.loc = loc
        if is_assign(stmt):
            stmt.value.loc = loc


def get_stmt_defs(stmt):
    if is_assign(stmt):
        return set([stmt.target.name])

    if type(stmt) in numba.core.analysis.ir_extension_usedefs:
        def_func = numba.core.analysis.ir_extension_usedefs[type(stmt)]
        _uses, defs = def_func(stmt)
        return defs

    return set()


def get_const_value(
    var,
    func_ir,
    err_msg,
    typemap=None,
    arg_types=None,
    file_info=None,
):
    """Get constant value of a variable if possible, otherwise raise error.
    If the variable is argument to the function, force recompilation with literal
    typing of the argument.
    """
    if hasattr(var, "loc"):
        loc = var.loc
    else:
        loc = None
    try:
        val = get_const_value_inner(
            func_ir, var, arg_types, typemap, file_info=file_info
        )
        if isinstance(val, ir.UndefinedType):
            name = func_ir.get_definition(var.name).name
            raise BodoError(f"name '{name}' is not defined", loc=loc)
    except GuardException:
        raise BodoError(err_msg, loc=loc)
    return val


def get_const_value_inner(
    func_ir,
    var,
    arg_types=None,
    typemap=None,
    updated_containers=None,
    file_info=None,
    pyobject_to_literal=False,
    literalize_args=True,
):
    """Check if a variable can be inferred as a constant and return the constant value.
    Otherwise, raise GuardException.
    """
    require(isinstance(var, ir.Var))
    var_def = get_definition(func_ir, var)

    # get type of variable if possible
    typ = None
    if typemap is not None:
        typ = typemap.get(var.name, None)
    if isinstance(var_def, ir.Arg) and arg_types is not None:
        typ = arg_types[var_def.index]

    # avoid updated containers like a = []; a.append("A")
    if updated_containers and var.name in updated_containers:
        raise BodoConstUpdatedError(
            f"variable '{var.name}' is updated inplace using '{updated_containers[var.name]}'"
        )

    # literal type case
    if is_literal_type(typ):
        return get_literal_value(typ)

    # constant value
    if isinstance(var_def, (ir.Const, ir.Global, ir.FreeVar)):
        val = var_def.value
        return val

    # argument dispatch, force literal only if argument can be literal
    if (
        literalize_args
        and isinstance(var_def, ir.Arg)
        and can_literalize_type(typ, pyobject_to_literal)
    ):
        raise numba.core.errors.ForceLiteralArg(
            {var_def.index},
            loc=var.loc,
            file_infos={var_def.index: file_info} if file_info is not None else None,
        )

    # binary op (s1 op s2)
    if is_expr(var_def, "binop"):
        # Embed string concat op inside file_info to recompute file name during load
        # from cache (to get new schema and compare to previous schema)
        # see [BE-690]
        if file_info and var_def.fn == operator.add:
            # at least one side should be a constant value not coming from an argument
            try:
                arg1 = get_const_value_inner(
                    func_ir,
                    var_def.lhs,
                    arg_types,
                    typemap,
                    updated_containers,
                    literalize_args=False,
                )
                file_info.set_concat(arg1, True)
                arg2 = get_const_value_inner(
                    func_ir,
                    var_def.rhs,
                    arg_types,
                    typemap,
                    updated_containers,
                    file_info,
                )
                return var_def.fn(arg1, arg2)
            except (GuardException, BodoConstUpdatedError):
                pass

            try:
                arg2 = get_const_value_inner(
                    func_ir,
                    var_def.rhs,
                    arg_types,
                    typemap,
                    updated_containers,
                    literalize_args=False,
                )
                file_info.set_concat(arg2, False)
                arg1 = get_const_value_inner(
                    func_ir,
                    var_def.lhs,
                    arg_types,
                    typemap,
                    updated_containers,
                    file_info,
                )
                return var_def.fn(arg1, arg2)
            except (GuardException, BodoConstUpdatedError):
                pass
            # TODO(ehsan): raise a warning if caching isn't possible here?

        arg1 = get_const_value_inner(
            func_ir, var_def.lhs, arg_types, typemap, updated_containers
        )
        arg2 = get_const_value_inner(
            func_ir, var_def.rhs, arg_types, typemap, updated_containers
        )
        return var_def.fn(arg1, arg2)

    # unary op (op s1)
    if is_expr(var_def, "unary"):
        val = get_const_value_inner(
            func_ir, var_def.value, arg_types, typemap, updated_containers
        )
        return var_def.fn(val)

    # df.columns case
    if is_expr(var_def, "getattr") and typemap:
        obj_typ = typemap.get(var_def.value.name, None)
        if (
            isinstance(obj_typ, bodo.hiframes.pd_dataframe_ext.DataFrameType)
            and var_def.attr == "columns"
        ):
            # pandas columns are Index objects (accurate object is needed for const
            # computations, see test_loc_col_select::impl3)
            return pd.Index(obj_typ.columns)
        # start/stop/step of slice
        if isinstance(obj_typ, types.SliceType):
            slice_def = get_definition(func_ir, var_def.value)
            require(is_call(slice_def))
            slice_callname = find_callname(func_ir, slice_def)

            # _normalize_slice may change slice attributes if negative or out-of-bounds
            # we can only return start = 0 or step = 1 if normalizes
            check_normalize = False
            if slice_callname == ("_normalize_slice", "numba.cpython.unicode"):
                require(var_def.attr in ("start", "step"))
                slice_def = get_definition(func_ir, slice_def.args[0])
                check_normalize = True

            require(find_callname(func_ir, slice_def) == ("slice", "builtins"))
            # slice(stop) call has start = 0 and step = 1 by default
            if len(slice_def.args) == 1:
                if var_def.attr == "start":
                    return 0
                if var_def.attr == "step":
                    return 1
                require(var_def.attr == "stop")
                return get_const_value_inner(
                    func_ir, slice_def.args[0], arg_types, typemap, updated_containers
                )

            # slice(start, stop[, step]) case
            if var_def.attr == "start":
                val = get_const_value_inner(
                    func_ir, slice_def.args[0], arg_types, typemap, updated_containers
                )
                if val is None:
                    val = 0
                if check_normalize:
                    require(val == 0)
                return val
            if var_def.attr == "stop":
                assert not check_normalize
                return get_const_value_inner(
                    func_ir, slice_def.args[1], arg_types, typemap, updated_containers
                )
            require(var_def.attr == "step")
            # step is 1 by default if not provided
            if len(slice_def.args) == 2:
                return 1
            else:
                val = get_const_value_inner(
                    func_ir, slice_def.args[2], arg_types, typemap, updated_containers
                )
                if val is None:
                    val = 1
                if check_normalize:
                    require(val == 1)
                return val

    if is_expr(var_def, "getattr"):
        return getattr(
            get_const_value_inner(
                func_ir, var_def.value, arg_types, typemap, updated_containers
            ),
            var_def.attr,
        )

    if is_expr(var_def, "getitem"):
        value = get_const_value_inner(
            func_ir, var_def.value, arg_types, typemap, updated_containers
        )
        index = get_const_value_inner(
            func_ir, var_def.index, arg_types, typemap, updated_containers
        )
        return value[index]

    # list/set/dict cases

    # try dict.keys()
    call_name = guard(find_callname, func_ir, var_def, typemap)
    if (
        call_name is not None
        and len(call_name) == 2
        and call_name[0] == "keys"
        and isinstance(call_name[1], ir.Var)
    ):
        call_func = var_def.func
        var_def = get_definition(func_ir, call_name[1])
        dict_varname = call_name[1].name
        if updated_containers and dict_varname in updated_containers:
            raise BodoConstUpdatedError(
                "variable '{}' is updated inplace using '{}'".format(
                    dict_varname, updated_containers[dict_varname]
                )
            )
        require(is_expr(var_def, "build_map"))
        vals = [v[0] for v in var_def.items]
        # HACK replace dict.keys getattr to avoid typing errors
        keys_getattr = guard(get_definition, func_ir, call_func)
        assert isinstance(keys_getattr, ir.Expr) and keys_getattr.attr == "keys"
        keys_getattr.attr = "copy"
        return [
            get_const_value_inner(func_ir, v, arg_types, typemap, updated_containers)
            for v in vals
        ]

    # dict case
    if is_expr(var_def, "build_map"):
        return {
            get_const_value_inner(
                func_ir, v[0], arg_types, typemap, updated_containers
            ): get_const_value_inner(
                func_ir, v[1], arg_types, typemap, updated_containers
            )
            for v in var_def.items
        }

    # tuple case
    if is_expr(var_def, "build_tuple"):
        return tuple(
            get_const_value_inner(func_ir, v, arg_types, typemap, updated_containers)
            for v in var_def.items
        )

    # list
    if is_expr(var_def, "build_list"):
        return [
            get_const_value_inner(func_ir, v, arg_types, typemap, updated_containers)
            for v in var_def.items
        ]

    # set
    if is_expr(var_def, "build_set"):
        return {
            get_const_value_inner(func_ir, v, arg_types, typemap, updated_containers)
            for v in var_def.items
        }

    # list() call
    if call_name == ("list", "builtins"):
        values = get_const_value_inner(
            func_ir, var_def.args[0], arg_types, typemap, updated_containers
        )
        # sort set values when converting to list to have consistent order across
        # processors (e.g. important for join keys, see test_merge_multi_int_key)
        if isinstance(values, set):
            values = sorted(values)
        return list(values)

    # set() call
    if call_name == ("set", "builtins"):
        return set(
            get_const_value_inner(
                func_ir, var_def.args[0], arg_types, typemap, updated_containers
            )
        )

    # range() call
    if call_name == ("range", "builtins") and len(var_def.args) == 1:
        return range(
            get_const_value_inner(
                func_ir, var_def.args[0], arg_types, typemap, updated_containers
            )
        )

    # slice() call
    if call_name == ("slice", "builtins"):
        return slice(
            *tuple(
                get_const_value_inner(
                    func_ir, v, arg_types, typemap, updated_containers
                )
                for v in var_def.args
            )
        )

    # str() call
    if call_name == ("str", "builtins"):
        return str(
            get_const_value_inner(
                func_ir, var_def.args[0], arg_types, typemap, updated_containers
            )
        )

    # bool() call
    if call_name == ("bool", "builtins"):
        return bool(
            get_const_value_inner(
                func_ir, var_def.args[0], arg_types, typemap, updated_containers
            )
        )

    # format() call
    if call_name == ("format", "builtins"):
        arg = get_const_value_inner(
            func_ir, var_def.args[0], arg_types, typemap, updated_containers
        )
        fmt_spec = (
            get_const_value_inner(
                func_ir, var_def.args[1], arg_types, typemap, updated_containers
            )
            if len(var_def.args) > 1
            else ""
        )
        return format(arg, fmt_spec)

    # Index calls
    if call_name in (
        ("init_binary_str_index", "bodo.hiframes.pd_index_ext"),
        ("init_numeric_index", "bodo.hiframes.pd_index_ext"),
        ("init_categorical_index", "bodo.hiframes.pd_index_ext"),
        ("init_datetime_index", "bodo.hiframes.pd_index_ext"),
        ("init_timedelta_index", "bodo.hiframes.pd_index_ext"),
        ("init_heter_index", "bodo.hiframes.pd_index_ext"),
    ):
        return pd.Index(
            get_const_value_inner(
                func_ir, var_def.args[0], arg_types, typemap, updated_containers
            )
        )

    if call_name == ("str_arr_from_sequence", "bodo.libs.str_arr_ext"):
        return np.array(
            get_const_value_inner(
                func_ir, var_def.args[0], arg_types, typemap, updated_containers
            )
        )

    if call_name == ("init_range_index", "bodo.hiframes.pd_index_ext"):
        return pd.RangeIndex(
            get_const_value_inner(
                func_ir, var_def.args[0], arg_types, typemap, updated_containers
            ),
            get_const_value_inner(
                func_ir, var_def.args[1], arg_types, typemap, updated_containers
            ),
            get_const_value_inner(
                func_ir, var_def.args[2], arg_types, typemap, updated_containers
            ),
        )

    # len(tuple)
    if (
        call_name == ("len", "builtins")
        and typemap
        and isinstance(typemap.get(var_def.args[0].name, None), types.BaseTuple)
    ):
        return len(typemap[var_def.args[0].name])

    # len(data)
    if call_name == ("len", "builtins"):
        # data may not be all constant, but it's length may be fixed (e.g. build_list)
        arg_def = guard(get_definition, func_ir, var_def.args[0])
        if isinstance(arg_def, ir.Expr) and arg_def.op in (
            "build_tuple",
            "build_list",
            "build_set",
            "build_map",
        ):
            return len(arg_def.items)
        return len(
            get_const_value_inner(
                func_ir, var_def.args[0], arg_types, typemap, updated_containers
            )
        )

    # pd.CategoricalDtype() calls
    if call_name == ("CategoricalDtype", "pandas"):
        kws = dict(var_def.kws)
        cats = get_call_expr_arg(
            "CategoricalDtype", var_def.args, kws, 0, "categories", ""
        )
        ordered = get_call_expr_arg(
            "CategoricalDtype", var_def.args, kws, 1, "ordered", False
        )
        if ordered is not False:
            ordered = get_const_value_inner(
                func_ir, ordered, arg_types, typemap, updated_containers
            )
        if cats == "":
            cats = None
        else:
            cats = get_const_value_inner(
                func_ir, cats, arg_types, typemap, updated_containers
            )
        return pd.CategoricalDtype(cats, ordered)

    # np.dtype() calls
    if call_name == ("dtype", "numpy"):
        return np.dtype(
            get_const_value_inner(
                func_ir, var_def.args[0], arg_types, typemap, updated_containers
            )
        )

    # np.int32(), ... calls (commonly generated by BodoSQL)
    if (
        call_name is not None
        and call_name[1] == "numpy"
        and call_name[0] in _np_type_names
    ):
        return getattr(np, call_name[0])(
            get_const_value_inner(
                func_ir, var_def.args[0], arg_types, typemap, updated_containers
            )
        )

    # pd.Int64Dtype(), ...
    if (
        call_name is not None
        and len(call_name) == 2
        and call_name[1] == "pandas"
        and call_name[0]
        in (
            "Int8Dtype",
            "Int16Dtype",
            "Int32Dtype",
            "Int64Dtype",
            "UInt8Dtype",
            "UInt16Dtype",
            "UInt32Dtype",
            "UInt64Dtype",
        )
    ):
        return getattr(pd, call_name[0])()

    # method call case: val.method(a, b, ...)
    if (
        call_name is not None
        and len(call_name) == 2
        and isinstance(call_name[1], ir.Var)
    ):
        val = get_const_value_inner(
            func_ir, call_name[1], arg_types, typemap, updated_containers
        )
        args = [
            get_const_value_inner(func_ir, v, arg_types, typemap, updated_containers)
            for v in var_def.args
        ]
        kws = {
            a[0]: get_const_value_inner(
                func_ir, a[1], arg_types, typemap, updated_containers
            )
            for a in var_def.kws
        }
        return getattr(val, call_name[0])(*args, **kws)

    # bodo data type calls like bodo.DataFrameType()
    if (
        call_name is not None
        and len(call_name) == 2
        and call_name[1] == "bodo"
        and call_name[0] in bodo_types_with_params
    ):
        args = tuple(
            get_const_value_inner(func_ir, v, arg_types, typemap, updated_containers)
            for v in var_def.args
        )
        kwargs = {
            name: get_const_value_inner(
                func_ir, v, arg_types, typemap, updated_containers
            )
            for name, v in dict(var_def.kws).items()
        }
        return getattr(bodo, call_name[0])(*args, **kwargs)

    # evaluate JIT function at compile time if arguments can be constant and it is a
    # "pure" function (has no side effects and only depends on input values for output)
    if (
        is_call(var_def)
        and typemap
        and isinstance(typemap.get(var_def.func.name, None), types.Dispatcher)
    ):
        py_func = typemap[var_def.func.name].dispatcher.py_func
        require(var_def.vararg is None)
        args = tuple(
            get_const_value_inner(func_ir, v, arg_types, typemap, updated_containers)
            for v in var_def.args
        )
        kwargs = {
            name: get_const_value_inner(
                func_ir, v, arg_types, typemap, updated_containers
            )
            for name, v in dict(var_def.kws).items()
        }
        arg_types = tuple(bodo.typeof(v) for v in args)
        kw_types = {k: bodo.typeof(v) for k, v in kwargs.items()}
        require(_func_is_pure(py_func, arg_types, kw_types))
        return py_func(*args, **kwargs)

    # BodoSQL optional getitem
    if call_name == ("scalar_optional_getitem", "bodo.utils.indexing"):
        value = get_const_value_inner(
            func_ir, var_def.args[0], arg_types, typemap, updated_containers
        )
        index = get_const_value_inner(
            func_ir, var_def.args[1], arg_types, typemap, updated_containers
        )
        return value[index]

    raise GuardException("Constant value not found")


def _func_is_pure(py_func, arg_types, kw_types):
    """return True if py_func is a pure function: output depends on input only
    (e.g. no I/O) and has no side-effects"""
    from bodo.hiframes.pd_dataframe_ext import DataFrameType
    from bodo.hiframes.pd_series_ext import SeriesType
    from bodo.ir.connector import Connector

    f_ir, typemap, _, _ = bodo.compiler.get_func_type_info(py_func, arg_types, kw_types)
    for block in f_ir.blocks.values():
        for stmt in block.body:
            # print
            if isinstance(stmt, ir.Print):
                return False
            # I/O nodes
            if isinstance(stmt, Connector):
                return False
            # setitem of input arguments like lists causes reflection
            if is_setitem(stmt) and isinstance(
                guard(get_definition, f_ir, stmt.target), ir.Arg
            ):
                return False
            if is_assign(stmt):
                rhs = stmt.value
                # generators keep state so not pure
                if isinstance(rhs, ir.Yield):
                    return False
                if is_call(rhs):
                    func_var_def = guard(get_definition, f_ir, rhs.func)
                    # assume objmode is not pure since we can't fully analyze it
                    if isinstance(func_var_def, ir.Const) and isinstance(
                        func_var_def.value, numba.core.dispatcher.ObjModeLiftedWith
                    ):
                        return False
                    fdef = guard(find_callname, f_ir, rhs)
                    # assume not pure if function call can't be analyzed
                    if fdef is None:
                        return False
                    func_name, func_mod = fdef
                    # check I/O functions
                    if func_mod == "pandas" and func_name.startswith("read_"):
                        return False
                    if fdef in (
                        ("fromfile", "numpy"),
                        ("file_read", "bodo.io.np_io"),
                    ):
                        return False
                    if fdef == ("File", "h5py"):
                        return False
                    if isinstance(func_mod, ir.Var):
                        typ = typemap[func_mod.name]
                        if isinstance(
                            typ, (DataFrameType, SeriesType)
                        ) and func_name in (
                            "to_csv",
                            "to_excel",
                            "to_json",
                            "to_sql",
                            "to_pickle",
                            "to_parquet",
                            "info",
                        ):
                            return False
                        if isinstance(typ, types.Array) and func_name == "tofile":
                            return False
                        # logging calls have side effects
                        if isinstance(typ, bodo.LoggingLoggerType):
                            return False
                        # matplotlib types
                        if str(typ).startswith("Mpl"):
                            return False
                        # inplace container update
                        if func_name in container_update_method_names and isinstance(
                            guard(get_definition, f_ir, func_mod), ir.Arg
                        ):
                            return False

                    # random functions are not pure since change across calls
                    # time() is not deterministic across calls
                    if func_mod in (
                        "numpy.random",
                        "time",
                        "logging",
                        "matplotlib.pyplot",
                    ):
                        return False

    return True


# similar to Dispatcher.fold_argument_types in Numba
# https://github.com/numba/numba/blob/0872b372ca6bcab3b7c3f979d92b3427885713ad/numba/core/dispatcher.py#L56
def fold_argument_types(pysig, args, kws):
    """
    Given positional and named argument types, fold keyword arguments
    and resolve defaults by inserting types.Omitted() instances.
    """

    def normal_handler(index, param, value):
        return value

    def default_handler(index, param, default):
        return types.Omitted(default)

    def stararg_handler(index, param, values):
        return types.StarArgTuple(values)

    args = fold_arguments(
        pysig, args, kws, normal_handler, default_handler, stararg_handler
    )
    return args


def get_const_func_output_type(
    func, arg_types, kw_types, typing_context, target_context, is_udf=True
):
    """Get output type of constant function 'func' when compiled with 'arg_types' as
    argument types.
    'func' can be a MakeFunctionLiteral (inline lambda) or FunctionLiteral (function)
    'is_udf' prepares the output for UDF cases like Series.apply()
    """
    from bodo.hiframes.pd_series_ext import HeterogeneousSeriesType, SeriesType

    py_func = None
    # MakeFunctionLiteral is not possible currently due to Numba's
    # MakeFunctionToJitFunction pass but may be possible later
    if isinstance(func, types.MakeFunctionLiteral):  # pragma: no cover
        code = func.literal_value.code
        _globals = {"np": np, "pd": pd, "numba": numba, "bodo": bodo}
        # XXX hack in untyped_pass to make globals available
        if hasattr(func.literal_value, "globals"):
            # TODO: use code.co_names to find globals actually used?
            _globals = func.literal_value.globals

        f_ir = numba.core.ir_utils.get_ir_of_code(_globals, code)
        fix_struct_return(f_ir)
        (
            typemap,
            f_return_type,
            calltypes,
            _,
        ) = numba.core.typed_passes.type_inference_stage(
            typing_context, target_context, f_ir, arg_types, None
        )
    elif isinstance(func, bodo.utils.typing.FunctionLiteral):
        py_func = func.literal_value
        f_ir, typemap, calltypes, f_return_type = bodo.compiler.get_func_type_info(
            py_func, arg_types, kw_types
        )
    elif isinstance(func, CPUDispatcher):
        py_func = func.py_func
        f_ir, typemap, calltypes, f_return_type = bodo.compiler.get_func_type_info(
            py_func, arg_types, kw_types
        )
    # TODO: [BE-129] If func is a built-in function we support, we should make use of it.
    else:
        if not isinstance(func, types.Dispatcher):  # pragma: no cover
            if isinstance(func, types.Function):
                raise BodoError(f"Bodo does not support built-in functions yet, {func}")
            else:
                raise BodoError(f"Function type expected, not {func}")
        py_func = func.dispatcher.py_func
        f_ir, typemap, calltypes, f_return_type = bodo.compiler.get_func_type_info(
            py_func, arg_types, kw_types
        )

    # replace returned dictionary with a StructType to enabling typing for
    # StructArrayType later
    if is_udf and isinstance(f_return_type, types.DictType):
        struct_key_names = guard(get_struct_keynames, f_ir, typemap)
        if struct_key_names is not None:
            f_return_type = StructType(
                (f_return_type.value_type,) * len(struct_key_names), struct_key_names
            )

    # add length/index info for constant Series output (required for output typing)
    if is_udf and isinstance(f_return_type, (SeriesType, HeterogeneousSeriesType)):
        # run SeriesPass to simplify Series calls (e.g. pd.Series)
        typingctx = numba.core.registry.cpu_target.typing_context
        targetctx = numba.core.registry.cpu_target.target_context
        # TODO: Can we capture parfor_metadata for error messages?
        series_pass = bodo.transforms.series_pass.SeriesPass(
            f_ir,
            typingctx,
            targetctx,
            typemap,
            calltypes,
            {},
        )

        changed = series_pass.run()
        # Needed, as series pass always causes changes for current
        # PR CI
        if changed:  # pragma: no cover
            changed = series_pass.run()
            if changed:
                series_pass.run()

        cfg = compute_cfg_from_blocks(f_ir.blocks)
        # get const info from all exit points and make sure they are consistent
        # checking for ir.Return since exit point could be for exception
        series_info = [
            guard(_get_const_series_info, f_ir.blocks[l], f_ir, typemap)
            for l in cfg.exit_points()
            if isinstance(f_ir.blocks[l].body[-1], ir.Return)
        ]
        if (
            None in series_info or len(pd.Series(series_info).unique()) != 1
        ):  # pragma: no cover
            f_return_type.const_info = None
        else:
            f_return_type.const_info = series_info[0]

    return f_return_type


def _get_const_series_info(block, f_ir, typemap):
    """get length and Index info for Series with constant length (homogeneous or heterogeneous values)"""
    from bodo.hiframes.pd_series_ext import HeterogeneousSeriesType

    assert isinstance(block.body[-1], ir.Return)
    return_var = block.body[-1].value
    ret_def = get_definition(f_ir, return_var)
    require(is_expr(ret_def, "cast"))
    ret_def = get_definition(f_ir, ret_def.value)

    require(
        is_call(ret_def)
        and find_callname(f_ir, ret_def)
        == ("init_series", "bodo.hiframes.pd_series_ext")
    )

    index_var = ret_def.args[1]
    index_vals = tuple(get_const_value_inner(f_ir, index_var, typemap=typemap))

    # length is known in type for heterogeneous Series
    if isinstance(typemap[return_var.name], HeterogeneousSeriesType):
        return len(typemap[return_var.name].data), index_vals

    # get length for homogeneous Series
    data_var = ret_def.args[0]
    data_def = get_definition(f_ir, data_var)

    func_name, mod_name = find_callname(f_ir, data_def)
    if is_call(data_def) and bodo.utils.utils.is_alloc_callname(func_name, mod_name):
        # If we have an allocation, we want to find the source of n if possible.
        alloc_len = data_def.args[0]
        total_len = get_const_value_inner(f_ir, alloc_len, typemap=typemap)
        return total_len, index_vals

    if is_call(data_def) and find_callname(f_ir, data_def) in [
        ("asarray", "numpy"),
        ("str_arr_from_sequence", "bodo.libs.str_arr_ext"),
        ("build_nullable_tuple", "bodo.libs.nullable_tuple_ext"),
    ]:
        data_var = data_def.args[0]
        data_def = get_definition(f_ir, data_var)

    require(is_expr(data_def, "build_tuple") or is_expr(data_def, "build_list"))
    return len(data_def.items), index_vals


def extract_keyvals_from_struct_map(f_ir, build_map, loc, scope, typemap=None):
    """extract keys and values from Expr.build_map of a struct value.
    Returns struct key names, tuple variables for keys/values, and assignments for
    generation of tuples.
    """
    keys = []
    key_strs = []
    values = []
    for k, v in build_map.items:
        k_str = find_const(f_ir, k)
        require(isinstance(k_str, str))
        key_strs.append(k_str)
        keys.append(k)
        values.append(v)

    # val_tup = (v1, v2)
    val_tup = ir.Var(scope, mk_unique_var("val_tup"), loc)
    val_tup_assign = ir.Assign(ir.Expr.build_tuple(values, loc), val_tup, loc)
    f_ir._definitions[val_tup.name] = [val_tup_assign.value]
    # key_tup = ("A", "B")
    key_tup = ir.Var(scope, mk_unique_var("key_tup"), loc)
    key_tup_assign = ir.Assign(ir.Expr.build_tuple(keys, loc), key_tup, loc)
    f_ir._definitions[key_tup.name] = [key_tup_assign.value]

    if typemap is not None:
        typemap[val_tup.name] = types.Tuple([typemap[v.name] for v in values])
        typemap[key_tup.name] = types.Tuple([typemap[v.name] for v in keys])

    return key_strs, val_tup, val_tup_assign, key_tup, key_tup_assign


def _replace_const_map_return(f_ir, block, label):
    """replaces constant dictionary return value with a struct if values are not
    homogeneous, e.g. {"A": 1, "B": 2.3} -> struct((1, 2.3), ("A", "B"))
    """
    # get const map in return
    require(isinstance(block.body[-1], ir.Return))
    return_val = block.body[-1].value
    cast_def = guard(get_definition, f_ir, return_val)
    require(is_expr(cast_def, "cast"))
    ret_def = guard(get_definition, f_ir, cast_def.value)
    require(is_expr(ret_def, "build_map"))
    require(len(ret_def.items) > 0)
    # {"A": v1, "B": v2} -> struct_if_heter_dict((v1, v2), ("A", "B"))
    loc = block.loc
    scope = block.scope
    (
        key_strs,
        val_tup,
        val_tup_assign,
        key_tup,
        key_tup_assign,
    ) = extract_keyvals_from_struct_map(f_ir, ret_def, loc, scope)

    # new_Var = struct_if_heter_dict(val_tup, key_tup)
    call_var = ir.Var(scope, mk_unique_var("conv_call"), loc)
    call_global = ir.Assign(
        ir.Global(
            "struct_if_heter_dict", bodo.utils.conversion.struct_if_heter_dict, loc
        ),
        call_var,
        loc,
    )
    f_ir._definitions[call_var.name] = [call_global.value]
    new_var = ir.Var(scope, mk_unique_var("struct_val"), loc)
    new_assign = ir.Assign(
        ir.Expr.call(call_var, [val_tup, key_tup], {}, loc), new_var, loc
    )
    f_ir._definitions[new_var.name] = [new_assign.value]
    cast_def.value = new_var
    # {"A": v1, "B": v2} -> {"A": "A", "B": "B"} to avoid typing errors
    ret_def.items = [(k, k) for (k, _) in ret_def.items]
    block.body = (
        block.body[:-2]
        + [val_tup_assign, key_tup_assign, call_global, new_assign]
        + block.body[-2:]
    )
    return tuple(key_strs)


def get_struct_keynames(f_ir, typemap):
    """returns the key names if output of f_ir is a struct created by
    struct_if_heter_dict(), otherwise None.
    """
    cfg = compute_cfg_from_blocks(f_ir.blocks)
    exit_label = list(cfg.exit_points())[0]
    block = f_ir.blocks[exit_label]
    require(isinstance(block.body[-1], ir.Return))
    return_val = block.body[-1].value
    cast_def = guard(get_definition, f_ir, return_val)
    require(is_expr(cast_def, "cast"))
    ret_def = guard(get_definition, f_ir, cast_def.value)
    require(
        is_call(ret_def)
        and find_callname(f_ir, ret_def)
        == ("struct_if_heter_dict", "bodo.utils.conversion")
    )
    return get_overload_const_list(typemap[ret_def.args[1].name])


def fix_struct_return(f_ir):
    """replaces constant dictionary return value with a struct for all return blocks
    in 'f_ir'. Returns the key names if output is a struct.
    """
    key_names = None
    cfg = compute_cfg_from_blocks(f_ir.blocks)
    for exit_label in cfg.exit_points():
        key_names = guard(
            _replace_const_map_return, f_ir, f_ir.blocks[exit_label], exit_label
        )
    return key_names


def update_node_list_definitions(node_list, func_ir):
    loc = ir.Loc("", 0)
    dumm_block = ir.Block(ir.Scope(None, loc), loc)
    dumm_block.body = node_list
    build_definitions({0: dumm_block}, func_ir._definitions)
    return


# sentinel for nested const tuple gen used below
NESTED_TUP_SENTINEL = "$BODO_NESTED_TUP"


def gen_const_val_str(c):
    """convert value 'c' to string constant"""
    # const nested constant tuples are not supported in Numba yet, need special handling
    # HACK: flatten tuple values but add a sentinel value that specifies how many
    # elements are from the nested tuple. Supports only one level nesting
    # TODO: fix nested const tuple handling in Numba
    if isinstance(c, tuple):
        return "'{}{}', ".format(NESTED_TUP_SENTINEL, len(c)) + ", ".join(
            gen_const_val_str(v) for v in c
        )
    if isinstance(c, str):
        return "'{}'".format(c)
    # TODO: Support actual timestamp, timedelta, float values
    if isinstance(c, (pd.Timestamp, pd.Timedelta, float)):
        # Timestamp has a space
        return "'{}'".format(c)
    return str(c)


def gen_const_tup(vals):
    """generate a constant tuple value as text"""
    val_seq = ", ".join(gen_const_val_str(c) for c in vals)
    return "({}{})".format(
        val_seq,
        "," if len(vals) == 1 else "",
    )


def get_const_tup_vals(c_typ):
    """get constant values from a tuple type generated using 'gen_const_tup'
    reverses the hack in 'gen_const_val_str'
    """
    vals = get_overload_const_list(c_typ)
    return _get_original_nested_tups(vals)


def _get_original_nested_tups(vals):
    """find potential translated nested tuples in vals and reverse to the original
    nested format (before gen_const_val_str translation).
    """
    # scan for sentinel from the end to handle nested cases properly
    # e.g. ('$BODO_NESTED_TUP2', '$BODO_NESTED_TUP2', 'B', 'sum', 'sum') ->
    # ((("B", "sum"), "sum"),)
    for i in range(len(vals) - 1, -1, -1):
        v = vals[i]
        if isinstance(v, str) and v.startswith(NESTED_TUP_SENTINEL):
            n_elem = int(v[len(NESTED_TUP_SENTINEL) :])
            # translate to nested tuple item and call function recursively
            return _get_original_nested_tups(
                tuple(vals[:i])
                + (tuple(vals[i + 1 : i + n_elem + 1]),)
                + tuple(vals[i + n_elem + 1 :])
            )

    return tuple(vals)


# dummy sentinel singleton to designate constant value not found for variable
class ConstNotFound:
    pass


CONST_NOT_FOUND = ConstNotFound()


def get_const_arg(
    f_name,
    args,
    kws,
    func_ir,
    func_arg_types,
    arg_no,
    arg_name,
    loc,
    default=None,
    err_msg: pt.Optional[str] = None,
    typ: pt.Optional[str] = None,
    use_default: bool = False,
):
    """Get constant value for a function call argument. Raise error if the value is
    not constant.
    """
    typ = "str" if typ is None else typ
    arg = CONST_NOT_FOUND
    if err_msg is None:
        err_msg = ("{} requires '{}' argument as a constant {}").format(
            f_name, arg_name, typ
        )

    arg_var = get_call_expr_arg(f_name, args, kws, arg_no, arg_name, "")

    try:
        arg = get_const_value_inner(func_ir, arg_var, arg_types=func_arg_types)
    except GuardException:
        # raise error if argument specified but not constant
        if arg_var != "":
            raise BodoError(err_msg, loc=loc)

    if arg is CONST_NOT_FOUND:
        # Provide use_default to allow letting None be the default value
        if use_default or default is not None:
            return default
        raise BodoError(err_msg, loc=loc)
    return arg


def get_call_expr_arg(
    f_name, args, kws, arg_no, arg_name, default=None, err_msg=None, use_default=False
):
    """get a specific argument from all argument variables of a call expr, which could
    be specified either as a positional argument or keyword argument.
    If argument is not specified, an error is raised unless if a default is specified.
    """
    arg = None
    # If an argument is kwonly, arg_no < 0
    if len(args) > arg_no and arg_no >= 0:
        arg = args[arg_no]
        if arg_name in kws:
            err_msg = f"{f_name}() got multiple values for argument '{arg_name}'"
            raise BodoError(err_msg)
    elif arg_name in kws:
        arg = kws[arg_name]

    if arg is None:
        # Check use_default to allow None as a default
        if use_default or default is not None:
            return default
        if err_msg is None:
            err_msg = "{} requires '{}' argument".format(f_name, arg_name)
        raise BodoError(err_msg)
    return arg


def set_call_expr_arg(var, args, kws, arg_no, arg_name, add_if_missing=False):
    """replaces call argument with a new variable.
    The add_if_missing flag appends the argument to the kws if it doesn't
    already exist. Otherwise this raises an error if argument was not specified.
    """
    if len(args) > arg_no:
        args[arg_no] = var
    elif add_if_missing or arg_name in kws:
        kws[arg_name] = var
    else:
        raise BodoError(
            "cannot set call argument since does not exist"
        )  # pragma: no cover


def set_ith_arg_to_omitted_value(
    pass_info,
    rhs: ir.Expr,
    i: int,
    new_value: pt.Any,
    expected_existing_value: pt.Any,
):
    """
    Sets the last argument of call expr 'rhs' to True, assuming that it is an Omitted
    value with the given expected_existing_value. This is done by modifying pass_info's
    typing information directly.

    This is usually used for Bodo overloads that have an extra flag as last argument
    to enable parallelism but more generic to allow communicating more complex parallelism
    semantics, such as a tuple of values.

    Args:
        pass_info (_type_): The pass information used to grab the call type.
        rhs (ir.Expr): The rhs call expression to modify.
        i (int): The index of the argument to replace using Python integer indexing.
        new_value (pt.Any): The new value for the last argument without wrapping the result
          in any "optional".
        expected_existing_value (pt.Any): The existing value of the last argument that
          should be replaced. This is used for assertion check.
    """
    call_type = pass_info.calltypes.pop(rhs)
    # normalize to simplify slicing.
    pos_idx = i if i >= 0 else len(call_type.args) + i
    assert call_type.args[pos_idx] == types.Omitted(
        expected_existing_value
    ), f"Omitted({expected_existing_value}) {pos_idx}th argument expected"
    pass_info.calltypes[rhs] = pass_info.typemap[rhs.func.name].get_call_type(
        pass_info.typingctx,
        call_type.args[:pos_idx]
        + (types.Omitted(new_value),)
        + call_type.args[pos_idx + 1 :],
        {},
    )


def set_last_arg_to_true(pass_info, rhs):
    """set last argument of call expr 'rhs' to True, assuming that it is an Omitted
    arg with value of False.
    This is usually used for Bodo overloads that have an extra flag as last argument
    to enable parallelism.
    """
    set_ith_arg_to_omitted_value(pass_info, rhs, -1, True, False)


def set_2nd_to_last_arg_to_true(pass_info, rhs):
    """Same as above but sets second to last argument to True"""
    set_ith_arg_to_omitted_value(pass_info, rhs, -2, True, False)


def avoid_udf_inline(py_func, arg_types, kw_types):
    """return True if UDF function should not be inlined because:
    1) it has assertions (which breaks prange)
    2) it has context manager like objmode blocks (BE-290)
    3) it has dataframe input so probably expensive (BE-265)
    """
    from bodo.hiframes.pd_dataframe_ext import DataFrameType

    f_ir = numba.core.compiler.run_frontend(py_func, inline_closures=True)

    # there is explicit _bodo_inline arg
    if "_bodo_inline" in kw_types and is_overload_constant_bool(
        kw_types["_bodo_inline"]
    ):
        return not get_overload_const_bool(kw_types["_bodo_inline"])

    # there is dataframe input
    if any(isinstance(t, DataFrameType) for t in arg_types + tuple(kw_types.values())):
        return True

    for block in f_ir.blocks.values():
        # assertions
        # TODO(ehsan): add TryRaise/StaticTryRaise/DynamicTryRaise?
        if isinstance(block.body[-1], (ir.Raise, ir.StaticRaise, ir.DynamicRaise)):
            return True
        # has context manager
        for stmt in block.body:
            if isinstance(stmt, ir.EnterWith):
                return True
    return False


def replace_func(
    pass_info,
    func,
    args,
    const=False,
    pre_nodes=None,
    extra_globals=None,
    pysig=None,
    kws=None,
    inline_bodo_calls=False,
    run_full_pipeline=False,
):
    """"""
    glbls = {"numba": numba, "np": np, "bodo": bodo, "pd": pd}
    if extra_globals is not None:
        glbls.update(extra_globals)
    func.__globals__.update(glbls)

    # create explicit arg variables for defaults if func has any
    # XXX: inline_closure_call() can't handle defaults properly
    if pysig is not None:
        pre_nodes = [] if pre_nodes is None else pre_nodes
        scope = next(iter(pass_info.func_ir.blocks.values())).scope
        loc = scope.loc

        def normal_handler(index, param, default):
            return default

        def default_handler(index, param, default):
            d_var = ir.Var(scope, mk_unique_var("defaults"), loc)
            # try to use a literal type if possible (as required by some overloads)
            try:
                pass_info.typemap[d_var.name] = types.literal(default)
            except:
                pass_info.typemap[d_var.name] = numba.typeof(default)
            node = ir.Assign(ir.Const(default, loc), d_var, loc)
            pre_nodes.append(node)
            return d_var

        # TODO: stararg needs special handling?
        args = numba.core.typing.fold_arguments(
            pysig, args, kws, normal_handler, default_handler, normal_handler
        )

    arg_typs = tuple(pass_info.typemap[v.name] for v in args)

    if const:
        new_args = []
        for i, arg in enumerate(args):
            val = guard(find_const, pass_info.func_ir, arg)
            if val:
                new_args.append(types.literal(val))
            else:
                new_args.append(arg_typs[i])
        arg_typs = tuple(new_args)
    return ReplaceFunc(
        func, arg_typs, args, glbls, inline_bodo_calls, run_full_pipeline, pre_nodes
    )


############################# UDF utils ############################


def is_var_size_item_array_type(t):
    """returns True if array type 't' has variable size items (e.g. strings)"""
    assert is_array_typ(t, False)
    return (
        t == string_array_type
        or isinstance(t, ArrayItemArrayType)
        or (
            isinstance(t, StructArrayType)
            and any(is_var_size_item_array_type(d) for d in t.data)
        )
    )


def gen_init_varsize_alloc_sizes(t):
    """generate initialization code as text for allocation sizes for arrays with
    variable items, e.g. total number of characters in string arrays
    """
    # TODO: handle all possible array types and nested cases, e.g. struct
    if t == string_array_type:
        vname = "num_chars_{}".format(ir_utils.next_label())
        return f"  {vname} = 0\n", (vname,)
    if isinstance(t, ArrayItemArrayType):
        inner_code, inner_vars = gen_init_varsize_alloc_sizes(t.dtype)
        vname = "num_items_{}".format(ir_utils.next_label())
        return f"  {vname} = 0\n" + inner_code, (vname,) + inner_vars
    return "", ()


def gen_varsize_item_sizes(t, item, var_names):
    """generate aggregation code as text for allocation sizes for arrays with
    variable items, e.g. total number of characters in string arrays
    """
    # TODO: handle all possible array types and nested cases, e.g. struct
    if t == string_array_type:
        return "    {} += bodo.libs.str_arr_ext.get_utf8_size({})\n".format(
            var_names[0], item
        )
    if isinstance(t, ArrayItemArrayType):
        return "    {} += len({})\n".format(
            var_names[0], item
        ) + gen_varsize_array_counts(t.dtype, item, var_names[1:])
    return ""


def gen_varsize_array_counts(t, item, var_names):
    """count the total number of elements in a nested array. e.g. total characters in a
    string array.
    """
    # TODO: other arrays
    if t == string_array_type:
        return "    {} += bodo.libs.str_arr_ext.get_num_total_chars({})\n".format(
            var_names[0], item
        )
    return ""


def get_type_alloc_counts(t):
    """get the number of counts needed for upfront allocation of array of type 't'.
    For example, ArrayItemArrayType(ArrayItemArrayType(array(int64))) returns 3.
    """
    if isinstance(t, (StructArrayType, TupleArrayType)):
        return 1 + sum(get_type_alloc_counts(d.dtype) for d in t.data)

    if t == string_array_type or t == bodo.binary_array_type:
        return 2

    if isinstance(t, ArrayItemArrayType):
        return 1 + get_type_alloc_counts(t.dtype)

    if isinstance(t, MapArrayType):
        # 2 counts are needed for length and total number of key/value pairs, which are
        # included since length of key/value arrays is counted twice
        return get_type_alloc_counts(t.key_arr_type) + get_type_alloc_counts(
            t.value_arr_type
        )

    if bodo.utils.utils.is_array_typ(t, False) or t == bodo.string_type:
        return 1

    if isinstance(t, StructType):
        return sum(get_type_alloc_counts(d) for d in t.data)

    if isinstance(t, types.BaseTuple):
        return sum(get_type_alloc_counts(d) for d in t.types)

    return 0


def find_udf_str_name(obj_dtype, func_name, typing_context, caller_name):
    """
    Given an obj_dtype, func_name, and a typing_context, this function
    finds a matching implementation to use inside apply.

    caller_name refers to the function calling this helper
    function and is used solely for error messages.

    Mimicing Pandas behavior, we first try and find a matching implementation
    which is method for obj_dtype. If there is no matching implementation
    then we attempt to find a matching Numpy function. If no match exists we
    raise a Bodo Error.

    https://github.com/pandas-dev/pandas/blob/8e07787bc1030e5d13d3ad5e83b5d060a519ef67/pandas/core/apply.py#L564

    If the Pandas method is not currently supported we may not match Pandas behavior.
    When the method is supported but not with the current type, we rely on the Bodo Error
    inside the implementation. If a method has no implementation we may incorrectly try
    and use a Numpy function.

    This function returns a 'return_type', which can be used to check typing, extract
    a return type, and find an implementation.
    """
    result = typing_context.resolve_getattr(obj_dtype, func_name)
    if result is None:
        # Find a Numpy implementation or raise an error
        numpy_mod = types.misc.Module(np)
        try:
            result = typing_context.resolve_getattr(numpy_mod, func_name)
        except AttributeError:
            # Numpy tries to look up getattr on func_name. If this doesn't exist the
            # error message is less clear
            result = None
        if result is None:
            raise BodoError(
                f"{caller_name}(): No Pandas method or Numpy function found with the name '{func_name}'."
            )
    return result


def get_udf_str_return_type(
    obj_dtype, func_name, typing_context, caller_name, axis=None
):
    """
    Given an obj_dtype, func_name, and a typing_context, this function returns
    the return type for the implementation used inside apply.

    These functions do not normally take arguments except arguments from apply may be forwarded.
    Currently we only support axis, which if supply we attempt to pass to any Pandas methods.
    If it fails we try again without axis.

    This function uses find_udf_str_name to find the correct overload and simply calls
    the function with the given types.
    """
    result = find_udf_str_name(obj_dtype, func_name, typing_context, caller_name)
    if isinstance(result, types.BoundFunction):
        # Methods are Bound Functions
        if axis is not None:
            # If axis is provided we may need to pass it to DataFrame methods.
            # Pandas is inconsistent about supporting the axis variable, but
            # recent changes suggest they will only allow it where it is supported.
            sig = result.get_call_type(typing_context, (), {"axis": axis})
        else:
            sig = result.get_call_type(typing_context, (), {})
        return sig.return_type
    else:
        if bodo.utils.typing.is_numpy_ufunc(result):
            # Functions require passing obj_dtype as an argument
            sig = result.get_call_type(typing_context, (obj_dtype,), {})
            return sig.return_type
        raise BodoError(
            f"{caller_name}(): Only Pandas methods and np.ufunc are supported as string literals. '{func_name}' not supported."
        )


def get_pandas_method_str_impl(
    obj_dtype, func_name, typing_context, caller_name, axis=None
):
    """
    Given an obj_dtype, func_name, and a typing_context, this function returns
    the function used that implements the provided Pandas method.

    These functions do not normally take arguments except arguments from apply may be forwarded.
    Currently we only support axis.

    This function uses find_udf_str_name to find the correct
    function and uses its internal information to find the implementation.
    If the function is a Numpy udf instead it returns None.
    """
    result = find_udf_str_name(obj_dtype, func_name, typing_context, caller_name)
    if isinstance(result, types.BoundFunction):
        # Methods are Bound Functions
        template = result.template
        # TODO: Handle situations where we don't have an overload?
        if axis is not None:
            return template._overload_func(obj_dtype, axis=axis)
        else:
            return template._overload_func(obj_dtype)
    return None


def dict_to_const_keys_var_values_lists(
    dict_var, func_ir, arg_types, typemap, updated_containers, require_const_map, label
):
    """
    Takes a dictionary variable, which should be created
    with build_map and returns a list of keys and a list of values.

    This is used for the case where a dictionary is required to have
    literal keys but may not be required to have literal values.

    It returns 2 values:
        - keys: Python list of literal values
        - values: Python list of values variables

    For each key that cannot be made a constant, it updates the require_const_map
    with the provided label. If there is any key that cnanot be a constant a
    GuardException is raised.
    """
    # Influenced by numba.core.ir_utils.find_build_sequence
    require(isinstance(dict_var, ir.Var))
    dict_def = get_definition(func_ir, dict_var)
    require(isinstance(dict_def, ir.Expr))
    require(dict_def.op == "build_map")
    dict_items = dict_def.items
    keys = []
    values = []
    needs_transform = False
    for i in range(len(dict_items)):
        key, value = dict_items[i]
        try:
            key_const = get_const_value_inner(
                func_ir,
                key,
                arg_types,
                typemap,
                updated_containers,
            )
            keys.append(key_const)
            values.append(value)
        except GuardException:
            # save for potential loop unrolling
            require_const_map[key] = label
            needs_transform = True
    if needs_transform:
        raise GuardException
    return keys, values


def list_to_vars_value_list(list_var, func_ir):
    """
    Takes a list variable, which should be created
    via build_list and returns a list of values.

    This is used for the case where the list must be a literal to
    determine the variables, but the variables don't need to be constant.
    """
    # Influenced by numba.core.ir_utils.find_build_sequence
    require(isinstance(list_var, ir.Var))
    list_def = get_definition(func_ir, list_var)
    require(isinstance(list_def, ir.Expr))
    require(list_def.op == "build_list")
    return list_def.items


def tuples_to_vars_value_list(tuple_var, func_ir):
    """
    Takes a tuple variable, which should be created
    via build_tuple and returns a list of values.

    This is used for the case where the list must be a literal to
    determine the variables, but the variables don't need to be constant.
    """
    # Influenced by numba.core.ir_utils.find_build_sequence
    require(isinstance(tuple_var, ir.Var))
    tuple_def = get_definition(func_ir, tuple_var)
    require(isinstance(tuple_def, ir.Expr))
    require(tuple_def.op == "build_tuple")
    return tuple_def.items


def _get_const_keys_from_dict(args, func_ir, build_map, err_msg, loc):
    # check keys to be string/int
    try:
        keys = tuple(
            get_const_value_inner(
                func_ir,
                t[0],
                args,
            )
            for t in build_map.items
        )
    except GuardException:
        raise BodoError(err_msg, loc)

    if not all(isinstance(c, (str, int)) for c in keys):
        raise BodoError(err_msg, loc)

    return keys


def _convert_const_key_dict(
    args, func_ir, build_map, err_msg, scope, loc, output_sentinel_tuple=False
):
    """converts a constant key dictionary build_map into either a tuple with sentinel, or two tuples
    of keys/values as a workaround to extract key/values in overloads
    """

    keys = _get_const_keys_from_dict(args, func_ir, build_map, err_msg, loc)

    new_nodes = []
    key_const_variables = [
        bodo.transforms.typing_pass._create_const_var(
            k, "dict_key", scope, loc, new_nodes
        )
        for k in keys
    ]
    value_variables = [t[1] for t in build_map.items]

    # create tuple with sentinel
    if output_sentinel_tuple:
        sentinel_var = ir.Var(scope, mk_unique_var("sentinel"), loc)
        tup_var = ir.Var(scope, mk_unique_var("dict_tup"), loc)
        new_nodes.append(ir.Assign(ir.Const("__bodo_tup", loc), sentinel_var, loc))
        tup_items = [sentinel_var] + key_const_variables + value_variables
        new_nodes.append(ir.Assign(ir.Expr.build_tuple(tup_items, loc), tup_var, loc))
        return (tup_var,), new_nodes
    else:
        val_tup_var = ir.Var(scope, mk_unique_var("values_tup"), loc)
        idx_tup_var = ir.Var(scope, mk_unique_var("idx_tup"), loc)

        new_nodes.append(
            ir.Assign(ir.Expr.build_tuple(value_variables, loc), val_tup_var, loc)
        )
        new_nodes.append(
            ir.Assign(ir.Expr.build_tuple(key_const_variables, loc), idx_tup_var, loc)
        )

        return (val_tup_var, idx_tup_var), new_nodes


def get_runtime_join_filter_terms(
    func_ir: ir.FunctionIR, _bodo_runtime_join_filters_arg: pt.Optional[ir.Expr]
) -> pt.Optional[list[tuple[ir.Var, tuple[int], tuple[int, int, str]]]]:
    """
    Takes a function IR and an expression for the runtime join filters argument to the function.
    Extracts the join state variables and column indices from the runtime join filters argument so
    we can access them at compile time to generate code to apply the runtime join filters.
    """
    if _bodo_runtime_join_filters_arg is None:
        # If the tuple is absent, then no runtime
        # join filters were provided so we can skip the codepath.
        rtjf_terms = None
    else:
        _bodo_runtime_join_filters_defn = numba.core.ir_utils.get_definition(
            func_ir,
            _bodo_runtime_join_filters_arg,
        )
        if (
            isinstance(_bodo_runtime_join_filters_defn, ir.Const)
            and _bodo_runtime_join_filters_defn.value is None
        ):
            # If the tuple is explicitly set to None, then no runtime
            # join filters were provided so we can skip the codepath.
            rtjf_terms = None
        else:
            # Otherwise, we create a list of (state_var, indices)
            # tuples from the raw arguments and store in the SqlReader
            rtjf_terms = []
            for rtjf_tuple in _bodo_runtime_join_filters_defn.items:
                tup_defn = numba.core.ir_utils.get_definition(func_ir, rtjf_tuple)
                # Verify that the tuple is well formed
                if len(tup_defn.items) != 3:
                    raise_bodo_error(
                        f"Invalid runtime join filter tuple. Expected 2 elements per tuple, instead had {len(tup_defn.items)}"
                    )
                # Extract the state variable
                state_var = tup_defn.items[0]
                if not isinstance(state_var, ir.Var):
                    raise_bodo_error(
                        f"Invalid runtime join filter tuple. Expected the first argument to be a Var, instead got {state_var}."
                    )
                # Extract the column indices
                col_indices_meta = numba.core.ir_utils.get_definition(
                    func_ir, tup_defn.items[1]
                )
                if not isinstance(col_indices_meta, ir.Global) or not isinstance(
                    col_indices_meta.value, bodo.utils.typing.MetaType
                ):
                    raise_bodo_error(
                        f"Invalid runtime join filter tuple. Expected the second argument to be a global MetaType tuple, instead got {col_indices_meta}."
                    )
                col_indices_tup = col_indices_meta.value.meta
                non_equality_meta = numba.core.ir_utils.get_definition(
                    func_ir, tup_defn.items[2]
                )
                if not isinstance(non_equality_meta, ir.Global) or not isinstance(
                    non_equality_meta.value, bodo.utils.typing.MetaType
                ):
                    raise_bodo_error(
                        f"Invalid runtime join filter tuple. Expected the second argument to be a global MetaType tuple, instead got {non_equality_meta}."
                    )
                non_equality_tup = non_equality_meta.value.meta
                rtjf_terms.append((state_var, col_indices_tup, non_equality_tup))
    return rtjf_terms


def create_nested_run_pass_event(pass_name: str, state, pass_obj):
    """
    Creates a nested call to "run_pass" from inside another Bodo compiler
    pass.

    Args:
        pass_name (str): The name of the pass for logging purposes.
        state (StateDict): The state object that contains the IR and type information and is used for invoking
        "run_pass" on the given state.
        pass_obj (_type_): Any compiler pass object that can invoke "run_pass" on the given
        state.
    """
    # Code is translated from Numba:
    # https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/compiler_machinery.py#L307
    # Note we removed most of the event details because they are unused and some of our calls may not have all of
    # of the necessary information.
    ev_details = dict(name=f"{pass_name} [...]")
    with event.trigger_event("numba:run_pass", data=ev_details):
        pass_obj.run_pass(state)
