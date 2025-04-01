"""IR node for the groupby"""

import ctypes
import operator
import types as pytypes
from collections import defaultdict, namedtuple

import numba
import numpy as np
import pandas as pd
from llvmlite import ir as lir
from numba.core import cgutils, compiler, ir, ir_utils, types
from numba.core.analysis import compute_use_defs
from numba.core.ir_utils import (
    build_definitions,
    compile_to_numba_ir,
    find_callname,
    find_const,
    find_topo_order,
    get_definition,
    get_ir_of_code,
    get_name_var_table,
    guard,
    is_getitem,
    mk_unique_var,
    next_label,
    remove_dels,
    replace_arg_nodes,
    replace_var_names,
    replace_vars_inner,
    visit_vars_inner,
)
from numba.core.typing import signature
from numba.core.typing.templates import AbstractTemplate, infer_global
from numba.extending import intrinsic
from numba.parfors.parfor import (
    Parfor,
    unwrap_parfor_blocks,
    wrap_parfor_blocks,
)

import bodo
from bodo.hiframes.datetime_date_ext import DatetimeDateArrayType
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.libs.array import (
    arr_info_list_to_table,
    array_from_cpp_table,
    array_to_info,
    cpp_table_to_py_data,
    delete_info,
    delete_table,
    groupby_and_aggregate,
    py_data_to_cpp_table,
)
from bodo.libs.array_item_arr_ext import (
    ArrayItemArrayType,
    pre_alloc_array_item_array,
)
from bodo.libs.binary_arr_ext import BinaryArrayType, pre_alloc_binary_array
from bodo.libs.bool_arr_ext import BooleanArrayType
from bodo.libs.decimal_arr_ext import DecimalArrayType, alloc_decimal_array
from bodo.libs.float_arr_ext import FloatingArrayType
from bodo.libs.int_arr_ext import IntDtype, IntegerArrayType
from bodo.libs.str_arr_ext import (
    StringArrayType,
    pre_alloc_string_array,
    string_array_type,
)
from bodo.libs.str_ext import string_type
from bodo.transforms import distributed_analysis, distributed_pass
from bodo.transforms.distributed_analysis import Distribution
from bodo.transforms.table_column_del_pass import (
    _compute_table_column_uses,
    _find_used_columns,
    ir_extension_table_column_use,
    remove_dead_column_extensions,
)
from bodo.utils.transform import create_nested_run_pass_event, get_call_expr_arg
from bodo.utils.typing import (
    BodoError,
    MetaType,
    decode_if_dict_array,
    dtype_to_array_type,
    get_index_data_arr_types,
    get_literal_value,
    get_overload_const_func,
    get_overload_const_list,
    get_overload_const_str,
    get_overload_constant_dict,
    is_overload_constant_dict,
    is_overload_constant_list,
    is_overload_constant_str,
    list_cumulative,
    to_str_arr_if_dict_array,
    type_has_unknown_cats,
    unwrap_typeref,
)
from bodo.utils.utils import (
    gen_getitem,
    get_const_or_build_tuple_of_consts,
    is_assign,
    is_call_assign,
    is_expr,
    is_null_pointer,
    is_var_assign,
)

# TODO: it's probably a bad idea for these to be global. Maybe try moving them
# to a context or dispatcher object somehow
# Maps symbol name to cfunc object that implements UDF for groupby. This dict
# is used only when compiling
gb_agg_cfunc = {}
# Maps symbol name to cfunc address (used when compiling and loading from cache)
# When compiling, this is populated in aggregate.py::gen_top_level_agg_func
# When loading from cache, this is populated in numba_compat.py::resolve_gb_agg_funcs
# when the compiled result is loaded from cache
gb_agg_cfunc_addr = {}


@intrinsic(prefer_literal=True)
def add_agg_cfunc_sym(typingctx, func, sym):
    """This "registers" a cfunc that implements part of groupby.agg UDF to ensure
    it can be cached. It does two things:
    - Generate a dummy call to the cfunc to make sure the symbol is not
      discarded during linking
    - Add cfunc library to the library of the Bodo function being compiled
      (necessary for caching so that the cfunc is part of the cached result)
    """

    def codegen(context, builder, signature, args):
        # generate dummy call to the cfunc
        sig = func.signature
        if sig == types.none(types.voidptr):
            # cfunc generated with gen_eval_cb has this signature
            fnty = lir.FunctionType(
                lir.VoidType(),
                [
                    lir.IntType(8).as_pointer(),
                ],
            )
            fn_tp = cgutils.get_or_insert_function(
                builder.module, fnty, sym._literal_value
            )
            builder.call(
                fn_tp,
                [
                    context.get_constant_null(sig.args[0]),
                ],
            )
        elif sig == types.none(types.int64, types.voidptr, types.voidptr):
            # cfunc generated with gen_general_udf_cb has this signature
            fnty = lir.FunctionType(
                lir.VoidType(),
                [
                    lir.IntType(64),
                    lir.IntType(8).as_pointer(),
                    lir.IntType(8).as_pointer(),
                ],
            )
            fn_tp = cgutils.get_or_insert_function(
                builder.module, fnty, sym._literal_value
            )
            builder.call(
                fn_tp,
                [
                    context.get_constant(types.int64, 0),
                    context.get_constant_null(sig.args[1]),
                    context.get_constant_null(sig.args[2]),
                ],
            )
        else:
            # Assume signature is none(voidptr, voidptr, int64*) (see gen_update_cb
            # and gen_combine_cb)
            fnty = lir.FunctionType(
                lir.VoidType(),
                [
                    lir.IntType(8).as_pointer(),
                    lir.IntType(8).as_pointer(),
                    lir.IntType(64).as_pointer(),
                ],
            )
            fn_tp = cgutils.get_or_insert_function(
                builder.module, fnty, sym._literal_value
            )
            builder.call(
                fn_tp,
                [
                    context.get_constant_null(sig.args[0]),
                    context.get_constant_null(sig.args[1]),
                    context.get_constant_null(sig.args[2]),
                ],
            )
        # add cfunc library to the library of the Bodo function being compiled.
        context.add_linking_libs([gb_agg_cfunc[sym._literal_value]._library])
        return

    return types.none(func, sym), codegen


@numba.njit
def get_agg_udf_addr(name):
    """Resolve address of cfunc given by its symbol name"""
    with bodo.no_warning_objmode(addr="int64"):
        addr = bodo.ir.aggregate.gb_agg_cfunc_addr[name]
    return addr


class AggUDFStruct:
    """Holds the compiled functions and information of groupby UDFs,
    used to generate the cfuncs that are called from C++"""

    def __init__(self, regular_udf_funcs=None, general_udf_funcs=None):
        assert regular_udf_funcs is not None or general_udf_funcs is not None
        self.regular_udfs = False
        self.general_udfs = False
        self.regular_udf_cfuncs = None
        self.general_udf_cfunc = None
        if regular_udf_funcs is not None:
            (
                self.var_typs,
                self.init_func,
                self.update_all_func,
                self.combine_all_func,
                self.eval_all_func,
            ) = regular_udf_funcs
            self.regular_udfs = True
        if general_udf_funcs is not None:
            self.general_udf_funcs = general_udf_funcs
            self.general_udfs = True

    def set_regular_cfuncs(self, update_cb, combine_cb, eval_cb):
        """Set the cfuncs that are called from C++ that apply regular UDFs"""
        assert self.regular_udfs and self.regular_udf_cfuncs is None
        self.regular_udf_cfuncs = [update_cb, combine_cb, eval_cb]

    def set_general_cfunc(self, general_udf_cb):
        """Set the cfunc that is called from C++ that applies general UDFs"""
        assert self.general_udfs and self.general_udf_cfunc is None
        self.general_udf_cfunc = general_udf_cb


AggFuncStruct = namedtuple("AggFuncStruct", ["func", "ftype"])


# !!! IMPORTANT: this is supposed to match the positions in
# Bodo_FTypes::FTypeEnum in _groupby_ftypes.h
supported_agg_funcs = [
    "no_op",  # needed to ensure that 0 value isn't matched with any function
    "ngroup",
    "head",
    "transform",
    "size",
    "shift",
    "sum",
    "count",
    "nunique",
    "median",
    "cumsum",
    "cumprod",
    "cummin",
    "cummax",
    "mean",
    "min",
    "max",
    "prod",
    "first",
    "last",
    "idxmin",
    "idxmax",
    "var_pop",
    "std_pop",
    "var",
    "std",
    "kurtosis",
    "skew",
    "boolor_agg",
    "booland_agg",
    "boolxor_agg",
    "bitor_agg",
    "bitand_agg",
    "bitxor_agg",
    "count_if",
    "listagg",
    "array_agg",
    "array_agg_distinct",
    "mode",
    "percentile_cont",
    "percentile_disc",
    "object_agg",
    "udf",
    "gen_udf",
    "window",
    "row_number",
    "min_row_number_filter",
    "rank",
    "dense_rank",
    "percent_rank",
    "cume_dist",
    "ntile",
    "ratio_to_report",
    "conditional_true_event",
    "conditional_change_event",
    "any_value",
    "grouping",
    "lead",
    "lag",
]

# This is just a list of the functions that can be used with
# bodo.utils.utils.ExtendedNamedAgg. Any function in this list
# should also be included in supported_agg_funcs
supported_extended_agg_funcs = [
    "array_agg",
    "array_agg_distinct",
    "listagg",
    "percentile_cont",
    "percentile_disc",
    "object_agg",
]

# Currently supported operations with transform
supported_transform_funcs = [
    "no_op",
    "sum",
    "count",
    "nunique",
    "median",
    "mean",
    "min",
    "max",
    "prod",
    "first",
    "last",
    "var",
    "std",
]
# Currently supported operations with window
supported_window_funcs = [
    "no_op",  # needed to ensure that 0 value isn't matched with any function
    "row_number",
    "min_row_number_filter",
    "rank",
    "dense_rank",
    "percent_rank",
    "cume_dist",
    "ntile",
    "ratio_to_report",
    "conditional_true_event",
    "conditional_change_event",
    "size",
    "count",
    "count_if",
    "var",
    "var_pop",
    "std",
    "std_pop",
    "mean",
    "any_value",
    "first",
    "last",
    "lead",
    "lag",
]


def get_agg_func(func_ir, func_name, rhs, series_type=None, typemap=None):
    """Returns specification of functions used by a groupby operation. It will
    either return:
    - A single function (case of a single function applied to all groupby
      input columns). For example: df.groupby("A").sum()
    - A list (element i of the list corresponds to a function(s) to apply
      to input column i)
        - The list can contain functions and list of functions, meaning
          that for each input column, a single function or list of
          functions can be applied.

    For pd.NamedAggs, rhs is a tuple containing the assigned column name,
    and the pd.NamedAgg/ExtendedAgg value. For example,
    in df.groupby("A").agg(New_B=pd.NamedAgg("B", "sum")), rhs is
    ("new_b", pd.NamedAgg("B", "sum")) (as the relevant numba types).

    For all other cases, it is the the full RHS of whatever operation is
    called on the groupby object.
    """
    if func_name == "no_op":
        raise BodoError("Unknown aggregation function used in groupby.")

    # FIXME: using float64 type as default to be compatible with old code
    # TODO: make groupby functions typed properly everywhere
    if series_type is None:
        series_type = SeriesType(types.float64)

    # Here we also set func.ncols_pre_shuffle and func.ncols_post_shuffle (see
    # below) for aggregation functions. These are the number of columns used
    # to compute the result of the function at runtime, before shuffle and
    # after shuffle, respectively. This is needed to generate code that invokes
    # udfs at runtime (see gen_update_cb, gen_combine_cb and gen_eval_cb),
    # to know which columns in the table received from C++ library correspond
    # to udfs and which to builtin functions
    if func_name in {"var_pop", "std_pop", "var", "std"}:
        func = pytypes.SimpleNamespace()
        func.ftype = func_name
        func.fname = func_name
        func.ncols_pre_shuffle = 3
        func.ncols_post_shuffle = 4
        return func
    elif func_name == "skew":
        func = pytypes.SimpleNamespace()
        func.ftype = func_name
        func.fname = func_name
        func.ncols_pre_shuffle = 4
        func.ncols_post_shuffle = 5
        return func
    elif func_name == "listagg":
        func = pytypes.SimpleNamespace()
        func.ftype = func_name
        func.fname = func_name
        func.ncols_pre_shuffle = 1
        func.ncols_post_shuffle = 1
        (
            func.listagg_sep,
            func.orderby,
            func.ascending,
            func.na_position_b,
        ) = handle_listagg_additional_args(func_ir, rhs)
        return func
    elif func_name in {"array_agg", "array_agg_distinct"}:
        func = pytypes.SimpleNamespace()
        func.ftype = func_name
        func.fname = func_name
        func.ncols_pre_shuffle = 1
        func.ncols_post_shuffle = 1
        (
            func.orderby,
            func.ascending,
            func.na_position_b,
        ) = handle_array_agg_additional_args(func_ir, rhs)
        return func
    elif func_name in {"percentile_cont", "percentile_disc"}:
        func = pytypes.SimpleNamespace()
        func.ftype = func_name
        func.fname = func_name
        func.ncols_pre_shuffle = 1
        func.ncols_post_shuffle = 1
        func.percentile = handle_percentile_additional_args(func_ir, rhs)
        return func
    elif func_name == "object_agg":
        func = pytypes.SimpleNamespace()
        func.ftype = func_name
        func.fname = func_name
        func.ncols_pre_shuffle = 1
        func.ncols_post_shuffle = 1
        func.key_col = handle_object_agg_additional_args(func_ir, rhs)
        return func
    elif func_name == "kurtosis":
        func = pytypes.SimpleNamespace()
        func.ftype = func_name
        func.fname = func_name
        func.ncols_pre_shuffle = 5
        func.ncols_post_shuffle = 6
        return func
    elif func_name == "mode":
        func = pytypes.SimpleNamespace()
        func.ftype = func_name
        func.fname = func_name
        func.ncols_pre_shuffle = 1
        func.ncols_post_shuffle = 1
        return func
    elif func_name in {"boolxor_agg"}:
        func = pytypes.SimpleNamespace()
        func.ftype = func_name
        func.fname = func_name
        func.ncols_pre_shuffle = 2
        func.ncols_post_shuffle = 3
        return func
    if func_name in {
        "first",
        "last",
        "boolor_agg",
        "booland_agg",
        "bitor_agg",
        "bitand_agg",
        "bitxor_agg",
        "count_if",
    }:
        # We don't have a function definition for first/last/boolor_agg/etc,
        # and it is not needed for the groupby C++ codepath, so we just use a dummy object.
        # Also NOTE: Series last and df.groupby.last() are different operations
        func = pytypes.SimpleNamespace()
        func.ftype = func_name
        func.fname = func_name
        func.ncols_pre_shuffle = 1
        func.ncols_post_shuffle = 1
        return func
    if func_name in {"idxmin", "idxmax"}:
        func = pytypes.SimpleNamespace()
        func.ftype = func_name
        func.fname = func_name
        func.ncols_pre_shuffle = 2
        func.ncols_post_shuffle = 2
        return func
    if func_name in list_cumulative or func_name in {
        "nunique",
        "shift",
        "head",
        "transform",
        "count",
        "sum",
        "max",
        "min",
        "size",
        "mean",
        "median",
        "prod",
        "ngroup",
    }:
        func = pytypes.SimpleNamespace()
        func.ftype = func_name
        func.fname = func_name
        func.ncols_pre_shuffle = 1
        func.ncols_post_shuffle = 1
        skip_na_data = True
        shift_periods_t = 1
        head_n = -1

        # Check for skipna/dropna argument, for the functions that support it.
        # (currently only "cumsum", "cumprod", "cummin", "cummax", and "nunique")
        if isinstance(rhs, ir.Expr):
            for erec in rhs.kws:
                # Type checking should be handled at the overload/bound_func level.
                # Any unknown kws at this stage should be naming the
                # output column.
                if func_name in list_cumulative:
                    if erec[0] == "skipna":
                        skip_na_data = guard(find_const, func_ir, erec[1])
                        if not isinstance(skip_na_data, bool):  # pragma: no cover
                            raise BodoError(
                                f"For {func_name} argument of skipna should be a boolean"
                            )
                if func_name == "nunique":
                    if erec[0] == "dropna":
                        skip_na_data = guard(find_const, func_ir, erec[1])
                        if not isinstance(skip_na_data, bool):  # pragma: no cover
                            raise BodoError(
                                "argument of dropna to nunique should be a boolean"
                            )

        # To handle shift(2) and shift(periods=2)
        if func_name == "shift" and (len(rhs.args) > 0 or len(rhs.kws) > 0):
            shift_periods_t = get_call_expr_arg(
                "shift",
                rhs.args,
                dict(rhs.kws),
                0,
                "periods",
                shift_periods_t,
            )
            shift_periods_t = guard(find_const, func_ir, shift_periods_t)
        # To handle head(2) and head(n=2)
        if func_name == "head":
            head_n = get_call_expr_arg(
                "head",
                rhs.args,
                dict(rhs.kws),
                0,
                "n",
                5,  # default value
            )
            # If we use the default value skip the constant search.
            if not isinstance(head_n, int):
                head_n = guard(find_const, func_ir, head_n)
            # Per Pandas docs: Does not work for negative values of n.
            if head_n < 0:
                raise BodoError(
                    f"groupby.{func_name} does not work with negative values."
                )
        func.skip_na_data = skip_na_data
        func.periods = shift_periods_t
        func.head_n = head_n
        if func_name == "transform":
            kws = dict(rhs.kws)
            func_var = get_call_expr_arg(func_name, rhs.args, kws, 0, "func", "")
            agg_func_typ = typemap[func_var.name]
            f_name = None
            if isinstance(agg_func_typ, str):
                f_name = agg_func_typ
            elif is_overload_constant_str(agg_func_typ):
                f_name = get_overload_const_str(agg_func_typ)
            elif bodo.utils.typing.is_builtin_function(agg_func_typ):
                # Builtin function case (e.g. df.groupby("B").transform(sum))
                f_name = bodo.utils.typing.get_builtin_function_name(agg_func_typ)
            if f_name not in bodo.ir.aggregate.supported_transform_funcs:
                raise BodoError(f"unsupported transform function {f_name}")
            # TODO: It could be user-defined
            func.transform_funcs = [supported_agg_funcs.index(f_name)]
        else:
            func.transform_funcs = [supported_agg_funcs.index("no_op")]
        return func
    if func_name == "window":
        func = pytypes.SimpleNamespace()
        func.ftype = func_name
        func.fname = func_name
        kws = dict(rhs.kws)
        func_vars = get_call_expr_arg(func_name, rhs.args, kws, 0, "func", "")
        if func_vars == ():  # pragma: no cover
            raise BodoError("window function requires a function")
        window_funcs = get_literal_value(typemap[func_vars.name])
        for window_func in window_funcs:
            window_func_name = window_func[0]
            if (
                window_func_name not in bodo.ir.aggregate.supported_window_funcs
            ):  # pragma: no cover
                raise BodoError(f"unsupported window function {window_func_name}")
        # Note: Orderby columns are in the gb_info
        ascending_var = get_call_expr_arg(func_name, rhs.args, kws, 2, "ascending", "")
        if ascending_var == "":  # pragma: no cover
            raise BodoError("window function requires an ascending argument")
        ascending = get_literal_value(typemap[ascending_var.name])
        na_position_var = get_call_expr_arg(
            func_name, rhs.args, kws, 3, "na_position", ""
        )
        if na_position_var == "":  # pragma: no cover
            raise BodoError("window function requires an na_position argument")
        na_position = get_literal_value(typemap[na_position_var.name])

        # Update the function information that may need be needed for the generated C++ code.
        # TODO: We may want to allow some update before shuffle for min_row_number_filter.
        func.ncols_pre_shuffle = 1
        func.ncols_post_shuffle = 1
        func.window_funcs = [
            supported_agg_funcs.index(window_func[0]) for window_func in window_funcs
        ]
        func.ascending = ascending
        func.na_position_b = [na_pos == "last" for na_pos in na_position]
        window_args = []
        n_input_cols = 0
        # window_funcs contains tuples in the form (function_name, arg0, arg1, ...)
        for window_func in window_funcs:
            func_name = window_func[0]
            func_args = window_func[1:]
            scalar_args, vector_args = bodo.hiframes.pd_groupby_ext.extract_window_args(
                func_name, func_args
            )
            window_args.extend(scalar_args)
            n_input_cols += len(vector_args)
        func.window_args = window_args
        func.n_input_cols = n_input_cols
        return func

    # agg case
    assert func_name in [
        "agg",
        "aggregate",
    ], f"Expected agg or aggregate function, found: {func_name}"

    # NOTE: assuming typemap is provided here
    assert typemap is not None
    kws = dict(rhs.kws)
    func_var = get_call_expr_arg(func_name, rhs.args, kws, 0, "func", "")
    # func is None in NamedAgg case
    if func_var == "":
        agg_func_typ = types.none
    else:
        agg_func_typ = typemap[func_var.name]

    # multi-function const dict case
    if is_overload_constant_dict(agg_func_typ):
        items = get_overload_constant_dict(agg_func_typ)
        # return a list, element i is function or list of functions to apply
        # to column i
        funcs = [
            get_agg_func_udf(func_ir, f_val, rhs, series_type, typemap)
            for f_val in items.values()
        ]
        return funcs

    # NamedAgg case
    if agg_func_typ == types.none:
        out_list = []

        for i in range(len(rhs.kws)):
            f_val_name = rhs.kws[i][1].name
            f_val_name_literal = get_literal_value(typemap[f_val_name])[1]
            namedaggCall = rhs.kws[i]

            udf = get_agg_func_udf(
                func_ir,
                f_val_name_literal,
                namedaggCall,
                series_type,
                typemap,
            )
            out_list.append(udf)
        return out_list

    # multi-function tuple case
    if isinstance(agg_func_typ, types.BaseTuple) or is_overload_constant_list(
        agg_func_typ
    ):
        funcs = []
        lambda_count = 0
        if is_overload_constant_list(agg_func_typ):
            # Lists find functions through their initial/literal values
            agg_func_vals = get_overload_const_list(agg_func_typ)
        else:
            # Tuples can find functions through their types
            agg_func_vals = agg_func_typ.types

        for t in agg_func_vals:
            if is_overload_constant_str(t):
                func_name = get_overload_const_str(t)
                funcs.append(
                    get_agg_func(func_ir, func_name, rhs, series_type, typemap)
                )
            else:
                assert typemap is not None, "typemap is required for agg UDF handling"
                func = _get_const_agg_func(t, func_ir)
                func.ftype = "udf"
                func.fname = _get_udf_name(func)
                # similar to _resolve_agg, TODO(ehsan): refactor
                # if tuple has lambdas they will be named <lambda_0>,
                # <lambda_1>, ... in output
                if func.fname == "<lambda>" and len(agg_func_vals) > 1:
                    func.fname = "<lambda_" + str(lambda_count) + ">"
                    lambda_count += 1
                funcs.append(func)
        # return a list containing one list of functions (applied to single
        # input column)
        return [funcs]

    # Single String use case
    if is_overload_constant_str(agg_func_typ):
        func_name = get_overload_const_str(agg_func_typ)
        return get_agg_func(func_ir, func_name, rhs, series_type, typemap)

    # Builtin function case (e.g. df.groupby("B").agg(sum))
    if bodo.utils.typing.is_builtin_function(agg_func_typ):
        func_name = bodo.utils.typing.get_builtin_function_name(agg_func_typ)
        return get_agg_func(func_ir, func_name, rhs, series_type, typemap)

    # typemap should be available for UDF case
    assert typemap is not None, "typemap is required for agg UDF handling"
    func = _get_const_agg_func(typemap[rhs.args[0].name], func_ir)
    func.ftype = "udf"
    func.fname = _get_udf_name(func)
    return func


def get_agg_func_udf(func_ir, f_val, rhs, series_type, typemap):
    """get udf value for agg call.

    For pd.NamedAggs, rhs is a tuple containing the assigned column name,
    and the pd.NamedAgg/ExtendedAgg value. For example,
    in df.groupby("A").agg(New_B=pd.NamedAgg("B", "sum")), rhs is
    ("new_b", pd.NamedAgg("B", "sum")) (as the relevant numba types).

    For all other cases, it is the the full RHS of whatever operation is
    called on the groupby object.
    """
    if isinstance(f_val, str):
        return get_agg_func(func_ir, f_val, rhs, series_type, typemap)
    if bodo.utils.typing.is_builtin_function(f_val):
        # Builtin function case (e.g. df.groupby("B").agg(sum))
        func_name = bodo.utils.typing.get_builtin_function_name(f_val)
        return get_agg_func(func_ir, func_name, rhs, series_type, typemap)
    if bodo.utils.typing.is_numpy_function(f_val):
        # Numpy function case (e.g. df.groupby("B").agg(np.var))
        func_name = bodo.hiframes.pd_groupby_ext.get_agg_name_for_numpy_method(
            bodo.utils.typing.get_builtin_function_name(f_val)
        )
        return get_agg_func(func_ir, func_name, rhs, series_type, typemap)
    if isinstance(f_val, (tuple, list)):
        lambda_count = 0
        out = []
        for f in f_val:
            func = get_agg_func_udf(func_ir, f, rhs, series_type, typemap)
            if func.fname == "<lambda>" and len(f_val) > 1:
                func.fname = f"<lambda_{lambda_count}>"
                lambda_count += 1
            out.append(func)
        return out
    else:
        assert is_expr(f_val, "make_function") or isinstance(
            f_val, (numba.core.registry.CPUDispatcher, types.Dispatcher)
        )
        assert typemap is not None, "typemap is required for agg UDF handling"
        func = _get_const_agg_func(f_val, func_ir)
        func.ftype = "udf"
        func.fname = _get_udf_name(func)
        return func


def _get_udf_name(func):
    """return name of UDF func"""
    code = func.code if hasattr(func, "code") else func.__code__
    f_name = code.co_name
    return f_name


def _get_const_agg_func(func_typ, func_ir):
    """get UDF function from its type. Wraps closures in functions."""
    agg_func = get_overload_const_func(func_typ, func_ir)

    # convert agg_func to a function if it is a make_function object
    # TODO: more robust handling, maybe reuse Numba's inliner code if possible
    if is_expr(agg_func, "make_function"):

        def agg_func_wrapper(A):  # pragma: no cover
            return A

        agg_func_wrapper.__code__ = agg_func.code
        agg_func = agg_func_wrapper
        return agg_func

    return agg_func


def handle_listagg_additional_args(func_ir, outcol_and_namedagg):  # pragma: no cover
    """
    Extract additional arguments for the listagg function.

    In this case, outcol_and_namedagg is a tuple containing the assigned column name,
    and the pd.NamedAgg/ExtendedAgg value. For example,
    in df.groupby("A").agg(New_B=pd.NamedAgg("B", "sum")), outcol_and_namedagg is
    ("new_b", pd.NamedAgg("B", "sum")) (as the relevant numba types).

    """
    additional_args_values = extract_extendedagg_additional_args_tuple(
        func_ir, outcol_and_namedagg
    )

    assert isinstance(additional_args_values[0], (ir.Global, ir.FreeVar, ir.Const)), (
        "Internal error in handle_listagg_additional_args: listagg_sep should be a constant value"
    )
    listagg_sep = additional_args_values[0].value
    orderby = get_const_or_build_tuple_of_consts(additional_args_values[1])
    ascending = list(get_const_or_build_tuple_of_consts(additional_args_values[2]))
    na_position_b = [
        na_pos == "last"
        for na_pos in get_const_or_build_tuple_of_consts(additional_args_values[3])
    ]
    return listagg_sep, orderby, ascending, na_position_b


def handle_array_agg_additional_args(func_ir, outcol_and_namedagg):  # pragma: no cover
    """
    Extract additional arguments for the array_agg function.

    In this case, outcol_and_namedagg is a tuple containing the assigned column name,
    and the pd.NamedAgg/ExtendedAgg value. For example,
    in df.groupby("A").agg(New_B=pd.NamedAgg("B", "sum")), outcol_and_namedagg is
    ("new_b", pd.NamedAgg("B", "sum")) (as the relevant numba types).
    """
    additional_args_values = extract_extendedagg_additional_args_tuple(
        func_ir, outcol_and_namedagg
    )

    orderby = get_const_or_build_tuple_of_consts(additional_args_values[0])
    ascending = list(get_const_or_build_tuple_of_consts(additional_args_values[1]))
    na_position_b = [
        na_pos == "last"
        for na_pos in get_const_or_build_tuple_of_consts(additional_args_values[2])
    ]
    return orderby, ascending, na_position_b


def handle_percentile_additional_args(func_ir, outcol_and_namedagg):  # pragma: no cover
    """
    Extract additional arguments for PERCENTILE_CONT/PERCENTILE_DISC.

    In this case, outcol_and_namedagg is a tuple containing the assigned column name,
    and the pd.NamedAgg/ExtendedAgg value. For example,
    in df.groupby("A").agg(New_B=pd.NamedAgg("B", "sum")), outcol_and_namedagg is
    ("new_b", pd.NamedAgg("B", "sum")) (as the relevant numba types).

    """
    additional_args_values = extract_extendedagg_additional_args_tuple(
        func_ir, outcol_and_namedagg
    )

    assert isinstance(additional_args_values[0], (ir.Global, ir.FreeVar, ir.Const)), (
        "Internal error in handle_percentile_additional_args: percentile should be a constant value"
    )
    return additional_args_values[0].value


def handle_object_agg_additional_args(func_ir, outcol_and_namedagg):  # pragma: no cover
    """
    Extract additional arguments for OBJECT_AGG.

    In this case, outcol_and_namedagg is a tuple containing the assigned column name,
    and the pd.NamedAgg/ExtendedAgg value. For example,
    in df.groupby("A").agg(New_B=pd.NamedAgg("B", "sum")), outcol_and_namedagg is
    ("new_b", pd.NamedAgg("B", "sum")) (as the relevant numba types).

    """
    additional_args_values = extract_extendedagg_additional_args_tuple(
        func_ir, outcol_and_namedagg
    )

    assert isinstance(additional_args_values[0], (ir.Global, ir.FreeVar, ir.Const)), (
        "Internal error in handle_object_agg_additional_args: key column should be a constant value"
    )
    return additional_args_values[0].value


def extract_extendedagg_additional_args_tuple(func_ir, outcol_and_namedagg):
    """
    Takes the output column + ExtendedNamedAgg call.
    extracts the values of the arguments in the additional args tuple,
    returning a list of variables. There will
    be a variable number of arguments in the additional args tuple, depending on which
    specific function is being called.
    """

    assert len(outcol_and_namedagg) == 2, (
        "bodo extended agg tuple should have 2 values (Output column name), and the additional arguments"
    )

    named_agg_args = guard(get_definition, func_ir, outcol_and_namedagg[1]).items
    extended_args_list = guard(get_definition, func_ir, named_agg_args[2]).items

    out_list = []
    for item in extended_args_list:
        out_list.append(guard(get_definition, func_ir, item))
    return out_list


# type(dtype) is called by np.full (used in agg_typer)
@infer_global(type)
class TypeDt64(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        if len(args) == 1 and isinstance(
            args[0], (types.NPDatetime, types.NPTimedelta)
        ):
            classty = types.DType(args[0])
            return signature(classty, *args)


class Aggregate(ir.Stmt):
    def __init__(
        self,
        df_out,
        df_in,
        key_names,
        gb_info_out,
        out_vars,
        in_vars,
        in_key_inds,
        df_in_type,
        out_type,
        input_has_index,
        same_index,
        return_key,
        loc,
        func_name,
        maintain_input_size,
        dropna,
        _num_shuffle_keys,
        _use_sql_rules,
    ):
        """IR node for groupby operations. It takes a logical table (input data can be
        in a table and arrays, or just arrays), and returns a logical table (table and
        arrays or just arrays). The actual computation is done in C++.

        Args:
            df_out (str): name of output variable, just for IR printing
            df_in (str): name of input variable, just for IR printing
            key_names (tuple(str)): names of key columns, just for IR printing
            gb_info_out (dict[int, tuple(tuple(int), func)]): out_col -> (tuple(in_col), func)
                map each output logical column number to the input logical number and
                function that creates it.
                Examples (["A", "B", "C"] input column names):
                For `df.groupby("A").agg({"B": "min", "C": "max"})`
                gb_info_out = {0: ((1,), min_func), 1: ((2,), max_func)}
                For `df.groupby("A").agg(
                   E=pd.NamedAgg(column="B", aggfunc=lambda A: A.sum()),
                   F=pd.NamedAgg(column="B", aggfunc="min"),
                )`
                gb_info_out = {0: ((1,), lambda_func), 1: ((1,), min_func)}
            out_vars (list(ir.Var)): list of output variables to assign
            in_vars (list(ir.Var)): list of variables with input data
            in_key_inds (list(int)): logical column number of keys in input (i.e. table
                column number in table format case or array index in list of variables
                in non-table case)
            df_in_type (types.Type): data type of input (always a dataframe)
            out_type (types.Type): data type of output (dataframe or Series)
            input_has_index (bool): whether input Index (last logical column) is used in
                computation.
                NOTE: MultiIndex isn't supported yet.
            same_index (bool): whether groupby returns the Index for output dataframe
                which has to match input Index.
                NOTE: MultiIndex isn't supported yet.
            return_key (bool): whether groupby returns key columns in output
            loc (ir.Loc): code location of the IR node
            func_name (str): name of the groupby function called (sum, agg, ...)
            maintain_input_size (bool): Is the output df the same length as the input
                df? Used for dead column elimination.
            dropna (bool): whether groupby drops NA values in computation.
            _num_shuffle_keys (int): How many of the keys should be used in the shuffle
                table to distribute table across ranks. This leads to shuffling by
                keys[:_num_shuffle_keys]. If _num_shuffle_keys == -1 then we use all
                of the keys, which is the common case.
            _use_sql_rules (bool): whether to use SQL rules for groupby aggregation
                or Pandas rules
        """
        self.df_out = df_out
        self.df_in = df_in
        self.key_names = key_names
        self.gb_info_out = gb_info_out
        self.out_vars = out_vars
        self.in_vars = in_vars
        self.in_key_inds = in_key_inds
        self.df_in_type = df_in_type
        self.out_type = out_type
        self.input_has_index = input_has_index
        self.same_index = same_index
        self.return_key = return_key
        self.loc = loc
        self.func_name = func_name
        self.maintain_input_size = maintain_input_size
        self.dropna = dropna
        self._num_shuffle_keys = _num_shuffle_keys
        self._use_sql_rules = _use_sql_rules
        # logical column number of dead inputs
        self.dead_in_inds = set()
        # logical column number of dead outputs
        self.dead_out_inds = set()

    def get_live_in_vars(self):
        """return input variables that are live (handles both table and non-table
        format cases)

        Returns:
            list(ir.Var): list of live variables
        """
        return [v for v in self.in_vars if v is not None]

    def get_live_out_vars(self):
        """return output variables that are live (table, possibly key arrays, possibly
        Index array)

        Returns:
            list(ir.Var): list of live output variables
        """
        return [v for v in self.out_vars if v is not None]

    @property
    def is_in_table_format(self):
        """True if input dataframe is in table format"""
        return self.df_in_type.is_table_format

    @property
    def n_in_table_arrays(self):
        """number of logical input columns that are part of the input table.
        Returns 1 if input is not in table format to simplify computations.
        """
        return len(self.df_in_type.columns) if self.df_in_type.is_table_format else 1

    @property
    def n_in_cols(self):
        """Number of logical input columns"""
        # table columns plus extra arrays
        return self.n_in_table_arrays + len(self.in_vars) - 1

    @property
    def in_col_types(self):
        """list of data types of all logical input columns"""
        return list(self.df_in_type.data) + list(
            get_index_data_arr_types(self.df_in_type.index)
        )

    @property
    def is_output_table(self):
        """True if output is in table format. We always use table format for dataframe
        output, but a single array for Series output.
        """
        return not isinstance(self.out_type, SeriesType)

    @property
    def n_out_table_arrays(self):
        """number of logical output columns that are part of the output table.
        Returns 1 if output is not in table format (Series case) to simplify
        computations.
        """
        return (
            len(self.out_type.table_type.arr_types)
            if not isinstance(self.out_type, SeriesType)
            else 1
        )

    @property
    def n_out_cols(self):
        """Number of logical output columns"""
        # table columns plus extra arrays
        return self.n_out_table_arrays + len(self.out_vars) - 1

    @property
    def out_col_types(self):
        """list of data types of all logical output columns"""
        data_col_types = (
            [self.out_type.data]
            if isinstance(self.out_type, SeriesType)
            else list(self.out_type.table_type.arr_types)
        )
        index_col_types = list(get_index_data_arr_types(self.out_type.index))
        return data_col_types + index_col_types

    def update_dead_col_info(self):
        """updates all internal data structures when there are more output dead columns
        added in dead_out_inds.
        gb_info_out and dead_in_inds need updated and dead input variables need
        to be set to None.
        """
        # remove dead output columns from gb_info_out
        for col_no in self.dead_out_inds:
            self.gb_info_out.pop(col_no, None)

        # Map live inputs
        live_in_inds = set(self.in_key_inds)
        # Index column (which is last) is not passed if input_as_index=False
        if self.input_has_index:
            live_in_inds.add(self.n_in_cols - 1)
        else:
            self.dead_in_inds.add(self.n_in_cols - 1)
            self.dead_out_inds.add(self.n_out_cols - 1)

        # remove dead inputs (have no live output)
        for in_cols, _ in self.gb_info_out.values():
            live_in_inds.update(in_cols)
        dead_in_inds = set(range(self.n_in_cols)) - live_in_inds
        self.dead_in_inds.update(dead_in_inds)

        # update input variables
        if self.is_in_table_format:
            if not (set(range(self.n_in_table_arrays)) - self.dead_in_inds):
                self.in_vars[0] = None
            for i in range(1, len(self.in_vars)):
                col_no = self.n_in_table_arrays + i - 1
                if col_no in self.dead_in_inds:
                    self.in_vars[i] = None
        else:
            for i in range(len(self.in_vars)):
                if i in self.dead_in_inds:
                    self.in_vars[i] = None

    def __repr__(self):  # pragma: no cover
        in_cols = ", ".join(v.name for v in self.get_live_in_vars())
        df_in_str = f"{self.df_in}{{{in_cols}}}"
        out_cols = ", ".join(v.name for v in self.get_live_out_vars())
        df_out_str = f"{self.df_out}{{{out_cols}}}"
        return f"Groupby (keys: {self.key_names} {self.in_key_inds}): {df_in_str} {df_out_str}"


def aggregate_usedefs(aggregate_node, use_set=None, def_set=None):
    """use/def analysis extension for Aggregate IR node

    Args:
        aggregate_node (Aggregate): Aggregate IR node
        use_set (set(str), optional): Existing set of used variables. Defaults to None.
        def_set (set(str), optional): Existing set of defined variables. Defaults to
            None.

    Returns:
        namedtuple('use_defs_result', 'usemap,defmap'): use/def sets
    """
    if use_set is None:
        use_set = set()
    if def_set is None:
        def_set = set()

    # input table/arrays are used
    use_set.update({v.name for v in aggregate_node.get_live_in_vars()})

    # output table/arrays are defined
    def_set.update({v.name for v in aggregate_node.get_live_out_vars()})

    return numba.core.analysis._use_defs_result(usemap=use_set, defmap=def_set)


numba.core.analysis.ir_extension_usedefs[Aggregate] = aggregate_usedefs


def remove_dead_aggregate(
    agg_node, lives_no_aliases, lives, arg_aliases, alias_map, func_ir, typemap
):
    """Dead code elimination for Aggregate IR node

    Args:
        agg_node (Aggregate): Aggregate IR node
        lives_no_aliases (set(str)): live variable names without their aliases
        lives (set(str)): live variable names with their aliases
        arg_aliases (set(str)): variables that are function arguments or alias them
        alias_map (dict(str, set(str))): mapping of variables names and their aliases
        func_ir (FunctionIR): full function IR
        typemap (dict(str, types.Type)): typemap of variables

    Returns:
        (Aggregate, optional): Aggregate IR node if not fully dead, None otherwise
    """

    # remove dead table and non-table variables
    out_data_var = agg_node.out_vars[0]
    if out_data_var is not None and out_data_var.name not in lives:
        agg_node.out_vars[0] = None
        # output is table in dataframe case but a single array in Series case
        if agg_node.is_output_table:
            dead_cols = set(range(agg_node.n_out_table_arrays))
            agg_node.dead_out_inds.update(dead_cols)
        else:
            agg_node.dead_out_inds.add(0)

    for i in range(1, len(agg_node.out_vars)):
        v = agg_node.out_vars[i]
        if v is not None and v.name not in lives:
            agg_node.out_vars[i] = None
            col_no = agg_node.n_out_table_arrays + i - 1
            agg_node.dead_out_inds.add(col_no)

    # remove empty aggregate node
    if all(v is None for v in agg_node.out_vars):
        return None

    agg_node.update_dead_col_info()

    return agg_node


ir_utils.remove_dead_extensions[Aggregate] = remove_dead_aggregate


def get_copies_aggregate(aggregate_node, typemap):
    """Aggregate IR node extension for variable copy analysis

    Args:
        aggregate_node (Aggregate): Aggregate IR node
        typemap (dict(str, ir.Var)): typemap of variables

    Returns:
        tuple(set(str), set(str)): set of copies generated or killed
    """
    # Aggregate doesn't generate copies, it just kills the output columns
    kill_set = {v.name for v in aggregate_node.get_live_out_vars()}
    return set(), kill_set


ir_utils.copy_propagate_extensions[Aggregate] = get_copies_aggregate


def apply_copies_aggregate(
    aggregate_node, var_dict, name_var_table, typemap, calltypes, save_copies
):
    """Aggregate IR node extension for applying variable copies pass

    Args:
        aggregate_node (Aggregate): Aggregate IR node
        var_dict (dict(str, ir.Var)): dictionary of variables to replace
        name_var_table (dict(str, ir.Var)): map variable name to its ir.Var object
        typemap (dict(str, ir.Var)): typemap of variables
        calltypes (dict[ir.Inst, Signature]): signature of callable nodes
        save_copies (list(tuple(str, ir.Var))): copies that were applied
    """
    for i in range(len(aggregate_node.in_vars)):
        if aggregate_node.in_vars[i] is not None:
            aggregate_node.in_vars[i] = replace_vars_inner(
                aggregate_node.in_vars[i], var_dict
            )

    for i in range(len(aggregate_node.out_vars)):
        if aggregate_node.out_vars[i] is not None:
            aggregate_node.out_vars[i] = replace_vars_inner(
                aggregate_node.out_vars[i], var_dict
            )


ir_utils.apply_copy_propagate_extensions[Aggregate] = apply_copies_aggregate


def visit_vars_aggregate(aggregate_node, callback, cbdata):
    """Aggregate IR node extension for visiting variables pass

    Args:
        aggregate_node (Aggregate): Aggregate IR node
        callback (function): callback to call on each variable (just passed along here)
        cbdata (object): data to pass to callback (just passed along here)
    """

    for i in range(len(aggregate_node.in_vars)):
        if aggregate_node.in_vars[i] is not None:
            aggregate_node.in_vars[i] = visit_vars_inner(
                aggregate_node.in_vars[i], callback, cbdata
            )

    for i in range(len(aggregate_node.out_vars)):
        if aggregate_node.out_vars[i] is not None:
            aggregate_node.out_vars[i] = visit_vars_inner(
                aggregate_node.out_vars[i], callback, cbdata
            )


# add call to visit aggregate variable
ir_utils.visit_vars_extensions[Aggregate] = visit_vars_aggregate


def aggregate_array_analysis(aggregate_node, equiv_set, typemap, array_analysis):
    """Array analysis for Aggregate IR node. Input arrays have the same size. Output
    arrays have the same size as well.
    But outputs are not the same size as inputs necessarily.

    Args:
        aggregate_node (ir.Aggregate): input Aggregate node
        equiv_set (SymbolicEquivSet): equivalence set object of Numba array analysis
        typemap (dict[str, types.Type]): typemap from analysis pass
        array_analysis (ArrayAnalysis): array analysis object for the pass

    Returns:
        tuple(list(ir.Stmt), list(ir.Stmt)): lists of IR statements to add to IR before
        this node and after this node.
    """

    # arrays of input df have same size in first dimension
    all_shapes = []
    for col_var in aggregate_node.get_live_in_vars():
        col_shape = equiv_set.get_shape(col_var)
        if col_shape is not None:
            all_shapes.append(col_shape[0])

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    # create correlations for output arrays
    # arrays of output df have same size in first dimension
    # gen size variables for output columns
    post = []
    all_shapes = []

    for col_var in aggregate_node.get_live_out_vars():
        typ = typemap[col_var.name]
        shape = array_analysis._gen_shape_call(equiv_set, col_var, typ.ndim, None, post)
        equiv_set.insert_equiv(col_var, shape)
        all_shapes.append(shape[0])
        equiv_set.define(col_var, set())

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    return [], post


numba.parfors.array_analysis.array_analysis_extensions[Aggregate] = (
    aggregate_array_analysis
)


def aggregate_distributed_analysis(aggregate_node, array_dists):
    """Distributed analysis for Aggregate IR node. Inputs and outputs have the same
    distribution, except that output of 1D is 1D_Var due to groupby/shuffling.

    Args:
        aggregate_node (Aggregate): Aggregate IR node
        array_dists (dict[str, Distribution]): distributions of arrays in the IR
            (variable name -> Distribution)
    """

    in_arrs = aggregate_node.get_live_in_vars()
    out_arrs = aggregate_node.get_live_out_vars()
    # input columns have same distribution
    in_dist = Distribution.OneD
    for col_var in in_arrs:
        in_dist = Distribution(min(in_dist.value, array_dists[col_var.name].value))

    # output is 1D_Var due to groupby/shuffle, has to meet input dist
    out_dist = Distribution(min(in_dist.value, Distribution.OneD_Var.value))

    # cumulative/transform/window functions don't aggregate values and have a reverse
    # shuffle, so output chunk size is the same as input size (can stay 1D)
    if aggregate_node.maintain_input_size:
        out_dist = in_dist

    for col_var in out_arrs:
        if col_var.name in array_dists:
            out_dist = Distribution(
                min(out_dist.value, array_dists[col_var.name].value)
            )

    # output can cause input REP
    if out_dist == Distribution.REP:
        in_dist = out_dist

    # set dists
    for col_var in in_arrs:
        array_dists[col_var.name] = in_dist

    for col_var in out_arrs:
        array_dists[col_var.name] = out_dist


distributed_analysis.distributed_analysis_extensions[Aggregate] = (
    aggregate_distributed_analysis
)


def build_agg_definitions(agg_node, definitions=None):
    """Aggregate IR node extension for building varibale definitions pass

    Args:
        agg_node (Aggregate): Aggregate IR node
        definitions (defaultdict(list), optional): Existing definitions list. Defaults
            to None.

    Returns:
        defaultdict(list): updated definitions
    """

    if definitions is None:
        definitions = defaultdict(list)

    # output arrays are defined
    for col_var in agg_node.get_live_out_vars():
        definitions[col_var.name].append(agg_node)

    return definitions


ir_utils.build_defs_extensions[Aggregate] = build_agg_definitions


def __update_redvars():
    pass


@infer_global(__update_redvars)
class UpdateDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        return signature(types.void, *args)


def __combine_redvars():
    pass


@infer_global(__combine_redvars)
class CombineDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        return signature(types.void, *args)


def __eval_res():
    pass


@infer_global(__eval_res)
class EvalDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        # takes the output array as first argument to know the output dtype
        return signature(args[0].dtype, *args)


def agg_distributed_run(
    agg_node, array_dists, typemap, calltypes, typingctx, targetctx
):
    """lowers Aggregate IR node to regular IR nodes. Uses the C++ implementation of
    groupby operations.

    Args:
        agg_node (Aggregate): Aggregate IR node to lower
        array_dists (dict(str, Distribution)): distribution of arrays
        typemap (dict(str, ir.Var)): typemap of variables
        calltypes (dict[ir.Inst, Signature]): signature of callable nodes
        typingctx (typing.Context): typing context for compiler pipeline
        targetctx (cpu.CPUContext): target context for compiler pipeline

    Returns:
        list(ir.Stmt): list of IR nodes that implement the input Aggregate IR node
    """
    parallel = False
    live_in_vars = agg_node.get_live_in_vars()
    live_out_vars = agg_node.get_live_out_vars()
    if array_dists is not None:
        parallel = True
        for v in live_in_vars + live_out_vars:
            if (
                array_dists[v.name] != distributed_pass.Distribution.OneD
                and array_dists[v.name] != distributed_pass.Distribution.OneD_Var
            ):
                parallel = False

    out_col_typs = agg_node.out_col_types

    # get column types
    # Type of input columns in the same order as passed to C++ and can include
    # repetition. C++ receives one input column for each (in_col,func) pair
    # and the same input column might not necessarily appear in consecutive
    # positions in that list (see NamedAgg examples)
    in_col_typs = []
    funcs = []
    func_out_types = []
    # See comment about use of gb_info_out in gen_top_level_agg_func
    # when laying out input columns and functions for C++
    for out_col, (in_cols, func) in agg_node.gb_info_out.items():
        for in_col in in_cols:
            t = agg_node.in_col_types[in_col]
            in_col_typs.append(t)
        funcs.append(func)
        func_out_types.append(out_col_typs[out_col])

    glbs = {
        "bodo": bodo,
        "np": np,
        "dt64_dtype": np.dtype("datetime64[ns]"),
        "td64_dtype": np.dtype("timedelta64[ns]"),
    }
    # TODO: Support for Categories not known at compile time
    for i, in_col_typ in enumerate(in_col_typs):
        if isinstance(in_col_typ, bodo.CategoricalArrayType):
            glbs.update({f"in_cat_dtype_{i}": in_col_typ})

    for i, out_col_typ in enumerate(out_col_typs):
        if isinstance(out_col_typ, bodo.CategoricalArrayType):
            glbs.update({f"out_cat_dtype_{i}": out_col_typ})

    udf_func_struct = get_udf_func_struct(
        funcs,
        in_col_typs,
        typingctx,
        targetctx,
    )

    out_var_types = [
        typemap[v.name] if v is not None else types.none for v in agg_node.out_vars
    ]

    func_text, f_glbs = gen_top_level_agg_func(
        agg_node,
        in_col_typs,
        out_col_typs,
        func_out_types,
        parallel,
        udf_func_struct,
        out_var_types,
        typemap,
    )
    glbs.update(f_glbs)
    glbs.update(
        {
            "pd": pd,
            "pre_alloc_string_array": pre_alloc_string_array,
            "pre_alloc_binary_array": pre_alloc_binary_array,
            "pre_alloc_array_item_array": pre_alloc_array_item_array,
            "string_array_type": string_array_type,
            "alloc_decimal_array": alloc_decimal_array,
            "array_to_info": array_to_info,
            "arr_info_list_to_table": arr_info_list_to_table,
            "coerce_to_array": bodo.utils.conversion.coerce_to_array,
            "groupby_and_aggregate": groupby_and_aggregate,
            "array_from_cpp_table": array_from_cpp_table,
            "delete_info": delete_info,
            "add_agg_cfunc_sym": add_agg_cfunc_sym,
            "get_agg_udf_addr": get_agg_udf_addr,
            "delete_table": delete_table,
            "decode_if_dict_array": decode_if_dict_array,
            "set_table_data": bodo.hiframes.table.set_table_data,
            "get_table_data": bodo.hiframes.table.get_table_data,
            "out_typs": out_col_typs,
        }
    )
    if udf_func_struct is not None:
        if udf_func_struct.regular_udfs:
            glbs.update(
                {
                    "__update_redvars": udf_func_struct.update_all_func,
                    "__init_func": udf_func_struct.init_func,
                    "__combine_redvars": udf_func_struct.combine_all_func,
                    "__eval_res": udf_func_struct.eval_all_func,
                    "cpp_cb_update": udf_func_struct.regular_udf_cfuncs[0],
                    "cpp_cb_combine": udf_func_struct.regular_udf_cfuncs[1],
                    "cpp_cb_eval": udf_func_struct.regular_udf_cfuncs[2],
                }
            )
        if udf_func_struct.general_udfs:
            glbs.update({"cpp_cb_general": udf_func_struct.general_udf_cfunc})

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    top_level_func = loc_vars["agg_top"]

    f_block = compile_to_numba_ir(
        top_level_func,
        glbs,
        typingctx=typingctx,
        targetctx=targetctx,
        arg_typs=tuple(typemap[v.name] for v in live_in_vars),
        typemap=typemap,
        calltypes=calltypes,
    ).blocks.popitem()[1]

    replace_arg_nodes(f_block, live_in_vars)

    # get return value from cast node, the last node before cast isn't output assignment
    ret_var = f_block.body[-2].value.value
    nodes = f_block.body[:-2]

    for i, v in enumerate(live_out_vars):
        gen_getitem(v, ret_var, i, calltypes, nodes)

    return nodes


distributed_pass.distributed_run_extensions[Aggregate] = agg_distributed_run


# TODO: Use `bodo.utils.utils.alloc_type` instead if possible
def _gen_dummy_alloc(t, colnum=0, is_input=False):
    """generate dummy allocation text for type `t`, used for creating dummy arrays that
    just pass data type to functions.
    """
    if isinstance(t, IntegerArrayType):
        int_typ_name = IntDtype(t.dtype).name
        assert int_typ_name.endswith("Dtype()")
        int_typ_name = int_typ_name[:-7]  # remove trailing "Dtype()"
        return f"bodo.hiframes.pd_series_ext.get_series_data(pd.Series([1], dtype='{int_typ_name}'))"
    elif isinstance(t, FloatingArrayType):
        # Float32 or Float64
        float_typ_name = str(t.dtype).capitalize()
        return f"bodo.hiframes.pd_series_ext.get_series_data(pd.Series([1.0], dtype='{float_typ_name}'))"
    elif isinstance(t, BooleanArrayType):
        return "bodo.libs.bool_arr_ext.alloc_bool_array(0)"
    elif isinstance(t, StringArrayType):
        return "pre_alloc_string_array(1, 1)"
    elif t == bodo.dict_str_arr_type:
        return "bodo.libs.dict_arr_ext.init_dict_arr(pre_alloc_string_array(1, 1), bodo.libs.int_arr_ext.alloc_int_array(1, np.int32), False, False, None)"
    elif isinstance(t, BinaryArrayType):
        return "pre_alloc_binary_array(1, 1)"
    elif t == ArrayItemArrayType(string_array_type):
        return "pre_alloc_array_item_array(1, (1,), 1, string_array_type)"
    elif isinstance(t, DecimalArrayType):
        return f"alloc_decimal_array(1, {t.precision}, {t.scale})"
    elif isinstance(t, DatetimeDateArrayType):
        return "bodo.hiframes.datetime_date_ext.init_datetime_date_array(np.empty(1, np.int64), np.empty(1, np.uint8))"
    elif isinstance(t, bodo.CategoricalArrayType):
        if t.dtype.categories is None:
            raise BodoError(
                "Groupby agg operations on Categorical types require constant categories"
            )
        # TODO: Support categories that aren't known at compile time
        starter = "in" if is_input else "out"
        return f"bodo.utils.utils.alloc_type(1, {starter}_cat_dtype_{colnum})"
    else:
        return f"np.empty(1, {_get_np_dtype(t.dtype)})"


def _get_np_dtype(t):
    if t == types.bool_:
        return "np.bool_"
    if t == types.NPDatetime("ns"):
        return "dt64_dtype"
    if t == types.NPTimedelta("ns"):
        return "td64_dtype"
    return f"np.{t}"


def gen_update_cb(
    udf_func_struct,
    allfuncs,
    n_keys,
    data_in_typs_,
    do_combine,
    func_idx_to_in_col,
    label_suffix,
):
    """
    Generates a Python function (to be compiled into a numba cfunc) which
    does the "update" step of an agg operation. The code is for a specific
    groupby.agg(). The update step performs the initial local aggregation.
    """
    red_var_typs = udf_func_struct.var_typs
    n_red_vars = len(red_var_typs)

    func_text = f"def bodo_gb_udf_update_local{label_suffix}(in_table, out_table, row_to_group):\n"
    func_text += "    if is_null_pointer(in_table):\n"  # this is dummy call
    func_text += "        return\n"

    # get redvars data types
    func_text += "    data_redvar_dummy = ({}{})\n".format(
        ",".join([f"np.empty(1, {_get_np_dtype(t)})" for t in red_var_typs]),
        "," if len(red_var_typs) == 1 else "",
    )

    # calculate the offsets of redvars of udfs in the table received from C++.
    # Note that the table can contain a mix of columns from udfs and builtins
    col_offset = n_keys  # keys are the first columns in the table, skip them
    in_col_offsets = []
    redvar_offsets = []  # offsets of redvars in the table received from C++
    data_in_typs = []
    if do_combine:
        # the groupby will do a combine after update and shuffle. This means
        # the table we are receiving is pre_shuffle
        for i, f in enumerate(allfuncs):
            if f.ftype != "udf":
                col_offset += f.ncols_pre_shuffle
            else:
                redvar_offsets += list(range(col_offset, col_offset + f.n_redvars))
                col_offset += f.n_redvars
                data_in_typs.append(data_in_typs_[func_idx_to_in_col[i]])
                in_col_offsets.append(func_idx_to_in_col[i] + n_keys)
    else:
        # a combine won't be done in this case (which means either a shuffle
        # was done before update, or no shuffle is necessary, so the table
        # we are getting is post_shuffle table
        for i, f in enumerate(allfuncs):
            if f.ftype != "udf":
                col_offset += f.ncols_post_shuffle
            else:
                # udfs in post_shuffle table have one column for output plus
                # redvars columns
                redvar_offsets += list(
                    range(col_offset + 1, col_offset + 1 + f.n_redvars)
                )
                col_offset += f.n_redvars + 1
                data_in_typs.append(data_in_typs_[func_idx_to_in_col[i]])
                in_col_offsets.append(func_idx_to_in_col[i] + n_keys)
    assert len(redvar_offsets) == n_red_vars, (
        "Internal error: redvar offsets lenth does not match number of redvars"
    )

    # get input data types
    n_data_cols = len(data_in_typs)
    data_in_dummy_text = []
    for i, t in enumerate(data_in_typs):
        data_in_dummy_text.append(_gen_dummy_alloc(t, i, True))
    func_text += "    data_in_dummy = ({}{})\n".format(
        ",".join(data_in_dummy_text), "," if len(data_in_typs) == 1 else ""
    )

    func_text += "\n    # initialize redvar cols\n"
    func_text += "    init_vals = __init_func()\n"
    for i in range(n_red_vars):
        func_text += f"    redvar_arr_{i} = array_from_cpp_table(out_table, {redvar_offsets[i]}, data_redvar_dummy[{i}])\n"
        func_text += f"    redvar_arr_{i}.fill(init_vals[{i}])\n"
    func_text += "    redvars = ({}{})\n".format(
        ",".join([f"redvar_arr_{i}" for i in range(n_red_vars)]),
        "," if n_red_vars == 1 else "",
    )

    func_text += "\n"
    for i in range(n_data_cols):
        func_text += f"    data_in_{i} = array_from_cpp_table(in_table, {in_col_offsets[i]}, data_in_dummy[{i}])\n"
    func_text += "    data_in = ({}{})\n".format(
        ",".join([f"data_in_{i}" for i in range(n_data_cols)]),
        "," if n_data_cols == 1 else "",
    )

    func_text += "\n"
    func_text += "    for i in range(len(data_in_0)):\n"
    func_text += "        w_ind = row_to_group[i]\n"
    func_text += "        if w_ind != -1:\n"
    func_text += "            __update_redvars(redvars, data_in, w_ind, i)\n"

    loc_vars = {}
    exec(
        func_text,
        {
            "bodo": bodo,
            "np": np,
            "pd": pd,
            "array_from_cpp_table": array_from_cpp_table,
            "pre_alloc_string_array": pre_alloc_string_array,
            "__init_func": udf_func_struct.init_func,
            "__update_redvars": udf_func_struct.update_all_func,
            "is_null_pointer": is_null_pointer,
            "dt64_dtype": np.dtype("datetime64[ns]"),
            "td64_dtype": np.dtype("timedelta64[ns]"),
        },
        loc_vars,
    )
    return loc_vars[f"bodo_gb_udf_update_local{label_suffix}"]


def gen_combine_cb(udf_func_struct, allfuncs, n_keys, label_suffix):
    """
    Generates a Python function (to be compiled into a numba cfunc) which
    does the "combine" step of an agg operation. The code is for a specific
    groupby.agg(). The combine step combines the received aggregated data from
    other processes.
    """
    red_var_typs = udf_func_struct.var_typs
    n_red_vars = len(red_var_typs)

    func_text = (
        f"def bodo_gb_udf_combine{label_suffix}(in_table, out_table, row_to_group):\n"
    )
    func_text += "    if is_null_pointer(in_table):\n"  # this is dummy call
    func_text += "        return\n"

    # get redvars data types
    func_text += "    data_redvar_dummy = ({}{})\n".format(
        ",".join([f"np.empty(1, {_get_np_dtype(t)})" for t in red_var_typs]),
        "," if len(red_var_typs) == 1 else "",
    )

    # calculate the offsets of redvars of udfs in the tables received from C++.
    # Note that the tables can contain a mix of columns from udfs and builtins.
    # The input table is the pre_shuffle table right after shuffling (so has
    # the same specs as pre_shuffle). post_shuffle is the output table from
    # combine operation
    col_offset_in = n_keys
    col_offset_out = n_keys
    redvar_offsets_in = []  # offsets of udf redvars in the table received from C++
    redvar_offsets_out = []  # offsets of udf redvars in the table received from C++
    for f in allfuncs:
        if f.ftype != "udf":
            col_offset_in += f.ncols_pre_shuffle
            col_offset_out += f.ncols_post_shuffle
        else:
            redvar_offsets_in += list(range(col_offset_in, col_offset_in + f.n_redvars))
            # udfs in post_shuffle table have one column for output plus
            # redvars columns
            redvar_offsets_out += list(
                range(col_offset_out + 1, col_offset_out + 1 + f.n_redvars)
            )
            col_offset_in += f.n_redvars
            col_offset_out += 1 + f.n_redvars
    assert len(redvar_offsets_in) == n_red_vars

    func_text += "\n    # initialize redvar cols\n"
    func_text += "    init_vals = __init_func()\n"
    for i in range(n_red_vars):
        func_text += f"    redvar_arr_{i} = array_from_cpp_table(out_table, {redvar_offsets_out[i]}, data_redvar_dummy[{i}])\n"
        func_text += f"    redvar_arr_{i}.fill(init_vals[{i}])\n"
    func_text += "    redvars = ({}{})\n".format(
        ",".join([f"redvar_arr_{i}" for i in range(n_red_vars)]),
        "," if n_red_vars == 1 else "",
    )

    func_text += "\n"
    for i in range(n_red_vars):
        func_text += f"    recv_redvar_arr_{i} = array_from_cpp_table(in_table, {redvar_offsets_in[i]}, data_redvar_dummy[{i}])\n"
    func_text += "    recv_redvars = ({}{})\n".format(
        ",".join([f"recv_redvar_arr_{i}" for i in range(n_red_vars)]),
        "," if n_red_vars == 1 else "",
    )

    func_text += "\n"
    if n_red_vars:  # if there is a parfor
        func_text += "    for i in range(len(recv_redvar_arr_0)):\n"
        func_text += "        w_ind = row_to_group[i]\n"
        func_text += "        __combine_redvars(redvars, recv_redvars, w_ind, i)\n"

    loc_vars = {}
    exec(
        func_text,
        {
            "np": np,
            "array_from_cpp_table": array_from_cpp_table,
            "__init_func": udf_func_struct.init_func,
            "__combine_redvars": udf_func_struct.combine_all_func,
            "is_null_pointer": is_null_pointer,
            "dt64_dtype": np.dtype("datetime64[ns]"),
            "td64_dtype": np.dtype("timedelta64[ns]"),
        },
        loc_vars,
    )
    return loc_vars[f"bodo_gb_udf_combine{label_suffix}"]


def gen_eval_cb(udf_func_struct, allfuncs, n_keys, out_data_typs_, label_suffix):
    """
    Generates a Python function (to be compiled into a numba cfunc) which
    does the "eval" step of an agg operation. The code is for a specific
    groupby.agg(). The eval step writes the final result to the output columns
    for each group.
    """
    red_var_typs = udf_func_struct.var_typs
    n_red_vars = len(red_var_typs)

    # calculate the offsets of redvars and output columns of udfs in the table
    # received from C++. Note that the table can contain a mix of columns from
    # udfs and builtins
    col_offset = n_keys
    redvar_offsets = []  # offsets of redvars in the table received from C++
    data_out_offsets = []  # offsets of data col in the table received from C++
    out_data_typs = []
    for i, f in enumerate(allfuncs):
        if f.ftype != "udf":
            col_offset += f.ncols_post_shuffle
        else:
            # udfs in post_shuffle table have one column for output plus
            # redvars columns
            data_out_offsets.append(col_offset)
            redvar_offsets += list(range(col_offset + 1, col_offset + 1 + f.n_redvars))
            col_offset += 1 + f.n_redvars
            out_data_typs.append(out_data_typs_[i])
    assert len(redvar_offsets) == n_red_vars
    n_data_cols = len(out_data_typs)

    func_text = f"def bodo_gb_udf_eval{label_suffix}(table):\n"
    func_text += "    if is_null_pointer(table):\n"  # this is dummy call
    func_text += "        return\n"

    func_text += "    data_redvar_dummy = ({}{})\n".format(
        ",".join([f"np.empty(1, {_get_np_dtype(t)})" for t in red_var_typs]),
        "," if len(red_var_typs) == 1 else "",
    )

    func_text += "    out_data_dummy = ({}{})\n".format(
        ",".join([f"np.empty(1, {_get_np_dtype(t.dtype)})" for t in out_data_typs]),
        "," if len(out_data_typs) == 1 else "",
    )

    for i in range(n_red_vars):
        func_text += f"    redvar_arr_{i} = array_from_cpp_table(table, {redvar_offsets[i]}, data_redvar_dummy[{i}])\n"
    func_text += "    redvars = ({}{})\n".format(
        ",".join([f"redvar_arr_{i}" for i in range(n_red_vars)]),
        "," if n_red_vars == 1 else "",
    )

    func_text += "\n"
    for i in range(n_data_cols):
        func_text += f"    data_out_{i} = array_from_cpp_table(table, {data_out_offsets[i]}, out_data_dummy[{i}])\n"
    func_text += "    data_out = ({}{})\n".format(
        ",".join([f"data_out_{i}" for i in range(n_data_cols)]),
        "," if n_data_cols == 1 else "",
    )

    func_text += "\n"
    func_text += "    for i in range(len(data_out_0)):\n"
    func_text += "        __eval_res(redvars, data_out, i)\n"

    loc_vars = {}
    exec(
        func_text,
        {
            "np": np,
            "array_from_cpp_table": array_from_cpp_table,
            "__eval_res": udf_func_struct.eval_all_func,
            "is_null_pointer": is_null_pointer,
            "dt64_dtype": np.dtype("datetime64[ns]"),
            "td64_dtype": np.dtype("timedelta64[ns]"),
        },
        loc_vars,
    )
    return loc_vars[f"bodo_gb_udf_eval{label_suffix}"]


def gen_general_udf_cb(
    udf_func_struct,
    allfuncs,
    n_keys,
    in_col_typs,
    out_col_typs,
    func_idx_to_in_col,
    label_suffix,
):
    """
    Generates a Python function and compiles it to a numba cfunc, which
    applies all general UDFs in a groupby operation. The code is for a specific
    groupby.agg().
    """
    col_offset = n_keys
    out_col_offsets = []  # offsets of general UDF output columns in the table received from C++
    for i, f in enumerate(allfuncs):
        if f.ftype == "gen_udf":
            out_col_offsets.append(col_offset)
            col_offset += 1
        elif f.ftype != "udf":
            col_offset += f.ncols_post_shuffle
        else:
            # udfs in post_shuffle table have one column for output plus redvars
            col_offset += f.n_redvars + 1

    func_text = f"def bodo_gb_apply_general_udfs{label_suffix}(num_groups, in_table, out_table):\n"
    func_text += "    if num_groups == 0:\n"  # this is dummy call
    func_text += "        return\n"
    for i, func in enumerate(udf_func_struct.general_udf_funcs):
        func_text += f"    # col {i}\n"
        func_text += f"    out_col = array_from_cpp_table(out_table, {out_col_offsets[i]}, out_col_{i}_typ)\n"
        func_text += "    for j in range(num_groups):\n"
        func_text += f"        in_col = array_from_cpp_table(in_table, {i}*num_groups + j, in_col_{i}_typ)\n"
        func_text += (
            f"        out_col[j] = func_{i}(pd.Series(in_col))  # func returns scalar\n"
        )

    glbs = {
        "pd": pd,
        "array_from_cpp_table": array_from_cpp_table,
    }
    gen_udf_offset = 0
    for i, func in enumerate(allfuncs):
        if func.ftype != "gen_udf":
            continue
        func = udf_func_struct.general_udf_funcs[gen_udf_offset]
        glbs[f"func_{gen_udf_offset}"] = func
        glbs[f"in_col_{gen_udf_offset}_typ"] = in_col_typs[func_idx_to_in_col[i]]
        glbs[f"out_col_{gen_udf_offset}_typ"] = out_col_typs[i]
        gen_udf_offset += 1
    loc_vars = {}
    exec(func_text, glbs, loc_vars)
    f = loc_vars[f"bodo_gb_apply_general_udfs{label_suffix}"]
    c_sig = types.void(types.int64, types.voidptr, types.voidptr)
    return numba.cfunc(c_sig, nopython=True)(f)


def gen_top_level_agg_func(
    agg_node,
    in_col_typs,
    out_col_typs,
    func_out_types,
    parallel,
    udf_func_struct,
    out_var_types,
    typemap,
):
    """create the top level aggregation function by generating text"""
    n_keys = len(agg_node.in_key_inds)
    n_out_vars = len(agg_node.out_vars)
    # If we output the index then we need to remove it from the list of variables.
    if agg_node.same_index:
        assert agg_node.input_has_index, (
            "agg codegen: input_has_index=True required for same_index=True"
        )

    # make sure array arg names have logical column number for easier codegen below
    # NOTE: input columns are not repeated in the arg list
    if agg_node.is_in_table_format:
        in_args = []
        if agg_node.in_vars[0] is not None:
            in_args.append("arg0")
        for i in range(agg_node.n_in_table_arrays, agg_node.n_in_cols):
            if i not in agg_node.dead_in_inds:
                in_args.append(f"arg{i}")
    else:
        in_args = [f"arg{i}" for i, v in enumerate(agg_node.in_vars) if v is not None]

    func_text = f"def agg_top({', '.join(in_args)}):\n"

    # convert arrays to cpp table, format: key columns, data columns, Index column
    # For each unique function applied to a given input column (i.e. each
    # (in_col, func) pair) we add the column to the table_info passed to C++
    # (in other words input columns can be repeated in the table info)
    # For NamedAgg case the order in which inputs are provided has to
    # match the output order, so we use agg_node.gb_info_out instead

    in_cpp_col_inds = []
    if agg_node.is_in_table_format:
        gb_info_in_cols = []
        for in_cols, _ in agg_node.gb_info_out.values():
            gb_info_in_cols.extend(in_cols)
        in_cpp_col_inds = agg_node.in_key_inds + gb_info_in_cols
        if agg_node.input_has_index:
            # Index is always last input
            in_cpp_col_inds.append(agg_node.n_in_cols - 1)
        comma = "," if len(agg_node.in_vars) - 1 == 1 else ""
        other_vars = []
        for i in range(agg_node.n_in_table_arrays, agg_node.n_in_cols):
            if i in agg_node.dead_in_inds:
                other_vars.append("None")
            else:
                other_vars.append(f"arg{i}")
        first_arg = "arg0" if agg_node.in_vars[0] is not None else "None"
        func_text += f"    table = py_data_to_cpp_table({first_arg}, ({', '.join(other_vars)}{comma}), in_col_inds, {agg_node.n_in_table_arrays})\n"
    else:
        key_in_arrs = [f"arg{i}" for i in agg_node.in_key_inds]
        data_in_arrs = []
        for in_cols, _ in agg_node.gb_info_out.values():
            for in_col in in_cols:
                data_in_arrs.append(f"arg{in_col}")
        all_in_arrs = key_in_arrs + data_in_arrs
        if agg_node.input_has_index:
            # Index is always last input
            all_in_arrs.append(f"arg{len(agg_node.in_vars) - 1}")

        # NOTE: Avoiding direct array_to_info calls to workaround possible Numba
        # refcount pruning bug. See https://bodo.atlassian.net/browse/BSE-1135
        in_cpp_col_inds = list(range(len(all_in_arrs)))
        comma = "," if len(all_in_arrs) == 1 else ""
        func_text += f"    table = py_data_to_cpp_table(None, ({', '.join(all_in_arrs)}{comma}), in_col_inds, 0)\n"

    # do_combine indicates whether GroupbyPipeline in C++ will need to do
    # `void combine()` operation or not
    do_combine = parallel
    # flat list of aggregation functions, one for each (input_col, func)
    # combination, each combination results in one output column
    allfuncs = []
    # index of first function (in allfuncs) of input col i
    func_offsets = []
    # map index of function i in allfuncs to the column in input table
    func_idx_to_in_col = []
    # Map the number of columns used by each function
    func_ncols = []
    # number of redvars for each udf function
    udf_ncols = []
    skip_na_data = False
    shift_periods = 1
    head_n = -1
    num_cum_funcs = 0
    transform_funcs = []
    n_window_calls_per_func = []
    window_ascending = []
    window_na_position = []
    window_args = []
    n_window_args = []
    n_input_cols = []

    funcs = []
    input_cols_lst = []
    for input_cols, func in agg_node.gb_info_out.values():
        funcs.append(func)
        input_cols_lst.append(input_cols)
    f_offset = 0
    for f_idx, func in enumerate(funcs):
        in_cols = input_cols_lst[f_idx]
        num_in_cols = len(in_cols)
        func_offsets.append(len(allfuncs))
        ascending = [False] * num_in_cols
        na_position = [False] * num_in_cols
        w_calls = 0
        w_args = []
        n_args = 0
        n_cols = 0
        if func.ftype in {
            "median",
            "nunique",
            "ngroup",
            "listagg",
            "array_agg",
            "array_agg_distinct",
            "mode",
            "percentile_cont",
            "percentile_disc",
            "object_agg",
        }:
            # these operations require shuffle at the beginning, so a
            # local aggregation followed by combine is not necessary
            do_combine = False
        if func.ftype in list_cumulative:
            num_cum_funcs += 1
        if hasattr(func, "skip_na_data"):
            skip_na_data = func.skip_na_data
        if func.ftype == "shift":
            shift_periods = func.periods
            do_combine = False  # See median/nunique note ^
        if func.ftype == "transform":
            transform_funcs.extend(func.transform_funcs)
            w_calls = len(func.transform_funcs)
            do_combine = False  # See median/nunique note ^
        if func.ftype == "window":
            transform_funcs.extend(func.window_funcs)
            do_combine = False  # See median/nunique note ^
            w_calls = len(func.window_funcs)
            ascending = func.ascending
            na_position = func.na_position_b
            w_args = func.window_args
            n_args = len(func.window_args)
            n_cols = func.n_input_cols
        if func.ftype == "listagg":
            do_combine = False  # See median/nunique note ^
            # length of ascending/na_position should be the same as the number of input columns passed to groupby_and_aggregate
            # Therefore, we add two extra columns to account for the listagg_sep column, and the data argument
            ascending = [False, False] + func.ascending
            na_position = [False, False] + func.na_position_b
        if func.ftype in {"array_agg", "array_agg_distinct"}:
            do_combine = False  # See median/nunique note ^
            # length of ascending/na_position should be the same as the number of input columns passed to groupby_and_aggregate
            # Therefore, we add an extra columns to account for the the data argument
            ascending = [False] + func.ascending
            na_position = [False] + func.na_position_b
        if func.ftype == "object_agg":
            do_combine = False  # See median/nunique note ^

        # Update the various window arguments
        n_window_calls_per_func.append(w_calls)
        window_ascending.extend(ascending)
        window_na_position.extend(na_position)
        window_args.extend(w_args)
        n_window_args.append(n_args)
        n_input_cols.append(n_cols)

        if func.ftype == "head":
            head_n = func.head_n
            do_combine = False  # This operation just retruns n rows. No combine needed.
        allfuncs.append(func)
        func_idx_to_in_col.append(f_offset)
        # Update the column start location for each function + indicate
        # how many columns are used by each function
        f_offset += num_in_cols
        func_ncols.append(num_in_cols)
        if func.ftype == "udf":
            udf_ncols.append(func.n_redvars)
        elif func.ftype == "gen_udf":
            udf_ncols.append(0)
            do_combine = False
    func_offsets.append(len(allfuncs))
    assert len(agg_node.gb_info_out) == len(allfuncs), (
        "invalid number of groupby outputs"
    )
    if num_cum_funcs > 0:
        if num_cum_funcs != len(allfuncs):
            raise BodoError(
                f"{agg_node.func_name}(): Cannot mix cumulative operations with other aggregation functions",
                loc=agg_node.loc,
            )
        do_combine = False  # same as median and nunique

    udf_types = []
    if udf_func_struct is not None:
        # there are user-defined functions
        udf_label = next_label()

        # generate cfuncs
        if udf_func_struct.regular_udfs:
            # generate update, combine and eval functions for the user-defined
            # functions and compile them to numba cfuncs, to be called from C++
            c_sig = types.void(
                types.voidptr, types.voidptr, types.CPointer(types.int64)
            )
            cpp_cb_update = numba.cfunc(c_sig, nopython=True)(
                gen_update_cb(
                    udf_func_struct,
                    allfuncs,
                    n_keys,
                    in_col_typs,
                    do_combine,
                    func_idx_to_in_col,
                    udf_label,
                )
            )
            cpp_cb_combine = numba.cfunc(c_sig, nopython=True)(
                gen_combine_cb(udf_func_struct, allfuncs, n_keys, udf_label)
            )
            cpp_cb_eval = numba.cfunc("void(voidptr)", nopython=True)(
                gen_eval_cb(
                    udf_func_struct, allfuncs, n_keys, func_out_types, udf_label
                )
            )

            udf_func_struct.set_regular_cfuncs(
                cpp_cb_update, cpp_cb_combine, cpp_cb_eval
            )
            for cfunc in udf_func_struct.regular_udf_cfuncs:
                gb_agg_cfunc[cfunc.native_name] = cfunc
                gb_agg_cfunc_addr[cfunc.native_name] = cfunc.address

        if udf_func_struct.general_udfs:
            cpp_cb_general = gen_general_udf_cb(
                udf_func_struct,
                allfuncs,
                n_keys,
                in_col_typs,
                func_out_types,
                func_idx_to_in_col,
                udf_label,
            )
            udf_func_struct.set_general_cfunc(cpp_cb_general)

        # generate a dummy (empty) table with correct type info for
        # output columns and reduction variables corresponding to UDFs,
        # so that the C++ runtime can allocate arrays
        red_var_typs = (
            udf_func_struct.var_typs if udf_func_struct.regular_udfs else None
        )

        redvar_offset = 0
        i = 0
        for out_col_ind, f in zip(agg_node.gb_info_out.keys(), allfuncs):
            if f.ftype in ("udf", "gen_udf"):
                udf_types.append(out_col_typs[out_col_ind])
                for j in range(redvar_offset, redvar_offset + udf_ncols[i]):
                    udf_types.append(dtype_to_array_type(red_var_typs[j]))
                redvar_offset += udf_ncols[i]
                i += 1

        func_text += f"    dummy_table = create_dummy_table(({', '.join(f'udf_type{i}' for i in range(len(udf_types)))}{',' if len(udf_types) == 1 else ''}))\n"
        func_text += f"    udf_table_dummy = py_data_to_cpp_table(dummy_table, (), udf_dummy_col_inds, {len(udf_types)})\n"

        # include cfuncs in library and insert a dummy call to make sure symbol
        # is not discarded
        if udf_func_struct.regular_udfs:
            func_text += (
                f"    add_agg_cfunc_sym(cpp_cb_update, '{cpp_cb_update.native_name}')\n"
            )
            func_text += f"    add_agg_cfunc_sym(cpp_cb_combine, '{cpp_cb_combine.native_name}')\n"
            func_text += (
                f"    add_agg_cfunc_sym(cpp_cb_eval, '{cpp_cb_eval.native_name}')\n"
            )
            func_text += f"    cpp_cb_update_addr = get_agg_udf_addr('{cpp_cb_update.native_name}')\n"
            func_text += f"    cpp_cb_combine_addr = get_agg_udf_addr('{cpp_cb_combine.native_name}')\n"
            func_text += f"    cpp_cb_eval_addr = get_agg_udf_addr('{cpp_cb_eval.native_name}')\n"
        else:
            func_text += "    cpp_cb_update_addr = 0\n"
            func_text += "    cpp_cb_combine_addr = 0\n"
            func_text += "    cpp_cb_eval_addr = 0\n"
        if udf_func_struct.general_udfs:
            cfunc = udf_func_struct.general_udf_cfunc
            gb_agg_cfunc[cfunc.native_name] = cfunc
            gb_agg_cfunc_addr[cfunc.native_name] = cfunc.address
            func_text += (
                f"    add_agg_cfunc_sym(cpp_cb_general, '{cfunc.native_name}')\n"
            )
            func_text += (
                f"    cpp_cb_general_addr = get_agg_udf_addr('{cfunc.native_name}')\n"
            )
        else:
            func_text += "    cpp_cb_general_addr = 0\n"
    else:
        # if there are no udfs we don't need udf table, so just create
        # an empty one-column table
        func_text += "    udf_table_dummy = arr_info_list_to_table([array_to_info(np.empty(1, dtype=np.int64))])\n"
        func_text += "    cpp_cb_update_addr = 0\n"
        func_text += "    cpp_cb_combine_addr = 0\n"
        func_text += "    cpp_cb_eval_addr = 0\n"
        func_text += "    cpp_cb_general_addr = 0\n"

    # NOTE: adding extra zero to make sure the list is never empty to avoid Numba
    # typing issues
    func_text += "    cols_per_func = np.array([{}], dtype=np.int8)\n".format(
        ", ".join([str(i) for i in func_ncols] + ["0"])
    )
    # NOTE: adding extra 0 to make sure the list is never empty to avoid Numba
    # typing issues
    func_text += "    transform_funcs = np.array([{}], dtype=np.int64)\n".format(
        ", ".join([str(i) for i in transform_funcs] + ["0"])
    )
    # NOTE: adding extra 0 to make sure the list is never empty to avoid Numba
    # typing issues
    func_text += "    n_window_calls_per_func = np.array([{}], dtype=np.int8)\n".format(
        ", ".join([str(i) for i in n_window_calls_per_func] + ["0"])
    )
    # NOTE: adding extra False to make sure the list is never empty to avoid Numba
    # typing issues
    func_text += "    window_ascending = np.array([{}], dtype=np.bool_)\n".format(
        ", ".join([str(i) for i in window_ascending] + ["False"])
    )
    # NOTE: adding extra False to make sure the list is never empty to avoid Numba
    # typing issues
    func_text += "    window_na_position = np.array([{}], dtype=np.bool_)\n".format(
        ", ".join([str(i) for i in window_na_position] + ["False"])
    )
    # NOTE: scalar window args get converted to a cpp table
    func_text += "    window_args_list = []\n"
    for i, window_arg in enumerate(window_args):
        func_text += f"    window_arg_arr_{i} = coerce_scalar_to_array({window_arg}, 1, unknown_type)\n"
        func_text += f"    window_arg_info_{i} = array_to_info(window_arg_arr_{i})\n"
        func_text += f"    window_args_list.append(window_arg_info_{i})\n"
    # add an extra 0 to avoid typing issues.
    func_text += "    window_arg_arr_n = coerce_scalar_to_array(0, 1, unknown_type)\n"
    func_text += "    window_arg_info_n = array_to_info(window_arg_arr_n)\n"
    func_text += "    window_args_list.append(window_arg_info_n)\n"
    func_text += "    window_args_table = arr_info_list_to_table(window_args_list)\n"

    # NOTE: adding extra 0 to make sure the list is never empty to avoid Numba
    # typing issues
    func_text += "    n_window_args_per_func = np.array([{}], dtype=np.int8)\n".format(
        ", ".join([str(i) for i in n_window_args] + ["0"])
    )

    # NOTE: adding extra 0 to make sure the list is never empty to avoid Numba
    # typing issues
    func_text += (
        "    n_window_inputs_per_func = np.array([{}], dtype=np.int32)\n".format(
            ", ".join([str(i) for i in n_input_cols] + ["0"])
        )
    )

    # NOTE: adding extra zero to make sure the list is never empty to avoid Numba
    # typing issues
    func_text += "    ftypes = np.array([{}], dtype=np.int32)\n".format(
        ", ".join([str(supported_agg_funcs.index(f.ftype)) for f in allfuncs] + ["0"])
    )
    # TODO: pass these constant arrays as globals to make compilation faster
    func_text += f"    func_offsets = np.array({str(func_offsets)}, dtype=np.int32)\n"
    if len(udf_ncols) > 0:
        func_text += f"    udf_ncols = np.array({str(udf_ncols)}, dtype=np.int32)\n"
    else:
        func_text += "    udf_ncols = np.array([0], np.int32)\n"  # dummy
    # single-element numpy array to return number of rows from C++
    func_text += "    total_rows_np = np.array([0], dtype=np.int64)\n"
    # call C++ groupby
    # We pass the logical arguments to the function (skip_na_data, return_key, same_index, ...)

    # Determine the subset of the keys to shuffle on
    n_shuffle_keys = (
        agg_node._num_shuffle_keys if agg_node._num_shuffle_keys != -1 else n_keys
    )

    func_text += (
        f"    out_table = groupby_and_aggregate(table, {n_keys}, cols_per_func.ctypes, n_window_calls_per_func.ctypes,"
        f"{len(allfuncs)}, {agg_node.input_has_index}, ftypes.ctypes, func_offsets.ctypes, "
        f"udf_ncols.ctypes, {parallel}, {skip_na_data}, {shift_periods}, "
        f"transform_funcs.ctypes, {head_n}, {agg_node.return_key}, {agg_node.same_index}, "
        f"{agg_node.dropna}, cpp_cb_update_addr, cpp_cb_combine_addr, cpp_cb_eval_addr, "
        f"cpp_cb_general_addr, udf_table_dummy, total_rows_np.ctypes, window_ascending.ctypes, "
        f"window_na_position.ctypes, window_args_table, n_window_args_per_func.ctypes, n_window_inputs_per_func.ctypes, "
        f"{agg_node.maintain_input_size}, {n_shuffle_keys}, {agg_node._use_sql_rules})\n"
    )

    out_cpp_col_inds = []
    idx = 0
    if agg_node.return_key:
        # output keys are in the beginning if as_index=False but after data if
        # as_index=True (part of Index)
        out_key_offset = (
            0
            if isinstance(agg_node.out_type.index, bodo.RangeIndexType)
            # number of data columns is all logical columns minus keys minus Index
            else agg_node.n_out_cols - len(agg_node.in_key_inds) - 1
        )
        for i in range(n_keys):
            col_no = out_key_offset + i
            # cpp returns all keys even if dead
            # TODO[BE-3182]: avoid returning dead key columns in cpp
            out_cpp_col_inds.append(
                col_no if col_no not in agg_node.dead_out_inds else -1
            )
            idx += 1

    offset = 0
    for out_col_ind in agg_node.gb_info_out.keys():
        # For window functions, the out_col_ind and possibly several numbers
        # immediately following since window can return multiple columns
        # for one single aggregation
        if agg_node.gb_info_out[out_col_ind][1].ftype == "window":
            n_window = len(agg_node.gb_info_out[0][1].window_funcs)
            for i in range(out_col_ind, out_col_ind + n_window):
                out_cpp_col_inds.append(offset + i)
            offset += n_window - 1
        else:
            out_cpp_col_inds.append(offset + out_col_ind)
        idx += 1

    # Index is always stored last
    if agg_node.same_index:
        if agg_node.out_vars[-1] is not None:
            out_cpp_col_inds.append(agg_node.n_out_cols - 1)

    # NOTE: cpp_table_to_py_data() needs a type for all logical arrays (even if None)
    # out_cpp_col_inds determines what arrays are dead
    comma = "," if n_out_vars == 1 else ""
    out_types_str = f"({', '.join(f'out_type{i}' for i in range(n_out_vars))}{comma})"

    # pass input arrays corresponding to output arrays with unknown categorical values
    # to cpp_table_to_py_data() to reuse categorical values for output arrays
    unknown_cat_arrs = []
    unknown_cat_out_inds = []
    for i, t in enumerate(out_col_typs):
        if i not in agg_node.dead_out_inds and type_has_unknown_cats(t):
            if i in agg_node.gb_info_out:
                in_cols = agg_node.gb_info_out[i][0]
                # Currently we only support functions that take exactly 1 input
                # array and output categorical data
                assert len(in_cols) == 1, (
                    "Internal error: Categorical output requires a groupby function with 1 input column"
                )
                in_col = in_cols[0]
            else:
                assert agg_node.return_key, (
                    "Internal error: groupby key output with unknown categoricals detected, but return_key is False"
                )
                key_no = i - out_key_offset
                in_col = agg_node.in_key_inds[key_no]
            unknown_cat_out_inds.append(i)
            if agg_node.is_in_table_format and in_col < agg_node.n_in_table_arrays:
                unknown_cat_arrs.append(f"get_table_data(arg0, {in_col})")
            else:
                unknown_cat_arrs.append(f"arg{in_col}")

    comma = "," if len(unknown_cat_arrs) == 1 else ""
    func_text += f"    out_data = cpp_table_to_py_data(out_table, out_col_inds, {out_types_str}, total_rows_np[0], {agg_node.n_out_table_arrays}, ({', '.join(unknown_cat_arrs)}{comma}), unknown_cat_out_inds)\n"
    # clean up
    func_text += (
        f"    ev_clean = bodo.utils.tracing.Event('tables_clean_up', {parallel})\n"
    )
    func_text += "    delete_table(out_table)\n"
    func_text += "    ev_clean.finalize()\n"

    func_text += "    return out_data\n"

    glbls = {f"out_type{i}": out_var_types[i] for i in range(n_out_vars)}
    glbls["out_col_inds"] = MetaType(tuple(out_cpp_col_inds))
    glbls["in_col_inds"] = MetaType(tuple(in_cpp_col_inds))
    glbls["cpp_table_to_py_data"] = cpp_table_to_py_data
    glbls["py_data_to_cpp_table"] = py_data_to_cpp_table
    glbls.update({f"udf_type{i}": t for i, t in enumerate(udf_types)})
    glbls["udf_dummy_col_inds"] = MetaType(tuple(range(len(udf_types))))
    glbls["create_dummy_table"] = create_dummy_table
    glbls["unknown_cat_out_inds"] = MetaType(tuple(unknown_cat_out_inds))
    glbls["get_table_data"] = bodo.hiframes.table.get_table_data
    glbls["wrap_window_arg"] = bodo.libs.distributed_api.value_to_ptr_as_int64
    glbls["coerce_scalar_to_array"] = bodo.utils.conversion.coerce_scalar_to_array
    glbls["array_to_info"] = bodo.libs.array.array_to_info
    glbls["arr_info_list_to_table"] = bodo.libs.array.arr_info_list_to_table
    glbls["unknown_type"] = types.unknown

    return func_text, glbls


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def create_dummy_table(data_types):
    """Creates a dummy TableType value with the specified array types.
    Currently used to pass UDF data types to C++ using py_data_to_cpp_table().

    Args:
        data_types (tuple(types.Type)): Array types in the dummy table

    Returns:
        TableType: dummy table with specified array types
    """
    arr_types = tuple(
        unwrap_typeref(data_types.types[i]) for i in range(len(data_types.types))
    )
    table_type = bodo.TableType(arr_types)
    glbls = {"table_type": table_type}

    func_text = "def impl(data_types):\n"
    func_text += "  py_table = init_table(table_type, False)\n"
    func_text += "  py_table = set_table_len(py_table, 1)\n"

    for typ, blk in table_type.type_to_blk.items():
        glbls[f"typ_list_{blk}"] = types.List(typ)
        glbls[f"typ_{blk}"] = typ
        n_arrs = len(table_type.block_to_arr_ind[blk])
        func_text += (
            f"  arr_list_{blk} = alloc_list_like(typ_list_{blk}, {n_arrs}, False)\n"
        )
        func_text += f"  for i in range(len(arr_list_{blk})):\n"
        func_text += f"    arr_list_{blk}[i] = alloc_type(1, typ_{blk}, (-1,))\n"
        func_text += f"  py_table = set_table_block(py_table, arr_list_{blk}, {blk})\n"

    func_text += "  return py_table\n"

    glbls.update(
        {
            "init_table": bodo.hiframes.table.init_table,
            "alloc_list_like": bodo.hiframes.table.alloc_list_like,
            "set_table_block": bodo.hiframes.table.set_table_block,
            "set_table_len": bodo.hiframes.table.set_table_len,
            "alloc_type": bodo.utils.utils.alloc_type,
        }
    )
    loc_vars = {}
    exec(func_text, glbls, loc_vars)
    return loc_vars["impl"]


def agg_table_column_use(
    agg_node, block_use_map, equiv_vars, typemap, table_col_use_map
):
    """Compute column uses in input table of groupby based on output table's uses.
    Uses gb_info_out to map output to input column number. Key columns are always used.

    Args:
        agg_node (Aggregate): Aggregate node to process
        block_use_map (Dict[str, Tuple[Set[int], bool, bool]]): column uses for current
            block.
        equiv_vars (Dict[str, Set[str]]): Dictionary
            mapping table variable names to a set of
            other table name aliases.
        typemap (dict[str, types.Type]): typemap of variables
        table_col_use_map (Dict[int, Dict[str, Tuple[Set[int], bool, bool]]]):
            Dictionary mapping block numbers to a dictionary of table names
            and "column uses". A column use is represented by the triple
                - used_cols: Set of used column numbers
                - use_all: Flag for if all columns are used. If True used_cols
                    is garbage
                - cannot_del_columns: Flag indicate this table is used in
                    an unsupported operation (e.g. passed to a DataFrame)
                    and therefore no columns can be deleted.
    """
    if not agg_node.is_in_table_format or agg_node.in_vars[0] is None:
        return

    rhs_table = agg_node.in_vars[0].name
    rhs_key = (rhs_table, None)

    (
        orig_used_cols,
        orig_use_all,
        orig_cannot_del_cols,
    ) = block_use_map[rhs_key]

    # skip if input already uses all columns or cannot delete the table
    if orig_use_all or orig_cannot_del_cols:
        return

    # get output's data column uses, which are only in first variable (table or array)
    if agg_node.is_output_table and agg_node.out_vars[0] is not None:
        out_var_key = (agg_node.out_vars[0].name, None)
        (
            out_used_cols,
            out_use_all,
            out_cannot_del_cols,
        ) = _compute_table_column_uses(out_var_key, table_col_use_map, equiv_vars)
        # we don't simply propagate use_all since all of output columns may not use all
        # of input columns (groupby has explicit column selection support)
        if out_use_all or out_cannot_del_cols:
            out_used_cols = set(range(agg_node.n_out_table_arrays))
    else:
        out_used_cols = {}
        # Series case, output only has column 0
        if agg_node.out_vars[0] is not None and 0 not in agg_node.dead_out_inds:
            out_used_cols = {0}

    # key columns are always used
    table_in_key_set = {
        i for i in agg_node.in_key_inds if i < agg_node.n_in_table_arrays
    }

    # get used input data columns
    in_used_cols = set()
    for i in out_used_cols:
        # output keys (as_index=False) are not part of gb_info_out
        if i in agg_node.gb_info_out:
            in_used_cols.update(agg_node.gb_info_out[i][0])
    in_used_cols |= table_in_key_set | orig_used_cols
    in_use_all = len(set(range(agg_node.n_in_table_arrays)) - in_used_cols) == 0

    block_use_map[rhs_key] = (
        in_used_cols,
        in_use_all,
        False,
    )


ir_extension_table_column_use[Aggregate] = agg_table_column_use


def agg_remove_dead_column(agg_node, column_live_map, equiv_vars, typemap):
    """Remove dead table columns from Aggregate node (if in table format).
    Updates all of Aggregate node's internal data structures (dead_out_inds,
    dead_in_inds, gb_info_out).

    Args:
        agg_node (Aggregate): Aggregate node to update
        column_live_map (Dict[Tuple[str, Optional[int]], Tuple[Set[int], bool, bool]]): column uses of each
            table found by the key for the current block.
        equiv_vars (Dict[Tuple[str, Optional[int]], Set[Tuple[str, Optional[int]]]]): Dictionary
            mapping tables to a set other tables via the key.
        typemap (dict[str, types.Type]): typemap of variables
    """
    if not agg_node.is_output_table or agg_node.out_vars[0] is None:
        return False

    n_table_cols = agg_node.n_out_table_arrays
    lhs_table = agg_node.out_vars[0].name
    lhs_key = (lhs_table, None)

    used_columns = _find_used_columns(
        lhs_key, n_table_cols, column_live_map, equiv_vars
    )

    # None means all columns are used so we can't prune any columns
    if used_columns is None:
        return False

    dead_columns = set(range(n_table_cols)) - used_columns
    removed = len(dead_columns - agg_node.dead_out_inds) != 0

    # update agg node's internal data structures
    if removed:
        agg_node.dead_out_inds.update(dead_columns)
        agg_node.update_dead_col_info()

    return removed


remove_dead_column_extensions[Aggregate] = agg_remove_dead_column


def compile_to_optimized_ir(func, arg_typs, typingctx, targetctx):
    # TODO: reuse Numba's compiler pipelines
    # XXX are outside function's globals needed?
    code = func.code if hasattr(func, "code") else func.__code__
    closure = func.closure if hasattr(func, "closure") else func.__closure__
    f_ir = get_ir_of_code(func.__globals__, code)
    replace_closures(f_ir, closure, code)

    # replace len(arr) calls (i.e. size of group) with a sentinel function that will be
    # replaced with a simple loop in series pass
    for block in f_ir.blocks.values():
        for stmt in block.body:
            if (
                is_call_assign(stmt)
                and find_callname(f_ir, stmt.value) == ("len", "builtins")
                and stmt.value.args[0].name == f_ir.arg_names[0]
            ):
                len_global = get_definition(f_ir, stmt.value.func)
                len_global.name = "dummy_agg_count"
                len_global.value = dummy_agg_count

    # rename all variables to avoid conflict (init and eval nodes)
    var_table = get_name_var_table(f_ir.blocks)
    new_var_dict = {}
    for name, _ in var_table.items():
        new_var_dict[name] = mk_unique_var(name)
    replace_var_names(f_ir.blocks, new_var_dict)
    f_ir._definitions = build_definitions(f_ir.blocks)

    assert f_ir.arg_count == 1, "agg function should have one input"
    # construct default flags similar to numba.core.compiler
    flags = numba.core.compiler.Flags()
    flags.nrt = True
    untyped_pass = bodo.transforms.untyped_pass.UntypedPass(
        f_ir, typingctx, arg_typs, {}, {}, flags
    )
    untyped_pass.run()
    f_ir._definitions = build_definitions(f_ir.blocks)
    typemap, return_type, calltypes, _ = numba.core.typed_passes.type_inference_stage(
        typingctx, targetctx, f_ir, arg_typs, None
    )

    options = numba.core.cpu.ParallelOptions(True)
    targetctx = numba.core.cpu.CPUContext(typingctx)

    DummyPipeline = namedtuple(
        "DummyPipeline",
        [
            "typingctx",
            "targetctx",
            "args",
            "func_ir",
            "typemap",
            "return_type",
            "calltypes",
            "type_annotation",
            "locals",
            "flags",
            "pipeline",
        ],
    )
    TypeAnnotation = namedtuple("TypeAnnotation", ["typemap", "calltypes"])
    ta = TypeAnnotation(typemap, calltypes)
    # The new Numba 0.50 inliner requires the pipeline state itself to be a member of
    # the pipeline state. To emulate it using a namedtuple (which is immutable), we
    # create a pipline first with the required data and add it to another one.
    pm = DummyPipeline(
        typingctx,
        targetctx,
        None,
        f_ir,
        typemap,
        return_type,
        calltypes,
        ta,
        {},
        flags,
        None,
    )
    untyped_pipeline = numba.core.compiler.DefaultPassBuilder.define_untyped_pipeline(
        pm
    )
    pm = DummyPipeline(
        typingctx,
        targetctx,
        None,
        f_ir,
        typemap,
        return_type,
        calltypes,
        ta,
        {},
        flags,
        untyped_pipeline,
    )
    # run overload inliner to inline Series implementations such as Series.max()
    inline_overload_pass = numba.core.typed_passes.InlineOverloads()
    create_nested_run_pass_event(inline_overload_pass.name(), pm, inline_overload_pass)

    # TODO: Can we capture parfor_metadata for error messages?
    series_pass = bodo.transforms.series_pass.SeriesPass(
        f_ir, typingctx, targetctx, typemap, calltypes, {}, optimize_inplace_ops=False
    )
    series_pass.run()
    # change the input type to UDF from Series to Array since Bodo passes Arrays to UDFs
    # Series functions should be handled by SeriesPass and there should be only
    # `get_series_data` Series function left to remove
    for block in f_ir.blocks.values():
        for stmt in block.body:
            if (
                is_assign(stmt)
                and isinstance(stmt.value, (ir.Arg, ir.Var))
                and isinstance(typemap[stmt.target.name], SeriesType)
            ):
                typ = typemap.pop(stmt.target.name)
                typemap[stmt.target.name] = typ.data
            if is_call_assign(stmt) and find_callname(f_ir, stmt.value) == (
                "get_series_data",
                "bodo.hiframes.pd_series_ext",
            ):
                f_ir._definitions[stmt.target.name].remove(stmt.value)
                stmt.value = stmt.value.args[0]
                f_ir._definitions[stmt.target.name].append(stmt.value)
            # remove isna() calls since NA cannot be handled in UDFs yet
            # TODO: support NA in UDFs
            if is_call_assign(stmt) and find_callname(f_ir, stmt.value) == (
                "isna",
                "bodo.libs.array_kernels",
            ):
                f_ir._definitions[stmt.target.name].remove(stmt.value)
                stmt.value = ir.Const(False, stmt.loc)
                f_ir._definitions[stmt.target.name].append(stmt.value)
            # remove setna() calls since NA cannot be handled in UDFs yet
            # TODO: support NA in UDFs
            if is_call_assign(stmt) and find_callname(f_ir, stmt.value) == (
                "setna",
                "bodo.libs.array_kernels",
            ):
                f_ir._definitions[stmt.target.name].remove(stmt.value)
                stmt.value = ir.Const(False, stmt.loc)
                f_ir._definitions[stmt.target.name].append(stmt.value)

    bodo.transforms.untyped_pass.remove_dead_branches(f_ir)
    preparfor_pass = numba.parfors.parfor.PreParforPass(
        f_ir, typemap, calltypes, typingctx, targetctx, options
    )
    preparfor_pass.run()
    f_ir._definitions = build_definitions(f_ir.blocks)
    state = numba.core.compiler.StateDict()
    state.func_ir = f_ir
    state.typemap = typemap
    state.calltypes = calltypes
    state.typingctx = typingctx
    state.targetctx = targetctx
    state.return_type = return_type
    numba.core.rewrites.rewrite_registry.apply("after-inference", state)
    parfor_pass = numba.parfors.parfor.ParforPass(
        f_ir, typemap, calltypes, return_type, typingctx, targetctx, options, flags, {}
    )
    parfor_pass.run()
    parfor_pass = numba.parfors.parfor.ParforFusionPass(
        f_ir, typemap, calltypes, return_type, typingctx, targetctx, options, flags, {}
    )
    parfor_pass.run()
    parfor_pass = numba.parfors.parfor.ParforPreLoweringPass(
        f_ir, typemap, calltypes, return_type, typingctx, targetctx, options, flags, {}
    )
    parfor_pass.run()

    # TODO(ehsan): remove when this PR is merged and released in Numba:
    # https://github.com/numba/numba/pull/6519
    remove_dels(f_ir.blocks)
    # make sure eval nodes are after the parfor for easier extraction
    # TODO: extract an eval func more robustly
    numba.parfors.parfor.maximize_fusion(f_ir, f_ir.blocks, typemap, False)
    return f_ir, pm


def replace_closures(f_ir, closure, code):
    """replace closure variables similar to inline_closure_call"""
    if closure:
        closure = f_ir.get_definition(closure)
        if isinstance(closure, tuple):
            cellget = ctypes.pythonapi.PyCell_Get
            cellget.restype = ctypes.py_object
            cellget.argtypes = (ctypes.py_object,)
            items = tuple(cellget(x) for x in closure)
        else:
            assert isinstance(closure, ir.Expr) and closure.op == "build_tuple"
            items = closure.items
        assert len(code.co_freevars) == len(items)
        numba.core.inline_closurecall._replace_freevars(f_ir.blocks, items)


class RegularUDFGenerator:
    """Generate code that applies UDFs to all columns that use them"""

    def __init__(
        self,
        in_col_types,
        typingctx,
        targetctx,
    ):
        self.in_col_types = in_col_types
        self.typingctx = typingctx
        self.targetctx = targetctx
        self.all_reduce_vars = []
        self.all_vartypes = []
        self.all_init_nodes = []
        self.all_eval_funcs = []
        self.all_update_funcs = []
        self.all_combine_funcs = []
        # offsets of reduce vars
        self.curr_offset = 0
        self.redvar_offsets = [0]

    def add_udf(self, in_col_typ, func):
        # convert dict-encoded string array to regular string array to make sure parfors
        # are generated instead of optimized dict-encoded calls like dict_arr_eq.
        # see test_groupby_agg_nullable_or
        in_series_typ = SeriesType(
            in_col_typ.dtype, to_str_arr_if_dict_array(in_col_typ), None, string_type
        )
        # compile UDF to IR
        f_ir, pm = compile_to_optimized_ir(
            func, (in_series_typ,), self.typingctx, self.targetctx
        )
        f_ir._definitions = build_definitions(f_ir.blocks)

        assert len(f_ir.blocks) == 1 and 0 in f_ir.blocks, (
            "only simple functions with one block supported for aggregation"
        )
        block = f_ir.blocks[0]

        # find and ignore arg and size/shape nodes for input arr
        block_body, arr_var = _rm_arg_agg_block(block, pm.typemap)

        parfor_ind = -1
        for i, stmt in enumerate(block_body):
            if isinstance(stmt, numba.parfors.parfor.Parfor):
                assert parfor_ind == -1, "only one parfor for aggregation function"
                parfor_ind = i

        # some UDFs could have no parfors (e.g. lambda x: 1)
        parfor = None
        if parfor_ind != -1:
            parfor = block_body[parfor_ind]
            # TODO(ehsan): remove when this PR is merged and released in Numba:
            # https://github.com/numba/numba/pull/6519
            remove_dels(parfor.loop_body)
            remove_dels({0: parfor.init_block})

        init_nodes = []
        if parfor:
            init_nodes = block_body[:parfor_ind] + parfor.init_block.body

        eval_nodes = block_body[parfor_ind + 1 :]

        redvars = []
        var_to_redvar = {}
        if parfor:
            redvars, var_to_redvar = get_parfor_reductions(
                parfor, parfor.params, pm.calltypes
            )

        func.ncols_pre_shuffle = len(redvars)
        func.ncols_post_shuffle = len(redvars) + 1  # one for output after eval
        func.n_redvars = len(redvars)

        # find reduce variables given their names
        reduce_vars = [0] * len(redvars)
        for stmt in init_nodes:
            if is_assign(stmt) and stmt.target.name in redvars:
                ind = redvars.index(stmt.target.name)
                reduce_vars[ind] = stmt.target
        var_types = [pm.typemap[v] for v in redvars]

        combine_func = gen_combine_func(
            f_ir,
            parfor,
            redvars,
            var_to_redvar,
            var_types,
            arr_var,
            pm,
            self.typingctx,
            self.targetctx,
        )

        init_nodes = _mv_read_only_init_vars(init_nodes, parfor, eval_nodes)

        # XXX: update mutates parfor body
        update_func = gen_update_func(
            parfor,
            redvars,
            var_to_redvar,
            var_types,
            arr_var,
            in_col_typ,
            pm,
            self.typingctx,
            self.targetctx,
        )

        eval_func = gen_eval_func(
            f_ir, eval_nodes, reduce_vars, var_types, pm, self.typingctx, self.targetctx
        )

        self.all_reduce_vars += reduce_vars
        self.all_vartypes += var_types
        self.all_init_nodes += init_nodes
        self.all_eval_funcs.append(eval_func)
        self.all_update_funcs.append(update_func)
        self.all_combine_funcs.append(combine_func)
        self.curr_offset += len(redvars)
        self.redvar_offsets.append(self.curr_offset)

    def gen_all_func(self):
        # return None if no regular UDFs
        if len(self.all_update_funcs) == 0:
            return None

        init_func = gen_init_func(
            self.all_init_nodes,
            self.all_reduce_vars,
            self.all_vartypes,
            self.typingctx,
            self.targetctx,
        )
        update_all_func = gen_all_update_func(
            self.all_update_funcs,
            self.in_col_types,
            self.redvar_offsets,
        )
        combine_all_func = gen_all_combine_func(
            self.all_combine_funcs,
            self.all_vartypes,
            self.redvar_offsets,
            self.typingctx,
            self.targetctx,
        )
        eval_all_func = gen_all_eval_func(
            self.all_eval_funcs,
            self.redvar_offsets,
        )
        return (
            self.all_vartypes,
            init_func,
            update_all_func,
            combine_all_func,
            eval_all_func,
        )


class GeneralUDFGenerator:
    def __init__(self):
        self.funcs = []

    def add_udf(self, func):
        self.funcs.append(bodo.jit(distributed=False)(func))
        func.ncols_pre_shuffle = 1  # does not apply
        func.ncols_post_shuffle = 1
        func.n_redvars = 0

    def gen_all_func(self):
        if len(self.funcs) > 0:
            return self.funcs
        else:
            return None


def get_udf_func_struct(
    agg_func,
    in_col_types,
    typingctx,
    targetctx,
):
    # Construct list of (input col type, aggregation func)
    # If multiple functions will be applied to the same input column, that
    # input column will appear multiple times in the generated list
    typ_and_func = []
    for t, f in zip(in_col_types, agg_func):
        typ_and_func.append((t, f))

    # Create UDF code generators
    regular_udf_gen = RegularUDFGenerator(
        in_col_types,
        typingctx,
        targetctx,
    )
    general_udf_gen = GeneralUDFGenerator()

    for in_col_typ, func in typ_and_func:
        if func.ftype not in ("udf", "gen_udf"):
            continue  # skip non-udf functions

        try:
            # First try to generate a regular UDF with one parfor and reduction
            # variables
            regular_udf_gen.add_udf(in_col_typ, func)
        except Exception:
            # Assume this UDF is a general function
            # NOTE that if there are general UDFs the groupby parallelization
            # will be less efficient
            general_udf_gen.add_udf(func)
            # XXX could same function be general and regular UDF depending
            # on input type?
            func.ftype = "gen_udf"

    # generate code that calls UDFs for all input columns with regular UDFs
    regular_udf_funcs = regular_udf_gen.gen_all_func()
    # generate code that calls UDFs for all input columns with general UDFs
    general_udf_funcs = general_udf_gen.gen_all_func()

    if regular_udf_funcs is not None or general_udf_funcs is not None:
        return AggUDFStruct(regular_udf_funcs, general_udf_funcs)
    else:
        # no user-defined functions found for groupby.agg()
        return None


def _mv_read_only_init_vars(init_nodes, parfor, eval_nodes):
    """move stmts that are only used in the parfor body to the beginning of
    parfor body. For example, in test_agg_seq_str, B='aa' should be moved.
    """
    if not parfor:
        return init_nodes

    # get parfor body usedefs
    use_defs = compute_use_defs(parfor.loop_body)
    parfor_uses = set()
    for s in use_defs.usemap.values():
        parfor_uses |= s
    parfor_defs = set()
    for s in use_defs.defmap.values():
        parfor_defs |= s

    # get uses of eval nodes
    dummy_block = ir.Block(ir.Scope(None, parfor.loc), parfor.loc)
    dummy_block.body = eval_nodes
    e_use_defs = compute_use_defs({0: dummy_block})
    e_uses = e_use_defs.usemap[0]

    # find stmts that are only used in parfor body
    i_uses = set()  # variables used later in init nodes
    new_init_nodes = []
    const_nodes = []
    for stmt in reversed(init_nodes):
        stmt_uses = {v.name for v in stmt.list_vars()}
        if is_assign(stmt):
            v = stmt.target.name
            stmt_uses.remove(v)
            # v is only used in parfor body
            if (
                v in parfor_uses
                and v not in i_uses
                and v not in e_uses
                and v not in parfor_defs
            ):
                const_nodes.append(stmt)
                # If we add a variable to the body, update the
                # uses + defs. Uses matter because there may
                # be depedencies
                # i.e.
                # $data.515.582 = const(int, 3)
                # rhs_arr.500 = $data.515.582
                # Defs shouldn't matter but keeps information correct
                parfor_uses |= stmt_uses
                parfor_defs.add(v)
                continue
        i_uses |= stmt_uses
        new_init_nodes.append(stmt)

    const_nodes.reverse()
    new_init_nodes.reverse()

    first_body_label = min(parfor.loop_body.keys())
    first_block = parfor.loop_body[first_body_label]
    first_block.body = const_nodes + first_block.body
    return new_init_nodes


def gen_init_func(init_nodes, reduce_vars, var_types, typingctx, targetctx):
    # parallelaccelerator adds functions that check the size of input array
    # these calls need to be removed
    _checker_calls = (
        numba.parfors.parfor.max_checker,
        numba.parfors.parfor.min_checker,
        numba.parfors.parfor.argmax_checker,
        numba.parfors.parfor.argmin_checker,
    )
    checker_vars = set()
    cleaned_init_nodes = []
    for stmt in init_nodes:
        if (
            is_assign(stmt)
            and isinstance(stmt.value, ir.Global)
            and isinstance(stmt.value.value, pytypes.FunctionType)
            and stmt.value.value in _checker_calls
        ):
            checker_vars.add(stmt.target.name)
        elif is_call_assign(stmt) and stmt.value.func.name in checker_vars:
            pass  # remove call
        else:
            cleaned_init_nodes.append(stmt)

    init_nodes = cleaned_init_nodes

    return_typ = types.Tuple(var_types)

    dummy_f = lambda: None
    f_ir = compile_to_numba_ir(dummy_f, {})
    block = list(f_ir.blocks.values())[0]
    loc = block.loc

    # return initialized reduce vars as tuple
    tup_var = ir.Var(block.scope, mk_unique_var("init_tup"), loc)
    tup_assign = ir.Assign(ir.Expr.build_tuple(reduce_vars, loc), tup_var, loc)
    block.body = block.body[-2:]
    block.body = init_nodes + [tup_assign] + block.body
    block.body[-2].value.value = tup_var

    # compile implementation to binary (Dispatcher)
    init_all_func = compiler.compile_ir(
        typingctx, targetctx, f_ir, (), return_typ, compiler.DEFAULT_FLAGS, {}
    )
    from numba.core.target_extension import cpu_target

    imp_dis = numba.core.target_extension.dispatcher_registry[cpu_target](dummy_f)
    imp_dis.add_overload(init_all_func)
    return imp_dis


def gen_all_update_func(
    update_funcs,
    in_col_types,
    redvar_offsets,
):
    out_num_cols = len(update_funcs)
    in_num_cols = len(in_col_types)

    func_text = "def update_all_f(redvar_arrs, data_in, w_ind, i):\n"
    for j in range(out_num_cols):
        redvar_access = ", ".join(
            [
                f"redvar_arrs[{i}][w_ind]"
                for i in range(redvar_offsets[j], redvar_offsets[j + 1])
            ]
        )
        if redvar_access:  # if there is a parfor
            func_text += f"  {redvar_access} = update_vars_{j}({redvar_access},  data_in[{0 if in_num_cols == 1 else j}][i])\n"
    func_text += "  return\n"

    glbs = {}
    for i, f in enumerate(update_funcs):
        glbs[f"update_vars_{i}"] = f
    loc_vars = {}
    exec(func_text, glbs, loc_vars)
    update_all_f = loc_vars["update_all_f"]
    return numba.njit(no_cpython_wrapper=True)(update_all_f)


def gen_all_combine_func(
    combine_funcs,
    reduce_var_types,
    redvar_offsets,
    typingctx,
    targetctx,
):
    reduce_arrs_tup_typ = types.Tuple(
        [types.Array(t, 1, "C") for t in reduce_var_types]
    )
    arg_typs = (
        reduce_arrs_tup_typ,
        reduce_arrs_tup_typ,
        types.intp,
        types.intp,
    )

    num_cols = len(redvar_offsets) - 1

    func_text = "def combine_all_f(redvar_arrs, recv_arrs, w_ind, i):\n"

    for j in range(num_cols):
        redvar_access = ", ".join(
            [
                f"redvar_arrs[{i}][w_ind]"
                for i in range(redvar_offsets[j], redvar_offsets[j + 1])
            ]
        )
        recv_access = ", ".join(
            [
                f"recv_arrs[{i}][i]"
                for i in range(redvar_offsets[j], redvar_offsets[j + 1])
            ]
        )
        if recv_access:  # if there is a parfor
            func_text += f"  {redvar_access} = combine_vars_{j}({redvar_access}, {recv_access})\n"
    func_text += "  return\n"
    glbs = {}
    for i, f in enumerate(combine_funcs):
        glbs[f"combine_vars_{i}"] = f
    loc_vars = {}
    exec(func_text, glbs, loc_vars)
    combine_all_f = loc_vars["combine_all_f"]

    f_ir = compile_to_numba_ir(combine_all_f, glbs)

    # compile implementation to binary (Dispatcher)
    combine_all_func = compiler.compile_ir(
        typingctx, targetctx, f_ir, arg_typs, types.none, compiler.DEFAULT_FLAGS, {}
    )

    from numba.core.target_extension import cpu_target

    imp_dis = numba.core.target_extension.dispatcher_registry[cpu_target](combine_all_f)
    imp_dis.add_overload(combine_all_func)
    return imp_dis


def gen_all_eval_func(
    eval_funcs,
    redvar_offsets,
):
    num_cols = len(redvar_offsets) - 1

    func_text = "def eval_all_f(redvar_arrs, out_arrs, j):\n"

    for j in range(num_cols):
        redvar_access = ", ".join(
            [
                f"redvar_arrs[{i}][j]"
                for i in range(redvar_offsets[j], redvar_offsets[j + 1])
            ]
        )
        func_text += f"  out_arrs[{j}][j] = eval_vars_{j}({redvar_access})\n"
    func_text += "  return\n"
    glbs = {}
    for i, f in enumerate(eval_funcs):
        glbs[f"eval_vars_{i}"] = f
    loc_vars = {}
    exec(func_text, glbs, loc_vars)
    eval_all_f = loc_vars["eval_all_f"]
    return numba.njit(no_cpython_wrapper=True)(eval_all_f)


def gen_eval_func(f_ir, eval_nodes, reduce_vars, var_types, pm, typingctx, targetctx):
    """Generates a Numba function for "eval" step of an agg operation.
    The eval step computes the final result for each group.
    """
    # eval func takes reduce vars and produces final result
    num_red_vars = len(var_types)
    in_names = [f"in{i}" for i in range(num_red_vars)]
    return_typ = types.unliteral(pm.typemap[eval_nodes[-1].value.name])

    # TODO: non-numeric return
    zero = return_typ(0)
    func_text = "def agg_eval({}):\n return _zero\n".format(", ".join(in_names))

    loc_vars = {}
    exec(func_text, {"_zero": zero}, loc_vars)
    agg_eval = loc_vars["agg_eval"]

    arg_typs = tuple(var_types)
    f_ir = compile_to_numba_ir(
        agg_eval,
        # TODO: add outside globals
        {"numba": numba, "bodo": bodo, "np": np, "_zero": zero},
        typingctx=typingctx,
        targetctx=targetctx,
        arg_typs=arg_typs,
        typemap=pm.typemap,
        calltypes=pm.calltypes,
    )

    # TODO: support multi block eval funcs
    block = list(f_ir.blocks.values())[0]

    # assign inputs to reduce vars used in computation
    assign_nodes = []
    for i, v in enumerate(reduce_vars):
        assign_nodes.append(ir.Assign(block.body[i].target, v, v.loc))
        # make sure all versions of the reduce variable have the right output
        # SSA changes in Numba 0.53.0rc2 may create extra versions of the reduce
        # variable
        for v_ver in v.versioned_names:
            assign_nodes.append(ir.Assign(v, ir.Var(v.scope, v_ver, v.loc), v.loc))
    block.body = block.body[:num_red_vars] + assign_nodes + eval_nodes

    # compile implementation to binary (Dispatcher)
    eval_func = compiler.compile_ir(
        typingctx, targetctx, f_ir, arg_typs, return_typ, compiler.DEFAULT_FLAGS, {}
    )

    from numba.core.target_extension import cpu_target

    imp_dis = numba.core.target_extension.dispatcher_registry[cpu_target](agg_eval)
    imp_dis.add_overload(eval_func)
    return imp_dis


def gen_combine_func(
    f_ir, parfor, redvars, var_to_redvar, var_types, arr_var, pm, typingctx, targetctx
):
    """generates a Numba function for the "combine" step of an agg operation.
    The combine step combines the received aggregated data from other processes.
    Example for a basic sum reduce:
        def agg_combine(v0, in0):
            v0 += in0
            return v0
    """

    # no need for combine if there is no parfor
    if not parfor:
        return numba.njit(lambda: ())

    num_red_vars = len(redvars)
    redvar_in_names = [f"v{i}" for i in range(num_red_vars)]
    in_names = [f"in{i}" for i in range(num_red_vars)]

    func_text = "def agg_combine({}):\n".format(", ".join(redvar_in_names + in_names))

    blocks = wrap_parfor_blocks(parfor)
    topo_order = find_topo_order(blocks)
    topo_order = topo_order[1:]  # ignore init block
    unwrap_parfor_blocks(parfor)

    special_combines = {}
    ignore_redvar_inds = []

    for label in topo_order:
        bl = parfor.loop_body[label]
        for stmt in bl.body:
            # reduction variables
            if is_assign(stmt) and stmt.target.name in redvars:
                red_var = stmt.target.name
                ind = redvars.index(red_var)
                if ind in ignore_redvar_inds:
                    continue
                if len(f_ir._definitions[red_var]) == 2:
                    # 0 is the actual func since init_block is traversed later
                    # in parfor.py:3039, TODO: make this detection more robust
                    # XXX trying both since init_prange doesn't work for min
                    var_def = f_ir._definitions[red_var][0]
                    func_text += _match_reduce_def(var_def, f_ir, ind)
                    var_def = f_ir._definitions[red_var][1]
                    func_text += _match_reduce_def(var_def, f_ir, ind)

    func_text += "    return {}".format(
        ", ".join([f"v{i}" for i in range(num_red_vars)])
    )
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    agg_combine = loc_vars["agg_combine"]

    # reduction variable types for new input and existing values
    arg_typs = tuple(2 * var_types)

    glbs = {"numba": numba, "bodo": bodo, "np": np}
    glbs.update(special_combines)
    f_ir = compile_to_numba_ir(
        agg_combine,
        glbs,  # TODO: add outside globals
        typingctx=typingctx,
        targetctx=targetctx,
        arg_typs=arg_typs,
        typemap=pm.typemap,
        calltypes=pm.calltypes,
    )

    block = list(f_ir.blocks.values())[0]

    return_typ = pm.typemap[block.body[-1].value.name]
    # compile implementation to binary (Dispatcher)
    combine_func = compiler.compile_ir(
        typingctx, targetctx, f_ir, arg_typs, return_typ, compiler.DEFAULT_FLAGS, {}
    )

    from numba.core.target_extension import cpu_target

    imp_dis = numba.core.target_extension.dispatcher_registry[cpu_target](agg_combine)
    imp_dis.add_overload(combine_func)
    return imp_dis


def _match_reduce_def(var_def, f_ir, ind):
    func_text = ""
    while isinstance(var_def, ir.Var):
        var_def = guard(get_definition, f_ir, var_def)
    # TODO: support other reductions
    if (
        isinstance(var_def, ir.Expr)
        and var_def.op == "inplace_binop"
        and var_def.fn in ("+=", operator.iadd)
    ):
        func_text = f"    v{ind} += in{ind}\n"
    if isinstance(var_def, ir.Expr) and var_def.op == "call":
        fdef = guard(find_callname, f_ir, var_def)
        if fdef == ("min", "builtins"):
            func_text = f"    v{ind} = min(v{ind}, in{ind})\n"
        if fdef == ("max", "builtins"):
            func_text = f"    v{ind} = max(v{ind}, in{ind})\n"
    return func_text


def gen_update_func(
    parfor,
    redvars,
    var_to_redvar,
    var_types,
    arr_var,
    in_col_typ,
    pm,
    typingctx,
    targetctx,
):
    """generates a Numba function for the "update" step of an agg operation.
    The update step performs the initial aggregation of local data before communication.
    Example for 'lambda a: (a=="AA").sum()':
        def agg_combine(v0, in0):
            v0 += in0 == "AA"
            return v0
    """

    # no need for update if there is no parfor
    if not parfor:
        return numba.njit(lambda A: ())

    num_red_vars = len(redvars)
    num_in_vars = 1

    # create input value variable for each reduction variable
    in_vars = []
    for i in range(num_in_vars):
        in_var = ir.Var(arr_var.scope, f"$input{i}", arr_var.loc)
        in_vars.append(in_var)

    # replace X[i] with input value
    index_var = parfor.loop_nests[0].index_variable
    red_ir_vars = [0] * num_red_vars
    for bl in parfor.loop_body.values():
        new_body = []
        for stmt in bl.body:
            # remove extra index assignment i = parfor_index for isna(A, i)
            if is_var_assign(stmt) and stmt.value.name == index_var.name:
                continue
            if is_getitem(stmt) and stmt.value.value.name == arr_var.name:
                stmt.value = in_vars[0]
            # XXX replace bodo.libs.array_kernels.isna(A, i) for now
            # TODO: handle actual NA
            # for test_agg_seq_count_str test
            if (
                is_call_assign(stmt)
                and guard(find_callname, pm.func_ir, stmt.value)
                == ("isna", "bodo.libs.array_kernels")
                and stmt.value.args[0].name == arr_var.name
            ):
                stmt.value = ir.Const(False, stmt.target.loc)
            # store reduction variables
            if is_assign(stmt) and stmt.target.name in redvars:
                ind = redvars.index(stmt.target.name)
                red_ir_vars[ind] = stmt.target
            new_body.append(stmt)
        bl.body = new_body

    redvar_in_names = [f"v{i}" for i in range(num_red_vars)]
    in_names = [f"in{i}" for i in range(num_in_vars)]

    func_text = "def agg_update({}):\n".format(", ".join(redvar_in_names + in_names))
    func_text += "    __update_redvars()\n"
    func_text += "    return {}".format(
        ", ".join([f"v{i}" for i in range(num_red_vars)])
    )

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    agg_update = loc_vars["agg_update"]

    # XXX input column type can be different than reduction variable type
    arg_typs = tuple(var_types + [in_col_typ.dtype] * num_in_vars)

    f_ir = compile_to_numba_ir(
        agg_update,
        # TODO: add outside globals
        {"__update_redvars": __update_redvars},
        typingctx=typingctx,
        targetctx=targetctx,
        arg_typs=arg_typs,
        typemap=pm.typemap,
        calltypes=pm.calltypes,
    )

    f_ir._definitions = build_definitions(f_ir.blocks)

    body = f_ir.blocks.popitem()[1].body
    return_typ = pm.typemap[body[-1].value.name]

    blocks = wrap_parfor_blocks(parfor)
    topo_order = find_topo_order(blocks)
    topo_order = topo_order[1:]  # ignore init block
    unwrap_parfor_blocks(parfor)

    f_ir.blocks = parfor.loop_body
    first_block = f_ir.blocks[topo_order[0]]
    last_block = f_ir.blocks[topo_order[-1]]

    # arg assigns
    initial_assigns = body[: (num_red_vars + num_in_vars)]
    if num_red_vars > 1:
        # return nodes: build_tuple, cast, return
        return_nodes = body[-3:]
        assert (
            is_assign(return_nodes[0])
            and isinstance(return_nodes[0].value, ir.Expr)
            and return_nodes[0].value.op == "build_tuple"
        )
    else:
        # return nodes: cast, return
        return_nodes = body[-2:]

    # assign input reduce vars
    # redvar_i = v_i
    for i in range(num_red_vars):
        arg_var = body[i].target
        node = ir.Assign(arg_var, red_ir_vars[i], arg_var.loc)
        initial_assigns.append(node)

    # assign input value vars
    # redvar_in_i = in_i
    for i in range(num_red_vars, num_red_vars + num_in_vars):
        arg_var = body[i].target
        node = ir.Assign(arg_var, in_vars[i - num_red_vars], arg_var.loc)
        initial_assigns.append(node)

    first_block.body = initial_assigns + first_block.body

    # assign ouput reduce vars
    # v_i = red_var_i
    after_assigns = []
    for i in range(num_red_vars):
        arg_var = body[i].target
        node = ir.Assign(red_ir_vars[i], arg_var, arg_var.loc)
        after_assigns.append(node)

    last_block.body += after_assigns + return_nodes

    # TODO: simplify f_ir
    # compile implementation to binary (Dispatcher)
    agg_impl_func = compiler.compile_ir(
        typingctx, targetctx, f_ir, arg_typs, return_typ, compiler.DEFAULT_FLAGS, {}
    )

    from numba.core.target_extension import cpu_target

    imp_dis = numba.core.target_extension.dispatcher_registry[cpu_target](agg_update)
    imp_dis.add_overload(agg_impl_func)
    return imp_dis


def _rm_arg_agg_block(block, typemap):
    block_body = []
    arr_var = None
    for i, stmt in enumerate(block.body):
        if is_assign(stmt) and isinstance(stmt.value, ir.Arg):
            arr_var = stmt.target
            arr_typ = typemap[arr_var.name]
            # array analysis generates shape only for ArrayCompatible types
            if not isinstance(arr_typ, types.ArrayCompatible):
                block_body += block.body[i + 1 :]
                break
            # XXX assuming shape/size nodes are right after arg
            shape_nd = block.body[i + 1]
            assert (
                is_assign(shape_nd)
                and isinstance(shape_nd.value, ir.Expr)
                and shape_nd.value.op == "getattr"
                and shape_nd.value.attr == "shape"
                and shape_nd.value.value.name == arr_var.name
            )
            shape_vr = shape_nd.target
            size_nd = block.body[i + 2]
            assert (
                is_assign(size_nd)
                and isinstance(size_nd.value, ir.Expr)
                and size_nd.value.op == "static_getitem"
                and size_nd.value.value.name == shape_vr.name
            )
            # ignore size/shape vars
            block_body += block.body[i + 3 :]
            break
        block_body.append(stmt)

    return block_body, arr_var


# adapted from numba/parfor.py
def get_parfor_reductions(
    parfor,
    parfor_params,
    calltypes,
    reduce_varnames=None,
    param_uses=None,
    var_to_param=None,
):
    """find variables that are updated using their previous values and an array
    item accessed with parfor index, e.g. s = s+A[i]
    """
    if reduce_varnames is None:
        reduce_varnames = []

    # for each param variable, find what other variables are used to update it
    # also, keep the related nodes
    if param_uses is None:
        param_uses = defaultdict(list)
    if var_to_param is None:
        var_to_param = {}

    blocks = wrap_parfor_blocks(parfor)
    topo_order = find_topo_order(blocks)
    topo_order = topo_order[1:]  # ignore init block
    unwrap_parfor_blocks(parfor)

    for label in reversed(topo_order):
        for stmt in reversed(parfor.loop_body[label].body):
            if isinstance(stmt, ir.Assign) and (
                stmt.target.name in parfor_params or stmt.target.name in var_to_param
            ):
                lhs = stmt.target.name
                rhs = stmt.value
                cur_param = lhs if lhs in parfor_params else var_to_param[lhs]
                used_vars = []
                if isinstance(rhs, ir.Var):
                    used_vars = [rhs.name]
                elif isinstance(rhs, ir.Expr):
                    used_vars = [v.name for v in stmt.value.list_vars()]
                param_uses[cur_param].extend(used_vars)
                for v in used_vars:
                    var_to_param[v] = cur_param
            if isinstance(stmt, Parfor):
                # recursive parfors can have reductions like test_prange8
                get_parfor_reductions(
                    stmt,
                    parfor_params,
                    calltypes,
                    reduce_varnames,
                    param_uses,
                    var_to_param,
                )

    for param, used_vars in param_uses.items():
        # a parameter is a reduction variable if its value is used to update it
        # check reduce_varnames since recursive parfors might have processed
        # param already
        if param in used_vars and param not in reduce_varnames:
            reduce_varnames.append(param)

    return reduce_varnames, var_to_param


# sentinel function for the use of len (length of group) in agg UDFs, which will be
# replaced with a dummy loop in series pass
@numba.extending.register_jitable
def dummy_agg_count(A):  # pragma: no cover
    return len(A)
