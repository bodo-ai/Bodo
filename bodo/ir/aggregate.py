# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""IR node for the groupby, pivot and cross_tabulation"""
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
from numba.extending import intrinsic, lower_builtin, overload
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
    array_to_info,
    compute_node_partition_by_hash,
    delete_info_decref_array,
    delete_table,
    delete_table_decref_arrays,
    groupby_and_aggregate,
    info_from_table,
    info_to_array,
    pivot_groupby_and_aggregate,
)
from bodo.libs.array_item_arr_ext import (
    ArrayItemArrayType,
    pre_alloc_array_item_array,
)
from bodo.libs.binary_arr_ext import BinaryArrayType, pre_alloc_binary_array
from bodo.libs.bool_arr_ext import BooleanArrayType
from bodo.libs.decimal_arr_ext import DecimalArrayType, alloc_decimal_array
from bodo.libs.int_arr_ext import IntDtype, IntegerArrayType
from bodo.libs.str_arr_ext import (
    StringArrayType,
    pre_alloc_string_array,
    string_array_type,
)
from bodo.libs.str_ext import string_type
from bodo.transforms import distributed_analysis, distributed_pass
from bodo.transforms.distributed_analysis import Distribution
from bodo.utils.transform import get_call_expr_arg
from bodo.utils.typing import (
    BodoError,
    get_literal_value,
    get_overload_const_func,
    get_overload_const_str,
    get_overload_constant_dict,
    is_overload_constant_dict,
    is_overload_constant_str,
    list_cumulative,
)
from bodo.utils.utils import (
    debug_prints,
    incref,
    is_assign,
    is_call_assign,
    is_expr,
    is_null_pointer,
    is_var_assign,
    sanitize_varname,
    unliteral_all,
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


@intrinsic
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


@numba.jit
def get_agg_udf_addr(name):
    """ Resolve address of cfunc given by its symbol name """
    with numba.objmode(addr="int64"):
        addr = gb_agg_cfunc_addr[name]
    return addr


class AggUDFStruct(object):
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
        """ Set the cfuncs that are called from C++ that apply regular UDFs """
        assert self.regular_udfs and self.regular_udf_cfuncs is None
        self.regular_udf_cfuncs = [update_cb, combine_cb, eval_cb]

    def set_general_cfunc(self, general_udf_cb):
        """ Set the cfunc that is called from C++ that applies general UDFs """
        assert self.general_udfs and self.general_udf_cfunc is None
        self.general_udf_cfunc = general_udf_cb


AggFuncStruct = namedtuple("AggFuncStruct", ["func", "ftype"])


# !!! IMPORTANT: this is supposed to match the positions in
# Bodo_FTypes::FTypeEnum in _groupby.cpp
supported_agg_funcs = [
    "no_op",  # needed to ensure that 0 value isn't matched with any function
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
    "var",
    "std",
    "udf",
    "gen_udf",
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
    if func_name in {"var", "std"}:
        func = pytypes.SimpleNamespace()
        func.ftype = func_name
        func.fname = func_name
        func.ncols_pre_shuffle = 3
        func.ncols_post_shuffle = 4
        return func
    if func_name in {"first", "last"}:
        # We don't have a function definition for first/last, and it is not needed
        # for the groupby C++ codepath, so we just use a dummy object.
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
    if func_name in supported_agg_funcs[:-8]:
        func = pytypes.SimpleNamespace()
        func.ftype = func_name
        func.fname = func_name
        func.ncols_pre_shuffle = 1
        func.ncols_post_shuffle = 1
        skipdropna = True
        shift_periods_t = 1
        if isinstance(rhs, ir.Expr):
            for erec in rhs.kws:
                # Type checking should be handled at the overload/bound_func level.
                # Any unknown kws at this stage should be naming the
                # output column.
                if func_name in list_cumulative:
                    if erec[0] == "skipna":
                        skipdropna = guard(find_const, func_ir, erec[1])
                        if not isinstance(skipdropna, bool):
                            raise BodoError(
                                "For {} argument of skipna should be a boolean".format(
                                    func_name
                                )
                            )
                if func_name == "nunique":
                    if erec[0] == "dropna":
                        skipdropna = guard(find_const, func_ir, erec[1])
                        if not isinstance(skipdropna, bool):
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
        func.skipdropna = skipdropna
        func.periods = shift_periods_t
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
            if f_name not in bodo.ir.aggregate.supported_transform_funcs[::]:
                raise BodoError(f"unsupported transform function {f_name}")
            # TODO: It could be user-defined
            func.transform_func = supported_agg_funcs.index(f_name)
        else:
            func.transform_func = supported_agg_funcs.index("no_op")
        return func

    # agg case
    assert func_name in ["agg", "aggregate"]

    # NOTE: assuming typemap is provided here
    # TODO: refactor old pivot code that doesn't provide typemap
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
        return [
            get_agg_func_udf(
                func_ir,
                get_literal_value(typemap[f_val.name])[1],
                rhs,
                series_type,
                typemap,
            )
            for f_val in kws.values()
        ]

    # multi-function tuple case
    if isinstance(agg_func_typ, types.BaseTuple):
        funcs = []
        lambda_count = 0
        for t in agg_func_typ.types:
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
                if func.fname == "<lambda>":
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
    """get udf value for agg call"""
    if isinstance(f_val, str):
        return get_agg_func(func_ir, f_val, rhs, series_type, typemap)
    if bodo.utils.typing.is_builtin_function(f_val):
        # Builtin function case (e.g. df.groupby("B").agg(sum))
        func_name = bodo.utils.typing.get_builtin_function_name(f_val)
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


# combine function takes the reduce vars in reverse order of their user
@numba.njit(no_cpython_wrapper=True)
def _var_combine(ssqdm_a, mean_a, nobs_a, ssqdm_b, mean_b, nobs_b):  # pragma: no cover
    nobs = nobs_a + nobs_b
    mean_x = (nobs_a * mean_a + nobs_b * mean_b) / nobs
    delta = mean_b - mean_a
    M2 = ssqdm_a + ssqdm_b + delta * delta * nobs_a * nobs_b / nobs
    return M2, mean_x, nobs


# XXX: njit doesn't work when bodo.jit() is used for agg_func in hiframes
# @numba.njit
def __special_combine(*args):  # pragma: no cover
    return


@infer_global(__special_combine)
class SpecialCombineTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        return signature(types.void, *unliteral_all(args))


@lower_builtin(__special_combine, types.VarArg(types.Any))
def lower_special_combine(context, builder, sig, args):
    return context.get_dummy_value()


class Aggregate(ir.Stmt):
    def __init__(
        self,
        df_out,
        df_in,
        key_names,
        gb_info_in,
        gb_info_out,
        out_key_vars,
        df_out_vars,  # NOTE: does not include output key vars (stored in out_key_vars)
        df_in_vars,
        key_arrs,
        input_has_index,
        same_index,
        return_key,
        loc,
        dropna=True,
        pivot_arr=None,
        pivot_values=None,
        is_crosstab=False,
    ):
        # name of output dataframe (just for printing purposes)
        self.df_out = df_out
        # name of input dataframe (just for printing purposes)
        self.df_in = df_in
        # key name (for printing)
        self.key_names = key_names
        # Store full info on how input columns and aggregation functions map
        # to output columns
        # gb_info_in: maps in_col -> list of (func, out) where out is output
        # column name for non-pivot/crosstab case or list of output columns otherwise
        # Examples:
        # For `df.groupby("A").agg({"B": "min", "C": "max"})`
        # gb_info_in = {"B": [(min_func, "B")]}
        # For `df.groupby("A").agg(
        #    E=pd.NamedAgg(column="B", aggfunc=lambda A: A.sum()),
        #    F=pd.NamedAgg(column="B", aggfunc="min"),
        # )`
        # gb_info_in = {"B": [(lambda_func, "E"), (min_func, "F")]}
        self.gb_info_in = gb_info_in
        # gb_info_out: maps out_col -> (in_col, func)
        self.gb_info_out = gb_info_out
        self.out_key_vars = out_key_vars
        self.df_out_vars = df_out_vars
        self.df_in_vars = df_in_vars
        self.key_arrs = key_arrs
        self.input_has_index = input_has_index
        self.same_index = same_index
        self.return_key = return_key
        self.loc = loc
        self.dropna = dropna
        # pivot_table handling
        self.pivot_arr = pivot_arr
        self.pivot_values = pivot_values
        self.is_crosstab = is_crosstab

    def __repr__(self):  # pragma: no cover
        out_cols = ""
        for (c, v) in self.df_out_vars.items():
            out_cols += "'{}':{}, ".format(c, v.name)
        df_out_str = "{}{{{}}}".format(self.df_out, out_cols)
        in_cols = ""
        for (c, v) in self.df_in_vars.items():
            in_cols += "'{}':{}, ".format(c, v.name)
        df_in_str = "{}{{{}}}".format(self.df_in, in_cols)
        pivot = (
            "pivot {}:{}".format(self.pivot_arr.name, self.pivot_values)
            if self.pivot_arr is not None
            else ""
        )
        key_names = ",".join(self.key_names)
        key_arrnames = ",".join([v.name for v in self.key_arrs])
        return "aggregate: {} = {} [key: {}:{}] {}".format(
            df_out_str, df_in_str, key_names, key_arrnames, pivot
        )

    def remove_out_col(self, out_col_name):
        """ Remove output column and associated input column if no longer needed """

        self.df_out_vars.pop(out_col_name)

        in_col, _ = self.gb_info_out.pop(out_col_name)
        if in_col is None and not self.is_crosstab:
            # size operation where output column doesn't have associated input column
            return

        outs = self.gb_info_in[in_col]
        if self.pivot_arr is not None:
            # pivot case, output corresponds to pivot values
            self.pivot_values.remove(out_col_name)
            # each (input_col, func) can produce multiple output columns
            for i, (func, out_cols) in enumerate(outs):
                try:
                    out_cols.remove(out_col_name)
                    if len(out_cols) == 0:
                        outs.pop(i)
                        break
                except ValueError:  # not found
                    continue
        else:
            for i, (func, out_col) in enumerate(outs):
                if out_col == out_col_name:
                    outs.pop(i)
                    break
        if len(outs) == 0:  # input column is no longer needed
            self.gb_info_in.pop(in_col)
            self.df_in_vars.pop(in_col)


def aggregate_usedefs(aggregate_node, use_set=None, def_set=None):
    if use_set is None:
        use_set = set()
    if def_set is None:
        def_set = set()

    # key array and input columns are used
    use_set.update({v.name for v in aggregate_node.key_arrs})
    use_set.update({v.name for v in aggregate_node.df_in_vars.values()})

    if aggregate_node.pivot_arr is not None:
        use_set.add(aggregate_node.pivot_arr.name)

    # output columns are defined
    def_set.update({v.name for v in aggregate_node.df_out_vars.values()})

    # return key is defined
    if aggregate_node.out_key_vars is not None:
        def_set.update({v.name for v in aggregate_node.out_key_vars})

    return numba.core.analysis._use_defs_result(usemap=use_set, defmap=def_set)


numba.core.analysis.ir_extension_usedefs[Aggregate] = aggregate_usedefs


def remove_dead_aggregate(
    aggregate_node, lives_no_aliases, lives, arg_aliases, alias_map, func_ir, typemap
):
    """remove dead input/output columns and agg functions"""

    # find dead output columns
    dead_cols = [
        col_name
        for col_name, col_var in aggregate_node.df_out_vars.items()
        if col_var.name not in lives
    ]

    # remove dead output columns and their corresponding input columns/functions
    for cname in dead_cols:
        aggregate_node.remove_out_col(cname)

    out_key_vars = aggregate_node.out_key_vars
    if out_key_vars is not None and all(v.name not in lives for v in out_key_vars):
        aggregate_node.out_key_vars = None

    # TODO: test agg remove
    # remove empty aggregate node
    if len(aggregate_node.df_out_vars) == 0 and aggregate_node.out_key_vars is None:
        return None

    return aggregate_node


ir_utils.remove_dead_extensions[Aggregate] = remove_dead_aggregate


def get_copies_aggregate(aggregate_node, typemap):
    # aggregate doesn't generate copies, it just kills the output columns
    kill_set = set(v.name for v in aggregate_node.df_out_vars.values())
    if aggregate_node.out_key_vars is not None:
        kill_set.update({v.name for v in aggregate_node.out_key_vars})
    return set(), kill_set


ir_utils.copy_propagate_extensions[Aggregate] = get_copies_aggregate


def apply_copies_aggregate(
    aggregate_node, var_dict, name_var_table, typemap, calltypes, save_copies
):
    """apply copy propagate in aggregate node"""
    for i in range(len(aggregate_node.key_arrs)):
        aggregate_node.key_arrs[i] = replace_vars_inner(
            aggregate_node.key_arrs[i], var_dict
        )

    for col_name in list(aggregate_node.df_in_vars.keys()):
        aggregate_node.df_in_vars[col_name] = replace_vars_inner(
            aggregate_node.df_in_vars[col_name], var_dict
        )
    for col_name in list(aggregate_node.df_out_vars.keys()):
        aggregate_node.df_out_vars[col_name] = replace_vars_inner(
            aggregate_node.df_out_vars[col_name], var_dict
        )

    if aggregate_node.out_key_vars is not None:
        for i in range(len(aggregate_node.out_key_vars)):
            aggregate_node.out_key_vars[i] = replace_vars_inner(
                aggregate_node.out_key_vars[i], var_dict
            )

    if aggregate_node.pivot_arr is not None:
        aggregate_node.pivot_arr = replace_vars_inner(
            aggregate_node.pivot_arr, var_dict
        )


ir_utils.apply_copy_propagate_extensions[Aggregate] = apply_copies_aggregate


def visit_vars_aggregate(aggregate_node, callback, cbdata):
    if debug_prints():  # pragma: no cover
        print("visiting aggregate vars for:", aggregate_node)
        print("cbdata: ", sorted(cbdata.items()))

    for i in range(len(aggregate_node.key_arrs)):
        aggregate_node.key_arrs[i] = visit_vars_inner(
            aggregate_node.key_arrs[i], callback, cbdata
        )

    for col_name in list(aggregate_node.df_in_vars.keys()):
        aggregate_node.df_in_vars[col_name] = visit_vars_inner(
            aggregate_node.df_in_vars[col_name], callback, cbdata
        )
    for col_name in list(aggregate_node.df_out_vars.keys()):
        aggregate_node.df_out_vars[col_name] = visit_vars_inner(
            aggregate_node.df_out_vars[col_name], callback, cbdata
        )

    if aggregate_node.out_key_vars is not None:
        for i in range(len(aggregate_node.out_key_vars)):
            aggregate_node.out_key_vars[i] = visit_vars_inner(
                aggregate_node.out_key_vars[i], callback, cbdata
            )

    if aggregate_node.pivot_arr is not None:
        aggregate_node.pivot_arr = visit_vars_inner(
            aggregate_node.pivot_arr, callback, cbdata
        )


# add call to visit aggregate variable
ir_utils.visit_vars_extensions[Aggregate] = visit_vars_aggregate


def aggregate_array_analysis(aggregate_node, equiv_set, typemap, array_analysis):
    # empty aggregate nodes should be deleted in remove dead
    assert (
        len(aggregate_node.df_out_vars) > 0
        or aggregate_node.out_key_vars is not None
        or aggregate_node.is_crosstab
    ), "empty aggregate in array analysis"

    # arrays of input df have same size in first dimension as key array
    all_shapes = []
    for key_arr in aggregate_node.key_arrs:
        col_shape = equiv_set.get_shape(key_arr)
        if col_shape:
            all_shapes.append(col_shape[0])

    if aggregate_node.pivot_arr is not None:
        col_shape = equiv_set.get_shape(aggregate_node.pivot_arr)
        if col_shape:
            all_shapes.append(col_shape[0])

    for col_var in aggregate_node.df_in_vars.values():
        col_shape = equiv_set.get_shape(col_var)
        if col_shape:
            all_shapes.append(col_shape[0])

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    # create correlations for output arrays
    # arrays of output df have same size in first dimension
    # gen size variable for an output column
    post = []
    all_shapes = []
    out_vars = list(aggregate_node.df_out_vars.values())
    if aggregate_node.out_key_vars is not None:
        out_vars.extend(aggregate_node.out_key_vars)

    for col_var in out_vars:
        typ = typemap[col_var.name]
        shape = array_analysis._gen_shape_call(equiv_set, col_var, typ.ndim, None, post)
        equiv_set.insert_equiv(col_var, shape)
        all_shapes.append(shape[0])
        equiv_set.define(col_var, set())

    if len(all_shapes) > 1:
        equiv_set.insert_equiv(*all_shapes)

    return [], post


numba.parfors.array_analysis.array_analysis_extensions[
    Aggregate
] = aggregate_array_analysis


def aggregate_distributed_analysis(aggregate_node, array_dists):
    # input columns have same distribution
    in_dist = Distribution.OneD
    for col_var in aggregate_node.df_in_vars.values():
        in_dist = Distribution(min(in_dist.value, array_dists[col_var.name].value))

    # key arrays
    for key_arr in aggregate_node.key_arrs:
        in_dist = Distribution(min(in_dist.value, array_dists[key_arr.name].value))

    # pivot case
    if aggregate_node.pivot_arr is not None:
        in_dist = Distribution(
            min(in_dist.value, array_dists[aggregate_node.pivot_arr.name].value)
        )
        array_dists[aggregate_node.pivot_arr.name] = in_dist

    for col_var in aggregate_node.df_in_vars.values():
        array_dists[col_var.name] = in_dist
    for key_arr in aggregate_node.key_arrs:
        array_dists[key_arr.name] = in_dist

    # output columns have same distribution
    out_dist = Distribution.OneD_Var
    for col_var in aggregate_node.df_out_vars.values():
        # output dist might not be assigned yet
        if col_var.name in array_dists:
            out_dist = Distribution(
                min(out_dist.value, array_dists[col_var.name].value)
            )

    if aggregate_node.out_key_vars is not None:
        for col_var in aggregate_node.out_key_vars:
            if col_var.name in array_dists:
                out_dist = Distribution(
                    min(out_dist.value, array_dists[col_var.name].value)
                )

    # out dist should meet input dist (e.g. REP in causes REP out)
    out_dist = Distribution(min(out_dist.value, in_dist.value))
    for col_var in aggregate_node.df_out_vars.values():
        array_dists[col_var.name] = out_dist

    if aggregate_node.out_key_vars is not None:
        for cvar in aggregate_node.out_key_vars:
            array_dists[cvar.name] = out_dist

    # output can cause input REP
    if out_dist != Distribution.OneD_Var:
        for key_arr in aggregate_node.key_arrs:
            array_dists[key_arr.name] = out_dist
        # pivot case
        if aggregate_node.pivot_arr is not None:
            array_dists[aggregate_node.pivot_arr.name] = out_dist
        for col_var in aggregate_node.df_in_vars.values():
            array_dists[col_var.name] = out_dist


distributed_analysis.distributed_analysis_extensions[
    Aggregate
] = aggregate_distributed_analysis


def build_agg_definitions(agg_node, definitions=None):
    if definitions is None:
        definitions = defaultdict(list)

    for col_var in agg_node.df_out_vars.values():
        definitions[col_var.name].append(agg_node)

    if agg_node.out_key_vars is not None:
        for cvar in agg_node.out_key_vars:
            definitions[cvar.name].append(agg_node)

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
    parallel = False
    if array_dists is not None:
        parallel = True
        for v in (
            list(agg_node.df_in_vars.values())
            + list(agg_node.df_out_vars.values())
            + agg_node.key_arrs
        ):
            if (
                array_dists[v.name] != distributed_pass.Distribution.OneD
                and array_dists[v.name] != distributed_pass.Distribution.OneD_Var
            ):
                parallel = False
            # TODO: check supported types
            # if (typemap[v.name] != types.Array(types.intp, 1, 'C')
            #         and typemap[v.name] != types.Array(types.float64, 1, 'C')):
            #     raise ValueError(
            #         "Only int64 and float64 columns are currently supported in aggregate")
            # if (typemap[left_key_var.name] != types.Array(types.intp, 1, 'C')
            #     or typemap[right_key_var.name] != types.Array(types.intp, 1, 'C')):
            # raise ValueError("Only int64 keys are currently supported in aggregate")

    # TODO: rebalance if output distributions are 1D instead of 1D_Var

    # TODO: handle key column being part of output

    key_typs = tuple(typemap[v.name] for v in agg_node.key_arrs)
    # get column variables
    in_col_vars = [v for (n, v) in agg_node.df_in_vars.items()]
    out_col_vars = [v for (n, v) in agg_node.df_out_vars.items()]
    # get column types
    # Type of input columns in the same order as passed to C++ and can include
    # repetition. C++ receives one input column for each (in_col,func) pair
    # and the same input column might not necessarily appear in consecutive
    # positions in that list (see NamedAgg examples)
    in_col_typs = []
    funcs = []
    # See comment about use of gb_info_in vs gb_info_out in gen_top_level_agg_func
    # when laying out input columns and functions for C++
    if agg_node.pivot_arr is not None:
        for in_col, outs in agg_node.gb_info_in.items():
            for func, _ in outs:
                if in_col is not None:
                    in_col_typs.append(typemap[agg_node.df_in_vars[in_col].name])
                funcs.append(func)
    else:
        for in_col, func in agg_node.gb_info_out.values():
            if in_col is not None:
                in_col_typs.append(typemap[agg_node.df_in_vars[in_col].name])
            funcs.append(func)
    out_col_typs = tuple(typemap[v.name] for v in out_col_vars)

    pivot_typ = (
        types.none if agg_node.pivot_arr is None else typemap[agg_node.pivot_arr.name]
    )
    arg_typs = tuple(
        key_typs + tuple(typemap[v.name] for v in in_col_vars) + (pivot_typ,)
    )

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
        agg_node.input_has_index,
        in_col_typs,
        out_col_typs,
        typingctx,
        targetctx,
        pivot_typ,
        agg_node.pivot_values,
        agg_node.is_crosstab,
    )

    top_level_func = gen_top_level_agg_func(
        agg_node,
        in_col_typs,
        out_col_typs,
        parallel,
        udf_func_struct,
    )
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
            "pivot_groupby_and_aggregate": pivot_groupby_and_aggregate,
            "compute_node_partition_by_hash": compute_node_partition_by_hash,
            "info_from_table": info_from_table,
            "info_to_array": info_to_array,
            "delete_info_decref_array": delete_info_decref_array,
            "delete_table": delete_table,
            "add_agg_cfunc_sym": add_agg_cfunc_sym,
            "get_agg_udf_addr": get_agg_udf_addr,
            "delete_table_decref_arrays": delete_table_decref_arrays,
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

    f_block = compile_to_numba_ir(
        top_level_func,
        glbs,
        typingctx=typingctx,
        targetctx=targetctx,
        arg_typs=arg_typs,
        typemap=typemap,
        calltypes=calltypes,
    ).blocks.popitem()[1]

    nodes = []
    if agg_node.pivot_arr is None:
        scope = agg_node.key_arrs[0].scope
        loc = agg_node.loc
        none_var = ir.Var(scope, mk_unique_var("dummy_none"), loc)
        typemap[none_var.name] = types.none
        nodes.append(ir.Assign(ir.Const(None, loc), none_var, loc))
        in_col_vars.append(none_var)
    else:
        in_col_vars.append(agg_node.pivot_arr)

    replace_arg_nodes(f_block, agg_node.key_arrs + in_col_vars)

    tuple_assign = f_block.body[-3]
    assert (
        is_assign(tuple_assign)
        and isinstance(tuple_assign.value, ir.Expr)
        and tuple_assign.value.op == "build_tuple"
    )
    nodes += f_block.body[:-3]

    out_vars = list(agg_node.df_out_vars.values())
    if agg_node.out_key_vars is not None:
        out_vars += agg_node.out_key_vars

    for i, var in enumerate(out_vars):
        out_var = tuple_assign.value.items[i]
        nodes.append(ir.Assign(out_var, var, var.loc))

    return nodes


distributed_pass.distributed_run_extensions[Aggregate] = agg_distributed_run


def get_numba_set(dtype):
    pass


@infer_global(get_numba_set)
class GetNumbaSetTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        arr = args[0]
        dtype = (
            types.Tuple([t.dtype for t in arr.types])
            if isinstance(arr, types.BaseTuple)
            else arr.dtype
        )
        if isinstance(arr, types.BaseTuple) and len(arr.types) == 1:
            dtype = arr.types[0].dtype
        return signature(types.Set(dtype), *args)


@lower_builtin(get_numba_set, types.Any)
def lower_get_numba_set(context, builder, sig, args):
    return numba.cpython.setobj.set_empty_constructor(context, builder, sig, args)


@infer_global(bool)
class BoolNoneTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        val_t = args[0]
        if val_t == types.none:
            return signature(types.boolean, *args)


@lower_builtin(bool, types.none)
def lower_column_mean_impl(context, builder, sig, args):
    res = context.compile_internal(builder, lambda a: False, sig, args)
    return res  # impl_ret_untracked(context, builder, sig.return_type, res)


def setitem_array_with_str(arr, i, v):  # pragma: no cover
    return


@overload(setitem_array_with_str)
def setitem_array_with_str_overload(arr, i, val):
    if arr == string_array_type:

        def setitem_str_arr(arr, i, val):  # pragma: no cover
            arr[i] = val

        return setitem_str_arr

    # return_key == False case where val could be string resulting in typing
    # issue, no need to set
    if val == string_type:
        return lambda arr, i, val: None

    def setitem_impl(arr, i, val):  # pragma: no cover
        arr[i] = val

    return setitem_impl


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
    elif isinstance(t, BooleanArrayType):
        return "bodo.libs.bool_arr_ext.init_bool_array(np.empty(0, np.bool_), np.empty(0, np.uint8))"
    elif isinstance(t, StringArrayType):
        return "pre_alloc_string_array(1, 1)"
    elif isinstance(t, BinaryArrayType):
        return "pre_alloc_binary_array(1, 1)"
    elif t == ArrayItemArrayType(string_array_type):
        return "pre_alloc_array_item_array(1, (1, 1), string_array_type)"
    elif isinstance(t, DecimalArrayType):
        return "alloc_decimal_array(1, {}, {})".format(t.precision, t.scale)
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
        return "np.empty(1, {})".format(_get_np_dtype(t.dtype))


def _get_np_dtype(t):
    if t == types.bool_:
        return "np.bool_"
    if t == types.NPDatetime("ns"):
        return "dt64_dtype"
    if t == types.NPTimedelta("ns"):
        return "td64_dtype"
    return "np.{}".format(t)


def gen_update_cb(
    udf_func_struct,
    allfuncs,
    n_keys,
    data_in_typs_,
    out_data_typs,
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

    func_text = (
        "def bodo_gb_udf_update_local{}(in_table, out_table, row_to_group):\n".format(
            label_suffix
        )
    )
    func_text += "    if is_null_pointer(in_table):\n"  # this is dummy call
    func_text += "        return\n"

    # get redvars data types
    func_text += "    data_redvar_dummy = ({}{})\n".format(
        ",".join(["np.empty(1, {})".format(_get_np_dtype(t)) for t in red_var_typs]),
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
    assert len(redvar_offsets) == n_red_vars

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
        func_text += "    redvar_arr_{} = info_to_array(info_from_table(out_table, {}), data_redvar_dummy[{}])\n".format(
            i, redvar_offsets[i], i
        )
        # incref needed so that arrays aren't deleted after this function exits
        func_text += "    incref(redvar_arr_{})\n".format(i)
        func_text += "    redvar_arr_{}.fill(init_vals[{}])\n".format(i, i)
    func_text += "    redvars = ({}{})\n".format(
        ",".join(["redvar_arr_{}".format(i) for i in range(n_red_vars)]),
        "," if n_red_vars == 1 else "",
    )

    func_text += "\n"
    for i in range(n_data_cols):
        func_text += "    data_in_{} = info_to_array(info_from_table(in_table, {}), data_in_dummy[{}])\n".format(
            i, in_col_offsets[i], i
        )
        # incref needed so that arrays aren't deleted after this function exits
        func_text += "    incref(data_in_{})\n".format(i)
    func_text += "    data_in = ({}{})\n".format(
        ",".join(["data_in_{}".format(i) for i in range(n_data_cols)]),
        "," if n_data_cols == 1 else "",
    )

    func_text += "\n"
    func_text += "    for i in range(len(data_in_0)):\n"
    func_text += "        w_ind = row_to_group[i]\n"
    func_text += "        if w_ind != -1:\n"
    func_text += (
        "            __update_redvars(redvars, data_in, w_ind, i, pivot_arr=None)\n"
    )

    loc_vars = {}
    exec(
        func_text,
        {
            "bodo": bodo,
            "np": np,
            "pd": pd,
            "info_to_array": info_to_array,
            "info_from_table": info_from_table,
            "incref": incref,
            "pre_alloc_string_array": pre_alloc_string_array,
            "__init_func": udf_func_struct.init_func,
            "__update_redvars": udf_func_struct.update_all_func,
            "is_null_pointer": is_null_pointer,
            "dt64_dtype": np.dtype("datetime64[ns]"),
            "td64_dtype": np.dtype("timedelta64[ns]"),
        },
        loc_vars,
    )
    return loc_vars["bodo_gb_udf_update_local{}".format(label_suffix)]


def gen_combine_cb(udf_func_struct, allfuncs, n_keys, out_data_typs, label_suffix):
    """
    Generates a Python function (to be compiled into a numba cfunc) which
    does the "combine" step of an agg operation. The code is for a specific
    groupby.agg(). The combine step combines the received aggregated data from
    other processes.
    """
    red_var_typs = udf_func_struct.var_typs
    n_red_vars = len(red_var_typs)

    func_text = (
        "def bodo_gb_udf_combine{}(in_table, out_table, row_to_group):\n".format(
            label_suffix
        )
    )
    func_text += "    if is_null_pointer(in_table):\n"  # this is dummy call
    func_text += "        return\n"

    # get redvars data types
    func_text += "    data_redvar_dummy = ({}{})\n".format(
        ",".join(["np.empty(1, {})".format(_get_np_dtype(t)) for t in red_var_typs]),
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
        func_text += "    redvar_arr_{} = info_to_array(info_from_table(out_table, {}), data_redvar_dummy[{}])\n".format(
            i, redvar_offsets_out[i], i
        )
        # incref needed so that arrays aren't deleted after this function exits
        func_text += "    incref(redvar_arr_{})\n".format(i)
        func_text += "    redvar_arr_{}.fill(init_vals[{}])\n".format(i, i)
    func_text += "    redvars = ({}{})\n".format(
        ",".join(["redvar_arr_{}".format(i) for i in range(n_red_vars)]),
        "," if n_red_vars == 1 else "",
    )

    func_text += "\n"
    for i in range(n_red_vars):
        func_text += "    recv_redvar_arr_{} = info_to_array(info_from_table(in_table, {}), data_redvar_dummy[{}])\n".format(
            i, redvar_offsets_in[i], i
        )
        # incref needed so that arrays aren't deleted after this function exits
        func_text += "    incref(recv_redvar_arr_{})\n".format(i)
    func_text += "    recv_redvars = ({}{})\n".format(
        ",".join(["recv_redvar_arr_{}".format(i) for i in range(n_red_vars)]),
        "," if n_red_vars == 1 else "",
    )

    func_text += "\n"
    if n_red_vars:  # if there is a parfor
        func_text += "    for i in range(len(recv_redvar_arr_0)):\n"
        func_text += "        w_ind = row_to_group[i]\n"
        func_text += "        __combine_redvars(redvars, recv_redvars, w_ind, i, pivot_arr=None)\n"

    loc_vars = {}
    exec(
        func_text,
        {
            "np": np,
            "info_to_array": info_to_array,
            "info_from_table": info_from_table,
            "incref": incref,
            "__init_func": udf_func_struct.init_func,
            "__combine_redvars": udf_func_struct.combine_all_func,
            "is_null_pointer": is_null_pointer,
            "dt64_dtype": np.dtype("datetime64[ns]"),
            "td64_dtype": np.dtype("timedelta64[ns]"),
        },
        loc_vars,
    )
    return loc_vars["bodo_gb_udf_combine{}".format(label_suffix)]


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

    func_text = "def bodo_gb_udf_eval{}(table):\n".format(label_suffix)
    func_text += "    if is_null_pointer(table):\n"  # this is dummy call
    func_text += "        return\n"

    func_text += "    data_redvar_dummy = ({}{})\n".format(
        ",".join(["np.empty(1, {})".format(_get_np_dtype(t)) for t in red_var_typs]),
        "," if len(red_var_typs) == 1 else "",
    )

    func_text += "    out_data_dummy = ({}{})\n".format(
        ",".join(
            ["np.empty(1, {})".format(_get_np_dtype(t.dtype)) for t in out_data_typs]
        ),
        "," if len(out_data_typs) == 1 else "",
    )

    for i in range(n_red_vars):
        func_text += "    redvar_arr_{} = info_to_array(info_from_table(table, {}), data_redvar_dummy[{}])\n".format(
            i, redvar_offsets[i], i
        )
        # incref needed so that arrays aren't deleted after this function exits
        func_text += "    incref(redvar_arr_{})\n".format(i)
    func_text += "    redvars = ({}{})\n".format(
        ",".join(["redvar_arr_{}".format(i) for i in range(n_red_vars)]),
        "," if n_red_vars == 1 else "",
    )

    func_text += "\n"
    for i in range(n_data_cols):
        func_text += "    data_out_{} = info_to_array(info_from_table(table, {}), out_data_dummy[{}])\n".format(
            i, data_out_offsets[i], i
        )
        # incref needed so that arrays aren't deleted after this function exits
        func_text += "    incref(data_out_{})\n".format(i)
    func_text += "    data_out = ({}{})\n".format(
        ",".join(["data_out_{}".format(i) for i in range(n_data_cols)]),
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
            "info_to_array": info_to_array,
            "info_from_table": info_from_table,
            "incref": incref,
            "__eval_res": udf_func_struct.eval_all_func,
            "is_null_pointer": is_null_pointer,
            "dt64_dtype": np.dtype("datetime64[ns]"),
            "td64_dtype": np.dtype("timedelta64[ns]"),
        },
        loc_vars,
    )
    return loc_vars["bodo_gb_udf_eval{}".format(label_suffix)]


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
    out_col_offsets = (
        []
    )  # offsets of general UDF output columns in the table received from C++
    for i, f in enumerate(allfuncs):
        if f.ftype == "gen_udf":
            out_col_offsets.append(col_offset)
            col_offset += 1
        elif f.ftype != "udf":
            col_offset += f.ncols_post_shuffle
        else:
            # udfs in post_shuffle table have one column for output plus redvars
            col_offset += f.n_redvars + 1

    func_text = (
        "def bodo_gb_apply_general_udfs{}(num_groups, in_table, out_table):\n".format(
            label_suffix
        )
    )
    func_text += "    if num_groups == 0:\n"  # this is dummy call
    func_text += "        return\n"
    for i, func in enumerate(udf_func_struct.general_udf_funcs):
        func_text += "    # col {}\n".format(i)
        func_text += "    out_col = info_to_array(info_from_table(out_table, {}), out_col_{}_typ)\n".format(
            out_col_offsets[i], i
        )
        # incref needed so that array isn't deleted after this function exits
        func_text += "    incref(out_col)\n"
        func_text += "    for j in range(num_groups):\n"
        func_text += "        in_col = info_to_array(info_from_table(in_table, {}*num_groups + j), in_col_{}_typ)\n".format(
            i, i
        )
        # incref needed so that array isn't deleted after this function exits
        func_text += "        incref(in_col)\n"
        func_text += "        out_col[j] = func_{}(pd.Series(in_col))  # func returns scalar\n".format(
            i
        )

    glbs = {
        "pd": pd,
        "info_to_array": info_to_array,
        "info_from_table": info_from_table,
        "incref": incref,
    }
    gen_udf_offset = 0
    for i, func in enumerate(allfuncs):
        if func.ftype != "gen_udf":
            continue
        func = udf_func_struct.general_udf_funcs[gen_udf_offset]
        glbs["func_{}".format(gen_udf_offset)] = func
        glbs["in_col_{}_typ".format(gen_udf_offset)] = in_col_typs[
            func_idx_to_in_col[i]
        ]
        glbs["out_col_{}_typ".format(gen_udf_offset)] = out_col_typs[i]
        gen_udf_offset += 1
    loc_vars = {}
    exec(func_text, glbs, loc_vars)
    f = loc_vars["bodo_gb_apply_general_udfs{}".format(label_suffix)]
    c_sig = types.void(types.int64, types.voidptr, types.voidptr)
    return numba.cfunc(c_sig, nopython=True)(f)


def gen_top_level_agg_func(
    agg_node,
    in_col_typs,
    out_col_typs,
    parallel,
    udf_func_struct,
):
    """create the top level aggregation function by generating text"""
    has_pivot_value = agg_node.pivot_arr is not None
    # If we output the index then we need to remove it from the list of variables.
    if agg_node.same_index:
        assert agg_node.input_has_index
    if agg_node.pivot_values is None:
        n_pivot = 1
    else:
        n_pivot = len(agg_node.pivot_values)

    # These are the names of arguments of agg_top function
    # NOTE that input columns are not repeated in the arg list
    key_arg_names = tuple("key_" + sanitize_varname(c) for c in agg_node.key_names)
    in_arg_names = {
        c: "in_{}".format(sanitize_varname(c))
        for c in agg_node.gb_info_in.keys()
        if c is not None
    }
    # out_names are the names of the output columns which are returned by agg_top
    out_names = {c: "out_" + sanitize_varname(c) for c in agg_node.gb_info_out.keys()}

    n_keys = len(agg_node.key_names)
    key_args = ", ".join(key_arg_names)
    in_args = ", ".join(in_arg_names.values())
    if in_args != "":
        in_args = ", " + in_args
    # If we put the index as argument, then it is the last argument of the
    # function.
    func_text = "def agg_top({}{}{}, pivot_arr):\n".format(
        key_args, in_args, ", index_arg" if agg_node.input_has_index else ""
    )

    # convert arrays to table
    # For each unique function applied to a given input column (i.e. each
    # (in_col, func) pair) we add the column to the table_info passed to C++
    # (in other words input columns can be repeated in the table info)
    if has_pivot_value:
        # For pivot case we can't use gb_info_out because multiple output
        # columns match to the same input column for a given function, but we
        # only want to add one input per (input_col, func)
        # Note that this path also works for df.groupby without NamedAggs
        # It doesn't work for NamedAgg case because inputs might not be consecutive
        in_names_all = []
        for in_col, outs in agg_node.gb_info_in.items():
            # in_col can be None in case of "size" operation
            if in_col is not None:
                for func, _ in outs:
                    in_names_all.append(in_arg_names[in_col])
    else:
        # For NamedAgg case the order in which inputs are provided has to
        # match the output order, so we use agg_node.gb_info_out instead
        in_names_all = tuple(
            in_arg_names[in_col]
            for in_col, _ in agg_node.gb_info_out.values()
            if in_col is not None
        )
    all_arrs = key_arg_names + tuple(in_names_all)
    func_text += "    info_list = [{}{}{}]\n".format(
        ", ".join("array_to_info({})".format(a) for a in all_arrs),
        ", array_to_info(index_arg)" if agg_node.input_has_index else "",
        ", array_to_info(pivot_arr)" if agg_node.is_crosstab else "",
    )
    func_text += "    table = arr_info_list_to_table(info_list)\n"

    for i, c in enumerate(agg_node.gb_info_out.keys()):
        out_name = out_names[c] + "_dummy"
        out_col_typ = out_col_typs[i]
        # Shift, Min, and Max maintain the same output and input type
        # We handle these separately with Categorical data because if a
        # CategoricalArrayType doesn't have types known at compile time then
        # we must associate it with another DType that has it's categories
        # set at runtime. The existing approach for other types just uses
        # Typerefs and those can't be resolved.
        in_col, func = agg_node.gb_info_out[c]
        if (
            isinstance(func, pytypes.SimpleNamespace)
            and func.fname in ["min", "max", "shift"]
            and isinstance(out_col_typ, bodo.CategoricalArrayType)
        ):
            func_text += "    {} = {}\n".format(out_name, in_arg_names[in_col])
        else:
            func_text += "    {} = {}\n".format(
                out_name, _gen_dummy_alloc(out_col_typ, i, False)
            )
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
    # number of redvars for each udf function
    udf_ncols = []
    skipdropna = False
    shift_periods = 1
    num_cum_funcs = 0
    transform_func = 0

    if not has_pivot_value:
        funcs = [func for _, func in agg_node.gb_info_out.values()]
    else:
        funcs = [func for func, _ in outs for outs in agg_node.gb_info_in.values()]
    for f_idx, func in enumerate(funcs):
        func_offsets.append(len(allfuncs))
        if func.ftype in {"median", "nunique"}:
            # these operations require shuffle at the beginning, so a
            # local aggregation followed by combine is not necessary
            do_combine = False
        if func.ftype in list_cumulative:
            num_cum_funcs += 1
        if hasattr(func, "skipdropna"):
            skipdropna = func.skipdropna
        if func.ftype == "shift":
            shift_periods = func.periods
            do_combine = False  # See median/nunique note ^
        if func.ftype in {"transform"}:
            transform_func = func.transform_func
            do_combine = False  # See median/nunique note ^
        allfuncs.append(func)
        func_idx_to_in_col.append(f_idx)
        if func.ftype == "udf":
            udf_ncols.append(func.n_redvars)
        elif func.ftype == "gen_udf":
            udf_ncols.append(0)
            do_combine = False
    func_offsets.append(len(allfuncs))
    if agg_node.is_crosstab:
        assert (
            len(agg_node.gb_info_out) == n_pivot
        ), "invalid number of groupby outputs for pivot"
    else:
        assert (
            len(agg_node.gb_info_out) == len(allfuncs) * n_pivot
        ), "invalid number of groupby outputs"
    if num_cum_funcs > 0:
        assert num_cum_funcs == len(
            allfuncs
        ), "Cannot mix cumulative operations with other aggregation functions"
        do_combine = False  # same as median and nunique

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
                    out_col_typs,
                    do_combine,
                    func_idx_to_in_col,
                    udf_label,
                )
            )
            cpp_cb_combine = numba.cfunc(c_sig, nopython=True)(
                gen_combine_cb(
                    udf_func_struct, allfuncs, n_keys, out_col_typs, udf_label
                )
            )
            cpp_cb_eval = numba.cfunc("void(voidptr)", nopython=True)(
                gen_eval_cb(udf_func_struct, allfuncs, n_keys, out_col_typs, udf_label)
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
                out_col_typs,
                func_idx_to_in_col,
                udf_label,
            )
            udf_func_struct.set_general_cfunc(cpp_cb_general)

        # generate a dummy (empty) table with correct type info for
        # output columns and reduction variables corresponding to UDFs,
        # so that the C++ runtime can allocate arrays
        udf_names_dummy = []
        redvar_offset = 0
        i = 0
        for out_name, f in zip(out_names.values(), allfuncs):
            if f.ftype in ("udf", "gen_udf"):
                udf_names_dummy.append(out_name + "_dummy")
                for j in range(redvar_offset, redvar_offset + udf_ncols[i]):
                    udf_names_dummy.append("data_redvar_dummy_" + str(j))
                redvar_offset += udf_ncols[i]
                i += 1

        if udf_func_struct.regular_udfs:
            red_var_typs = udf_func_struct.var_typs
            for i, t in enumerate(red_var_typs):
                func_text += "    data_redvar_dummy_{} = np.empty(1, {})\n".format(
                    i, _get_np_dtype(t)
                )

        func_text += "    out_info_list_dummy = [{}]\n".format(
            ", ".join("array_to_info({})".format(a) for a in udf_names_dummy)
        )
        func_text += (
            "    udf_table_dummy = arr_info_list_to_table(out_info_list_dummy)\n"
        )

        # include cfuncs in library and insert a dummy call to make sure symbol
        # is not discarded
        if udf_func_struct.regular_udfs:
            func_text += "    add_agg_cfunc_sym(cpp_cb_update, '{}')\n".format(
                cpp_cb_update.native_name
            )
            func_text += "    add_agg_cfunc_sym(cpp_cb_combine, '{}')\n".format(
                cpp_cb_combine.native_name
            )
            func_text += "    add_agg_cfunc_sym(cpp_cb_eval, '{}')\n".format(
                cpp_cb_eval.native_name
            )
            func_text += "    cpp_cb_update_addr = get_agg_udf_addr('{}')\n".format(
                cpp_cb_update.native_name
            )
            func_text += "    cpp_cb_combine_addr = get_agg_udf_addr('{}')\n".format(
                cpp_cb_combine.native_name
            )
            func_text += "    cpp_cb_eval_addr = get_agg_udf_addr('{}')\n".format(
                cpp_cb_eval.native_name
            )
        else:
            func_text += "    cpp_cb_update_addr = 0\n"
            func_text += "    cpp_cb_combine_addr = 0\n"
            func_text += "    cpp_cb_eval_addr = 0\n"
        if udf_func_struct.general_udfs:
            cfunc = udf_func_struct.general_udf_cfunc
            gb_agg_cfunc[cfunc.native_name] = cfunc
            gb_agg_cfunc_addr[cfunc.native_name] = cfunc.address
            func_text += "    add_agg_cfunc_sym(cpp_cb_general, '{}')\n".format(
                cfunc.native_name
            )
            func_text += "    cpp_cb_general_addr = get_agg_udf_addr('{}')\n".format(
                cfunc.native_name
            )
        else:
            func_text += "    cpp_cb_general_addr = 0\n"
    else:
        # if there are no udfs we don't need udf table, so just create
        # an empty one-column table
        func_text += "    udf_table_dummy = arr_info_list_to_table([array_to_info(np.empty(1))])\n"
        func_text += "    cpp_cb_update_addr = 0\n"
        func_text += "    cpp_cb_combine_addr = 0\n"
        func_text += "    cpp_cb_eval_addr = 0\n"
        func_text += "    cpp_cb_general_addr = 0\n"

    # NOTE: adding extra zero to make sure the list is never empty to avoid Numba
    # typing issues
    func_text += "    ftypes = np.array([{}, 0], dtype=np.int32)\n".format(
        ", ".join([str(supported_agg_funcs.index(f.ftype)) for f in allfuncs] + ["0"])
    )
    func_text += "    func_offsets = np.array({}, dtype=np.int32)\n".format(
        str(func_offsets)
    )
    if len(udf_ncols) > 0:
        func_text += "    udf_ncols = np.array({}, dtype=np.int32)\n".format(
            str(udf_ncols)
        )
    else:
        func_text += "    udf_ncols = np.array([0], np.int32)\n"  # dummy
    # call C++ groupby
    # We pass the logical arguments to the function (skipdropna, return_key, same_index, ...)

    if has_pivot_value:
        func_text += "    arr_type = coerce_to_array({})\n".format(
            agg_node.pivot_values
        )
        func_text += "    arr_info = array_to_info(arr_type)\n"
        func_text += "    dispatch_table = arr_info_list_to_table([arr_info])\n"
        func_text += "    pivot_info = array_to_info(pivot_arr)\n"
        func_text += "    dispatch_info = arr_info_list_to_table([pivot_info])\n"
        func_text += "    out_table = pivot_groupby_and_aggregate(table, {}, dispatch_table, dispatch_info, {}," " ftypes.ctypes, func_offsets.ctypes, udf_ncols.ctypes, {}, {}, {}, {}, {}, cpp_cb_update_addr, cpp_cb_combine_addr, cpp_cb_eval_addr, udf_table_dummy)\n".format(
            n_keys,
            agg_node.input_has_index,
            parallel,
            agg_node.is_crosstab,
            skipdropna,
            agg_node.return_key,
            agg_node.same_index,
        )
        func_text += "    delete_info_decref_array(pivot_info)\n"
        func_text += "    delete_info_decref_array(arr_info)\n"
    else:
        func_text += "    out_table = groupby_and_aggregate(table, {}, {}," " ftypes.ctypes, func_offsets.ctypes, udf_ncols.ctypes, {}, {}, {}, {}, {}, {}, {}, cpp_cb_update_addr, cpp_cb_combine_addr, cpp_cb_eval_addr, cpp_cb_general_addr, udf_table_dummy)\n".format(
            n_keys,
            agg_node.input_has_index,
            parallel,
            skipdropna,
            shift_periods,
            transform_func,
            agg_node.return_key,
            agg_node.same_index,
            agg_node.dropna,
        )

    idx = 0
    if agg_node.return_key:
        for i, key_name in enumerate(key_arg_names):
            func_text += (
                "    {} = info_to_array(info_from_table(out_table, {}), {})\n".format(
                    key_name, idx, key_name
                )
            )
            idx += 1
    for out_name in out_names.values():
        func_text += (
            "    {} = info_to_array(info_from_table(out_table, {}), {})\n".format(
                out_name, idx, out_name + "_dummy"
            )
        )
        idx += 1
    # The index as last argument in output as well.
    if agg_node.same_index:
        func_text += "    out_index_arg = info_to_array(info_from_table(out_table, {}), index_arg)\n".format(
            idx
        )
        idx += 1
    # clean up
    func_text += (
        f"    ev_clean = bodo.utils.tracing.Event('tables_clean_up', {parallel})\n"
    )
    func_text += "    delete_table_decref_arrays(table)\n"
    func_text += "    delete_table_decref_arrays(udf_table_dummy)\n"
    func_text += "    delete_table(out_table)\n"
    func_text += f"    ev_clean.finalize()\n"

    ret_names = tuple(out_names.values())
    if agg_node.return_key:
        ret_names += tuple(key_arg_names)
    func_text += "    return ({},{})\n".format(
        ", ".join(ret_names), " out_index_arg," if agg_node.same_index else ""
    )

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    agg_top = loc_vars["agg_top"]
    return agg_top


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
    # The new Numba 0.50 inliner requires the pipline state itselft to be a member of
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
    inline_overload_pass.run_pass(pm)

    series_pass = bodo.transforms.series_pass.SeriesPass(
        f_ir, typingctx, targetctx, typemap, calltypes, {}, False
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


class RegularUDFGenerator(object):
    """ Generate code that applies UDFs to all columns that use them """

    def __init__(
        self,
        in_col_types,
        out_col_types,
        pivot_typ,
        pivot_values,
        is_crosstab,
        typingctx,
        targetctx,
    ):
        self.in_col_types = in_col_types
        self.out_col_types = out_col_types
        self.pivot_typ = pivot_typ
        self.pivot_values = pivot_values
        self.is_crosstab = is_crosstab
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
        in_series_typ = SeriesType(in_col_typ.dtype, in_col_typ, None, string_type)
        # compile UDF to IR
        f_ir, pm = compile_to_optimized_ir(
            func, (in_series_typ,), self.typingctx, self.targetctx
        )
        f_ir._definitions = build_definitions(f_ir.blocks)

        assert len(f_ir.blocks) == 1 and 0 in f_ir.blocks, (
            "only simple functions" " with one block supported for aggregation"
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

        self.all_vartypes = (
            self.all_vartypes * len(self.pivot_values)
            if self.pivot_values is not None
            else self.all_vartypes
        )

        self.all_reduce_vars = (
            self.all_reduce_vars * len(self.pivot_values)
            if self.pivot_values is not None
            else self.all_reduce_vars
        )

        init_func = gen_init_func(
            self.all_init_nodes,
            self.all_reduce_vars,
            self.all_vartypes,
            self.typingctx,
            self.targetctx,
        )
        update_all_func = gen_all_update_func(
            self.all_update_funcs,
            self.all_vartypes,
            self.in_col_types,
            self.redvar_offsets,
            self.typingctx,
            self.targetctx,
            self.pivot_typ,
            self.pivot_values,
            self.is_crosstab,
        )
        combine_all_func = gen_all_combine_func(
            self.all_combine_funcs,
            self.all_vartypes,
            self.redvar_offsets,
            self.typingctx,
            self.targetctx,
            self.pivot_typ,
            self.pivot_values,
        )
        eval_all_func = gen_all_eval_func(
            self.all_eval_funcs,
            self.all_vartypes,
            self.redvar_offsets,
            self.out_col_types,
            self.typingctx,
            self.targetctx,
            self.pivot_values,
        )
        return (
            self.all_vartypes,
            init_func,
            update_all_func,
            combine_all_func,
            eval_all_func,
        )


class GeneralUDFGenerator(object):
    # TODO pivot and crosstab
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
    input_has_index,
    in_col_types,
    out_col_types,
    typingctx,
    targetctx,
    pivot_typ,
    pivot_values,
    is_crosstab,
):
    if is_crosstab and len(in_col_types) == 0:
        # use dummy int input type for crosstab since doesn't have input
        in_col_types = [types.Array(types.intp, 1, "C")]

    # Construct list of (input col type, aggregation func)
    # If multiple functions will be applied to the same input column, that
    # input column will appear multiple times in the generated list
    typ_and_func = []
    for t, f in zip(in_col_types, agg_func):
        typ_and_func.append((t, f))

    # Create UDF code generators
    regular_udf_gen = RegularUDFGenerator(
        in_col_types,
        out_col_types,
        pivot_typ,
        pivot_values,
        is_crosstab,
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
        except:
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
                i_uses |= stmt_uses
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
    reduce_var_types,
    in_col_types,
    redvar_offsets,
    typingctx,
    targetctx,
    pivot_typ,
    pivot_values,
    is_crosstab,
):

    out_num_cols = len(update_funcs)
    in_num_cols = len(in_col_types)
    if pivot_values is not None:
        assert in_num_cols == 1

    func_text = "def update_all_f(redvar_arrs, data_in, w_ind, i, pivot_arr):\n"
    if pivot_values is not None:
        num_redvars = redvar_offsets[in_num_cols]
        func_text += "  pv = pivot_arr[i]\n"
        for j, pv in enumerate(pivot_values):
            el = "el" if j != 0 else ""
            func_text += "  {}if pv == '{}':\n".format(el, pv)  # TODO: non-string pivot
            init_offset = num_redvars * j
            redvar_access = ", ".join(
                [
                    "redvar_arrs[{}][w_ind]".format(i)
                    for i in range(
                        init_offset + redvar_offsets[0], init_offset + redvar_offsets[1]
                    )
                ]
            )
            data_access = "data_in[0][i]"
            if is_crosstab:  # TODO: crosstab with values arg
                data_access = "0"
            func_text += "    {} = update_vars_0({}, {})\n".format(
                redvar_access, redvar_access, data_access
            )
    else:
        for j in range(out_num_cols):
            redvar_access = ", ".join(
                [
                    "redvar_arrs[{}][w_ind]".format(i)
                    for i in range(redvar_offsets[j], redvar_offsets[j + 1])
                ]
            )
            if redvar_access:  # if there is a parfor
                func_text += "  {} = update_vars_{}({},  data_in[{}][i])\n".format(
                    redvar_access, j, redvar_access, 0 if in_num_cols == 1 else j
                )
    func_text += "  return\n"

    glbs = {}
    for i, f in enumerate(update_funcs):
        glbs["update_vars_{}".format(i)] = f
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
    pivot_typ,
    pivot_values,
):

    reduce_arrs_tup_typ = types.Tuple(
        [types.Array(t, 1, "C") for t in reduce_var_types]
    )
    arg_typs = (
        reduce_arrs_tup_typ,
        reduce_arrs_tup_typ,
        types.intp,
        types.intp,
        pivot_typ,
    )

    num_cols = len(redvar_offsets) - 1
    num_redvars = redvar_offsets[num_cols]

    func_text = "def combine_all_f(redvar_arrs, recv_arrs, w_ind, i, pivot_arr):\n"

    if pivot_values is not None:
        assert num_cols == 1
        for k in range(len(pivot_values)):
            init_offset = num_redvars * k
            redvar_access = ", ".join(
                [
                    "redvar_arrs[{}][w_ind]".format(i)
                    for i in range(
                        init_offset + redvar_offsets[0], init_offset + redvar_offsets[1]
                    )
                ]
            )
            recv_access = ", ".join(
                [
                    "recv_arrs[{}][i]".format(i)
                    for i in range(
                        init_offset + redvar_offsets[0], init_offset + redvar_offsets[1]
                    )
                ]
            )
            func_text += "  {} = combine_vars_0({}, {})\n".format(
                redvar_access, redvar_access, recv_access
            )
    else:
        for j in range(num_cols):
            redvar_access = ", ".join(
                [
                    "redvar_arrs[{}][w_ind]".format(i)
                    for i in range(redvar_offsets[j], redvar_offsets[j + 1])
                ]
            )
            recv_access = ", ".join(
                [
                    "recv_arrs[{}][i]".format(i)
                    for i in range(redvar_offsets[j], redvar_offsets[j + 1])
                ]
            )
            if recv_access:  # if there is a parfor
                func_text += "  {} = combine_vars_{}({}, {})\n".format(
                    redvar_access, j, redvar_access, recv_access
                )
    func_text += "  return\n"
    glbs = {}
    for i, f in enumerate(combine_funcs):
        glbs["combine_vars_{}".format(i)] = f
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
    reduce_var_types,
    redvar_offsets,
    out_col_typs,
    typingctx,
    targetctx,
    pivot_values,
):

    reduce_arrs_tup_typ = types.Tuple(
        [types.Array(t, 1, "C") for t in reduce_var_types]
    )
    out_col_typs = types.Tuple(out_col_typs)

    num_cols = len(redvar_offsets) - 1

    num_redvars = redvar_offsets[num_cols]

    func_text = "def eval_all_f(redvar_arrs, out_arrs, j):\n"

    if pivot_values is not None:
        assert num_cols == 1
        for j in range(len(pivot_values)):
            init_offset = num_redvars * j
            redvar_access = ", ".join(
                [
                    "redvar_arrs[{}][j]".format(i)
                    for i in range(
                        init_offset + redvar_offsets[0], init_offset + redvar_offsets[1]
                    )
                ]
            )
            func_text += "  out_arrs[{}][j] = eval_vars_0({})\n".format(
                j, redvar_access
            )
    else:
        for j in range(num_cols):
            redvar_access = ", ".join(
                [
                    "redvar_arrs[{}][j]".format(i)
                    for i in range(redvar_offsets[j], redvar_offsets[j + 1])
                ]
            )
            func_text += "  out_arrs[{}][j] = eval_vars_{}({})\n".format(
                j, j, redvar_access
            )
    func_text += "  return\n"
    glbs = {}
    for i, f in enumerate(eval_funcs):
        glbs["eval_vars_{}".format(i)] = f
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
            if is_call_assign(stmt) and (
                guard(find_callname, f_ir, stmt.value)
                == ("__special_combine", "bodo.ir.aggregate")
            ):
                args = stmt.value.args
                l_argnames = []
                r_argnames = []
                for v in args[:-1]:
                    ind = redvars.index(v.name)
                    ignore_redvar_inds.append(ind)
                    l_argnames.append("v{}".format(ind))
                    r_argnames.append("in{}".format(ind))
                comb_name = "__special_combine__{}".format(len(special_combines))
                func_text += "    ({},) = {}({})\n".format(
                    ", ".join(l_argnames), comb_name, ", ".join(l_argnames + r_argnames)
                )
                dummy_call = ir.Expr.call(args[-1], [], (), bl.loc)
                sp_func = guard(find_callname, f_ir, dummy_call)
                # XXX: only var supported for now
                # TODO: support general functions
                assert sp_func == ("_var_combine", "bodo.ir.aggregate")
                sp_func = bodo.ir.aggregate._var_combine
                special_combines[comb_name] = sp_func

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
        ", ".join(["v{}".format(i) for i in range(num_red_vars)])
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
        func_text = "    v{} += in{}\n".format(ind, ind)
    if isinstance(var_def, ir.Expr) and var_def.op == "call":
        fdef = guard(find_callname, f_ir, var_def)
        if fdef == ("min", "builtins"):
            func_text = "    v{} = min(v{}, in{})\n".format(ind, ind, ind)
        if fdef == ("max", "builtins"):
            func_text = "    v{} = max(v{}, in{})\n".format(ind, ind, ind)
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
    var_types = [pm.typemap[v] for v in redvars]

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

    redvar_in_names = ["v{}".format(i) for i in range(num_red_vars)]
    in_names = ["in{}".format(i) for i in range(num_in_vars)]

    func_text = "def agg_update({}):\n".format(", ".join(redvar_in_names + in_names))
    func_text += "    __update_redvars()\n"
    func_text += "    return {}".format(
        ", ".join(["v{}".format(i) for i in range(num_red_vars)])
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
