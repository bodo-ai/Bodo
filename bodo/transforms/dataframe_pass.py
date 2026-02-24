"""
converts data frame operations to Series and Array operations
as much as possible to provide implementation and enable optimization.
Creates specialized IR nodes for complex operations like Join.
"""

import datetime
import operator
import warnings

import numba
import numpy as np
import pandas as pd
from numba.core import ir, types
from numba.core.ir_utils import (
    find_build_sequence,
    find_callname,
    find_const,
    find_topo_order,
    get_definition,
    guard,
    mk_unique_var,
)
from pandas.core.common import flatten

import bodo
import bodo.hiframes.dataframe_impl  # noqa # side effect: install DataFrame overloads
import bodo.hiframes.pd_rolling_ext
from bodo.hiframes.dataframe_indexing import (
    DataFrameIatType,
    DataFrameILocType,
    DataFrameLocType,
    DataFrameType,
)
from bodo.hiframes.pd_groupby_ext import (
    DataFrameGroupByType,
    _get_groupby_apply_udf_out_type,
)
from bodo.hiframes.pd_index_ext import RangeIndexType
from bodo.hiframes.pd_multi_index_ext import MultiIndexType
from bodo.hiframes.pd_series_ext import HeterogeneousSeriesType, SeriesType
from bodo.hiframes.table import TableType
from bodo.ir.aggregate import get_agg_func
from bodo.libs.bool_arr_ext import BooleanArrayType
from bodo.utils.transform import (
    compile_func_single_block,
    gen_const_tup,
    get_build_sequence_vars,
    get_call_expr_arg,
    get_const_value,
    get_const_value_inner,
    replace_func,
)
from bodo.utils.typing import (
    INDEX_SENTINEL,
    BodoError,
    ColNamesMetaType,
    gen_bodosql_case_func,
    get_index_data_arr_types,
    get_literal_value,
    get_overload_const_func,
    get_overload_const_list,
    get_overload_const_str,
    get_overload_const_tuple,
    handle_bodosql_case_init_code,
    is_literal_type,
    is_overload_constant_list,
    is_overload_constant_str,
    is_overload_constant_tuple,
    is_overload_none,
    list_cumulative,
    unwrap_typeref,
)
from bodo.utils.utils import (
    find_build_tuple,
    get_getsetitem_index_var,
    is_array_typ,
    is_assign,
    is_expr,
    sanitize_varname,
)

binary_op_names = [f.__name__ for f in bodo.hiframes.pd_series_ext.series_binary_ops]


class DataFramePass:
    """
    This pass converts data frame operations to Series and Array operations as much as
    possible to provide implementation and enable optimization. Creates specialized
    IR nodes for complex operations like Join.
    """

    def __init__(self, func_ir, typingctx, targetctx, typemap, calltypes):
        self.func_ir = func_ir
        self.typingctx = typingctx
        self.targetctx = targetctx
        self.typemap = typemap
        self.calltypes = calltypes
        # Loc object of current location being translated
        self.curr_loc = self.func_ir.loc

    def _run_assign(self, assign):
        lhs = assign.target
        rhs = assign.value

        if isinstance(rhs, ir.Expr):
            if rhs.op == "getattr":
                return self._run_getattr(assign, rhs)

            if rhs.op == "binop":
                return self._run_binop(assign, rhs)

            # # XXX handling inplace_binop similar to binop for now
            # # TODO handle inplace alignment
            if rhs.op == "inplace_binop":
                return self._run_binop(assign, rhs)

            if rhs.op == "unary":
                return self._run_unary(assign, rhs)

            # replace getitems on dataframe
            if rhs.op in ("getitem", "static_getitem"):
                return self._run_getitem(assign, rhs)

            if rhs.op == "call":
                return self._run_call(assign, lhs, rhs)

        return None

    def _run_getitem(self, assign, rhs):
        nodes = []
        index_var = get_getsetitem_index_var(rhs, self.typemap, nodes)
        index_typ = self.typemap[index_var.name]
        target = rhs.value
        target_typ = self.typemap[target.name]

        # inline DataFrame getitem
        if isinstance(target_typ, DataFrameType):
            impl = bodo.hiframes.dataframe_indexing.df_getitem_overload(
                target_typ, index_typ
            )
            return replace_func(self, impl, [target, index_var], pre_nodes=nodes)

        # inline DataFrame.iloc[] getitem
        if isinstance(target_typ, DataFrameILocType):
            impl = bodo.hiframes.dataframe_indexing.overload_iloc_getitem(
                target_typ, index_typ
            )
            return replace_func(self, impl, [target, index_var], pre_nodes=nodes)

        # inline DataFrame.loc[] getitem
        if isinstance(target_typ, DataFrameLocType):
            impl = bodo.hiframes.dataframe_indexing.overload_loc_getitem(
                target_typ, index_typ
            )
            return replace_func(self, impl, [target, index_var], pre_nodes=nodes)

        # inline DataFrame.iat[] getitem
        if isinstance(target_typ, DataFrameIatType):
            impl = bodo.hiframes.dataframe_indexing.overload_iat_getitem(
                target_typ, index_typ
            )
            return replace_func(self, impl, [target, index_var], pre_nodes=nodes)

        return None

    def _run_setitem(self, inst):
        target_typ = self.typemap[inst.target.name]
        nodes = []
        index_var = get_getsetitem_index_var(inst, self.typemap, nodes)
        index_typ = self.typemap[index_var.name]

        # inline DataFrame.iat[] setitem
        if isinstance(target_typ, DataFrameIatType):
            impl = bodo.hiframes.dataframe_indexing.overload_iat_setitem(
                target_typ, index_typ, self.typemap[inst.value.name]
            )
            return replace_func(
                self, impl, [inst.target, index_var, inst.value], pre_nodes=nodes
            )

        return None

    def _run_getattr(self, assign, rhs):
        rhs_type = self.typemap[rhs.value.name]  # get type of rhs value "df"

        # replace attribute access with overload
        if isinstance(rhs_type, DataFrameType) and rhs.attr in (
            "values",
            "size",
            "shape",
            "empty",
        ):
            overload_name = "overload_dataframe_" + rhs.attr
            overload_func = getattr(bodo.hiframes.dataframe_impl, overload_name)
            impl = overload_func(rhs_type)
            return replace_func(self, impl, [rhs.value])

        # S = df.A (get dataframe column)
        # Note we skip has_runtime_cols without error checking because this should
        # have already been caught in DataFrameAttribute.
        # TODO: check invalid df.Attr?
        if (
            isinstance(rhs_type, DataFrameType)
            and not rhs_type.has_runtime_cols
            and rhs.attr in rhs_type.columns
        ):
            nodes = []
            col_name = rhs.attr
            arr = self._get_dataframe_data(rhs.value, col_name, nodes)
            index = self._get_dataframe_index(rhs.value, nodes)
            name = ir.Var(arr.scope, mk_unique_var("df_col_name"), arr.loc)
            self.typemap[name.name] = types.StringLiteral(col_name)
            nodes.append(ir.Assign(ir.Const(col_name, arr.loc), name, arr.loc))
            return replace_func(
                self,
                eval(
                    "lambda arr, index, name: bodo.hiframes.pd_series_ext.init_series(arr, index, name)"
                ),
                [arr, index, name],
                pre_nodes=nodes,
            )

        # level selection in multi-level df
        # Note we skip has_runtime_cols without error checking because this should
        # have already been caught in DataFrameAttribute.
        if (
            isinstance(rhs_type, DataFrameType)
            and not rhs_type.has_runtime_cols
            and len(rhs_type.columns) > 0
            and isinstance(rhs_type.columns[0], tuple)
            and any(v[0] == rhs.attr for v in rhs_type.columns)
        ):
            nodes = []
            index = self._get_dataframe_index(rhs.value, nodes)
            new_names = []
            new_data = []
            for i, v in enumerate(rhs_type.columns):
                if v[0] != rhs.attr:
                    continue
                # output names are str in 2 level case, not tuple
                # TODO: test more than 2 levels
                new_names.append(v[1] if len(v) == 2 else v[1:])
                new_data.append(self._get_dataframe_data(rhs.value, v, nodes))
            _init_df = _gen_init_df_dataframe_pass(new_names, "index")
            return nodes + compile_func_single_block(
                _init_df, new_data + [index], assign.target, self
            )

        # replace df.iloc._obj with df
        if (
            isinstance(
                rhs_type, (DataFrameILocType, DataFrameLocType, DataFrameIatType)
            )
            and rhs.attr == "_obj"
        ):
            assign.value = guard(get_definition, self.func_ir, rhs.value).value
            return [assign]

        return None

    def _run_binop(self, assign, rhs):
        """transform ir.Expr.binop nodes"""

        arg1, arg2 = rhs.lhs, rhs.rhs
        typ1, typ2 = self.typemap[arg1.name], self.typemap[arg2.name]
        if not (isinstance(typ1, DataFrameType) or isinstance(typ2, DataFrameType)):
            return None

        if rhs.fn in bodo.hiframes.pd_series_ext.series_binary_ops:
            overload_func = bodo.hiframes.dataframe_impl.create_binary_op_overload(
                rhs.fn
            )
            impl = overload_func(typ1, typ2)
            return replace_func(self, impl, [arg1, arg2])

        if rhs.fn in bodo.hiframes.pd_series_ext.series_inplace_binary_ops:
            overload_func = (
                bodo.hiframes.dataframe_impl.create_inplace_binary_op_overload(rhs.fn)
            )
            impl = overload_func(typ1, typ2)
            return replace_func(self, impl, [arg1, arg2])

        return [assign]  # XXX should reach here, check it properly

    def _run_unary(self, assign, rhs):
        arg = rhs.value
        typ = self.typemap[arg.name]

        if isinstance(typ, DataFrameType):
            assert rhs.fn in bodo.hiframes.pd_series_ext.series_unary_ops
            overload_func = bodo.hiframes.dataframe_impl.create_unary_op_overload(
                rhs.fn
            )
            impl = overload_func(typ)
            return replace_func(self, impl, (arg,))

        return None

    def _run_call(self, assign, lhs, rhs):
        fdef = guard(find_callname, self.func_ir, rhs, self.typemap)
        if fdef is None:
            from numba.stencils.stencil import StencilFunc

            # could be make_function from list comprehension which is ok
            func_def = guard(get_definition, self.func_ir, rhs.func)
            if isinstance(func_def, ir.Expr) and func_def.op == "make_function":
                return None
            # ignore objmode block calls
            if isinstance(func_def, ir.Const) and isinstance(
                func_def.value, numba.core.dispatcher.ObjModeLiftedWith
            ):
                return None
            if isinstance(func_def, ir.Global) and isinstance(
                func_def.value, StencilFunc
            ):
                return None
            # Numba generates const function calls for some operators sometimes instead
            # of expressions. This normalizes them to regular unary/binop expressions
            # so that Bodo transforms handle them properly.
            if (
                isinstance(func_def, (ir.Const, ir.FreeVar, ir.Global))
                and func_def.value in numba.core.utils.OPERATORS_TO_BUILTINS
            ):  # pragma: no cover
                return self._convert_op_call_to_expr(assign, rhs, func_def.value)
            # input to _bodo_groupby_apply_impl() is a UDF dispatcher
            elif isinstance(func_def, ir.Arg) and isinstance(
                self.typemap[rhs.func.name], types.Dispatcher
            ):
                return [assign]
            # If df.apply fails to inline an externally compiled JIT
            # function, then we may have a CPUDispatcher
            # (see test_df_apply_heterogeneous_series).
            elif isinstance(func_def, ir.Const) and isinstance(
                self.typemap[rhs.func.name], types.Dispatcher
            ):
                return [assign]
            # Cases like dtype(value) in np.linspace implementation
            elif isinstance(
                func_def, (ir.Const, ir.FreeVar, ir.Global)
            ) and func_def.value in (
                int,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
                float,
                np.float32,
                np.float64,
                complex,
                np.complex64,
                np.complex128,
            ):
                return [assign]

            warnings.warn("function call couldn't be found for dataframe analysis")
            return None
        else:
            func_name, func_mod = fdef

        # df binary operators call builtin array operators directly,
        # convert to binop node to be parallelized by PA
        # TODO: add support to PA
        # if (func_mod == '_operator' and func_name in binary_op_names
        #         and len(rhs.args) > 0
        #         and (is_array_typ(self.typemap[rhs.args[0].name])
        #             or is_array_typ(self.typemap[rhs.args[1].name]))):
        #     func = getattr(operator, func_name)
        #     return [ir.Assign(ir.Expr.binop(
        #         func, rhs.args[0], rhs.args[1], rhs.loc),
        #         assign.target,
        #         rhs.loc)]

        if fdef == ("len", "builtins") and self._is_df_var(rhs.args[0]):
            return self._run_call_len(lhs, rhs.args[0], assign)

        if fdef == ("set_df_col", "bodo.hiframes.dataframe_impl"):
            return self._run_call_set_df_column(assign, lhs, rhs)

        if fdef == (
            "__bodosql_replace_columns_dummy",
            "bodo.hiframes.dataframe_impl",
        ):  # pragma: no cover
            return self._df_pass_run_call_bodosql_replace_columns(assign, lhs, rhs)

        if fdef == ("get_dataframe_table", "bodo.hiframes.pd_dataframe_ext"):
            # If we loaded the table from a DataFrame we may be able to eliminate
            # the original DataFrame.
            df_var = rhs.args[0]
            df_def = guard(get_definition, self.func_ir, df_var)
            call_def = guard(find_callname, self.func_ir, df_def, self.typemap)
            if not self._is_updated_df(df_var.name) and call_def == (
                "init_dataframe",
                "bodo.hiframes.pd_dataframe_ext",
            ):
                assert self.typemap[df_var.name].is_table_format, (
                    "set_table_data called on a DataFrame without a table format"
                )
                # Extra the original table to enable eliminating the original DataFrame
                tuple_var = df_def.args[0]
                tuple_def = guard(get_definition, self.func_ir, tuple_var)
                if is_expr(tuple_def, "build_tuple"):
                    assert len(tuple_def.items), (
                        "init_dataframe with Table called on a tuple with multiple data values"
                    )
                    assign.value = tuple_def.items[0]

        if fdef == ("get_dataframe_data", "bodo.hiframes.pd_dataframe_ext"):
            df_var = rhs.args[0]
            ind = guard(find_const, self.func_ir, rhs.args[1])
            var_def = guard(get_definition, self.func_ir, df_var)
            call_def = guard(find_callname, self.func_ir, var_def, self.typemap)
            if not self._is_updated_df(df_var.name) and call_def == (
                "init_dataframe",
                "bodo.hiframes.pd_dataframe_ext",
            ):
                seq_info = guard(find_build_sequence, self.func_ir, var_def.args[0])
                if seq_info is not None:
                    if self.typemap[df_var.name].is_table_format:
                        # If we have a table format we replace the get_dataframe_data
                        # with a get_table_data call so we can perform dead column
                        # elimination.
                        table_var = seq_info[0][0]
                        return compile_func_single_block(
                            eval(
                                "lambda table, ind: bodo.hiframes.table.get_table_data(table, ind)"
                            ),
                            (table_var, rhs.args[1]),
                            lhs,
                            self,
                        )
                    assign.value = seq_info[0][ind]

        if fdef == ("get_dataframe_index", "bodo.hiframes.pd_dataframe_ext"):
            df_var = rhs.args[0]
            var_def = guard(get_definition, self.func_ir, df_var)
            call_def = guard(find_callname, self.func_ir, var_def, self.typemap)
            if call_def == ("init_dataframe", "bodo.hiframes.pd_dataframe_ext"):
                assign.value = var_def.args[1]

        # make sure n_table_cols_t argument is passed as a constant since it's necessary
        # after optimizations (when input table arg may be eliminated)
        if fdef == ("logical_table_to_table", "bodo.hiframes.table"):
            n_cols = guard(find_const, self.func_ir, rhs.args[3])
            if n_cols is not None:
                self.typemap.pop(rhs.args[3].name)
                self.typemap[rhs.args[3].name] = types.IntegerLiteral(n_cols)

        # Optimize out get_table_data() calls which are outputs of
        # logical_table_to_table().
        if fdef == ("get_table_data", "bodo.hiframes.table"):
            table_var = rhs.args[0]
            ind = guard(find_const, self.func_ir, rhs.args[1])
            table_def = guard(get_definition, self.func_ir, table_var)
            call_def = guard(find_callname, self.func_ir, table_def, self.typemap)
            if call_def == (
                "logical_table_to_table",
                "bodo.hiframes.table",
            ):
                # first argument is input "table", which could be in actual table
                # format or tuple of arrays.
                in_table_var = table_def.args[0]
                in_table_type = self.typemap[in_table_var.name]
                n_in_table_cols = (
                    len(in_table_type.arr_types)
                    if isinstance(in_table_type, TableType)
                    else len(in_table_type.types)
                )
                extra_arrs_var = table_def.args[1]
                in_col_inds = self.typemap[table_def.args[2].name].instance_type.meta
                in_ind = in_col_inds[ind]

                # column is in extra arrays
                if in_ind >= n_in_table_cols:
                    return compile_func_single_block(
                        eval(
                            f"lambda extra_arrs: extra_arrs[{in_ind - n_in_table_cols}]"
                        ),
                        (extra_arrs_var,),
                        lhs,
                        self,
                    )

                # column is in the input table
                return compile_func_single_block(
                    eval(
                        f"lambda table: bodo.hiframes.table.get_table_data(table, {in_ind})"
                        if isinstance(in_table_type, TableType)
                        else f"lambda table: table[{in_ind}]"
                    ),
                    (in_table_var,),
                    lhs,
                    self,
                )

        # inline get_dataframe_all_data() to enable optimizations
        if fdef == ("get_dataframe_all_data", "bodo.hiframes.pd_dataframe_ext"):
            df_var = rhs.args[0]
            df_type = self.typemap[df_var.name]
            if df_type.is_table_format:
                # just return the data table in table format case
                nodes = []
                in_table_var = self._get_dataframe_table(df_var, nodes)
                assign.value = in_table_var
                nodes.append(assign)
                return nodes

            # create a tuple of data arrays
            nodes = []
            in_vars = [
                self._get_dataframe_data(df_var, c, nodes) for c in df_type.columns
            ]
            tuple_var = ir.Var(
                df_var.scope, mk_unique_var("$table_tuple_var"), df_var.loc
            )
            self.typemap[tuple_var.name] = types.BaseTuple.from_types(df_type.data)
            tuple_call = ir.Expr.build_tuple(in_vars, df_var.loc)
            nodes.append(ir.Assign(tuple_call, tuple_var, df_var.loc))
            assign.value = tuple_var
            nodes.append(assign)
            return nodes

        if fdef == ("join_dummy", "bodo.hiframes.pd_dataframe_ext"):
            return self._run_call_join(assign, lhs, rhs)

        if fdef == ("bodosql_case_placeholder", "bodo.utils.typing"):
            return self._run_call_bodosql_case_placeholder(rhs)

        # df/series/groupby.pipe()
        if (
            isinstance(func_mod, ir.Var)
            and isinstance(
                self.typemap[func_mod.name],
                (
                    DataFrameType,
                    bodo.hiframes.pd_series_ext.SeriesType,
                    DataFrameGroupByType,
                ),
            )
            and func_name == "pipe"
        ):
            return self._run_call_pipe(rhs, func_mod)

        if isinstance(func_mod, ir.Var) and isinstance(
            self.typemap[func_mod.name], DataFrameType
        ):
            return self._run_call_dataframe(
                assign, assign.target, rhs, func_mod, func_name
            )

        if isinstance(func_mod, ir.Var) and isinstance(
            self.typemap[func_mod.name], DataFrameGroupByType
        ):
            return self._run_call_groupby(
                assign, assign.target, rhs, func_mod, func_name
            )

        if fdef == ("crosstab_dummy", "bodo.hiframes.pd_groupby_ext"):
            return self._run_call_crosstab(assign, lhs, rhs)

        if fdef == ("sort_values_dummy", "bodo.hiframes.pd_dataframe_ext"):
            return self._run_call_df_sort_values(assign, lhs, rhs)

        if fdef == ("itertuples_dummy", "bodo.hiframes.pd_dataframe_ext"):
            return self._run_call_df_itertuples(assign, lhs, rhs)

        if fdef == ("query_dummy", "bodo.hiframes.pd_dataframe_ext"):
            return self._run_call_query(assign, lhs, rhs)

        # match dummy function created in _run_call_query and raise error if output of
        # expression in df.query() is not a boolean Series
        if fdef == ("_check_query_series_bool", "bodo.transforms.dataframe_pass"):
            if (
                not isinstance(
                    self.typemap[lhs.name],
                    bodo.hiframes.pd_series_ext.SeriesType,
                )
                or self.typemap[lhs.name].dtype != types.bool_
            ):
                raise BodoError(
                    "query(): expr does not evaluate to a 1D boolean array."
                    " Only 1D boolean array is supported right now."
                )
            assign.value = rhs.args[0]
            return [assign]

        # Numba generates operator calls instead of binop nodes so needs normalized
        if len(fdef) == 2 and fdef[1] == "_operator":
            op = getattr(operator, fdef[0], None)
            if op in numba.core.utils.OPERATORS_TO_BUILTINS:
                return self._convert_op_call_to_expr(assign, rhs, op)

        return None

    def _run_call_dataframe(self, assign, lhs, rhs, df_var, func_name):
        if func_name in ("count", "query"):
            rhs.args.insert(0, df_var)
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}

            impl = getattr(
                bodo.hiframes.dataframe_impl, "overload_dataframe_" + func_name
            )(*arg_typs, **kw_typs)
            return replace_func(
                self,
                impl,
                rhs.args,
                pysig=numba.core.utils.pysignature(impl),
                kws=dict(rhs.kws),
            )

        if func_name in ("mask", "where"):
            rhs.args.insert(0, df_var)
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}

            overload_impl = (
                bodo.hiframes.dataframe_impl.create_dataframe_mask_where_overload(
                    func_name
                )
            )
            impl = overload_impl(*arg_typs, **kw_typs)
            return replace_func(
                self,
                impl,
                rhs.args,
                pysig=numba.core.utils.pysignature(impl),
                kws=dict(rhs.kws),
            )

        if func_name == "sort_values":
            rhs.args.insert(0, df_var)
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}

            impl = bodo.hiframes.dataframe_impl.overload_dataframe_sort_values(
                *arg_typs, **kw_typs
            )
            return replace_func(
                self,
                impl,
                rhs.args,
                pysig=numba.core.utils.pysignature(impl),
                kws=dict(rhs.kws),
            )

        if func_name == "pivot_table":
            rhs.args.insert(0, df_var)
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}
            impl = bodo.hiframes.dataframe_impl.overload_dataframe_pivot_table(
                *arg_typs, **kw_typs
            )
            stub = eval(
                "lambda df, values=None, index=None, columns=None, aggfunc='mean', fill_value=None, margins=False, dropna=True, margins_name='All', observed=True, _pivot_values=None: None"
            )
            return replace_func(
                self,
                impl,
                rhs.args,
                pysig=numba.core.utils.pysignature(stub),
                kws=dict(rhs.kws),
            )
        # df.apply(lambda a:..., axis=1)
        if func_name == "apply":
            return self._run_call_dataframe_apply(assign, lhs, rhs, df_var)

        return [assign]

    def _run_call_dataframe_apply(self, assign, lhs, rhs, df_var):
        """generate IR nodes for df.apply() with UDFs"""
        df_typ = self.typemap[df_var.name]
        # get apply function
        kws = dict(rhs.kws)
        func_var = get_call_expr_arg("apply", rhs.args, kws, 0, "func")
        func_type = self.typemap[func_var.name]
        axis_var = get_call_expr_arg(
            "apply", rhs.args, kws, 1, "axis", default=types.literal(0)
        )
        # Handle builtin functions passed by strings.
        if is_overload_constant_str(func_type):
            func_name = get_overload_const_str(func_type)
            var_kws = {}
            try:
                axis_var = get_call_expr_arg("apply", rhs.args, kws, 1, "axis")
                axis_type = self.typemap[axis_var.name]
                var_kws["axis"] = axis_var
            except BodoError:
                # If we have an exception we don't pass an axis value
                axis_type = None
            # Manually inline the implementation for efficiency
            impl = bodo.utils.transform.get_pandas_method_str_impl(
                df_typ,
                func_name,
                self.typingctx,
                "DataFrame.apply",
                axis_type,
            )
            if impl is not None:
                return replace_func(
                    self,
                    impl,
                    [df_var],
                    pysig=numba.core.utils.pysignature(impl),
                    kws=var_kws,
                    # Some DataFrame functions may require methods or
                    # attributes that need to be inlined by the full
                    # pipeline.
                    run_full_pipeline=True,
                )
            # TODO: Support Numpy funcs (requires overload/typing)

        func = get_overload_const_func(func_type, self.func_ir)
        out_typ = self.typemap[lhs.name]
        is_df_output = isinstance(out_typ, DataFrameType)
        out_arr_types = out_typ.data
        out_arr_types = out_arr_types if is_df_output else [out_arr_types]
        n_out_cols = len(out_arr_types)
        extra_args = get_call_expr_arg("apply", rhs.args, kws, 4, "args", [])
        nodes = []
        if extra_args:
            extra_args = get_build_sequence_vars(
                self.func_ir, self.typemap, self.calltypes, extra_args, nodes
            )

        # find kw arguments to UDF (pop apply() args first)
        kws.pop("func", None)
        kws.pop("axis", None)
        kws.pop("raw", None)
        kws.pop("result_type", None)
        kws.pop("args", None)
        udf_arg_names = (
            ", ".join(f"e{i}" for i in range(len(extra_args)))
            + (", " if extra_args else "")
            + ", ".join(f"{a}=e{i + len(extra_args)}" for i, a in enumerate(kws.keys()))
        )
        extra_args += list(kws.values())
        extra_arg_names = ", ".join(f"e{i}" for i in range(len(extra_args)))

        # find which columns are actually used if possible
        used_cols = _get_df_apply_used_cols(func, df_typ.columns)
        # avoid empty data which results in errors
        if not used_cols:
            used_cols = [df_typ.columns[0]]

        # prange func to inline
        col_name_args = ", ".join(["c" + str(i) for i in range(len(used_cols))])
        row_args = ", ".join(
            [
                f"bodo.utils.conversion.box_if_dt64(c{i}[i])"
                for i in range(len(used_cols))
            ]
        )

        func_text = f"def f({col_name_args}, df_index, {extra_arg_names}):\n"
        func_text += "  numba.parfors.parfor.init_prange()\n"
        func_text += "  n = len(c0)\n"
        func_text += "  index_arr = bodo.utils.conversion.coerce_to_array(df_index)\n"

        for i in range(n_out_cols):
            func_text += (
                f"  S{i} = bodo.utils.utils.alloc_type(n, _arr_typ{i}, (-1,))\n"
            )
        func_text += "  for i in numba.parfors.parfor.internal_prange(n):\n"
        # TODO: unbox to array value if necessary (e.g. Timestamp to dt64)
        func_text += f"    row_idx = bodo.hiframes.pd_index_ext.init_heter_index({gen_const_tup(used_cols)}, bodo.utils.conversion.box_if_dt64(index_arr[i]))\n"
        # Determine if we have a heterogenous or homogeneous series
        null_values_list = [
            f"bodo.libs.array_kernels.isna(c{i}, i)" for i in range(len(used_cols))
        ]
        null_args = ", ".join(null_values_list)
        func_text += f"    row_data = bodo.libs.nullable_tuple_ext.build_nullable_tuple(({row_args},), ({null_args},))\n"
        func_text += "    row = bodo.hiframes.pd_series_ext.init_series(row_data, row_idx, bodo.utils.conversion.box_if_dt64(index_arr[i]))\n"
        func_text += f"    v = map_func(row, {udf_arg_names})\n"
        if is_df_output:
            func_text += "    v_vals = bodo.hiframes.pd_series_ext.get_series_data(v)\n"
            for i in range(n_out_cols):
                func_text += f"    v{i} = v_vals[{i}]\n"
        else:
            func_text += "    v0 = v\n"
        for i in range(n_out_cols):
            if is_df_output:
                func_text += f"    if bodo.libs.array_kernels.isna(v_vals, {i}):\n"
                func_text += f"      bodo.libs.array_kernels.setna(S{i}, i)\n"
                func_text += "    else:\n"
                # Add extra indent
                func_text += "  "
            func_text += f"    S{i}[i] = bodo.utils.conversion.unbox_if_tz_naive_timestamp(v{i})\n"
        if is_df_output:
            data_arrs = ", ".join(f"S{i}" for i in range(n_out_cols))
            func_text += f"  return bodo.hiframes.pd_dataframe_ext.init_dataframe(({data_arrs},), df_index, __col_name_meta_value_dataframe_apply)\n"
        else:
            func_text += (
                "  return bodo.hiframes.pd_series_ext.init_series(S0, df_index, None)\n"
            )

        loc_vars = {}
        glbls = {}
        if is_df_output:
            glbls.update(
                {
                    "__col_name_meta_value_dataframe_apply": ColNamesMetaType(
                        self.typemap[lhs.name].columns
                    )
                }
            )
        exec(func_text, glbls, loc_vars)
        f = loc_vars["f"]

        col_vars = [self._get_dataframe_data(df_var, c, nodes) for c in used_cols]
        df_index_var = self._get_dataframe_index(df_var, nodes)
        map_func = bodo.compiler.udf_jit(func)

        glbls.update(
            {
                "numba": numba,
                "np": np,
                "bodo": bodo,
                "map_func": map_func,
                "init_nested_counts": bodo.utils.indexing.init_nested_counts,
                "add_nested_counts": bodo.utils.indexing.add_nested_counts,
            }
        )
        for i in range(n_out_cols):
            glbls[f"_arr_typ{i}"] = out_arr_types[i]

        return replace_func(
            self,
            f,
            col_vars + [df_index_var] + extra_args,
            extra_globals=glbls,
            pre_nodes=nodes,
        )

    def _run_call_bodosql_case_placeholder(self, rhs):
        """generate code for BodoSQL CASE statement and replace the placeholder call.
        BodoSQL provides the body of the loop as well as the output data type

        Args:
            rhs (ir.Expr.call): placeholder call

        Returns:
            ReplaceFunc: generated function for replacement
        """
        import re

        import bodosql

        init_code = self.typemap[rhs.args[2].name].instance_type.meta
        body_code = self.typemap[rhs.args[3].name].instance_type.meta
        arr_variable_name = get_overload_const_str(self.typemap[rhs.args[4].name])
        indexing_variable_name = get_overload_const_str(self.typemap[rhs.args[5].name])
        out_arr_type = unwrap_typeref(self.typemap[rhs.args[6].name])

        named_params = dict(rhs.kws)
        named_param_args = ", ".join(named_params.keys())

        # As a simple heuristic we determined complex case statements by looking at the
        # number of lines in the body of the case
        is_complex = len(body_code.split("\n")) > bodo.COMPLEX_CASE_THRESHOLD

        var_names, must_inline = handle_bodosql_case_init_code(init_code)

        if (
            not must_inline
            and is_complex
            and len(var_names) > 0
            and not named_param_args
        ):
            f, glbls = gen_bodosql_case_func(
                init_code,
                body_code,
                named_param_args,
                var_names,
                arr_variable_name,
                indexing_variable_name,
                out_arr_type,
                self.func_ir.func_id.func.__globals__,
            )
        else:
            func_text = f"def f(arrs, n, {named_param_args}):\n"
            func_text += init_code
            func_text += "  numba.parfors.parfor.init_prange()\n"
            func_text += f"  {arr_variable_name} = bodo.utils.utils.alloc_type(n, out_arr_type, (-1,))\n"
            func_text += f"  for {indexing_variable_name} in numba.parfors.parfor.internal_prange(n):\n"
            func_text += body_code
            func_text += f"  return {arr_variable_name}\n"

            loc_vars = {}
            glbls = {
                "numba": numba,
                "pd": pd,
                "np": np,
                "re": re,
                "bodo": bodo,
                "bodosql": bodosql,
                "out_arr_type": out_arr_type,
                "datetime": datetime,
            }
            # Globals generated by BodoSQL (accessible from main function) may be
            # necessary too.
            # See https://bodo.atlassian.net/browse/BSE-1941
            glbls.update(self.func_ir.func_id.func.__globals__)
            exec(func_text, glbls, loc_vars)
            f = loc_vars["f"]

        return replace_func(
            self,
            f,
            rhs.args[:2] + list(named_params.values()),
            extra_globals=glbls,
            run_full_pipeline=True,
        )

    def _run_call_df_sort_values(self, assign, lhs, rhs):
        """Implements support for df.sort_values().
        Translates sort_values_dummy() to a Sort IR node.
        """
        (
            df_var,
            by_var,
            ascending_var,
            inplace_var,
            na_position_var,
            _bodo_chunk_bounds,
            _bodo_interval_sort_var,
        ) = rhs.args
        df_typ = self.typemap[df_var.name]
        is_table_format = self.typemap[lhs.name].is_table_format
        inplace = guard(find_const, self.func_ir, inplace_var)
        # error_msg should be unused
        error_msg = "df.sort_values(): 'na_position' must be a literal constant of type str or a constant list of str with 1 entry per key column"
        na_position = guard(
            get_const_value,
            na_position_var,
            self.func_ir,
            error_msg,
            typemap=self.typemap,
        )
        _bodo_interval_sort = get_const_value(
            _bodo_interval_sort_var,
            self.func_ir,
            "df.sort_values(): '_bodo_interval_sort' must be a literal constant of type boolean.",
        )

        # find key array for sort ('by' arg)
        by_type = self.typemap[by_var.name]
        if is_overload_constant_tuple(by_type):
            key_names = [get_overload_const_tuple(by_type)]
        else:
            key_names = get_overload_const_list(by_type)
        valid_keys_set = set(df_typ.columns)
        index_name = "$_bodo_unset_"
        if not is_overload_none(df_typ.index.name_typ):
            index_name = df_typ.index.name_typ.literal_value
            valid_keys_set.add(index_name)
        if INDEX_SENTINEL in key_names:
            index_name = INDEX_SENTINEL
            valid_keys_set.add(index_name)
        # "A" is equivalent to ("A", "")
        key_names = [(k, "") if (k, "") in valid_keys_set else k for k in key_names]
        ascending_list = self._get_list_value_spec_length(
            ascending_var,
            len(key_names),
            err_msg="ascending should be bool or a list of bool of the number of keys",
        )
        # already checked in validate_sort_values_spec() so assertion is enough
        assert all(k in valid_keys_set for k in key_names), (
            f"invalid sort keys {key_names}"
        )

        nodes = []
        if is_table_format:
            in_table_var = self._get_dataframe_table(df_var, nodes)
            in_vars = [in_table_var]
        else:
            in_vars = [
                self._get_dataframe_data(df_var, c, nodes) for c in df_typ.columns
            ]
        in_index_var = self._gen_array_from_index(df_var, nodes)
        in_vars.append(in_index_var)

        key_inds = tuple(
            len(df_typ.columns) if c == index_name else df_typ.column_index[c]
            for c in key_names
        )
        if inplace:
            arr_types = df_typ.data + (self.typemap[in_index_var.name],)
            if any(arr_types[i] == bodo.types.dict_str_arr_type for i in key_inds):
                raise BodoError(
                    "inplace sort not supported for dictionary-encoded string arrays yet",
                    loc=rhs.loc,
                )
            out_vars = in_vars.copy()
            out_index_var = in_index_var
        else:
            out_vars = []
            if is_table_format:
                out_var = ir.Var(
                    lhs.scope, mk_unique_var(sanitize_varname("out_table")), lhs.loc
                )
                self.typemap[out_var.name] = df_typ.table_type
                out_vars.append(out_var)
            else:
                for ind, k in enumerate(df_typ.columns):
                    out_var = ir.Var(
                        lhs.scope, mk_unique_var(sanitize_varname(k)), lhs.loc
                    )
                    self.typemap[out_var.name] = df_typ.data[ind]
                    out_vars.append(out_var)
            # index var
            out_index_var = ir.Var(lhs.scope, mk_unique_var("_index_"), lhs.loc)
            self.typemap[out_index_var.name] = self.typemap[in_index_var.name]
            out_vars.append(out_index_var)

        nodes.append(
            bodo.ir.sort.Sort(
                df_var.name,
                lhs.name,
                in_vars,
                out_vars,
                key_inds,
                inplace,
                lhs.loc,
                ascending_list,
                na_position,
                _bodo_chunk_bounds,
                _bodo_interval_sort,
                is_table_format=is_table_format,
                num_table_arrays=len(df_typ.columns) if is_table_format else 0,
            )
        )

        # output from input index
        in_df_index = self._get_dataframe_index(df_var, nodes)
        in_df_index_name = self._get_index_name(in_df_index, nodes)
        out_index = self._gen_index_from_array(out_index_var, in_df_index_name, nodes)

        _init_df = _gen_init_df_dataframe_pass(
            df_typ.columns, "index", is_table_format=is_table_format
        )

        # return new df even for inplace case, since typing pass replaces input variable
        # using output of the call
        return nodes + compile_func_single_block(
            _init_df, out_vars[:-1] + [out_index], lhs, self
        )

    def _convert_op_call_to_expr(self, assign, rhs, op):
        """converts calls to operators (e.g. operator.add) to equivalent Expr nodes such
        as binop to be handled properly later.
        """
        old_calltype = self.calltypes[rhs]
        if len(rhs.args) == 1:
            rhs = ir.Expr.unary(op, rhs.args[0], rhs.loc)
            self.calltypes[rhs] = old_calltype
            assign.value = rhs
            return self._run_unary(assign, rhs)
        # arguments for contains() are reversed in operator
        if op == operator.contains:
            rhs.args = [rhs.args[1], rhs.args[0]]
        # inplace binop case
        if op in numba.core.utils.INPLACE_BINOPS_TO_OPERATORS.values():
            # get non-inplace version to pass to inplace_binop()
            op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
            assert op_str.endswith("=")
            immuop = numba.core.utils.BINOPS_TO_OPERATORS[op_str[:-1]]
            rhs = ir.Expr.inplace_binop(op, immuop, rhs.args[0], rhs.args[1], rhs.loc)
        else:
            rhs = ir.Expr.binop(op, rhs.args[0], rhs.args[1], rhs.loc)
        self.calltypes[rhs] = old_calltype
        assign.value = rhs
        return self._run_binop(assign, rhs)

    def _gen_array_from_index(self, df_var, nodes):
        func_text = (
            ""
            "def _get_index(df):\n"
            "    return bodo.utils.conversion.index_to_array(\n"
            "        bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)\n"
            "    )\n"
        )

        loc_vars = {}
        exec(func_text, globals(), loc_vars)
        nodes += compile_func_single_block(
            loc_vars["_get_index"], (df_var,), None, self
        )
        return nodes[-1].target

    def _gen_index_from_array(self, arr_var, name_var, nodes):
        func_text = (
            ""
            "def _get_index(arr, name):\n"
            "    return bodo.utils.conversion.index_from_array(arr, name)\n"
        )

        loc_vars = {}
        exec(func_text, globals(), loc_vars)
        nodes += compile_func_single_block(
            loc_vars["_get_index"], (arr_var, name_var), None, self
        )
        return nodes[-1].target

    def _run_call_df_itertuples(self, assign, lhs, rhs):
        """pass df column names and variables to get_itertuples() to be able
        to create the iterator.
        e.g. get_itertuples("A", "B", A_arr, B_arr)
        """
        df_var = rhs.args[0]
        df_typ = self.typemap[df_var.name]

        col_name_args = ", ".join(["c" + str(i) for i in range(len(df_typ.columns))])
        name_consts = ", ".join([f"'{c}'" for c in df_typ.columns])

        func_text = f"def f({col_name_args}):\n"
        func_text += f"  return bodo.hiframes.dataframe_impl.get_itertuples({name_consts}, {col_name_args})\n"

        loc_vars = {}
        exec(func_text, {}, loc_vars)
        f = loc_vars["f"]

        nodes = []
        col_vars = [self._get_dataframe_data(df_var, c, nodes) for c in df_typ.columns]
        return replace_func(self, f, col_vars, pre_nodes=nodes)

    def _run_call_col_reduce(self, assign, lhs, rhs, func_name):
        """support functions that reduce columns to single output and create
        a Series like mean, std, max, ..."""
        # TODO: refactor
        df_var = rhs.args[0]
        df_typ = self.typemap[df_var.name]

        # impl: for each column, convert data to series, call S.mean(), get
        # output data and create a new indexed Series
        n_cols = len(df_typ.columns)
        data_args = tuple(f"data{i}" for i in range(n_cols))

        func_text = "def _mean_impl({}):\n".format(", ".join(data_args))
        for d in data_args:
            ind = f"bodo.hiframes.pd_index_ext.init_range_index(0, len({d}), 1, None)"
            func_text += (
                "  {} = bodo.hiframes.pd_series_ext.init_series({}, {})\n".format(
                    d + "_S", d, ind
                )
            )
            func_text += "  {} = {}.{}()\n".format(d + "_O", d + "_S", func_name)
        func_text += "  data = np.array(({},))\n".format(
            ", ".join(d + "_O" for d in data_args)
        )
        func_text += (
            "  index = bodo.libs.str_arr_ext.str_arr_from_sequence(({},))\n".format(
                ", ".join(f"'{c}'" for c in df_typ.columns)
            )
        )
        func_text += "  return bodo.hiframes.pd_series_ext.init_series(data, index)\n"
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        _mean_impl = loc_vars["_mean_impl"]

        nodes = []
        col_vars = [self._get_dataframe_data(df_var, c, nodes) for c in df_typ.columns]
        return replace_func(self, _mean_impl, col_vars, pre_nodes=nodes)

    def _run_call_query(self, assign, lhs, rhs):
        """Transform query expr to Numba IR using the expr parser in Pandas."""
        # FIXME: local variables could be renamed by previous passes, including initial
        # renaming of Numba (e.g. a -> a.1 in some cases).
        # we need to develop a way to preserve initial variable names
        df_var, expr_var = rhs.args
        df_typ = self.typemap[df_var.name]

        # get expression string
        err_msg = (
            "df.query() expr arg should be constant string or argument to jit function"
        )
        expr = get_const_value(expr_var, self.func_ir, err_msg, self.typemap)

        # check expr is a non-empty string
        if len(expr) == 0:
            raise BodoError("query(): expr argument cannot be an empty string")

        # check expr is not multiline expression
        if len([e.strip() for e in expr.splitlines() if e.strip() != ""]) > 1:
            raise BodoError("query(): multiline expressions not supported yet")

        # parse expression
        parsed_expr, parsed_expr_str, used_cols = self._parse_query_expr(expr, df_typ)

        # check no columns nor index in selcted in expr
        if len(used_cols) == 0 and "index" not in expr:
            raise BodoError("query(): no column/index is selected in expr")

        # local variables
        sentinel = pd.core.computation.ops.LOCAL_TAG
        loc_ref_vars = {
            c: c.replace(sentinel, "")
            for c in parsed_expr.names
            if isinstance(c, str) and c.startswith(sentinel)
        }
        in_args = list(used_cols.values()) + ["index"] + list(loc_ref_vars.keys())
        func_text = "def _query_impl({}):\n".format(", ".join(in_args))
        # convert array to Series to support cases such as C.str.contains
        for c_var in used_cols.values():
            ind = (
                f"bodo.hiframes.pd_index_ext.init_range_index(0, len({c_var}), 1, None)"
            )
            func_text += (
                f"  {c_var} = bodo.hiframes.pd_series_ext.init_series({c_var}, {ind})\n"
            )
        # use dummy function to catch data type error
        func_text += f"  return _check_query_series_bool({parsed_expr_str})"
        loc_vars = {}
        global _check_query_series_bool
        exec(
            func_text, {"_check_query_series_bool": _check_query_series_bool}, loc_vars
        )
        _query_impl = loc_vars["_query_impl"]

        # data frame column inputs
        nodes = []
        args = [self._get_dataframe_data(df_var, c, nodes) for c in used_cols.keys()]
        args.append(self._gen_array_from_index(df_var, nodes))
        # local referenced variables
        args += [ir.Var(lhs.scope, v, lhs.loc) for v in loc_ref_vars.values()]

        return replace_func(
            self, _query_impl, args, pre_nodes=nodes, run_full_pipeline=True
        )

    def _parse_query_expr(self, expr, df_typ):
        """Parses query expression using Pandas parser but avoids issues such as
        early evaluation of string expressions by Pandas.
        """
        clean_name = pd.core.computation.parsing.clean_column_name
        cleaned_columns = [clean_name(c) for c in df_typ.columns]
        resolver = dict.fromkeys(cleaned_columns, 0)
        resolver["index"] = 0
        # create fake environment for Expr that just includes the symbol names to
        # enable parsing
        glbs = self.func_ir.func_id.func.__globals__
        lcls = dict.fromkeys(self.func_ir.func_id.code.co_varnames, 0)
        env = pd.core.computation.scope.ensure_scope(2, glbs, lcls, (resolver,))
        index_name = df_typ.index.name_typ
        return bodo.hiframes.dataframe_impl._parse_query_expr(
            expr, env, df_typ.columns, cleaned_columns, index_name
        )

    def _run_call_set_df_column(self, assign, lhs, rhs):
        """transform set_df_column() to handle reflection/inplace cases properly if
        needed. Otherwise, just create a new dataframe with the new column.
        """

        df_var = rhs.args[0]
        cname = guard(find_const, self.func_ir, rhs.args[1])
        new_arr = rhs.args[2]
        # inplace = guard(find_const, self.func_ir, rhs.args[3])
        inplace = guard(get_definition, self.func_ir, rhs.args[3]).value
        df_typ = self.typemap[df_var.name]
        out_df_type = self.typemap[assign.target.name]
        nodes = []

        # transform df['col2'] = df['col1'][arr] since we don't support index alignment
        # since columns should have the same size, output is filled with NaNs
        # TODO: make sure col1 and col2 are in the same df
        # TODO: compare df index and Series index and match them in setitem
        arr_def = guard(get_definition, self.func_ir, new_arr)
        if guard(find_callname, self.func_ir, arr_def, self.typemap) == (
            "init_series",
            "bodo.hiframes.pd_series_ext",
        ):  # pragma: no cover
            arr_def = guard(get_definition, self.func_ir, arr_def.args[0])
        if (
            is_expr(arr_def, "getitem")
            and is_array_typ(self.typemap[arr_def.value.name])
            and self.is_bool_arr(arr_def.index.name)
        ):
            orig_arr = arr_def.value
            bool_arr = arr_def.index
            nodes += compile_func_single_block(
                eval(
                    "lambda arr, bool_arr: bodo.hiframes.series_impl.series_filter_bool(arr, bool_arr)"
                ),
                (orig_arr, bool_arr),
                None,
                self,
            )
            new_arr = nodes[-1].target

        # set unboxed df column with reflection
        df_def = guard(get_definition, self.func_ir, df_var)
        # TODO: consider dataframe alias cases where definition is not directly ir.Arg
        # but dataframe has a parent object
        if isinstance(df_def, ir.Arg) or guard(
            find_callname, self.func_ir, df_def, self.typemap
        ) == ("set_df_column_with_reflect", "bodo.hiframes.pd_dataframe_ext"):
            return replace_func(
                self,
                eval(
                    "lambda df, cname, arr: bodo.hiframes.pd_dataframe_ext.set_df_column_with_reflect("
                    "    df,"
                    "    cname,"
                    "    bodo.utils.conversion.coerce_to_array("
                    "        arr, scalar_to_arr_len=len(df)"
                    "    ),"
                    ")"
                ),
                [df_var, rhs.args[1], new_arr],
                pre_nodes=nodes,
            )

        if inplace:
            if cname not in df_typ.columns:
                raise BodoError(
                    "Setting new dataframe columns inplace is not supported in conditionals/loops or for dataframe arguments",
                    loc=rhs.loc,
                )
            return replace_func(
                self,
                eval(
                    "lambda df, arr: bodo.hiframes.pd_dataframe_ext.set_dataframe_data("
                    "    df,"
                    "    c_ind,"
                    "    bodo.utils.conversion.coerce_to_array("
                    "        arr, scalar_to_arr_len=len(df)"
                    "    ),"
                    ")"
                ),
                [df_var, new_arr],
                pre_nodes=nodes,
                extra_globals={"c_ind": df_typ.columns.index(cname)},
            )

        n_cols = len(df_typ.columns)
        df_index_var = self._get_dataframe_index(df_var, nodes)

        col_ind = out_df_type.columns.index(cname)

        if df_typ.is_table_format:
            # in this case, output dominates the input so we can reuse its internal data
            # see _run_df_set_column()
            in_table_var = self._get_dataframe_table(df_var, nodes)
            in_arrs = [in_table_var]
            data_args = "T1"
            new_arr_arg = "new_val"
        else:
            in_arrs = [
                self._get_dataframe_data(df_var, c, nodes) for c in df_typ.columns
            ]
            data_args = [f"data{i}" for i in range(n_cols)]

            # if column is being added
            if cname not in df_typ.columns:
                data_args.append("new_arr")
                in_arrs.append(new_arr)
                new_arr_arg = "new_arr"
            else:  # updating existing column
                in_arrs[col_ind] = new_arr
                new_arr_arg = f"data{col_ind}"

            data_args = ", ".join(data_args)

        # TODO: fix list, Series data
        df_index = "df_index"
        if n_cols == 0:
            df_index = "bodo.utils.conversion.extract_index_if_none(new_val, None)\n"
        func_text = f"def _init_df({data_args}, df_index, df, new_val):\n"
        # using len(df) instead of len(df_index) since len optimization works better for
        # dataframes
        func_text += f"  {new_arr_arg} = bodo.utils.conversion.coerce_to_array({new_arr_arg}, scalar_to_arr_len=len(df))\n"
        if df_typ.is_table_format:
            func_text += f"  T2 = bodo.hiframes.table.set_table_data(T1, {col_ind}, {new_arr_arg})\n"
            func_text += f"  df = bodo.hiframes.pd_dataframe_ext.init_dataframe((T2,), {df_index}, __col_name_meta_value_set_df_column)\n"
            func_text += "  return df\n"
        else:
            func_text += f"  return bodo.hiframes.pd_dataframe_ext.init_dataframe(({data_args},), {df_index}, __col_name_meta_value_set_df_column)\n"
        loc_vars = {}
        __col_name_meta_value_set_df_column = ColNamesMetaType(out_df_type.columns)
        exec(
            func_text,
            {
                "__col_name_meta_value_set_df_column": __col_name_meta_value_set_df_column
            },
            loc_vars,
        )
        _init_df = loc_vars["_init_df"]
        return replace_func(
            self,
            _init_df,
            in_arrs + [df_index_var, df_var, new_arr],
            pre_nodes=nodes,
            extra_globals={
                "__col_name_meta_value_set_df_column": __col_name_meta_value_set_df_column
            },
        )

    def _df_pass_run_call_bodosql_replace_columns(
        self, assign, lhs, rhs
    ):  # pragma: no cover
        """
        transforms __bodosql_replace_columns_dummy. This is heavily copied from _run_call_set_df_column. However, we only need to handle a subset of the cases found here,
        as we have some assurances due to the way that bodosql handles code generation, namely:
        1. The input dataframe is has at least one column
        2. All of the columns that are being set are already present in the input dataframe
        3. We don't have to handle reflection and/or updating in place
        """

        df_var = rhs.args[0]
        col_indicies = guard(find_build_tuple, self.func_ir, rhs.args[1])
        assert col_indicies is not None, (
            "Internal error, unable to find build tuple for arg1 of __bodosql_replace_columns_dummy"
        )
        # column names must be consts
        col_names_to_replace = [
            guard(find_const, self.func_ir, col_indicie) for col_indicie in col_indicies
        ]

        for name in col_names_to_replace:
            assert name is not None, (
                "Internal error, unable to find build tuple for arg1 of __bodosql_replace_columns_dummy"
            )

        new_arrs = guard(find_build_tuple, self.func_ir, rhs.args[2])
        assert new_arrs is not None, (
            "Internal error, unable to find build tuple for arg2 of __bodosql_replace_columns_dummy"
        )

        in_df_typ = self.typemap[df_var.name]
        out_df_typ = self.typemap[assign.target.name]
        nodes = []

        # This shouldn't be possible given our codegen, but better safe than sorry
        for col_name in col_names_to_replace:
            if col_name not in in_df_typ.column_index:
                raise BodoError(
                    "Internal error, invalid code generated for __bodosql_replace_columns_dummy: column name that is not present in the input dataframe passed into arg1",
                    loc=rhs.loc,
                )

        # We only generate a __bodosql_replace_columns_dummy call if there are columns that need to be converted
        # Therefore, we should always have at least one column in the input
        n_cols = len(in_df_typ.columns)
        if n_cols == 0:
            raise BodoError(
                "Internal error, invalid code generated for __bodosql_replace_columns_dummy: df in arg0 has no columns",
                loc=rhs.loc,
            )
        df_index_var = self._get_dataframe_index(df_var, nodes)

        if in_df_typ.is_table_format:
            # in this case, output dominates the input so we can reuse its internal data
            # see _run_df_set_column()
            in_table_var = self._get_dataframe_table(df_var, nodes)
            in_arrs = [in_table_var]
            data_args = ["T0"]
            # Note: col_inds is only defined/used in tableformat path
            col_inds = []
            for i in range(len(col_names_to_replace)):
                col_name = col_names_to_replace[i]
                new_arr = new_arrs[i]
                col_inds.append(in_df_typ.column_index[col_name])
                data_args.append(f"data{i}")
                in_arrs.append(new_arr)
        else:
            in_arrs = [
                self._get_dataframe_data(df_var, c, nodes) for c in in_df_typ.columns
            ]
            data_args = [f"data{i}" for i in range(n_cols)]
            init_table_args = data_args.copy()
            for i in range(len(col_names_to_replace)):
                new_arr = new_arrs[i]
                col_name = col_names_to_replace[i]
                # we should always updating existing column
                col_ind = in_df_typ.column_index[col_name]
                data_args[col_ind] = f"new_arr_{col_ind}"
                in_arrs[col_ind] = new_arrs[i]
                init_table_args[col_ind] = (
                    f"bodo.utils.conversion.coerce_to_array(new_arr_{col_ind})"
                )
                in_arrs[col_ind] = new_arr

        data_args_full_string = ", ".join(data_args)
        func_text = f"def _init_df({data_args_full_string}, df_index):\n"
        if in_df_typ.is_table_format:
            # For now, we're just going to generate a list of set_table_data's. IE:
            #
            # T1 = bodo.hiframes.table.set_table_data(T0, 0, bodo.utils.conversion.coerce_to_array(data_args0))
            # T2 = bodo.hiframes.table.set_table_data(T1, 1, bodo.utils.conversion.coerce_to_array(data_args1))
            # ...
            # df = bodo.hiframes.pd_dataframe_ext.init_dataframe((T10, ), df_index, out_df_type)
            # return df
            #
            # In the future, we could likely restructure set_table_data to allow multiple simultaneous set actions
            for i in range(len(col_inds)):
                col_ind = col_inds[i]
                func_text += f"  T{i + 1} = bodo.hiframes.table.set_table_data(T{i}, {col_ind}, bodo.utils.conversion.coerce_to_array({data_args[i + 1]}))\n"
            func_text += f"  df = bodo.hiframes.pd_dataframe_ext.init_dataframe((T{len(col_names_to_replace)},), df_index, out_col_names)\n"
            func_text += "  return df\n"
        else:
            init_table_args_full_string = ", ".join(init_table_args)
            func_text += f"  return bodo.hiframes.pd_dataframe_ext.init_dataframe(({init_table_args_full_string},), df_index, out_col_names)\n"
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        _init_df = loc_vars["_init_df"]
        return replace_func(
            self,
            _init_df,
            in_arrs + [df_index_var],
            pre_nodes=nodes,
            extra_globals={"out_col_names": ColNamesMetaType(out_df_typ.columns)},
        )

    def _run_call_len(self, lhs, df_var, assign):
        df_typ = self.typemap[df_var.name]

        # DataFrames with columns determined at runtime aren't
        # transformed
        if df_typ.has_runtime_cols:
            return [assign]

        # run len on one of the columns
        # FIXME: it could potentially avoid remove dead for the column if
        # array analysis doesn't replace len() with it's size
        nodes = []
        if len(df_typ.columns) == 0:
            # 0 column DataFrame has the same length as the index
            arr = self._get_dataframe_index(df_var, nodes)
        else:
            arr = self._get_dataframe_data(df_var, df_typ.columns[0], nodes)

        func_text = """def f(df_arr):\n  return len(df_arr)\n"""

        loc_vars = {}
        exec(func_text, globals(), loc_vars)
        return replace_func(self, loc_vars["f"], [arr], pre_nodes=nodes)

    def _run_call_join(self, assign, lhs, rhs):
        """Transform join_dummy() generated from calls like df.merge() into ir.Join node"""
        (
            left_df,
            right_df,
            left_on_var,
            right_on_var,
            how_var,
            suffix_x_var,
            suffix_y_var,
            is_join_var,
            is_indicator_var,
            _bodo_na_equal_var,
            _bodo_rebalance_output_if_skewed_var,
            gen_cond_var,
        ) = rhs.args

        left_keys = self._get_const_or_list(left_on_var)
        right_keys = self._get_const_or_list(right_on_var)

        # While all of these variables use guard, they must not
        # be None as they are all set internally by Bodo.
        how = guard(find_const, self.func_ir, how_var)
        assert how is not None, "Internal error with Join IR node"
        suffix_x = guard(find_const, self.func_ir, suffix_x_var)
        assert suffix_x is not None, "Internal error with Join IR node"
        suffix_y = guard(find_const, self.func_ir, suffix_y_var)
        assert suffix_y is not None, "Internal error with Join IR node"
        is_join = guard(find_const, self.func_ir, is_join_var)
        assert is_join is not None, "Internal error with Join IR node"
        is_indicator = guard(find_const, self.func_ir, is_indicator_var)
        assert is_indicator is not None, "Internal error with Join IR node"
        is_na_equal = guard(find_const, self.func_ir, _bodo_na_equal_var)
        assert is_na_equal is not None, "Internal error with Join IR node"
        rebalance_output_if_skewed = guard(
            find_const, self.func_ir, _bodo_rebalance_output_if_skewed_var
        )
        assert rebalance_output_if_skewed is not None, (
            "Internal error with Join IR node"
        )
        gen_cond_expr = guard(find_const, self.func_ir, gen_cond_var)
        assert gen_cond_expr is not None, "Internal error with Join IR node"
        out_typ = self.typemap[lhs.name]
        # convert right join to left join
        is_left = how in {"left", "outer"}
        is_right = how in {"right", "outer"}

        nodes = []

        # Obtain the output and input IR variables. Output is just a
        # table.
        out_data_vars = [
            ir.Var(
                lhs.scope, mk_unique_var(sanitize_varname("join_out_table")), lhs.loc
            ),
        ]
        out_table_typ = TableType(out_typ.data)
        # Add the type for the table.
        self.typemap[out_data_vars[0].name] = out_table_typ

        left_df_type = self.typemap[left_df.name]
        if left_df_type.is_table_format:
            left_vars = [self._get_dataframe_table(left_df, nodes)]
        else:
            left_vars = [
                self._get_dataframe_data(left_df, c, nodes)
                for c in left_df_type.columns
            ]
        right_df_type = self.typemap[right_df.name]
        if right_df_type.is_table_format:
            right_vars = [self._get_dataframe_table(right_df, nodes)]
        else:
            right_vars = [
                self._get_dataframe_data(right_df, c, nodes)
                for c in right_df_type.columns
            ]
        # In the case of pd.merge we have following behavior for the index:
        # ---if the key is a normal column then the joined table has a trivial index.
        # ---if one of the key is an index then it becomes an index.
        # In the case of df1.join(df2, ....) we have following behavior for the index:
        # ---the index of df1 is used for the merging. and the index of df2 or some other column.
        # ---The index of the joined table is assigned from the non-joined column.

        has_index_var = False
        in_df_index_name = None
        right_index = INDEX_SENTINEL in right_keys
        left_index = INDEX_SENTINEL in left_keys
        # The index variables
        right_df_index = self._get_dataframe_index(right_df, nodes)
        right_df_index_name = self._get_index_name(right_df_index, nodes)
        right_index_var = self._gen_array_from_index(right_df, nodes)
        left_df_index = self._get_dataframe_index(left_df, nodes)
        left_df_index_name = self._get_index_name(left_df_index, nodes)
        left_index_var = self._gen_array_from_index(left_df, nodes)

        # If left_index=True or right_index=True the output index comes
        # from an existing table. This matches the corresponding code
        # at the end of JoinTyper (left_index_as_output is cases 1 and 3
        # and right_index_as_output is case 2).
        left_index_as_output = (left_index and right_index and how != "asof") or (
            not left_index and right_index
        )
        right_index_as_output = left_index and not right_index

        if right_index_as_output:
            has_index_var = True
            in_df_index_name = right_df_index_name
        elif left_index_as_output:
            has_index_var = True
            in_df_index_name = left_df_index_name
        # Update the type info
        if has_index_var:
            index_var = ir.Var(lhs.scope, mk_unique_var("out_index"), lhs.loc)
            out_data_vars.append(index_var)
            self.typemap[index_var.name] = get_index_data_arr_types(
                self.typemap[lhs.name].index
            )[0]
            left_vars.append(left_index_var)
            right_vars.append(right_index_var)
        else:
            out_data_vars.append(None)
            left_vars.append(None)
            right_vars.append(None)
        if is_indicator:
            # This indicator column number is always last.
            indicator_col_num = len(out_typ.data) - 1
        else:
            indicator_col_num = -1

        # cross join needs input lengths to handle dead input cases
        left_len_var = self._gen_len_call(left_df, nodes) if how == "cross" else None
        right_len_var = self._gen_len_call(right_df, nodes) if how == "cross" else None

        nodes.append(
            bodo.ir.join.Join(
                left_keys,
                right_keys,
                out_data_vars,
                out_typ,
                left_vars,
                left_df_type,
                right_vars,
                right_df_type,
                how,
                suffix_x,
                suffix_y,
                lhs.loc,
                is_left,
                is_right,
                is_join,
                left_index,
                right_index,
                indicator_col_num,
                is_na_equal,
                rebalance_output_if_skewed,
                gen_cond_expr,
                left_len_var,
                right_len_var,
            )
        )

        # Output variables returned by Join (Table/Index)
        out_vars = [out_data_vars[0]]
        if has_index_var:
            # Index does not come from the input so we generate a new index.
            out_index = self._gen_index_from_array(
                out_data_vars[1], in_df_index_name, nodes
            )
            out_vars.append(out_index)
            _init_df = _gen_init_df_dataframe_pass(
                out_typ.columns, "index", is_table_format=True
            )
        else:
            _init_df = _gen_init_df_dataframe_pass(
                out_typ.columns, is_table_format=True
            )

        return nodes + compile_func_single_block(_init_df, out_vars, lhs, self)

    def _gen_len_call(self, var, nodes):
        """generate a len() call on 'var' and append the IR nodes to 'nodes'

        Args:
            var (ir.Var): input variable to call len() on (array/series/dataframe type)
            nodes (list(ir.Stmt)): IR node list to append nodes of len() call

        Returns:
            ir.Var: output variable of len() call
        """
        nodes += compile_func_single_block(
            eval("lambda A: len(A)"),
            (var,),
            None,
            self,
        )
        return nodes[-1].target

    def _gen_gb_info_out(self, rhs, grp_typ, df_type, out_typ, func_name):
        args = tuple([self.typemap[v.name] for v in rhs.args])
        kws = {k: self.typemap[v.name] for k, v in rhs.kws}

        # gb_info maps (in_cols, additional_args, func_name) -> [out_col_1, out_col_2, ...]
        _, gb_info = bodo.hiframes.pd_groupby_ext.resolve_gb(
            grp_typ,
            args,
            kws,
            func_name,
            numba.core.registry.cpu_target.typing_context,
            numba.core.registry.cpu_target.target_context,
        )

        # Populate gb_info_out (to store in Aggregate node)
        # gb_info_out: maps out_col -> (in_cols, func)
        gb_info_out = {}
        agg_func = get_agg_func(self.func_ir, func_name, rhs, typemap=self.typemap)
        if not isinstance(agg_func, list):
            # agg_func is SimpleNamespace or a Python function for UDFs
            agg_funcs = [agg_func for _ in range(len(gb_info))]
        else:
            # TODO Might be possible to simplify get_agg_func by returning a flat list
            agg_funcs = list(flatten(agg_func))

        used_colnames = set(grp_typ.keys)
        i = 0
        for (in_cols, additional_args, fname), out_col_list in gb_info.items():
            in_col_inds = tuple(df_type.column_index[in_col] for in_col in in_cols)

            # TODO(keaton): handle the case that we have duplicate aggregation functions
            # https://bodo.atlassian.net/browse/BSE-609
            # previously, this was segfaulting. This doesn't resolve the issue, but it at least
            # prevents segfaults.
            if len(out_col_list) != 1:
                raise BodoError(
                    "Internal error in _gen_gb_info_out: encountered duplicate aggregation functions"
                )

            out_col = out_col_list[0]
            out_col_ind = (
                out_typ.column_index[out_col]
                if not isinstance(out_typ, SeriesType)
                else 0
            )
            f = agg_funcs[i]
            assert out_col_ind not in gb_info_out, (
                "Internal error in _gen_gb_info_out: duplicate output column assignment"
            )
            assert fname == f.fname, (
                "Internal error in _gen_gb_info_out: fname mismatch"
            )
            gb_info_out[out_col_ind] = (in_col_inds, f)
            used_colnames.update(in_cols)
            i += 1

        return gb_info_out, used_colnames

    def _run_call_groupby(self, assign, lhs, rhs, grp_var, func_name):
        """Transform groupby calls into an Aggregate IR node"""
        if func_name == "apply":
            return self._run_call_groupby_apply(assign, lhs, rhs, grp_var)

        # DataFrameGroupByType instance with initial typing info
        grp_typ = self.typemap[grp_var.name]
        # get dataframe variable of groupby
        df_var = self._get_groupby_df_obj(grp_var)
        df_type = self.typemap[df_var.name]
        # Type of df.groupby() output
        out_typ = self.typemap[lhs.name]

        # check for duplicate output columns
        out_colnames = (
            grp_typ.selection if isinstance(out_typ, SeriesType) else out_typ.columns
        )
        if len(out_colnames) != len(set(out_colnames)):
            raise BodoError("aggregate with duplication in output is not allowed")

        gb_info_out, used_colnames = self._gen_gb_info_out(
            rhs, grp_typ, df_type, out_typ, func_name
        )

        # input_has_index=True means we have to pass Index of input dataframe to groupby
        # since it's needed in creating output, e.g. in shift() but not sum()
        input_has_index = False
        # same_index=True if groupby returns the Index for output dataframe since its
        # values have to match the input (cumulative operations not using RangeIndex)
        same_index = False
        # return_key=True if groupby returns group keys since needed for output.
        # e.g. for sum() but not shift()
        return_key = True
        # Is the output df defined to maintain the same size as the input.
        maintain_input_size = False

        for _, func in gb_info_out.values():
            if func.ftype in (list_cumulative | {"shift", "transform", "ngroup"}):
                input_has_index = True
                same_index = True
                return_key = False
                maintain_input_size = True
            elif func.ftype in {"idxmin", "idxmax"}:
                input_has_index = True
            elif func.ftype == "head":
                input_has_index = True
                same_index = True
                return_key = False
            elif func.ftype == "window":
                return_key = False
                maintain_input_size = True

        # TODO: comment on when this case is possible and necessary
        if (
            same_index
            and isinstance(out_typ.index, RangeIndexType)
            # unlike cumulative operations gb.head()/gb.ngroup() will always
            # return the same Index in all cases including RangeIndexType.
            and func.ftype not in ("head", "ngroup")
        ):
            same_index = False
            input_has_index = False

        in_key_inds = [df_type.column_index[c] for c in grp_typ.keys]

        # get input variables (data columns first then Index if needed)
        nodes = []
        if df_type.is_table_format:
            in_table_var = self._get_dataframe_table(df_var, nodes)
            in_vars = [in_table_var]
        else:
            # pass all data columns to Aggregate node since it uses logical column indexes
            # minor optimization: avoid generating get_dataframe_data for columns that are
            # obviously not used
            in_vars = [
                self._get_dataframe_data(df_var, c, nodes)
                if c in used_colnames
                else None
                for c in df_type.columns
            ]

        if input_has_index:
            in_index_var = self._gen_array_from_index(df_var, nodes)
            in_vars.append(in_index_var)
        else:
            in_vars.append(None)

        # output data is a single array in case of Series, otherwise always a table
        # if keys are returned, they could be part of the table (as_index=False), or be
        # separate array(s) that go to output dataframe's Index
        out_var = ir.Var(lhs.scope, mk_unique_var("groupby_out_data"), lhs.loc)
        self.typemap[out_var.name] = (
            out_typ.data if isinstance(out_typ, SeriesType) else out_typ.table_type
        )
        out_vars = [out_var]

        # add key column variables
        # extra variables are necessary if keys are part of Index and not table
        if return_key and not isinstance(out_typ.index, RangeIndexType):
            for k in grp_typ.keys:
                out_key_var = ir.Var(
                    lhs.scope, mk_unique_var(sanitize_varname(k)), lhs.loc
                )
                ind = df_type.column_index[k]
                self.typemap[out_key_var.name] = df_type.data[ind]
                out_vars.append(out_key_var)

        if same_index:
            out_index_var = ir.Var(lhs.scope, mk_unique_var("out_index"), lhs.loc)
            self.typemap[out_index_var.name] = self.typemap[in_index_var.name]
            out_vars.append(out_index_var)
        else:
            out_vars.append(None)

        agg_node = bodo.ir.aggregate.Aggregate(
            lhs.name,
            df_var.name,  # input dataframe var name
            grp_typ.keys,  # name of key columns, for printing only
            gb_info_out,
            out_vars,
            in_vars,
            in_key_inds,
            df_type,
            out_typ,
            input_has_index,
            same_index,
            return_key,
            lhs.loc,
            func_name,
            maintain_input_size,
            grp_typ.dropna,
            # Subset of keys to use as the key when shuffling across
            # ranks, keys[:grp_typ._num_shuffle_keys].
            # If grp_typ._num_shuffle_keys == -1 then we use
            # all of the keys, which is the common case.
            grp_typ._num_shuffle_keys,
            # Should we use SQL or Pandas rules instead groupby
            grp_typ._use_sql_rules,
        )

        nodes.append(agg_node)

        if same_index:
            in_df_index = self._get_dataframe_index(df_var, nodes)
            in_df_index_name = self._get_index_name(in_df_index, nodes)
            index_var = self._gen_index_from_array(
                out_index_var, in_df_index_name, nodes
            )
        elif isinstance(out_typ.index, RangeIndexType):
            # as_index=False case generates trivial RangeIndex
            # See test_groupby_asindex_no_values
            # We get the length from the first output assuming it won't cause trouble
            # for dead code/column elimination
            output_column_ir_var = out_vars[0]
            nodes += compile_func_single_block(
                eval(
                    "lambda A: bodo.hiframes.pd_index_ext.init_range_index(0, len(A), 1, None)"
                ),
                (output_column_ir_var,),
                None,
                self,
            )
            index_var = nodes[-1].target
        elif isinstance(out_typ.index, MultiIndexType):
            # gen MultiIndex init function
            arg_names = ", ".join(f"in{i}" for i in range(len(grp_typ.keys)))
            names_tup = ", ".join(f"'{k}'" for k in grp_typ.keys)
            func_text = f"def _multi_inde_impl({arg_names}):\n"
            func_text += f"    return bodo.hiframes.pd_multi_index_ext.init_multi_index(({arg_names}), ({names_tup}))\n"
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            _multi_inde_impl = loc_vars["_multi_inde_impl"]
            # the key columns are right before Index which is last
            index_arrs = out_vars[-len(grp_typ.keys) - 1 : -1]
            nodes += compile_func_single_block(_multi_inde_impl, index_arrs, None, self)
            index_var = nodes[-1].target
        else:
            # the key column is right before Index which is last
            index_arr = out_vars[-2]
            index_name = grp_typ.keys[0]
            nodes += compile_func_single_block(
                eval(
                    "lambda A: bodo.utils.conversion.index_from_array(A, _index_name)"
                ),
                (index_arr,),
                None,
                self,
                extra_globals={"_index_name": index_name},
            )
            index_var = nodes[-1].target

        # NOTE: output becomes series if single output and explicitly selected
        # or as_index=True and only size
        # or ngroup
        if isinstance(out_typ, SeriesType):
            assert (
                len(grp_typ.selection) == 1
                and grp_typ.series_select
                and grp_typ.as_index
            ) or (grp_typ.as_index and func_name in ("size", "ngroup"))
            name_val = None if func_name in ("size", "ngroup") else grp_typ.selection[0]
            name_var = ir.Var(lhs.scope, mk_unique_var("S_name"), lhs.loc)
            self.typemap[name_var.name] = (
                types.none
                if func_name in ("size", "ngroup")
                else types.StringLiteral(name_val)
            )
            nodes.append(ir.Assign(ir.Const(name_val, lhs.loc), name_var, lhs.loc))
            return replace_func(
                self,
                eval(
                    "lambda A, I, name: bodo.hiframes.pd_series_ext.init_series(A, I, name)"
                ),
                [out_vars[0], index_var, name_var],
                pre_nodes=nodes,
            )

        _init_df = _gen_init_df_dataframe_pass(
            out_typ.columns, "index", is_table_format=True
        )
        return nodes + compile_func_single_block(
            _init_df, [out_vars[0], index_var], lhs, self
        )

    def _run_call_groupby_apply(self, assign, lhs, rhs, grp_var):
        """generate IR nodes for df.groupby().apply() with UDFs.
        Generates a separate function call '_bodo_groupby_apply_impl()' that includes
        the actual implementation since generating IR here directly may confuse
        distributed analysis. Regular overload doesn't work since the UDF may have
        keyword arguments (not supported by Numba).
        """
        grp_typ = self.typemap[grp_var.name]
        df_var = self._get_groupby_df_obj(grp_var)
        df_type = self.typemap[df_var.name]
        out_typ = self.typemap[lhs.name]
        n_out_cols = 1 if isinstance(out_typ, SeriesType) else len(out_typ.columns)

        # get apply function
        kws = dict(rhs.kws)
        func_var = get_call_expr_arg("GroupBy.apply", rhs.args, kws, 0, "func")
        func = get_overload_const_func(self.typemap[func_var.name], self.func_ir)

        extra_args = [] if len(rhs.args) < 2 else rhs.args[1:]

        # find kw arguments to UDF (pop apply() args first)
        kws.pop("func", None)
        udf_arg_names = (
            ", ".join(f"e{i}" for i in range(len(extra_args)))
            + (", " if extra_args else "")
            + ", ".join(f"{a}=e{i + len(extra_args)}" for i, a in enumerate(kws.keys()))
        )
        udf_arg_types = [self.typemap[v.name] for v in extra_args]
        udf_kw_types = {k: self.typemap[v.name] for k, v in kws.items()}
        udf_return_type = _get_groupby_apply_udf_out_type(
            bodo.utils.typing.FunctionLiteral(func),
            grp_typ,
            udf_arg_types,
            udf_kw_types,
            self.typingctx,
            self.targetctx,
        )

        extra_args += list(kws.values())
        extra_arg_names = ", ".join(f"e{i}" for i in range(len(extra_args)))

        in_col_names = df_type.columns
        if grp_typ.explicit_select:
            in_col_names = tuple(grp_typ.selection)

        # find which columns are actually used if possible
        used_cols = _get_df_apply_used_cols(func, in_col_names)
        # avoid empty data which results in errors
        if not used_cols:
            used_cols = [df_type.columns[0]]

        n_keys = len(grp_typ.keys)
        key_names = [f"k{i}" for i in range(n_keys)]
        col_names = [f"c{i}" for i in range(len(used_cols))]
        key_name_args = ", ".join(key_names)
        col_name_args = ", ".join(col_names)

        if extra_arg_names:
            extra_arg_names += ", "

        func_text = (
            f"def f({key_name_args}, {col_name_args}, df_index, {extra_arg_names}):\n"
        )
        # Distributed pass shuffles in_df, which converts RangeIndex to NumericIndex.
        # We need to avoid RangeIndex to make sure data type of in_df doesn't change.
        # Otherwise, _bodo_groupby_apply_impl would have to be recompiled in distributed
        # pass to avoid errors (extra compilation time overhead).
        if isinstance(df_type.index, RangeIndexType):
            func_text += "  df_index = range_index_to_numeric(df_index)\n"

        func_text += f"  in_df = bodo.hiframes.pd_dataframe_ext.init_dataframe(({col_name_args},), df_index, __col_name_meta_value_inner)\n"

        # pass shuffle info and is_parallel (not literal) to keep data types constant
        # after distributed pass parallelization (avoid recompilation).
        func_text += f"  return _bodo_groupby_apply_impl(({key_name_args},), in_df, {extra_arg_names} get_null_shuffle_info(), unliteral_val(False))\n"

        loc_vars = {}
        glbls = {
            "__col_name_meta_value_inner": ColNamesMetaType(tuple(used_cols)),
            "get_null_shuffle_info": bodo.libs.array.get_null_shuffle_info,
            "unliteral_val": bodo.utils.typing.unliteral_val,
            "range_index_to_numeric": bodo.hiframes.pd_index_ext.range_index_to_numeric,
        }
        exec(func_text, glbls, loc_vars)
        f = loc_vars["f"]

        nodes = []
        key_vars = [self._get_dataframe_data(df_var, c, nodes) for c in grp_typ.keys]
        col_vars = [self._get_dataframe_data(df_var, c, nodes) for c in used_cols]
        df_index_var = self._get_dataframe_index(df_var, nodes)
        map_func = bodo.compiler.udf_jit(func)

        func_text = self._gen_groupby_apply_func(
            grp_typ,
            n_keys,
            n_out_cols,
            extra_arg_names,
            udf_arg_names,
            udf_return_type,
            out_typ,
        )

        glbs = {
            "numba": numba,
            "np": np,
            "bodo": bodo,
            "get_group_indices": bodo.hiframes.pd_groupby_ext.get_group_indices,
            "generate_slices": bodo.hiframes.pd_groupby_ext.generate_slices,
            "delete_shuffle_info": bodo.libs.array.delete_shuffle_info,
            "dist_reduce": bodo.libs.distributed_api.dist_reduce,
            "map_func": map_func,
            "reverse_shuffle": bodo.hiframes.pd_groupby_ext.reverse_shuffle,
        }
        if isinstance(out_typ, DataFrameType):
            glbs.update(
                {
                    "__col_name_meta_value_groupby_apply": ColNamesMetaType(
                        tuple(out_typ.columns)
                    )
                }
            )
        out_arr_types = out_typ.data
        out_arr_types = (
            out_arr_types if isinstance(out_typ, DataFrameType) else [out_arr_types]
        )
        for i in range(n_out_cols):
            glbs[f"_arr_typ{i}"] = out_arr_types[i]
        loc_vars = {}
        exec(func_text, glbs, loc_vars)
        _bodo_groupby_apply_impl = bodo.jit(distributed=False)(
            loc_vars["_bodo_groupby_apply_impl"]
        )

        glbs = {
            "numba": numba,
            "np": np,
            "bodo": bodo,
            "out_type": out_typ,
            "_bodo_groupby_apply_impl": _bodo_groupby_apply_impl,
        }

        return nodes + compile_func_single_block(
            f,
            key_vars + col_vars + [df_index_var] + extra_args,
            lhs,
            self,
            extra_globals=glbs,
        )

    def _gen_groupby_apply_func(
        self,
        grp_typ,
        n_keys,
        n_out_cols,
        extra_arg_names,
        udf_arg_names,
        udf_return_type,
        out_typ,
    ):
        """generate groupby apply function that groups input rows, calls the UDF, and
        constructs the output.
        """
        # TODO: [BE-1266] A suggested refactor code
        # remove NA groups from both out_labels and sort_idx when (dropna=True)
        # when dropna=True, we don't need to keep track of groupby labels and sort_idx for NA values

        func_text = f"def _bodo_groupby_apply_impl(keys, in_df, {extra_arg_names}shuffle_info, _is_parallel):\n"
        func_text += "  ev_apply = bodo.utils.tracing.Event('gb.apply', _is_parallel)\n"

        # get groupby info
        func_text += "  ev_gp_indices_data = bodo.utils.tracing.Event('group_indices_data_key', _is_parallel)\n"
        func_text += f"  sort_idx, group_indices, ngroups = get_group_indices(keys, {grp_typ.dropna}, _is_parallel)\n"
        # TODO This can be done in C++ as well.
        # This will avoid returning group_indices back.
        func_text += (
            "  starts, ends = generate_slices(group_indices[sort_idx], ngroups)\n"
        )
        func_text += "  ev_gp_indices_data.add_attribute('g_ngroups', ngroups)\n"
        func_text += f"  ev_gp_indices_data.add_attribute('n_keys', {n_keys})\n"
        # sort keys and data
        for i in range(n_keys):
            func_text += f"  s_key{i} = keys[{i}][sort_idx]\n"
        is_series_in = grp_typ.series_select and len(grp_typ.selection) == 1
        func_text += "  in_data = in_df.iloc[sort_idx{}]\n".format(
            ",0" if is_series_in else ""
        )
        func_text += "  ev_gp_indices_data.add_attribute('in_data', len(in_data))\n"
        func_text += "  ev_gp_indices_data.finalize()\n"

        # whether UDF returns a single row (as Series) or scalar
        if (
            isinstance(udf_return_type, (SeriesType, HeterogeneousSeriesType))
            and udf_return_type.const_info is not None
        ) or not isinstance(udf_return_type, (SeriesType, DataFrameType)):
            func_text += self._gen_groupby_apply_row_loop(
                grp_typ, udf_return_type, out_typ, udf_arg_names, n_out_cols, n_keys
            )
        else:
            func_text += self._gen_groupby_apply_acc_loop(
                grp_typ, udf_return_type, out_typ, udf_arg_names, n_out_cols, n_keys
            )
        return func_text

    def _gen_groupby_apply_row_loop(
        self, grp_typ, udf_return_type, out_typ, udf_arg_names, n_out_cols, n_keys
    ):
        """generate groupby apply loop in cases where the UDF output returns a single
        row of output
        """

        n_out_keys = 0 if grp_typ.as_index else n_keys
        sum_no = bodo.libs.distributed_api.Reduce_Type.Sum.value

        func_text = ""
        func_text += "  ev = bodo.utils.tracing.Event('_gen_groupby_apply_row_loop', _is_parallel)\n"

        df = grp_typ.df_type
        # output always has input keys (either Index or regular columns)
        for i in range(n_keys):
            # Get key array type to compare if it's dictionary string encoded.
            # For dict-encoded string, allocate array for indices only.
            # Then, create in_key_arrs as dictionary-encoded string at the end.
            key_typ = df.data[df.column_index[grp_typ.keys[i]]]
            if key_typ == bodo.types.dict_str_arr_type:
                func_text += f"  dict_key_indices_arrs{i} = bodo.libs.int_arr_ext.alloc_int_array(ngroups, np.int32)\n"
            else:
                func_text += f"  in_key_arrs{i} = bodo.utils.utils.alloc_type(ngroups, s_key{i}, (-1,))\n"
        for i in range(n_out_cols - n_out_keys):
            func_text += f"  arrs{i} = bodo.utils.utils.alloc_type(ngroups, _arr_typ{i + n_out_keys}, (-1,))\n"
        # as_index=False includes group number as Index
        # NOTE: Pandas assigns group numbers in sorted order to Index when
        # as_index=False. Matching it exactly requires expensive sorting, so we assign
        # numbers in the order of groups across processors (using exscan)
        if not grp_typ.as_index:
            func_text += "  out_index_arr = np.empty(ngroups, np.int64)\n"
            func_text += "  n_prev_groups = 0\n"
            func_text += "  if _is_parallel:\n"
            func_text += f"    n_prev_groups = bodo.libs.distributed_api.dist_exscan(ngroups, np.int32({sum_no}))\n"

        # loop over groups and call UDF
        func_text += "  for i in range(ngroups):\n"
        func_text += "    piece = in_data[starts[i]:ends[i]]\n"

        func_text += f"    out = map_func(piece, {udf_arg_names})\n"
        if isinstance(udf_return_type, (SeriesType, HeterogeneousSeriesType)):
            func_text += (
                "    out_vals = bodo.hiframes.pd_series_ext.get_series_data(out)\n"
            )
            for i in range(n_out_cols - n_out_keys):
                func_text += f"    arrs{i}[i] = bodo.utils.conversion.unbox_if_tz_naive_timestamp(out_vals[{i}])\n"
        else:
            func_text += "    arrs0[i] = bodo.utils.conversion.unbox_if_tz_naive_timestamp(out)\n"
        for i in range(n_keys):
            key_typ = df.data[df.column_index[grp_typ.keys[i]]]
            if key_typ == bodo.types.dict_str_arr_type:
                func_text += (
                    f"    if bodo.libs.array_kernels.isna(s_key{i}, starts[i]):\n"
                )
                func_text += f"      bodo.libs.array_kernels.setna(dict_key_indices_arrs{i}, i)\n"
                func_text += "    else:\n"
                func_text += f"      dict_key_indices_arrs{i}[i] = s_key{i}._indices[starts[i]]\n"
            else:
                func_text += f"    bodo.libs.array_kernels.copy_array_element(in_key_arrs{i}, i, s_key{i}, starts[i])\n"
        if not grp_typ.as_index:
            func_text += "    out_index_arr[i] = n_prev_groups + i\n"

        # create output dataframe
        if grp_typ.as_index:
            index_names = ", ".join(
                f"'{v}'" if isinstance(v, str) else f"{v}" for v in grp_typ.keys
            )
        else:
            index_names = "None"
        for i in range(n_keys):
            key_typ = df.data[df.column_index[grp_typ.keys[i]]]
            if key_typ == bodo.types.dict_str_arr_type:
                func_text += f"  in_key_arrs{i} = bodo.libs.dict_arr_ext.init_dict_arr(s_key{i}._data, dict_key_indices_arrs{i}, False, False, s_key{i}._dict_id)\n"

        if isinstance(out_typ.index, MultiIndexType):
            out_key_arr_names = ", ".join(f"in_key_arrs{i}" for i in range(n_keys))
            func_text += f"  out_index = bodo.hiframes.pd_multi_index_ext.init_multi_index(({out_key_arr_names},), ({index_names},), None)\n"
        else:
            out_index_arr = "out_index_arr" if not grp_typ.as_index else "in_key_arrs0"
            func_text += f"  out_index = bodo.utils.conversion.index_from_array({out_index_arr}, {index_names})\n"

        out_data = ", ".join(f"arrs{i}" for i in range(n_out_cols - n_out_keys))
        if not grp_typ.as_index:
            out_data = (
                ", ".join(f"in_key_arrs{i}" for i in range(n_keys)) + ", " + out_data
            )
        # parallel shuffle clean up
        func_text += "  if _is_parallel:\n"
        func_text += "    delete_shuffle_info(shuffle_info)\n"
        func_text += "  ev.finalize()\n"
        func_text += "  ev_apply.finalize()\n"

        if isinstance(out_typ, DataFrameType):
            # This column name metadata value is generated at the point of execution of the functext in _run_call_groupby_apply when the out type is a dataframe
            func_text += f"  return bodo.hiframes.pd_dataframe_ext.init_dataframe(({out_data},), out_index, __col_name_meta_value_groupby_apply)\n"
        else:
            func_text += "  return bodo.hiframes.pd_series_ext.init_series(arrs0, out_index, None)\n"
        return func_text

    def _gen_groupby_apply_acc_loop(
        self, grp_typ, udf_return_type, out_typ, udf_arg_names, n_out_cols, n_keys
    ):
        """generate groupby apply loop in cases where the UDF output is multiple rows
        and needs to be accumulated properly
        """

        sum_no = bodo.libs.distributed_api.Reduce_Type.Sum.value

        func_text = ""
        func_text += "  ev = bodo.utils.tracing.Event('_gen_groupby_apply_acc_loop', _is_parallel)\n"

        # gather output array, index and keys in lists to concatenate for output
        for i in range(n_out_cols):
            func_text += f"  arrs{i} = []\n"
        if grp_typ.as_index:
            for i in range(n_keys):
                func_text += f"  in_key_arrs{i} = []\n"
        else:
            func_text += "  in_key_arr = []\n"
        func_text += "  arrs_index = []\n"

        # NOTE: Pandas assigns group numbers in sorted order to Index when
        # as_index=False. Matching it exactly requires expensive sorting, so we assign
        # numbers in the order of groups across processors (using exscan)
        if not grp_typ.as_index:
            func_text += "  n_prev_groups = 0\n"
            func_text += "  if _is_parallel:\n"
            func_text += f"    n_prev_groups = bodo.libs.distributed_api.dist_exscan(ngroups, np.int32({sum_no}))\n"
        # NOTE: Pandas tracks whether output Index is same as input, and reorders output
        # to match input if Index hasn't changed
        # https://github.com/pandas-dev/pandas/blob/9ee8674a9fb593f138e66d7b108a097beaaab7f2/pandas/_libs/reduction.pyx#L369
        func_text += "  mutated = False\n"

        # loop over groups and call UDF
        func_text += "  for i in range(ngroups):\n"
        func_text += "    piece = in_data[starts[i]:ends[i]]\n"

        func_text += f"    out_df = map_func(piece, {udf_arg_names})\n"
        if isinstance(udf_return_type, SeriesType):
            func_text += (
                "    out_idx = bodo.hiframes.pd_series_ext.get_series_index(out_df)\n"
            )
        else:
            func_text += "    out_idx = bodo.hiframes.pd_dataframe_ext.get_dataframe_index(out_df)\n"
        func_text += "    mutated |= out_idx is not piece.index\n"
        func_text += (
            "    arrs_index.append(bodo.utils.conversion.index_to_array(out_idx))\n"
        )
        # all rows of returned df will get the same key value in output index
        df = grp_typ.df_type
        if grp_typ.as_index:
            for i in range(n_keys):
                # all rows of output get the input keys as Index. Hence, create an array
                # of key values with same length as output
                key_typ = df.data[df.column_index[grp_typ.keys[i]]]
                if key_typ == bodo.types.dict_str_arr_type:
                    func_text += f"    in_key_arrs{i}.append(bodo.utils.utils.full_type(len(out_df), s_key{i}._indices[starts[i]], s_key{i}._indices))\n"
                else:
                    func_text += f"    in_key_arrs{i}.append(bodo.utils.utils.full_type(len(out_df), s_key{i}[starts[i]], s_key{i}))\n"
        else:
            func_text += (
                "    in_key_arr.append(np.full(len(out_df), n_prev_groups + i))\n"
            )
        if isinstance(udf_return_type, SeriesType):
            func_text += "    arrs0.append(bodo.hiframes.pd_series_ext.get_series_data(out_df))\n"
        else:
            for i in range(n_out_cols):
                func_text += f"    arrs{i}.append(bodo.hiframes.pd_dataframe_ext.get_dataframe_data(out_df, {i}))\n"
        for i in range(n_out_cols):
            func_text += f"  out_arr{i} = bodo.libs.array_kernels.concat(arrs{i})\n"
        if grp_typ.as_index:
            for i in range(n_keys):
                key_typ = df.data[df.column_index[grp_typ.keys[i]]]
                if key_typ == bodo.types.dict_str_arr_type:
                    func_text += f"  out_key_arr_dict_index{i} = bodo.libs.array_kernels.concat(in_key_arrs{i})\n"
                    func_text += f"  out_key_arr{i} = bodo.libs.dict_arr_ext.init_dict_arr(s_key{i}._data, out_key_arr_dict_index{i}, s_key{i}._has_global_dictionary, s_key{i}._has_unique_local_dictionary, s_key{i}._dict_id)\n"
                else:
                    func_text += f"  out_key_arr{i} = bodo.libs.array_kernels.concat(in_key_arrs{i})\n"

            out_key_arr_names = ", ".join(f"out_key_arr{i}" for i in range(n_keys))
        else:
            func_text += "  out_key_arr = bodo.libs.array_kernels.concat(in_key_arr)\n"
            out_key_arr_names = "out_key_arr"

        # create output dataframe
        # TODO(ehsan): support MultiIndex in input and UDF output
        if grp_typ.as_index:
            index_names = ", ".join(
                f"'{v}'" if isinstance(v, str) else f"{v}" for v in grp_typ.keys
            )
        else:
            index_names = "None"
        index_names += ", in_df.index.name"
        func_text += "  out_idx_arr_all = bodo.libs.array_kernels.concat(arrs_index)\n"
        func_text += f"  out_index = bodo.hiframes.pd_multi_index_ext.init_multi_index(({out_key_arr_names}, out_idx_arr_all), ({index_names},), None)\n"

        # Unset global flag of output dictionary-encoded arrays in case map_func
        # kernel reuses input dictionaries which are global after shuffle.
        # The ranks with empty data won't have the global flag on which causes a hang
        # in reverse shuffle below.
        # See https://bodo.atlassian.net/browse/BSE-2566
        for i in range(n_out_cols):
            func_text += (
                f"  out_arr{i} = bodo.libs.dict_arr_ext.unset_dict_global(out_arr{i})\n"
            )

        # reorder output to match input if UDF Index is same as input
        func_text += "  if _is_parallel:\n"
        # synchronize since some ranks may avoid mutation due to corner cases
        func_text += (
            f"    mutated = bool(dist_reduce(int(mutated), np.int32({sum_no})))\n"
        )
        func_text += "  if not mutated:\n"

        # Use Bodo's implementation of argsort instead of numba (sort_idx.argsort())
        # Numba's implementation is slow.
        # https://bodo.atlassian.net/browse/BE-4053
        func_text += "    rev_idx = bodo.hiframes.series_impl.argsort(sort_idx)\n"
        func_text += "    out_index = out_index[rev_idx]\n"
        for i in range(n_out_cols):
            func_text += f"    out_arr{i} = out_arr{i}[rev_idx]\n"
        func_text += "    if _is_parallel:\n"
        func_text += "      out_index = reverse_shuffle(out_index, shuffle_info)\n"
        for i in range(n_out_cols):
            func_text += (
                f"      out_arr{i} = reverse_shuffle(out_arr{i}, shuffle_info)\n"
            )

        # parallel shuffle clean up
        func_text += "  if _is_parallel:\n"
        func_text += "    delete_shuffle_info(shuffle_info)\n"

        out_data = ", ".join(f"out_arr{i}" for i in range(n_out_cols))
        if isinstance(out_typ, SeriesType):
            # some ranks may have empty data after shuffle (ngroups == 0), so call the
            # UDF with empty data to get the name of the output Series
            func_text += f"  out_name = out_df.name if ngroups else map_func(in_data[0:0], {udf_arg_names}).name\n"
            func_text += "  ev.finalize()\n"
            func_text += "  ev_apply.finalize()\n"
            func_text += "  return bodo.hiframes.pd_series_ext.init_series(out_arr0, out_index, out_name)\n"
        else:
            func_text += "  ev.finalize()\n"
            func_text += "  ev_apply.finalize()\n"
            # This column name metadata value is generated at the point of execution of the functext in _run_call_groupby_apply when the out type is a dataframe
            func_text += f"  return bodo.hiframes.pd_dataframe_ext.init_dataframe(({out_data},), out_index, __col_name_meta_value_groupby_apply)\n"

        return func_text

    def _run_call_pipe(self, rhs, obj_var):
        """generate IR nodes for df/series/groupby.pipe().
        Transform: grp.pipe(f, args) -> f(grp, args)
        """
        # get pipe function and args
        kws = dict(rhs.kws)
        func_var = get_call_expr_arg("pipe", rhs.args, kws, 0, "func")
        func = get_overload_const_func(self.typemap[func_var.name], self.func_ir)
        extra_args = [] if len(rhs.args) < 2 else rhs.args[1:]
        args = [obj_var] + list(extra_args)

        return replace_func(
            self,
            func,
            args,
            kws=rhs.kws,
            pysig=numba.core.utils.pysignature(func),
            run_full_pipeline=True,
        )

    def _get_groupby_df_obj(self, obj_var):
        """get df object for groupby()
        e.g. groupby('A')['B'], groupby('A')['B', 'C'], groupby('A')
        """
        select_def = guard(get_definition, self.func_ir, obj_var)
        if isinstance(select_def, ir.Expr) and select_def.op in (
            "getitem",
            "static_getitem",
            "getattr",
        ):
            obj_var = select_def.value

        obj_call = guard(get_definition, self.func_ir, obj_var)
        # find dataframe
        call_def = guard(find_callname, self.func_ir, obj_call, self.typemap)
        if call_def == ("init_groupby", "bodo.hiframes.pd_groupby_ext"):
            return obj_call.args[0]
        else:  # pragma: no cover
            # TODO(ehsan): support groupby obj through control flow & function args
            raise BodoError("Invalid groupby call", loc=obj_var.loc)

    def _get_const_tup(self, tup_var):
        tup_def = guard(get_definition, self.func_ir, tup_var)
        if isinstance(tup_def, ir.Expr):
            if tup_def.op == "binop" and tup_def.fn in ("+", operator.add):
                return self._get_const_tup(tup_def.lhs) + self._get_const_tup(
                    tup_def.rhs
                )
            if tup_def.op in ("build_tuple", "build_list"):
                return tup_def.items
        raise BodoError("constant tuple expected")

    def _get_dataframe_data(self, df_var, col_name, nodes):
        # optimization: return data var directly if not ambiguous
        # (no multiple init_dataframe calls for the same df_var with control
        # flow)
        # e.g. A = init_dataframe(A, None, 'A')
        # XXX assuming init_dataframe is the only call to create a dataframe
        # and dataframe._data is never overwritten
        df_typ = self.typemap[df_var.name]
        ind = df_typ.columns.index(col_name)
        var_def = guard(get_definition, self.func_ir, df_var)
        call_def = guard(find_callname, self.func_ir, var_def, self.typemap)
        if not self._is_updated_df(df_var.name) and call_def == (
            "init_dataframe",
            "bodo.hiframes.pd_dataframe_ext",
        ):
            seq_info = guard(find_build_sequence, self.func_ir, var_def.args[0])
            if seq_info is not None:
                if self.typemap[df_var.name].is_table_format:
                    # If we have a table format we replace the get_dataframe_data
                    # with a get_table_data call so we can perform dead column
                    # elimination.
                    table_var = seq_info[0][0]
                    loc = df_var.loc
                    ind_var = ir.Var(df_var.scope, mk_unique_var("col_ind"), loc)
                    self.typemap[ind_var.name] = types.IntegerLiteral(ind)
                    nodes.append(ir.Assign(ir.Const(ind, loc), ind_var, loc))
                    nodes += compile_func_single_block(
                        eval(
                            "lambda table, ind: bodo.hiframes.table.get_table_data(table, ind)"
                        ),
                        (table_var, ind_var),
                        None,
                        self,
                    )
                    return nodes[-1].target
                return seq_info[0][ind]

        loc = df_var.loc
        ind_var = ir.Var(df_var.scope, mk_unique_var("col_ind"), loc)
        self.typemap[ind_var.name] = types.IntegerLiteral(ind)
        nodes.append(ir.Assign(ir.Const(ind, loc), ind_var, loc))
        # XXX use get_series_data() for getting data instead of S._data
        # to enable alias analysis
        nodes += compile_func_single_block(
            eval(
                "lambda df, c_ind: bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, c_ind)"
            ),
            (df_var, ind_var),
            None,
            self,
        )
        return nodes[-1].target

    def _get_dataframe_index(self, df_var, nodes):
        var_def = guard(get_definition, self.func_ir, df_var)
        call_def = guard(find_callname, self.func_ir, var_def, self.typemap)
        # TODO(ehsan): make sure dataframe index is not updated elsewhere
        if call_def == ("init_dataframe", "bodo.hiframes.pd_dataframe_ext"):
            return var_def.args[1]

        # XXX use get_series_data() for getting data instead of S._data
        # to enable alias analysis
        nodes += compile_func_single_block(
            eval("lambda df: bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)"),
            (df_var,),
            None,
            self,
        )
        return nodes[-1].target

    def _get_dataframe_table(self, df_var: ir.Var, nodes: list[ir.Stmt]):
        """Returns the table used by a DataFrame with
        table format. If the DataFrame's init_dataframe call is in the
        IR, it extracts an existing ir.Var for the table. Otherwise
        it generates a get_dataframe_table call.

        Args:
            df_var (ir.Var): IR Variable for the DataFrame
            nodes (List[ir.Stmt]): List of IR statements preceding this call.
                If code must be generated it should be added to this list.

        Returns:
            ir.Var: Returns an IR Variable for the table.
        """
        df_typ = self.typemap[df_var.name]
        assert df_typ.is_table_format, "_get_dataframe_table requires table format"
        var_def = guard(get_definition, self.func_ir, df_var)
        call_def = guard(find_callname, self.func_ir, var_def, self.typemap)
        if not self._is_updated_df(df_var.name) and call_def == (
            "init_dataframe",
            "bodo.hiframes.pd_dataframe_ext",
        ):
            seq_info = guard(find_build_sequence, self.func_ir, var_def.args[0])
            if seq_info is not None:
                # The table is always the first/only element of the tuple.
                return seq_info[0][0]

        nodes += compile_func_single_block(
            eval("lambda df: bodo.hiframes.pd_dataframe_ext.get_dataframe_table(df)"),
            (df_var,),
            None,
            self,
        )
        return nodes[-1].target

    def _get_index_name(self, dt_var, nodes):
        var_def = guard(get_definition, self.func_ir, dt_var)
        call_def = guard(find_callname, self.func_ir, var_def, self.typemap)
        if (
            call_def
            in (
                ("init_timedelta_index", "bodo.hiframes.pd_index_ext"),
                ("init_binary_str_index", "bodo.hiframes.pd_index_ext"),
                ("init_numeric_index", "bodo.hiframes.pd_index_ext"),
                ("init_categorical_index", "bodo.hiframes.pd_index_ext"),
                ("init_heter_index", "bodo.hiframes.pd_index_ext"),
            )
            and len(var_def.args) == 2
        ) or (
            call_def == ("init_datetime_index", "bodo.hiframes.pd_index_ext")
            and len(var_def.args) == 3
        ):
            return var_def.args[1]

        f = eval("lambda S: bodo.hiframes.pd_index_ext.get_index_name(S)")
        if self.typemap[dt_var.name] == types.none:
            f = eval("lambda S: None")

        nodes += compile_func_single_block(f, (dt_var,), None, self)
        return nodes[-1].target

    def _is_updated_df(self, varname):
        """returns True if columns of dataframe 'varname' may be updated inplace
        somewhere in the program.
        """
        if varname in self._updated_dataframes:
            return True
        if varname in self._visited_updated_dataframes:
            return False
        self._visited_updated_dataframes.add(varname)
        updated_df = any(
            self._is_updated_df(v.name)
            for v in self.func_ir._definitions[varname]
            if (
                isinstance(v, ir.Var) and v.name not in self._visited_updated_dataframes
            )
        )
        # Cache updated dataframes to avoid redundant checks.
        if updated_df:
            self._updated_dataframes.add(varname)
        return updated_df

    def _is_df_var(self, var):
        return isinstance(self.typemap[var.name], DataFrameType)

    def is_bool_arr(self, varname):
        typ = self.typemap[varname]
        return (
            isinstance(typ, (SeriesType, types.Array, BooleanArrayType))
            and typ.dtype == types.bool_
        )

    def _get_const_or_list(
        self, by_arg, list_only=False, default=None, err_msg=None, typ=None
    ):
        var_typ = self.typemap[by_arg.name]
        if isinstance(var_typ, types.Optional):
            var_typ = var_typ.type
        if is_overload_constant_list(var_typ):
            return get_overload_const_list(var_typ)
        if is_literal_type(var_typ):
            return [get_literal_value(var_typ)]

        typ = str if typ is None else typ
        by_arg_def = guard(find_build_sequence, self.func_ir, by_arg)
        if by_arg_def is None:
            # try single key column
            by_arg_def = guard(find_const, self.func_ir, by_arg)
            if by_arg_def is None:
                if default is not None:
                    return default
                raise BodoError(err_msg)
            if isinstance(var_typ, types.BaseTuple):
                assert isinstance(by_arg_def, tuple)
                return by_arg_def
            key_colnames = (by_arg_def,)
        else:
            if list_only and by_arg_def[1] != "build_list":
                if default is not None:
                    return default
                raise BodoError(err_msg)
            key_colnames = tuple(
                guard(find_const, self.func_ir, v) for v in by_arg_def[0]
            )
            if any(not isinstance(v, typ) for v in key_colnames):
                if default is not None:
                    return default
                raise BodoError(err_msg)
        return key_colnames

    def _get_list_value_spec_length(self, by_arg, n_key, err_msg=None):
        """Used to returning a list of values of length n_key.
        If by_arg is a list of values then check that the list of length n_key.
        If by_arg is just a single value, then return the list of length n_key of this value.
        """
        var_typ = self.typemap[by_arg.name]
        if is_overload_constant_list(var_typ):
            vals = get_overload_const_list(var_typ)
            if len(vals) != n_key:
                raise BodoError(err_msg)
            return vals
        # try single key column
        by_arg_def = guard(find_const, self.func_ir, by_arg)
        if by_arg_def is None:
            raise BodoError(err_msg)
        key_colnames = (by_arg_def,) * n_key
        return key_colnames


func_text = """
def _check_query_series_bool(S):
    # a dummy function used in _run_call_query to catch data type error later in the
    # pipeline (S should be a Series(bool)).
    return S
"""
exec(func_text)
numba.extending.register_jitable(globals()["_check_query_series_bool"])


def _gen_init_df_dataframe_pass(columns, index=None, is_table_format=False):
    n_cols = len(columns)
    if is_table_format:
        # Table always uses only a single variable
        data_args = "data0"
    else:
        data_args = ", ".join(f"data{i}" for i in range(n_cols))
    args = data_args

    if index is None:
        assert n_cols > 0
        index = "bodo.hiframes.pd_index_ext.init_range_index(0, len(data0), 1, None)"
    else:
        args += ", " + index

    func_text = f"def _init_df({args}):\n"
    func_text += f"  return bodo.hiframes.pd_dataframe_ext.init_dataframe(({data_args},), {index}, __col_name_meta_value_gen_init_df_2)\n"
    loc_vars = {}
    exec(
        func_text,
        {
            "bodo": bodo,
            "__col_name_meta_value_gen_init_df_2": ColNamesMetaType(tuple(columns)),
        },
        loc_vars,
    )
    _init_df = loc_vars["_init_df"]
    return _init_df


def _get_df_apply_used_cols(func, columns):
    """find which df columns are actually used in UDF 'func' inside df.apply(func) or
    df.groupby().apply(func) if possible (has to be conservative and assume all columns
    are used when it cannot analyze the IR properly)
    """
    from bodo.utils.python_310_bytecode_pass import _transform_list_appends

    lambda_ir = numba.core.compiler.run_frontend(func)
    columns_set = set(columns)

    used_cols = set()
    l_topo_order = find_topo_order(lambda_ir.blocks)
    first_stmt = lambda_ir.blocks[l_topo_order[0]].body[0]
    assert isinstance(first_stmt, ir.Assign) and isinstance(first_stmt.value, ir.Arg)
    # NOTE: UDF argument can be dataframe or Series
    arg_var = first_stmt.target
    use_all_cols = False
    for bl in lambda_ir.blocks.values():
        # df.iloc[:,[c1, c2,..]] may have long constant lists
        _transform_list_appends(lambda_ir, bl)
        for stmt in bl.body:
            # ignore ir.Arg
            if is_assign(stmt) and isinstance(stmt.value, ir.Arg):
                continue

            # ignore df.loc getattr
            if (
                is_assign(stmt)
                and is_expr(stmt.value, "getattr")
                and stmt.value.attr == "loc"
            ):
                continue

            # match df.C column access
            if (
                is_assign(stmt)
                and is_expr(stmt.value, "getattr")
                and stmt.value.value.name == arg_var.name
                and stmt.value.attr in columns_set
            ):
                used_cols.add(stmt.value.attr)
                continue

            # match df["C"], df[["C", "D"]], df.loc[:, ["C", "D"]]
            if is_assign(stmt) and is_expr(stmt.value, "getitem"):
                # df["C"], df[["C", "D"]]
                if stmt.value.value.name == arg_var.name:
                    cols = guard(get_const_value_inner, lambda_ir, stmt.value.index)
                    if not isinstance(cols, list):
                        cols = [cols]
                    cols_set = set(cols)
                    # make sure getitem indices are column names since input argument
                    # can be a Series which allows integer indexing, e.g. r[[1, 2]]
                    if not (cols_set - columns_set):
                        used_cols.update(cols_set)
                        continue

                # df.loc[:, ["C", "D"]]
                val_def = guard(get_definition, lambda_ir, stmt.value.value)
                if (
                    is_expr(val_def, "getattr")
                    and val_def.attr == "loc"
                    and val_def.value.name == arg_var.name
                ):
                    idx = guard(find_build_tuple, lambda_ir, stmt.value.index)
                    if idx is not None and len(idx) == 2:
                        cols = guard(get_const_value_inner, lambda_ir, idx[1])
                        cols_set = set(cols)
                        if not (cols_set - columns_set):
                            used_cols.update(cols_set)
                            continue

            vnames = {v.name for v in stmt.list_vars()}
            if arg_var.name in vnames:
                # argument is used in some other form
                # be conservative and use all cols
                use_all_cols = True
                used_cols = columns_set
                break

        if use_all_cols:
            break

    # remove duplicates with set() since a column can be used multiple times
    # keep the order the same as original columns to avoid errors with int getitem on
    # rows
    # Create a dictionary for scaling to large numbers of columns.
    cols_dict = {name: i for i, name in enumerate(columns)}
    used_cols = [c for (_, c) in sorted((cols_dict[v], v) for v in used_cols)]
    return used_cols
