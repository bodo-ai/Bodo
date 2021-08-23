# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
converts data frame operations to Series and Array operations
as much as possible to provide implementation and enable optimization.
Creates specialized IR nodes for complex operations like Join.
"""
import operator
import warnings
from collections import defaultdict

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
from bodo.ir.aggregate import get_agg_func
from bodo.libs.bool_arr_ext import BooleanArrayType
from bodo.utils.transform import (
    compile_func_single_block,
    gen_const_tup,
    get_call_expr_arg,
    get_const_value,
    replace_func,
)
from bodo.utils.typing import (
    BodoError,
    get_literal_value,
    get_overload_const_func,
    get_overload_const_list,
    get_overload_const_str,
    get_overload_const_tuple,
    is_literal_type,
    is_overload_constant_list,
    is_overload_constant_tuple,
    is_overload_none,
    list_cumulative,
)
from bodo.utils.utils import (
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

    def __init__(self, func_ir, typingctx, typemap, calltypes):
        self.func_ir = func_ir
        self.typingctx = typingctx
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
        lhs = assign.target
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
        # TODO: check invalid df.Attr?
        if isinstance(rhs_type, DataFrameType) and rhs.attr in rhs_type.columns:
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
        if (
            isinstance(rhs_type, DataFrameType)
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
            _init_df = _gen_init_df(new_names, "index")
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
            return self._run_call_len(lhs, rhs.args[0])

        if fdef == ("set_df_col", "bodo.hiframes.dataframe_impl"):
            return self._run_call_set_df_column(assign, lhs, rhs)

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
                assert seq_info is not None
                assign.value = seq_info[0][ind]

        if fdef == ("get_dataframe_index", "bodo.hiframes.pd_dataframe_ext"):
            df_var = rhs.args[0]
            var_def = guard(get_definition, self.func_ir, df_var)
            call_def = guard(find_callname, self.func_ir, var_def, self.typemap)
            if call_def == ("init_dataframe", "bodo.hiframes.pd_dataframe_ext"):
                assign.value = var_def.args[1]

        if fdef == ("join_dummy", "bodo.hiframes.pd_dataframe_ext"):
            return self._run_call_join(assign, lhs, rhs)

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

        if fdef == ("pivot_table_dummy", "bodo.hiframes.pd_groupby_ext"):
            return self._run_call_pivot_table(assign, lhs, rhs)

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
        func = get_overload_const_func(self.typemap[func_var.name], self.func_ir)
        out_typ = self.typemap[lhs.name]
        is_df_output = isinstance(out_typ, DataFrameType)
        out_arr_types = out_typ.data
        out_arr_types = out_arr_types if is_df_output else [out_arr_types]
        n_out_cols = len(out_arr_types)
        extra_args = get_call_expr_arg("apply", rhs.args, kws, 4, "args", [])
        if extra_args:
            extra_args = guard(find_build_sequence, self.func_ir, extra_args)
            extra_args = [] if extra_args is None else extra_args[0]

        # find kw arguments to UDF (pop apply() args first)
        kws.pop("func", None)
        kws.pop("axis", None)
        kws.pop("raw", None)
        kws.pop("result_type", None)
        kws.pop("args", None)
        udf_arg_names = (
            ", ".join("e{}".format(i) for i in range(len(extra_args)))
            + (", " if extra_args else "")
            + ", ".join(
                "{}=e{}".format(a, i + len(extra_args))
                for i, a in enumerate(kws.keys())
            )
        )
        extra_args += list(kws.values())
        extra_arg_names = ", ".join("e{}".format(i) for i in range(len(extra_args)))

        # find which columns are actually used if possible
        used_cols = _get_df_apply_used_cols(func, df_typ.columns)
        # avoid empty data which results in errors
        if not used_cols:
            used_cols = [df_typ.columns[0]]

        # prange func to inline
        col_name_args = ", ".join(["c" + str(i) for i in range(len(used_cols))])
        row_args = ", ".join(
            [
                "bodo.utils.conversion.box_if_dt64(c{}[i])".format(i)
                for i in range(len(used_cols))
            ]
        )

        func_text = "def f({}, df_index, {}):\n".format(col_name_args, extra_arg_names)
        func_text += "  numba.parfors.parfor.init_prange()\n"
        func_text += "  n = len(c0)\n"
        func_text += "  index_arr = bodo.utils.conversion.coerce_to_array(df_index)\n"

        for i in range(n_out_cols):
            func_text += (
                f"  S{i} = bodo.utils.utils.alloc_type(n, _arr_typ{i}, (-1,))\n"
            )
        func_text += "  for i in numba.parfors.parfor.internal_prange(n):\n"
        # TODO: unbox to array value if necessary (e.g. Timestamp to dt64)
        func_text += "    row_idx = bodo.hiframes.pd_index_ext.init_heter_index({}, bodo.utils.conversion.box_if_dt64(index_arr[i]))\n".format(
            gen_const_tup(used_cols)
        )
        # TODO: pass df_index[i] as row name (after issue with RangeIndex getitem in
        # test_df_apply_assertion is resolved)
        func_text += "    row = bodo.hiframes.pd_series_ext.init_series(({},), row_idx, bodo.utils.conversion.box_if_dt64(index_arr[i]))\n".format(
            row_args
        )
        func_text += "    v = map_func(row, {})\n".format(udf_arg_names)
        if is_df_output:
            func_text += "    v_vals = bodo.hiframes.pd_series_ext.get_series_data(v)\n"
            for i in range(n_out_cols):
                func_text += f"    v{i} = v_vals[{i}]\n"
        else:
            func_text += f"    v0 = v\n"
        for i in range(n_out_cols):
            func_text += (
                f"    S{i}[i] = bodo.utils.conversion.unbox_if_timestamp(v{i})\n"
            )
        if is_df_output:
            data_arrs = ", ".join(f"S{i}" for i in range(n_out_cols))
            col_names = gen_const_tup(self.typemap[lhs.name].columns)
            func_text += f"  return bodo.hiframes.pd_dataframe_ext.init_dataframe(({data_arrs},), df_index, {col_names})\n"
        else:
            func_text += (
                "  return bodo.hiframes.pd_series_ext.init_series(S0, df_index, None)\n"
            )

        loc_vars = {}
        exec(func_text, {}, loc_vars)
        f = loc_vars["f"]

        nodes = []
        col_vars = [self._get_dataframe_data(df_var, c, nodes) for c in used_cols]
        df_index_var = self._get_dataframe_index(df_var, nodes)
        map_func = bodo.compiler.udf_jit(func)

        glbs = {
            "numba": numba,
            "np": np,
            "bodo": bodo,
            "map_func": map_func,
            "init_nested_counts": bodo.utils.indexing.init_nested_counts,
            "add_nested_counts": bodo.utils.indexing.add_nested_counts,
        }
        for i in range(n_out_cols):
            glbs[f"_arr_typ{i}"] = out_arr_types[i]

        return replace_func(
            self,
            f,
            col_vars + [df_index_var] + extra_args,
            extra_globals=glbs,
            pre_nodes=nodes,
        )

    def _run_call_df_sort_values(self, assign, lhs, rhs):
        """Implements support for df.sort_values().
        Translates sort_values_dummy() to a Sort IR node.
        """
        df_var, by_var, ascending_var, inplace_var, na_position_var = rhs.args
        df_typ = self.typemap[df_var.name]
        inplace = guard(find_const, self.func_ir, inplace_var)
        na_position = guard(find_const, self.func_ir, na_position_var)

        # find key array for sort ('by' arg)
        by_type = self.typemap[by_var.name]
        if is_overload_constant_tuple(by_type):
            key_names = [get_overload_const_tuple(by_type)]
        else:
            key_names = get_overload_const_list(by_type)
        valid_keys_set = set(df_typ.columns)
        index_is_key = False
        index_name = "unset"
        if not is_overload_none(df_typ.index.name_typ):
            index_name = df_typ.index.name_typ.literal_value
            valid_keys_set.add(index_name)
            if index_name in key_names:
                index_is_key = True
        if "$_bodo_index_" in key_names:
            index_is_key = True
            index_name = "$_bodo_index_"
            valid_keys_set.add(index_name)
        # "A" is equivalent to ("A", "")
        key_names = [(k, "") if (k, "") in valid_keys_set else k for k in key_names]
        ascending_list = self._get_list_value_spec_length(
            ascending_var,
            len(key_names),
            err_msg="ascending should be bool or a list of bool of the number of keys",
        )
        # already checked in validate_sort_values_spec() so assertion is enough
        assert all(
            k in valid_keys_set for k in key_names
        ), f"invalid sort keys {key_names}"

        nodes = []
        in_vars = {
            c: self._get_dataframe_data(df_var, c, nodes) for c in df_typ.columns
        }
        in_index_var = self._gen_array_from_index(df_var, nodes)
        in_vars["$_bodo_index_"] = in_index_var

        # remove key from dfs (only data is kept)
        def get_value(c):
            if c == index_name:
                return in_index_var
            return in_vars.pop(c)

        in_key_arrs = [get_value(c) for c in key_names]
        if inplace:
            out_key_vars = in_key_arrs.copy()
            out_vars = in_vars.copy()
            out_index_var = in_index_var
        else:
            out_vars = {}
            for k in df_typ.columns:
                out_var = ir.Var(lhs.scope, mk_unique_var(sanitize_varname(k)), lhs.loc)
                ind = df_typ.columns.index(k)
                self.typemap[out_var.name] = df_typ.data[ind]
                out_vars[k] = out_var
            # index var
            out_index_var = ir.Var(lhs.scope, mk_unique_var("_index_"), lhs.loc)
            self.typemap[out_index_var.name] = self.typemap[in_index_var.name]
            out_key_vars = []
            if not index_is_key:
                out_vars["$_bodo_index_"] = out_index_var
            for k in key_names:
                if index_is_key and k == index_name:
                    out_key_vars.append(out_index_var)
                else:
                    out_key_vars.append(out_vars.pop(k))

        nodes.append(
            bodo.ir.sort.Sort(
                df_var.name,
                lhs.name,
                in_key_arrs,
                out_key_vars,
                in_vars,
                out_vars,
                inplace,
                lhs.loc,
                ascending_list,
                na_position,
            )
        )

        # output from input index
        in_df_index = self._get_dataframe_index(df_var, nodes)
        in_df_index_name = self._get_index_name(in_df_index, nodes)
        out_index = self._gen_index_from_array(out_index_var, in_df_index_name, nodes)

        _init_df = _gen_init_df(df_typ.columns, "index")

        # XXX the order of output variables passed should match out_typ.columns
        out_arrs = []
        for c in df_typ.columns:
            if c in key_names:
                ind = key_names.index(c)
                out_arrs.append(out_key_vars[ind])
            else:
                out_arrs.append(out_vars[c])
        out_arrs.append(out_index)

        # return new df even for inplace case, since typing pass replaces input variable
        # using output of the call
        return nodes + compile_func_single_block(_init_df, out_arrs, lhs, self)

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
        name_consts = ", ".join(["'{}'".format(c) for c in df_typ.columns])

        func_text = "def f({}):\n".format(col_name_args)
        func_text += (
            "  return bodo.hiframes.dataframe_impl.get_itertuples({}, {})\n".format(
                name_consts, col_name_args
            )
        )

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
        data_args = tuple("data{}".format(i) for i in range(n_cols))

        func_text = "def _mean_impl({}):\n".format(", ".join(data_args))
        for d in data_args:
            ind = "bodo.hiframes.pd_index_ext.init_range_index(0, len({}), 1, None)".format(
                d
            )
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
                ", ".join("'{}'".format(c) for c in df_typ.columns)
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
        parsed_expr, parsed_expr_str, used_cols = self._parse_query_expr(
            expr, df_typ.columns
        )

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
            ind = "bodo.hiframes.pd_index_ext.init_range_index(0, len({}), 1, None)".format(
                c_var
            )
            func_text += (
                "  {0} = bodo.hiframes.pd_series_ext.init_series({0}, {1})\n".format(
                    c_var, ind
                )
            )
        # use dummy function to catch data type error
        func_text += "  return _check_query_series_bool({})".format(parsed_expr_str)
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

    def _parse_query_expr(self, expr, columns):
        """Parses query expression using Pandas parser but avoids issues such as
        early evaluation of string expressions by Pandas.
        """
        clean_name = pd.core.computation.parsing.clean_column_name
        cleaned_columns = [clean_name(c) for c in columns]
        resolver = {c: 0 for c in cleaned_columns}
        resolver["index"] = 0
        used_cols = {}
        # create fake environment for Expr that just includes the symbol names to
        # enable parsing
        glbs = self.func_ir.func_id.func.__globals__
        lcls = {a: 0 for a in self.func_ir.func_id.code.co_varnames}
        env = pd.core.computation.scope.ensure_scope(2, glbs, lcls, (resolver,))

        # avoid rewrite of operations in Pandas such as early evaluation of string exprs
        def _rewrite_membership_op(self, node, left, right):
            op_instance = node.op
            op = self.visit(op_instance)
            return op, op_instance, left, right

        def _maybe_evaluate_binop(
            self,
            op,
            op_class,
            lhs,
            rhs,
            eval_in_python=("in", "not in"),
            maybe_eval_in_python=("==", "!=", "<", ">", "<=", ">="),
        ):
            # currently, & and | are not supported in expr
            res = op(lhs, rhs)
            return res

        # avoid early evaluation of getattr such as C.str.contains().
        # functions like C.str.contains are saved and handled similar to
        # intrinsic functions like sqrt instead of evaluation.
        new_funcs = []

        class NewFuncNode(pd.core.computation.ops.FuncNode):
            def __init__(self, name):

                if name not in pd.core.computation.ops.MATHOPS or (
                    pd.core.computation.check._NUMEXPR_INSTALLED
                    and pd.core.computation.check_NUMEXPR_VERSION
                    < pd.core.computation.ops.LooseVersion("2.6.9")
                    and name in ("floor", "ceil")
                ):
                    if name not in new_funcs:
                        raise BodoError(
                            '"{0}" is not a supported function'.format(name)
                        )

                self.name = name
                if name in new_funcs:
                    self.func = name
                else:
                    self.func = getattr(np, name)

            def __call__(self, *args):
                return pd.core.computation.ops.MathCall(self, args)

            # __repr__ is needed if this attr node is not called, e.g. A.dt.year
            def __repr__(self):
                return pd.io.formats.printing.pprint_thing(self.name)

        def visit_Attribute(self, node, **kwargs):
            """handles value.attr cases such as C.str.contains()
            functions are turned into NewFuncNode. Intermediate values like C.str
            are added to local scope as local variable to avoid evaluation.
            """
            attr = node.attr
            value = node.value
            sentinel = pd.core.computation.ops.LOCAL_TAG

            if attr in ("str", "dt"):
                # check the case where df.column.str where column is not in df
                try:
                    value_str = str(self.visit(value))
                except pd.core.computation.ops.UndefinedVariableError as e:
                    col_name = e.args[0].split("'")[1]
                    raise BodoError(
                        "df.query(): column {} is not found in dataframe columns {}".format(
                            col_name, columns
                        )
                    )
            else:
                value_str = str(self.visit(value))
            name = value_str + "." + attr
            if name.startswith(sentinel):
                name = name[len(sentinel) :]

            # make local variable in case of C.str
            if attr in ("str", "dt"):
                orig_col_name = columns[cleaned_columns.index(value_str)]
                used_cols[orig_col_name] = value_str
                self.env.scope[name] = 0
                return self.term_type(sentinel + name, self.env)

            # make function node
            new_funcs.append(name)
            return NewFuncNode(name)

        # make sure string literals are printed correctly in expression
        def __str__(self):
            if isinstance(self.value, list):
                return "{}".format(self.value)
            if isinstance(self.value, str):
                return "'{}'".format(self.value)
            return pd.io.formats.printing.pprint_thing(self.name)

        # handle math calls
        def math__str__(self):
            """makes math calls compilable by adding "np." and Series functions"""
            # avoid change if it is a dummy attribute call
            if self.op in new_funcs:
                return pd.io.formats.printing.pprint_thing(
                    "{0}({1})".format(self.op, ",".join(map(str, self.operands)))
                )

            operands = map(
                lambda a: "bodo.hiframes.pd_series_ext.get_series_data({})".format(
                    str(a)
                ),
                self.operands,
            )
            op = "np.{}".format(self.op)
            ind = "bodo.hiframes.pd_index_ext.init_range_index(0, len({}), 1, None)".format(
                str(self.operands[0])
            )
            return pd.io.formats.printing.pprint_thing(
                "bodo.hiframes.pd_series_ext.init_series({0}({1}), {2})".format(
                    op, ",".join(operands), ind
                )
            )

        # replace 'in' operator with dummy function to convert to prange later
        def op__str__(self):
            parened = (
                "({0})".format(pd.io.formats.printing.pprint_thing(opr))
                for opr in self.operands
            )
            if self.op == "in":
                return pd.io.formats.printing.pprint_thing(
                    "bodo.hiframes.pd_dataframe_ext.val_isin_dummy({})".format(
                        ", ".join(parened)
                    )
                )
            if self.op == "not in":
                return pd.io.formats.printing.pprint_thing(
                    "bodo.hiframes.pd_dataframe_ext.val_notin_dummy({})".format(
                        ", ".join(parened)
                    )
                )
            return pd.io.formats.printing.pprint_thing(
                " {0} ".format(self.op).join(parened)
            )

        saved_rewrite_membership_op = (
            pd.core.computation.expr.BaseExprVisitor._rewrite_membership_op
        )
        saved_maybe_evaluate_binop = (
            pd.core.computation.expr.BaseExprVisitor._maybe_evaluate_binop
        )
        saved_visit_Attribute = pd.core.computation.expr.BaseExprVisitor.visit_Attribute
        saved__maybe_downcast_constants = (
            pd.core.computation.expr.BaseExprVisitor._maybe_downcast_constants
        )
        saved__str__ = pd.core.computation.ops.Term.__str__
        saved_math__str__ = pd.core.computation.ops.MathCall.__str__
        saved_op__str__ = pd.core.computation.ops.Op.__str__
        saved__disallow_scalar_only_bool_ops = (
            pd.core.computation.ops.BinOp._disallow_scalar_only_bool_ops
        )
        try:
            pd.core.computation.expr.BaseExprVisitor._rewrite_membership_op = (
                _rewrite_membership_op
            )
            pd.core.computation.expr.BaseExprVisitor._maybe_evaluate_binop = (
                _maybe_evaluate_binop
            )
            pd.core.computation.expr.BaseExprVisitor.visit_Attribute = visit_Attribute
            # _maybe_downcast_constants accesses actual value which is not possible
            pd.core.computation.expr.BaseExprVisitor._maybe_downcast_constants = (
                lambda self, left, right: (
                    left,
                    right,
                )
            )
            pd.core.computation.ops.Term.__str__ = __str__
            pd.core.computation.ops.MathCall.__str__ = math__str__
            pd.core.computation.ops.Op.__str__ = op__str__
            # _disallow_scalar_only_bool_ops accesses actual value which is not possible
            pd.core.computation.ops.BinOp._disallow_scalar_only_bool_ops = (
                lambda self: None
            )
            parsed_expr = pd.core.computation.expr.Expr(expr, env=env)
            parsed_expr_str = str(parsed_expr)
        except pd.core.computation.ops.UndefinedVariableError as e:
            # catch undefined variable error
            index_name = self.typemap["arg.df"].index.name_typ
            if (
                not is_overload_none(index_name)
                and get_overload_const_str(index_name) == e.args[0].split("'")[1]
            ):
                # currently do not support named index appears in expr
                raise BodoError(
                    "df.query(): Refering to named"
                    " index ('{}') by name is not supported".format(
                        get_overload_const_str(index_name)
                    )
                )
            else:
                # throw other errors
                # this includes: columns does not exist in dataframe,
                #                undefined local variable using @
                raise BodoError("df.query(): undefined variable, {}".format(e))
        finally:
            pd.core.computation.expr.BaseExprVisitor._rewrite_membership_op = (
                saved_rewrite_membership_op
            )
            pd.core.computation.expr.BaseExprVisitor._maybe_evaluate_binop = (
                saved_maybe_evaluate_binop
            )
            pd.core.computation.expr.BaseExprVisitor.visit_Attribute = (
                saved_visit_Attribute
            )
            pd.core.computation.expr.BaseExprVisitor._maybe_downcast_constants = (
                saved__maybe_downcast_constants
            )
            pd.core.computation.ops.Term.__str__ = saved__str__
            pd.core.computation.ops.MathCall.__str__ = saved_math__str__
            pd.core.computation.ops.Op.__str__ = saved_op__str__
            pd.core.computation.ops.BinOp._disallow_scalar_only_bool_ops = (
                saved__disallow_scalar_only_bool_ops
            )

        used_cols.update(
            {c: clean_name(c) for c in columns if clean_name(c) in parsed_expr.names}
        )
        return parsed_expr, parsed_expr_str, used_cols

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
        if isinstance(df_def, ir.Arg):
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
        in_arrs = [self._get_dataframe_data(df_var, c, nodes) for c in df_typ.columns]
        data_args = ["data{}".format(i) for i in range(n_cols)]
        out_columns = list(df_typ.columns)

        # if column is being added
        if cname not in df_typ.columns:
            data_args.append("new_arr")
            out_columns.append(cname)
            in_arrs.append(new_arr)
            new_arr_arg = "new_arr"
        else:  # updating existing column
            col_ind = df_typ.columns.index(cname)
            in_arrs[col_ind] = new_arr
            new_arr_arg = "data{}".format(col_ind)

        data_args = ", ".join(data_args)

        # TODO: fix list, Series data
        col_var = gen_const_tup(out_columns)
        df_index = "df_index"
        if n_cols == 0:
            df_index = "bodo.utils.conversion.extract_index_if_none(new_val, None)\n"
        func_text = "def _init_df({}, df_index, df, new_val):\n".format(data_args)
        # using len(df) instead of len(df_index) since len optimization works better for
        # dataframes
        func_text += "  {} = bodo.utils.conversion.coerce_to_array({}, scalar_to_arr_len=len(df))\n".format(
            new_arr_arg, new_arr_arg
        )
        func_text += "  return bodo.hiframes.pd_dataframe_ext.init_dataframe(({},), {}, {})\n".format(
            data_args, df_index, col_var
        )
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        _init_df = loc_vars["_init_df"]
        return replace_func(
            self, _init_df, in_arrs + [df_index_var, df_var, new_arr], pre_nodes=nodes
        )

    def _run_call_len(self, lhs, df_var):
        df_typ = self.typemap[df_var.name]

        # empty dataframe has 0 len
        if len(df_typ.columns) == 0:
            return [ir.Assign(ir.Const(0, lhs.loc), lhs, lhs.loc)]

        # run len on one of the columns
        # FIXME: it could potentially avoid remove dead for the column if
        # array analysis doesn't replace len() with it's size
        nodes = []
        arr = self._get_dataframe_data(df_var, df_typ.columns[0], nodes)

        func_text = "" "def f(df_arr):\n" "    return len(df_arr)\n"

        loc_vars = {}
        exec(func_text, globals(), loc_vars)
        return replace_func(self, loc_vars["f"], [arr], pre_nodes=nodes)

    def _run_call_join(self, assign, lhs, rhs):
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
        ) = rhs.args

        left_keys = self._get_const_or_list(left_on_var)
        right_keys = self._get_const_or_list(right_on_var)
        how = guard(find_const, self.func_ir, how_var)
        suffix_x = guard(find_const, self.func_ir, suffix_x_var)
        suffix_y = guard(find_const, self.func_ir, suffix_y_var)
        is_join = guard(find_const, self.func_ir, is_join_var)
        is_indicator = guard(find_const, self.func_ir, is_indicator_var)
        is_na_equal = guard(find_const, self.func_ir, _bodo_na_equal_var)
        out_typ = self.typemap[lhs.name]
        # convert right join to left join
        is_left = how in {"left", "outer"}
        is_right = how in {"right", "outer"}

        nodes = []
        out_data_vars = {
            c: ir.Var(lhs.scope, mk_unique_var(sanitize_varname(c)), lhs.loc)
            for c in out_typ.columns
        }
        for v, t in zip(out_data_vars.values(), out_typ.data):
            self.typemap[v.name] = t

        left_vars = {
            c: self._get_dataframe_data(left_df, c, nodes)
            for c in self.typemap[left_df.name].columns
        }
        right_vars = {
            c: self._get_dataframe_data(right_df, c, nodes)
            for c in self.typemap[right_df.name].columns
        }
        # In the case of pd.merge we have following behavior for the index:
        # ---if the key is a normal column then the joined table has a trivial index.
        # ---if one of the key is an index then it becomes an index.
        # In the case of df1.join(df2, ....) we have following behavior for the index:
        # ---the index of df1 is used for the merging. and the index of df2 or some other column.
        # ---The index of the joined table is assigned from the non-joined column.

        in_index_var = None
        out_index_var = None
        in_df_index_name = None
        right_index = "$_bodo_index_" in right_keys
        left_index = "$_bodo_index_" in left_keys
        # The index variables
        right_df_index = self._get_dataframe_index(right_df, nodes)
        right_df_index_name = self._get_index_name(right_df_index, nodes)
        right_index_var = self._gen_array_from_index(right_df, nodes)
        left_df_index = self._get_dataframe_index(left_df, nodes)
        left_df_index_name = self._get_index_name(left_df_index, nodes)
        left_index_var = self._gen_array_from_index(left_df, nodes)

        if left_index and not right_index:
            out_index_var = ir.Var(lhs.scope, mk_unique_var("out_index"), lhs.loc)
            self.typemap[out_index_var.name] = self.typemap[right_index_var.name]
            out_data_vars["$_bodo_index_"] = out_index_var
            left_vars["$_bodo_index_"] = left_index_var
            right_vars["$_bodo_index_"] = right_index_var
            in_df_index_name = right_df_index_name
        if right_index:
            out_index_var = ir.Var(lhs.scope, mk_unique_var("out_index"), lhs.loc)
            self.typemap[out_index_var.name] = self.typemap[left_index_var.name]
            out_data_vars["$_bodo_index_"] = out_index_var
            left_vars["$_bodo_index_"] = left_index_var
            right_vars["$_bodo_index_"] = right_index_var
            in_df_index_name = left_df_index_name
        nodes.append(
            bodo.ir.join.Join(
                lhs.name,
                left_df.name,
                right_df.name,
                left_keys,
                right_keys,
                out_data_vars,
                left_vars,
                right_vars,
                how,
                suffix_x,
                suffix_y,
                lhs.loc,
                is_left,
                is_right,
                is_join,
                left_index,
                right_index,
                is_indicator,
                is_na_equal,
            )
        )

        out_arrs = list(out_data_vars.values())
        if out_index_var is not None:
            out_index = self._gen_index_from_array(
                out_index_var, in_df_index_name, nodes
            )
            out_arrs = [v for c, v in out_data_vars.items() if c != "$_bodo_index_"]
            out_arrs.append(out_index)
            _init_df = _gen_init_df(out_typ.columns, "index")
        else:
            _init_df = _gen_init_df(out_typ.columns)

        return nodes + compile_func_single_block(_init_df, out_arrs, lhs, self)

    def _run_call_groupby(self, assign, lhs, rhs, grp_var, func_name):
        """Transform groupby calls into an Aggregate IR node"""
        if func_name == "apply":
            return self._run_call_groupby_apply(assign, lhs, rhs, grp_var)

        grp_typ = self.typemap[
            grp_var.name
        ]  # DataFrameGroupByType instance with initial typing info
        df_var = self._get_groupby_df_obj(
            grp_var
        )  # IR Var associated with groupby input dataframe
        df_type = self.typemap[df_var.name]  # Input dataframe type
        out_typ = self.typemap[lhs.name]  # Type of df.groupby() output

        nodes = []

        args = tuple([self.typemap[v.name] for v in rhs.args])
        kws = {k: self.typemap[v.name] for k, v in rhs.kws}
        # gb_info maps (in_col, func_name) -> out_col
        _, gb_info = bodo.hiframes.pd_groupby_ext.resolve_gb(
            grp_typ, args, kws, func_name, numba.core.registry.cpu_target.typing_context
        )

        # Populate gb_info_in and gb_info_out (to store in Aggregate node)
        # gb_info_in: maps in_col -> list of (func, out_col)
        # gb_info_out: maps out_col -> (in_col, func)
        gb_info_in = defaultdict(list)
        gb_info_out = {}
        agg_func = get_agg_func(self.func_ir, func_name, rhs, typemap=self.typemap)
        if not isinstance(agg_func, list):
            # agg_func is SimpleNamespace or a Python function for UDFs
            agg_funcs = [agg_func for _ in range(len(gb_info))]
        else:
            # TODO Might be possible to simplify get_agg_func by returning a flat list
            agg_funcs = list(flatten(agg_func))

        i = 0
        for (in_col, fname), out_col in gb_info.items():
            f = agg_funcs[i]
            assert fname == f.fname
            gb_info_in[in_col].append((f, out_col))
            gb_info_out[out_col] = (in_col, f)
            i += 1

        input_has_index = False
        same_index = False
        return_key = True
        # return_key is True if we return the keys from the table. In case
        # of aggregate on cumsum or other cumulative function, there is no such need.
        # same_index is True if we return the index from the table (which is the case for
        # cumulative operations not using RangeIndex)
        for funcs in gb_info_in.values():
            for func, _ in funcs:
                if func.ftype in (list_cumulative | {"shift", "transform"}):
                    input_has_index = True
                    same_index = True
                    return_key = False
                elif func.ftype in {"idxmin", "idxmax"}:
                    input_has_index = True
        if same_index and isinstance(grp_typ.df_type.index, RangeIndexType):
            same_index = False
            input_has_index = False

        df_in_vars = {
            c: self._get_dataframe_data(df_var, c, nodes)
            for c in gb_info_in.keys()
            if c is not None
        }

        in_key_arrs = [self._get_dataframe_data(df_var, c, nodes) for c in grp_typ.keys]

        out_key_vars = None
        if return_key and (grp_typ.as_index is False or out_typ.index != types.none):
            out_key_vars = []
            for k in grp_typ.keys:
                out_key_var = ir.Var(
                    lhs.scope, mk_unique_var(sanitize_varname(k)), lhs.loc
                )
                ind = df_type.columns.index(k)
                self.typemap[out_key_var.name] = df_type.data[ind]
                out_key_vars.append(out_key_var)

        if input_has_index:
            in_index_var = self._gen_array_from_index(df_var, nodes)
            df_in_vars["$_bodo_index_"] = in_index_var

        if same_index:
            out_index_var = ir.Var(lhs.scope, mk_unique_var("out_index"), lhs.loc)
            self.typemap[out_index_var.name] = self.typemap[in_index_var.name]
            if out_key_vars == None:
                out_key_vars = []
            out_key_vars.append(out_index_var)

        df_out_vars = {}
        out_colnames = (
            grp_typ.selection if isinstance(out_typ, SeriesType) else out_typ.columns
        )
        if func_name == "size":
            var = ir.Var(lhs.scope, mk_unique_var("size"), lhs.loc)
            self.typemap[var.name] = types.Array(types.int64, 1, "C")
            df_out_vars["size"] = var
        else:
            for c in out_colnames:
                # output key columns are stored in out_key_vars and shouldn't be duplicated
                if isinstance(c, tuple) and len(c) > 1 and c[1] == "":
                    if c[0] in grp_typ.keys:
                        continue
                elif c in grp_typ.keys and not c in grp_typ.selection:
                    continue
                # output column name can be a string or tuple of strings. the
                # latter case occurs when doing this:
                # df.groupby(...).agg({"A": [f1, f2]})
                # In this case, output names have 2 levels: (A, f1) and (A, f2)
                var = ir.Var(
                    lhs.scope, mk_unique_var(sanitize_varname(str(c))), lhs.loc
                )
                self.typemap[var.name] = (
                    out_typ.data
                    if isinstance(out_typ, SeriesType)
                    else out_typ.data[out_typ.columns.index(c)]
                )
                df_out_vars[c] = var

            if len(out_colnames) != len(set(out_colnames)):
                raise BodoError("aggregate with duplication in output is not allowed")

        agg_node = bodo.ir.aggregate.Aggregate(
            lhs.name,
            df_var.name,  # input dataframe var name
            grp_typ.keys,  # name of key columns
            gb_info_in,
            gb_info_out,
            out_key_vars,
            df_out_vars,
            df_in_vars,
            in_key_arrs,
            input_has_index,
            same_index,
            return_key,
            lhs.loc,
            grp_typ.dropna,
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
            nodes += compile_func_single_block(
                eval(
                    "lambda A: bodo.hiframes.pd_index_ext.init_range_index(0, len(A), 1, None)"
                ),
                (list(df_out_vars.values())[0],),
                None,
                self,
            )
            index_var = nodes[-1].target
        elif isinstance(out_typ.index, MultiIndexType):
            # gen MultiIndex init function
            arg_names = ", ".join("in{}".format(i) for i in range(len(grp_typ.keys)))
            names_tup = ", ".join("'{}'".format(k) for k in grp_typ.keys)
            func_text = "def _multi_inde_impl({}):\n".format(arg_names)
            func_text += "    return bodo.hiframes.pd_multi_index_ext.init_multi_index(({}), ({}))\n".format(
                arg_names, names_tup
            )
            loc_vars = {}
            exec(func_text, {}, loc_vars)
            _multi_inde_impl = loc_vars["_multi_inde_impl"]
            nodes += compile_func_single_block(
                _multi_inde_impl, out_key_vars, None, self
            )
            index_var = nodes[-1].target
        else:
            index_arr = out_key_vars[0]
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

        # XXX output becomes series if single output and explicitly selected
        if isinstance(out_typ, SeriesType):
            assert (
                len(grp_typ.selection) == 1
                and grp_typ.series_select
                and grp_typ.as_index
            ) or (grp_typ.as_index and func_name == "size")
            name_val = None if func_name == "size" else list(df_out_vars.keys())[0]
            name_var = ir.Var(lhs.scope, mk_unique_var("S_name"), lhs.loc)
            self.typemap[name_var.name] = (
                types.none if func_name == "size" else types.StringLiteral(name_val)
            )
            nodes.append(ir.Assign(ir.Const(name_val, lhs.loc), name_var, lhs.loc))
            return replace_func(
                self,
                eval(
                    "lambda A, I, name: bodo.hiframes.pd_series_ext.init_series(A, I, name)"
                ),
                list(df_out_vars.values()) + [index_var, name_var],
                pre_nodes=nodes,
            )

        _init_df = _gen_init_df(out_typ.columns, "index")

        # XXX the order of output variables passed should match out_typ.columns
        out_vars = []
        for c in out_typ.columns:
            is_key = False
            if isinstance(c, tuple) and len(c) > 1 and c[1] == "":
                if c[0] in grp_typ.keys:
                    is_key = True
                    c = c[0]
            elif c in grp_typ.keys:
                is_key = True
            if is_key:
                assert not grp_typ.as_index
                ind = grp_typ.keys.index(c)
                out_vars.append(out_key_vars[ind])
            else:
                out_vars.append(df_out_vars[c])

        out_vars.append(index_var)
        return nodes + compile_func_single_block(_init_df, out_vars, lhs, self)

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
            ", ".join("e{}".format(i) for i in range(len(extra_args)))
            + (", " if extra_args else "")
            + ", ".join(
                "{}=e{}".format(a, i + len(extra_args))
                for i, a in enumerate(kws.keys())
            )
        )
        udf_arg_types = [self.typemap[v.name] for v in extra_args]
        udf_kw_types = {k: self.typemap[v.name] for k, v in kws.items()}
        udf_return_type = _get_groupby_apply_udf_out_type(
            bodo.utils.typing.FunctionLiteral(func),
            grp_typ,
            udf_arg_types,
            udf_kw_types,
            self.typingctx,
        )

        extra_args += list(kws.values())
        extra_arg_names = ", ".join("e{}".format(i) for i in range(len(extra_args)))

        in_col_names = df_type.columns
        if grp_typ.explicit_select:
            in_col_names = tuple(grp_typ.selection)

        # find which columns are actually used if possible
        used_cols = _get_df_apply_used_cols(func, in_col_names)
        # avoid empty data which results in errors
        if not used_cols:
            used_cols = [df_type.columns[0]]

        n_keys = len(grp_typ.keys)
        key_names = ["k" + str(i) for i in range(n_keys)]
        col_names = ["c" + str(i) for i in range(len(used_cols))]
        key_name_args = ", ".join(key_names)
        col_name_args = ", ".join(col_names)

        func_text = (
            f"def f({key_name_args}, {col_name_args}, df_index, {extra_arg_names}):\n"
        )

        func_text += f"  in_df = bodo.hiframes.pd_dataframe_ext.init_dataframe(({col_name_args},), df_index, {gen_const_tup(used_cols)})\n"

        func_text += f"  return _bodo_groupby_apply_impl(({key_name_args},), in_df, {extra_arg_names})\n"

        loc_vars = {}
        exec(func_text, {}, loc_vars)
        f = loc_vars["f"]

        nodes = []
        key_vars = [self._get_dataframe_data(df_var, c, nodes) for c in grp_typ.keys]
        col_vars = [self._get_dataframe_data(df_var, c, nodes) for c in used_cols]
        df_index_var = self._get_dataframe_index(df_var, nodes)
        map_func = bodo.compiler.udf_jit(func)

        if extra_arg_names:
            extra_arg_names += ", "

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
            "shuffle_dataframe": bodo.hiframes.pd_groupby_ext.shuffle_dataframe,
            "reverse_shuffle": bodo.hiframes.pd_groupby_ext.reverse_shuffle,
            "delete_shuffle_info": bodo.libs.array.delete_shuffle_info,
            "dist_reduce": bodo.libs.distributed_api.dist_reduce,
            "map_func": map_func,
        }
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

        return replace_func(
            self,
            f,
            key_vars + col_vars + [df_index_var] + extra_args,
            extra_globals=glbs,
            pre_nodes=nodes,
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

        func_text = f"def _bodo_groupby_apply_impl(keys, in_df, {extra_arg_names}_is_parallel=False):\n"
        func_text += f"  if _is_parallel:\n"
        func_text += f"    in_df, keys, shuffle_info = shuffle_dataframe(in_df, keys)\n"

        # get groupby info
        func_text += f"  sort_idx, group_indices, ngroups = get_group_indices(keys, {grp_typ.dropna})\n"
        # TODO This can be done in C++ as well.
        # This will avoid returning group_indices back.
        func_text += (
            "  starts, ends = generate_slices(group_indices[sort_idx], ngroups)\n"
        )
        # sort keys and data
        for i in range(n_keys):
            func_text += f"  s_key{i} = keys[{i}][sort_idx]\n"
        is_series_in = grp_typ.series_select and len(grp_typ.selection) == 1
        func_text += "  in_data = in_df.iloc[sort_idx{}]\n".format(
            ",0" if is_series_in else ""
        )

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

        # output always has input keys (either Index or regular columns)
        for i in range(n_keys):
            func_text += f"  in_key_arrs{i} = bodo.utils.utils.alloc_type(ngroups, s_key{i}, (-1,))\n"
        for i in range(n_out_cols - n_out_keys):
            func_text += f"  arrs{i} = bodo.utils.utils.alloc_type(ngroups, _arr_typ{i+n_out_keys}, (-1,))\n"
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
        func_text += f"  for i in range(ngroups):\n"
        func_text += "    piece = in_data[starts[i]:ends[i]]\n"

        func_text += f"    out = map_func(piece, {udf_arg_names})\n"
        if isinstance(udf_return_type, (SeriesType, HeterogeneousSeriesType)):
            func_text += (
                "    out_vals = bodo.hiframes.pd_series_ext.get_series_data(out)\n"
            )
            for i in range(n_out_cols - n_out_keys):
                func_text += f"    arrs{i}[i] = bodo.utils.conversion.unbox_if_timestamp(out_vals[{i}])\n"
        else:
            func_text += (
                f"    arrs0[i] = bodo.utils.conversion.unbox_if_timestamp(out)\n"
            )
        for i in range(n_keys):
            func_text += f"    in_key_arrs{i}[i] = s_key{i}[starts[i]]\n"
        if not grp_typ.as_index:
            func_text += f"    out_index_arr[i] = n_prev_groups + i\n"

        # create output dataframe
        if grp_typ.as_index:
            index_names = ", ".join(
                f"'{v}'" if isinstance(v, str) else f"{v}" for v in grp_typ.keys
            )
        else:
            index_names = "None"
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
        func_text += f"  if _is_parallel:\n"
        func_text += f"    delete_shuffle_info(shuffle_info)\n"

        if isinstance(out_typ, DataFrameType):
            func_text += f"  return bodo.hiframes.pd_dataframe_ext.init_dataframe(({out_data},), out_index, {gen_const_tup(out_typ.columns)})\n"
        else:
            func_text += f"  return bodo.hiframes.pd_series_ext.init_series(arrs0, out_index, None)\n"
        return func_text

    def _gen_groupby_apply_acc_loop(
        self, grp_typ, udf_return_type, out_typ, udf_arg_names, n_out_cols, n_keys
    ):
        """generate groupby apply loop in cases where the UDF output is multiple rows
        and needs to be accumulated properly
        """

        sum_no = bodo.libs.distributed_api.Reduce_Type.Sum.value

        func_text = ""

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
        func_text += f"  mutated = False\n"

        # loop over groups and call UDF
        func_text += f"  for i in range(ngroups):\n"
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
        if grp_typ.as_index:
            for i in range(n_keys):
                # all rows of output get the input keys as Index. Hence, create an array
                # of key values with same length as output
                func_text += f"    in_key_arrs{i}.append(bodo.utils.utils.full_type(len(out_df), s_key{i}[starts[i]], s_key{i}))\n"
        else:
            func_text += (
                f"    in_key_arr.append(np.full(len(out_df), n_prev_groups + i))\n"
            )
        if isinstance(udf_return_type, SeriesType):
            func_text += f"    arrs0.append(bodo.hiframes.pd_series_ext.get_series_data(out_df))\n"
        else:
            for i in range(n_out_cols):
                func_text += f"    arrs{i}.append(bodo.hiframes.pd_dataframe_ext.get_dataframe_data(out_df, {i}))\n"
        for i in range(n_out_cols):
            func_text += f"  out_arr{i} = bodo.libs.array_kernels.concat(arrs{i})\n"
        if grp_typ.as_index:
            for i in range(n_keys):
                func_text += f"  out_key_arr{i} = bodo.libs.array_kernels.concat(in_key_arrs{i})\n"

            out_key_arr_names = ", ".join(f"out_key_arr{i}" for i in range(n_keys))
        else:
            func_text += f"  out_key_arr = bodo.libs.array_kernels.concat(in_key_arr)\n"
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
        func_text += f"  out_idx_arr_all = bodo.libs.array_kernels.concat(arrs_index)\n"
        func_text += f"  out_index = bodo.hiframes.pd_multi_index_ext.init_multi_index(({out_key_arr_names}, out_idx_arr_all), ({index_names},), None)\n"

        # reorder output to match input if UDF Index is same as input
        func_text += "  if _is_parallel:\n"
        # synchronize since some ranks may avoid mutation due to corner cases
        func_text += (
            f"    mutated = bool(dist_reduce(int(mutated), np.int32({sum_no})))\n"
        )
        func_text += "  if not mutated:\n"
        func_text += f"    rev_idx = sort_idx.argsort()\n"
        func_text += f"    out_index = out_index[rev_idx]\n"
        for i in range(n_out_cols):
            func_text += f"    out_arr{i} = out_arr{i}[rev_idx]\n"
        func_text += f"    if _is_parallel:\n"
        func_text += f"      out_index = reverse_shuffle(out_index, shuffle_info)\n"
        for i in range(n_out_cols):
            func_text += (
                f"      out_arr{i} = reverse_shuffle(out_arr{i}, shuffle_info)\n"
            )

        # parallel shuffle clean up
        func_text += f"  if _is_parallel:\n"
        func_text += f"    delete_shuffle_info(shuffle_info)\n"

        out_data = ", ".join("out_arr{}".format(i) for i in range(n_out_cols))
        if isinstance(out_typ, SeriesType):
            # some ranks may have empty data after shuffle (ngroups == 0), so call the
            # UDF with empty data to get the name of the output Series
            func_text += f"  out_name = out_df.name if ngroups else map_func(in_data[0:0], {udf_arg_names}).name\n"
            func_text += f"  return bodo.hiframes.pd_series_ext.init_series(out_arr0, out_index, out_name)\n"
        else:
            func_text += f"  return bodo.hiframes.pd_dataframe_ext.init_dataframe(({out_data},), out_index, {gen_const_tup(out_typ.columns)})\n"

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

    def _run_call_pivot_table(self, assign, lhs, rhs):
        df_var, values, index, columns, aggfunc, _pivot_values = rhs.args
        func_name = self.typemap[aggfunc.name].literal_value
        values = self.typemap[values.name].literal_value
        index = self.typemap[index.name].literal_value
        columns = self.typemap[columns.name].literal_value
        pivot_values = self.typemap[_pivot_values.name].meta
        df_type = self.typemap[df_var.name]
        out_typ = self.typemap[lhs.name]

        nodes = []
        df_in_vars = {values: self._get_dataframe_data(df_var, values, nodes)}

        df_out_vars = {
            col: ir.Var(lhs.scope, mk_unique_var(sanitize_varname(col)), lhs.loc)
            for col in pivot_values
        }
        for v in df_out_vars.values():
            self.typemap[v.name] = out_typ.data[0]

        pivot_arr = self._get_dataframe_data(df_var, columns, nodes)
        index_arr = self._get_dataframe_data(df_var, index, nodes)
        agg_func = get_agg_func(self.func_ir, func_name, rhs, typemap=self.typemap)
        gb_info_in = {}
        for in_col in values:
            # TODO: multiple functions
            assert not isinstance(agg_func, list)
            gb_info_in[in_col] = [(agg_func, list(pivot_values))]
        gb_info_out = {col: (values, agg_func) for col in pivot_values}

        out_key_vars = []
        out_index_var = ir.Var(
            lhs.scope, mk_unique_var(sanitize_varname(index)), lhs.loc
        )
        ind = df_type.columns.index(index)
        self.typemap[out_index_var.name] = df_type.data[ind]
        out_key_vars.append(out_index_var)

        input_has_index = False
        same_index = False
        return_key = True
        dropna = True
        agg_node = bodo.ir.aggregate.Aggregate(
            lhs.name,
            df_var.name,
            [index],
            gb_info_in,
            gb_info_out,
            out_key_vars,
            df_out_vars,
            df_in_vars,
            [index_arr],
            input_has_index,
            same_index,
            return_key,
            lhs.loc,
            dropna,
            pivot_arr,
            pivot_values,
        )
        nodes.append(agg_node)

        nodes += compile_func_single_block(
            eval("lambda A: bodo.utils.conversion.index_from_array(A, _index_name)"),
            (out_index_var,),
            None,
            self,
            extra_globals={"_index_name": index},
        )
        index_var = nodes[-1].target

        _init_df = _gen_init_df(out_typ.columns, "index")

        # XXX the order of output variables passed should match out_typ.columns
        out_vars = [df_out_vars[c] for c in out_typ.columns]
        out_vars.append(index_var)

        return nodes + compile_func_single_block(_init_df, out_vars, lhs, self)

    def _run_call_crosstab(self, assign, lhs, rhs):
        index, columns, _pivot_values = rhs.args
        pivot_values = self.typemap[_pivot_values.name].meta
        out_typ = self.typemap[lhs.name]

        index_typ = out_typ.index
        index_name_typ = index_typ.name_typ
        index_name = index_name_typ.literal_value

        nodes = []
        if isinstance(self.typemap[index.name], SeriesType):
            nodes += compile_func_single_block(
                eval("lambda S: bodo.hiframes.pd_series_ext.get_series_data(S)"),
                (index,),
                None,
                self,
            )
            index = nodes[-1].target

        if isinstance(self.typemap[columns.name], SeriesType):
            nodes += compile_func_single_block(
                eval("lambda S: bodo.hiframes.pd_series_ext.get_series_data(S)"),
                (columns,),
                None,
                self,
            )
            columns = nodes[-1].target

        # The index is added.
        # TODO: Add the name of the index to the construction. Pandas does it.
        out_key_vars = []
        out_index_var = ir.Var(lhs.scope, mk_unique_var("index"), lhs.loc)
        self.typemap[out_index_var.name] = self.typemap[index.name]
        out_key_vars.append(out_index_var)

        df_in_vars = {}

        df_out_vars = {
            col: ir.Var(lhs.scope, mk_unique_var(sanitize_varname(col)), lhs.loc)
            for col in pivot_values
        }
        for i, v in enumerate(df_out_vars.values()):
            self.typemap[v.name] = out_typ.data[i]

        pivot_arr = columns
        agg_func = get_agg_func(self.func_ir, "count", rhs, typemap=self.typemap)
        gb_info_in = {None: [(agg_func, list(pivot_values))]}
        gb_info_out = {col: (None, agg_func) for col in pivot_values}

        # TODO: make out_key_var an index column
        input_has_index = False
        same_index = False
        return_key = True
        dropna = True
        agg_node = bodo.ir.aggregate.Aggregate(
            lhs.name,
            "crosstab",
            ["index"],
            gb_info_in,
            gb_info_out,
            out_key_vars,
            df_out_vars,
            df_in_vars,
            [index],
            input_has_index,
            same_index,
            return_key,
            lhs.loc,
            dropna,
            pivot_arr,
            pivot_values,
            True,
        )
        nodes.append(agg_node)

        _init_df = _gen_init_df(out_typ.columns, "index")

        # XXX the order of output variables passed should match out_typ.columns
        out_vars = [df_out_vars[c] for c in out_typ.columns]

        nodes += compile_func_single_block(
            eval("lambda A: bodo.utils.conversion.index_from_array(A, _index_name)"),
            (out_index_var,),
            None,
            self,
            extra_globals={"_index_name": index_name},
        )
        out_index = nodes[-1].target

        out_vars.append(out_index)
        return nodes + compile_func_single_block(_init_df, out_vars, lhs, self)

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
            assert seq_info is not None
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
        df_typ = self.typemap[df_var.name]
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

    def _get_index_name(self, dt_var, nodes):
        var_def = guard(get_definition, self.func_ir, dt_var)
        call_def = guard(find_callname, self.func_ir, var_def, self.typemap)
        if (
            call_def
            in (
                ("init_datetime_index", "bodo.hiframes.pd_index_ext"),
                ("init_timedelta_index", "bodo.hiframes.pd_index_ext"),
                ("init_string_index", "bodo.hiframes.pd_index_ext"),
                ("init_numeric_index", "bodo.hiframes.pd_index_ext"),
                ("init_categorical_index", "bodo.hiframes.pd_index_ext"),
                ("init_heter_index", "bodo.hiframes.pd_index_ext"),
            )
            and len(var_def.args) == 2
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


def _gen_init_df(columns, index=None):
    n_cols = len(columns)
    data_args = ", ".join("data{}".format(i) for i in range(n_cols))
    args = data_args

    if index is None:
        assert n_cols > 0
        index = "bodo.hiframes.pd_index_ext.init_range_index(0, len(data0), 1, None)"
    else:
        args += ", " + index

    col_var = gen_const_tup(columns)
    func_text = "def _init_df({}):\n".format(args)
    func_text += "  return bodo.hiframes.pd_dataframe_ext.init_dataframe(({},), {}, {})\n".format(
        data_args, index, col_var
    )
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    _init_df = loc_vars["_init_df"]

    return _init_df


def _get_df_apply_used_cols(func, columns):
    """find which df columns are actually used in UDF 'func' inside df.apply(func) if
    possible (has to be conservative and assume all columns are used when it cannot
    analyze the IR properly)
    """
    lambda_ir = numba.core.compiler.run_frontend(func)

    used_cols = []
    l_topo_order = find_topo_order(lambda_ir.blocks)
    first_stmt = lambda_ir.blocks[l_topo_order[0]].body[0]
    assert isinstance(first_stmt, ir.Assign) and isinstance(first_stmt.value, ir.Arg)
    arg_var = first_stmt.target
    use_all_cols = False
    for bl in lambda_ir.blocks.values():
        for stmt in bl.body:
            vnames = [v.name for v in stmt.list_vars()]
            if arg_var.name in vnames:
                if is_assign(stmt) and isinstance(stmt.value, ir.Arg):
                    continue
                # match x.C column access
                elif (
                    is_assign(stmt)
                    and is_expr(stmt.value, "getattr")
                    and stmt.value.value.name == arg_var.name
                    and stmt.value.attr in columns
                ):
                    used_cols.append(stmt.value.attr)
                # match x["C"] column access
                elif (
                    is_assign(stmt)
                    and is_expr(stmt.value, "getitem")
                    and stmt.value.value.name == arg_var.name
                    and guard(find_const, lambda_ir, stmt.value.index) in columns
                ):
                    used_cols.append(guard(find_const, lambda_ir, stmt.value.index))
                else:
                    # argument is used in some other form
                    # be conservative and use all cols
                    use_all_cols = True
                    used_cols = columns
                    break

        if use_all_cols:
            break

    # remove duplicates with set() since a column can be used multiple times
    # keep the order the same as original columns to avoid errors with int getitem on
    # rows
    used_cols = [c for (_, c) in sorted((columns.index(v), v) for v in set(used_cols))]
    return used_cols
