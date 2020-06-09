# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
converts data frame operations to Series and Array operations
as much as possible to provide implementation and enable optimization.
Creates specialized IR nodes for complex operations like Join.
"""
import operator
from collections import namedtuple
import numpy as np
import pandas as pd
import warnings

import numba
from numba.core import ir, ir_utils, types
from bodo.utils.typing import is_overload_none
from numba.core.ir_utils import (
    replace_arg_nodes,
    compile_to_numba_ir,
    find_topo_order,
    get_definition,
    guard,
    find_callname,
    find_const,
    mk_unique_var,
    dprint_func_ir,
    build_definitions,
    find_build_sequence,
    GuardException,
    compute_cfg_from_blocks,
)
from numba.core.inline_closurecall import inline_closure_call

import bodo
from bodo.utils.typing import list_cumulative
from bodo import hiframes
from bodo.utils.utils import (
    debug_prints,
    is_array_typ,
    is_assign,
    sanitize_varname,
    get_getsetitem_index_var,
)
from bodo.hiframes.dataframe_indexing import (
    DataFrameType,
    DataFrameLocType,
    DataFrameILocType,
    DataFrameIatType,
)
from bodo.hiframes.pd_series_ext import SeriesType
import bodo.hiframes.dataframe_impl  # side effect: install DataFrame overloads
import bodo.hiframes.pd_groupby_ext
from bodo.hiframes.pd_groupby_ext import DataFrameGroupByType
import bodo.hiframes.pd_rolling_ext
from bodo.hiframes.pd_rolling_ext import RollingType
from bodo.ir.aggregate import get_agg_func
from bodo.utils.transform import (
    compile_func_single_block,
    update_locs,
    get_const_value,
    get_call_expr_arg,
    gen_const_tup,
    ReplaceFunc,
    replace_func,
)
from bodo.utils.utils import gen_getitem
from bodo.libs.str_arr_ext import (
    string_array_type,
    get_utf8_size,
    pre_alloc_string_array,
)
from bodo.hiframes.split_impl import string_array_split_view_type
from bodo.libs.list_str_arr_ext import list_string_array_type
from bodo.libs.bool_arr_ext import BooleanArrayType
from bodo.hiframes.pd_index_ext import RangeIndexType
from bodo.hiframes.pd_multi_index_ext import MultiIndexType
from bodo.utils.typing import (
    get_index_data_arr_types,
    is_overload_none,
    is_overload_constant_str,
    get_overload_const_func,
    get_overload_const_str,
    BodoError,
    is_overload_constant_dict,
    get_overload_constant_dict,
    is_overload_constant_list,
    get_overload_const_list,
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

    def run(self):
        dprint_func_ir(self.func_ir, "starting dataframe pass")
        ir_utils.remove_dels(self.func_ir.blocks)
        blocks = self.func_ir.blocks
        # topo_order necessary so DataFrame data replacement optimization can
        # be performed in one pass
        topo_order = find_topo_order(blocks)
        work_list = list((l, blocks[l]) for l in reversed(topo_order))
        while work_list:
            label, block = work_list.pop()

            new_body = []
            replaced = False
            for i, inst in enumerate(block.body):
                out_nodes = [inst]
                self.curr_loc = inst.loc

                try:
                    if isinstance(inst, ir.Assign):
                        self.func_ir._definitions[inst.target.name].remove(inst.value)
                        out_nodes = self._run_assign(inst)
                    elif isinstance(inst, (ir.SetItem, ir.StaticSetItem)):
                        out_nodes = self._run_setitem(inst)
                except BodoError as e:
                    raise BodoError(inst.loc.strformat() + "\n" + str(e))

                if isinstance(out_nodes, list):
                    # TODO: process these nodes
                    new_body.extend(out_nodes)
                    self._update_definitions(out_nodes)
                if isinstance(out_nodes, ReplaceFunc):
                    rp_func = out_nodes
                    if rp_func.pre_nodes is not None:
                        new_body.extend(rp_func.pre_nodes)
                        self._update_definitions(rp_func.pre_nodes)
                    # replace inst.value to a call with target args
                    # as expected by inline_closure_call
                    inst.value = ir.Expr.call(
                        ir.Var(block.scope, "dummy", inst.loc),
                        rp_func.args,
                        (),
                        inst.loc,
                    )
                    # replace "target" of Setitem nodes since inline_closure_call
                    # assumes an assignment and sets "target" to return value
                    if isinstance(inst, (ir.SetItem, ir.StaticSetItem)):
                        inst.target = ir.Var(block.scope, "dummy", inst.loc)
                    block.body = new_body + block.body[i:]
                    callee_blocks, _ = inline_closure_call(
                        self.func_ir,
                        rp_func.glbls,
                        block,
                        len(new_body),
                        rp_func.func,
                        self.typingctx,
                        rp_func.arg_types,
                        self.typemap,
                        self.calltypes,
                    )
                    # add blocks in reversed topo order to enable dead branch
                    # pruning (merge example)
                    # TODO: fix inline_closure_call()
                    topo_order = find_topo_order(self.func_ir.blocks)
                    for c_label in reversed(topo_order):
                        if c_label in callee_blocks:
                            c_block = callee_blocks[c_label]
                            # update loc info
                            c_block.loc = self.curr_loc
                            update_locs(c_block.body, self.curr_loc)
                            # include the new block created after callee used
                            # to split the original block
                            # find it using jumps out of callee (returns
                            # originally) but include only once
                            if isinstance(c_block.body[-1], ir.Jump):
                                target_label = c_block.body[-1].target
                                if (
                                    target_label not in callee_blocks
                                    and target_label not in work_list
                                ):
                                    work_list.append(
                                        (
                                            target_label,
                                            self.func_ir.blocks[target_label],
                                        )
                                    )
                            work_list.append((c_label, c_block))
                    replaced = True
                    break

            if not replaced:
                blocks[label].body = new_body

        self.func_ir.blocks = ir_utils.simplify_CFG(self.func_ir.blocks)
        # can't call remove dead since Series transform is not done yet and
        # aliases like S.values are not known, see test_1D_Var_alloc3
        # TODO: merge dist pass and series pass
        # while ir_utils.remove_dead(self.func_ir.blocks, self.func_ir.arg_names,
        #                            self.func_ir, self.typemap):
        #     pass

        self.func_ir._definitions = build_definitions(self.func_ir.blocks)
        dprint_func_ir(self.func_ir, "after dataframe pass")
        return

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

        return [assign]

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

        nodes.append(assign)
        return nodes

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

        return nodes + [inst]

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
                lambda arr, index, name: bodo.hiframes.pd_series_ext.init_series(
                    arr, index, name
                ),
                [arr, index, name],
                pre_nodes=nodes,
            )

        # level selection in multi-level df
        if (
            isinstance(rhs_type, DataFrameType)
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

        return [assign]

    def _run_binop(self, assign, rhs):

        arg1, arg2 = rhs.lhs, rhs.rhs
        typ1, typ2 = self.typemap[arg1.name], self.typemap[arg2.name]
        if not (isinstance(typ1, DataFrameType) or isinstance(typ2, DataFrameType)):
            return [assign]

        if rhs.fn in bodo.hiframes.pd_series_ext.series_binary_ops:
            overload_func = bodo.hiframes.dataframe_impl.create_binary_op_overload(
                rhs.fn
            )
            impl = overload_func(typ1, typ2)
            return replace_func(self, impl, [arg1, arg2])

        if rhs.fn in bodo.hiframes.pd_series_ext.series_inplace_binary_ops:
            overload_func = bodo.hiframes.dataframe_impl.create_inplace_binary_op_overload(
                rhs.fn
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

        return [assign]

    def _run_call(self, assign, lhs, rhs):
        fdef = guard(find_callname, self.func_ir, rhs, self.typemap)
        if fdef is None:
            from numba.stencils.stencil import StencilFunc

            # could be make_function from list comprehension which is ok
            func_def = guard(get_definition, self.func_ir, rhs.func)
            if isinstance(func_def, ir.Expr) and func_def.op == "make_function":
                return [assign]
            # ignore objmode block calls
            if isinstance(func_def, ir.Const) and isinstance(
                func_def.value, numba.core.dispatcher.ObjModeLiftedWith
            ):
                return [assign]
            if isinstance(func_def, ir.Global) and isinstance(
                func_def.value, StencilFunc
            ):
                return [assign]
            warnings.warn("function call couldn't be found for dataframe analysis")
            return [assign]
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

        if fdef == ("join_dummy", "bodo.hiframes.pd_dataframe_ext"):
            return self._run_call_join(assign, lhs, rhs)

        if isinstance(func_mod, ir.Var) and isinstance(
            self.typemap[func_mod.name], DataFrameType
        ):
            return self._run_call_dataframe(
                assign, assign.target, rhs, func_mod, func_name
            )

        if fdef == ("add_consts_to_type", "bodo.utils.typing"):
            assign.value = rhs.args[0]
            return [assign]

        if isinstance(func_mod, ir.Var) and isinstance(
            self.typemap[func_mod.name], DataFrameGroupByType
        ):
            return self._run_call_groupby(
                assign, assign.target, rhs, func_mod, func_name
            )

        if isinstance(func_mod, ir.Var) and isinstance(
            self.typemap[func_mod.name], RollingType
        ):
            return self._run_call_rolling(
                assign, assign.target, rhs, func_mod, func_name
            )

        if fdef == ("pivot_table_dummy", "bodo.hiframes.pd_groupby_ext"):
            return self._run_call_pivot_table(assign, lhs, rhs)

        if fdef == ("crosstab_dummy", "bodo.hiframes.pd_groupby_ext"):
            return self._run_call_crosstab(assign, lhs, rhs)

        if fdef == ("concat_dummy", "bodo.hiframes.pd_dataframe_ext") and isinstance(
            self.typemap[lhs.name], DataFrameType
        ):
            return self._run_call_concat(assign, lhs, rhs)

        if fdef == ("sort_values_dummy", "bodo.hiframes.pd_dataframe_ext"):
            return self._run_call_df_sort_values(assign, lhs, rhs)

        if fdef == ("itertuples_dummy", "bodo.hiframes.pd_dataframe_ext"):
            return self._run_call_df_itertuples(assign, lhs, rhs)

        if fdef == ("fillna_dummy", "bodo.hiframes.pd_dataframe_ext"):
            return self._run_call_df_fillna(assign, lhs, rhs)

        if fdef == ("dropna_dummy", "bodo.hiframes.pd_dataframe_ext"):
            return self._run_call_df_dropna(assign, lhs, rhs)

        if fdef == ("reset_index_dummy", "bodo.hiframes.pd_dataframe_ext"):
            return self._run_call_reset_index(assign, lhs, rhs)

        if fdef == ("query_dummy", "bodo.hiframes.pd_dataframe_ext"):
            return self._run_call_query(assign, lhs, rhs)

        return [assign]

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

        if func_name == "pivot_table":
            rhs.args.insert(0, df_var)
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}
            impl = bodo.hiframes.pd_dataframe_ext.pivot_table_overload(
                *arg_typs, **kw_typs
            )
            stub = (
                lambda df, values=None, index=None, columns=None, aggfunc="mean", fill_value=None, margins=False, dropna=True, margins_name="All", _pivot_values=None: None
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
        """generate IR nodes for df.apply() with UDFs
        """
        df_typ = self.typemap[df_var.name]
        # get apply function
        kws = dict(rhs.kws)
        func_var = get_call_expr_arg("apply", rhs.args, kws, 0, "func")
        func = get_overload_const_func(self.typemap[func_var.name])

        # find which columns are actually used if possible
        used_cols = _get_df_apply_used_cols(func, df_typ.columns)

        Row = namedtuple(sanitize_varname(df_var.name), used_cols)
        # prange func to inline
        col_name_args = ", ".join(["c" + str(i) for i in range(len(used_cols))])
        row_args = ", ".join(
            [
                "bodo.utils.conversion.box_if_dt64(c{}[i])".format(i)
                for i in range(len(used_cols))
            ]
        )

        func_text = "def f({}, df_index):\n".format(col_name_args)

        if self.typemap[lhs.name].data == string_array_type:
            func_text += "  numba.parfors.parfor.init_prange()\n"
            func_text += "  n = len(c0)\n"
            func_text += "  n_chars = 0\n"
            func_text += "  for i in numba.parfors.parfor.internal_prange(n):\n"
            func_text += "     row = Row({})\n".format(row_args)
            func_text += "     n_chars += get_utf8_size(map_func(row))\n"
            func_text += "  S = pre_alloc_string_array(n, n_chars)\n"
        else:
            func_text += "  numba.parfors.parfor.init_prange()\n"
            func_text += "  n = len(c0)\n"
            func_text += "  S = bodo.utils.utils.alloc_type(n, _arr_typ)\n"
        func_text += "  for i in numba.parfors.parfor.internal_prange(n):\n"
        func_text += "     row = Row({})\n".format(row_args)
        func_text += "     S[i] = map_func(row)\n"
        func_text += (
            "  return bodo.hiframes.pd_series_ext.init_series(S, df_index, None)\n"
        )

        loc_vars = {}
        exec(func_text, {}, loc_vars)
        f = loc_vars["f"]

        arr_typ = self.typemap[lhs.name].data
        nodes = []
        col_vars = [self._get_dataframe_data(df_var, c, nodes) for c in used_cols]
        df_index_var = self._get_dataframe_index(df_var, nodes)

        # using Bodo's sequential/inline pipeline for the UDF to make sure nested calls
        # are inlined and not distributed. Otherwise, generated barriers cause hangs
        # see: test_df_apply_func_case2
        parallel = {
            "comprehension": True,
            "setitem": False,
            "reduction": True,
            "numpy": True,
            "stencil": True,
            "fusion": True,
        }
        map_func = numba.njit(
            func, parallel=parallel, pipeline_class=bodo.compiler.BodoCompilerSeqInline
        )

        return replace_func(
            self,
            f,
            col_vars + [df_index_var],
            extra_globals={
                "numba": numba,
                "np": np,
                "Row": Row,
                "bodo": bodo,
                "_arr_typ": arr_typ,
                "get_utf8_size": get_utf8_size,
                "pre_alloc_string_array": pre_alloc_string_array,
                "map_func": map_func,
            },
            pre_nodes=nodes,
        )

    def _run_call_df_sort_values(self, assign, lhs, rhs):
        df_var, by_var, ascending_var, inplace_var, na_position_var = rhs.args
        df_typ = self.typemap[df_var.name]
        inplace = guard(find_const, self.func_ir, inplace_var)
        na_position = guard(find_const, self.func_ir, na_position_var)

        # find key array for sort ('by' arg)
        key_names = self._get_const_or_list(by_var)
        set_possible_keys = set(df_typ.columns)
        index_is_key = False
        index_name = "unset"
        if not is_overload_none(df_typ.index.name_typ):
            index_name = df_typ.index.name_typ.literal_value
            set_possible_keys.add(index_name)
            if index_name in key_names:
                index_is_key = True
        if "$_bodo_index_" in key_names:
            index_is_key = True
            index_name = "$_bodo_index_"
            set_possible_keys.add(index_name)
        ascending_list = self._get_list_value_spec_length(
            ascending_var,
            len(key_names),
            err_msg="ascending should be bool or a list of bool of the number of keys",
        )
        if not all(k in set_possible_keys for k in key_names):
            raise ValueError("invalid sort keys {}".format(key_names))

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
                out_var = ir.Var(lhs.scope, mk_unique_var(k), lhs.loc)
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

    def _gen_array_from_index(self, df_var, nodes):
        def _get_index(df):  # pragma: no cover
            return bodo.utils.conversion.index_to_array(
                bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)
            )

        nodes += compile_func_single_block(_get_index, (df_var,), None, self)
        return nodes[-1].target

    def _gen_index_from_array(self, arr_var, name_var, nodes):
        def _get_index(arr, name):  # pragma: no cover
            return bodo.utils.conversion.index_from_array(arr, name)

        nodes += compile_func_single_block(_get_index, (arr_var, name_var), None, self)
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
        func_text += "  return bodo.hiframes.dataframe_impl.get_itertuples({}, {})\n".format(
            name_consts, col_name_args
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
            func_text += "  {} = bodo.hiframes.pd_series_ext.init_series({}, {})\n".format(
                d + "_S", d, ind
            )
            func_text += "  {} = {}.{}()\n".format(d + "_O", d + "_S", func_name)
        func_text += "  data = np.array(({},))\n".format(
            ", ".join(d + "_O" for d in data_args)
        )
        func_text += "  index = bodo.libs.str_arr_ext.str_arr_from_sequence(({},))\n".format(
            ", ".join("'{}'".format(c) for c in df_typ.columns)
        )
        func_text += "  return bodo.hiframes.pd_series_ext.init_series(data, index)\n"
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        _mean_impl = loc_vars["_mean_impl"]

        nodes = []
        col_vars = [self._get_dataframe_data(df_var, c, nodes) for c in df_typ.columns]
        return replace_func(self, _mean_impl, col_vars, pre_nodes=nodes)

    def _run_call_df_fillna(self, assign, lhs, rhs):
        df_var = rhs.args[0]
        value = rhs.args[1]
        inplace_var = rhs.args[2]
        inplace = guard(find_const, self.func_ir, inplace_var)
        df_typ = self.typemap[df_var.name]

        # impl: for each column, convert data to series, call S.fillna(), get
        # output data and create a new dataframe
        n_cols = len(df_typ.columns)
        data_args = tuple("data{}".format(i) for i in range(n_cols))

        col_var = gen_const_tup(df_typ.columns)
        func_text = "def _fillna_impl({}, df_index, val):\n".format(
            ", ".join(data_args)
        )
        for d in data_args:
            func_text += "  ind_{0} = bodo.hiframes.pd_index_ext.init_range_index(0, len({0}), 1, None)\n".format(
                d
            )
            func_text += "  {0} = bodo.hiframes.pd_series_ext.init_series({1}, ind_{1})\n".format(
                d + "_S", d
            )
            if not inplace:
                func_text += "  {} = {}.fillna(val)\n".format(d + "_S", d + "_S")
                func_text += "  {} = bodo.hiframes.pd_series_ext.get_series_data({})\n".format(
                    d + "_O", d + "_S"
                )
            else:
                func_text += "  {}.fillna(val, inplace=True)\n".format(d + "_S")

        if not inplace:
            func_text += "  return bodo.hiframes.pd_dataframe_ext.init_dataframe(({},), df_index, {})\n".format(
                ", ".join(d + "_O" for d in data_args), col_var
            )
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        _fillna_impl = loc_vars["_fillna_impl"]

        nodes = []
        col_vars = [self._get_dataframe_data(df_var, c, nodes) for c in df_typ.columns]
        index_arg = self._get_dataframe_index(df_var, nodes)
        args = col_vars + [index_arg, value]
        return nodes + compile_func_single_block(_fillna_impl, args, lhs, self)

    def _run_call_df_dropna(self, assign, lhs, rhs):
        # TODO: refactor, support/test all array types
        df_var = rhs.args[0]
        df_typ = self.typemap[df_var.name]

        nodes = []
        col_vars = [self._get_dataframe_data(df_var, c, nodes) for c in df_typ.columns]

        index_arr_types = get_index_data_arr_types(df_typ.index)
        n_ind_arrs = len(index_arr_types)
        arr_types = df_typ.data + index_arr_types
        arg_names = ["data" + str(i) for i in range(len(arr_types))]
        out_names = ["out" + str(i) for i in range(len(arr_types))]

        str_colnames = [
            arg_names[i] for i, t in enumerate(arr_types) if t == string_array_type
        ]
        list_str_colnames = [
            arg_names[i] for i, t in enumerate(arr_types) if t == list_string_array_type
        ]
        split_view_colnames = [
            arg_names[i]
            for i, t in enumerate(arr_types)
            if t == string_array_split_view_type
        ]

        # Pandas ignores NAs in index arrays
        isna_calls = [
            "bodo.libs.array_kernels.isna({}, i)".format(v)
            for v in arg_names[:-n_ind_arrs]
        ]

        func_text = "def _dropna_imp({}):\n".format(", ".join(arg_names))
        # find new length, and number of characters etc. if necessary
        func_text += "  old_len = len({})\n".format(arg_names[0])
        func_text += "  new_len = 0\n"
        for c in str_colnames:
            func_text += "  num_chars_{} = 0\n".format(c)
        for c in list_str_colnames:
            func_text += "  num_lists_{} = 0\n".format(c)
            func_text += "  num_chars_{} = 0\n".format(c)
        func_text += "  for i in numba.parfors.parfor.internal_prange(old_len):\n"
        func_text += "    if not ({}):\n".format(" or ".join(isna_calls))
        func_text += "      new_len += 1\n"
        for c in str_colnames:
            func_text += "      num_chars_{0} += get_utf8_size({0}[i])\n".format(c)
        for c in list_str_colnames:
            func_text += "      v_{0} = {0}[i]\n".format(c)
            func_text += "      num_lists_{0} += len(v_{0})\n".format(c)
            func_text += "      for s_{0} in v_{0}:\n".format(c)
            func_text += "        num_chars_{0} += get_utf8_size(s_{0})\n".format(c)
        # allocate new arrays
        func_text += "  curr_ind = 0\n"
        for v, out in zip(arg_names, out_names):
            if v in str_colnames:
                func_text += "  {} = bodo.libs.str_arr_ext.pre_alloc_string_array(new_len, num_chars_{})\n".format(
                    out, v
                )
            elif v in list_str_colnames:
                func_text += "  {0} = bodo.libs.list_str_arr_ext.pre_alloc_list_string_array(new_len, num_lists_{1}, num_chars_{1})\n".format(
                    out, c
                )
            elif v in split_view_colnames:
                # TODO support dropna() for split view
                func_text += "  {} = {}\n".format(out, v)
            else:
                func_text += "  {} = np.empty(new_len, {}.dtype)\n".format(out, v)
            if v in list_str_colnames:
                func_text += "  index_offsets_{} = bodo.libs.list_str_arr_ext.get_index_offset_ptr({})\n".format(
                    c, out
                )
                func_text += "  data_offsets_{} = bodo.libs.list_str_arr_ext.get_data_offset_ptr({})\n".format(
                    c, out
                )
                func_text += "  curr_s_offset_{} = 0\n".format(c)
                func_text += "  curr_d_offset_{} = 0\n".format(c)
        # fill new array
        func_text += "  for ii in numba.parfors.parfor.internal_prange(old_len):\n"
        func_text += "    if not ({}):\n".format(
            " or ".join(
                [
                    "bodo.libs.array_kernels.isna({}, ii)".format(v)
                    for v in arg_names[:-n_ind_arrs]
                ]
            )
        )
        for v, out in zip(arg_names, out_names):
            if v in split_view_colnames:
                continue
            if v in list_str_colnames:
                func_text += "      l_start_offset = {0}._index_offsets[ii]\n".format(v)
                func_text += "      l_end_offset = {0}._index_offsets[ii + 1]\n".format(
                    v
                )
                func_text += "      n_str = l_end_offset - l_start_offset\n"
                func_text += "      str_ind = 0\n"
                func_text += "      for jj in range(l_start_offset, l_end_offset):\n"
                func_text += "          data_offsets_{0}[curr_s_offset_{0} + str_ind] = curr_d_offset_{0}\n".format(
                    c
                )
                func_text += "          n_char = {0}._data_offsets[jj + 1] - {0}._data_offsets[jj]\n".format(
                    v
                )
                func_text += (
                    "          in_ptr = bodo.hiframes.split_impl.get_c_arr_ptr(\n"
                )
                func_text += "              {0}._data, {0}._data_offsets[jj]\n".format(
                    v
                )
                func_text += "          )\n"
                func_text += (
                    "          out_ptr = bodo.hiframes.split_impl.get_c_arr_ptr(\n"
                )
                func_text += "              {}._data, curr_d_offset_{}\n".format(out, c)
                func_text += "          )\n"
                func_text += "          bodo.libs.str_arr_ext._memcpy(out_ptr, in_ptr, n_char, 1)\n"
                func_text += "          curr_d_offset_{0} += n_char\n".format(c)
                func_text += "          str_ind += 1\n"
                func_text += "          index_offsets_{0}[curr_ind] = curr_s_offset_{0}\n".format(
                    c
                )
                func_text += "      curr_s_offset_{0} += n_str\n".format(c)
                continue
            func_text += "      {}[curr_ind] = {}[ii]\n".format(out, v)
        func_text += "      curr_ind += 1\n"
        if v in list_str_colnames:
            func_text += "  index_offsets_{0}[new_len] = curr_s_offset_{0}\n".format(c)
            func_text += "  data_offsets_{0}[curr_s_offset_{0}] = curr_d_offset_{0}\n".format(
                c
            )

        col_var = gen_const_tup(df_typ.columns)
        # TODO: support MultiIndex
        func_text += "  index = bodo.utils.conversion.index_from_array({})\n".format(
            out_names[-1]
        )
        func_text += "  return bodo.hiframes.pd_dataframe_ext.init_dataframe(({},), index, {})\n".format(
            ", ".join(out_names[:-n_ind_arrs]), col_var
        )

        loc_vars = {}
        exec(
            func_text, {"get_utf8_size": bodo.libs.str_arr_ext.get_utf8_size}, loc_vars
        )
        _dropna_imp = loc_vars["_dropna_imp"]

        in_index_var = self._gen_array_from_index(df_var, nodes)
        args = col_vars + [in_index_var]

        return replace_func(self, _dropna_imp, args, pre_nodes=nodes)

    def _run_call_reset_index(self, assign, lhs, rhs):
        # TODO: reflection
        df_var = rhs.args[0]
        drop = guard(find_const, self.func_ir, rhs.args[1])
        inplace = guard(find_const, self.func_ir, rhs.args[2])
        df_typ = self.typemap[df_var.name]
        out_df_typ = self.typemap[lhs.name]
        n_ind = len(out_df_typ.columns) - len(df_typ.columns)
        assert drop or n_ind != 0  # there are index columns when not dropping index

        # impl: for each column, copy data and create a new dataframe
        n_cols = len(out_df_typ.columns)
        data_args = ["data{}".format(i) for i in range(n_cols)]

        col_var = gen_const_tup(out_df_typ.columns)
        func_text = "def _reset_index_impl({}):\n".format(", ".join(data_args))
        for i, d in enumerate(data_args):
            if not inplace and i >= n_ind:
                func_text += "  {} = {}.copy()\n".format(d, d)
        func_text += "  index = bodo.hiframes.pd_index_ext.init_range_index(0, len({}), 1, None)\n".format(
            data_args[0]
        )
        func_text += "  return bodo.hiframes.pd_dataframe_ext.init_dataframe(({},), index, {})\n".format(
            ", ".join(data_args), col_var
        )
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        _reset_index_impl = loc_vars["_reset_index_impl"]

        nodes = []
        args = [self._get_dataframe_data(df_var, c, nodes) for c in df_typ.columns]
        # add index array arguments if not dropping index
        if not drop:
            if isinstance(df_typ.index, MultiIndexType):
                # MultiIndex case takes multiple arrays from MultiIndex._data
                ind_var = self._get_dataframe_index(df_var, nodes)
                nodes += compile_func_single_block(
                    lambda ind: ind._data, [ind_var], None, self
                )
                arr_tup = nodes[-1].target
                arr_args = []
                for i in range(n_ind):
                    arr_var = ir.Var(ind_var.scope, mk_unique_var("ind_arr"), lhs.loc)
                    self.typemap[arr_var.name] = df_typ.index.array_types[i]
                    gen_getitem(arr_var, arr_tup, i, self.calltypes, nodes)
                    arr_args.append(arr_var)
                args = arr_args + args
            else:
                ind_var = self._gen_array_from_index(df_var, nodes)
                args = [ind_var] + args

        # return new df even for inplace case, since typing pass replaces input variable
        # using output of the call
        return nodes + compile_func_single_block(_reset_index_impl, args, lhs, self)

    def _run_call_query(self, assign, lhs, rhs):
        """Transform query expr to Numba IR using the expr parser in Pandas.
        """
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
        sentinel = pd.core.computation.ops._LOCAL_TAG
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
            func_text += "  {0} = bodo.hiframes.pd_series_ext.init_series({0}, {1})\n".format(
                c_var, ind
            )
        func_text += "  return {}".format(parsed_expr_str)
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        _query_impl = loc_vars["_query_impl"]

        # data frame column inputs
        nodes = []
        args = [self._get_dataframe_data(df_var, c, nodes) for c in used_cols.keys()]
        args.append(self._gen_array_from_index(df_var, nodes))
        # local referenced variables
        args += [ir.Var(lhs.scope, v, lhs.loc) for v in loc_ref_vars.values()]

        nodes += compile_func_single_block(_query_impl, args, lhs, self)

        # check whether the output of generated function is a boolean array
        if (
            not isinstance(
                self.typemap[nodes[-1].value.name],
                bodo.hiframes.pd_series_ext.SeriesType,
            )
            or self.typemap[nodes[-1].value.name].dtype != types.bool_
        ):
            raise BodoError(
                "query(): expr does not evaluate to a 1D boolean array."
                " Only 1D boolean array is supported right now."
            )

        return nodes

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

                if name not in pd.core.computation.ops._mathops or (
                    pd.core.computation.check._NUMEXPR_INSTALLED
                    and pd.core.computation.check_NUMEXPR_VERSION
                    < pd.core.computation.ops.LooseVersion("2.6.9")
                    and name in ("floor", "ceil")
                ):
                    if name not in new_funcs:
                        raise ValueError(
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
            sentinel = pd.core.computation.ops._LOCAL_TAG

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
            """makes math calls compilable by adding "np." and Series functions
            """
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
            pd.core.computation.expr.BaseExprVisitor._maybe_downcast_constants = lambda self, left, right: (
                left,
                right,
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

        df_var = rhs.args[0]
        cname = guard(find_const, self.func_ir, rhs.args[1])
        new_arr = rhs.args[2]
        # inplace = guard(find_const, self.func_ir, rhs.args[3])
        inplace = guard(get_definition, self.func_ir, rhs.args[3]).value
        df_typ = self.typemap[df_var.name]
        nodes = []

        # find df['col2'] = df['col1'][arr]
        # since columns should have the same size, output is filled with NaNs
        # TODO: make sure col1 and col2 are in the same df
        # TODO: compare df index and Series index and match them in setitem
        arr_def = guard(get_definition, self.func_ir, new_arr)
        if (
            isinstance(arr_def, ir.Expr)
            and arr_def.op == "getitem"
            and is_array_typ(self.typemap[arr_def.value.name])
            and self.is_bool_arr(arr_def.index.name)
        ):
            orig_arr = arr_def.value
            bool_arr = arr_def.index
            nodes += compile_func_single_block(
                lambda arr, bool_arr: bodo.hiframes.series_impl.series_filter_bool(
                    arr, bool_arr
                ),
                (orig_arr, bool_arr),
                None,
                self,
            )
            new_arr = nodes[-1].target

        # set unboxed df column with reflection
        if df_typ.has_parent:
            return replace_func(
                self,
                lambda df, cname, arr, inplace: bodo.hiframes.pd_dataframe_ext.set_df_column_with_reflect(
                    df,
                    cname,
                    bodo.utils.conversion.coerce_to_array(
                        arr, scalar_to_arr_len=len(df)
                    ),
                    inplace,
                ),
                [df_var, rhs.args[1], new_arr, rhs.args[3]],
                pre_nodes=nodes,
            )

        if inplace:
            return replace_func(
                self,
                lambda df, arr: bodo.hiframes.pd_dataframe_ext.set_dataframe_data(
                    df,
                    c_ind,
                    bodo.utils.conversion.coerce_to_array(
                        arr, scalar_to_arr_len=len(df)
                    ),
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

        def f(df_arr):  # pragma: no cover
            return len(df_arr)

        return replace_func(self, f, [arr], pre_nodes=nodes)

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
        ) = rhs.args

        left_keys = self._get_const_or_list(left_on_var)
        right_keys = self._get_const_or_list(right_on_var)
        how = guard(find_const, self.func_ir, how_var)
        suffix_x = guard(find_const, self.func_ir, suffix_x_var)
        suffix_y = guard(find_const, self.func_ir, suffix_y_var)
        is_join = guard(find_const, self.func_ir, is_join_var)
        out_typ = self.typemap[lhs.name]
        # convert right join to left join
        is_left = how in {"left", "outer"}
        is_right = how in {"right", "outer"}

        nodes = []
        out_data_vars = {
            c: ir.Var(lhs.scope, mk_unique_var(c), lhs.loc) for c in out_typ.columns
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
        grp_typ = self.typemap[grp_var.name]
        df_var = self._get_df_obj_select(grp_var, "groupby")
        df_type = self.typemap[df_var.name]
        out_typ = self.typemap[lhs.name]
        nodes = []
        if isinstance(out_typ, SeriesType) or func_name in ("agg", "aggregate"):
            in_cols = grp_typ.selection
        else:
            in_cols = [c for c in grp_typ.selection if c in out_typ.columns]
        if func_name in ("agg", "aggregate"):
            func_var = get_call_expr_arg(func_name, rhs.args, dict(rhs.kws), 0, "func")
            agg_func_typ = self.typemap[func_var.name]
            if is_overload_constant_dict(agg_func_typ):
                func_dict = get_overload_constant_dict(agg_func_typ)
                # multi-function const dict case:
                # in this case, the input columns are the ones in the dict
                in_cols = [name for name in func_dict.keys()]
        agg_func = get_agg_func(self.func_ir, func_name, rhs, typemap=self.typemap)
        same_index = False
        return_key = True
        # allfuncs is the set of all functions used
        allfuncs = bodo.ir.aggregate.gen_allfuncs(agg_func, len(in_cols))
        # return_key is True if we return the keys from the table. In case
        # of aggregate on cumsum or other cumulative function, there is no such need.
        # same_index is True if we return the index from the table (which is the case for
        # cumulative operations not using RangeIndex
        for func in allfuncs:
            if func.ftype in list_cumulative:
                same_index = True
                return_key = False
        if same_index and isinstance(grp_typ.df_type.index, RangeIndexType):
            same_index = False

        df_in_vars = {c: self._get_dataframe_data(df_var, c, nodes) for c in in_cols}

        in_key_arrs = [self._get_dataframe_data(df_var, c, nodes) for c in grp_typ.keys]

        out_key_vars = None
        if return_key and (grp_typ.as_index is False or out_typ.index != types.none):
            out_key_vars = []
            for k in grp_typ.keys:
                out_key_var = ir.Var(lhs.scope, mk_unique_var(k), lhs.loc)
                ind = df_type.columns.index(k)
                self.typemap[out_key_var.name] = df_type.data[ind]
                out_key_vars.append(out_key_var)

        if same_index:
            in_index_var = self._gen_array_from_index(df_var, nodes)
            df_in_vars["$_bodo_index_"] = in_index_var
            out_index_var = ir.Var(lhs.scope, mk_unique_var("out_index"), lhs.loc)
            self.typemap[out_index_var.name] = self.typemap[in_index_var.name]
            if out_key_vars == None:
                out_key_vars = []
            out_key_vars.append(out_index_var)

        df_out_vars = {}
        out_colnames = (
            grp_typ.selection if isinstance(out_typ, SeriesType) else out_typ.columns
        )
        for c in out_colnames:
            # output column name can be a string or tuple of strings. the
            # latter case occurs when doing this:
            # df.groupby(...).agg({"A": [f1, f2]})
            # In this case, output names have 2 levels: (A, f1) and (A, f2)
            var = ir.Var(lhs.scope, mk_unique_var(str(c)), lhs.loc)
            self.typemap[var.name] = (
                out_typ.data
                if isinstance(out_typ, SeriesType)
                else out_typ.data[out_typ.columns.index(c)]
            )
            df_out_vars[c] = var

        if len(out_colnames) != len(df_out_vars):
            raise BodoError("aggregate with duplication in output is not allowed")

        agg_node = bodo.ir.aggregate.Aggregate(
            lhs.name,
            df_var.name,
            grp_typ.keys,
            out_key_vars,
            df_out_vars,
            df_in_vars,
            in_key_arrs,
            agg_func,
            same_index,
            return_key,
            lhs.loc,
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
                lambda A: bodo.hiframes.pd_index_ext.init_range_index(
                    0, len(A), 1, None
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
                lambda A: bodo.utils.conversion.index_from_array(A, _index_name),
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
                and grp_typ.explicit_select
                and grp_typ.as_index
            )
            name_str = list(df_out_vars.keys())[0]
            name_var = ir.Var(lhs.scope, mk_unique_var("S_name"), lhs.loc)
            self.typemap[name_var.name] = types.StringLiteral(name_str)
            nodes.append(ir.Assign(ir.Const(name_str, lhs.loc), name_var, lhs.loc))
            return replace_func(
                self,
                lambda A, I, name: bodo.hiframes.pd_series_ext.init_series(A, I, name),
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
        in_vars = {values: self._get_dataframe_data(df_var, values, nodes)}

        df_col_map = {
            col: ir.Var(lhs.scope, mk_unique_var(col), lhs.loc) for col in pivot_values
        }
        for v in df_col_map.values():
            self.typemap[v.name] = out_typ.data[0]

        pivot_arr = self._get_dataframe_data(df_var, columns, nodes)
        index_arr = self._get_dataframe_data(df_var, index, nodes)
        agg_func = get_agg_func(self.func_ir, func_name, rhs)

        same_index = False
        return_key = True
        agg_node = bodo.ir.aggregate.Aggregate(
            lhs.name,
            df_var.name,
            [index],
            None,
            df_col_map,
            in_vars,
            [index_arr],
            agg_func,
            same_index,
            return_key,
            lhs.loc,
            pivot_arr,
            pivot_values,
        )
        nodes.append(agg_node)

        _init_df = _gen_init_df(out_typ.columns)

        # XXX the order of output variables passed should match out_typ.columns
        out_vars = []
        for c in out_typ.columns:
            out_vars.append(df_col_map[c])

        return nodes + compile_func_single_block(_init_df, out_vars, lhs, self)

    def _run_call_crosstab(self, assign, lhs, rhs):
        index, columns, _pivot_values = rhs.args
        pivot_values = self.typemap[_pivot_values.name].meta
        out_typ = self.typemap[lhs.name]

        nodes = []
        if isinstance(self.typemap[index.name], SeriesType):
            nodes += compile_func_single_block(
                lambda S: bodo.hiframes.pd_series_ext.get_series_data(S),
                (index,),
                None,
                self,
            )
            index = nodes[-1].target

        if isinstance(self.typemap[columns.name], SeriesType):
            nodes += compile_func_single_block(
                lambda S: bodo.hiframes.pd_series_ext.get_series_data(S),
                (columns,),
                None,
                self,
            )
            columns = nodes[-1].target

        in_vars = {}

        df_col_map = {
            col: ir.Var(lhs.scope, mk_unique_var(col), lhs.loc) for col in pivot_values
        }
        for i, v in enumerate(df_col_map.values()):
            self.typemap[v.name] = out_typ.data[i]

        pivot_arr = columns

        def _agg_len_impl(in_arr):  # pragma: no cover
            return len(in_arr)

        # TODO: make out_key_var an index column
        same_index = False
        return_key = True
        agg_node = bodo.ir.aggregate.Aggregate(
            lhs.name,
            "crosstab",
            [index.name],
            None,
            df_col_map,
            in_vars,
            [index],
            _agg_len_impl,
            same_index,
            return_key,
            lhs.loc,
            pivot_arr,
            pivot_values,
            True,
        )
        nodes.append(agg_node)

        _init_df = _gen_init_df(out_typ.columns)

        # XXX the order of output variables passed should match out_typ.columns
        out_vars = []
        for c in out_typ.columns:
            out_vars.append(df_col_map[c])

        return nodes + compile_func_single_block(_init_df, out_vars, lhs, self)

    def _run_call_rolling(self, assign, lhs, rhs, rolling_var, func_name):
        rolling_typ = self.typemap[rolling_var.name]
        dummy_call = guard(get_definition, self.func_ir, rolling_var)
        df_var, window, center, on = dummy_call.args
        df_type = self.typemap[df_var.name]
        out_typ = self.typemap[lhs.name]

        # handle 'on' arg
        if self.typemap[on.name] == types.none:
            on = None
        else:
            assert isinstance(self.typemap[on.name], types.StringLiteral)
            on = self.typemap[on.name].literal_value

        nodes = []
        window_const = guard(find_const, self.func_ir, window)
        # convert string offset window statically to nanos
        # TODO: support dynamic conversion
        # TODO: support other offsets types (time delta, etc.)
        if on is not None and not isinstance(window_const, str):
            raise ValueError(
                "window argument to rolling should be constant"
                "string in the offset case (variable window)"
            )

        if isinstance(window_const, str):
            window = pd.tseries.frequencies.to_offset(window_const).nanos
            window_var = ir.Var(lhs.scope, mk_unique_var("window"), lhs.loc)
            self.typemap[window_var.name] = types.int64
            nodes.append(ir.Assign(ir.Const(window, lhs.loc), window_var, lhs.loc))
            window = window_var

        in_vars = {
            c: self._get_dataframe_data(df_var, c, nodes) for c in rolling_typ.selection
        }

        on_arr = self._get_dataframe_data(df_var, on, nodes) if on is not None else None
        out_index_var = self._get_dataframe_index(df_var, nodes)
        on_index_arr = on_arr
        if isinstance(window_const, str) and on_arr is None:
            on_index_arr = self._gen_array_from_index(df_var, nodes)

        df_col_map = {}
        for c in rolling_typ.selection:
            var = ir.Var(lhs.scope, mk_unique_var(c), lhs.loc)
            self.typemap[var.name] = (
                out_typ.data
                if isinstance(out_typ, SeriesType)
                else out_typ.data[out_typ.columns.index(c)]
            )
            df_col_map[c] = var

        if on is not None:
            df_col_map[on] = on_arr  # TODO: copy array?

        other = None
        if func_name in ("cov", "corr"):
            other = rhs.args[0]

        for cname, out_col_var in df_col_map.items():
            if cname == on:
                continue
            in_col_var = in_vars[cname]
            if func_name in ("cov", "corr"):
                # TODO: Series as other
                if cname not in self.typemap[other.name].columns:
                    continue  # nan column handled below
                rhs.args[0] = self._get_dataframe_data(other, cname, nodes)
            nodes += self._gen_rolling_call(
                in_col_var,
                out_col_var,
                window,
                center,
                rhs.args,
                func_name,
                on_index_arr,
            )

        # in corr/cov case, Pandas makes non-common columns NaNs
        if func_name in ("cov", "corr"):
            nan_cols = list(
                sorted(set(self.typemap[other.name].columns) ^ set(df_type.columns))
            )
            len_arr = list(in_vars.values())[0]
            for cname in nan_cols:

                def f(arr):  # pragma: no cover
                    return np.full(len(arr), np.nan)

                nodes += compile_func_single_block(f, [len_arr], None, self)
                df_col_map[cname] = nodes[-1].target

        # XXX output becomes series if single output and explicitly selected
        if isinstance(out_typ, SeriesType):
            # TODO: test (needs support in typing?)
            assert (
                len(rolling_typ.selection) == 1
                and rolling_typ.explicit_select
                and rolling_typ.as_index
            )
            return replace_func(
                self,
                lambda A, I: bodo.hiframes.pd_series_ext.init_series(A, I, _name),
                list(df_col_map.values()) + [out_index_var],
                pre_nodes=nodes,
                extra_globals={"_name": list(df_col_map.keys())[0]},
            )

        _init_df = _gen_init_df(out_typ.columns, "index")

        # XXX the order of output variables passed should match out_typ.columns
        out_vars = []
        for c in out_typ.columns:
            out_vars.append(df_col_map[c])
        out_vars.append(out_index_var)

        return nodes + compile_func_single_block(_init_df, out_vars, lhs, self)

    def _gen_rolling_call(
        self, in_col_var, out_col_var, window, center, args, func_name, on_arr
    ):
        nodes = []
        if func_name in ("cov", "corr"):
            other = args[0]
            if on_arr is not None:
                if func_name == "cov":

                    def f(arr, other, on_arr, w, center):  # pragma: no cover
                        return bodo.hiframes.rolling.rolling_cov(
                            arr, other, on_arr, w, center
                        )

                if func_name == "corr":

                    def f(arr, other, on_arr, w, center):  # pragma: no cover
                        return bodo.hiframes.rolling.rolling_corr(
                            arr, other, on_arr, w, center
                        )

                args = [in_col_var, other, on_arr, window, center]
            else:
                if func_name == "cov":

                    def f(arr, other, w, center):  # pragma: no cover
                        return bodo.hiframes.rolling.rolling_cov(arr, other, w, center)

                if func_name == "corr":

                    def f(arr, other, w, center):  # pragma: no cover
                        return bodo.hiframes.rolling.rolling_corr(arr, other, w, center)

                args = [in_col_var, other, window, center]
        # variable window case
        elif on_arr is not None:
            if func_name == "apply":

                def f(arr, on_arr, w, center, func):  # pragma: no cover
                    return bodo.hiframes.rolling.rolling_variable(
                        arr, on_arr, w, center, False, func
                    )

                args = [in_col_var, on_arr, window, center, args[0]]
            else:

                def f(arr, on_arr, w, center):  # pragma: no cover
                    return bodo.hiframes.rolling.rolling_variable(
                        arr, on_arr, w, center, False, _func_name
                    )

                args = [in_col_var, on_arr, window, center]
        else:  # fixed window
            # apply case takes the passed function instead of just name
            if func_name == "apply":

                def f(arr, w, center, func):  # pragma: no cover
                    return bodo.hiframes.rolling.rolling_fixed(
                        arr, w, center, False, func
                    )

                args = [in_col_var, window, center, args[0]]
            else:

                def f(arr, w, center):  # pragma: no cover
                    return bodo.hiframes.rolling.rolling_fixed(
                        arr, w, center, False, _func_name
                    )

                args = [in_col_var, window, center]

        return nodes + compile_func_single_block(
            f, args, out_col_var, self, extra_globals={"_func_name": func_name}
        )

    def _run_call_concat(self, assign, lhs, rhs):
        # TODO: handle non-numerical (e.g. string, datetime) columns
        nodes = []
        out_typ = self.typemap[lhs.name]
        df_list = self._get_const_tup(rhs.args[0])
        axis = guard(find_const, self.func_ir, rhs.args[1])
        if axis == 1:
            return self._run_call_concat_columns(df_list, out_typ, lhs)

        # generate a concat call for each output column
        # gen concat function
        arg_names = ", ".join(["in{}".format(i) for i in range(len(df_list))])
        func_text = "def _concat_imp({}):\n".format(arg_names)
        func_text += "    return bodo.libs.array_kernels.concat(({}))\n".format(
            arg_names
        )
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        _concat_imp = loc_vars["_concat_imp"]

        out_vars = []
        for cname in out_typ.columns:
            # find an example input array for this output array to use for typing in
            # gen_na_array()
            example_arr = None
            for df in df_list:
                df_typ = self.typemap[df.name]
                if cname in df_typ.columns:
                    example_arr = self._get_dataframe_data(df, cname, nodes)
                    break
            # arguments to the generated function
            args = []
            # get input columns
            for df in df_list:
                df_typ = self.typemap[df.name]
                # generate full NaN column
                if cname not in df_typ.columns:
                    arr = self._get_dataframe_data(df, df_typ.columns[0], nodes)
                    nodes += compile_func_single_block(
                        lambda arr, A: bodo.libs.array_kernels.gen_na_array(
                            len(arr), A
                        ),
                        (arr, example_arr),
                        None,
                        self,
                    )
                    args.append(nodes[-1].target)
                else:
                    arr = self._get_dataframe_data(df, cname, nodes)
                    args.append(arr)

            nodes += compile_func_single_block(_concat_imp, args, None, self)
            out_vars.append(nodes[-1].target)

        _init_df = _gen_init_df(out_typ.columns)

        return nodes + compile_func_single_block(_init_df, out_vars, lhs, self)

    def _run_call_concat_columns(self, objs, out_typ, lhs):
        nodes = []
        out_vars = []
        for obj in objs:
            nodes += compile_func_single_block(
                lambda S: bodo.hiframes.pd_series_ext.get_series_data(S),
                (obj,),
                None,
                self,
            )
            out_vars.append(nodes[-1].target)

        _init_df = _gen_init_df(out_typ.columns)

        return nodes + compile_func_single_block(_init_df, out_vars, lhs, self)

    def _get_df_obj_select(self, obj_var, obj_name):
        """get df object for groupby() or rolling()
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
        call_def = guard(find_callname, self.func_ir, obj_call)
        assert (
            call_def is not None
            and call_def[0] == obj_name
            and isinstance(call_def[1], ir.Var)
            and self._is_df_var(call_def[1])
        )
        df_var = call_def[1]

        return df_var

    def _get_const_tup(self, tup_var):
        tup_def = guard(get_definition, self.func_ir, tup_var)
        if isinstance(tup_def, ir.Expr):
            if tup_def.op == "binop" and tup_def.fn in ("+", operator.add):
                return self._get_const_tup(tup_def.lhs) + self._get_const_tup(
                    tup_def.rhs
                )
            if tup_def.op in ("build_tuple", "build_list"):
                return tup_def.items
        raise ValueError("constant tuple expected")

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
        call_def = guard(find_callname, self.func_ir, var_def)
        if call_def == ("init_dataframe", "bodo.hiframes.pd_dataframe_ext"):
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
            lambda df, c_ind: bodo.hiframes.pd_dataframe_ext.get_dataframe_data(
                df, c_ind
            ),
            (df_var, ind_var),
            None,
            self,
        )
        return nodes[-1].target

    def _get_dataframe_index(self, df_var, nodes):
        df_typ = self.typemap[df_var.name]
        var_def = guard(get_definition, self.func_ir, df_var)
        call_def = guard(find_callname, self.func_ir, var_def)
        if call_def == ("init_dataframe", "bodo.hiframes.pd_dataframe_ext"):
            return var_def.args[1]

        # XXX use get_series_data() for getting data instead of S._data
        # to enable alias analysis
        nodes += compile_func_single_block(
            lambda df: bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df),
            (df_var,),
            None,
            self,
        )
        return nodes[-1].target

    def _get_index_name(self, dt_var, nodes):
        var_def = guard(get_definition, self.func_ir, dt_var)
        call_def = guard(find_callname, self.func_ir, var_def)
        if (
            call_def
            in (
                ("init_datetime_index", "bodo.hiframes.pd_index_ext"),
                ("init_timedelta_index", "bodo.hiframes.pd_index_ext"),
                ("init_string_index", "bodo.hiframes.pd_index_ext"),
                ("init_numeric_index", "bodo.hiframes.pd_index_ext"),
            )
            and len(var_def.args) == 2
        ):
            return var_def.args[1]

        f = lambda S: bodo.hiframes.pd_index_ext.get_index_name(S)
        if self.typemap[dt_var.name] == types.none:
            f = lambda S: None

        nodes += compile_func_single_block(f, (dt_var,), None, self)
        return nodes[-1].target

    def _is_df_var(self, var):
        return isinstance(self.typemap[var.name], DataFrameType)

    def is_bool_arr(self, varname):
        typ = self.typemap[varname]
        return (
            isinstance(typ, (SeriesType, types.Array, BooleanArrayType))
            and typ.dtype == types.bool_
        )

    def is_int_list_or_arr(self, varname):
        typ = self.typemap[varname]
        return isinstance(typ, (SeriesType, types.Array, types.List)) and isinstance(
            typ.dtype, types.Integer
        )

    def _is_const_none(self, var):
        var_def = guard(get_definition, self.func_ir, var)
        return isinstance(var_def, ir.Const) and var_def.value is None

    def _update_definitions(self, node_list):
        loc = ir.Loc("", 0)
        dumm_block = ir.Block(ir.Scope(None, loc), loc)
        dumm_block.body = node_list
        build_definitions({0: dumm_block}, self.func_ir._definitions)
        return

    def _gen_arr_copy(self, in_arr, nodes):
        nodes += compile_func_single_block(lambda A: A.copy(), (in_arr,), None, self)
        return nodes[-1].target

    def _get_const_or_list(
        self, by_arg, list_only=False, default=None, err_msg=None, typ=None
    ):
        var_typ = self.typemap[by_arg.name]
        if isinstance(var_typ, types.Optional):
            var_typ = var_typ.type
        if isinstance(var_typ, bodo.utils.typing.ListLiteral):
            return var_typ.literal_value

        typ = str if typ is None else typ
        by_arg_def = guard(find_build_sequence, self.func_ir, by_arg)
        if by_arg_def is None:
            # try single key column
            by_arg_def = guard(find_const, self.func_ir, by_arg)
            if by_arg_def is None:
                if default is not None:
                    return default
                raise ValueError(err_msg)
            if isinstance(var_typ, types.BaseTuple):
                assert isinstance(by_arg_def, tuple)
                return by_arg_def
            key_colnames = (by_arg_def,)
        else:
            if list_only and by_arg_def[1] != "build_list":
                if default is not None:
                    return default
                raise ValueError(err_msg)
            key_colnames = tuple(
                guard(find_const, self.func_ir, v) for v in by_arg_def[0]
            )
            if any(not isinstance(v, typ) for v in key_colnames):
                if default is not None:
                    return default
                raise ValueError(err_msg)
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
    assert isinstance(first_stmt, ir.Assign) and isinstance(
        first_stmt.value, ir.Arg
    )
    arg_var = first_stmt.target
    use_all_cols = False
    for bl in lambda_ir.blocks.values():
        for stmt in bl.body:
            vnames = [v.name for v in stmt.list_vars()]
            if arg_var.name in vnames:
                if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Arg):
                    continue
                if (
                    isinstance(stmt, ir.Assign)
                    and isinstance(stmt.value, ir.Expr)
                    and stmt.value.op == "getattr"
                ):
                    assert stmt.value.attr in columns
                    used_cols.append(stmt.value.attr)
                else:
                    # argument is used in some other form
                    # be conservative and use all cols
                    use_all_cols = True
                    used_cols = columns
                    break

        if use_all_cols:
            break

    # remove duplicates with set() since a column can be used multiple times
    used_cols = sorted(set(used_cols))
    return used_cols


def _eval_const_var(func_ir, var):
    try:
        return find_const(func_ir, var)
    except GuardException:
        pass
    var_def = guard(get_definition, func_ir, var)
    if isinstance(var_def, ir.Expr) and var_def.op == "binop":
        return var_def.fn(
            _eval_const_var(func_ir, var_def.lhs), _eval_const_var(func_ir, var_def.rhs)
        )

    raise GuardException
