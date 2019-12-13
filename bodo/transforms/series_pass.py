# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
converts Series operations to array operations as much as possible
to provide implementation and enable optimization.
"""
import operator
from collections import defaultdict, namedtuple
import re
import numpy as np
import pandas as pd
import warnings
import datetime

import numba
from numba import ir, ir_utils, types
from numba.ir_utils import (
    replace_arg_nodes,
    compile_to_numba_ir,
    find_topo_order,
    gen_np_call,
    get_definition,
    guard,
    find_callname,
    mk_alloc,
    find_const,
    is_setitem,
    is_getitem,
    mk_unique_var,
    dprint_func_ir,
    build_definitions,
    find_build_sequence,
)
from numba.inline_closurecall import inline_closure_call
from numba.typing.templates import Signature, bound_function, signature
from numba.typing.arraydecl import ArrayAttribute
from numba.extending import overload
from numba.typing.templates import infer_global, AbstractTemplate, signature
import bodo
from bodo import hiframes
from bodo.utils.utils import (
    debug_prints,
    inline_new_blocks,
    ReplaceFunc,
    is_whole_slice,
    get_getsetitem_index_var,
    is_expr,
)
from bodo.libs.str_ext import string_type, unicode_to_std_str, std_str_to_unicode
from bodo.libs.list_str_arr_ext import list_string_array_type
from bodo.libs.str_arr_ext import (
    string_array_type,
    StringArrayType,
    is_str_arr_typ,
    pre_alloc_string_array,
    get_utf8_size,
)
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.bool_arr_ext import boolean_array
from bodo.hiframes.pd_categorical_ext import CategoricalArray
from bodo.hiframes.pd_series_ext import (
    SeriesType,
    is_str_series_typ,
    series_to_array_type,
    is_dt64_series_typ,
    is_bool_series_typ,
    if_series_to_array_type,
    is_series_type,
    SeriesRollingType,
)
from bodo.hiframes.pd_index_ext import DatetimeIndexType, TimedeltaIndexType
from bodo.io.h5_api import h5dataset_type
from bodo.hiframes.rolling import get_rolling_setup_args
import bodo.hiframes.series_impl  # side effect: install Series overloads
import bodo.hiframes.series_indexing  # side effect: install Series overloads
import bodo.hiframes.series_str_impl  # side effect: install Series overloads
import bodo.hiframes.series_dt_impl  # side effect: install Series overloads
from bodo.hiframes.series_dt_impl import SeriesDatetimePropertiesType
from bodo.hiframes.series_str_impl import SeriesStrMethodType
from bodo.hiframes.series_indexing import SeriesIatType
from bodo.ir.aggregate import Aggregate
from bodo.hiframes import series_kernels, split_impl
from bodo.hiframes.series_kernels import series_replace_funcs
from bodo.hiframes.split_impl import (
    string_array_split_view_type,
    StringArraySplitViewType,
    getitem_c_arr,
    get_array_ctypes_ptr,
    get_split_view_index,
    get_split_view_data_ptr,
)
from bodo.utils.transform import compile_func_single_block, update_locs


ufunc_names = set(f.__name__ for f in numba.typing.npydecl.supported_ufuncs)


_dt_index_binops = (
    "==",
    "!=",
    ">=",
    ">",
    "<=",
    "<",
    "-",
    operator.eq,
    operator.ne,
    operator.ge,
    operator.gt,
    operator.le,
    operator.lt,
    operator.sub,
)

_string_array_comp_ops = (
    operator.eq,
    operator.ne,
    operator.ge,
    operator.gt,
    operator.le,
    operator.lt,
)

_binop_to_str = {
    operator.eq: "==",
    operator.ne: "!=",
    operator.ge: ">=",
    operator.gt: ">",
    operator.le: "<=",
    operator.lt: "<",
    operator.sub: "-",
    operator.add: "+",
    operator.mul: "*",
    operator.truediv: "/",
    operator.floordiv: "//",
    operator.mod: "%",
    operator.pow: "**",
    "==": "==",
    "!=": "!=",
    ">=": ">=",
    ">": ">",
    "<=": "<=",
    "<": "<",
    "-": "-",
    "+": "+",
    "*": "*",
    "/": "/",
    "//": "//",
    "%": "%",
    "**": "**",
}


class SeriesPass(object):
    """Analyze and transform hiframes calls after typing"""

    def __init__(self, func_ir, typingctx, typemap, calltypes):
        self.func_ir = func_ir
        self.typingctx = typingctx
        self.typemap = typemap
        self.calltypes = calltypes
        # Loc object of current location being translated
        self.curr_loc = self.func_ir.loc
        # keep track of tuple variables change by to_const_tuple
        self._type_changed_vars = []

    def run(self):
        blocks = self.func_ir.blocks
        ir_utils.remove_dels(blocks)
        # topo_order necessary so Series data replacement optimization can be
        # performed in one pass
        topo_order = find_topo_order(blocks)
        work_list = list((l, blocks[l]) for l in reversed(topo_order))
        while work_list:
            label, block = work_list.pop()
            new_body = []
            replaced = False
            for i, inst in enumerate(block.body):
                out_nodes = [inst]
                self.curr_loc = inst.loc

                if isinstance(inst, ir.Assign):
                    self.func_ir._definitions[inst.target.name].remove(inst.value)
                    out_nodes = self._run_assign(inst)
                elif isinstance(inst, (ir.SetItem, ir.StaticSetItem)):
                    out_nodes = self._run_setitem(inst)
                else:
                    if isinstance(
                        inst,
                        (
                            Aggregate,
                            bodo.ir.sort.Sort,
                            bodo.ir.join.Join,
                            bodo.ir.filter.Filter,
                            bodo.ir.csv_ext.CsvReader,
                        ),
                    ):
                        out_nodes = self._handle_hiframes_nodes(inst)

                if isinstance(out_nodes, list):
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
                        work_list,
                    )
                    # update Loc objects
                    for c_block in callee_blocks.values():
                        c_block.loc = self.curr_loc
                        update_locs(c_block.body, self.curr_loc)
                    replaced = True
                    break
                if isinstance(out_nodes, dict):
                    block.body = new_body + block.body[i:]
                    inline_new_blocks(self.func_ir, block, i, out_nodes, work_list)
                    replaced = True
                    break

            if not replaced:
                blocks[label].body = new_body

        self.func_ir.blocks = ir_utils.simplify_CFG(self.func_ir.blocks)
        while ir_utils.remove_dead(
            self.func_ir.blocks, self.func_ir.arg_names, self.func_ir, self.typemap
        ):
            pass

        self.func_ir._definitions = build_definitions(self.func_ir.blocks)
        dprint_func_ir(self.func_ir, "after series pass")
        return

    def _run_assign(self, assign):
        lhs = assign.target.name
        rhs = assign.value

        # fix type of lhs if type of rhs has been changed
        if isinstance(rhs, ir.Var) and rhs.name in self._type_changed_vars:
            self.typemap.pop(lhs)
            self.typemap[lhs] = self.typemap[rhs.name]
            self._type_changed_vars.append(lhs)

        if isinstance(rhs, ir.Expr):
            if rhs.op == "getattr":
                return self._run_getattr(assign, rhs)

            if rhs.op == "binop":
                return self._run_binop(assign, rhs)

            # XXX handling inplace_binop similar to binop for now
            # TODO handle inplace alignment similar to
            # add_special_arithmetic_methods() in pandas ops.py
            # TODO: inplace of str array?
            if rhs.op == "inplace_binop":
                return self._run_binop(assign, rhs)

            if rhs.op == "unary":
                return self._run_unary(assign, rhs)

            # replace getitems on Series.iat
            if rhs.op in ("getitem", "static_getitem"):
                return self._run_getitem(assign, rhs)

            if rhs.op == "call":
                return self._run_call(assign, lhs, rhs)

        return [assign]

    def _run_getitem(self, assign, rhs):
        target = rhs.value
        target_typ = self.typemap[target.name]
        nodes = []
        idx = get_getsetitem_index_var(rhs, self.typemap, nodes)
        idx_typ = self.typemap[idx.name]

        # optimize out getitem on built_tuple, important for pd.DataFrame()
        # since dictionary is converted to tuple
        if isinstance(target_typ, types.BaseTuple) and isinstance(
            idx_typ, types.IntegerLiteral
        ):
            val_def = guard(get_definition, self.func_ir, rhs.value)
            if isinstance(val_def, ir.Expr) and val_def.op == "build_tuple":
                assign.value = val_def.items[idx_typ.literal_value]
                return [assign]

        if is_series_type(target_typ):
            impl = bodo.hiframes.series_indexing.overload_series_getitem(
                self.typemap[target.name], self.typemap[idx.name]
            )
            return self._replace_func(impl, (target, idx), pre_nodes=nodes)

        # inline index getitem, TODO: test
        if bodo.hiframes.pd_index_ext.is_pd_index_type(target_typ):
            typ1, typ2 = self.typemap[target.name], self.typemap[idx.name]
            if isinstance(target_typ, bodo.hiframes.pd_index_ext.RangeIndexType):
                impl = bodo.hiframes.pd_index_ext.overload_range_index_getitem(
                    typ1, typ2
                )
            elif isinstance(target_typ, bodo.hiframes.pd_index_ext.DatetimeIndexType):
                impl = bodo.hiframes.pd_index_ext.overload_datetime_index_getitem(
                    typ1, typ2
                )
            else:
                # TODO: test timedelta
                impl = bodo.hiframes.pd_index_ext.overload_index_getitem(typ1, typ2)
            return self._replace_func(impl, (target, idx), pre_nodes=nodes)

        # TODO: reimplement Iat optimization
        # if isinstance(self.typemap[rhs.value.name], SeriesIatType):
        #     val_def = guard(get_definition, self.func_ir, rhs.value)
        #     assert (isinstance(val_def, ir.Expr) and val_def.op == 'getattr'
        #         and val_def.attr in ('iat', 'iloc', 'loc'))
        #     series_var = val_def.value
        #     rhs.value = series_var

        nodes.append(assign)
        return nodes

    def _run_setitem(self, inst):
        target_typ = self.typemap[inst.target.name]
        # Series as index
        # TODO: handle all possible cases
        nodes = []

        if target_typ == h5dataset_type:
            return self._handle_h5_write(inst.target, inst.index, inst.value)

        # TODO: proper iat/iloc/loc optimization
        # if isinstance(target_typ, SeriesIatType):
        #     val_def = guard(get_definition, self.func_ir, inst.target)
        #     assert (isinstance(val_def, ir.Expr) and val_def.op == 'getattr'
        #         and val_def.attr in ('iat', 'iloc', 'loc'))
        #     series_var = val_def.value
        #     inst.target = series_var
        #     target_typ = target_typ.stype

        nodes.append(inst)
        return nodes

    def _run_getattr(self, assign, rhs):
        rhs_type = self.typemap[rhs.value.name]  # get type of rhs value "S"

        if isinstance(rhs_type, SeriesStrMethodType) and rhs.attr == "_obj":
            rhs_def = guard(get_definition, self.func_ir, rhs.value)
            # conditional on S.str is not supported since aliasing, ... can't
            # be handled for getattr. This is probably a rare case.
            # TODO: handle, example:
            # if flag:
            #    S_str = S1.str
            # else:
            #    S_str = S2.str
            if rhs_def is None:
                raise ValueError("Invalid Series.str, cannot handle conditional yet")
            assert is_expr(rhs_def, "getattr")
            assign.value = rhs_def.value
            return [assign]

        if isinstance(rhs_type, SeriesDatetimePropertiesType) and rhs.attr == "_obj":
            rhs_def = guard(get_definition, self.func_ir, rhs.value)
            if rhs_def is None:
                raise ValueError("Invalid Series.dt, cannot handle conditional yet")
            assert is_expr(rhs_def, "getattr")
            assign.value = rhs_def.value
            return [assign]

        if isinstance(rhs_type, SeriesDatetimePropertiesType):
            if rhs.attr == "date":
                impl = bodo.hiframes.series_dt_impl.series_dt_date_overload(rhs_type)
            else:
                impl = bodo.hiframes.series_dt_impl.create_date_field_overload(rhs.attr)(rhs_type)
            return self._replace_func(impl, [rhs.value])


        # replace arr.dtype for dt64 since PA replaces with
        # np.datetime64[ns] which invalid, TODO: fix PA
        if (
            rhs.attr == "dtype"
            and (is_series_type(rhs_type) or isinstance(rhs_type, types.Array))
            and isinstance(rhs_type.dtype, (types.NPDatetime, types.NPTimedelta))
        ):
            assign.value = ir.Global("numpy.datetime64", rhs_type.dtype, rhs.loc)
            return [assign]

        # replace arr.dtype since PA replacement inserts in the
        # beginning of block, preventing fusion. TODO: fix PA
        if rhs.attr == "dtype" and isinstance(
            if_series_to_array_type(rhs_type), types.Array
        ):
            typ_str = str(rhs_type.dtype)
            assign.value = ir.Global(
                "np.dtype({})".format(typ_str), np.dtype(typ_str), rhs.loc
            )
            return [assign]

        # replace attribute access with overload
        if isinstance(rhs_type, SeriesType) and rhs.attr in (
            "index",
            "values",
            "shape",
            "dtype",
            "ndim",
            "size",
            "T",
            "hasnans",
            "empty",
            "dtypes",
            "name",
        ):
            #
            overload_name = "overload_series_" + rhs.attr
            overload_func = getattr(bodo.hiframes.series_impl, overload_name)
            impl = overload_func(rhs_type)
            return self._replace_func(impl, [rhs.value])

        if isinstance(rhs_type, SeriesType) and rhs.attr == "values":
            # simply return the column
            nodes = []
            var = self._get_series_data(rhs.value, nodes)
            assign.value = var
            nodes.append(assign)
            return nodes

        if isinstance(rhs_type, SeriesType) and rhs.attr == "shape":
            nodes = []
            data = self._get_series_data(rhs.value, nodes)
            return self._replace_func(lambda A: (len(A),), [data], pre_nodes=nodes)

        if isinstance(rhs_type, DatetimeIndexType) and rhs.attr == "values":
            # simply return the data array
            nodes = []
            var = self._get_index_data(rhs.value, nodes)
            assign.value = var
            nodes.append(assign)
            return nodes

        if isinstance(rhs_type, DatetimeIndexType):
            # TODO: test this inlining
            if rhs.attr in bodo.hiframes.pd_timestamp_ext.date_fields:
                impl = bodo.hiframes.pd_index_ext.gen_dti_field_impl(rhs.attr)
                return self._replace_func(impl, [rhs.value])
            if rhs.attr == "date":
                impl = bodo.hiframes.pd_index_ext.overload_datetime_index_date(rhs_type)
                return self._replace_func(impl, [rhs.value])

        if isinstance(rhs_type, TimedeltaIndexType):
            # TODO: test this inlining
            if rhs.attr in bodo.hiframes.pd_timestamp_ext.timedelta_fields:
                impl = bodo.hiframes.pd_index_ext.gen_tdi_field_impl(rhs.attr)
                return self._replace_func(impl, [rhs.value])

        if isinstance(rhs_type, SeriesType) and rhs.attr in ("size", "shape"):
            # simply return the column
            nodes = []
            var = self._get_series_data(rhs.value, nodes)
            rhs.value = var
            nodes.append(assign)
            return nodes

        # TODO: test ndim and T
        if isinstance(rhs_type, SeriesType) and rhs.attr == "ndim":
            rhs.value = ir.Const(1, rhs.loc)
            return [assign]

        if isinstance(rhs_type, SeriesType) and rhs.attr == "T":
            rhs = rhs.value
            return [assign]

        return [assign]

    def _run_binop(self, assign, rhs):
        res = self._handle_string_array_expr(assign, rhs)
        if res is not None:
            return res

        arg1, arg2 = rhs.lhs, rhs.rhs
        typ1, typ2 = self.typemap[arg1.name], self.typemap[arg2.name]

        if isinstance(typ1, CategoricalArray) and isinstance(typ2, types.StringLiteral):
            impl = bodo.hiframes.pd_categorical_ext.overload_cat_arr_eq_str(typ1, typ2)
            return self._replace_func(impl, [arg1, arg2])

        # both dt64
        if is_dt64_series_typ(typ1) and is_dt64_series_typ(typ2):
            func = rhs.fn

            def impl(S1, S2):
                arr1 = bodo.hiframes.pd_series_ext.get_series_data(S1)
                arr2 = bodo.hiframes.pd_series_ext.get_series_data(S2)
                l = len(arr1)
                S = np.empty(l, dtype=np.bool_)
                nulls = np.empty((l + 7) >> 3, dtype=np.uint8)
                for i in numba.parfor.internal_prange(l):
                    S[i] = func(arr1[i], arr2[i])
                    bodo.libs.int_arr_ext.set_bit_to_arr(nulls, i, 1)
                return bodo.hiframes.pd_series_ext.init_series(
                    bodo.libs.bool_arr_ext.init_bool_array(S, nulls)
                )

            return self._replace_func(impl, [arg1, arg2])

        # inline overloaded
        # TODO: use overload inlining when available
        if rhs.fn == operator.sub:
            if (
                isinstance(typ1, DatetimeIndexType)
                and typ2 == bodo.hiframes.pd_timestamp_ext.pandas_timestamp_type
            ) or (
                isinstance(typ2, DatetimeIndexType)
                and typ1 == bodo.hiframes.pd_timestamp_ext.pandas_timestamp_type
            ):
                impl = bodo.hiframes.pd_index_ext.overload_datetime_index_sub(
                    typ1, typ2
                )
                return self._replace_func(impl, [arg1, arg2])

        # string comparison with DatetimeIndex
        if rhs.fn in (
            operator.eq,
            operator.ne,
            operator.ge,
            operator.gt,
            operator.le,
            operator.lt,
        ) and (
            (
                isinstance(typ1, DatetimeIndexType)
                and types.unliteral(typ2) == string_type
            )
            or (
                isinstance(typ2, DatetimeIndexType)
                and types.unliteral(typ1) == string_type
            )
        ):
            impl = bodo.hiframes.pd_index_ext.overload_binop_dti_str(rhs.fn)(typ1, typ2)
            return self._replace_func(impl, [arg1, arg2])

        if self._is_dt_index_binop(rhs):
            return self._handle_dt_index_binop(assign, rhs)

        if rhs.fn in numba.typing.npydecl.NumpyRulesArrayOperator._op_map.keys() and any(
            isinstance(t, IntegerArrayType) for t in (typ1, typ2)
        ):
            overload_func = bodo.libs.int_arr_ext.create_op_overload(rhs.fn, 2)
            impl = overload_func(typ1, typ2)
            return self._replace_func(impl, [arg1, arg2])

        if rhs.fn in numba.typing.npydecl.NumpyRulesInplaceArrayOperator._op_map.keys() and any(
            isinstance(t, IntegerArrayType) for t in (typ1, typ2)
        ):
            overload_func = bodo.libs.int_arr_ext.create_op_overload(rhs.fn, 2)
            impl = overload_func(typ1, typ2)
            return self._replace_func(impl, [arg1, arg2])

        if rhs.fn in numba.typing.npydecl.NumpyRulesArrayOperator._op_map.keys() and any(
            t == boolean_array for t in (typ1, typ2)
        ):
            overload_func = bodo.libs.bool_arr_ext.create_op_overload(rhs.fn, 2)
            impl = overload_func(typ1, typ2)
            return self._replace_func(impl, [arg1, arg2])

        if rhs.fn in numba.typing.npydecl.NumpyRulesInplaceArrayOperator._op_map.keys() and any(
            t == boolean_array for t in (typ1, typ2)
        ):
            overload_func = bodo.libs.bool_arr_ext.create_op_overload(rhs.fn, 2)
            impl = overload_func(typ1, typ2)
            return self._replace_func(impl, [arg1, arg2])

        if not (isinstance(typ1, SeriesType) or isinstance(typ2, SeriesType)):
            return [assign]

        if rhs.fn in bodo.hiframes.pd_series_ext.series_binary_ops:
            overload_func = bodo.hiframes.series_impl.create_binary_op_overload(rhs.fn)
            impl = overload_func(typ1, typ2)
            return self._replace_func(impl, [arg1, arg2])

        if rhs.fn in bodo.hiframes.pd_series_ext.series_inplace_binary_ops:
            overload_func = bodo.hiframes.series_impl.create_inplace_binary_op_overload(
                rhs.fn
            )
            impl = overload_func(typ1, typ2)
            return self._replace_func(impl, [arg1, arg2])

        # TODO: remove this code
        nodes = []
        # TODO: support alignment, dt, etc.
        # S3 = S1 + S2 ->
        # S3_data = S1_data + S2_data; S3 = init_series(S3_data)
        if isinstance(typ1, SeriesType):
            arg1 = self._get_series_data(arg1, nodes)
        if isinstance(typ2, SeriesType):
            arg2 = self._get_series_data(arg2, nodes)

        rhs.lhs, rhs.rhs = arg1, arg2
        self._convert_series_calltype(rhs)

        # output stays as Array in A += B where A is Array
        if isinstance(self.typemap[assign.target.name], types.Array):
            assert isinstance(self.calltypes[rhs].return_type, types.Array)
            nodes.append(assign)
            return nodes

        out_data = ir.Var(
            arg1.scope, mk_unique_var(assign.target.name + "_data"), rhs.loc
        )
        self.typemap[out_data.name] = self.calltypes[rhs].return_type
        nodes.append(ir.Assign(rhs, out_data, rhs.loc))
        return self._replace_func(
            lambda data: bodo.hiframes.pd_series_ext.init_series(data, None, None),
            [out_data],
            pre_nodes=nodes,
        )

    def _run_unary(self, assign, rhs):
        arg = rhs.value
        typ = self.typemap[arg.name]

        if isinstance(typ, SeriesType):
            assert rhs.fn in bodo.hiframes.pd_series_ext.series_unary_ops
            overload_func = bodo.hiframes.series_impl.create_unary_op_overload(rhs.fn)
            impl = overload_func(typ)
            return self._replace_func(impl, [arg])

        if isinstance(typ, IntegerArrayType):
            assert rhs.fn in (operator.neg, operator.invert, operator.pos)
            overload_func = bodo.libs.int_arr_ext.create_op_overload(rhs.fn, 1)
            impl = overload_func(typ)
            return self._replace_func(impl, [arg])

        if typ == boolean_array:
            assert rhs.fn in (operator.neg, operator.invert, operator.pos)
            overload_func = bodo.libs.bool_arr_ext.create_op_overload(rhs.fn, 1)
            impl = overload_func(typ)
            return self._replace_func(impl, [arg])

        return [assign]

    def _run_call(self, assign, lhs, rhs):
        fdef = guard(find_callname, self.func_ir, rhs, self.typemap)
        if fdef is None:
            from numba.stencil import StencilFunc

            # could be make_function from list comprehension which is ok
            func_def = guard(get_definition, self.func_ir, rhs.func)
            if isinstance(func_def, ir.Expr) and func_def.op == "make_function":
                return [assign]
            if isinstance(func_def, ir.Global) and isinstance(
                func_def.value, StencilFunc
            ):
                return [assign]
            if isinstance(func_def, ir.Const):
                return self._run_const_call(assign, lhs, rhs, func_def.value)
            warnings.warn("function call couldn't be found for initial analysis")
            return [assign]
        else:
            func_name, func_mod = fdef

        # support call ufuncs on Series
        if (
            func_mod in ("numpy", "ufunc")
            and func_name in ufunc_names
            and any(isinstance(self.typemap[a.name], SeriesType) for a in rhs.args)
        ):
            return self._handle_ufuncs(func_name, rhs.args)

        # inline ufuncs on IntegerArray
        if (
            func_mod in ("numpy", "ufunc")
            and func_name in ufunc_names
            and any(
                isinstance(self.typemap[a.name], IntegerArrayType) for a in rhs.args
            )
        ):
            return self._handle_ufuncs_int_arr(func_name, rhs.args)

        # inline ufuncs on BooleanArray
        if (
            func_mod in ("numpy", "ufunc")
            and func_name in ufunc_names
            and any(self.typemap[a.name] == boolean_array for a in rhs.args)
        ):
            return self._handle_ufuncs_bool_arr(func_name, rhs.args)

        if fdef == ("apply_null_mask", "bodo.libs.int_arr_ext"):
            in_typs = tuple(self.typemap[a.name] for a in rhs.args)
            impl = bodo.libs.int_arr_ext.apply_null_mask.py_func(*in_typs)
            return self._replace_func(impl, rhs.args)

        if fdef == ("merge_bitmaps", "bodo.libs.int_arr_ext"):
            in_typs = tuple(self.typemap[a.name] for a in rhs.args)
            impl = bodo.libs.int_arr_ext.merge_bitmaps.py_func(*in_typs)
            return self._replace_func(impl, rhs.args)

        if fdef == ("set_cmp_out_for_nan", "bodo.libs.bool_arr_ext"):
            in_typs = tuple(self.typemap[a.name] for a in rhs.args)
            impl = bodo.libs.bool_arr_ext.overload_set_cmp_out_for_nan(*in_typs)
            return self._replace_func(impl, rhs.args)

        if fdef == ("get_int_arr_data", "bodo.libs.int_arr_ext"):
            var_def = guard(get_definition, self.func_ir, rhs.args[0])
            call_def = guard(find_callname, self.func_ir, var_def)
            if call_def == ("init_integer_array", "bodo.libs.int_arr_ext"):
                assign.value = var_def.args[0]
                return [assign]

        if fdef == ("get_int_arr_bitmap", "bodo.libs.int_arr_ext"):
            var_def = guard(get_definition, self.func_ir, rhs.args[0])
            call_def = guard(find_callname, self.func_ir, var_def)
            if call_def == ("init_integer_array", "bodo.libs.int_arr_ext"):
                assign.value = var_def.args[1]
                return [assign]

        if fdef == ("get_bool_arr_data", "bodo.libs.bool_arr_ext"):
            var_def = guard(get_definition, self.func_ir, rhs.args[0])
            call_def = guard(find_callname, self.func_ir, var_def)
            if call_def == ("init_bool_array", "bodo.libs.bool_arr_ext"):
                assign.value = var_def.args[0]
                return [assign]

        if fdef == ("get_bool_arr_bitmap", "bodo.libs.bool_arr_ext"):
            var_def = guard(get_definition, self.func_ir, rhs.args[0])
            call_def = guard(find_callname, self.func_ir, var_def)
            if call_def == ("init_bool_array", "bodo.libs.bool_arr_ext"):
                assign.value = var_def.args[1]
                return [assign]

        # inline IntegerArrayType.copy()
        if (
            isinstance(func_mod, ir.Var)
            and isinstance(self.typemap[func_mod.name], IntegerArrayType)
            and func_name in ("copy", "astype")
        ):
            rhs.args.insert(0, func_mod)
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}

            impl = getattr(bodo.libs.int_arr_ext, "overload_int_arr_" + func_name)(
                *arg_typs, **kw_typs
            )
            return self._replace_func(
                impl, rhs.args, pysig=numba.utils.pysignature(impl), kws=dict(rhs.kws)
            )

        # inline BooleanArray.copy()
        if (
            isinstance(func_mod, ir.Var)
            and self.typemap[func_mod.name] == boolean_array
            and func_name in ("copy", "astype")
        ):
            rhs.args.insert(0, func_mod)
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}

            impl = getattr(bodo.libs.bool_arr_ext, "overload_bool_arr_" + func_name)(
                *arg_typs, **kw_typs
            )
            return self._replace_func(
                impl, rhs.args, pysig=numba.utils.pysignature(impl), kws=dict(rhs.kws)
            )

        if isinstance(func_mod, ir.Var) and isinstance(
            self.typemap[func_mod.name], SeriesStrMethodType
        ):
            rhs.args.insert(0, func_mod)
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}

            if func_name in bodo.hiframes.pd_series_ext.str2str_methods:
                impl = bodo.hiframes.series_str_impl.create_str2str_methods_overload(
                    func_name
                )(self.typemap[func_mod.name])
            elif func_name in bodo.hiframes.pd_series_ext.str2bool_methods:
                impl = bodo.hiframes.series_str_impl.create_str2bool_methods_overload(
                    func_name
                )(self.typemap[func_mod.name])
            else:
                impl = getattr(
                    bodo.hiframes.series_str_impl, "overload_str_method_" + func_name
                )(*arg_typs, **kw_typs)
            return self._replace_func(
                impl, rhs.args, pysig=numba.utils.pysignature(impl), kws=dict(rhs.kws)
            )

        if fdef == ("cat_array_to_int", "bodo.hiframes.pd_categorical_ext"):
            assign.value = rhs.args[0]
            return [assign]

        # replace _get_type_max_value(arr.dtype) since parfors
        # arr.dtype transformation produces invalid code for dt64
        # TODO: min
        if fdef == ("_get_type_max_value", "bodo.transforms.series_pass"):
            if self.typemap[rhs.args[0].name] == types.DType(types.NPDatetime("ns")):
                return self._replace_func(
                    lambda: bodo.hiframes.pd_timestamp_ext.integer_to_dt64(
                        numba.targets.builtins.get_type_max_value(numba.types.uint64)
                    ),
                    [],
                )
            return self._replace_func(
                lambda d: numba.targets.builtins.get_type_max_value(d), rhs.args
            )

        if fdef == ("h5_read_dummy", "bodo.io.h5_api"):
            ndim = guard(find_const, self.func_ir, rhs.args[1])
            dtype_str = guard(find_const, self.func_ir, rhs.args[2])
            index_var = rhs.args[3]
            index_tp = self.typemap[index_var.name]
            # index is either a single value (e.g. slice) or a tuple (e.g. slices)
            index_types = (
                index_tp.types if isinstance(index_tp, types.BaseTuple) else [index_tp]
            )
            filter_read = False

            # check index types
            for i, t in enumerate(index_types):
                if i == 0 and t == types.Array(types.bool_, 1, "C"):
                    filter_read = True
                else:
                    assert (
                        isinstance(t, types.SliceType) and t.has_step == False
                    ) or isinstance(
                        t, types.Integer
                    ), "only simple slice without step supported for reading hdf5"

            func_text = "def _h5_read_impl(dset_id, ndim, dtype_str, index):\n"

            # get array size and start/count of slices
            for i in range(ndim):
                if i == 0 and filter_read:
                    # TODO: check index format for this case
                    assert isinstance(self.typemap[index_var.name], types.BaseTuple)
                    func_text += "  read_indices = bodo.io.h5_api.get_filter_read_indices(index{})\n".format(
                        "[0]" if isinstance(index_tp, types.BaseTuple) else ""
                    )
                    func_text += "  start_0 = 0\n"
                    func_text += "  size_0 = len(read_indices)\n"
                else:
                    func_text += "  start_{0} = 0\n".format(i)
                    func_text += "  size_{0} = bodo.io.h5_api.h5size(dset_id, np.int32({0}))\n".format(
                        i
                    )
                    if i < len(index_types):
                        if isinstance(index_types[i], types.SliceType):
                            func_text += "  slice_idx_{0} = numba.unicode._normalize_slice(index{1}, size_{0})\n".format(
                                i,
                                "[{}]".format(i)
                                if isinstance(index_tp, types.BaseTuple)
                                else "",
                            )
                            func_text += "  start_{0} = slice_idx_{0}.start\n".format(i)
                            func_text += "  size_{0} = numba.unicode._slice_span(slice_idx_{0})\n".format(
                                i
                            )
                        else:
                            assert isinstance(index_types[i], types.Integer)
                            func_text += "  start_{0} = index{1}\n".format(
                                i,
                                "[{}]".format(i)
                                if isinstance(index_tp, types.BaseTuple)
                                else "",
                            )
                            func_text += "  size_{0} = 1\n".format(i)

            # array dimensions can be less than dataset due to integer selection
            func_text += "  arr_shape = ({},)\n".format(
                ", ".join(
                    [
                        "size_{}".format(i)
                        for i in range(ndim)
                        if not (
                            i < len(index_types)
                            and isinstance(index_types[i], types.Integer)
                        )
                    ]
                )
            )
            func_text += "  A = np.empty(arr_shape, np.{})\n".format(dtype_str)

            func_text += "  start_tup = ({},)\n".format(
                ", ".join(["start_{}".format(i) for i in range(ndim)])
            )
            func_text += "  count_tup = ({},)\n".format(
                ", ".join(["size_{}".format(i) for i in range(ndim)])
            )

            if filter_read:
                func_text += "  err = bodo.io.h5_api.h5read_filter(dset_id, np.int32({}), start_tup, count_tup, 0, A, read_indices)\n".format(
                    ndim
                )
            else:
                func_text += "  err = bodo.io.h5_api.h5read(dset_id, np.int32({}), start_tup, count_tup, 0, A)\n".format(
                    ndim
                )
            func_text += "  return A\n"
            # print(func_text)

            loc_vars = {}
            exec(func_text, {}, loc_vars)
            _h5_read_impl = loc_vars["_h5_read_impl"]
            return self._replace_func(_h5_read_impl, rhs.args)

        if fdef == ("DatetimeIndex", "pandas"):
            return self._run_pd_DatetimeIndex(assign, assign.target, rhs)

        if fdef == ("TimedeltaIndex", "pandas"):
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}
            impl = bodo.hiframes.pd_index_ext.pd_timedelta_index_overload(
                *arg_typs, **kw_typs
            )
            return self._replace_func(
                impl, rhs.args, pysig=self.calltypes[rhs].pysig, kws=dict(rhs.kws)
            )

        if fdef == ("Int64Index", "pandas"):
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}
            impl = bodo.hiframes.pd_index_ext.create_numeric_constructor(
                pd.Int64Index, np.int64
            )(*arg_typs, **kw_typs)
            return self._replace_func(
                impl, rhs.args, pysig=self.calltypes[rhs].pysig, kws=dict(rhs.kws)
            )

        if fdef == ("UInt64Index", "pandas"):
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}
            impl = bodo.hiframes.pd_index_ext.create_numeric_constructor(
                pd.UInt64Index, np.uint64
            )(*arg_typs, **kw_typs)
            return self._replace_func(
                impl, rhs.args, pysig=self.calltypes[rhs].pysig, kws=dict(rhs.kws)
            )

        if fdef == ("Float64Index", "pandas"):
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}
            impl = bodo.hiframes.pd_index_ext.create_numeric_constructor(
                pd.Float64Index, np.float64
            )(*arg_typs, **kw_typs)
            return self._replace_func(
                impl, rhs.args, pysig=self.calltypes[rhs].pysig, kws=dict(rhs.kws)
            )

        if fdef == ("Series", "pandas"):
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}

            impl = bodo.hiframes.pd_series_ext.pd_series_overload(*arg_typs, **kw_typs)
            return self._replace_func(
                impl, rhs.args, pysig=self.calltypes[rhs].pysig, kws=dict(rhs.kws)
            )

        if fdef == ("concat", "bodo.libs.array_kernels"):
            # concat() case where tuple type changes by to_const_type()
            if any([a.name in self._type_changed_vars for a in rhs.args]):
                argtyps = tuple(self.typemap[a.name] for a in rhs.args)
                old_sig = self.calltypes.pop(rhs)
                new_sig = self.typemap[rhs.func.name].get_call_type(
                    self.typingctx, argtyps, dict(rhs.kws)
                )
                self.calltypes[rhs] = new_sig

            # replace tuple of Series with tuple of Arrays
            in_vars, _ = guard(find_build_sequence, self.func_ir, rhs.args[0])
            nodes = []
            s_arrs = [
                self._get_series_data(v, nodes)
                if isinstance(self.typemap[v.name], SeriesType)
                else v
                for v in in_vars
            ]
            loc = assign.target.loc
            scope = assign.target.scope
            new_tup = ir.Expr.build_tuple(s_arrs, loc)
            new_arg = ir.Var(
                scope, mk_unique_var(rhs.args[0].name + "_arrs"), loc
            )
            self.typemap[new_arg.name] = self.typemap[rhs.args[0].name]
            nodes.append(ir.Assign(new_tup, new_arg, loc))
            rhs.args[0] = new_arg
            nodes.append(assign)
            self.calltypes.pop(rhs)
            new_sig = self.typemap[rhs.func.name].get_call_type(
                self.typingctx, (self.typemap[new_arg.name],), dict(rhs.kws)
            )
            self.calltypes[rhs] = new_sig
            return nodes

        # replace isna early to enable more optimization in PA
        # TODO: handle more types
        if fdef == ("isna", "bodo.libs.array_kernels"):
            arr = rhs.args[0]
            ind = rhs.args[1]
            arr_typ = self.typemap[arr.name]
            if isinstance(arr_typ, types.Array):
                if isinstance(arr_typ.dtype, types.Float):
                    func = lambda arr, i: np.isnan(arr[i])
                    return self._replace_func(func, [arr, ind])
                elif isinstance(arr_typ.dtype, (types.NPDatetime, types.NPTimedelta)):
                    nat = arr_typ.dtype("NaT")
                    # TODO: replace with np.isnat
                    return self._replace_func(lambda arr, i: arr[i] == nat, [arr, ind])
                elif isinstance(arr_typ.dtype, types.Integer):
                    return self._replace_func(lambda arr, i: False, [arr, ind])
            return [assign]

        if fdef == ("argsort", "bodo.hiframes.series_impl"):
            lhs = assign.target
            data = rhs.args[0]
            nodes = []

            def _get_indices(S):  # pragma: no cover
                n = len(S)
                return np.arange(n)

            nodes += compile_func_single_block(_get_indices, (data,), None, self)
            index_var = nodes[-1].target

            # dummy output data arrays for results
            out_data = ir.Var(lhs.scope, mk_unique_var(data.name + "_data"), lhs.loc)
            self.typemap[out_data.name] = self.typemap[data.name]

            in_df = {"inds": index_var}
            out_df = {"inds": lhs}
            in_keys = [data]
            out_keys = [out_data]
            ascending = True

            # Sort node
            nodes.append(
                bodo.ir.sort.Sort(
                    data.name,
                    lhs.name,
                    in_keys,
                    out_keys,
                    in_df,
                    out_df,
                    False,
                    lhs.loc,
                    ascending,
                )
            )

            return nodes

        if fdef == ("sort", "bodo.hiframes.series_impl"):
            lhs = assign.target
            data = rhs.args[0]
            index_arr = rhs.args[1]
            ascending = find_const(self.func_ir, rhs.args[2])
            inplace = find_const(self.func_ir, rhs.args[3])

            nodes = []

            out_data = ir.Var(lhs.scope, mk_unique_var(data.name + "_data"), lhs.loc)
            self.typemap[out_data.name] = self.typemap[data.name]
            out_index = ir.Var(
                lhs.scope, mk_unique_var(index_arr.name + "_index"), lhs.loc
            )
            self.typemap[out_index.name] = self.typemap[index_arr.name]

            in_df = {"inds": index_arr}
            out_df = {"inds": out_index}
            in_keys = [data]
            out_keys = [out_data]

            # Sort node
            nodes.append(
                bodo.ir.sort.Sort(
                    data.name,
                    lhs.name,
                    in_keys,
                    out_keys,
                    in_df,
                    out_df,
                    inplace,
                    lhs.loc,
                    ascending,
                )
            )

            return nodes + compile_func_single_block(
                lambda A, B: (A, B), (out_data, out_index), lhs, self
            )

        if fdef == ("to_numeric", "bodo.hiframes.series_impl"):
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            impl = bodo.hiframes.series_impl.to_numeric_overload(*arg_typs)
            return self._replace_func(impl, rhs.args)

        if fdef == ("series_filter_bool", "bodo.hiframes.series_impl"):
            nodes = []
            in_arr = rhs.args[0]
            bool_arr = rhs.args[1]
            if is_series_type(self.typemap[in_arr.name]):
                in_arr = self._get_series_data(in_arr, nodes)
            if is_series_type(self.typemap[bool_arr.name]):
                bool_arr = self._get_series_data(bool_arr, nodes)

            return self._replace_func(
                series_kernels._column_filter_impl, [in_arr, bool_arr], pre_nodes=nodes
            )

        if fdef == ("get_itertuples", "bodo.hiframes.dataframe_impl"):
            nodes = []
            new_args = []
            for arg in rhs.args:
                if isinstance(self.typemap[arg.name], SeriesType):
                    new_args.append(self._get_series_data(arg, nodes))
                else:
                    new_args.append(arg)

            self._convert_series_calltype(rhs)
            rhs.args = new_args

            nodes.append(assign)
            return nodes

        if fdef == ("get_series_data_tup", "bodo.hiframes.pd_series_ext"):
            arg = rhs.args[0]
            impl = bodo.hiframes.pd_series_ext.overload_get_series_data_tup(
                self.typemap[arg.name]
            )
            return compile_func_single_block(impl, (arg,), assign.target, self)

        if fdef == ("get_index_data", "bodo.hiframes.pd_index_ext"):
            var_def = guard(get_definition, self.func_ir, rhs.args[0])
            call_def = guard(find_callname, self.func_ir, var_def)
            if call_def in (
                ("init_datetime_index", "bodo.hiframes.pd_index_ext"),
                ("init_timedelta_index", "bodo.hiframes.pd_index_ext"),
                ("init_string_index", "bodo.hiframes.pd_index_ext"),
                ("init_numeric_index", "bodo.hiframes.pd_index_ext"),
            ):
                assign.value = var_def.args[0]
            return [assign]

        if fdef == ("get_index_name", "bodo.hiframes.pd_index_ext"):
            var_def = guard(get_definition, self.func_ir, rhs.args[0])
            call_def = guard(find_callname, self.func_ir, var_def)
            if (
                call_def
                in (
                    ("init_datetime_index", "bodo.hiframes.pd_index_ext"),
                    ("init_timedelta_index", "bodo.hiframes.pd_index_ext"),
                    ("init_string_index", "bodo.hiframes.pd_index_ext"),
                    ("init_numeric_index", "bodo.hiframes.pd_index_ext"),
                )
                and len(var_def.args) > 1
            ):
                assign.value = var_def.args[1]
            return [assign]

        # pd.DataFrame() calls init_series for even Series since it's untyped
        # remove the call since it is invalid for analysis here
        # XXX remove when df pass is typed? (test_pass_series2)
        if fdef == ("init_series", "bodo.hiframes.pd_series_ext"):
            if isinstance(self.typemap[rhs.args[0].name], SeriesType):
                assign.value = rhs.args[0]
            return [assign]

        if fdef == ("get_series_data", "bodo.hiframes.pd_series_ext"):
            # TODO: make sure data is not altered using update_series_data()
            # or other functions, using any reference to payload
            var_def = guard(get_definition, self.func_ir, rhs.args[0])
            call_def = guard(find_callname, self.func_ir, var_def)
            if call_def == ("init_series", "bodo.hiframes.pd_series_ext"):
                assign.value = var_def.args[0]
            return [assign]

        if fdef == ("get_series_index", "bodo.hiframes.pd_series_ext"):
            # TODO: make sure index is not altered using update_series_index()
            # or other functions, using any reference to payload
            var_def = guard(get_definition, self.func_ir, rhs.args[0])
            call_def = guard(find_callname, self.func_ir, var_def)
            if (
                call_def == ("init_series", "bodo.hiframes.pd_series_ext")
                and len(var_def.args) > 1
            ):
                assign.value = var_def.args[1]
            return [assign]

        if fdef == ("get_series_name", "bodo.hiframes.pd_series_ext"):
            # TODO: make sure name is not altered
            var_def = guard(get_definition, self.func_ir, rhs.args[0])
            call_def = guard(find_callname, self.func_ir, var_def)
            if (
                call_def == ("init_series", "bodo.hiframes.pd_series_ext")
                and len(var_def.args) > 2
            ):
                assign.value = var_def.args[2]
            return [assign]

        if fdef == ("update_series_data", "bodo.hiframes.pd_series_ext"):
            return [assign]

        if fdef == ("update_series_index", "bodo.hiframes.pd_series_ext"):
            return [assign]

        if func_mod == "bodo.hiframes.rolling":
            return self._run_call_rolling(assign, assign.target, rhs, func_name)

        if fdef == ("empty_like", "numpy"):
            return self._handle_empty_like(assign, lhs, rhs)

        if fdef == ("alloc_type", "bodo.utils.utils"):
            typ = self.typemap[rhs.args[1].name].instance_type
            if typ.dtype == bodo.hiframes.pd_timestamp_ext.datetime_date_type:
                impl = lambda n, t: bodo.hiframes.datetime_date_ext.np_arr_to_array_datetime_date(
                    np.empty(n, np.int64)
                )
            elif isinstance(typ, IntegerArrayType):
                impl = lambda n, t: bodo.libs.int_arr_ext.init_integer_array(
                    np.empty(n, _dtype), np.empty((n + 7) >> 3, np.uint8)
                )
            elif typ == boolean_array:
                impl = lambda n, t: bodo.libs.bool_arr_ext.init_bool_array(
                    np.empty(n, _dtype), np.empty((n + 7) >> 3, np.uint8)
                )
            else:
                impl = lambda n, t: np.empty(n, _dtype)
            return compile_func_single_block(
                impl, rhs.args, assign.target, self, extra_globals={"_dtype": typ.dtype}
            )

        if isinstance(func_mod, ir.Var) and is_series_type(self.typemap[func_mod.name]):
            return self._run_call_series(
                assign, assign.target, rhs, func_mod, func_name
            )

        if isinstance(func_mod, ir.Var) and isinstance(
            self.typemap[func_mod.name], SeriesRollingType
        ):
            return self._run_call_series_rolling(
                assign, assign.target, rhs, func_mod, func_name
            )

        if isinstance(func_mod, ir.Var) and isinstance(
            self.typemap[func_mod.name], DatetimeIndexType
        ):
            rhs.args.insert(0, func_mod)
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}
            if func_name == "min":
                impl = bodo.hiframes.pd_index_ext.overload_datetime_index_min(
                    *arg_typs, **kw_typs
                )
            else:
                assert func_name == "max"
                impl = bodo.hiframes.pd_index_ext.overload_datetime_index_max(
                    *arg_typs, **kw_typs
                )
            stub = lambda dti, axis=None, skipna=True: None
            return self._replace_func(
                impl, rhs.args, pysig=numba.utils.pysignature(stub), kws=dict(rhs.kws)
            )

        if isinstance(func_mod, ir.Var) and bodo.hiframes.pd_index_ext.is_pd_index_type(
            self.typemap[func_mod.name]
        ):
            return self._run_call_index(assign, assign.target, rhs, func_mod, func_name)

        if fdef == ("concat_dummy", "bodo.hiframes.pd_dataframe_ext") and isinstance(
            self.typemap[lhs], SeriesType
        ):
            return self._run_call_concat(assign, lhs, rhs)

        # handle sorted() with key lambda input
        if fdef == ("sorted", "builtins") and "key" in dict(rhs.kws):
            return self._handle_sorted_by_key(rhs)

        if fdef == ("init_dataframe", "bodo.hiframes.pd_dataframe_ext"):
            return [assign]

        if fdef == ("len", "builtins") and is_series_type(
            self.typemap[rhs.args[0].name]
        ):
            return self._replace_func(
                lambda S: len(bodo.hiframes.pd_series_ext.get_series_data(S)), rhs.args
            )

        # XXX sometimes init_dataframe() can't be resolved in dataframe_pass
        # and there are get_dataframe_data() calls that could be optimized
        # example: test_sort_parallel
        if fdef == ("get_dataframe_data", "bodo.hiframes.pd_dataframe_ext"):
            df_var = rhs.args[0]
            ind = guard(find_const, self.func_ir, rhs.args[1])
            var_def = guard(get_definition, self.func_ir, df_var)
            call_def = guard(find_callname, self.func_ir, var_def)
            if call_def == ("init_dataframe", "bodo.hiframes.pd_dataframe_ext"):
                seq_info = guard(find_build_sequence, self.func_ir, var_def.args[0])
                assert seq_info is not None
                assign.value = seq_info[0][ind]

        if fdef == ("get_dataframe_index", "bodo.hiframes.pd_dataframe_ext"):
            df_var = rhs.args[0]
            var_def = guard(get_definition, self.func_ir, df_var)
            call_def = guard(find_callname, self.func_ir, var_def)
            if call_def == ("init_dataframe", "bodo.hiframes.pd_dataframe_ext"):
                assign.value = var_def.args[1]

        if fdef == ("to_const_tuple", "bodo.utils.typing"):
            tup = rhs.args[0]
            tup_items = self._get_const_tup(tup)
            new_tup = ir.Expr.build_tuple(tup_items, tup.loc)
            assign.value = new_tup
            # fix type and definition of lhs
            self.typemap.pop(lhs)
            self._type_changed_vars.append(lhs)
            self.typemap[lhs] = types.Tuple(
                tuple(self.typemap[a.name] for a in tup_items)
            )
            return [assign]

        # inline conversion functions to enable optimization
        if func_mod == "bodo.utils.conversion" and func_name != "flatten_array":
            # TODO: use overload IR inlining when available
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}
            overload_func = getattr(bodo.utils.conversion, "overload_" + func_name)
            impl = overload_func(*arg_typs, **kw_typs)
            return self._replace_func(
                impl, rhs.args, pysig=self.calltypes[rhs].pysig, kws=dict(rhs.kws)
            )

        if func_mod == "bodo.libs.distributed_api" and func_name in (
            "dist_return",
            "threaded_return",
        ):
            return [assign]

        if fdef == ("val_isin_dummy", "bodo.hiframes.pd_dataframe_ext"):
            def impl(S, vals):
                arr = bodo.hiframes.pd_series_ext.get_series_data(S)
                numba.parfor.init_prange()
                n = len(arr)
                out = np.empty(n, np.bool_)
                for i in numba.parfor.internal_prange(n):
                    out[i] = arr[i] in vals
                return bodo.hiframes.pd_series_ext.init_series(out)
            return self._replace_func(impl, rhs.args)

        if fdef == ("val_notin_dummy", "bodo.hiframes.pd_dataframe_ext"):
            def impl(S, vals):
                arr = bodo.hiframes.pd_series_ext.get_series_data(S)
                numba.parfor.init_prange()
                n = len(arr)
                out = np.empty(n, np.bool_)
                for i in numba.parfor.internal_prange(n):
                    # TODO: why don't these work?
                    # out[i] = (arr[i] not in vals)
                    # out[i] = not (arr[i] in vals)
                    _in = (arr[i] in vals)
                    out[i] = not _in
                return bodo.hiframes.pd_series_ext.init_series(out)
            # import pdb; pdb.set_trace()
            return self._replace_func(impl, rhs.args)

        # inline np.where() for 3 arg case with 1D input
        if (fdef == ("where", "numpy")
                or fdef == ("where_impl", "bodo.hiframes.series_impl")) and (
                len(rhs.args) == 3 and self.typemap[rhs.args[0].name].ndim == 1):
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}
            impl = bodo.hiframes.series_impl.overload_np_where(*arg_typs, **kw_typs)
            return self._replace_func(
                impl, rhs.args, pysig=self.calltypes[rhs].pysig, kws=dict(rhs.kws)
            )

        # convert Series to Array for unhandled calls
        # TODO check all the functions that get here and handle if necessary
        # e.g. np.sum, prod, min, max, argmin, argmax, mean, var, and std
        if any(isinstance(self.typemap[arg.name], SeriesType) for arg in rhs.args):
            return self._fix_unhandled_calls(assign, lhs, rhs)

        return [assign]

    def _fix_unhandled_calls(self, assign, lhs, rhs):
        # TODO: test
        nodes = []
        new_args = []
        series_arg = None
        for arg in rhs.args:
            if isinstance(self.typemap[arg.name], SeriesType):
                new_args.append(self._get_series_data(arg, nodes))
                series_arg = arg
            else:
                new_args.append(arg)

        self._convert_series_calltype(rhs)
        rhs.args = new_args
        if isinstance(self.typemap[lhs], SeriesType):
            scope = assign.target.scope
            new_lhs = ir.Var(scope, mk_unique_var(lhs + "_data"), rhs.loc)
            self.typemap[new_lhs.name] = self.calltypes[rhs].return_type
            nodes.append(ir.Assign(rhs, new_lhs, rhs.loc))
            index = self._get_series_index(series_arg, nodes)
            name = self._get_series_name(series_arg, nodes)
            return self._replace_func(
                lambda A, index, name: bodo.hiframes.pd_series_ext.init_series(A, index, name),
                (new_lhs, index, name),
                pre_nodes=nodes,
            )
        else:
            nodes.append(assign)
            return nodes

    def _run_const_call(self, assign, lhs, rhs, func):
        # replace direct calls to operators with Expr binop nodes to enable
        # ParallelAccelerator transformtions

        # inline bool arr operators
        if any(self.typemap[a.name] == boolean_array for a in rhs.args):
            n_args = len(rhs.args)
            overload_func = bodo.libs.bool_arr_ext.create_op_overload(func, n_args)
            impl = overload_func(*tuple(self.typemap[a.name] for a in rhs.args))
            return self._replace_func(impl, rhs.args)

        if func in bodo.hiframes.pd_series_ext.series_binary_ops:
            expr = ir.Expr.binop(func, rhs.args[0], rhs.args[1], rhs.loc)
            self.calltypes[expr] = self.calltypes[rhs]
            return self._run_binop(ir.Assign(expr, assign.target, rhs.loc), expr)

        if func in bodo.hiframes.pd_series_ext.series_inplace_binary_ops:
            # TODO: test
            imm_fn = bodo.hiframes.pd_series_ext.inplace_binop_to_imm[func]
            expr = ir.Expr.inplace_binop(
                func, imm_fn, rhs.args[0], rhs.args[1], rhs.loc
            )
            self.calltypes[expr] = self.calltypes[rhs]
            return [ir.Assign(expr, assign.target, rhs.loc)]

        # TODO: this fails test_series_unary_op with pos for some reason
        if func in bodo.hiframes.pd_series_ext.series_unary_ops:
            expr = ir.Expr.unary(func, rhs.args[0], rhs.loc)
            self.calltypes[expr] = self.calltypes[rhs]
            return [ir.Assign(expr, assign.target, rhs.loc)]

        # TODO: handle other calls
        return [assign]

    def _handle_ufuncs(self, ufunc_name, args):
        """hanlde ufuncs with any Series in arguments.
        Output is Series using index and name of original Series.
        """
        np_ufunc = getattr(np, ufunc_name)
        if np_ufunc.nin == 1:

            def impl(S):
                arr = bodo.hiframes.pd_series_ext.get_series_data(S)
                index = bodo.hiframes.pd_series_ext.get_series_index(S)
                name = bodo.hiframes.pd_series_ext.get_series_name(S)
                out_arr = _ufunc(arr)
                return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

            # impl.__globals__['_ufunc'] = np_ufunc
            return self._replace_func(impl, args, extra_globals={"_ufunc": np_ufunc})
        elif np_ufunc.nin == 2:
            if isinstance(self.typemap[args[0].name], SeriesType):

                def impl(S1, S2):
                    arr = bodo.hiframes.pd_series_ext.get_series_data(S1)
                    index = bodo.hiframes.pd_series_ext.get_series_index(S1)
                    name = bodo.hiframes.pd_series_ext.get_series_name(S1)
                    other_arr = bodo.utils.conversion.get_array_if_series_or_index(S2)
                    out_arr = _ufunc(arr, other_arr)
                    return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

                return self._replace_func(
                    impl, args, extra_globals={"_ufunc": np_ufunc}
                )
            else:
                assert isinstance(self.typemap[args[1].name], SeriesType)

                def impl(S1, S2):
                    arr = bodo.utils.conversion.get_array_if_series_or_index(S1)
                    other_arr = bodo.hiframes.pd_series_ext.get_series_data(S2)
                    index = bodo.hiframes.pd_series_ext.get_series_index(S2)
                    name = bodo.hiframes.pd_series_ext.get_series_name(S2)
                    out_arr = _ufunc(arr, other_arr)
                    return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

                return self._replace_func(
                    impl, args, extra_globals={"_ufunc": np_ufunc}
                )
        else:
            raise ValueError("Unsupported numpy ufunc {}".format(ufunc_name))

    def _handle_ufuncs_int_arr(self, ufunc_name, args):
        np_ufunc = getattr(np, ufunc_name)
        overload_func = bodo.libs.int_arr_ext.create_op_overload(np_ufunc, np_ufunc.nin)
        in_typs = tuple(self.typemap[a.name] for a in args)
        impl = overload_func(*in_typs)
        return self._replace_func(impl, args)

    def _handle_ufuncs_bool_arr(self, ufunc_name, args):
        np_ufunc = getattr(np, ufunc_name)
        overload_func = bodo.libs.bool_arr_ext.create_op_overload(
            np_ufunc, np_ufunc.nin
        )
        in_typs = tuple(self.typemap[a.name] for a in args)
        impl = overload_func(*in_typs)
        return self._replace_func(impl, args)

    def _run_call_series(self, assign, lhs, rhs, series_var, func_name):
        if func_name in (
            "sum",
            "prod",
            "mean",
            "var",
            "std",
            "cumsum",
            "cumprod",
            "abs",
            "count",
            "unique",
            "get_values",
            "to_numpy",
            "min",
            "max",
            "median",
            "idxmin",
            "idxmax",
            "rename",
            "corr",
            "cov",
            "nunique",
            "describe",
            "put",
            "isna",
            "isnull",
            "quantile",
            "fillna",
            "dropna",
            "shift",
            "pct_change",
            "nlargest",
            "notna",
            "nsmallest",
            "head",
            "tail",
            "argsort",
            "sort_values",
            "take",
            "append",
            "copy",
        ):
            if func_name == "isnull":
                func_name = "isna"
            rhs.args.insert(0, series_var)
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}
            overload_func = getattr(
                bodo.hiframes.series_impl, "overload_series_" + func_name
            )
            impl = overload_func(*arg_typs, **kw_typs)
            return self._replace_func(
                impl, rhs.args, pysig=numba.utils.pysignature(impl), kws=dict(rhs.kws)
            )

        if func_name in bodo.hiframes.series_impl.explicit_binop_funcs.values():
            rhs.args.insert(0, series_var)
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}
            op = getattr(operator, func_name)
            overload_func = bodo.hiframes.series_impl.create_explicit_binary_op_overload(
                op
            )
            impl = overload_func(*arg_typs, **kw_typs)
            return self._replace_func(
                impl, rhs.args, pysig=numba.utils.pysignature(impl), kws=dict(rhs.kws)
            )

        if func_name == "rolling":
            # XXX: remove rolling setup call, assuming still available in definitions
            self.func_ir._definitions[lhs.name].append(rhs)
            return []

        if func_name == "combine":
            return self._handle_series_combine(assign, lhs, rhs, series_var)

        if func_name in ("map", "apply"):
            kws = dict(rhs.kws)
            if func_name == "apply":
                func_var = self._get_arg("apply", rhs.args, kws, 0, "func")
            else:
                func_var = self._get_arg("map", rhs.args, kws, 0, "arg")
            return self._handle_series_map(assign, lhs, rhs, series_var, func_var)

        if func_name == "value_counts":
            nodes = []
            data = self._get_series_data(series_var, nodes)
            name = self._get_series_name(series_var, nodes)
            # reusing aggregate/count
            # TODO: write optimized implementation
            # data of input becomes both key and data for aggregate input
            # data of output is the counts
            out_key_var = ir.Var(lhs.scope, mk_unique_var(lhs.name + "_index"), lhs.loc)
            self.typemap[out_key_var.name] = self.typemap[data.name]
            out_data_var = ir.Var(lhs.scope, mk_unique_var(lhs.name + "_data"), lhs.loc)
            self.typemap[out_data_var.name] = self.typemap[lhs.name].data
            agg_func = series_replace_funcs["count"]
            agg_func.ftype = bodo.ir.aggregate.supported_agg_funcs.index("count")
            agg_func.builtin = True
            agg_node = bodo.ir.aggregate.Aggregate(
                lhs.name,
                "series",
                ["series"],
                [out_key_var],
                {"data": out_data_var},
                {"data": data},
                [data],
                agg_func,
                lhs.loc,
            )
            nodes.append(agg_node)
            # TODO: handle args like sort=False
            func = lambda A, B, name: bodo.hiframes.pd_series_ext.init_series(
                A, bodo.utils.conversion.convert_to_index(B), name
            ).sort_values(ascending=False)
            return self._replace_func(
                func, [out_data_var, out_key_var, name], pre_nodes=nodes
            )

        # astype with string output
        if func_name == "astype":
            # just return input if both input/output are strings
            # TODO: removing this opt causes a crash in test_series_astype_str
            # TODO: copy if not packed string array
            if is_str_series_typ(self.typemap[lhs.name]) and is_str_series_typ(
                self.typemap[series_var.name]
            ):
                return self._replace_func(lambda a: a, [series_var])

            rhs.args.insert(0, series_var)
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}
            overload_func = getattr(
                bodo.hiframes.series_impl, "overload_series_" + func_name
            )
            impl = overload_func(*arg_typs, **kw_typs)
            stub = lambda S, dtype, copy=True, errors="raise": None
            return self._replace_func(
                impl, rhs.args, pysig=numba.utils.pysignature(stub), kws=dict(rhs.kws)
            )

        # functions we revert to Numpy for now, otherwise warning
        # TODO: fix distributed cumsum/cumprod
        # TODO: handle series-specific cases for this funcs
        if not func_name.startswith("values."):
            warnings.warn(
                "unknown Series call {}, reverting to Numpy".format(func_name)
            )

        return [assign]

    def _run_call_series_fillna(self, assign, lhs, rhs, series_var):
        dtype = self.typemap[series_var.name].dtype
        val = rhs.args[0]
        nodes = []
        data = self._get_series_data(series_var, nodes)
        index = self._get_series_index(series_var, nodes)
        name = self._get_series_name(series_var, nodes)
        kws = dict(rhs.kws)
        inplace = False
        if "inplace" in kws:
            inplace = guard(find_const, self.func_ir, kws["inplace"])
            if inplace == None:  # pragma: no cover
                raise ValueError("inplace arg to fillna should be constant")

        if inplace:
            if dtype == string_type:
                # optimization: just set null bit if fill is empty
                if guard(find_const, self.func_ir, val) == "":
                    return self._replace_func(
                        lambda A: bodo.libs.str_arr_ext.set_null_bits(A),
                        [data],
                        pre_nodes=nodes,
                    )
                # Since string arrays can't be changed, we have to create a new
                # array and assign it back to the same Series variable
                # result back to the same variable
                # TODO: handle string array reflection
                fill_var = rhs.args[0]
                assign.target = series_var  # replace output
                return self._replace_func(
                    series_kernels._series_fillna_str_alloc_impl,
                    (data, fill_var, index, name), pre_nodes=nodes
                )
            else:
                return self._replace_func(
                    series_kernels._column_fillna_impl,
                    [data, data, val],
                    pre_nodes=nodes,
                )
        else:
            if dtype == string_type:
                func = series_replace_funcs["fillna_str_alloc"]
            else:
                func = series_replace_funcs["fillna_alloc"]
            return self._replace_func(func, [data, val, index, name], pre_nodes=nodes)

    def _run_call_series_dropna(self, assign, lhs, rhs, series_var):
        dtype = self.typemap[series_var.name].dtype
        kws = dict(rhs.kws)
        inplace = False
        if "inplace" in kws:
            inplace = guard(find_const, self.func_ir, kws["inplace"])
            if inplace == None:  # pragma: no cover
                raise ValueError("inplace arg to dropna should be constant")

        nodes = []
        data = self._get_series_data(series_var, nodes)
        name = self._get_series_name(series_var, nodes)
        if dtype == string_type:
            func = series_replace_funcs["dropna_str_alloc"]
        elif isinstance(dtype, types.Float):
            func = series_replace_funcs["dropna_float"]
        else:
            # integer case, TODO: bool, date etc.
            func = lambda A, name: bodo.hiframes.pd_series_ext.init_series(A, None, name)

        if inplace:
            # Since arrays can't resize inplace, we have to create a new
            # array and assign it back to the same Series variable
            # result back to the same variable
            assign.target = series_var  # replace output

        return self._replace_func(func, [data, name], pre_nodes=nodes)

    def _handle_series_map(self, assign, lhs, rhs, series_var, func_var):
        """translate df.A.map(lambda a:...) to prange()
        """
        func = guard(get_definition, self.func_ir, func_var)
        if func is None or not (
            isinstance(func, ir.Expr) and func.op == "make_function"
        ):
            raise ValueError("lambda for map not found")

        dtype = self.typemap[series_var.name].dtype
        nodes = []
        data = self._get_series_data(series_var, nodes)
        index = self._get_series_index(series_var, nodes)
        name = self._get_series_name(series_var, nodes)
        out_typ = self.typemap[lhs.name].dtype

        # TODO: handle all array types like list(str)
        if out_typ == string_type:
            # prange func to inline
            func_text = "def f(A, index, name):\n"
            func_text += "  numba.parfor.init_prange()\n"
            func_text += "  n = len(A)\n"
            func_text += "  n_chars = 0\n"
            func_text += "  for i in numba.parfor.internal_prange(n):\n"
            if dtype == types.NPDatetime("ns"):
                func_text += "    t = bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(np.int64(A[i]))\n"
            elif isinstance(dtype, types.BaseTuple):
                func_text += "    t = bodo.utils.typing.convert_rec_to_tup(A[i])\n"
            else:
                func_text += "    t = A[i]\n"
            func_text += "    n_chars += get_utf8_size(map_func(t))\n"
            func_text += "  S = pre_alloc_string_array(n, n_chars)\n"
            func_text += "  for i in numba.parfor.internal_prange(n):\n"
            if dtype == types.NPDatetime("ns"):
                func_text += "    t = bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(np.int64(A[i]))\n"
            elif isinstance(dtype, types.BaseTuple):
                func_text += "    t = bodo.utils.typing.convert_rec_to_tup(A[i])\n"
            else:
                func_text += "    t = A[i]\n"
            func_text += "    v = map_func(t)\n"
            func_text += "    S[i] = v\n"
            # func_text += "    print(S[i])\n"
            func_text += "  return bodo.hiframes.pd_series_ext.init_series(S, index, name)\n"
        else:
            func_text = "def f(A, index, name):\n"
            func_text += "  numba.parfor.init_prange()\n"
            func_text += "  n = len(A)\n"
            func_text += "  S = np.empty(n, out_dtype)\n"
            func_text += "  for i in numba.parfor.internal_prange(n):\n"
            if dtype == types.NPDatetime("ns"):
                func_text += "    t = bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(np.int64(A[i]))\n"
            elif isinstance(dtype, types.BaseTuple):
                func_text += "    t = bodo.utils.typing.convert_rec_to_tup(A[i])\n"
            else:
                func_text += "    t = A[i]\n"
            func_text += "    v = map_func(t)\n"
            if isinstance(out_typ, types.BaseTuple):
                func_text += "    S[i] = bodo.utils.typing.convert_tup_to_rec(v)\n"
            else:
                func_text += "    S[i] = v\n"
            # func_text += "    print(S[i])\n"
            func_text += "  return bodo.hiframes.pd_series_ext.init_series(S, index, name)\n"

        loc_vars = {}
        exec(func_text, {}, loc_vars)
        f = loc_vars["f"]

        out_dtype = self.typemap[lhs.name].dtype
        if isinstance(out_dtype, types.BaseTuple):
            out_dtype = np.dtype(",".join(str(t) for t in out_dtype.types), align=True)

        _globals = self.func_ir.func_id.func.__globals__
        f_ir = compile_to_numba_ir(
            f,
            {"numba": numba, "np": np, "pd": pd, "bodo": bodo, "out_dtype": out_dtype,
                            "get_utf8_size": get_utf8_size,
                "pre_alloc_string_array": pre_alloc_string_array,
            },
        )

        # fix definitions to enable finding sentinel
        f_ir._definitions = build_definitions(f_ir.blocks)
        topo_order = find_topo_order(f_ir.blocks)

        # find sentinel function and replace with user func
        for l in topo_order:
            block = f_ir.blocks[l]
            for i, stmt in enumerate(block.body):
                if (
                    isinstance(stmt, ir.Assign)
                    and isinstance(stmt.value, ir.Expr)
                    and stmt.value.op == "call"
                ):
                    fdef = guard(get_definition, f_ir, stmt.value.func)
                    if isinstance(fdef, ir.Global) and fdef.name == "map_func":
                        inline_closure_call(f_ir, _globals, block, i, func)

        # remove sentinel global to avoid type inference issues
        ir_utils.remove_dead(f_ir.blocks, f_ir.arg_names, f_ir)
        f_ir._definitions = build_definitions(f_ir.blocks)
        arg_typs = (
            self.typemap[data.name],
            self.typemap[index.name],
            self.typemap[name.name],
        )
        f_typemap, _f_ret_t, f_calltypes = numba.typed_passes.type_inference_stage(
            self.typingctx, f_ir, arg_typs, self.typemap[lhs.name]
        )
        # remove argument entries like arg.a from typemap
        arg_names = [vname for vname in f_typemap if vname.startswith("arg.")]
        for a in arg_names:
            f_typemap.pop(a)
        self.typemap.update(f_typemap)
        self.calltypes.update(f_calltypes)
        replace_arg_nodes(f_ir.blocks[topo_order[0]], [data, index, name])
        f_ir.blocks[topo_order[0]].body = nodes + f_ir.blocks[topo_order[0]].body
        return f_ir.blocks

    def _run_call_index(self, assign, lhs, rhs, index_var, func_name):
        if func_name in ("isna", "take"):
            if func_name == "isnull":
                func_name = "isna"
            rhs.args.insert(0, index_var)
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}
            overload_func = getattr(
                bodo.hiframes.pd_index_ext, "overload_index_" + func_name
            )
            impl = overload_func(*arg_typs, **kw_typs)
            return self._replace_func(
                impl, rhs.args, pysig=numba.utils.pysignature(impl), kws=dict(rhs.kws)
            )

    def _run_call_rolling(self, assign, lhs, rhs, func_name):
        # replace input arguments with data arrays from Series
        nodes = []
        new_args = []
        for arg in rhs.args:
            if isinstance(self.typemap[arg.name], SeriesType):
                new_args.append(self._get_series_data(arg, nodes))
            else:
                new_args.append(arg)

        self._convert_series_calltype(rhs)
        rhs.args = new_args

        if func_name == "rolling_corr":

            def rolling_corr_impl(arr, other, win, center):
                cov = bodo.hiframes.rolling.rolling_cov(arr, other, win, center)
                a_std = bodo.hiframes.rolling.rolling_fixed(
                    arr, win, center, False, "std"
                )
                b_std = bodo.hiframes.rolling.rolling_fixed(
                    other, win, center, False, "std"
                )
                return cov / (a_std * b_std)

            return self._replace_func(rolling_corr_impl, rhs.args, pre_nodes=nodes)
        if func_name == "rolling_cov":

            def rolling_cov_impl(arr, other, w, center):  # pragma: no cover
                ddof = 1
                X = arr.astype(np.float64)
                Y = other.astype(np.float64)
                XpY = X + Y
                XtY = X * Y
                count = bodo.hiframes.rolling.rolling_fixed(
                    XpY, w, center, False, "count"
                )
                mean_XtY = bodo.hiframes.rolling.rolling_fixed(
                    XtY, w, center, False, "mean"
                )
                mean_X = bodo.hiframes.rolling.rolling_fixed(
                    X, w, center, False, "mean"
                )
                mean_Y = bodo.hiframes.rolling.rolling_fixed(
                    Y, w, center, False, "mean"
                )
                bias_adj = count / (count - ddof)
                return (mean_XtY - mean_X * mean_Y) * bias_adj

            return self._replace_func(rolling_cov_impl, rhs.args, pre_nodes=nodes)
        # replace apply function with dispatcher obj, now the type is known
        if func_name == "rolling_fixed" and isinstance(
            self.typemap[rhs.args[4].name], types.MakeFunctionLiteral
        ):
            # for apply case, create a dispatcher for the kernel and pass it
            # TODO: automatically handle lambdas in Numba
            dtype = self.typemap[rhs.args[0].name].dtype
            out_dtype = self.typemap[lhs.name].dtype
            func_node = guard(get_definition, self.func_ir, rhs.args[4])
            imp_dis = self._handle_rolling_apply_func(func_node, dtype, out_dtype)

            def f(arr, w, center):  # pragma: no cover
                return bodo.hiframes.rolling.rolling_fixed(arr, w, center, False, _func)

            return nodes + compile_func_single_block(
                f, rhs.args[:-2], lhs, self, extra_globals={"_func": imp_dis}
            )
        elif func_name == "rolling_variable" and isinstance(
            self.typemap[rhs.args[5].name], types.MakeFunctionLiteral
        ):
            # for apply case, create a dispatcher for the kernel and pass it
            # TODO: automatically handle lambdas in Numba
            dtype = self.typemap[rhs.args[0].name].dtype
            out_dtype = self.typemap[lhs.name].dtype
            func_node = guard(get_definition, self.func_ir, rhs.args[5])
            imp_dis = self._handle_rolling_apply_func(func_node, dtype, out_dtype)

            def f(arr, on_arr, w, center):  # pragma: no cover
                return bodo.hiframes.rolling.rolling_variable(
                    arr, on_arr, w, center, False, _func
                )

            return nodes + compile_func_single_block(
                f, rhs.args[:-2], lhs, self, extra_globals={"_func": imp_dis}
            )

        nodes.append(assign)
        return nodes

    def _handle_series_combine(self, assign, lhs, rhs, series_var):
        """translate s1.combine(s2, lambda x1,x2 :...) to prange()
        """
        kws = dict(rhs.kws)
        other_var = self._get_arg("combine", rhs.args, kws, 0, "other")
        func_var = self._get_arg("combine", rhs.args, kws, 1, "func")
        fill_var = self._get_arg("combine", rhs.args, kws, 2, "fill_value", default="")

        func = guard(get_definition, self.func_ir, func_var)
        if func is None or not (
            isinstance(func, ir.Expr) and func.op == "make_function"
        ):
            raise ValueError("lambda for combine not found")

        nodes = []
        data = self._get_series_data(series_var, nodes)
        index = self._get_series_index(series_var, nodes)
        name = self._get_series_name(series_var, nodes)
        other_data = self._get_series_data(other_var, nodes)

        # Use NaN if fill_value is not provided
        use_nan = fill_var is "" or self.typemap[fill_var.name] == types.none

        # prange func to inline
        if use_nan:
            func_text = "def f(A, B, index, name):\n"
        else:
            func_text = "def f(A, B, C, index, name):\n"
        func_text += "  n1 = len(A)\n"
        func_text += "  n2 = len(B)\n"
        func_text += "  n = max(n1, n2)\n"
        if not isinstance(self.typemap[series_var.name].dtype, types.Float) and use_nan:
            func_text += "  assert n1 == n, 'can not use NAN for non-float series, with different length'\n"
        if not isinstance(self.typemap[other_var.name].dtype, types.Float) and use_nan:
            func_text += "  assert n2 == n, 'can not use NAN for non-float series, with different length'\n"
        func_text += "  numba.parfor.init_prange()\n"
        func_text += "  S = np.empty(n, out_dtype)\n"
        func_text += "  for i in numba.parfor.internal_prange(n):\n"
        if use_nan and isinstance(self.typemap[series_var.name].dtype, types.Float):
            func_text += "    t1 = np.nan\n"
            func_text += "    if i < n1:\n"
            func_text += "      t1 = A[i]\n"
        # length is equal, due to assertion above
        elif use_nan:
            func_text += "    t1 = A[i]\n"
        else:
            func_text += "    t1 = C\n"
            func_text += "    if i < n1:\n"
            func_text += "      t1 = A[i]\n"
        # same, but for 2nd argument
        if use_nan and isinstance(self.typemap[other_var.name].dtype, types.Float):
            func_text += "    t2 = np.nan\n"
            func_text += "    if i < n2:\n"
            func_text += "      t2 = B[i]\n"
        elif use_nan:
            func_text += "    t2 = B[i]\n"
        else:
            func_text += "    t2 = C\n"
            func_text += "    if i < n2:\n"
            func_text += "      t2 = B[i]\n"
        func_text += "    S[i] = map_func(t1, t2)\n"
        # TODO: Pandas combine ignores name for some reason!
        func_text += "  return bodo.hiframes.pd_series_ext.init_series(S, index, None)\n"

        loc_vars = {}
        exec(func_text, {}, loc_vars)
        f = loc_vars["f"]

        _globals = self.func_ir.func_id.func.__globals__
        f_ir = compile_to_numba_ir(
            f,
            {
                "numba": numba,
                "np": np,
                "pd": pd,
                "bodo": bodo,
                "out_dtype": self.typemap[lhs.name].dtype,
            },
        )

        # fix definitions to enable finding sentinel
        f_ir._definitions = build_definitions(f_ir.blocks)
        topo_order = find_topo_order(f_ir.blocks)

        # find sentinel function and replace with user func
        for l in topo_order:
            block = f_ir.blocks[l]
            for i, stmt in enumerate(block.body):
                if (
                    isinstance(stmt, ir.Assign)
                    and isinstance(stmt.value, ir.Expr)
                    and stmt.value.op == "call"
                ):
                    fdef = guard(get_definition, f_ir, stmt.value.func)
                    if isinstance(fdef, ir.Global) and fdef.name == "map_func":
                        inline_closure_call(f_ir, _globals, block, i, func)
                        break

        # remove sentinel global to avoid type inference issues
        ir_utils.remove_dead(f_ir.blocks, f_ir.arg_names, f_ir)
        f_ir._definitions = build_definitions(f_ir.blocks)
        arg_typs = (self.typemap[data.name], self.typemap[other_data.name])
        if not use_nan:
            arg_typs += (self.typemap[fill_var.name],)
        arg_typs += (self.typemap[index.name], self.typemap[name.name])
        f_typemap, _f_ret_t, f_calltypes = numba.typed_passes.type_inference_stage(
            self.typingctx, f_ir, arg_typs, self.typemap[lhs.name]
        )
        # remove argument entries like arg.a from typemap
        arg_names = [vname for vname in f_typemap if vname.startswith("arg.")]
        for a in arg_names:
            f_typemap.pop(a)
        self.typemap.update(f_typemap)
        self.calltypes.update(f_calltypes)
        func_args = [data, other_data]
        if not use_nan:
            func_args.append(fill_var)
        func_args += [index, name]
        first_block = f_ir.blocks[topo_order[0]]
        replace_arg_nodes(first_block, func_args)
        first_block.body = nodes + first_block.body
        return f_ir.blocks

    def _run_call_series_rolling(self, assign, lhs, rhs, rolling_var, func_name):
        """
        Handle Series rolling calls like:
          A = df.column.rolling(3).sum()
        """
        rolling_call = guard(get_definition, self.func_ir, rolling_var)
        assert isinstance(rolling_call, ir.Expr) and rolling_call.op == "call"
        call_def = guard(get_definition, self.func_ir, rolling_call.func)
        assert isinstance(call_def, ir.Expr) and call_def.op == "getattr"
        series_var = call_def.value
        nodes = []
        data = self._get_series_data(series_var, nodes)
        index = self._get_series_index(series_var, nodes)
        name = self._get_series_name(series_var, nodes)

        window, center, _on = get_rolling_setup_args(self.func_ir, rolling_call, False)
        if not isinstance(center, ir.Var):
            center_var = ir.Var(lhs.scope, mk_unique_var("center"), lhs.loc)
            self.typemap[center_var.name] = types.bool_
            nodes.append(ir.Assign(ir.Const(center, lhs.loc), center_var, lhs.loc))
            center = center_var

        if func_name in ("cov", "corr"):
            # TODO: variable window
            if len(rhs.args) == 1:
                other = self._get_series_data(rhs.args[0], nodes)
            else:
                other = data
            if func_name == "cov":
                f = lambda a, b, w, c, i, n: bodo.hiframes.pd_series_ext.init_series(
                    bodo.hiframes.rolling.rolling_cov(a, b, w, c), i, n
                )
            if func_name == "corr":
                f = lambda a, b, w, c, i, n: bodo.hiframes.pd_series_ext.init_series(
                    bodo.hiframes.rolling.rolling_corr(a, b, w, c), i, n
                )
            return self._replace_func(
                f, [data, other, window, center, index, name], pre_nodes=nodes
            )
        elif func_name == "apply":
            func_node = guard(get_definition, self.func_ir, rhs.args[0])
            dtype = self.typemap[data.name].dtype
            out_dtype = self.typemap[lhs.name].dtype
            func_global = self._handle_rolling_apply_func(func_node, dtype, out_dtype)
        else:
            func_global = func_name

        def f(arr, w, center, index, name):  # pragma: no cover
            return bodo.hiframes.pd_series_ext.init_series(
                bodo.hiframes.rolling.rolling_fixed(arr, w, center, False, _func),
                index,
                name,
            )

        args = [data, window, center, index, name]
        return self._replace_func(
            f, args, pre_nodes=nodes, extra_globals={"_func": func_global}
        )

    def _handle_rolling_apply_func(self, func_node, dtype, out_dtype):
        if func_node is None:
            raise ValueError("cannot find kernel function for rolling.apply() call")
        # TODO: more error checking on the kernel to make sure it doesn't
        # use global/closure variables
        if func_node.closure is not None:
            raise ValueError(
                "rolling apply kernel functions cannot have closure variables"
            )
        if func_node.defaults is not None:
            raise ValueError(
                "rolling apply kernel functions cannot have default arguments"
            )
        # create a function from the code object
        glbs = self.func_ir.func_id.func.__globals__
        lcs = {}
        exec("def f(A): return A", glbs, lcs)
        kernel_func = lcs["f"]
        kernel_func.__code__ = func_node.code
        kernel_func.__name__ = func_node.code.co_name
        # use bodo's sequential pipeline to enable pandas operations
        # XXX seq pipeline used since dist pass causes a hang
        m = numba.ir_utils._max_label
        impl_disp = numba.njit(
            kernel_func, pipeline_class=bodo.compiler.BodoCompilerSeq
        )
        # precompile to avoid REP counting conflict in testing
        sig = out_dtype(types.Array(dtype, 1, "C"))
        impl_disp.compile(sig)
        numba.ir_utils._max_label += m
        return impl_disp

    def _run_pd_DatetimeIndex(self, assign, lhs, rhs):
        """transform pd.DatetimeIndex() call with string array argument
        """
        arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
        kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}
        impl = bodo.hiframes.pd_index_ext.pd_datetimeindex_overload(
            *arg_typs, **kw_typs
        )
        return self._replace_func(
            impl, rhs.args, pysig=self.calltypes[rhs].pysig, kws=dict(rhs.kws)
        )

    def _is_dt_index_binop(self, rhs):
        if rhs.op != "binop":
            return False

        if rhs.fn not in _dt_index_binops:
            return False

        arg1, arg2 = self.typemap[rhs.lhs.name], self.typemap[rhs.rhs.name]
        # one of them is dt_index but not both
        if (is_dt64_series_typ(arg1) or is_dt64_series_typ(arg2)) and not (
            is_dt64_series_typ(arg1) and is_dt64_series_typ(arg2)
        ):
            return True

        if (
            isinstance(arg1, DatetimeIndexType) or isinstance(arg2, DatetimeIndexType)
        ) and not (
            isinstance(arg1, DatetimeIndexType) and isinstance(arg2, DatetimeIndexType)
        ):
            return True

        return False

    def _handle_dt_index_binop(self, assign, rhs):
        arg1, arg2 = rhs.lhs, rhs.rhs

        def _is_allowed_type(t):
            return is_dt64_series_typ(t) or t in (string_type, types.NPDatetime("ns"))

        # TODO: this has to be more generic to support all combinations.
        if (
            is_dt64_series_typ(self.typemap[arg1.name])
            and self.typemap[arg2.name]
            == bodo.hiframes.pd_timestamp_ext.pandas_timestamp_type
            and rhs.fn in ("-", operator.sub)
        ):
            return self._replace_func(
                series_kernels._column_sub_impl_datetime_series_timestamp, [arg1, arg2]
            )

        if not _is_allowed_type(
            types.unliteral(self.typemap[arg1.name])
        ) or not _is_allowed_type(types.unliteral(self.typemap[arg2.name])):
            raise ValueError("Series(dt64) operation not supported")

        # string comparison with Series(dt64)
        op_str = _binop_to_str[rhs.fn]
        typ1 = types.unliteral(self.typemap[arg1.name])
        typ2 = types.unliteral(self.typemap[arg2.name])
        nodes = []

        func_text = "def f(arg1, arg2):\n"
        if is_dt64_series_typ(typ1):
            arg1 = self._get_series_data(arg1, nodes)
            func_text += "  dt_index, _str = arg1, arg2\n"
            comp = "dt_index[i] {} other".format(op_str)
            other_typ = typ2
        else:
            arg2 = self._get_series_data(arg2, nodes)
            func_text += "  dt_index, _str = arg2, arg1\n"
            comp = "other {} dt_index[i]".format(op_str)
            other_typ = typ1
        func_text += "  l = len(dt_index)\n"
        if other_typ == string_type:
            func_text += (
                "  other = bodo.hiframes.pd_timestamp_ext.parse_datetime_str(_str)\n"
            )
        else:
            func_text += "  other = _str\n"
        func_text += "  S = np.empty(l, dtype=np.bool_)\n"
        func_text += "  nulls = np.empty((l + 7) >> 3, dtype=np.uint8)\n"
        func_text += "  for i in numba.parfor.internal_prange(l):\n"
        func_text += "    S[i] = {}\n".format(comp)
        func_text += "    bodo.libs.int_arr_ext.set_bit_to_arr(nulls, i, 1)\n"
        func_text += "  return bodo.hiframes.pd_series_ext.init_series(bodo.libs.bool_arr_ext.init_bool_array(S, nulls))\n"
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        f = loc_vars["f"]
        # print(func_text)
        return self._replace_func(f, [arg1, arg2], pre_nodes=nodes)

    def _handle_string_array_expr(self, assign, rhs):
        # convert str_arr==str into parfor
        if (
            rhs.fn in _string_array_comp_ops
            and is_str_arr_typ(self.typemap[rhs.lhs.name])
            or is_str_arr_typ(self.typemap[rhs.rhs.name])
        ):
            nodes = []
            arg1 = rhs.lhs
            arg2 = rhs.rhs
            is_series = False
            if is_str_series_typ(self.typemap[arg1.name]):
                arg1 = self._get_series_data(arg1, nodes)
                is_series = True
            if is_str_series_typ(self.typemap[arg2.name]):
                arg2 = self._get_series_data(arg2, nodes)
                is_series = True

            if not is_series:
                # just inline string array ops
                # TODO: remove when binop inlining is supported
                f = bodo.libs.str_arr_ext.create_binary_op_overload(rhs.fn)(
                    self.typemap[rhs.lhs.name], self.typemap[rhs.rhs.name]
                )
                return self._replace_func(f, [arg1, arg2], pre_nodes=nodes)

            arg1_access = "A"
            arg2_access = "B"
            len_call = "len(A)"
            if is_str_arr_typ(self.typemap[arg1.name]):
                arg1_access = "A[i]"
                # replace type now for correct typing of len, etc.
                self.typemap.pop(arg1.name)
                self.typemap[arg1.name] = string_array_type

            if is_str_arr_typ(self.typemap[arg2.name]):
                arg1_access = "B[i]"
                len_call = "len(B)"
                self.typemap.pop(arg2.name)
                self.typemap[arg2.name] = string_array_type

            op_str = _binop_to_str[rhs.fn]

            func_text = "def f(A, B):\n"
            func_text += "  l = {}\n".format(len_call)
            func_text += "  S = np.empty(l, dtype=np.bool_)\n"
            if is_series:
                func_text += "  nulls = np.empty((l + 7) >> 3, dtype=np.uint8)\n"
            func_text += "  for i in numba.parfor.internal_prange(l):\n"
            func_text += "    S[i] = {} {} {}\n".format(
                arg1_access, op_str, arg2_access
            )
            # TODO: proper NAs
            if is_series:
                func_text += "    bodo.libs.int_arr_ext.set_bit_to_arr(nulls, i, 1)\n"
                func_text += "  return bodo.hiframes.pd_series_ext.init_series(bodo.libs.bool_arr_ext.init_bool_array(S, nulls))\n"
            else:
                func_text += "  return S\n"

            loc_vars = {}
            exec(func_text, {}, loc_vars)
            f = loc_vars["f"]
            return self._replace_func(f, [arg1, arg2], pre_nodes=nodes)

        return None

    def _handle_empty_like(self, assign, lhs, rhs):
        # B = empty_like(A) -> B = empty(len(A), dtype)
        in_arr = rhs.args[0]

        if self.typemap[in_arr.name].ndim == 1:
            # generate simpler len() for 1D case
            def f(_in_arr):  # pragma: no cover
                _alloc_size = len(_in_arr)
                _out_arr = np.empty(_alloc_size, _in_arr.dtype)

        else:

            def f(_in_arr):  # pragma: no cover
                _alloc_size = _in_arr.shape
                _out_arr = np.empty(_alloc_size, _in_arr.dtype)

        f_block = compile_to_numba_ir(
            f,
            {"np": np},
            self.typingctx,
            (if_series_to_array_type(self.typemap[in_arr.name]),),
            self.typemap,
            self.calltypes,
        ).blocks.popitem()[1]
        replace_arg_nodes(f_block, [in_arr])
        nodes = f_block.body[:-3]  # remove none return
        nodes[-1].target = assign.target
        return nodes

    def _run_call_concat(self, assign, lhs, rhs):
        nodes = []
        series_list = guard(get_definition, self.func_ir, rhs.args[0]).items
        arrs = [self._get_series_data(v, nodes) for v in series_list]
        arr_tup = ir.Var(rhs.args[0].scope, mk_unique_var("arr_tup"), rhs.args[0].loc)
        self.typemap[arr_tup.name] = types.Tuple([self.typemap[a.name] for a in arrs])
        tup_expr = ir.Expr.build_tuple(arrs, arr_tup.loc)
        nodes.append(ir.Assign(tup_expr, arr_tup, arr_tup.loc))
        # TODO: index and name
        return self._replace_func(
            lambda arr_list: bodo.hiframes.pd_series_ext.init_series(
                bodo.libs.array_kernels.concat(arr_list)
            ),
            [arr_tup],
            pre_nodes=nodes,
        )

    def _handle_h5_write(self, dset, index, arr):
        if index != slice(None):
            raise ValueError("Only HDF5 write of full array supported")
        assert isinstance(self.typemap[arr.name], types.Array)
        ndim = self.typemap[arr.name].ndim

        func_text = "def _h5_write_impl(dset_id, arr):\n"
        func_text += "  zero_tup = ({},)\n".format(", ".join(["0"] * ndim))
        # TODO: remove after support arr.shape in parallel
        func_text += "  arr_shape = ({},)\n".format(
            ", ".join(["arr.shape[{}]".format(i) for i in range(ndim)])
        )
        func_text += "  err = bodo.io.h5_api.h5write(dset_id, np.int32({}), zero_tup, arr_shape, 0, arr)\n".format(
            ndim
        )

        loc_vars = {}
        exec(func_text, {}, loc_vars)
        _h5_write_impl = loc_vars["_h5_write_impl"]
        return compile_func_single_block(_h5_write_impl, (dset, arr), None, self)

    def _handle_sorted_by_key(self, rhs):
        """generate a sort function with the given key lambda
        """
        # TODO: handle reverse
        from numba.targets import quicksort

        # get key lambda
        key_lambda_var = dict(rhs.kws)["key"]
        key_lambda = guard(get_definition, self.func_ir, key_lambda_var)
        if key_lambda is None or not (
            isinstance(key_lambda, ir.Expr) and key_lambda.op == "make_function"
        ):
            raise ValueError("sorted(): lambda for key not found")

        # wrap lambda in function
        def key_lambda_wrapper(A):
            return A

        key_lambda_wrapper.__code__ = key_lambda.code
        key_func = numba.njit(key_lambda_wrapper)

        # make quicksort with new lt
        def lt(a, b):
            return key_func(a) < key_func(b)

        sort_func = quicksort.make_jit_quicksort(lt=lt).run_quicksort

        return self._replace_func(
            lambda a: _sort_func(a),
            rhs.args,
            extra_globals={"_sort_func": numba.njit(sort_func)},
        )

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

    def _get_index_data(self, dt_var, nodes):
        var_def = guard(get_definition, self.func_ir, dt_var)
        call_def = guard(find_callname, self.func_ir, var_def)
        if call_def in (
            ("init_datetime_index", "bodo.hiframes.pd_index_ext"),
            ("init_timedelta_index", "bodo.hiframes.pd_index_ext"),
            ("init_string_index", "bodo.hiframes.pd_index_ext"),
            ("init_numeric_index", "bodo.hiframes.pd_index_ext"),
        ):
            return var_def.args[0]

        nodes += compile_func_single_block(
            lambda S: bodo.hiframes.pd_index_ext.get_index_data(S), (dt_var,), None, self
        )
        return nodes[-1].target

    def _get_series_data(self, series_var, nodes):
        # optimization: return data var directly if series has a single
        # definition by init_series()
        # e.g. S = init_series(A, None)
        # XXX assuming init_series() is the only call to create a series
        # and series._data is never overwritten
        var_def = guard(get_definition, self.func_ir, series_var)
        call_def = guard(find_callname, self.func_ir, var_def)
        if call_def == ("init_series", "bodo.hiframes.pd_series_ext"):
            return var_def.args[0]

        # XXX use get_series_data() for getting data instead of S._data
        # to enable alias analysis
        nodes += compile_func_single_block(
            lambda S: bodo.hiframes.pd_series_ext.get_series_data(S), (series_var,), None, self
        )
        return nodes[-1].target

    def _get_series_index(self, series_var, nodes):
        # XXX assuming init_series is the only call to create a series
        # and series._index is never overwritten
        var_def = guard(get_definition, self.func_ir, series_var)
        call_def = guard(find_callname, self.func_ir, var_def)
        if call_def == ("init_series", "bodo.hiframes.pd_series_ext") and (
            len(var_def.args) >= 2 and not self._is_const_none(var_def.args[1])
        ):
            return var_def.args[1]

        # XXX use get_series_index() for getting data instead of S._index
        # to enable alias analysis
        nodes += compile_func_single_block(
            lambda S: bodo.hiframes.pd_series_ext.get_series_index(S), (series_var,), None, self
        )
        return nodes[-1].target

    def _get_series_name(self, series_var, nodes):
        var_def = guard(get_definition, self.func_ir, series_var)
        call_def = guard(find_callname, self.func_ir, var_def)
        if call_def == ("init_series", "bodo.hiframes.pd_series_ext") and len(var_def.args) == 3:
            return var_def.args[2]

        nodes += compile_func_single_block(
            lambda S: bodo.hiframes.pd_series_ext.get_series_name(S), (series_var,), None, self
        )
        return nodes[-1].target

    def _replace_func(
        self,
        func,
        args,
        const=False,
        pre_nodes=None,
        extra_globals=None,
        pysig=None,
        kws=None,
    ):
        glbls = {"numba": numba, "np": np, "bodo": bodo, "pd": pd}
        if extra_globals is not None:
            glbls.update(extra_globals)
        func.__globals__.update(glbls)

        # create explicit arg variables for defaults if func has any
        # XXX: inine_closure_call() can't handle defaults properly
        if pysig is not None:
            pre_nodes = [] if pre_nodes is None else pre_nodes
            scope = next(iter(self.func_ir.blocks.values())).scope
            loc = scope.loc

            def normal_handler(index, param, default):
                return default

            def default_handler(index, param, default):
                d_var = ir.Var(scope, mk_unique_var("defaults"), loc)
                self.typemap[d_var.name] = numba.typeof(default)
                node = ir.Assign(ir.Const(default, loc), d_var, loc)
                pre_nodes.append(node)
                return d_var

            # TODO: stararg needs special handling?
            args = numba.typing.fold_arguments(
                pysig, args, kws, normal_handler, default_handler, normal_handler
            )

        arg_typs = tuple(self.typemap[v.name] for v in args)

        if const:
            new_args = []
            for i, arg in enumerate(args):
                val = guard(find_const, self.func_ir, arg)
                if val:
                    new_args.append(types.literal(val))
                else:
                    new_args.append(arg_typs[i])
            arg_typs = tuple(new_args)
        return ReplaceFunc(func, arg_typs, args, glbls, pre_nodes)

    def _convert_series_calltype(self, call):
        sig = self.calltypes[call]
        if sig is None:
            return
        assert isinstance(sig, Signature)

        # XXX using replace() since it copies, otherwise cached overload
        # functions fail
        new_sig = sig.replace(return_type=if_series_to_array_type(sig.return_type))
        new_sig.args = tuple(map(if_series_to_array_type, sig.args))

        # XXX: side effect: force update of call signatures
        if isinstance(call, ir.Expr) and call.op == "call":
            # StencilFunc requires kws for typing so sig.args can't be used
            # reusing sig.args since some types become Const in sig
            argtyps = new_sig.args[: len(call.args)]
            kwtyps = {name: self.typemap[v.name] for name, v in call.kws}
            sig = new_sig
            new_sig = self.typemap[call.func.name].get_call_type(
                self.typingctx, argtyps, kwtyps
            )
            # calltypes of things like BoundFunction (array.call) need to
            # be updated for lowering to work
            # XXX: new_sig could be None for things like np.int32()
            if call in self.calltypes and new_sig is not None:
                old_sig = self.calltypes[call]
                # fix types with undefined dtypes in empty_inferred, etc.
                return_type = _fix_typ_undefs(new_sig.return_type, old_sig.return_type)
                args = tuple(
                    _fix_typ_undefs(a, b) for a, b in zip(new_sig.args, old_sig.args)
                )
                new_sig = Signature(return_type, args, new_sig.recvr, new_sig.pysig)

        if new_sig is not None:
            # XXX sometimes new_sig is None for some reason
            # FIXME e.g. test_series_nlargest_parallel1 np.int32()
            self.calltypes.pop(call)
            self.calltypes[call] = new_sig
        return

    def is_bool_arr(self, varname):
        typ = self.typemap[varname]
        return (
            isinstance(if_series_to_array_type(typ), types.Array)
            and typ.dtype == types.bool_
        )

    def _is_const_none(self, var):
        var_def = guard(get_definition, self.func_ir, var)
        return isinstance(var_def, ir.Const) and var_def.value is None

    def _handle_hiframes_nodes(self, inst):
        if isinstance(inst, Aggregate):
            # now that type inference is done, remove type vars to
            # enable dead code elimination
            inst.out_typer_vars = None
            use_vars = inst.key_arrs + list(inst.df_in_vars.values())
            if inst.pivot_arr is not None:
                use_vars.append(inst.pivot_arr)
            def_vars = list(inst.df_out_vars.values())
            if inst.out_key_vars is not None:
                def_vars += inst.out_key_vars
            apply_copies_func = bodo.ir.aggregate.apply_copies_aggregate
        elif isinstance(inst, bodo.ir.sort.Sort):
            use_vars = inst.key_arrs + list(inst.df_in_vars.values())
            def_vars = []
            if not inst.inplace:
                def_vars = inst.out_key_arrs + list(inst.df_out_vars.values())
            apply_copies_func = bodo.ir.sort.apply_copies_sort
        elif isinstance(inst, bodo.ir.join.Join):
            use_vars = list(inst.right_vars.values()) + list(inst.left_vars.values())
            def_vars = list(inst.df_out_vars.values())
            apply_copies_func = bodo.ir.join.apply_copies_join
        elif isinstance(inst, bodo.ir.csv_ext.CsvReader):
            use_vars = []
            def_vars = inst.out_vars
            apply_copies_func = bodo.ir.csv_ext.apply_copies_csv
        else:
            assert isinstance(inst, bodo.ir.filter.Filter)
            use_vars = list(inst.df_in_vars.values())
            if isinstance(self.typemap[inst.bool_arr.name], SeriesType):
                use_vars.append(inst.bool_arr)
            def_vars = list(inst.df_out_vars.values())
            apply_copies_func = bodo.ir.filter.apply_copies_filter

        out_nodes = self._convert_series_hiframes_nodes(
            inst, use_vars, def_vars, apply_copies_func
        )

        return out_nodes

    def _update_definitions(self, node_list):
        loc = ir.Loc("", 0)
        dumm_block = ir.Block(ir.Scope(None, loc), loc)
        dumm_block.body = node_list
        build_definitions({0: dumm_block}, self.func_ir._definitions)
        return

    def _convert_series_hiframes_nodes(
        self, inst, use_vars, def_vars, apply_copies_func
    ):
        #
        out_nodes = []
        varmap = {
            v.name: self._get_series_data(v, out_nodes)
            for v in use_vars
            if isinstance(self.typemap[v.name], SeriesType)
        }
        apply_copies_func(inst, varmap, None, None, None, None)
        out_nodes.append(inst)

        for v in def_vars:
            self.func_ir._definitions[v.name].remove(inst)
        varmap = {}
        for v in def_vars:
            if not isinstance(self.typemap[v.name], SeriesType):
                continue
            data_var = ir.Var(v.scope, mk_unique_var(v.name + "data"), v.loc)
            self.typemap[data_var.name] = series_to_array_type(self.typemap[v.name])
            f_block = compile_to_numba_ir(
                lambda A: bodo.hiframes.pd_series_ext.init_series(A),
                {"bodo": bodo},
                self.typingctx,
                (self.typemap[data_var.name],),
                self.typemap,
                self.calltypes,
            ).blocks.popitem()[1]
            replace_arg_nodes(f_block, [data_var])
            out_nodes += f_block.body[:-2]
            out_nodes[-1].target = v
            varmap[v.name] = data_var

        apply_copies_func(inst, varmap, None, None, None, None)
        return out_nodes

    def _get_arg(self, f_name, args, kws, arg_no, arg_name, default=None, err_msg=None):
        arg = None
        if len(args) > arg_no:
            arg = args[arg_no]
        elif arg_name in kws:
            arg = kws[arg_name]

        if arg is None:
            if default is not None:
                return default
            if err_msg is None:
                err_msg = "{} requires '{}' argument".format(f_name, arg_name)
            raise ValueError(err_msg)
        return arg


def _fix_typ_undefs(new_typ, old_typ):
    if isinstance(old_typ, (types.Array, SeriesType)):
        assert isinstance(
            new_typ,
            (
                types.Array,
                SeriesType,
                StringArrayType,
                types.List,
                StringArraySplitViewType,
            ),
        )
        if new_typ.dtype == types.undefined:
            return new_typ.copy(old_typ.dtype)
    if isinstance(old_typ, types.BaseTuple):
        return types.Tuple(
            [_fix_typ_undefs(t, u) for t, u in zip(new_typ.types, old_typ.types)]
        )
    # TODO: fix List, Set
    return new_typ


def get_stmt_writes(stmt):
    # TODO: test bodo nodes
    writes = set()
    if isinstance(stmt, (ir.Assign, ir.SetItem, ir.StaticSetItem)):
        writes.add(stmt.target.name)
    if isinstance(stmt, Aggregate):
        writes = {v.name for v in stmt.df_out_vars.values()}
        if stmt.out_key_vars is not None:
            writes.update({v.name for v in stmt.out_key_vars})
    if isinstance(stmt, (bodo.ir.csv_ext.CsvReader, bodo.ir.parquet_ext.ParquetReader)):
        writes = {v.name for v in stmt.out_vars}
    if isinstance(stmt, (bodo.ir.filter.Filter, bodo.ir.join.Join)):
        writes = {v.name for v in stmt.df_out_vars.values()}
    if isinstance(stmt, bodo.ir.sort.Sort):
        if not stmt.inplace:
            writes.update({v.name for v in stmt.out_key_arrs})
            writes.update({v.name for v in stmt.df_out_vars.values()})
    return writes


# XXX override stmt write function use in parfor fusion
# TODO: implement support for nodes properly
ir_utils.get_stmt_writes = get_stmt_writes
