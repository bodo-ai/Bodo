"""
converts Series operations to array operations as much as possible
to provide implementation and enable optimization.
"""

import operator
import sys
import warnings

import numba
import numpy as np
import pandas as pd
from numba.core import ir, ir_utils, types
from numba.core.inline_closurecall import inline_closure_call
from numba.core.ir_utils import (
    build_definitions,
    compile_to_numba_ir,
    dprint_func_ir,
    find_callname,
    find_const,
    find_topo_order,
    get_definition,
    guard,
    mk_unique_var,
    replace_arg_nodes,
    require,
)
from numba.core.typing.templates import Signature

import bodo
import bodo.hiframes.series_dt_impl  # noqa # side effect: install Series overloads
import bodo.hiframes.series_impl  # noqa # side effect: install Series overloads
import bodo.hiframes.series_indexing  # noqa # side effect: install Series overloads
import bodo.hiframes.series_str_impl  # noqa # side effect: install Series overloads
from bodo.hiframes import series_kernels
from bodo.hiframes.datetime_date_ext import datetime_date_array_type
from bodo.hiframes.datetime_datetime_ext import datetime_datetime_type
from bodo.hiframes.datetime_timedelta_ext import (
    datetime_timedelta_type,
    timedelta_array_type,
)
from bodo.hiframes.pd_categorical_ext import CategoricalArrayType
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.hiframes.pd_groupby_ext import DataFrameGroupByType
from bodo.hiframes.pd_index_ext import (
    BinaryIndexType,
    CategoricalIndexType,
    DatetimeIndexType,
    HeterogeneousIndexType,
    NumericIndexType,
    PeriodIndexType,
    RangeIndexType,
    StringIndexType,
    TimedeltaIndexType,
    is_index_type,
)
from bodo.hiframes.pd_series_ext import (
    HeterogeneousSeriesType,
    SeriesType,
    if_series_to_array_type,
    is_dt64_series_typ,
    is_series_type,
    is_str_series_typ,
    is_timedelta64_series_typ,
)
from bodo.hiframes.pd_timestamp_ext import timedelta_methods
from bodo.hiframes.series_dt_impl import SeriesDatetimePropertiesType
from bodo.hiframes.series_indexing import (
    HeterogeneousSeriesIlocType,
    SeriesIatType,
    SeriesIlocType,
    SeriesLocType,
)
from bodo.hiframes.series_str_impl import (
    SeriesCatMethodType,
    SeriesStrMethodType,
)
from bodo.hiframes.split_impl import StringArraySplitViewType
from bodo.libs.array_item_arr_ext import ArrayItemArrayType
from bodo.libs.bool_arr_ext import (
    BooleanArrayType,
    boolean_array_type,
    is_valid_boolean_array_logical_op,
)
from bodo.libs.decimal_arr_ext import DecimalArrayType
from bodo.libs.float_arr_ext import FloatingArrayType
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.map_arr_ext import MapArrayType
from bodo.libs.str_arr_ext import StringArrayType, string_array_type
from bodo.libs.str_ext import string_type
from bodo.libs.struct_arr_ext import StructArrayType
from bodo.libs.tuple_arr_ext import TupleArrayType
from bodo.transforms.dataframe_pass import DataFramePass
from bodo.transforms.typing_pass import _get_state_defining_call
from bodo.utils.transform import (
    ReplaceFunc,
    avoid_udf_inline,
    compile_func_single_block,
    extract_keyvals_from_struct_map,
    get_build_sequence_vars,
    get_call_expr_arg,
    replace_func,
    set_2nd_to_last_arg_to_true,
    set_last_arg_to_true,
    update_locs,
)
from bodo.utils.typing import (
    BodoError,
    ColNamesMetaType,
    MetaType,
    get_literal_value,
    get_overload_const_func,
    get_overload_const_int,
    get_overload_const_str,
    get_overload_const_tuple,
    is_bodosql_context_type,
    is_literal_type,
    is_overload_constant_str,
    is_overload_constant_tuple,
    is_overload_false,
    is_overload_none,
    is_str_arr_type,
    unwrap_typeref,
)
from bodo.utils.utils import (
    find_build_tuple,
    gen_getitem,
    get_getsetitem_index_var,
    is_array_typ,
    is_assign,
    is_call,
    is_call_assign,
    is_expr,
    is_whole_slice,
)

ufunc_names = {f.__name__ for f in numba.core.typing.npydecl.supported_ufuncs}


_string_array_comp_ops = (
    operator.eq,
    operator.ne,
    operator.ge,
    operator.gt,
    operator.le,
    operator.lt,
)


class SeriesPass:
    """
    This pass converts Series operations to array operations as much as possible to
    provide implementation and enable optimization.
    """

    def __init__(
        self,
        func_ir,
        typingctx,
        targetctx,
        typemap,
        calltypes,
        _locals,
        optimize_inplace_ops=True,
        avoid_copy_propagation=False,
        parfor_metadata=None,
    ):
        self.func_ir = func_ir
        self.typingctx = typingctx
        self.targetctx = targetctx
        self.typemap = typemap
        self.calltypes = calltypes
        self.locals = _locals
        # DataFrame transformation module to try on each statement
        self.dataframe_pass = DataFramePass(
            func_ir, typingctx, targetctx, typemap, calltypes
        )
        # flag to enable inplace array op optimization: A[i] == v -> inplace_eq(A, i, v)
        self.optimize_inplace_ops = optimize_inplace_ops
        # Loc object of current location being translated
        self.curr_loc = self.func_ir.loc
        # Skip copy propagation. This is used for testing purposes where
        # copy propagation interferes with simple checks of the IR.
        self.avoid_copy_propagation = avoid_copy_propagation
        # Metadata information for parfor used to track user variables
        self.parfor_metadata = parfor_metadata

    def run(self):
        """run series/dataframe transformations"""
        blocks = self.func_ir.blocks
        # topo_order necessary so Series data replacement optimization can be
        # performed in one pass
        topo_order = find_topo_order(blocks)
        # find the potentially updated DataFrames to avoid optimizing out
        # get_dataframe_data() calls incorrectly
        self.dataframe_pass._updated_dataframes = self._get_updated_dataframes(
            blocks, topo_order
        )
        # Keep track of the updated DataFrames already visited on this pass.
        self.dataframe_pass._visited_updated_dataframes = set()

        # NOTE: this is iterating in topological order; we're popping from the reversed ordering.
        work_list = [(l, blocks[l]) for l in reversed(topo_order)]
        changed = False
        while work_list:
            label, block = work_list.pop()
            new_body = []
            replaced = False
            for i, inst in enumerate(block.body):
                if not changed:
                    # Create a str copy for determining if we have had a change.
                    # This is done since there is no copy() method implemented
                    # for IR.
                    inst_str_copy = str(inst)
                out_nodes = [inst]
                self.curr_loc = inst.loc
                self.dataframe_pass.curr_loc = self.curr_loc

                try:
                    if isinstance(inst, ir.Assign):
                        self.func_ir._definitions[inst.target.name].remove(inst.value)
                        # first try DataFrame transformations. If not applicable (None
                        # return), try Series transformations
                        out_nodes = self.dataframe_pass._run_assign(inst)
                        if out_nodes is None:
                            out_nodes = self._run_assign(inst)
                    elif isinstance(inst, (ir.SetItem, ir.StaticSetItem)):
                        out_nodes = self.dataframe_pass._run_setitem(inst)
                        if out_nodes is None:
                            out_nodes = self._run_setitem(inst)
                    # raise error if df.columns is still in the IR at this point, since
                    # typing pass should have replaced it, unless if new column names
                    # are not constant
                    elif (
                        isinstance(inst, ir.SetAttr)
                        and isinstance(self.typemap[inst.target.name], DataFrameType)
                        and inst.attr == "columns"
                    ):
                        raise BodoError(
                            "DataFrame.columns: new column names should be a constant list"
                        )
                    elif isinstance(inst, bodo.ir.csv_ext.CsvReader):
                        # This is a typing check. It doesn't impact typing, but it avoids an
                        # extra pass over the IR (reducing compilation time).
                        bodo.ir.csv_ext.check_node_typing(inst, self.typemap)
                except BodoError as e:
                    msg = f"{self.curr_loc.strformat()}\n{str(e)}"
                    raise BodoError(msg)

                if isinstance(out_nodes, list):
                    new_body.extend(out_nodes)
                    self._update_definitions(out_nodes)
                    if not changed:
                        # out_nodes is reset each iteration,
                        # so if there are no changes to the IR
                        # it should only contain the original
                        # instruction.
                        changed = (
                            len(out_nodes) != 1 or str(out_nodes[0]) != inst_str_copy
                        )
                if isinstance(out_nodes, ReplaceFunc):
                    rp_func = out_nodes
                    if rp_func.pre_nodes is not None:
                        new_body.extend(rp_func.pre_nodes)
                        self._update_definitions(rp_func.pre_nodes)
                    # replace inst.value to a call with target args
                    # as expected by inline_closure_call
                    inst.value = ir.Expr.call(
                        ir.Var(block.scope, mk_unique_var("dummy"), inst.loc),
                        rp_func.args,
                        (),
                        inst.loc,
                    )
                    # replace "target" of Setitem nodes since inline_closure_call
                    # assumes an assignment and sets "target" to return value
                    if isinstance(inst, (ir.SetItem, ir.StaticSetItem)):
                        dummy_varname = mk_unique_var("dummy")
                        inst.target = ir.Var(block.scope, dummy_varname, inst.loc)
                        # Append the dummy var to the typemap for correctness.
                        self.typemap[dummy_varname] = types.none
                    block.body = new_body + block.body[i:]
                    # save work_list length to know how many new items are added
                    n_prev_work_items = len(work_list)
                    # workaround: inline_closure_call doesn't run the full pipeline to
                    # generate callee's IR, which can cause problems for UDFs (e.g. may
                    # require untyped pass).
                    # so we use its callee_validator mechanism to
                    # replace the IR with the proper one
                    # see: test_series_map_full_pipeline
                    # TODO(ehsan): update inline_closure_call() to run full pipeline
                    callee_validator = None
                    if rp_func.run_full_pipeline:
                        f_ir, _, _, _ = bodo.compiler.get_func_type_info(
                            rp_func.func, rp_func.arg_types, {}
                        )

                        def replace_blocks(new_ir):
                            new_ir.blocks = f_ir.blocks

                        callee_validator = replace_blocks
                    callee_blocks, _ = inline_closure_call(
                        self.func_ir,
                        rp_func.glbls,
                        block,
                        len(new_body),
                        rp_func.func,
                        typingctx=self.typingctx,
                        targetctx=self.targetctx,
                        arg_typs=rp_func.arg_types,
                        typemap=self.typemap,
                        calltypes=self.calltypes,
                        work_list=work_list,
                        callee_validator=callee_validator,
                    )
                    # recursively inline Bodo functions if necessary (UDF case)
                    if rp_func.inline_bodo_calls:
                        # account for the new block inline_closure_call adds for code
                        # before the call in working block
                        n_prev_work_items += 1
                        # blocks of newly inlined function
                        inline_worklist = work_list[n_prev_work_items:]
                        new_labels = bodo.compiler.inline_calls(
                            self.func_ir,
                            self.locals,
                            inline_worklist,
                            self.typingctx,
                            self.targetctx,
                            self.typemap,
                            self.calltypes,
                        )
                        # add blocks added by inliner to be processed
                        work_list = work_list[:n_prev_work_items]  # avoid duplication
                        for l in new_labels:
                            # detect if the newly inlined function updates dataframe
                            # columns inplace, see test_set_column_detect_update3
                            self._get_updated_dataframes(
                                {0: blocks[l]},
                                [0],
                                self.dataframe_pass._updated_dataframes,
                            )
                            work_list.append((l, blocks[l]))
                    # Loc objects are not updated for user Bodo functions to keep source
                    # mapping
                    else:
                        # update Loc objects
                        for c_block in callee_blocks.values():
                            c_block.loc = self.curr_loc
                            update_locs(c_block.body, self.curr_loc)
                    replaced = True
                    break

            if not replaced:
                blocks[label].body = new_body
            else:
                changed = True

        # simplify CFG and run dead code elimination
        simplified_ir = self._simplify_IR()
        changed = changed or simplified_ir
        dprint_func_ir(self.func_ir, "after series pass")
        return changed

    def _run_assign(self, assign):
        lhs = assign.target.name
        rhs = assign.value

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

        # Start of Table Operations

        if isinstance(target_typ, bodo.types.TableType):
            # Inline all table getitems
            impl = bodo.hiframes.table.overload_table_getitem(target_typ, idx_typ)
            return nodes + compile_func_single_block(
                impl, (target, idx), ret_var=assign.target, typing_info=self
            )

        # Start of Series Operations

        # optimize out trivial slicing on series types
        if is_series_type(target_typ) and guard(
            is_whole_slice, self.typemap, self.func_ir, idx
        ):
            return [ir.Assign(target, assign.target, assign.loc)]

        if is_series_type(target_typ) and not isinstance(
            target_typ.index, HeterogeneousIndexType
        ):
            impl = bodo.hiframes.series_indexing.overload_series_getitem(
                self.typemap[target.name], self.typemap[idx.name]
            )
            return replace_func(self, impl, (target, idx), pre_nodes=nodes)

        # heterogenous Series.iloc[]
        if isinstance(target_typ, HeterogeneousSeriesIlocType):
            impl = bodo.hiframes.series_indexing.overload_heter_series_iloc_getitem(
                self.typemap[target.name], self.typemap[idx.name]
            )
            return replace_func(self, impl, (target, idx), pre_nodes=nodes)

        # Series.iloc[]
        if isinstance(target_typ, SeriesIlocType):
            impl = bodo.hiframes.series_indexing.overload_series_iloc_getitem(
                self.typemap[target.name], self.typemap[idx.name]
            )
            return replace_func(self, impl, (target, idx), pre_nodes=nodes)

        # Series.loc[]
        if isinstance(target_typ, SeriesLocType):
            impl = bodo.hiframes.series_indexing.overload_series_loc_getitem(
                self.typemap[target.name], self.typemap[idx.name]
            )
            return replace_func(self, impl, (target, idx), pre_nodes=nodes)

        # Series.iat[]
        if isinstance(target_typ, SeriesIatType):
            impl = bodo.hiframes.series_indexing.overload_series_iat_getitem(
                self.typemap[target.name], self.typemap[idx.name]
            )
            return replace_func(self, impl, (target, idx), pre_nodes=nodes)

        # simplify getitem on Series with constant Index values
        # used for df.apply() UDF optimization
        if (
            isinstance(target_typ, (SeriesType, HeterogeneousSeriesType))
            and isinstance(target_typ.index, HeterogeneousIndexType)
            and is_overload_constant_tuple(target_typ.index.data)
        ):
            indices = get_overload_const_tuple(target_typ.index.data)
            # Pandas falls back to positional indexing for int keys if index has no ints
            if isinstance(idx_typ, types.Integer) and not any(
                isinstance(a, int) for a in indices
            ):
                return compile_func_single_block(
                    eval(
                        "lambda S, idx: bodo.hiframes.pd_series_ext.get_series_data(S)[idx]"
                    ),
                    [rhs.value, idx],
                    assign.target,
                    self,
                )  # pragma: no cover

            if is_literal_type(idx_typ):
                idx_val = get_literal_value(idx_typ)
                if idx_val in indices:
                    arr_ind = indices.index(idx_val)
                    return compile_func_single_block(
                        eval(
                            "lambda S: bodo.hiframes.pd_series_ext.get_series_data(S)[_arr_ind]"
                        ),
                        [rhs.value],
                        assign.target,
                        self,
                        extra_globals={"_arr_ind": arr_ind},
                    )  # pragma: no cover

        if isinstance(target_typ, SeriesStrMethodType):
            impl = bodo.hiframes.series_str_impl.overload_str_method_getitem(
                target_typ, idx_typ
            )
            return replace_func(self, impl, (target, idx), pre_nodes=nodes)

        # Start of Index Operations

        # optimize out trivial slicing on index types
        if (
            isinstance(target_typ, bodo.hiframes.pd_multi_index_ext.MultiIndexType)
            or bodo.hiframes.pd_index_ext.is_pd_index_type(target_typ)
        ) and guard(is_whole_slice, self.typemap, self.func_ir, idx):
            return [ir.Assign(target, assign.target, assign.loc)]

        # inline index getitem, TODO: test
        if bodo.hiframes.pd_index_ext.is_pd_index_type(target_typ) and not isinstance(
            target_typ, HeterogeneousIndexType
        ):
            typ1, typ2 = self.typemap[target.name], self.typemap[idx.name]
            if isinstance(target_typ, RangeIndexType):
                # avoid inlining slice getitem of RangeIndex since it causes issues for
                # 1D_Var parallelization, see test_getitem_slice
                if isinstance(idx_typ, types.SliceType):
                    return [assign]
                impl = bodo.hiframes.pd_index_ext.overload_range_index_getitem(
                    typ1, typ2
                )
            elif isinstance(target_typ, bodo.hiframes.pd_index_ext.DatetimeIndexType):
                impl = bodo.hiframes.pd_index_ext.overload_datetime_index_getitem(
                    typ1, typ2
                )
            elif isinstance(target_typ, bodo.hiframes.pd_index_ext.TimedeltaIndexType):
                impl = bodo.hiframes.pd_index_ext.overload_timedelta_index_getitem(
                    typ1, typ2
                )
            elif isinstance(
                target_typ, bodo.hiframes.pd_index_ext.CategoricalIndexType
            ):
                impl = bodo.hiframes.pd_index_ext.overload_categorical_index_getitem(
                    typ1, typ2
                )
            else:
                impl = bodo.hiframes.pd_index_ext.overload_index_getitem(typ1, typ2)
            return replace_func(self, impl, (target, idx), pre_nodes=nodes)

        # Start of Array Operations

        # optimize out trivial slicing on arrays
        if is_array_typ(target_typ, False) and guard(
            is_whole_slice, self.typemap, self.func_ir, idx
        ):
            return [ir.Assign(target, assign.target, assign.loc)]

        # Optimize getitem of constant array, generated by UDFs that return a Series
        # e.g. t = (a, b); A = str_arr_from_sequence(t); val = A[0] -> val = a
        if is_array_typ(target_typ, False) and isinstance(
            idx_typ, types.IntegerLiteral
        ):
            t_def = guard(get_definition, self.func_ir, target)
            # TODO: support other const array calls
            if guard(find_callname, self.func_ir, t_def, self.typemap) in (
                ("str_arr_from_sequence", "bodo.libs.str_arr_ext"),
                ("asarray", "numpy"),
            ):
                a_def = guard(get_definition, self.func_ir, t_def.args[0])
                if is_expr(a_def, "build_tuple"):
                    assign.value = a_def.items[idx_typ.literal_value]
                    return [assign]

        # Start of Misc Operations

        # optimize out getitem on built_tuple, important for pd.DataFrame()
        # since dictionary is converted to tuple
        if isinstance(target_typ, types.BaseTuple) and isinstance(
            idx_typ, types.IntegerLiteral
        ):
            val_def = guard(get_definition, self.func_ir, rhs.value)
            if isinstance(val_def, ir.Expr) and val_def.op == "build_tuple":
                assign.value = val_def.items[idx_typ.literal_value]
                return [assign]

        # optimize out getitem on build_nullable_tuple,
        # important for df.apply since the row is converted
        # to a nullable tuple.
        if isinstance(target_typ, bodo.types.NullableTupleType) and isinstance(
            idx_typ, types.IntegerLiteral
        ):
            val_def = guard(get_definition, self.func_ir, rhs.value)
            if isinstance(val_def, ir.Expr) and val_def.op == "call":
                call_name = guard(find_callname, self.func_ir, val_def, self.typemap)
                if call_name == (
                    "build_nullable_tuple",
                    "bodo.libs.nullable_tuple_ext",
                ):
                    # Replace the target array with the original data.
                    # If this is a tuple series_pass should optimize it out.
                    assign.value.value = val_def.args[0]
                    return [assign]

        # replace namedtuple access with original value if possible
        # for example: r = Row(a, b); c = r["R1"] -> c = a
        # used for df.apply() UDF optimization
        if isinstance(target_typ, types.BaseNamedTuple) and isinstance(
            idx_typ, (types.StringLiteral, types.IntegerLiteral)
        ):
            named_tup_def = guard(get_definition, self.func_ir, rhs.value)
            # TODO: support kws
            if is_expr(named_tup_def, "call") and not named_tup_def.kws:
                if isinstance(idx_typ, types.StringLiteral):
                    arg_no = target_typ.instance_class._fields.index(
                        idx_typ.literal_value
                    )
                else:
                    arg_no = idx_typ.literal_value
                assign.value = named_tup_def.args[arg_no]

        nodes.append(assign)
        return nodes

    def _run_setitem(self, inst):
        target_typ = self.typemap[inst.target.name]
        value_type = self.typemap[inst.value.name]
        # Series as index
        # TODO: handle all possible cases
        nodes = []
        index_var = get_getsetitem_index_var(inst, self.typemap, nodes)
        index_typ = self.typemap[index_var.name]

        # Start of Series Operations

        # inline Series.loc setitem
        if isinstance(target_typ, SeriesLocType):
            impl = bodo.hiframes.series_indexing.overload_series_loc_setitem(
                target_typ, index_typ, value_type
            )
            # NOTE: using 'replace_func' instead of 'compile_func_single_block' to make
            # sure the newly added IR is transformed ("I._obj" is optimized away)
            return replace_func(
                self, impl, [inst.target, index_var, inst.value], pre_nodes=nodes
            )

        # TODO: proper iat/iloc/loc optimization
        # if isinstance(target_typ, SeriesIatType):
        #     val_def = guard(get_definition, self.func_ir, inst.target)
        #     assert (isinstance(val_def, ir.Expr) and val_def.op == 'getattr'
        #         and val_def.attr in ('iat', 'iloc', 'loc'))
        #     series_var = val_def.value
        #     inst.target = series_var
        #     target_typ = target_typ.stype

        # Start of Array Operations

        # support A[i] = None array setitem using our array NA setting function
        if (
            is_array_typ(target_typ, False)
            and isinstance(index_typ, types.Integer)
            and self.typemap[inst.value.name] == types.none
        ):
            return nodes + compile_func_single_block(
                eval("lambda A, idx: bodo.libs.array_kernels.setna(A, idx)"),
                [inst.target, index_var],
                None,
                self,
            )

        # Start of Misc Operations

        # handle struct setitem (needed for UDF inlining, see test_series_map_dict)
        # S[i] = {"A": v1, "B": v2} -> S[i] = struct_if_heter_dict((v1, v2), ("A", "B"))
        if isinstance(target_typ, StructArrayType):
            val_def = guard(get_definition, self.func_ir, inst.value)
            if is_expr(val_def, "build_map"):
                (
                    _,
                    val_tup,
                    val_tup_assign,
                    key_tup,
                    key_tup_assign,
                ) = extract_keyvals_from_struct_map(
                    self.func_ir, val_def, inst.loc, inst.target.scope, self.typemap
                )
                return (
                    nodes
                    + [val_tup_assign, key_tup_assign]
                    + compile_func_single_block(
                        eval(
                            "lambda vals, keys: bodo.utils.conversion.struct_if_heter_dict(vals, keys)"
                        ),
                        (val_tup, key_tup),
                        inst.value,
                        self,
                    )
                    + [inst]
                )

        if "h5py" in sys.modules and target_typ == bodo.io.h5_api.h5dataset_type:
            return self._handle_h5_write(inst.target, inst.index, inst.value)

        # optimize simple string value copy across arrays to avoid string allocation
        # B[j] = A[i] -> get_str_arr_item_copy(B, j, A, i)
        if target_typ == string_array_type and isinstance(index_typ, types.Integer):
            val_def = guard(get_definition, self.func_ir, inst.value)
            if (
                is_expr(val_def, "getitem") or is_expr(val_def, "static_getitem")
            ) and is_str_arr_type(self.typemap[val_def.value.name]):
                val_idx = get_getsetitem_index_var(val_def, self.typemap, nodes)
                return nodes + compile_func_single_block(
                    eval(
                        "lambda B, j, A, i: bodo.libs.str_arr_ext.get_str_arr_item_copy(B, j, A, i)"
                    ),
                    (inst.target, index_var, val_def.value, val_idx),
                    inst.value,
                    self,
                )

        nodes.append(inst)
        return nodes

    def _run_getattr(self, assign, rhs):
        rhs_type = self.typemap[rhs.value.name]  # get type of rhs value "S"

        # Start of Series Operations

        # conditional on S.str is not supported since aliasing, ... can't
        # be handled for getattr. This is probably a rare case.
        # TODO: handle, example:
        # if flag:
        #    S_str = S1.str
        # else:
        #    S_str = S2.str

        if (
            isinstance(
                rhs_type,
                (
                    SeriesStrMethodType,
                    SeriesCatMethodType,
                    SeriesDatetimePropertiesType,
                ),
            )
            and rhs.attr == "_obj"
        ):
            rhs_def = guard(get_definition, self.func_ir, rhs.value)

            if rhs_def is None:
                if isinstance(rhs_type, SeriesStrMethodType):
                    raise BodoError("Invalid Series.str, cannot handle conditional yet")
                elif isinstance(rhs_type, SeriesCatMethodType):
                    raise BodoError("Invalid Series.cat, cannot handle conditional yet")
                else:
                    assert isinstance(rhs_type, SeriesDatetimePropertiesType)
                    raise BodoError("Invalid Series.dt, cannot handle conditional yet")
            assert is_expr(rhs_def, "getattr")
            assign.value = rhs_def.value
            return [assign]

        if (
            isinstance(rhs_type, (SeriesIlocType, SeriesLocType, SeriesIatType))
            and rhs.attr == "_obj"
        ):
            arg = guard(get_definition, self.func_ir, rhs.value)
            if arg:
                assign.value = arg.value
            return [assign]

        # inline Series.cat.codes
        if isinstance(rhs_type, SeriesCatMethodType) and rhs.attr == "codes":
            impl = bodo.hiframes.series_str_impl.series_cat_codes_overload(rhs_type)
            return replace_func(self, impl, [rhs.value])

        # simplify getattr access on Series with constant Index values
        # used for df.apply() UDF optimization
        if (
            isinstance(rhs_type, (SeriesType, HeterogeneousSeriesType))
            and isinstance(rhs_type.index, HeterogeneousIndexType)
            and is_overload_constant_tuple(rhs_type.index.data)
        ):
            indices = get_overload_const_tuple(rhs_type.index.data)
            if rhs.attr in indices:
                arr_ind = indices.index(rhs.attr)
                nodes = compile_func_single_block(
                    eval(
                        "lambda S: bodo.hiframes.pd_series_ext.get_series_data(S)[_arr_ind]"
                    ),
                    [rhs.value],
                    assign.target,
                    self,
                    extra_globals={"_arr_ind": arr_ind},
                )
                return nodes

        # inline Series.dt.field
        if (
            isinstance(rhs_type, SeriesDatetimePropertiesType)
            and rhs.attr not in timedelta_methods
        ):
            if rhs_type.stype.dtype == types.NPDatetime("ns") or isinstance(
                rhs_type.stype.dtype,
                bodo.libs.pd_datetime_arr_ext.PandasDatetimeTZDtype,
            ):
                if rhs.attr == "date":
                    impl = bodo.hiframes.series_dt_impl.series_dt_date_overload(
                        rhs_type
                    )
                    return replace_func(self, impl, [rhs.value])
                elif rhs.attr in bodo.hiframes.pd_timestamp_ext.date_fields:
                    impl = bodo.hiframes.series_dt_impl.create_date_field_overload(
                        rhs.attr
                    )(rhs_type)
                    return replace_func(self, impl, [rhs.value])
            else:
                if rhs.attr in ("nanoseconds", "microseconds", "seconds", "days"):
                    impl = bodo.hiframes.series_dt_impl.create_timedelta_field_overload(
                        rhs.attr
                    )(rhs_type)
                    return replace_func(self, impl, [rhs.value])

        # inline rolling.attr access
        attr_ind = {"obj": 0, "window": 1, "min_periods": 2, "center": 3}
        if (
            isinstance(rhs_type, bodo.hiframes.pd_rolling_ext.RollingType)
            and (not rhs_type.selection or rhs.attr not in rhs_type.selection)
            and rhs.attr in attr_ind
        ):
            # get init_rolling() call
            rhs_def = guard(get_definition, self.func_ir, rhs.value)
            # handle explicit column selection case
            if is_expr(rhs_def, "static_getitem") or is_expr(rhs_def, "getattr"):
                rhs_def = guard(get_definition, self.func_ir, rhs_def.value)
            if rhs_def is not None:
                assert is_call(rhs_def), "invalid rolling object creation"
                arg_ind = attr_ind[rhs.attr]
                assign.value = rhs_def.args[arg_ind]
                return [assign]

        # replace series/arr.dtype for dt64 since PA replaces with
        # np.datetime64[ns] which invalid, TODO: fix PA
        if (
            rhs.attr == "dtype"
            and (is_series_type(rhs_type) or isinstance(rhs_type, types.Array))
            and isinstance(rhs_type.dtype, (types.NPDatetime, types.NPTimedelta))
        ):
            assign.value = ir.Global("numpy.datetime64", rhs_type.dtype, rhs.loc)
            return [assign]

        # replace attribute access with overload
        if isinstance(rhs_type, SeriesType) and rhs.attr in (
            "values",
            "shape",
            "size",
            "empty",
        ):
            overload_name = "overload_series_" + rhs.attr
            overload_func = getattr(bodo.hiframes.series_impl, overload_name)
            impl = overload_func(rhs_type)
            return replace_func(self, impl, [rhs.value])

        # Replace T.shape to optimize out T.shape[1] (can be generated in BodoSQL)
        if (
            isinstance(rhs_type, bodo.types.TableType)
            and rhs.attr == "shape"
            and not rhs_type.has_runtime_cols
        ):
            n_cols = len(rhs_type.arr_types)
            return compile_func_single_block(
                eval(f"lambda T: (len(T), {n_cols})"),
                [rhs.value],
                assign.target,
                self,
            )

        # replace series/arr.dtype since PA replacement inserts in the
        # beginning of block, preventing fusion. TODO: fix PA
        if rhs.attr == "dtype" and isinstance(
            if_series_to_array_type(rhs_type), types.Array
        ):
            typ_str = str(rhs_type.dtype)
            assign.value = ir.Global(f"np.dtype({typ_str})", np.dtype(typ_str), rhs.loc)
            return [assign]

        # Start of Index Operations

        if (
            isinstance(
                rhs_type,
                (
                    NumericIndexType,
                    StringIndexType,
                    BinaryIndexType,
                    PeriodIndexType,
                    CategoricalIndexType,
                    DatetimeIndexType,
                    TimedeltaIndexType,
                ),
            )
            and rhs.attr == "values"
        ):
            # simply return the data array
            nodes = []
            var = self._get_index_data(rhs.value, nodes)
            assign.value = var
            nodes.append(assign)
            return nodes

        if isinstance(rhs_type, RangeIndexType) and rhs.attr == "values":
            return replace_func(
                self,
                eval("lambda A: bodo.utils.conversion.coerce_to_ndarray(A)"),
                [rhs.value],
            )

        if isinstance(rhs_type, DatetimeIndexType):
            if (
                rhs.attr in bodo.hiframes.pd_timestamp_ext.date_fields
                and not rhs.attr == "is_leap_year"
            ):
                impl = bodo.hiframes.pd_index_ext.gen_dti_field_impl(rhs.attr)
                return replace_func(self, impl, [rhs.value])
            if rhs.attr == "date":
                impl = bodo.hiframes.pd_index_ext.overload_datetime_index_date(rhs_type)
                return replace_func(self, impl, [rhs.value])
            if rhs.attr == "is_leap_year":
                impl = bodo.hiframes.pd_index_ext.overload_datetime_index_is_leap_year(
                    rhs_type
                )
                return replace_func(self, impl, [rhs.value])

        if isinstance(rhs_type, TimedeltaIndexType):
            if rhs.attr in bodo.hiframes.pd_timestamp_ext.timedelta_fields:
                impl = bodo.hiframes.pd_index_ext.gen_tdi_field_impl(rhs.attr)
                return replace_func(self, impl, [rhs.value])

        # optimize away RangeIndex._start/_stop/_step if definition can be found
        # no need for checking other references to the RangeIndex since it is immutable
        if isinstance(rhs_type, RangeIndexType) and rhs.attr in (
            "_start",
            "_stop",
            "_step",
        ):
            r_def = guard(get_definition, self.func_ir, rhs.value)
            # find init_range_index(start, stop, step) call and replace value
            if r_def is not None and guard(
                find_callname, self.func_ir, r_def, self.typemap
            ) == (
                "init_range_index",
                "bodo.hiframes.pd_index_ext",
            ):
                if rhs.attr == "_start":
                    assign.value = r_def.args[0]
                if rhs.attr == "_stop":
                    assign.value = r_def.args[1]
                if rhs.attr == "_step":
                    assign.value = r_def.args[2]
            return [assign]

        # Start of Misc Operations

        # replace namedtuple access with original value if possible
        # for example: r = Row(a, b); c = r.R1 -> c = a
        # used for df.apply() UDF optimization
        if isinstance(rhs_type, types.BaseNamedTuple):
            named_tup_def = guard(get_definition, self.func_ir, rhs.value)
            # TODO: support kws
            if is_expr(named_tup_def, "call") and not named_tup_def.kws:
                arg_no = rhs_type.instance_class._fields.index(rhs.attr)
                assign.value = named_tup_def.args[arg_no]

        # optimize away bodo_sql_context.dataframes if
        # init_sql_context(names, dataframes) can be found
        #
        # Note we delay checking BodoSQLContextType until we find a possible match
        # to avoid paying the import overhead for Bodo calls with no BodoSQL.
        if hasattr(rhs, "attr") and rhs.attr == "dataframes":  # pragma: no cover
            sql_ctx_def = guard(get_definition, self.func_ir, rhs.value)
            if guard(find_callname, self.func_ir, sql_ctx_def, self.typemap) == (
                "init_sql_context",
                "bodosql.context_ext",
            ):
                if is_bodosql_context_type(rhs_type):
                    assign.value = sql_ctx_def.args[1]
                    return [assign]

        # optimize out spark_df._df if possible
        if (
            "bodo.libs.pyspark_ext" in sys.modules
            and isinstance(rhs_type, bodo.libs.pyspark_ext.SparkDataFrameType)
            and rhs.attr == "_df"
        ):
            rhs_def = guard(get_definition, self.func_ir, rhs.value)
            if guard(find_callname, self.func_ir, rhs_def) == (
                "init_spark_df",
                "bodo.libs.pyspark_ext",
            ):
                assign.value = rhs_def.args[0]
                return [assign]

        return [assign]

    def _run_binop(self, assign, rhs):
        """Handle binary operators. Mostly inlining overloads since not possible in
        Numba yet.
        """

        arg1, arg2 = rhs.lhs, rhs.rhs
        typ1, typ2 = self.typemap[arg1.name], self.typemap[arg2.name]
        cmp_ops = (
            operator.eq,
            operator.ne,
            operator.ge,
            operator.gt,
            operator.le,
            operator.lt,
        )

        # Start of Series Operations

        # series(dt64) - (timestamp/datetime.timedelta/datetime.datetime/series(timedelta64)/series(dt64))
        # or (timestamp or datetime.datetime) - series(dt64)
        if rhs.fn == operator.sub and (
            is_dt64_series_typ(typ1)
            and (
                typ2 == bodo.hiframes.pd_timestamp_ext.pd_timestamp_tz_naive_type
                or typ2 == datetime_timedelta_type
                or typ2 == datetime_datetime_type
                or is_timedelta64_series_typ(typ2)
                or is_dt64_series_typ(typ2)
            )
            or (
                is_dt64_series_typ(typ2)
                and (
                    typ1 == bodo.hiframes.pd_timestamp_ext.pd_timestamp_tz_naive_type
                    or typ1 == datetime_datetime_type
                )
            )
        ):
            impl = bodo.hiframes.series_dt_impl.create_bin_op_overload(rhs.fn)(
                typ1, typ2
            )
            return replace_func(self, impl, [arg1, arg2])

        # series(dt64) + (datetime.timedelta/series(timedelta64))
        # or (datetime.timedelta/series(timedelta64)) + series(dt64)
        if rhs.fn == operator.add and (
            is_dt64_series_typ(typ1)
            and (typ2 == datetime_timedelta_type or is_timedelta64_series_typ(typ2))
            or (
                is_dt64_series_typ(typ2)
                and (typ1 == datetime_timedelta_type or is_timedelta64_series_typ(typ1))
            )
        ):
            impl = bodo.hiframes.series_dt_impl.create_bin_op_overload(rhs.fn)(
                typ1, typ2
            )
            return replace_func(self, impl, [arg1, arg2])

        # series(timedelta64) - datetime.timedelta
        # or datetime.timedelta - series(timedelta64)
        if rhs.fn in [operator.sub, operator.add] and (
            (is_timedelta64_series_typ(typ1) and typ2 == datetime_timedelta_type)
            or (is_timedelta64_series_typ(typ2) and typ1 == datetime_timedelta_type)
        ):
            impl = bodo.hiframes.series_dt_impl.create_bin_op_overload(rhs.fn)(
                typ1, typ2
            )
            return replace_func(self, impl, [arg1, arg2])

        # series(timedelta) comp op timedelta
        if rhs.fn in cmp_ops and (
            is_timedelta64_series_typ(typ1)
            and typ2 == bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_type
            or (
                is_timedelta64_series_typ(typ2)
                and typ1 == bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_type
            )
        ):
            impl = bodo.hiframes.series_dt_impl.create_cmp_op_overload(rhs.fn)(
                typ1, typ2
            )
            return replace_func(self, impl, [arg1, arg2])

        # series(dt64) comp ops with pandas.Timestamp/string type
        if rhs.fn in cmp_ops and (
            is_dt64_series_typ(typ1)
            and (
                typ2 == bodo.hiframes.pd_timestamp_ext.pd_timestamp_tz_naive_type
                or typ2 == string_type
                or bodo.utils.typing.is_overload_constant_str(typ2)
            )
            or (
                is_dt64_series_typ(typ2)
                and (
                    typ1 == bodo.hiframes.pd_timestamp_ext.pd_timestamp_tz_naive_type
                    or typ1 == string_type
                    or bodo.utils.typing.is_overload_constant_str(typ1)
                )
            )
        ):
            impl = bodo.hiframes.series_dt_impl.create_cmp_op_overload(rhs.fn)(
                typ1, typ2
            )
            return replace_func(self, impl, [arg1, arg2])

        # catch the rest of the series ops
        if isinstance(typ1, SeriesType) or isinstance(typ2, SeriesType):
            if rhs.fn in bodo.hiframes.pd_series_ext.series_inplace_binary_ops:
                overload_func = (
                    bodo.hiframes.series_impl.create_inplace_binary_op_overload(rhs.fn)
                )
                impl = overload_func(typ1, typ2)
                return replace_func(self, impl, [arg1, arg2])

            if rhs.fn in bodo.hiframes.pd_series_ext.series_binary_ops:
                overload_func = bodo.hiframes.series_impl.create_binary_op_overload(
                    rhs.fn
                )
                impl = overload_func(typ1, typ2)
                return replace_func(self, impl, [arg1, arg2])

            # replace matmul '@' operator with np.dot
            if rhs.fn == operator.matmul:
                nodes = []
                if isinstance(typ1, SeriesType):
                    arg1 = self._get_series_data(arg1, nodes)
                if isinstance(typ2, SeriesType):
                    arg2 = self._get_series_data(arg2, nodes)
                return nodes + compile_func_single_block(
                    eval("lambda A, B: np.dot(A, B)"),
                    [arg1, arg2],
                    assign.target,
                    self,
                )
            return [assign]

        # Start of Index Operations

        # inline overloaded
        # TODO: use overload inlining when available
        if rhs.fn == operator.sub and (
            (
                isinstance(typ1, DatetimeIndexType)
                and typ2 == bodo.hiframes.pd_timestamp_ext.pd_timestamp_tz_naive_type
            )
            or (
                isinstance(typ2, DatetimeIndexType)
                and typ1 == bodo.hiframes.pd_timestamp_ext.pd_timestamp_tz_naive_type
            )
        ):
            impl = bodo.hiframes.pd_index_ext.overload_sub_operator_datetime_index(
                typ1, typ2
            )
            return replace_func(self, impl, [arg1, arg2])

        # string comparison with DatetimeIndex
        if rhs.fn in cmp_ops and (
            isinstance(typ1, DatetimeIndexType)
            and types.unliteral(typ2) == string_type
            or (
                isinstance(typ2, DatetimeIndexType)
                and types.unliteral(typ1) == string_type
            )
        ):
            impl = bodo.hiframes.pd_index_ext.overload_binop_dti_str(rhs.fn)(typ1, typ2)
            return replace_func(self, impl, [arg1, arg2])

        # inline Index ops if no input is Series
        # No input is series, since we just checked for it.
        if (
            is_index_type(typ1) or is_index_type(typ2)
        ) and rhs.fn in bodo.hiframes.pd_series_ext.series_binary_ops:
            overload_func = bodo.hiframes.pd_index_ext.create_binary_op_overload(rhs.fn)
            impl = overload_func(typ1, typ2)
            return replace_func(self, impl, [arg1, arg2])

        # Start of Array Operations

        if rhs.fn == operator.add and (is_str_arr_type(typ1) or is_str_arr_type(typ2)):
            impl = bodo.libs.str_arr_ext.overload_add_operator_string_array(typ1, typ2)
            return replace_func(self, impl, [arg1, arg2])

        # Add for tz-aware
        if rhs.fn == operator.add and (
            isinstance(typ1, bodo.types.DatetimeArrayType)
            or isinstance(typ2, bodo.types.DatetimeArrayType)
        ):
            impl = bodo.libs.pd_datetime_arr_ext.overload_add_operator_datetime_arr(
                typ1, typ2
            )
            return replace_func(self, impl, [arg1, arg2])

        if rhs.fn == operator.sub and (
            isinstance(typ1, bodo.types.DatetimeArrayType)
            or isinstance(typ2, bodo.types.DatetimeArrayType)
        ):
            impl = bodo.libs.pd_datetime_arr_ext.overload_sub_operator_datetime_arr(
                typ1, typ2
            )
            return replace_func(self, impl, [arg1, arg2])

        if rhs.fn == operator.sub and typ2 == datetime_timedelta_type:
            if typ1 == datetime_date_array_type:
                impl = (
                    bodo.hiframes.datetime_date_ext.overload_sub_operator_datetime_date(
                        typ1, typ2
                    )
                )
                return replace_func(self, impl, [arg1, arg2])
            elif typ1 == timedelta_array_type:
                impl = bodo.hiframes.datetime_timedelta_ext.overload_sub_operator_datetime_timedelta(
                    typ1, typ2
                )
                return replace_func(self, impl, [arg1, arg2])

        if (
            rhs.fn == operator.sub
            and typ1 == datetime_timedelta_type
            and typ2 == timedelta_array_type
        ):
            impl = bodo.hiframes.datetime_timedelta_ext.overload_sub_operator_datetime_timedelta(
                typ1, typ2
            )
            return replace_func(self, impl, [arg1, arg2])

        # categorical array comparison
        if rhs.fn in (operator.eq, operator.ne) and isinstance(
            typ1, CategoricalArrayType
        ):
            impl = bodo.hiframes.pd_categorical_ext.create_cmp_op_overload(rhs.fn)(
                typ1, typ2
            )
            return replace_func(self, impl, [arg1, arg2])

        # inline string array comparison ops
        if (
            rhs.fn in _string_array_comp_ops
            and (is_str_arr_type(typ1) or is_str_arr_type(typ2))
            and all(
                types.unliteral(t)
                in (string_array_type, bodo.types.dict_str_arr_type, string_type)
                for t in (typ1, typ2)
            )
        ):
            f = bodo.libs.str_arr_ext.create_binary_op_overload(rhs.fn)(
                self.typemap[rhs.lhs.name], self.typemap[rhs.rhs.name]
            )
            return replace_func(self, f, [arg1, arg2])

        # inline binary array comparison ops
        if rhs.fn in _string_array_comp_ops and bodo.libs.binops_ext.binary_array_cmp(
            typ1, typ2
        ):
            f = bodo.libs.binary_arr_ext.create_binary_cmp_op_overload(rhs.fn)(
                self.typemap[rhs.lhs.name], self.typemap[rhs.rhs.name]
            )
            return replace_func(self, f, [arg1, arg2])

        # optimize string array element comparison to operate inplace and avoid string
        # allocation overhead
        # A[i] == val -> inplace_eq(A, i, val)
        if self.optimize_inplace_ops and typ1 == string_type and rhs.fn == operator.eq:
            arg1_def = guard(get_definition, self.func_ir, arg1)
            if (
                is_expr(arg1_def, "getitem")
                and self.typemap[arg1_def.value.name] == string_array_type
            ):
                return compile_func_single_block(
                    eval(
                        "lambda A, i, val: bodo.libs.str_arr_ext.inplace_eq(A, i, val)"
                    ),
                    (arg1_def.value, arg1_def.index, arg2),
                    assign.target,
                    self,
                )

        # TODO: Make sure this doesn't conflict with Numba implementations in 0.52
        if (
            rhs.fn == operator.contains
            and bodo.utils.utils.is_array_typ(typ1, False)
            and typ1.dtype == types.unliteral(typ2)
        ):
            # Currently only supported for our array types.
            return replace_func(
                self, bodo.libs.array_kernels.arr_contains(typ1, typ2), [arg1, arg2]
            )

        # Inline tz-aware array operations
        if rhs.fn in cmp_ops and (
            isinstance(typ1, bodo.libs.pd_datetime_arr_ext.DatetimeArrayType)
            or isinstance(typ2, bodo.libs.pd_datetime_arr_ext.DatetimeArrayType)
        ):
            impl = bodo.libs.pd_datetime_arr_ext.create_cmp_op_overload_arr(rhs.fn)(
                typ1, typ2
            )
            return replace_func(self, impl, [arg1, arg2])

        # Inline tz-naive array + date operations
        if (
            rhs.fn in cmp_ops
            and (
                isinstance(typ1, types.Array)
                and typ1.dtype == bodo.types.datetime64ns
                and typ2
                in (bodo.types.datetime_date_array_type, bodo.types.datetime_date_type)
            )
            or (
                typ1
                in (bodo.types.datetime_date_array_type, bodo.types.datetime_date_type)
                and isinstance(typ2, types.Array)
                and typ2.dtype == bodo.types.datetime64ns
            )
        ):
            impl = bodo.hiframes.datetime_date_ext.create_datetime_array_date_cmp_op_overload(
                rhs.fn
            )(typ1, typ2)
            return replace_func(self, impl, [arg1, arg2])

        # datetime_date_array operations
        if rhs.fn in cmp_ops and (
            typ1 == datetime_date_array_type or typ2 == datetime_date_array_type
        ):
            impl = bodo.hiframes.datetime_date_ext.create_cmp_op_overload_arr(rhs.fn)(
                typ1, typ2
            )
            return replace_func(self, impl, [arg1, arg2])

        # datetime_timedelta_array operations
        if rhs.fn in cmp_ops and (
            typ1 == timedelta_array_type or typ2 == timedelta_array_type
        ):
            impl = bodo.hiframes.datetime_timedelta_ext.create_cmp_op_overload_arr(
                rhs.fn
            )(typ1, typ2)
            return replace_func(self, impl, [arg1, arg2])

        # Manually inline binops between pd.Timedelta and int arrays.
        # Failure to inline seems to produce issues in parfor pass in Numba
        # due to how MUL with timedelta64 is handled.
        if bodo.libs.binops_ext.args_td_and_int_array(typ1, typ2):
            impl = bodo.libs.int_arr_ext.get_int_array_op_pd_td(rhs.fn)(typ1, typ2)
            return replace_func(self, impl, [arg1, arg2])

        # inline remaining Integer array ops
        if (
            rhs.fn in numba.core.typing.npydecl.NumpyRulesArrayOperator._op_map.keys()
            and any(isinstance(t, IntegerArrayType) for t in (typ1, typ2))
            # NOTE: decimal array comparison isn't inlined since it uses Arrow compute
            and not (
                any(isinstance(t, DecimalArrayType) for t in (typ1, typ2))
                and rhs.fn in cmp_ops
            )
        ):
            overload_func = bodo.libs.int_arr_ext.create_op_overload(rhs.fn, 2)
            impl = overload_func(typ1, typ2)
            return replace_func(self, impl, [arg1, arg2])

        if (
            rhs.fn
            in numba.core.typing.npydecl.NumpyRulesInplaceArrayOperator._op_map.keys()
            and any(isinstance(t, IntegerArrayType) for t in (typ1, typ2))
        ):
            overload_func = bodo.libs.int_arr_ext.create_op_overload(rhs.fn, 2)
            impl = overload_func(typ1, typ2)
            return replace_func(self, impl, [arg1, arg2])

        # inline remaining Float array ops
        if (
            rhs.fn in numba.core.typing.npydecl.NumpyRulesArrayOperator._op_map.keys()
            and any(isinstance(t, FloatingArrayType) for t in (typ1, typ2))
            # NOTE: decimal array comparison isn't inlined since it uses Arrow compute
            and not (
                any(isinstance(t, DecimalArrayType) for t in (typ1, typ2))
                and rhs.fn in cmp_ops
            )
        ):
            overload_func = bodo.libs.float_arr_ext.create_op_overload(rhs.fn, 2)
            impl = overload_func(typ1, typ2)
            return replace_func(self, impl, [arg1, arg2])

        if (
            rhs.fn
            in numba.core.typing.npydecl.NumpyRulesInplaceArrayOperator._op_map.keys()
            and any(isinstance(t, FloatingArrayType) for t in (typ1, typ2))
        ):  # pragma: no cover
            overload_func = bodo.libs.float_arr_ext.create_op_overload(rhs.fn, 2)
            impl = overload_func(typ1, typ2)
            return replace_func(self, impl, [arg1, arg2])

        # inline operator.or_ and operator.and_ for boolean arrays
        if (rhs.fn in [operator.or_, operator.and_]) and any(
            t == boolean_array_type for t in (typ1, typ2)
        ):
            if is_valid_boolean_array_logical_op(typ1, typ2):
                impl = bodo.libs.bool_arr_ext.create_nullable_logical_op_overload(
                    rhs.fn
                )(typ1, typ2)
                assert impl != None
                return replace_func(self, impl, [arg1, arg2])

        # inline Boolean array ops
        if (
            rhs.fn in numba.core.typing.npydecl.NumpyRulesArrayOperator._op_map.keys()
            and any(t == boolean_array_type for t in (typ1, typ2))
            # Don't inline operators between array + Series. These should be handled
            # by Series
            and not any(isinstance(t, SeriesType) for t in (typ1, typ2))
        ):
            overload_func = bodo.libs.bool_arr_ext.create_op_overload(rhs.fn, 2)
            impl = overload_func(typ1, typ2)
            return replace_func(self, impl, [arg1, arg2])

        if (
            rhs.fn
            in numba.core.typing.npydecl.NumpyRulesInplaceArrayOperator._op_map.keys()
            and any(t == boolean_array_type for t in (typ1, typ2))
        ):
            overload_func = bodo.libs.bool_arr_ext.create_op_overload(rhs.fn, 2)
            impl = overload_func(typ1, typ2)
            return replace_func(self, impl, [arg1, arg2])

        # Start of Misc remaining Operations

        # replace matmul '@' operator with np.dot
        if rhs.fn == operator.matmul:
            return compile_func_single_block(
                eval("lambda A, B: np.dot(A, B)"),
                [arg1, arg2],
                assign.target,
                self,
            )

        # replace a +/- 0 with a (e.g. in index calc of test_loc_range_index_loop)
        if (
            rhs.fn in (operator.add, operator.sub)
            and guard(find_const, self.func_ir, arg2) == 0
            and typ1 == typ2
        ):
            assign.value = arg1
            return [assign]

        # replace a // 1 with a (e.g. in index calc of test_loc_range_index_loop)
        if (
            rhs.fn == operator.floordiv
            and guard(find_const, self.func_ir, arg2) == 1
            and typ1 == typ2
        ):
            assign.value = arg1
            return [assign]

        return [assign]

    def _run_unary(self, assign, rhs):
        arg = rhs.value
        typ = self.typemap[arg.name]

        if isinstance(typ, SeriesType):
            assert rhs.fn in bodo.hiframes.pd_series_ext.series_unary_ops
            overload_func = bodo.hiframes.series_impl.create_unary_op_overload(rhs.fn)
            impl = overload_func(typ)
            return replace_func(self, impl, [arg])

        if isinstance(typ, IntegerArrayType):  # pragma: no cover
            assert rhs.fn in (operator.neg, operator.invert, operator.pos)
            overload_func = bodo.libs.int_arr_ext.create_op_overload(rhs.fn, 1)
            impl = overload_func(typ)
            return replace_func(self, impl, [arg])

        if isinstance(typ, FloatingArrayType):  # pragma: no cover
            assert rhs.fn in (operator.neg, operator.pos)
            overload_func = bodo.libs.float_arr_ext.create_op_overload(rhs.fn, 1)
            impl = overload_func(typ)
            return replace_func(self, impl, [arg])

        if typ == boolean_array_type:
            assert rhs.fn in (operator.neg, operator.invert, operator.pos)
            overload_func = bodo.libs.bool_arr_ext.create_op_overload(rhs.fn, 1)
            impl = overload_func(typ)
            return replace_func(self, impl, [arg])

        return [assign]

    def _run_call(self, assign, lhs, rhs):
        fdef = guard(find_callname, self.func_ir, rhs, self.typemap)
        if fdef is None:
            from numba.stencils.stencil import StencilFunc

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
            # input to _bodo_groupby_apply_impl() is a UDF dispatcher
            elif isinstance(func_def, ir.Arg) and isinstance(
                self.typemap[rhs.func.name], types.Dispatcher
            ):
                return [assign]
            warnings.warn("function call couldn't be found for initial analysis")
            return [assign]
        else:
            func_name, func_mod = fdef

        # inline UDFs to enable more optimization
        # avoid_udf_inline() decides about inlining. For example,
        # cannot inline if function has assertions. see test_df_apply_assertion
        func_type = self.typemap[rhs.func.name]
        if bodo.compiler.is_udf_call(func_type) and not avoid_udf_inline(
            func_type.dispatcher.py_func,
            tuple(self.typemap[v.name] for v in rhs.args),
            {k: self.typemap[v.name] for k, v in dict(rhs.kws).items()},
        ):
            assert rhs.vararg is None, "vararg not supported for inlining UDFs"
            assert rhs.varkwarg is None, "varkwarg not supported for inlining UDFs"
            return replace_func(
                self,
                func_type.dispatcher.py_func,
                rhs.args,
                kws=rhs.kws,
                pysig=func_type.dispatcher._compiler.pysig,
                inline_bodo_calls=True,
                run_full_pipeline=True,
            )

        # remove unnecessary bool() calls generated for branch conditions by Numba to
        # simplify the IR
        if (
            fdef == ("bool", "builtins")
            and self.typemap[rhs.args[0].name] == types.bool_
        ):
            assign.value = rhs.args[0]
            return [assign]

        # support call ufuncs on Series
        if (
            func_mod in ("numpy", "ufunc")
            and func_name in ufunc_names
            and any(isinstance(self.typemap[a.name], SeriesType) for a in rhs.args)
        ):
            return self._handle_ufuncs(func_name, rhs.args)

        # inline builtin calls on Series
        if (
            func_mod == "builtins"
            and func_name in ("min", "max", "sum")
            and len(rhs.args) == 1
            and isinstance(self.typemap[rhs.args[0].name], SeriesType)
        ):
            impl = getattr(bodo.hiframes.series_impl, "overload_series_" + func_name)(
                self.typemap[rhs.args[0].name]
            )
            return replace_func(
                self, impl, rhs.args, pysig=numba.core.utils.pysignature(impl), kws=()
            )

        # inline builtin calls on Bodo nullable arrays (NOTE: our overload
        # implementation which has inline='always' may not be pick up by Numba since
        # Bodo arrays are iterables, see test_int_array.py::test_min)
        if (
            func_mod == "builtins"
            and func_name in ("min", "max", "sum")
            and len(rhs.args) == 1
            and isinstance(
                self.typemap[rhs.args[0].name],
                (IntegerArrayType, FloatingArrayType, BooleanArrayType),
            )
        ):
            impl = getattr(bodo.libs.array_kernels, "overload_array_" + func_name)(
                self.typemap[rhs.args[0].name]
            )
            return replace_func(
                self, impl, rhs.args, pysig=numba.core.utils.pysignature(impl), kws=()
            )

        # inline Series methods (necessary since used in min/max/sum array overloads
        # above)
        if (
            isinstance(func_mod, ir.Var)
            and isinstance(self.typemap[func_mod.name], SeriesType)
            and func_name in ("min", "max", "sum")
            and len(rhs.args) == 0
        ):
            rhs.args.insert(0, func_mod)
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}

            impl = getattr(bodo.hiframes.series_impl, "overload_series_" + func_name)(
                *arg_typs, **kw_typs
            )
            return replace_func(
                self,
                impl,
                rhs.args,
                pysig=numba.core.utils.pysignature(impl),
                kws=dict(rhs.kws),
            )

        # inline ufuncs on IntegerArray
        if (
            func_mod in ("numpy", "ufunc")
            and func_name in ufunc_names
            and any(
                isinstance(self.typemap[a.name], IntegerArrayType) for a in rhs.args
            )
        ):
            return self._handle_ufuncs_int_arr(func_name, rhs.args)

        # inline ufuncs on FloatingArray
        if (
            func_mod in ("numpy", "ufunc")
            and func_name in ufunc_names
            and any(
                isinstance(self.typemap[a.name], FloatingArrayType) for a in rhs.args
            )
        ):  # pragma: no cover
            return self._handle_ufuncs_float_arr(func_name, rhs.args)

        # inline ufuncs on BooleanArray
        if (
            func_mod in ("numpy", "ufunc")
            and func_name in ufunc_names
            and any(self.typemap[a.name] == boolean_array_type for a in rhs.args)
        ):
            return self._handle_ufuncs_bool_arr(func_name, rhs.args)

        # Set input_dicts_unified flag if input is already unified in previous streaming
        # operator
        if fdef == ("table_builder_append", "bodo.libs.table_builder"):
            state_def = guard(
                _get_state_defining_call,
                self.func_ir,
                rhs.args[0],
                ("init_table_builder_state", "bodo.libs.table_builder"),
            )
            if (
                state_def is not None
                and self.calltypes[state_def].args[-1] == types.Omitted(False)
                and guard(self._is_unified_streaming_output, rhs.args[1])
            ):
                set_last_arg_to_true(self, state_def)

        # Set input_dicts_unified flag if input is already unified in previous streaming
        # operator
        if fdef == (
            "snowflake_writer_append_table",
            "bodo.io.snowflake_write",
        ):
            state_def = guard(
                _get_state_defining_call,
                self.func_ir,
                rhs.args[0],
                ("snowflake_writer_init", "bodo.io.snowflake_write"),
            )
            if (
                state_def is not None
                and self.calltypes[state_def].args[-2] == types.Omitted(False)
                and guard(self._is_unified_streaming_output, rhs.args[1])
            ):
                set_2nd_to_last_arg_to_true(self, state_def)

        if fdef == (
            "iceberg_writer_append_table",
            "bodo.io.iceberg.stream_iceberg_write",
        ):
            state_def = guard(
                _get_state_defining_call,
                self.func_ir,
                rhs.args[0],
                ("iceberg_writer_init", "bodo.io.iceberg.stream_iceberg_write"),
            )
            if (
                state_def is not None
                and self.calltypes[state_def].args[-2] == types.Omitted(False)
                and guard(self._is_unified_streaming_output, rhs.args[1])
            ):
                set_2nd_to_last_arg_to_true(self, state_def)

        if fdef == (
            "parquet_writer_append_table",
            "bodo.io.stream_parquet_write",
        ):
            state_def = guard(
                _get_state_defining_call,
                self.func_ir,
                rhs.args[0],
                ("parquet_writer_init", "bodo.io.stream_parquet_write"),
            )
            if (
                state_def is not None
                and self.calltypes[state_def].args[-2] == types.Omitted(False)
                and guard(self._is_unified_streaming_output, rhs.args[1])
            ):
                set_2nd_to_last_arg_to_true(self, state_def)

        # support matplot lib calls
        if "bodo.libs.matplotlib_ext" in sys.modules:
            # matplotlib.pyplot functions
            if (
                func_mod == "matplotlib.pyplot"
                and func_name in bodo.libs.matplotlib_ext.mpl_plt_kwargs_funcs
            ):
                return self._run_call_matplotlib(lhs, rhs, func_mod, func_name)

            # matplotlib methods
            if isinstance(func_mod, ir.Var):
                # axes
                if (
                    self.typemap[func_mod.name] == types.mpl_axes_type
                    and func_name in bodo.libs.matplotlib_ext.mpl_axes_kwargs_funcs
                ):
                    return self._run_call_matplotlib(lhs, rhs, func_mod, func_name)

                # figs
                if (
                    self.typemap[func_mod.name] == types.mpl_figure_type
                    and func_name in bodo.libs.matplotlib_ext.mpl_figure_kwargs_funcs
                ):
                    return self._run_call_matplotlib(lhs, rhs, func_mod, func_name)

        # Handle inlining to avoid conflict with Numba np.hstack definition
        if fdef == ("hstack", "numpy"):
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}

            impl = bodo.libs.array_kernels.np_hstack(*arg_typs, **kw_typs)
            return replace_func(
                self,
                impl,
                rhs.args,
                pysig=numba.core.utils.pysignature(impl),
                kws=dict(rhs.kws),
            )

        # Handle inlining 1D Numpy arrays
        if fdef == ("any", "numpy"):
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}

            impl = bodo.libs.array_kernels.np_any(*arg_typs, **kw_typs)
            if impl is not None:
                # This section is never entered by current code. It is a safety
                # net if a 1D array Numba implementation is taken
                return replace_func(
                    self,
                    impl,
                    rhs.args,
                    pysig=numba.core.utils.pysignature(impl),
                    kws=dict(rhs.kws),
                )  # pragma: no cover

        # Handle inlining 1D Numpy arrays
        if fdef == ("all", "numpy"):
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}

            impl = bodo.libs.array_kernels.np_all(*arg_typs, **kw_typs)
            if impl is not None:
                # This section is never entered by current code. It is a safety
                # net if a 1D array Numba implementation is taken
                return replace_func(
                    self,
                    impl,
                    rhs.args,
                    pysig=numba.core.utils.pysignature(impl),
                    kws=dict(rhs.kws),
                )  # pragma: no cover

        if fdef == ("get_int_arr_data", "bodo.libs.int_arr_ext"):
            var_def = guard(get_definition, self.func_ir, rhs.args[0])
            call_def = guard(find_callname, self.func_ir, var_def, self.typemap)
            if call_def == ("init_integer_array", "bodo.libs.int_arr_ext"):
                assign.value = var_def.args[0]
                return [assign]

        if fdef == ("get_int_arr_bitmap", "bodo.libs.int_arr_ext"):
            var_def = guard(get_definition, self.func_ir, rhs.args[0])
            call_def = guard(find_callname, self.func_ir, var_def, self.typemap)
            if call_def == ("init_integer_array", "bodo.libs.int_arr_ext"):
                assign.value = var_def.args[1]
                return [assign]

        if fdef == ("get_float_arr_data", "bodo.libs.float_arr_ext"):
            var_def = guard(get_definition, self.func_ir, rhs.args[0])
            call_def = guard(find_callname, self.func_ir, var_def, self.typemap)
            if call_def == ("init_float_array", "bodo.libs.float_arr_ext"):
                assign.value = var_def.args[0]
                return [assign]

        if fdef == ("get_float_arr_bitmap", "bodo.libs.float_arr_ext"):
            var_def = guard(get_definition, self.func_ir, rhs.args[0])
            call_def = guard(find_callname, self.func_ir, var_def, self.typemap)
            if call_def == ("init_float_array", "bodo.libs.float_arr_ext"):
                assign.value = var_def.args[1]
                return [assign]

        # inline IntegerArrayType.astype()
        if (
            isinstance(func_mod, ir.Var)
            and isinstance(self.typemap[func_mod.name], IntegerArrayType)
            and func_name in ("astype", "sum")
        ):
            rhs.args.insert(0, func_mod)
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}

            impl = getattr(bodo.libs.int_arr_ext, "overload_int_arr_" + func_name)(
                *arg_typs, **kw_typs
            )
            return replace_func(
                self,
                impl,
                rhs.args,
                pysig=numba.core.utils.pysignature(impl),
                kws=dict(rhs.kws),
            )

        # inline FloatingArrayType.astype()
        if (
            isinstance(func_mod, ir.Var)
            and isinstance(self.typemap[func_mod.name], FloatingArrayType)
            and func_name in ("astype", "sum")
        ):  # pragma: no cover
            rhs.args.insert(0, func_mod)
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}

            impl = getattr(bodo.libs.float_arr_ext, "overload_float_arr_" + func_name)(
                *arg_typs, **kw_typs
            )
            return replace_func(
                self,
                impl,
                rhs.args,
                pysig=numba.core.utils.pysignature(impl),
                kws=dict(rhs.kws),
            )

        # inline BooleanArray.astype()
        if (
            isinstance(func_mod, ir.Var)
            and self.typemap[func_mod.name] == boolean_array_type
            and func_name in ("astype",)
        ):
            rhs.args.insert(0, func_mod)
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}

            impl = getattr(bodo.libs.bool_arr_ext, "overload_bool_arr_" + func_name)(
                *arg_typs, **kw_typs
            )
            return replace_func(
                self,
                impl,
                rhs.args,
                pysig=numba.core.utils.pysignature(impl),
                kws=dict(rhs.kws),
            )

        # inline CategoricalArrayType.astype()
        # TODO(ehsan): inline Series.astype() using inline="always" and avoid this
        if (
            isinstance(func_mod, ir.Var)
            and isinstance(self.typemap[func_mod.name], CategoricalArrayType)
            and func_name == "astype"
        ):
            rhs.args.insert(0, func_mod)
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}

            impl = bodo.hiframes.pd_categorical_ext.overload_cat_arr_astype(
                *arg_typs, **kw_typs
            )
            return replace_func(
                self,
                impl,
                rhs.args,
                pysig=numba.core.utils.pysignature(impl),
                kws=dict(rhs.kws),
            )

        # inline pd.CategoricalArrayType()
        if fdef in (("Categorical", "pandas"), ("Categorical", "bodo.pandas")):
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}

            impl = bodo.hiframes.pd_categorical_ext.pd_categorical_overload(
                *arg_typs, **kw_typs
            )
            return replace_func(
                self,
                impl,
                rhs.args,
                pysig=numba.core.utils.pysignature(impl),
                kws=dict(rhs.kws),
            )

        # inlining SeriesStrMethod methods is necessary since they may be used in
        # df.query() which is handled in dataframe pass currently (TODO: use overload)
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

            return replace_func(
                self,
                impl,
                rhs.args,
                pysig=numba.core.utils.pysignature(impl),
                kws=dict(rhs.kws),
            )

        # Inlining astype with nullable tuple can lead to to_datetime
        # or to_timedelta remaining in the IR. Since we have already
        # passed the inlining stage we need to manually inline.
        # BodoSQL also uses these and could be inside of CASE so needs inlined.
        # see test_datetime_fns.py::test_to_date_scalar
        if fdef in (
            ("to_datetime", "pandas"),
            ("to_timedelta", "pandas"),
            ("to_datetime", "bodo.pandas"),
            ("to_timedelta", "bodo.pandas"),
            ("to_datetime", "pandas.core.tools.datetimes"),
            ("to_timedelta", "pandas.core.tools.timedeltas"),
        ):
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}
            impl = getattr(bodo.hiframes.pd_timestamp_ext, f"overload_{fdef[0]}")(
                *arg_typs, **kw_typs
            )
            return replace_func(
                self,
                impl,
                rhs.args,
                pysig=numba.core.utils.pysignature(impl),
                kws=dict(rhs.kws),
            )

        # replace _get_type_max_value(arr.dtype) since parfors
        # arr.dtype transformation produces invalid code for dt64
        # TODO: min
        if fdef in (
            ("_get_type_max_value", "bodo.transforms.series_pass"),
            ("_get_type_max_value", "bodo.hiframes.series_kernels"),
        ):
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}
            impl = getattr(
                bodo.hiframes.series_kernels, "_get_type_max_value_overload"
            )(*arg_typs, **kw_typs)
            return replace_func(
                self,
                impl,
                rhs.args,
                pysig=numba.core.utils.pysignature(impl),
                kws=dict(rhs.kws),
            )

        if fdef in (
            ("_get_type_min_value", "bodo.transforms.series_pass"),
            ("_get_type_min_value", "bodo.hiframes.series_kernels"),
        ):
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}
            impl = getattr(
                bodo.hiframes.series_kernels, "_get_type_min_value_overload"
            )(*arg_typs, **kw_typs)
            return replace_func(
                self,
                impl,
                rhs.args,
                pysig=numba.core.utils.pysignature(impl),
                kws=dict(rhs.kws),
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
                    ) or isinstance(t, types.Integer), (
                        "only simple slice without step supported for reading hdf5"
                    )

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
                    func_text += f"  start_{i} = 0\n"
                    func_text += (
                        f"  size_{i} = bodo.io.h5_api.h5size(dset_id, np.int32({i}))\n"
                    )
                    if i < len(index_types):
                        if isinstance(index_types[i], types.SliceType):
                            func_text += "  slice_idx_{0} = numba.cpython.unicode._normalize_slice(index{1}, size_{0})\n".format(
                                i,
                                f"[{i}]"
                                if isinstance(index_tp, types.BaseTuple)
                                else "",
                            )
                            func_text += f"  start_{i} = slice_idx_{i}.start\n"
                            func_text += f"  size_{i} = numba.cpython.unicode._slice_span(slice_idx_{i})\n"
                        else:
                            assert isinstance(
                                types.unliteral(index_types[i]), types.Integer
                            )
                            func_text += "  start_{} = index{}\n".format(
                                i,
                                f"[{i}]"
                                if isinstance(index_tp, types.BaseTuple)
                                else "",
                            )
                            func_text += f"  size_{i} = 1\n"

            # array dimensions can be less than dataset due to integer selection
            func_text += "  arr_shape = ({},)\n".format(
                ", ".join(
                    [
                        f"size_{i}"
                        for i in range(ndim)
                        if not (
                            i < len(index_types)
                            and isinstance(
                                types.unliteral(index_types[i]), types.Integer
                            )
                        )
                    ]
                )
            )
            func_text += f"  A = np.empty(arr_shape, np.{dtype_str})\n"

            func_text += "  start_tup = ({},)\n".format(
                ", ".join([f"start_{i}" for i in range(ndim)])
            )
            func_text += "  count_tup = ({},)\n".format(
                ", ".join([f"size_{i}" for i in range(ndim)])
            )

            if filter_read:
                func_text += f"  err = bodo.io.h5_api.h5read_filter(dset_id, np.int32({ndim}), start_tup, count_tup, 0, A, read_indices)\n"
            else:
                func_text += f"  err = bodo.io.h5_api.h5read(dset_id, np.int32({ndim}), start_tup, count_tup, 0, A)\n"
            func_text += "  return A\n"

            loc_vars = {}
            exec(func_text, {}, loc_vars)
            _h5_read_impl = loc_vars["_h5_read_impl"]
            return replace_func(self, _h5_read_impl, rhs.args)

        if fdef in (("DatetimeIndex", "pandas"), ("DatetimeIndex", "bodo.pandas")):
            return self._run_pd_DatetimeIndex(assign, assign.target, rhs)

        if fdef in (("TimedeltaIndex", "pandas"), ("TimedeltaIndex", "bodo.pandas")):
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}
            impl = bodo.hiframes.pd_index_ext.pd_timedelta_index_overload(
                *arg_typs, **kw_typs
            )
            return replace_func(
                self, impl, rhs.args, pysig=self.calltypes[rhs].pysig, kws=dict(rhs.kws)
            )

        if fdef in (("Series", "pandas"), ("Series", "bodo.pandas")):
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}

            impl = bodo.hiframes.pd_series_ext.pd_series_overload(*arg_typs, **kw_typs)
            return replace_func(
                self, impl, rhs.args, pysig=self.calltypes[rhs].pysig, kws=dict(rhs.kws)
            )

        # inline pd.notna() since used in BodoSQL CASE codegen in dataframe pass (which
        # doesn't have inline pass)
        if (
            fdef
            in (
                ("notna", "pandas"),
                ("notnull", "pandas"),
                ("notna", "bodo.pandas"),
                ("notnull", "bodo.pandas"),
            )
            and not rhs.kws
        ):
            impl = bodo.hiframes.dataframe_impl.overload_notna(
                self.typemap[rhs.args[0].name]
            )
            return compile_func_single_block(
                impl,
                rhs.args,
                assign.target,
                self,
            )

        # inline np.var()/np.std() on nullable float arrays to be parallelized
        if (
            fdef in (("std", "numpy"), ("var", "numpy"))
            and not rhs.kws
            and len(rhs.args) == 1
            and isinstance(self.typemap[rhs.args[0].name], FloatingArrayType)
        ):
            impl = getattr(bodo.libs.float_arr_ext, "overload_" + func_name)(
                self.typemap[rhs.args[0].name]
            )
            return compile_func_single_block(
                impl,
                rhs.args,
                assign.target,
                self,
            )

        # pattern match pd.isna(A[i]) and replace it with array_kernels.isna(A, i)
        if fdef in (
            ("isna", "pandas"),
            ("isnull", "pandas"),
            ("isna", "bodo.pandas"),
            ("isnull", "bodo.pandas"),
        ):
            obj = get_call_expr_arg(fdef[0], rhs.args, dict(rhs.kws), 0, "obj")
            obj_def = guard(get_definition, self.func_ir, obj)
            # Timestamp/Timedelta values are boxed before passing to UDF
            if guard(find_callname, self.func_ir, obj_def, self.typemap) in (
                ("convert_datetime64_to_timestamp", "bodo.hiframes.pd_timestamp_ext"),
                (
                    "convert_numpy_timedelta64_to_pd_timedelta",
                    "bodo.hiframes.pd_timestamp_ext",
                ),
            ):
                obj_def = guard(get_definition, self.func_ir, obj_def.args[0])
            # TODO: Remove static_getitem from Numba
            if (is_expr(obj_def, "getitem") or is_expr(obj_def, "static_getitem")) and (
                is_array_typ(self.typemap[obj_def.value.name], False)
                or isinstance(
                    self.typemap[obj_def.value.name], bodo.types.NullableTupleType
                )
            ):
                if is_expr(obj_def, "getitem"):
                    index_var = obj_def.index
                else:
                    index_var = obj_def.index_var
                return compile_func_single_block(
                    eval("lambda A, i: bodo.libs.array_kernels.isna(A, i)"),
                    (obj_def.value, index_var),
                    assign.target,
                    self,
                )

        # replace isna early to enable more optimization in PA
        # TODO: handle more types
        if fdef == ("isna", "bodo.libs.array_kernels"):
            arr = rhs.args[0]
            arr_typ = self.typemap[arr.name]
            if isinstance(arr_typ, types.Array):
                arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
                impl = bodo.libs.array_kernels.overload_isna(*arg_typs)
                return replace_func(self, impl, rhs.args)
            # Optimize out the nullable tuple if using a static index
            if isinstance(arr_typ, bodo.types.NullableTupleType) and isinstance(
                self.typemap[rhs.args[1].name], types.IntegerLiteral
            ):
                val_def = guard(get_definition, self.func_ir, arr)
                if isinstance(val_def, ir.Expr) and val_def.op == "call":
                    call_name = guard(
                        find_callname, self.func_ir, val_def, self.typemap
                    )
                    if call_name == (
                        "build_nullable_tuple",
                        "bodo.libs.nullable_tuple_ext",
                    ):
                        null_values = val_def.args[1]
                        null_idx = get_overload_const_int(
                            self.typemap[rhs.args[1].name]
                        )
                        # Replace the call with a getitem on the null values tuple. This
                        # should enable optimizing out the series.
                        return compile_func_single_block(
                            eval(f"lambda tup: tup[{null_idx}]"),
                            (null_values,),
                            assign.target,
                            self,
                        )
            return [assign]

        # Inline pd.Timestamp() calls on constant strings to avoid going to objmode
        # (generated by BodoSQL for constant datetimes)
        if (
            fdef in (("Timestamp", "pandas"), ("Timestamp", "bodo.pandas"))
            and len(rhs.args) == 1
            and not rhs.kws
            and is_overload_constant_str(self.typemap[rhs.args[0].name])
        ):
            time_val = get_overload_const_str(self.typemap[rhs.args[0].name])
            gb_name = f"_bodo_timestamp_const_{ir_utils.next_label()}"
            assign.value = ir.Global(gb_name, pd.Timestamp(time_val), rhs.loc)
            return [assign]

        if fdef == ("argsort", "bodo.hiframes.series_impl"):
            lhs = assign.target
            data = rhs.args[0]
            nodes = []

            func_text = (
                "def _get_indices(S):\n    n = len(S)\n    return np.arange(n)\n"
            )

            loc_vars = {}
            exec(func_text, globals(), loc_vars)
            nodes += compile_func_single_block(
                loc_vars["_get_indices"], (data,), None, self
            )
            index_var = nodes[-1].target

            # dummy output data arrays for results
            out_data = ir.Var(lhs.scope, mk_unique_var(data.name + "_data"), lhs.loc)
            self.typemap[out_data.name] = self.typemap[data.name]

            in_vars = [data, index_var]
            out_vars = [out_data, lhs]
            key_inds = (0,)
            inplace = False
            ascending = True
            na_position = "last"

            # Sort node
            nodes.append(
                bodo.ir.sort.Sort(
                    data.name,
                    lhs.name,
                    in_vars,
                    out_vars,
                    key_inds,
                    inplace,
                    lhs.loc,
                    ascending,
                    na_position,
                )
            )

            return nodes

        # inline StringArray.astype()
        if (
            isinstance(func_mod, ir.Var)
            and is_str_arr_type(self.typemap[func_mod.name])
            and func_name == "astype"
        ):
            rhs.args.insert(0, func_mod)
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}

            impl = bodo.libs.str_arr_ext.overload_str_arr_astype(*arg_typs, **kw_typs)
            return replace_func(
                self,
                impl,
                rhs.args,
                pysig=numba.core.utils.pysignature(impl),
                kws=dict(rhs.kws),
            )

        if fdef == ("series_filter_bool", "bodo.hiframes.series_impl"):
            nodes = []
            in_arr = rhs.args[0]
            bool_arr = rhs.args[1]
            if is_series_type(self.typemap[in_arr.name]):
                in_arr = self._get_series_data(in_arr, nodes)
            if is_series_type(self.typemap[bool_arr.name]):
                bool_arr = self._get_series_data(bool_arr, nodes)

            return replace_func(
                self,
                series_kernels._column_filter_impl,
                [in_arr, bool_arr],
                pre_nodes=nodes,
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

        if fdef == ("get_index_data", "bodo.hiframes.pd_index_ext"):
            var_def = guard(get_definition, self.func_ir, rhs.args[0])
            call_def = guard(find_callname, self.func_ir, var_def, self.typemap)
            if call_def in (
                ("init_datetime_index", "bodo.hiframes.pd_index_ext"),
                ("init_timedelta_index", "bodo.hiframes.pd_index_ext"),
                ("init_binary_str_index", "bodo.hiframes.pd_index_ext"),
                ("init_numeric_index", "bodo.hiframes.pd_index_ext"),
                ("init_categorical_index", "bodo.hiframes.pd_index_ext"),
                ("init_heter_index", "bodo.hiframes.pd_index_ext"),
            ):
                assign.value = var_def.args[0]
            return [assign]

        if fdef == ("get_index_name", "bodo.hiframes.pd_index_ext"):
            var_def = guard(get_definition, self.func_ir, rhs.args[0])
            call_def = guard(find_callname, self.func_ir, var_def, self.typemap)
            if (
                call_def
                in (
                    ("init_datetime_index", "bodo.hiframes.pd_index_ext"),
                    ("init_timedelta_index", "bodo.hiframes.pd_index_ext"),
                    ("init_binary_str_index", "bodo.hiframes.pd_index_ext"),
                    ("init_numeric_index", "bodo.hiframes.pd_index_ext"),
                    ("init_categorical_index", "bodo.hiframes.pd_index_ext"),
                    ("init_heter_index", "bodo.hiframes.pd_index_ext"),
                )
                and len(var_def.args) > 1
            ):
                assign.value = var_def.args[1]
            elif (
                call_def == ("init_range_index", "bodo.hiframes.pd_index_ext")
                and len(var_def.args) > 3
            ):
                assign.value = var_def.args[3]
            return [assign]

        # optimize out decode_if_dict_array() if not needed
        if (
            fdef == ("decode_if_dict_array", "bodo.utils.typing")
            and self.typemap[rhs.args[0].name] == self.typemap[assign.target.name]
        ):
            assign.value = rhs.args[0]
            return [assign]

        # optimize out unwrap_tz_array() if not needed
        if (
            fdef == ("unwrap_tz_array", "bodo.libs.pd_datetime_arr_ext")
            and self.typemap[rhs.args[0].name] == self.typemap[assign.target.name]
        ):
            assign.value = rhs.args[0]
            return [assign]

        # pd.DataFrame() calls init_series for even Series since it's untyped
        # remove the call since it is invalid for analysis here
        # XXX remove when df pass is typed? (test_pass_series2)
        if fdef == ("init_series", "bodo.hiframes.pd_series_ext"):
            if isinstance(self.typemap[rhs.args[0].name], SeriesType):
                assign.value = rhs.args[0]
            return [assign]

        if fdef == ("get_series_data", "bodo.hiframes.pd_series_ext"):
            # or other functions, using any reference to payload
            var_def = guard(get_definition, self.func_ir, rhs.args[0])
            call_def = guard(find_callname, self.func_ir, var_def, self.typemap)
            if call_def == ("init_series", "bodo.hiframes.pd_series_ext"):
                assign.value = var_def.args[0]
            return [assign]

        if fdef == ("get_series_index", "bodo.hiframes.pd_series_ext"):
            # or other functions, using any reference to payload
            var_def = guard(get_definition, self.func_ir, rhs.args[0])
            call_def = guard(find_callname, self.func_ir, var_def, self.typemap)
            if (
                call_def == ("init_series", "bodo.hiframes.pd_series_ext")
                and len(var_def.args) > 1
            ):
                assign.value = var_def.args[1]
            return [assign]

        if fdef == ("get_series_name", "bodo.hiframes.pd_series_ext"):
            # TODO: make sure name is not altered
            var_def = guard(get_definition, self.func_ir, rhs.args[0])
            call_def = guard(find_callname, self.func_ir, var_def, self.typemap)
            if (
                call_def == ("init_series", "bodo.hiframes.pd_series_ext")
                and len(var_def.args) > 2
            ):
                assign.value = var_def.args[2]
            return [assign]

        if func_mod == "bodo.hiframes.rolling":
            return self._run_call_rolling(assign, assign.target, rhs, func_name)

        if fdef == ("empty_like", "numpy"):
            return self._handle_empty_like(assign, lhs, rhs)

        if fdef == ("full", "numpy"):
            return self._handle_np_full(assign, lhs, rhs)

        if func_mod == "bodo.libs.array_ops" and "array_op_" in func_name:
            return self._handle_array_ops(assign, rhs, func_name)

        if fdef == ("alloc_type", "bodo.utils.utils"):
            impl = bodo.utils.utils.overload_alloc_type(
                *tuple(self.typemap[v.name] for v in rhs.args)
            )
            dict_ref_type = (
                self.typemap[rhs.args[3].name] if len(rhs.args) > 3 else types.none
            )
            # create new functions for cases that need dtype since 'dtype' becomes a
            # freevar in overload and doesn't work properly currently.
            # TODO: fix freevar support
            typ = self.typemap[rhs.args[1].name]
            if isinstance(typ, types.TypeRef):
                typ = typ.instance_type
            dtype = None
            # Add dict_ref_arr=None to args if not provided to avoid errors
            args = rhs.args
            nodes = []
            if len(args) == 3:
                scope = assign.target.scope
                none_var = ir.Var(scope, mk_unique_var("none_var"), rhs.loc)
                self.typemap[none_var.name] = types.none
                args.append(none_var)
                nodes.append(ir.Assign(ir.Const(None, rhs.loc), none_var, rhs.loc))

            # nullable int array
            if isinstance(typ, IntegerArrayType):  # pragma: no cover
                dtype = typ.dtype
                impl = eval(
                    "lambda n, t, s=None, dict_ref_arr=None: bodo.libs.int_arr_ext.alloc_int_array(n, _dtype)"
                )
            elif isinstance(typ, FloatingArrayType):  # pragma: no cover
                dtype = typ.dtype
                impl = eval(
                    "lambda n, t, s=None, dict_ref_arr=None: bodo.libs.float_arr_ext.alloc_float_array(n, _dtype)"
                )
            elif isinstance(typ, types.Array):
                dtype = typ.dtype
                # avoid dt64 errors in np.empty, TODO: fix Numba
                if dtype == types.NPDatetime("ns"):
                    dtype = np.dtype("datetime64[ns]")
                impl = eval(
                    "lambda n, t, s=None, dict_ref_arr=None: np.empty(n, _dtype)"
                )
            elif isinstance(typ, ArrayItemArrayType):
                dtype = typ.dtype
                if is_overload_none(dict_ref_type):
                    impl = eval(
                        "lambda n, t, s=None, dict_ref_arr=None: bodo.libs.array_item_arr_ext.pre_alloc_array_item_array(n, s, _dtype)"
                    )
                else:
                    impl = eval(
                        "lambda n, t, s=None, dict_ref_arr=None: bodo.libs.array_item_arr_ext.pre_alloc_array_item_array(n, s, bodo.libs.array_item_arr_ext.get_data(dict_ref_arr))"
                    )
            elif isinstance(typ, CategoricalArrayType):
                if isinstance(self.typemap[rhs.args[1].name], types.TypeRef):
                    # If we have a type ref we must have types that are compile time constants
                    if typ.dtype.categories is None:
                        raise BodoError(
                            "UDFs that return Categorical values must have categories known at compile time."
                        )
                    # create the new categorical dtype inside the function instead of passing as
                    # constant. This avoids constant lowered Index inside the dtype, which can
                    # be slow since it cannot have a dictionary.
                    # see https://github.com/bodo-ai/Bodo/pull/3563
                    is_ordered = typ.dtype.ordered
                    int_type = typ.dtype.int_type
                    new_cats_arr = bodo.utils.utils.create_categorical_type(
                        typ.dtype.categories, typ.dtype.data.data, is_ordered
                    )
                    new_cats_tup = MetaType(typ.dtype.categories)
                    dtype = "bodo.hiframes.pd_categorical_ext.init_cat_dtype(bodo.utils.conversion.index_from_array(new_cats_arr), is_ordered, int_type, new_cats_tup)"
                    impl = eval(
                        f"lambda n, t, s=None, dict_ref_arr=None: bodo.hiframes.pd_categorical_ext.alloc_categorical_array(n, {dtype})"
                    )
                    return nodes + compile_func_single_block(
                        impl,
                        args,
                        assign.target,
                        self,
                        extra_globals={
                            "is_ordered": is_ordered,
                            "new_cats_arr": new_cats_arr,
                            "new_cats_tup": new_cats_tup,
                            "int_type": int_type,
                        },
                    )
                else:
                    # TODO: Fix the infrastructure so types will match when input-type == output-type
                    impl = eval(
                        "lambda n, t, s=None, dict_ref_arr=None: bodo.hiframes.pd_categorical_ext.alloc_categorical_array(n, t.dtype)"
                    )
            elif isinstance(typ, StructArrayType):
                dtypes = typ.data
                names = typ.names
                return nodes + compile_func_single_block(
                    eval(
                        "lambda n, t, s=None, dict_ref_arr=None: bodo.libs.struct_arr_ext.pre_alloc_struct_array(n, s, _dtypes, _names, dict_ref_arr)"
                    ),
                    args,
                    assign.target,
                    self,
                    extra_globals={"_dtypes": dtypes, "_names": names},
                )
            elif isinstance(typ, MapArrayType):
                struct_typ = StructArrayType(
                    (typ.key_arr_type, typ.value_arr_type), ("key", "value")
                )
                return nodes + compile_func_single_block(
                    eval(
                        "lambda n, t, s=None, dict_ref_arr=None: bodo.libs.map_arr_ext.pre_alloc_map_array(n, s, _struct_type, dict_ref_arr)"
                    ),
                    args,
                    assign.target,
                    self,
                    extra_globals={"_struct_type": struct_typ},
                )
            elif isinstance(typ, TupleArrayType):
                dtypes = typ.data
                return nodes + compile_func_single_block(
                    eval(
                        "lambda n, t, s=None, dict_ref_arr=None: bodo.libs.tuple_arr_ext.pre_alloc_tuple_array(n, s, _dtypes)"
                    ),
                    args,
                    assign.target,
                    self,
                    extra_globals={"_dtypes": dtypes},
                )
            elif isinstance(typ, DecimalArrayType):
                precision = typ.dtype.precision
                scale = typ.dtype.scale
                return nodes + compile_func_single_block(
                    eval(
                        "lambda n, t, s=None, dict_ref_arr=None: bodo.libs.decimal_arr_ext.alloc_decimal_array(n, _precision, _scale)"
                    ),
                    args,
                    assign.target,
                    self,
                    extra_globals={"_precision": precision, "_scale": scale},
                )
            elif isinstance(typ, bodo.types.DatetimeArrayType):
                tz = typ.tz
                return nodes + compile_func_single_block(
                    eval(
                        "lambda n, t, s=None, dict_ref_arr=None: bodo.libs.pd_datetime_arr_ext.alloc_pd_datetime_array(n, _tz)"
                    ),
                    args,
                    assign.target,
                    self,
                    extra_globals={"_tz": tz},
                )
            elif isinstance(typ, bodo.types.TimeArrayType):
                precision = typ.precision
                return nodes + compile_func_single_block(
                    eval(
                        "lambda n, t, s=None, dict_ref_arr=None: bodo.hiframes.time_ext.alloc_time_array(n, _precision)"
                    ),
                    args,
                    assign.target,
                    self,
                    extra_globals={"_precision": precision},
                )
            elif typ == bodo.types.timestamptz_array_type:
                return nodes + compile_func_single_block(
                    eval(
                        "lambda n, t, s=None, dict_ref_arr=None: bodo.hiframes.timestamptz_ext.alloc_timestamptz_array(n)"
                    ),
                    args,
                    assign.target,
                    self,
                )

            return nodes + compile_func_single_block(
                impl, args, assign.target, self, extra_globals={"_dtype": dtype}
            )

        if isinstance(func_mod, ir.Var) and is_series_type(self.typemap[func_mod.name]):
            return self._run_call_series(
                assign, assign.target, rhs, func_mod, func_name
            )

        if (
            isinstance(func_mod, ir.Var)
            and isinstance(self.typemap[func_mod.name], DatetimeIndexType)
            and func_name in {"min", "max"}
        ):
            rhs.args.insert(0, func_mod)
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}
            if func_name == "min":
                impl = bodo.hiframes.pd_index_ext.overload_datetime_index_min(
                    *arg_typs, **kw_typs
                )
            else:
                impl = bodo.hiframes.pd_index_ext.overload_datetime_index_max(
                    *arg_typs, **kw_typs
                )
            stub = eval("lambda dti, axis=None, skipna=True: None")
            return replace_func(
                self,
                impl,
                rhs.args,
                pysig=numba.core.utils.pysignature(stub),
                kws=dict(rhs.kws),
            )

        if isinstance(func_mod, ir.Var) and bodo.hiframes.pd_index_ext.is_pd_index_type(
            self.typemap[func_mod.name]
        ):
            return self._run_call_index(assign, assign.target, rhs, func_mod, func_name)

        if fdef == ("init_dataframe", "bodo.hiframes.pd_dataframe_ext"):
            return [assign]

        if fdef == ("len", "builtins") and is_series_type(
            self.typemap[rhs.args[0].name]
        ):
            return replace_func(
                self,
                eval("lambda S: len(bodo.hiframes.pd_series_ext.get_series_data(S))"),
                rhs.args,
            )

        if (
            fdef == ("len", "builtins")
            and not isinstance(self.typemap[rhs.args[0].name], RangeIndexType)
            and bodo.hiframes.pd_index_ext.is_pd_index_type(
                self.typemap[rhs.args[0].name]
            )
        ):
            nodes = []
            arr = self._get_index_data(rhs.args[0], nodes)
            func_text = """def f(df_arr):\n  return len(df_arr)\n"""
            loc_vars = {}
            exec(func_text, globals(), loc_vars)
            return replace_func(self, loc_vars["f"], [arr], pre_nodes=nodes)

        # optimize len(A[i]) -> get_str_arr_str_length(A, i)
        if (
            fdef == ("len", "builtins")
            and self.typemap[rhs.args[0].name] == string_type
        ):
            val_def = guard(get_definition, self.func_ir, rhs.args[0])
            if (
                is_expr(val_def, "getitem") or is_expr(val_def, "static_getitem")
            ) and self.typemap[val_def.value.name] == string_array_type:
                nodes = []
                val_idx = get_getsetitem_index_var(val_def, self.typemap, nodes)
                return nodes + compile_func_single_block(
                    eval(
                        "lambda A, i: bodo.libs.str_arr_ext.get_str_arr_str_length(A, i)"
                    ),
                    (val_def.value, val_idx),
                    assign.target,
                    self,
                )

        # inline conversion functions to enable optimization
        if func_mod == "bodo.utils.conversion" and func_name not in (
            "flatten_array",
            "make_replicated_array",
            "list_to_array",
            "np_to_nullable_array",
        ):
            # TODO: use overload IR inlining when available
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}
            overload_func = getattr(bodo.utils.conversion, "overload_" + func_name)
            impl = overload_func(*arg_typs, **kw_typs)
            return replace_func(
                self, impl, rhs.args, pysig=self.calltypes[rhs].pysig, kws=dict(rhs.kws)
            )

        if func_mod == "bodo.libs.distributed_api" and func_name in (
            "dist_return",
            "rep_return",
            "threaded_return",
        ):
            return [assign]

        if fdef == ("val_isin_dummy", "bodo.hiframes.pd_dataframe_ext"):
            func_text = (
                ""
                "def impl(S, vals):\n"
                "    arr = bodo.hiframes.pd_series_ext.get_series_data(S)\n"
                "    index = bodo.hiframes.pd_series_ext.get_series_index(S)\n"
                "    numba.parfors.parfor.init_prange()\n"
                "    n = len(arr)\n"
                "    out = bodo.libs.bool_arr_ext.alloc_bool_array(n)\n"
                "    for i in numba.parfors.parfor.internal_prange(n):\n"
                "        out[i] = arr[i] in vals\n"
                "    return bodo.hiframes.pd_series_ext.init_series(out, index)\n"
            )

            loc_vars = {}
            exec(func_text, globals(), loc_vars)
            return replace_func(self, loc_vars["impl"], rhs.args)

        # optimize out trivial slicing on tables
        if (
            func_mod == "bodo.hiframes.table"
            and func_name in ("table_filter", "table_local_filter")
            and guard(is_whole_slice, self.typemap, self.func_ir, rhs.args[1])
        ):
            return [ir.Assign(rhs.args[0], assign.target, assign.loc)]

        if fdef == ("val_notin_dummy", "bodo.hiframes.pd_dataframe_ext"):
            func_text = (
                ""
                "def impl(S, vals):\n"
                "    arr = bodo.hiframes.pd_series_ext.get_series_data(S)\n"
                "    index = bodo.hiframes.pd_series_ext.get_series_index(S)\n"
                "    numba.parfors.parfor.init_prange()\n"
                "    n = len(arr)\n"
                "    out = bodo.libs.bool_arr_ext.alloc_bool_array(n)\n"
                "    for i in numba.parfors.parfor.internal_prange(n):\n"
                "        # TODO: why don't these work?\n"
                "        # out[i] = (arr[i] not in vals)\n"
                "        # out[i] = not (arr[i] in vals)\n"
                "        _in = arr[i] in vals\n"
                "        out[i] = not _in\n"
                "    return bodo.hiframes.pd_series_ext.init_series(out, index)\n"
            )

            loc_vars = {}
            exec(func_text, globals(), loc_vars)
            return replace_func(self, loc_vars["impl"], rhs.args)

        # Optimize out trivial table_astype() to avoid corner case in
        # test_table_astype_copy_false_bug
        if fdef == ("table_astype", "bodo.utils.table_utils"):
            kws = dict(rhs.kws)
            in_table_var = get_call_expr_arg("table_astype", rhs.args, kws, 0, "table")
            new_table_typ_var = get_call_expr_arg(
                "table_astype", rhs.args, kws, 1, "new_table_typ"
            )
            copy_var = get_call_expr_arg("table_astype", rhs.args, kws, 2, "copy")
            _bodo_nan_to_str_var = get_call_expr_arg(
                "table_astype", rhs.args, kws, 3, "_bodo_nan_to_str"
            )
            used_cols_var = get_call_expr_arg(
                "table_astype", rhs.args, kws, 4, "used_cols", default=types.none
            )
            used_cols_type = (
                self.typemap[used_cols_var.name]
                if isinstance(used_cols_var, ir.Var)
                else used_cols_var
            )

            in_table_type = self.typemap[in_table_var.name]
            new_table_type = unwrap_typeref(self.typemap[new_table_typ_var.name])

            if (
                in_table_type == new_table_type
                and is_overload_false(self.typemap[copy_var.name])
                and is_overload_false(self.typemap[_bodo_nan_to_str_var.name])
                and is_overload_none(used_cols_type)
            ):
                assign.value = in_table_var
                return [assign]

        # inline np.where() for 3 arg case with 1D input
        if (
            fdef == ("where", "numpy")
            or fdef == ("where_impl", "bodo.hiframes.series_impl")
        ) and (len(rhs.args) == 3 and self.typemap[rhs.args[0].name].ndim == 1):
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}
            impl = bodo.hiframes.series_impl.overload_np_where(*arg_typs, **kw_typs)
            return replace_func(
                self, impl, rhs.args, pysig=self.calltypes[rhs].pysig, kws=dict(rhs.kws)
            )

        # inline np.where() for 3 arg case with 1D input
        if (
            fdef == ("where", "numpy")
            or fdef == ("where_impl", "bodo.hiframes.series_impl")
        ) and (len(rhs.args) == 3 and self.typemap[rhs.args[0].name].ndim == 1):
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}
            impl = bodo.hiframes.series_impl.overload_np_where(*arg_typs, **kw_typs)
            return replace_func(
                self, impl, rhs.args, pysig=self.calltypes[rhs].pysig, kws=dict(rhs.kws)
            )

        # Manually inline np.where() for 1 arg case with 1D input
        # This is done because Numba doesn't seem to different inline
        # options across overloads
        if (
            fdef == ("where", "numpy")
            or fdef == ("where_impl_one_arg", "bodo.hiframes.series_impl")
            and (
                (len(rhs.args) == 1 and len(dict(rhs.kws)) == 0)
                and (
                    isinstance(self.typemap[rhs.args[0].name], SeriesType)
                    or (
                        bodo.utils.utils.is_array_typ(
                            self.typemap[rhs.args[0].name], False
                        )
                        and self.typemap[rhs.args[0].name].ndim == 1
                    )
                )
            )
        ):
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}
            impl = bodo.hiframes.series_impl.overload_np_where_one_arg(
                *arg_typs, **kw_typs
            )
            return replace_func(
                self, impl, rhs.args, pysig=self.calltypes[rhs].pysig, kws=dict(rhs.kws)
            )

        # Replace with Bodo's parallel implementation (numba's version isn't parallel)
        if fdef == ("nan_to_num", "numpy"):
            in_data = get_call_expr_arg("nan_to_num", rhs.args, dict(rhs.kws), 0, "x")
            in_data_type = self.typemap[in_data.name]
            if (
                bodo.utils.utils.is_array_typ(in_data_type, False)
                and in_data_type.ndim <= 2
            ):
                arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
                kw_typs = {
                    name: self.typemap[v.name] for name, v in dict(rhs.kws).items()
                }
                impl = bodo.libs.array_kernels.np_nan_to_num(*arg_typs, **kw_typs)
                return replace_func(
                    self,
                    impl,
                    rhs.args,
                    pysig=numba.core.utils.pysignature(impl),
                    kws=dict(rhs.kws),
                )

        # Replace np.linspace with Bodo's parallel implementation
        if fdef == ("linspace", "numpy"):
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}
            impl = bodo.libs.array_kernels.np_linspace(*arg_typs, **kw_typs)
            return replace_func(
                self,
                impl,
                rhs.args,
                pysig=numba.core.utils.pysignature(impl),
                kws=dict(rhs.kws),
            )

        # dummy count loop to support len of group in agg UDFs
        if fdef == ("dummy_agg_count", "bodo.ir.aggregate"):
            func_text = (
                ""
                "def impl_agg_c(A):\n"
                "    c = 0\n"
                "    for _ in numba.parfors.parfor.internal_prange(len(A)):\n"
                "        c += 1\n"
                "    return c\n"
            )

            loc_vars = {}
            exec(func_text, globals(), loc_vars)
            return replace_func(
                self,
                loc_vars["impl_agg_c"],
                rhs.args,
                pysig=self.calltypes[rhs].pysig,
                kws=dict(rhs.kws),
            )

        if fdef == ("dt64_arr_sub", "bodo.hiframes.series_impl"):
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}
            impl = bodo.hiframes.series_impl.overload_dt64_arr_sub(*arg_typs, **kw_typs)
            return replace_func(
                self, impl, rhs.args, pysig=self.calltypes[rhs].pysig, kws=dict(rhs.kws)
            )

        # inlining here to avoid errors with Numba 0.54, see comment in the overload
        if fdef == ("_nan_argmax", "bodo.libs.array_kernels"):
            impl = bodo.libs.array_kernels._overload_nan_argmax(
                self.typemap[rhs.args[0].name]
            )
            return replace_func(self, impl, rhs.args)

        # inlining here to avoid errors with Numba 0.54, see comment in the overload
        if fdef == ("_nan_argmin", "bodo.libs.array_kernels"):
            impl = bodo.libs.array_kernels._overload_nan_argmin(
                self.typemap[rhs.args[0].name]
            )
            return replace_func(self, impl, rhs.args)

        # convert Series to Array for unhandled calls
        # TODO check all the functions that get here and handle if necessary
        # e.g. np.sum, prod, min, max, argmin, argmax, mean, var, and std
        if func_mod in ("numpy", "ufunc") and any(
            isinstance(self.typemap[arg.name], SeriesType) for arg in rhs.args
        ):
            return self._fix_unhandled_calls(assign, lhs, rhs)

        # If we encounter the explode kernel see if we can replace it
        if fdef == ("explode", "bodo.libs.array_kernels"):
            assert len(rhs.args) == 2, "invalid explode() args"

            array_src = guard(get_definition, self.func_ir, rhs.args[0].name)
            array_src_call = guard(find_callname, self.func_ir, array_src, self.typemap)
            if array_src_call == (
                "str_split",
                "bodo.libs.str_ext",
            ):
                # Replace the function only if we inherited directly from str_split
                args = array_src.args + [rhs.args[1]]

                return replace_func(
                    self,
                    eval(
                        "lambda arr, pat, n, index_arr: bodo.libs.array_kernels.explode_str_split("
                        "    arr, pat, n, index_arr"
                        ")"
                    ),
                    args,
                )

            if array_src_call == (
                "compute_split_view",
                "bodo.hiframes.split_impl",
            ):
                # Replace the function only if we inherited directly from str_split
                args = array_src.args + [rhs.args[1]]

                return replace_func(
                    self,
                    eval(
                        "lambda arr, pat, index_arr: bodo.libs.array_kernels.explode_str_split("
                        "    arr, pat, -1, index_arr"
                        ")"
                    ),
                    args,
                )

        # Fuse consequtive calls to concat_ws
        if fdef == ("concat_ws", "bodosql.kernels"):
            args = rhs.args
            sep_typ = self.typemap[args[1].name]
            if is_overload_constant_str(sep_typ):
                # We can only fuse concat calls if we are certain that the separator
                # is the same.
                sep = get_overload_const_str(sep_typ)
                # Extract the tuple list to check for any repeat calls to
                tup_list = guard(find_build_tuple, self.func_ir, rhs.args[0].name)
                new_tup_list = []
                # We use an indicator variable instead of the size because concat with a single
                # argument is legal and we still want to optimize out those calls.
                is_fused = False
                if tup_list is not None:
                    for tup_var in tup_list:
                        # Check each variable for an input to concat_ws. Note that we don't do
                        # multiple levels of nesting because we assume these will be fused in
                        # prior steps.
                        tup_var_def = guard(get_definition, self.func_ir, tup_var)
                        func_call = guard(
                            find_callname, self.func_ir, tup_var_def, self.typemap
                        )
                        if (
                            func_call == ("concat_ws", "bodosql.kernels")
                            and is_overload_constant_str(
                                self.typemap[tup_var_def.args[1].name]
                            )
                            and get_overload_const_str(
                                self.typemap[tup_var_def.args[1].name]
                            )
                            == sep
                        ):
                            # The separator is the same so we can fuse this call directly.
                            nested_tup_list = guard(
                                find_build_tuple, self.func_ir, tup_var_def.args[0].name
                            )
                            if nested_tup_list is None:
                                # If we can't find the tuple don't fuse the input.
                                new_tup_list.append(tup_var)
                            else:
                                is_fused = True
                                new_tup_list.extend(nested_tup_list)
                        else:
                            new_tup_list.append(tup_var)
                if is_fused:
                    import bodosql

                    # If we are fusing generate a new function call.
                    replace_vars = ", ".join(
                        f"tup_var{i}" for i in range(len(new_tup_list))
                    )
                    total_vars = tuple(new_tup_list + [rhs.args[1]])
                    glbls = {"bodosql": bodosql}
                    locals = {}
                    return replace_func(
                        self,
                        eval(
                            f"lambda {replace_vars}, sep: bodosql.kernels.concat_ws(({replace_vars},), sep)",
                            glbls,
                            locals,
                        ),
                        total_vars,
                    )

        # inline SparkDataFrame.select() here since inline_closurecall() cannot handle
        # stararg yet. TODO: support
        if (
            "bodo.libs.pyspark_ext" in sys.modules
            and isinstance(func_mod, ir.Var)
            and isinstance(
                self.typemap[func_mod.name], bodo.libs.pyspark_ext.SparkDataFrameType
            )
            and func_name == "select"
        ):
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            impl = bodo.libs.pyspark_ext._gen_df_select(
                self.typemap[func_mod.name], arg_typs, True
            )
            return replace_func(self, impl, (func_mod,))

        # Replace str.format because we need to expand kwargs
        if isinstance(func_mod, ir.Var) and (
            self.typemap[func_mod.name] == bodo.types.string_type
            or is_overload_constant_str(self.typemap[func_mod.name])
        ):
            return self._run_call_string(
                assign, assign.target, rhs, func_mod, func_name
            )

        if isinstance(func_mod, ir.Var) and isinstance(
            self.typemap[func_mod.name], bodo.types.LoggingLoggerType
        ):
            return self._run_call_logger(
                assign, assign.target, rhs, func_mod, func_name
            )

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
            return replace_func(
                self,
                eval(
                    "lambda A, index, name: bodo.hiframes.pd_series_ext.init_series(A, index, name)"
                ),
                (new_lhs, index, name),
                pre_nodes=nodes,
            )
        else:
            nodes.append(assign)
            return nodes

    def _handle_array_ops(self, assign, rhs, func_name):
        """
        Helper function that inlines array op implementations
        for Series/DataFrame that must be manually inlined.
        """
        op = func_name.split("_")[-1]

        if op in (
            "any",
            "all",
            "isna",
            "count",
            "min",
            "max",
            "mean",
            "sum",
            "prod",
            "var",
            "std",
            "quantile",
        ):
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}
            overload_func = getattr(bodo.libs.array_ops, "overload_array_op_" + op)
            impl = overload_func(*arg_typs, **kw_typs)
            return replace_func(
                self,
                impl,
                rhs.args,
                pysig=numba.core.utils.pysignature(impl),
                kws=dict(rhs.kws),
            )
        else:
            return [assign]

    def _run_const_call(self, assign, lhs, rhs, func):
        # replace direct calls to operators with Expr binop nodes to enable
        # ParallelAccelerator transformtions

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

        # inline bool arr operators
        if any(self.typemap[a.name] == boolean_array_type for a in rhs.args):
            n_args = len(rhs.args)
            overload_func = bodo.libs.bool_arr_ext.create_op_overload(func, n_args)
            impl = overload_func(*tuple(self.typemap[a.name] for a in rhs.args))
            return replace_func(self, impl, rhs.args)

        # TODO: handle other calls
        return [assign]

    def _handle_ufuncs(self, ufunc_name, args):
        """hanlde ufuncs with any Series in arguments.
        Output is Series using index and name of original Series.
        """
        np_ufunc = getattr(np, ufunc_name)
        if np_ufunc.nin == 1:
            func_text = (
                ""
                "def impl(S):\n"
                "    arr = bodo.hiframes.pd_series_ext.get_series_data(S)\n"
                "    index = bodo.hiframes.pd_series_ext.get_series_index(S)\n"
                "    name = bodo.hiframes.pd_series_ext.get_series_name(S)\n"
                "    out_arr = _ufunc(arr)\n"
                "    return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)\n"
            )

            # impl.__globals__['_ufunc'] = np_ufunc
            loc_vars = {}
            exec(func_text, globals(), loc_vars)
            return replace_func(
                self, loc_vars["impl"], args, extra_globals={"_ufunc": np_ufunc}
            )
        elif np_ufunc.nin == 2:
            if isinstance(self.typemap[args[0].name], SeriesType):
                func_text = (
                    ""
                    "def impl(S1, S2):\n"
                    "    arr = bodo.hiframes.pd_series_ext.get_series_data(S1)\n"
                    "    index = bodo.hiframes.pd_series_ext.get_series_index(S1)\n"
                    "    name = bodo.hiframes.pd_series_ext.get_series_name(S1)\n"
                    "    other_arr = bodo.utils.conversion.get_array_if_series_or_index(S2)\n"
                    "    out_arr = _ufunc(arr, other_arr)\n"
                    "    return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)\n"
                )

                loc_vars = {}
                exec(func_text, globals(), loc_vars)
                return replace_func(
                    self, loc_vars["impl"], args, extra_globals={"_ufunc": np_ufunc}
                )
            else:
                assert isinstance(self.typemap[args[1].name], SeriesType)

                func_text = (
                    ""
                    "def impl(S1, S2):\n"
                    "    arr = bodo.utils.conversion.get_array_if_series_or_index(S1)\n"
                    "    other_arr = bodo.hiframes.pd_series_ext.get_series_data(S2)\n"
                    "    index = bodo.hiframes.pd_series_ext.get_series_index(S2)\n"
                    "    name = bodo.hiframes.pd_series_ext.get_series_name(S2)\n"
                    "    out_arr = _ufunc(arr, other_arr)\n"
                    "    return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)\n"
                )

                loc_vars = {}
                exec(func_text, globals(), loc_vars)
                return replace_func(
                    self, loc_vars["impl"], args, extra_globals={"_ufunc": np_ufunc}
                )
        else:
            raise BodoError(f"Unsupported numpy ufunc {ufunc_name}")

    def _handle_ufuncs_int_arr(self, ufunc_name, args):
        np_ufunc = getattr(np, ufunc_name)
        overload_func = bodo.libs.int_arr_ext.create_op_overload(np_ufunc, np_ufunc.nin)
        in_typs = tuple(self.typemap[a.name] for a in args)
        impl = overload_func(*in_typs)
        return replace_func(self, impl, args)

    def _handle_ufuncs_float_arr(self, ufunc_name, args):
        np_ufunc = getattr(np, ufunc_name)
        overload_func = bodo.libs.float_arr_ext.create_op_overload(
            np_ufunc, np_ufunc.nin
        )
        in_typs = tuple(self.typemap[a.name] for a in args)
        impl = overload_func(*in_typs)
        return replace_func(self, impl, args)

    def _handle_ufuncs_bool_arr(self, ufunc_name, args):
        np_ufunc = getattr(np, ufunc_name)
        overload_func = bodo.libs.bool_arr_ext.create_op_overload(
            np_ufunc, np_ufunc.nin
        )
        in_typs = tuple(self.typemap[a.name] for a in args)
        impl = overload_func(*in_typs)
        return replace_func(self, impl, args)

    def _run_call_matplotlib(self, lhs, rhs, func_mod, func_name):
        """
        Replace functions in matplotlib that use *args or **kwargs and
        cannot be handled with a regular overload. These implementations
        call into objmode using a helper function and call the original
        matplotlib implementation.

        For functions that display output or modify the figure's data, we
        only call the function on rank 0. For functions that only produce or
        modify configuration objects (i.e. set_xlabel), we call on all ranks
        as it will not impact correctness.
        """
        kws = dict(rhs.kws)
        keys = list(kws.keys())
        header_args = (
            ", ".join(f"e{i}" for i in range(len(rhs.args)))
            + (", " if rhs.args else "")
            + ", ".join(f"e{i + len(rhs.args)}" for i in range(len(keys)))
        )
        arg_names = (
            ", ".join(f"e{i}" for i in range(len(rhs.args)))
            + (", " if rhs.args else "")
            + ", ".join(f"{a}=e{i + len(rhs.args)}" for i, a in enumerate(keys))
        )
        method_var = "matplotlib_obj" if isinstance(func_mod, ir.Var) else ""
        if method_var:
            full_header = method_var + ", " + header_args
        else:
            full_header = header_args

        primary_func = self._generate_mpl_primary_func(
            lhs, rhs, func_name, method_var, full_header, arg_names
        )
        helper_func = self._generate_mpl_helper_func(
            lhs, rhs, func_mod, func_name, method_var, full_header, arg_names
        )
        glbs = {
            f"helper_{func_name}": helper_func,
        }
        args = (
            ([func_mod] if method_var else []) + rhs.args + [kws[key] for key in keys]
        )
        return replace_func(
            self,
            primary_func,
            args,
            extra_globals=glbs,
        )

    def _generate_mpl_primary_func(
        self, lhs, rhs, func_name, method_var, full_header, arg_names
    ):
        """
        Generates the primary function that used in matplotlib function
        replacements. This function gathers data if necessary and calls
        a helper function that enters objmode to avoid objmode limitations.
        """
        # Define the primary function
        func_text = f"def f({full_header}):\n"
        if func_name in bodo.libs.matplotlib_ext.mpl_gather_plots:
            for i in range(len(rhs.args)):
                arg_typ = self.typemap[rhs.args[i].name]
                if bodo.utils.utils.is_array_typ(arg_typ, False):
                    # Gather any data for plotting distributed arrays
                    func_text += f"    e{i} = bodo.gatherv(e{i}, warn_if_rep=False)\n"
            # Only output any plots on rank 0
            func_text += "    if bodo.get_rank() == 0:\n"
            extra_indent = "     "
        else:
            extra_indent = ""
        func_text += f"    {extra_indent}return helper_{func_name}({full_header})\n"
        loc_vars = {}
        exec(func_text, {"bodo": bodo}, loc_vars)
        f = loc_vars["f"]
        return f

    def _generate_mpl_helper_func(
        self, lhs, rhs, func_mod, func_name, method_var, full_header, arg_names
    ):
        """
        Generates the helper that is used in matplotlib function replacements.
        This function calls the original matplotlib implementation in objmode.
        """
        import matplotlib

        if lhs in self.typemap:
            output_type = self.typemap[lhs]
        else:
            output_type = types.none
        type_name = str(output_type)
        if not hasattr(types, type_name):
            type_name = f"objmode_type{ir_utils.next_label()}"
            setattr(types, type_name, output_type)

        func_text = f"def helper_{func_name}({full_header}):\n"
        func_text += (
            f"    with bodo.ir.object_mode.no_warning_objmode(res='{type_name}'):\n"
        )
        if method_var:
            func_text += f"        res = {method_var}.{func_name}({arg_names})\n"
        else:
            func_text += f"        res = {func_mod}.{func_name}({arg_names})\n"
        # if axes is np.array, we convert to nested tuples
        # TODO: Replace with np.array when we can handle objs
        if func_name == "subplots" and isinstance(output_type[1], types.BaseTuple):
            func_text += "        fig, axes = res\n"
            func_text += "        axes = tuple([tuple(elem) if isinstance(elem, np.ndarray) else elem for elem in axes])\n"
            func_text += "        res = (fig, axes)\n"

        func_text += "    return res\n"
        loc_vars = {}
        exec(func_text, {"matplotlib": matplotlib, "bodo": bodo, "np": np}, loc_vars)
        helper_func = numba.njit(loc_vars[f"helper_{func_name}"])
        return helper_func

    def _run_call_string(self, assign, lhs, rhs, string_var, func_name):
        """String operations that need to be replaced because they require kwargs"""
        if func_name == "format":
            kws = dict(rhs.kws)
            keys = list(kws.keys())
            header_args = (
                ", ".join(f"e{i}" for i in range(len(rhs.args)))
                + (", " if rhs.args else "")
                + ", ".join(f"e{i + len(rhs.args)}" for i in range(len(keys)))
            )
            arg_names = (
                ", ".join(f"e{i}" for i in range(len(rhs.args)))
                + (", " if rhs.args else "")
                + ", ".join(f"{a}=e{i + len(rhs.args)}" for i, a in enumerate(keys))
            )
            func_text = f"def f(string, {header_args}):\n"
            func_text += f"    return format_func(string, {header_args})\n"

            format_func_text = f"def format_func(string, {header_args}):\n"
            format_func_text += (
                "    with bodo.ir.object_mode.no_warning_objmode(res='unicode_type'):\n"
            )
            format_func_text += f"        res = string.format({arg_names})\n"
            format_func_text += "    return res\n"

            loc_vars = {}
            exec(func_text, {}, loc_vars)
            f = loc_vars["f"]
            loc_vars = {}
            exec(format_func_text, globals(), loc_vars)
            format_func = numba.njit(loc_vars["format_func"])
            glbs = {
                "format_func": format_func,
            }
            args = [string_var] + rhs.args + [kws[key] for key in keys]
            return replace_func(
                self,
                f,
                args,
                extra_globals=glbs,
            )
        else:
            return [assign]

    def _run_call_logger(self, assign, lhs, rhs, logger_var, func_name):
        """
        Provide transformations for logging module functions that cannot be supported in regular overloads
        """
        func_names = (
            "debug",
            "warning",
            "warn",
            "info",
            "error",
            "exception",
            "critical",
            "log",
            "setLevel",
        )
        if func_name in func_names:
            kws = dict(rhs.kws)
            keys = list(kws.keys())
            header_args = (
                ", ".join(f"e{i}" for i in range(len(rhs.args)))
                + (", " if rhs.args else "")
                + ", ".join(f"e{i + len(rhs.args)}" for i in range(len(keys)))
            )
            arg_names = (
                ", ".join(f"e{i}" for i in range(len(rhs.args)))
                + (", " if rhs.args else "")
                + ", ".join(f"{a}=e{i + len(rhs.args)}" for i, a in enumerate(keys))
            )
            func_text = f"def f(logger, {header_args}):\n"
            func_text += f"    return format_func(logger, {header_args})\n"

            format_func_text = f"def format_func(logger, {header_args}):\n"
            format_func_text += "    with bodo.ir.object_mode.no_warning_objmode():\n"
            format_func_text += f"        logger.{func_name}({arg_names})\n"

            loc_vars = {}
            exec(func_text, {}, loc_vars)
            f = loc_vars["f"]
            loc_vars = {}
            exec(format_func_text, globals(), loc_vars)
            format_func = numba.njit(loc_vars["format_func"])
            glbs = {
                "format_func": format_func,
            }
            args = [logger_var] + rhs.args + [kws[key] for key in keys]
            return replace_func(
                self,
                f,
                args,
                extra_globals=glbs,
            )
        else:
            return [assign]

    def _run_call_series(self, assign, lhs, rhs, series_var, func_name):
        # NOTE: some operations are used in Bodo's kernels and overload inline="always"
        # may require inlining them eventually (since lowered impl doesn't exist)
        # see bodo/tests/test_series.py::test_series_astype_cat"[S0]"
        if func_name in (
            "count",
            "fillna",
            "sort_values",
            "dropna",
            "notna",
            "isna",
            "bfill",
            "ffill",
            "pad",
            "backfill",
            "mask",
            "where",
        ):
            rhs.args.insert(0, series_var)
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}
            if func_name in ("bfill", "ffill", "pad", "backfill"):
                overload_func = (
                    bodo.hiframes.series_impl.create_fillna_specific_method_overload(
                        func_name
                    )
                )
            elif func_name in ("mask", "where"):
                overload_func = (
                    bodo.hiframes.series_impl.create_series_mask_where_overload(
                        func_name
                    )
                )
            else:
                overload_func = getattr(
                    bodo.hiframes.series_impl, "overload_series_" + func_name
                )
            impl = overload_func(*arg_typs, **kw_typs)
            return replace_func(
                self,
                impl,
                rhs.args,
                pysig=numba.core.utils.pysignature(impl),
                kws=dict(rhs.kws),
            )

        def get_operator_func_name(f_name):
            if f_name == "div":
                return "truediv"
            return f_name

        # inline Series.add/pow/...
        if func_name in bodo.hiframes.series_impl.explicit_binop_funcs:
            rhs.args.insert(0, series_var)
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}
            op = getattr(operator, get_operator_func_name(func_name))
            overload_func = (
                bodo.hiframes.series_impl.create_explicit_binary_op_overload(op)
            )
            impl = overload_func(*arg_typs, **kw_typs)
            return replace_func(
                self,
                impl,
                rhs.args,
                pysig=numba.core.utils.pysignature(impl),
                kws=dict(rhs.kws),
            )

        # inline S.radd/rpow/...
        if (
            func_name[0] == "r"
            and func_name[1:] in bodo.hiframes.series_impl.explicit_binop_funcs
        ):
            rhs.args.insert(0, series_var)
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}
            op = getattr(operator, get_operator_func_name(func_name[1:]))
            overload_func = (
                bodo.hiframes.series_impl.create_explicit_binary_reverse_op_overload(op)
            )
            impl = overload_func(*arg_typs, **kw_typs)
            return replace_func(
                self,
                impl,
                rhs.args,
                pysig=numba.core.utils.pysignature(impl),
                kws=dict(rhs.kws),
            )

        if func_name == "combine":
            return self._handle_series_combine(assign, lhs, rhs, series_var)

        if func_name in ("map", "apply"):
            nodes = []
            kws = dict(rhs.kws)
            if func_name == "apply":
                func_var = get_call_expr_arg("apply", rhs.args, kws, 0, "func")
            else:
                func_var = get_call_expr_arg("map", rhs.args, kws, 0, "arg")
            extra_args = get_call_expr_arg(func_name, rhs.args, kws, 2, "args", [])
            if extra_args:
                extra_args = get_build_sequence_vars(
                    self.func_ir, self.typemap, self.calltypes, extra_args, nodes
                )
            na_action = None
            if func_name == "map":
                kws.pop("arg", None)
                na_action = get_call_expr_arg(
                    "map", rhs.args, kws, 1, "na_action", use_default=True
                )
                kws.pop("na_action", None)
                if na_action is not None:
                    na_action = self.typemap[na_action.name]
                    if is_overload_none(na_action):
                        na_action = None
                    else:
                        assert is_overload_constant_str(na_action), (
                            "Series.map(): na_action should be a constant string"
                        )
                        na_action = get_overload_const_str(na_action)
                        assert na_action == "ignore", (
                            "Series.map(): na_action should be None or 'ignore''"
                        )
            else:
                kws.pop("func", None)
                kws.pop("convert_dtype", None)
            kws.pop("args", None)
            return self._handle_series_map(
                assign,
                lhs,
                rhs,
                series_var,
                func_var,
                na_action,
                extra_args,
                kws,
                nodes,
            )

        # astype with string output
        if func_name == "astype":
            # just return input if both input/output are strings
            # TODO: removing this opt causes a crash in test_series_astype_str
            # TODO: copy if not packed string array
            if is_str_series_typ(self.typemap[lhs.name]) and is_str_series_typ(
                self.typemap[series_var.name]
            ):
                return replace_func(self, eval("lambda a: a"), [series_var])

            rhs.args.insert(0, series_var)
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}
            overload_func = getattr(
                bodo.hiframes.series_impl, "overload_series_" + func_name
            )
            impl = overload_func(*arg_typs, **kw_typs)
            stub = eval("lambda S, dtype, copy=True, errors='raise': None")
            return replace_func(
                self,
                impl,
                rhs.args,
                pysig=numba.core.utils.pysignature(stub),
                kws=dict(rhs.kws),
            )

        return [assign]

    def _handle_series_map(
        self, assign, lhs, rhs, series_var, func_var, na_action, extra_args, kws, nodes
    ):
        """translate df.A.map(lambda a:...) to prange()"""
        # get the function object (ir.Expr.make_function or actual python function)
        func_type = self.typemap[func_var.name]
        data = self._get_series_data(series_var, nodes)
        index = self._get_series_index(series_var, nodes)
        name = self._get_series_name(series_var, nodes)

        # dictionary input case
        if isinstance(func_type, types.DictType):
            func_text = (
                ""
                "def impl(A, index, name, d):\n"
                "    numba.parfors.parfor.init_prange()\n"
                "    n = len(A)\n"
                "    S0 = bodo.utils.utils.alloc_type(n, _arr_typ0, (-1,))\n"
                "    for i in numba.parfors.parfor.internal_prange(n):\n"
                "        if bodo.libs.array_kernels.isna(A, i):\n"
                "            bodo.libs.array_kernels.setna(S0, i)\n"
                "        v = bodo.utils.conversion.box_if_dt64(A[i])\n"
                "        if v in d:\n"
                "            S0[i] = bodo.utils.conversion.unbox_if_tz_naive_timestamp(d[v])\n"
                "        else:\n"
                "            bodo.libs.array_kernels.setna(S0, i)\n"
                "    return bodo.hiframes.pd_series_ext.init_series(S0, index, name)\n"
            )

            glbs = {
                "numba": numba,
                "bodo": bodo,
                "_arr_typ0": self.typemap[lhs.name].data,
            }

            loc_vars = {}
            exec(func_text, globals(), loc_vars)

            args = [data, index, name, func_var]
            return replace_func(
                self,
                loc_vars["impl"],
                args,
                extra_globals=glbs,
                pre_nodes=nodes,
            )

        # Generate a numpy ufunc
        run_ufunc = False

        # Handle builtin functions passed by strings.
        if is_overload_constant_str(func_type):
            func_name = get_overload_const_str(func_type)
            # Manually inline the implementation for efficiency
            impl = bodo.utils.transform.get_pandas_method_str_impl(
                self.typemap[series_var.name],
                func_name,
                self.typingctx,
                "Series.apply",
            )
            if impl is not None:
                return replace_func(
                    self,
                    impl,
                    [series_var],
                    pysig=numba.core.utils.pysignature(impl),
                    kws={},
                    # Some Series functions may require methods or
                    # attributes that need to be inlined by the full
                    # pipeline.
                    run_full_pipeline=True,
                )
            else:
                run_ufunc = True

        if bodo.utils.typing.is_numpy_ufunc(func_type):
            run_ufunc = True
            # Ufunc name can be found in the __name__ of
            # the typing key.
            func_name = func_type.typing_key.__name__

        if run_ufunc:
            return self._handle_ufuncs(func_name, (series_var,))

        func = get_overload_const_func(func_type, self.func_ir)

        is_df_output = isinstance(self.typemap[lhs.name], DataFrameType)
        out_arr_types = self.typemap[lhs.name].data
        out_arr_types = out_arr_types if is_df_output else [out_arr_types]
        n_out_cols = len(out_arr_types)
        udf_arg_names = (
            ", ".join(f"e{i}" for i in range(len(extra_args)))
            + (", " if extra_args else "")
            + ", ".join(f"{a}=e{i + len(extra_args)}" for i, a in enumerate(kws.keys()))
        )
        extra_args += list(kws.values())
        extra_arg_names = (", " if extra_args else "") + ", ".join(
            f"e{i}" for i in range(len(extra_args))
        )

        func_text = f"def f(A, index, name{extra_arg_names}):\n"
        func_text += "  numba.parfors.parfor.init_prange()\n"
        func_text += "  n = len(A)\n"
        for i in range(n_out_cols):
            func_text += (
                f"  S{i} = bodo.utils.utils.alloc_type(n, _arr_typ{i}, (-1,))\n"
            )
        func_text += "  for i in numba.parfors.parfor.internal_prange(n):\n"
        if na_action == "ignore":
            func_text += "    if bodo.libs.array_kernels.isna(A, i):\n"
            for i in range(n_out_cols):
                func_text += f"      bodo.libs.array_kernels.setna(S{i}, i)\n"
            func_text += "      continue\n"
        func_text += "    t2 = bodo.utils.conversion.box_if_dt64(A[i])\n"
        func_text += f"    v = map_func(t2, {udf_arg_names})\n"
        if is_df_output:
            func_text += "    v_vals = bodo.hiframes.pd_series_ext.get_series_data(v)\n"
            for i in range(n_out_cols):
                func_text += f"    v{i} = v_vals[{i}]\n"
        else:
            func_text += "    v0 = v\n"
        for i in range(n_out_cols):
            func_text += f"    S{i}[i] = bodo.utils.conversion.unbox_if_tz_naive_timestamp(v{i})\n"
        glbls = {}
        if is_df_output:
            data_arrs = ", ".join(f"S{i}" for i in range(n_out_cols))
            func_text += f"  return bodo.hiframes.pd_dataframe_ext.init_dataframe(({data_arrs},), index, __col_name_meta_value_series_map)\n"
            glbls.update(
                {
                    "__col_name_meta_value_series_map": ColNamesMetaType(
                        self.typemap[lhs.name].columns
                    )
                }
            )
        else:
            func_text += (
                "  return bodo.hiframes.pd_series_ext.init_series(S0, index, name)\n"
            )

        loc_vars = {}
        exec(func_text, glbls, loc_vars)
        f = loc_vars["f"]
        map_func = bodo.compiler.udf_jit(func)
        glbls.update(
            {
                "numba": numba,
                "bodo": bodo,
                "map_func": map_func,
                "init_nested_counts": bodo.utils.indexing.init_nested_counts,
                "add_nested_counts": bodo.utils.indexing.add_nested_counts,
            }
        )
        for i in range(n_out_cols):
            glbls[f"_arr_typ{i}"] = out_arr_types[i]
            glbls[f"data_arr_type{i}"] = out_arr_types[i].dtype

        args = [data, index, name] + extra_args
        return replace_func(
            self,
            f,
            args,
            extra_globals=glbls,
            pre_nodes=nodes,
        )

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
            return replace_func(
                self,
                impl,
                rhs.args,
                pysig=numba.core.utils.pysignature(impl),
                kws=dict(rhs.kws),
            )

        return [assign]

    def _run_call_rolling(self, assign, lhs, rhs, func_name):
        """inline implementation for rolling_corr/cov functions"""

        if func_name == "rolling_corr":
            func_text = (
                ""
                "def rolling_corr_impl(arr, other, win, minp, center):\n"
                "    cov = bodo.hiframes.rolling.rolling_cov(arr, other, win, minp, center)\n"
                "    a_std = bodo.hiframes.rolling.rolling_fixed(\n"
                "        arr, None, win, minp, center, 'std'\n"
                "    )\n"
                "    b_std = bodo.hiframes.rolling.rolling_fixed(\n"
                "        other, None, win, minp, center, 'std'\n"
                "    )\n"
                "    return cov / (a_std * b_std)\n"
            )

            loc_vars = {}
            exec(func_text, globals(), loc_vars)
            return replace_func(self, loc_vars["rolling_corr_impl"], rhs.args)

        if func_name == "rolling_cov":
            func_text = (
                ""
                "def rolling_cov_impl(arr, other, w, minp, center):\n"
                "    ddof = 1\n"
                "    X = arr.astype(np.float64)\n"
                "    Y = other.astype(np.float64)\n"
                "    XpY = X + Y\n"
                "    XtY = X * Y\n"
                "    count = bodo.hiframes.rolling.rolling_fixed(\n"
                "        XpY, None, w, minp, center, 'count'\n"
                "    )\n"
                "    mean_XtY = bodo.hiframes.rolling.rolling_fixed(\n"
                "        XtY, None, w, minp, center, 'mean'\n"
                "    )\n"
                "    mean_X = bodo.hiframes.rolling.rolling_fixed(\n"
                "        X, None, w, minp, center, 'mean'\n"
                "    )\n"
                "    mean_Y = bodo.hiframes.rolling.rolling_fixed(\n"
                "        Y, None, w, minp, center, 'mean'\n"
                "    )\n"
                "    bias_adj = count / (count - ddof)\n"
                "    return (mean_XtY - mean_X * mean_Y) * bias_adj\n"
            )

            loc_vars = {}
            exec(func_text, globals(), loc_vars)
            return replace_func(self, loc_vars["rolling_cov_impl"], rhs.args)

        if func_name == "alloc_shift":
            arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
            impl = bodo.hiframes.rolling.alloc_shift_overload(*arg_typs)
            return replace_func(self, impl, rhs.args)

        return [assign]

    def _handle_series_combine(self, assign, lhs, rhs, series_var):
        """translate s1.combine(s2, lambda x1,x2 :...) to prange()"""
        kws = dict(rhs.kws)
        other_var = get_call_expr_arg("combine", rhs.args, kws, 0, "other")
        func_var = get_call_expr_arg("combine", rhs.args, kws, 1, "func")
        fill_var = get_call_expr_arg(
            "combine", rhs.args, kws, 2, "fill_value", default=""
        )

        func = get_overload_const_func(self.typemap[func_var.name], self.func_ir)

        nodes = []
        data = self._get_series_data(series_var, nodes)
        index = self._get_series_index(series_var, nodes)
        name = self._get_series_name(series_var, nodes)
        other_data = self._get_series_data(other_var, nodes)

        # Use NaN if fill_value is not provided
        use_nan = fill_var == "" or self.typemap[fill_var.name] == types.none

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
        func_text += "  numba.parfors.parfor.init_prange()\n"
        func_text += "  S = np.empty(n, out_dtype)\n"
        func_text += "  for i in numba.parfors.parfor.internal_prange(n):\n"
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
        func_text += (
            "  return bodo.hiframes.pd_series_ext.init_series(S, index, None)\n"
        )

        loc_vars = {}
        exec(func_text, {}, loc_vars)
        f = loc_vars["f"]

        func_args = [data, other_data]
        if not use_nan:
            func_args.append(fill_var)
        func_args += [index, name]
        return replace_func(
            self,
            f,
            func_args,
            extra_globals={
                "numba": numba,
                "np": np,
                "pd": pd,
                "bodo": bodo,
                "out_dtype": self.typemap[lhs.name].dtype,
                "map_func": numba.njit(func),
            },
            pre_nodes=nodes,
        )

    def _run_pd_DatetimeIndex(self, assign, lhs, rhs):
        """transform pd.DatetimeIndex() call with string array argument"""
        arg_typs = tuple(self.typemap[v.name] for v in rhs.args)
        kw_typs = {name: self.typemap[v.name] for name, v in dict(rhs.kws).items()}
        impl = bodo.hiframes.pd_index_ext.pd_datetimeindex_overload(
            *arg_typs, **kw_typs
        )
        return replace_func(
            self, impl, rhs.args, pysig=self.calltypes[rhs].pysig, kws=dict(rhs.kws)
        )

    def _handle_empty_like(self, assign, lhs, rhs):
        # B = empty_like(A) -> B = empty(len(A), dtype)
        in_arr = rhs.args[0]

        if self.typemap[in_arr.name].ndim == 1:
            # generate simpler len() for 1D case
            func_text = (
                ""
                "def f(_in_arr):\n"
                "    _alloc_size = len(_in_arr)\n"
                "    _out_arr = np.empty(_alloc_size, _in_arr.dtype)\n"
            )

        else:
            func_text = (
                ""
                "def f(_in_arr):\n"
                "    _alloc_size = _in_arr.shape\n"
                "    _out_arr = np.empty(_alloc_size, _in_arr.dtype)\n"
            )

        loc_vars = {}
        exec(func_text, globals(), loc_vars)
        f_block = compile_to_numba_ir(
            loc_vars["f"],
            {"np": np},
            typingctx=self.typingctx,
            targetctx=self.targetctx,
            arg_typs=(if_series_to_array_type(self.typemap[in_arr.name]),),
            typemap=self.typemap,
            calltypes=self.calltypes,
        ).blocks.popitem()[1]
        replace_arg_nodes(f_block, [in_arr])
        nodes = f_block.body[:-3]  # remove none return
        nodes[-1].target = assign.target
        return nodes

    def _handle_np_full(self, assign, lhs, rhs):
        """parallelize np.full() since Numba doesn't support it"""
        kws = dict(rhs.kws)
        shape_var = get_call_expr_arg("full", rhs.args, kws, 0, "shape")
        fill_value_var = get_call_expr_arg("full", rhs.args, kws, 1, "fill_value")
        return_type = self.typemap[lhs]
        dtype = return_type.dtype
        nodes = []
        if return_type.ndim == 1:
            # convert (n,) to n for internal_prange()
            if isinstance(self.typemap[shape_var.name], types.BaseTuple):
                new_shape_var = ir.Var(shape_var.scope, mk_unique_var("shape"), rhs.loc)
                self.typemap[new_shape_var.name] = types.int64
                gen_getitem(new_shape_var, shape_var, 0, self.calltypes, nodes)
                shape_var = new_shape_var

            func_text = (
                ""
                "def full_impl(shape, fill_value):\n"
                "    numba.parfors.parfor.init_prange()\n"
                "    arr = np.empty(shape, dtype)\n"
                "    for i in numba.parfors.parfor.internal_prange(shape):\n"
                "        arr[i] = fill_value\n"
                "    return arr\n"
            )

        else:
            func_text = (
                ""
                "def full_impl(shape, fill_value):\n"
                "    numba.parfors.parfor.init_prange()\n"
                "    arr = np.empty(shape, dtype)\n"
                "    for i in numba.pndindex(shape):\n"
                "        arr[i] = fill_value\n"
                "    return arr\n"
            )

        loc_vars = {}
        exec(func_text, globals(), loc_vars)
        return replace_func(
            self,
            loc_vars["full_impl"],
            [shape_var, fill_value_var],
            pre_nodes=nodes,
            extra_globals={"dtype": dtype},
        )

    def _handle_h5_write(self, dset, index, arr):
        if index != slice(None):
            raise BodoError("Only HDF5 write of full array supported")
        assert isinstance(self.typemap[arr.name], types.Array)
        ndim = self.typemap[arr.name].ndim

        func_text = "def _h5_write_impl(dset_id, arr):\n"
        func_text += "  zero_tup = ({},)\n".format(", ".join(["0"] * ndim))
        # TODO: remove after support arr.shape in parallel
        func_text += "  arr_shape = ({},)\n".format(
            ", ".join([f"arr.shape[{i}]" for i in range(ndim)])
        )
        func_text += f"  err = bodo.io.h5_api.h5write(dset_id, np.int32({ndim}), zero_tup, arr_shape, 0, arr)\n"

        loc_vars = {}
        exec(func_text, {}, loc_vars)
        _h5_write_impl = loc_vars["_h5_write_impl"]
        return compile_func_single_block(_h5_write_impl, (dset, arr), None, self)

    def _is_unified_streaming_output(self, table_var):
        """Return True if table_var is output a streaming operator that unifies
        dictionaries. Currently only supports join.
        TODO[BSE-1197]: support other operators
        """
        table_def = get_definition(self.func_ir, table_var)

        # handle projection
        if is_call(table_def) and find_callname(
            self.func_ir, table_def, self.typemap
        ) == ("table_subset", "bodo.hiframes.table"):
            return self._is_unified_streaming_output(table_def.args[0])

        # join_probe_consume_batch's output is a tuple that is unpacked
        require(is_expr(table_def, "static_getitem"))
        table_def = get_definition(self.func_ir, table_def.value)
        require(is_expr(table_def, "exhaust_iter"))
        table_def = get_definition(self.func_ir, table_def.value)

        require(is_call(table_def))
        source_fname = find_callname(self.func_ir, table_def, self.typemap)
        return source_fname == (
            "join_probe_consume_batch",
            "bodo.libs.streaming.join",
        )

    def _simplify_IR(self):
        """Simplify IR after Series pass transforms."""
        changed = False
        self.func_ir.blocks = ir_utils.simplify_CFG(self.func_ir.blocks)
        if not self.avoid_copy_propagation:
            # Apply copy propagation. There will be many extra assignments if we
            # inline code.
            in_cps, _ = ir_utils.copy_propagate(self.func_ir.blocks, self.typemap)
            save_copies = ir_utils.apply_copy_propagate(
                self.func_ir.blocks,
                in_cps,
                ir_utils.get_name_var_table(self.func_ir.blocks),
                self.typemap,
                self.calltypes,
            )
            # Restore any user variable names.
            # Note: user variables may still be eliminated because of dead code elimination
            # but we attempt to capture them in metadata.
            var_rename_map = ir_utils.restore_copy_var_names(
                self.func_ir.blocks, save_copies, self.typemap
            )
            if self.parfor_metadata is not None:
                if "var_rename_map" not in self.parfor_metadata:
                    self.parfor_metadata["var_rename_map"] = {}
                self.parfor_metadata["var_rename_map"].update(var_rename_map)

        while ir_utils.remove_dead(
            self.func_ir.blocks, self.func_ir.arg_names, self.func_ir, self.typemap
        ):
            changed = True

        removed_branch = bodo.transforms.untyped_pass.remove_dead_branches(self.func_ir)
        changed = changed or removed_branch
        self.func_ir._definitions = build_definitions(self.func_ir.blocks)
        return changed

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

    def _get_index_data(self, dt_var, nodes):
        var_def = guard(get_definition, self.func_ir, dt_var)
        call_def = guard(find_callname, self.func_ir, var_def, self.typemap)
        if call_def in (
            ("init_datetime_index", "bodo.hiframes.pd_index_ext"),
            ("init_timedelta_index", "bodo.hiframes.pd_index_ext"),
            ("init_binary_str_index", "bodo.hiframes.pd_index_ext"),
            ("init_numeric_index", "bodo.hiframes.pd_index_ext"),
            ("init_categorical_index", "bodo.hiframes.pd_index_ext"),
            ("init_heter_index", "bodo.hiframes.pd_index_ext"),
        ):
            return var_def.args[0]

        nodes += compile_func_single_block(
            eval("lambda S: bodo.hiframes.pd_index_ext.get_index_data(S)"),
            (dt_var,),
            None,
            self,
        )
        return nodes[-1].target

    def _get_series_data(self, series_var, nodes):
        # optimization: return data var directly if series has a single
        # definition by init_series()
        # e.g. S = init_series(A, None)
        # XXX assuming init_series() is the only call to create a series
        # and series._data is never overwritten
        var_def = guard(get_definition, self.func_ir, series_var)
        call_def = guard(find_callname, self.func_ir, var_def, self.typemap)
        if call_def == ("init_series", "bodo.hiframes.pd_series_ext"):
            return var_def.args[0]

        # XXX use get_series_data() for getting data instead of S._data
        # to enable alias analysis
        nodes += compile_func_single_block(
            eval("lambda S: bodo.hiframes.pd_series_ext.get_series_data(S)"),
            (series_var,),
            None,
            self,
        )
        return nodes[-1].target

    def _get_series_index(self, series_var, nodes):
        # XXX assuming init_series is the only call to create a series
        # and series._index is never overwritten
        var_def = guard(get_definition, self.func_ir, series_var)
        call_def = guard(find_callname, self.func_ir, var_def, self.typemap)
        if call_def == ("init_series", "bodo.hiframes.pd_series_ext") and (
            len(var_def.args) >= 2 and not self._is_const_none(var_def.args[1])
        ):
            return var_def.args[1]

        # XXX use get_series_index() for getting data instead of S._index
        # to enable alias analysis
        nodes += compile_func_single_block(
            eval("lambda S: bodo.hiframes.pd_series_ext.get_series_index(S)"),
            (series_var,),
            None,
            self,
        )
        return nodes[-1].target

    def _get_series_name(self, series_var, nodes):
        var_def = guard(get_definition, self.func_ir, series_var)
        call_def = guard(find_callname, self.func_ir, var_def, self.typemap)
        if (
            call_def == ("init_series", "bodo.hiframes.pd_series_ext")
            and len(var_def.args) == 3
        ):
            return var_def.args[2]

        nodes += compile_func_single_block(
            eval("lambda S: bodo.hiframes.pd_series_ext.get_series_name(S)"),
            (series_var,),
            None,
            self,
        )
        return nodes[-1].target

    def _get_updated_dataframes(self, blocks, topo_order, updated_dfs=None):
        """find the potentially updated dataframes to avoid optimizing out
        get_dataframe_data() calls incorrectly.
        Looks for dataframe column set calls, dataframe args to JIT calls and UDFs.
        NOTE: This assumes that Bodo implementations of other APIs do not include
        setting dataframe columns inplace.
        """
        if updated_dfs is None:
            updated_dfs = set()
        for label in reversed(topo_order):
            block = blocks[label]
            for stmt in reversed(block.body):
                if (
                    is_assign(stmt)
                    and isinstance(stmt.value, ir.Var)
                    and stmt.target.name in updated_dfs
                ):
                    updated_dfs.add(stmt.value.name)
                if is_call_assign(stmt):
                    rhs = stmt.value
                    func_type = self.typemap[rhs.func.name]
                    fdef = guard(find_callname, self.func_ir, rhs, self.typemap)
                    if fdef in (
                        (
                            "set_df_column_with_reflect",
                            "bodo.hiframes.pd_dataframe_ext",
                        ),
                        ("set_dataframe_data", "bodo.hiframes.pd_dataframe_ext"),
                        ("set_df_col", "bodo.hiframes.dataframe_impl"),
                    ):
                        updated_dfs.add(rhs.args[0].name)
                    # If a user called a function with a bodo.jit and the input is a dataframe,
                    # the user may modify the bodo function in place
                    # TODO: Determine if we should internally switch to a whitelist of functions.
                    if bodo.compiler.is_user_dispatcher(func_type):
                        for arg in rhs.args + list(dict(rhs.kws).values()):
                            self._set_add_if_df(updated_dfs, arg.name)
                    # apply calls take both positional and kw args
                    if (
                        fdef
                        and fdef[0] == "apply"
                        and isinstance(fdef[1], ir.Var)
                        and isinstance(
                            self.typemap[fdef[1].name],
                            (DataFrameType, SeriesType, DataFrameGroupByType),
                        )
                    ):
                        for arg in rhs.args + list(dict(rhs.kws).values()):
                            self._set_add_if_df(updated_dfs, arg.name)
        # Iterate over statements again in Forward order to catches copies that
        # come after the update triggering df in the IR.
        # TODO: Add a test where this is necessary/remove this if it
        # isn't necessary.
        for label in topo_order:
            block = blocks[label]
            for stmt in block.body:
                if (
                    is_assign(stmt)
                    and isinstance(stmt.value, ir.Var)
                    and stmt.value.name in updated_dfs
                ):
                    updated_dfs.add(stmt.target.name)
        return updated_dfs

    def _set_add_if_df(self, updated_dfs, varname):
        """add 'varname' to 'updated_dfs' if it is a dataframe. Handles tuples
        recursively as well.
        TODO: support dataframe containers (list/dict) that may include updated dfs.
        """
        var_type = self.typemap[varname]
        if isinstance(var_type, DataFrameType):
            updated_dfs.add(varname)
        if isinstance(var_type, types.BaseTuple):
            tup_list = guard(find_build_tuple, self.func_ir, varname)
            if tup_list is not None:
                for v in tup_list:
                    if isinstance(v, ir.Var):  # pragma: no cover
                        self._set_add_if_df(updated_dfs, v.name)

    def _convert_series_calltype(self, call):
        sig = self.calltypes[call]
        if sig is None:
            return
        assert isinstance(sig, Signature)

        # XXX using replace() since it copies, otherwise cached overload
        # functions fail
        new_sig = sig.replace(return_type=if_series_to_array_type(sig.return_type))
        new_sig = new_sig.replace(args=tuple(map(if_series_to_array_type, sig.args)))

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

    def _is_const_none(self, var):
        var_def = guard(get_definition, self.func_ir, var)
        return isinstance(var_def, ir.Const) and var_def.value is None

    def _update_definitions(self, node_list):
        loc = ir.Loc("", 0)
        dumm_block = ir.Block(ir.Scope(None, loc), loc)
        dumm_block.body = node_list
        build_definitions({0: dumm_block}, self.func_ir._definitions)
        return


def _fix_typ_undefs(new_typ, old_typ):
    if isinstance(old_typ, (types.Array, SeriesType)):
        assert (
            isinstance(
                new_typ,
                (
                    types.Array,
                    IntegerArrayType,
                    FloatingArrayType,
                    SeriesType,
                    StringArrayType,
                    ArrayItemArrayType,
                    StructArrayType,
                    TupleArrayType,
                    bodo.hiframes.pd_categorical_ext.CategoricalArrayType,
                    types.List,
                    StringArraySplitViewType,
                ),
            )
            or new_typ == bodo.types.dict_str_arr_type
        )
        if new_typ.dtype == types.undefined:
            return new_typ.copy(old_typ.dtype)
    if isinstance(old_typ, types.BaseTuple):
        return types.Tuple(
            [_fix_typ_undefs(t, u) for t, u in zip(new_typ.types, old_typ.types)]
        )
    # TODO: fix List, Set
    return new_typ
