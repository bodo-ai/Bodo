# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Parallelizes the IR for distributed execution and inserts MPI calls.
"""
import copy
import math
import operator
import types as pytypes  # avoid confusion with numba.core.types
import warnings
from collections import defaultdict

import numba
import numpy as np

try:
    import sklearn
except:
    pass
import llvmlite.binding as ll
import numpy as np
from numba.core import ir, ir_utils, postproc, types
from numba.core.ir_utils import (
    GuardException,
    build_definitions,
    compile_to_numba_ir,
    compute_cfg_from_blocks,
    dprint_func_ir,
    find_build_sequence,
    find_callname,
    find_const,
    find_topo_order,
    get_call_table,
    get_definition,
    get_global_func_typ,
    get_name_var_table,
    get_tuple_table,
    guard,
    is_get_setitem,
    mk_alloc,
    mk_unique_var,
    remove_dead,
    remove_dels,
    rename_labels,
    replace_arg_nodes,
    replace_vars_inner,
    require,
    simplify,
)
from numba.parfors.parfor import (
    Parfor,
    _lower_parfor_sequential_block,
    get_parfor_params,
    get_parfor_reductions,
    unwrap_parfor_blocks,
    wrap_parfor_blocks,
)

import bodo
import bodo.utils.utils
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.io import csv_cpp
from bodo.io.h5_api import h5file_type, h5group_type
from bodo.libs.distributed_api import Reduce_Type
from bodo.libs.str_arr_ext import string_array_type
from bodo.libs.str_ext import (
    string_type,
    unicode_to_utf8,
    unicode_to_utf8_and_len,
)
from bodo.transforms.distributed_analysis import (
    DistributedAnalysis,
    Distribution,
    _get_array_accesses,
    get_reduce_op,
)
from bodo.utils.transform import (
    ReplaceFunc,
    compile_func_single_block,
    get_call_expr_arg,
    get_const_value_inner,
    replace_func,
)
from bodo.utils.typing import BodoError, BooleanLiteral, list_cumulative
from bodo.utils.utils import (
    debug_prints,
    find_build_tuple,
    gen_getitem,
    get_getsetitem_index_var,
    get_slice_step,
    is_alloc_callname,
    is_assign,
    is_call,
    is_call_assign,
    is_expr,
    is_np_array_typ,
    is_slice_equiv_arr,
    is_whole_slice,
)

ll.add_symbol("csv_output_is_dir", csv_cpp.csv_output_is_dir)


distributed_run_extensions = {}

# analysis data for debugging
dist_analysis = None
fir_text = None
saved_array_analysis = None

_csv_write = types.ExternalFunction(
    "csv_write",
    types.void(
        types.voidptr,
        types.voidptr,
        types.int64,
        types.int64,
        types.bool_,
        types.voidptr,
    ),
)

_csv_output_is_dir = types.ExternalFunction(
    "csv_output_is_dir",
    types.int8(types.voidptr),
)

_json_write = types.ExternalFunction(
    "json_write",
    types.void(
        types.voidptr,
        types.voidptr,
        types.int64,
        types.int64,
        types.bool_,
        types.bool_,
        types.voidptr,
    ),
)


class DistributedPass:
    """
    This pass analyzes the IR to decide parallelism of arrays and parfors for
    distributed transformation, then parallelizes the IR for distributed execution and
    inserts MPI calls.
    Specialized IR nodes are also transformed to regular IR here since all analysis and
    transformations are done.
    """

    def __init__(
        self, func_ir, typingctx, targetctx, typemap, calltypes, metadata, flags
    ):
        self.func_ir = func_ir
        self.typingctx = typingctx
        self.targetctx = targetctx
        self.typemap = typemap
        self.calltypes = calltypes
        # Loc object of current location being translated
        self.curr_loc = self.func_ir.loc
        self.metadata = metadata
        self.flags = flags
        self.arr_analysis = numba.parfors.array_analysis.ArrayAnalysis(
            self.typingctx, self.func_ir, self.typemap, self.calltypes
        )

        self._dist_analysis = None
        self._T_arrs = None  # set of transposed arrays (taken from analysis)
        # For each 1D parfor, map index variable name for the first dimension loop to
        # distributed start variable of the parfor
        self._1D_parfor_starts = {}
        # same as above but for 1D_Var parfors
        self._1D_Var_parfor_starts = {}
        # map 1D_Var arrays to index variable names for 1D_Var array accesses
        self._1D_Var_array_accesses = defaultdict(list)
        # keep start vars for 1D dist to reuse in parfor loop array accesses
        self._start_vars = {}

    def run(self):
        remove_dels(self.func_ir.blocks)
        dprint_func_ir(self.func_ir, "starting distributed pass")
        self.func_ir._definitions = build_definitions(self.func_ir.blocks)
        self.arr_analysis.run(self.func_ir.blocks)
        # saves array analysis to replace dead arrays in array.shape
        # (see test_csv_remove_col0_used_for_len and bodo_remove_dead_block)
        global saved_array_analysis
        try:
            saved_array_analysis = self.arr_analysis
            while ir_utils.remove_dead(
                self.func_ir.blocks, self.func_ir.arg_names, self.func_ir, self.typemap
            ):
                pass
        finally:
            saved_array_analysis = None
        self.func_ir._definitions = build_definitions(self.func_ir.blocks)
        self.arr_analysis.run(self.func_ir.blocks)

        dist_analysis_pass = DistributedAnalysis(
            self.func_ir,
            self.typemap,
            self.calltypes,
            self.typingctx,
            self.metadata,
            self.flags,
            self.arr_analysis,
        )
        self._dist_analysis = dist_analysis_pass.run()
        # dprint_func_ir(self.func_ir, "after analysis distributed")

        self._T_arrs = dist_analysis_pass._T_arrs
        self._parallel_accesses = dist_analysis_pass._parallel_accesses
        if debug_prints():  # pragma: no cover
            print("distributions: ", self._dist_analysis)

        self.func_ir._definitions = build_definitions(self.func_ir.blocks)

        # transform
        self._gen_init_code(self.func_ir.blocks)
        self.func_ir.blocks = self._run_dist_pass(self.func_ir.blocks)

        while remove_dead(
            self.func_ir.blocks, self.func_ir.arg_names, self.func_ir, self.typemap
        ):
            pass
        dprint_func_ir(self.func_ir, "after distributed pass")
        lower_parfor_sequential(
            self.typingctx, self.func_ir, self.typemap, self.calltypes
        )
        if bodo.multithread_mode:
            # parfor params need to be updated for multithread_mode since some
            # new variables like alloc_start are introduced by distributed pass
            # and are used in later parfors
            _parfor_ids = get_parfor_params(
                self.func_ir.blocks, True, defaultdict(list)
            )

        # save data for debug and test
        global dist_analysis, fir_text
        dist_analysis = self._dist_analysis
        import io

        str_io = io.StringIO()
        self.func_ir.dump(str_io)
        fir_text = str_io.getvalue()
        str_io.close()

    def _run_dist_pass(self, blocks, init_avail=None):
        # init liveness info
        cfg = compute_cfg_from_blocks(blocks)
        all_avail_vars = find_available_vars(blocks, cfg, init_avail)
        topo_order = find_topo_order(blocks)
        work_list = list((l, blocks[l]) for l in reversed(topo_order))
        while work_list:
            label, block = work_list.pop()
            # XXX can't change the block structure due to array analysis
            # XXX can't run transformation again on already converted code
            # since e.g. global sizes become invalid
            equiv_set = self.arr_analysis.get_equiv_set(label)
            avail_vars = all_avail_vars[label].copy()
            new_body = []
            for inst in block.body:
                self.curr_loc = inst.loc
                out_nodes = None
                if type(inst) in distributed_run_extensions:
                    f = distributed_run_extensions[type(inst)]
                    out_nodes = f(
                        inst,
                        self._dist_analysis.array_dists,
                        self.typemap,
                        self.calltypes,
                        self.typingctx,
                        self.targetctx,
                    )
                elif isinstance(inst, Parfor):
                    out_nodes = self._run_parfor(inst, equiv_set, avail_vars)
                    # run dist pass recursively
                    p_blocks = wrap_parfor_blocks(inst)
                    self._run_dist_pass(p_blocks, avail_vars)
                    unwrap_parfor_blocks(inst)
                elif isinstance(inst, ir.Assign):
                    rhs = inst.value
                    # concat reduction variables don't need transformation
                    # see test_concat_reduction
                    if inst.target.name in self._dist_analysis.concat_reduce_varnames:
                        out_nodes = [inst]
                    elif isinstance(rhs, ir.Expr):
                        out_nodes = self._run_expr(inst, equiv_set, avail_vars)
                elif isinstance(inst, (ir.StaticSetItem, ir.SetItem)):
                    out_nodes = []
                    index_var = get_getsetitem_index_var(inst, self.typemap, out_nodes)
                    out_nodes += self._run_getsetitem(
                        inst.target, index_var, inst, inst, equiv_set, avail_vars
                    )
                elif isinstance(inst, ir.Return):
                    out_nodes = self._gen_barrier() + [inst]
                # avoid replicated prints, print on all PEs only when there is dist arg
                elif isinstance(inst, ir.Print) and all(
                    self._is_REP(v.name) for v in inst.args
                ):
                    out_nodes = self._run_print(inst)

                if out_nodes is None:
                    out_nodes = [inst]

                assert isinstance(out_nodes, list), "invalid dist pass out nodes"
                self._update_avail_vars(avail_vars, out_nodes)
                new_body += out_nodes

            blocks[label].body = new_body

        return blocks

    def _run_expr(self, inst, equiv_set, avail_vars):
        rhs = inst.value

        if rhs.op == "call":
            return self._run_call(inst, equiv_set, avail_vars)

        if rhs.op in ("getitem", "static_getitem"):
            nodes = []
            index_var = get_getsetitem_index_var(rhs, self.typemap, nodes)
            return nodes + self._run_getsetitem(
                rhs.value, index_var, rhs, inst, equiv_set, avail_vars
            )

        # array.shape
        if (
            rhs.op == "getattr"
            and rhs.attr == "shape"
            and self._is_1D_or_1D_Var_arr(rhs.value.name)
        ):
            # concat reduction variables don't need transformation
            # see test_concat_reduction
            if rhs.value.name in self._dist_analysis.concat_reduce_varnames:
                return [inst]
            return self._run_array_shape(inst.target, rhs.value, equiv_set)

        # array.size
        if (
            rhs.op == "getattr"
            and rhs.attr == "size"
            and self._is_1D_or_1D_Var_arr(rhs.value.name)
        ):
            return self._run_array_size(inst.target, rhs.value, equiv_set)

        # RangeIndex._stop, get global value
        if (
            rhs.op == "getattr"
            and rhs.attr == "_stop"
            and isinstance(
                self.typemap[rhs.value.name], bodo.hiframes.pd_index_ext.RangeIndexType
            )
            and self._is_1D_or_1D_Var_arr(rhs.value.name)
        ):
            return [inst] + compile_func_single_block(
                lambda r: bodo.libs.distributed_api.dist_reduce(r._stop, _op),
                (rhs.value,),
                inst.target,
                self,
                extra_globals={"_op": np.int32(Reduce_Type.Max.value)},
            )

        # RangeIndex._start, get global value
        # XXX: assuming global start is 0
        # TODO: support all RangeIndex inputs
        if (
            rhs.op == "getattr"
            and rhs.attr == "_start"
            and isinstance(
                self.typemap[rhs.value.name], bodo.hiframes.pd_index_ext.RangeIndexType
            )
            and self._is_1D_or_1D_Var_arr(rhs.value.name)
        ):
            return [inst] + compile_func_single_block(
                lambda r: bodo.libs.distributed_api.dist_reduce(r._start, _op),
                (rhs.value,),
                inst.target,
                self,
                extra_globals={"_op": np.int32(Reduce_Type.Min.value)},
            )

        return [inst]

    def _run_call(self, assign, equiv_set, avail_vars):
        lhs = assign.target.name
        rhs = assign.value
        scope = assign.target.scope
        loc = assign.target.loc
        out = [assign]

        func_name = ""
        func_mod = ""
        fdef = guard(find_callname, self.func_ir, rhs, self.typemap)
        if fdef is None:
            # FIXME: since parfors are transformed and then processed
            # recursively, some funcs don't have definitions. The generated
            # arrays should be assigned REP and the var definitions added.
            # warnings.warn(
            #     "function call couldn't be found for distributed pass")
            return out
        else:
            func_name, func_mod = fdef

        if (
            func_name == "fit"
            and isinstance(func_mod, numba.core.ir.Var)
            and isinstance(
                self.typemap[func_mod.name],
                bodo.libs.sklearn_ext.BodoRandomForestClassifierType,
            )
        ):
            if self._is_1D_or_1D_Var_arr(rhs.args[0].name):
                self._set_last_arg_to_true(assign.value)
                return [assign]
        if (
            func_name == "score"
            and isinstance(func_mod, numba.core.ir.Var)
            and isinstance(
                self.typemap[func_mod.name],
                bodo.libs.sklearn_ext.BodoRandomForestClassifierType,
            )
        ):
            if self._is_1D_or_1D_Var_arr(rhs.args[0].name):
                self._set_last_arg_to_true(assign.value)
                return [assign]

        if (
            func_name == "fit"
            and isinstance(func_mod, numba.core.ir.Var)
            and isinstance(
                self.typemap[func_mod.name],
                bodo.libs.sklearn_ext.BodoSGDClassifierType,
            )
        ):
            if self._is_1D_or_1D_Var_arr(rhs.args[0].name):
                self._set_last_arg_to_true(assign.value)
                return [assign]
        if (
            func_name == "score"
            and isinstance(func_mod, numba.core.ir.Var)
            and isinstance(
                self.typemap[func_mod.name],
                bodo.libs.sklearn_ext.BodoSGDClassifierType,
            )
        ):
            if self._is_1D_or_1D_Var_arr(rhs.args[0].name):
                self._set_last_arg_to_true(assign.value)
                return [assign]

        if (
            func_name == "fit"
            and isinstance(func_mod, numba.core.ir.Var)
            and isinstance(
                self.typemap[func_mod.name],
                bodo.libs.sklearn_ext.BodoSGDRegressorType,
            )
        ):
            if self._is_1D_or_1D_Var_arr(rhs.args[0].name):
                self._set_last_arg_to_true(assign.value)
                return [assign]
        if (
            func_name == "score"
            and isinstance(func_mod, numba.core.ir.Var)
            and isinstance(
                self.typemap[func_mod.name],
                bodo.libs.sklearn_ext.BodoSGDRegressorType,
            )
        ):
            if self._is_1D_or_1D_Var_arr(rhs.args[0].name):
                self._set_last_arg_to_true(assign.value)
                return [assign]

        if (
            func_name == "fit"
            and isinstance(func_mod, numba.core.ir.Var)
            and isinstance(
                self.typemap[func_mod.name],
                bodo.libs.sklearn_ext.BodoKMeansClusteringType,
            )
        ):
            if self._is_1D_or_1D_Var_arr(rhs.args[0].name):
                self._set_last_arg_to_true(assign.value)
                return [assign]
        if (
            func_name == "score"
            and isinstance(func_mod, numba.core.ir.Var)
            and isinstance(
                self.typemap[func_mod.name],
                bodo.libs.sklearn_ext.BodoKMeansClusteringType,
            )
        ):
            if self._is_1D_or_1D_Var_arr(rhs.args[0].name):
                self._set_last_arg_to_true(assign.value)
                return [assign]

        if (
            func_mod == "sklearn.metrics._classification"
            and func_name == "precision_score"
        ):
            if self._is_1D_or_1D_Var_arr(rhs.args[0].name):
                rhs = assign.value
                kws = dict(rhs.kws)
                nodes = []

                y_true = get_call_expr_arg(
                    "sklearn.metrics.precision_score", rhs.args, kws, 0, "y_true"
                )
                y_pred = get_call_expr_arg(
                    "sklearn.metrics.precision_score", rhs.args, kws, 1, "y_pred"
                )

                # TODO other arguments
                average_var = ir.Var(
                    assign.target.scope,
                    mk_unique_var("precision_score_average"),
                    rhs.loc,
                )
                nodes.append(
                    ir.Assign(ir.Const("binary", rhs.loc), average_var, rhs.loc)
                )
                self.typemap[average_var.name] = types.StringLiteral("binary")
                # average cannot be specified positionally
                average = get_call_expr_arg(
                    "precision_score", rhs.args, kws, 1e6, "average", average_var
                )

                f = lambda y_true, y_pred, average: sklearn.metrics.precision_score(
                    y_true, y_pred, average=average, _is_data_distributed=True
                )
                return nodes + compile_func_single_block(
                    f,
                    [y_true, y_pred, average],
                    assign.target,
                    self,
                    extra_globals={"sklearn": sklearn},
                )

        if (
            func_mod == "sklearn.metrics._classification"
            and func_name == "recall_score"
        ):
            if self._is_1D_or_1D_Var_arr(rhs.args[0].name):
                rhs = assign.value
                kws = dict(rhs.kws)
                nodes = []

                y_true = get_call_expr_arg(
                    "sklearn.metrics.recall_score", rhs.args, kws, 0, "y_true"
                )
                y_pred = get_call_expr_arg(
                    "sklearn.metrics.recall_score", rhs.args, kws, 1, "y_pred"
                )

                # TODO other arguments
                average_var = ir.Var(
                    assign.target.scope, mk_unique_var("recall_score_average"), rhs.loc
                )
                nodes.append(
                    ir.Assign(ir.Const("binary", rhs.loc), average_var, rhs.loc)
                )
                self.typemap[average_var.name] = types.StringLiteral("binary")
                # average cannot be specified positionally
                average = get_call_expr_arg(
                    "recall_score", rhs.args, kws, 1e6, "average", average_var
                )

                f = lambda y_true, y_pred, average: sklearn.metrics.recall_score(
                    y_true, y_pred, average=average, _is_data_distributed=True
                )
                return nodes + compile_func_single_block(
                    f,
                    [y_true, y_pred, average],
                    assign.target,
                    self,
                    extra_globals={"sklearn": sklearn},
                )

        if func_mod == "sklearn.metrics._classification" and func_name == "f1_score":
            if self._is_1D_or_1D_Var_arr(rhs.args[0].name):
                rhs = assign.value
                kws = dict(rhs.kws)
                nodes = []

                y_true = get_call_expr_arg(
                    "sklearn.metrics.f1_score", rhs.args, kws, 0, "y_true"
                )
                y_pred = get_call_expr_arg(
                    "sklearn.metrics.f1_score", rhs.args, kws, 1, "y_pred"
                )

                # TODO other arguments
                average_var = ir.Var(
                    assign.target.scope, mk_unique_var("f1_score_average"), rhs.loc
                )
                nodes.append(
                    ir.Assign(ir.Const("binary", rhs.loc), average_var, rhs.loc)
                )
                self.typemap[average_var.name] = types.StringLiteral("binary")
                # average cannot be specified positionally
                average = get_call_expr_arg(
                    "f1_score", rhs.args, kws, 1e6, "average", average_var
                )

                f = lambda y_true, y_pred, average: sklearn.metrics.f1_score(
                    y_true, y_pred, average=average, _is_data_distributed=True
                )
                return nodes + compile_func_single_block(
                    f,
                    [y_true, y_pred, average],
                    assign.target,
                    self,
                    extra_globals={"sklearn": sklearn},
                )

        # divide 1D alloc
        # XXX allocs should be matched before going to _run_call_np
        if self._is_1D_arr(lhs) and is_alloc_callname(func_name, func_mod):
            # XXX for pre_alloc_string_array(n, nc), we assume nc is local
            # value (updated only in parfor like _str_replace_regex_impl)
            size_var = rhs.args[0]
            out, new_size_var = self._run_alloc(size_var, scope, loc)
            # empty_inferred is tuple for some reason
            rhs.args = list(rhs.args)
            rhs.args[0] = new_size_var
            out.append(assign)
            return out

        # fix 1D_Var allocs in case global len of another 1DVar is used
        if self._is_1D_Var_arr(lhs) and is_alloc_callname(func_name, func_mod):
            size_var = rhs.args[0]
            size_def = guard(get_definition, self.func_ir, size_var)
            # local 1D_Var arrays don't need transformation
            if is_expr(size_def, "call") and guard(
                find_callname, self.func_ir, size_def, self.typemap
            ) == ("local_alloc_size", "bodo.libs.distributed_api"):
                return out
            out, new_size_var = self._fix_1D_Var_alloc(
                size_var, scope, loc, equiv_set, avail_vars
            )
            # empty_inferred is tuple for some reason
            rhs.args = list(rhs.args)
            rhs.args[0] = new_size_var
            out.append(assign)
            return out

        if func_mod == "bodo.libs.array_kernels" and func_name in {"cummin", "cummax"}:
            if self._is_1D_or_1D_Var_arr(rhs.args[0].name):
                in_arr_var = rhs.args[0]
                lhs_var = assign.target
                # TODO: compute inplace if input array is dead
                def impl(A):  # pragma: no cover
                    B = np.empty_like(A)
                    _func(A, B)
                    return B

                func = getattr(bodo.libs.distributed_api, "dist_" + func_name)
                return compile_func_single_block(
                    impl, [in_arr_var], lhs_var, self, extra_globals={"_func": func}
                )

        # numpy direct functions
        if isinstance(func_mod, str) and func_mod == "numpy":
            return self._run_call_np(
                lhs, func_name, assign, rhs.args, dict(rhs.kws), equiv_set
            )

        # array.func calls
        if isinstance(func_mod, ir.Var) and is_np_array_typ(
            self.typemap[func_mod.name]
        ):
            return self._run_call_array(
                lhs, func_mod, func_name, assign, rhs.args, equiv_set, avail_vars
            )

        # df.func calls
        if isinstance(func_mod, ir.Var) and isinstance(
            self.typemap[func_mod.name], DataFrameType
        ):
            return self._run_call_df(lhs, func_mod, func_name, assign, rhs.args)

        if fdef == ("permutation", "numpy.random"):
            if self.typemap[rhs.args[0].name] == types.int64:
                return self._run_permutation_int(assign, rhs.args)

        # len(A) if A is 1D or 1D_Var
        if (
            fdef == ("len", "builtins")
            and rhs.args
            and self._is_1D_or_1D_Var_arr(rhs.args[0].name)
        ):
            arr = rhs.args[0]
            # concat reduction variables don't need transformation
            # see test_concat_reduction
            if arr.name in self._dist_analysis.concat_reduce_varnames:
                return [assign]
            nodes = []
            assign.value = self._get_dist_var_len(arr, nodes, equiv_set)
            nodes.append(assign)
            return nodes

        if fdef == ("File", "h5py"):
            # create and save a variable holding 1, in case we need to use it
            # to parallelize this call in _file_open_set_parallel()
            one_var = ir.Var(scope, mk_unique_var("$one"), loc)
            self.typemap[one_var.name] = types.IntegerLiteral(1)
            self._set1_var = one_var
            return [ir.Assign(ir.Const(1, loc), one_var, loc), assign]

        if (
            bodo.config._has_h5py
            and (
                func_mod == "bodo.io.h5_api"
                and func_name in ("h5read", "h5write", "h5read_filter")
            )
            and self._is_1D_arr(rhs.args[5].name)
        ):
            # TODO: make create_dataset/create_group collective
            arr = rhs.args[5]
            # dataset dimensions can be different than array due to integer selection
            ndims = len(self.typemap[rhs.args[2].name])
            nodes = []

            # divide 1st dimension
            size_var = self._get_dist_var_len(arr, nodes, equiv_set)
            start_var = self._get_1D_start(size_var, avail_vars, nodes)
            count_var = self._get_1D_count(size_var, nodes)

            # const value 1
            one_var = ir.Var(scope, mk_unique_var("$one"), loc)
            self.typemap[one_var.name] = types.IntegerLiteral(1)
            nodes.append(ir.Assign(ir.Const(1, loc), one_var, loc))

            # new starts
            starts_var = ir.Var(scope, mk_unique_var("$h5_starts"), loc)
            self.typemap[starts_var.name] = types.UniTuple(types.int64, ndims)
            prev_starts = self._get_tuple_varlist(rhs.args[2], nodes)
            start_tuple_call = ir.Expr.build_tuple([start_var] + prev_starts[1:], loc)
            starts_assign = ir.Assign(start_tuple_call, starts_var, loc)
            rhs.args[2] = starts_var

            # new counts
            counts_var = ir.Var(scope, mk_unique_var("$h5_counts"), loc)
            self.typemap[counts_var.name] = types.UniTuple(types.int64, ndims)
            prev_counts = self._get_tuple_varlist(rhs.args[3], nodes)
            count_tuple_call = ir.Expr.build_tuple([count_var] + prev_counts[1:], loc)
            counts_assign = ir.Assign(count_tuple_call, counts_var, loc)

            nodes += [starts_assign, counts_assign, assign]
            rhs.args[3] = counts_var
            rhs.args[4] = one_var

            # set parallel arg in file open
            file_varname = rhs.args[0].name
            self._file_open_set_parallel(file_varname)
            return nodes

        if (
            fdef
            == (
                "get_split_view_index",
                "bodo.hiframes.split_impl",
            )
            and self._dist_arr_needs_adjust(rhs.args[0].name, rhs.args[1].name)
        ):
            arr = rhs.args[0]
            index_var = self._fix_index_var(rhs.args[1])
            start_var, nodes = self._get_parallel_access_start_var(
                arr, equiv_set, index_var, avail_vars
            )
            sub_nodes = self._get_ind_sub(index_var, start_var)
            out = nodes + sub_nodes
            rhs.args[1] = sub_nodes[-1].target
            out.append(assign)
            return out

        if (
            fdef
            == (
                "setitem_str_arr_ptr",
                "bodo.libs.str_arr_ext",
            )
            and self._dist_arr_needs_adjust(rhs.args[0].name, rhs.args[1].name)
        ):
            arr = rhs.args[0]
            index_var = self._fix_index_var(rhs.args[1])
            start_var, nodes = self._get_parallel_access_start_var(
                arr, equiv_set, index_var, avail_vars
            )
            sub_nodes = self._get_ind_sub(index_var, start_var)
            out = nodes + sub_nodes
            rhs.args[1] = sub_nodes[-1].target
            out.append(assign)
            return out

        # adjust array index variable to be within current processor's data chunk
        if (
            fdef
            in (
                (
                    "inplace_eq",
                    "bodo.libs.str_arr_ext",
                ),
                ("str_arr_setitem_int_to_str", "bodo.libs.str_arr_ext"),
                ("str_arr_setitem_NA_str", "bodo.libs.str_arr_ext"),
                ("str_arr_set_not_na", "bodo.libs.str_arr_ext"),
            )
            and self._dist_arr_needs_adjust(rhs.args[0].name, rhs.args[1].name)
        ):
            arr = rhs.args[0]
            index_var = self._fix_index_var(rhs.args[1])
            start_var, nodes = self._get_parallel_access_start_var(
                arr, equiv_set, index_var, avail_vars
            )
            sub_nodes = self._get_ind_sub(index_var, start_var)
            out = nodes + sub_nodes
            rhs.args[1] = sub_nodes[-1].target
            out.append(assign)
            return out

        if (
            fdef
            == (
                "str_arr_item_to_numeric",
                "bodo.libs.str_arr_ext",
            )
            and self._dist_arr_needs_adjust(rhs.args[0].name, rhs.args[1].name)
        ):
            arr = rhs.args[0]
            index_var = self._fix_index_var(rhs.args[1])
            start_var, nodes = self._get_parallel_access_start_var(
                arr, equiv_set, index_var, avail_vars
            )
            sub_nodes = self._get_ind_sub(index_var, start_var)
            out = nodes + sub_nodes
            rhs.args[1] = sub_nodes[-1].target
            # input string array
            arr = rhs.args[2]
            index_var = self._fix_index_var(rhs.args[3])
            start_var, nodes = self._get_parallel_access_start_var(
                arr, equiv_set, index_var, avail_vars
            )
            sub_nodes = self._get_ind_sub(index_var, start_var)
            out += nodes + sub_nodes
            rhs.args[3] = sub_nodes[-1].target
            out.append(assign)
            return out

        if fdef == ("setna", "bodo.libs.array_kernels") and self._dist_arr_needs_adjust(
            rhs.args[0].name, rhs.args[1].name
        ):
            arr = rhs.args[0]
            index_var = self._fix_index_var(rhs.args[1])
            start_var, nodes = self._get_parallel_access_start_var(
                arr, equiv_set, index_var, avail_vars
            )
            sub_nodes = self._get_ind_sub(index_var, start_var)
            out = nodes + sub_nodes
            rhs.args[1] = sub_nodes[-1].target
            out.append(assign)
            return out

        if (
            fdef
            in (
                ("isna", "bodo.libs.array_kernels"),
                ("get_bit_bitmap_arr", "bodo.libs.int_arr_ext"),
                ("set_bit_to_arr", "bodo.libs.int_arr_ext"),
                ("get_str_arr_item_length", "bodo.libs.str_arr_ext"),
            )
            and self._dist_arr_needs_adjust(rhs.args[0].name, rhs.args[1].name)
        ):
            # fix index in call to isna
            arr = rhs.args[0]
            ind = self._fix_index_var(rhs.args[1])
            start_var, out = self._get_parallel_access_start_var(
                arr, equiv_set, ind, avail_vars
            )
            out += self._get_ind_sub(ind, start_var)
            rhs.args[1] = out[-1].target
            out.append(assign)

        if (
            fdef
            == (
                "rolling_fixed",
                "bodo.hiframes.rolling",
            )
            and self._is_1D_or_1D_Var_arr(rhs.args[0].name)
        ):
            # set parallel flag to true
            true_var = ir.Var(scope, mk_unique_var("true_var"), loc)
            self.typemap[true_var.name] = BooleanLiteral(True)
            rhs.args[3] = true_var
            # fix parallel arg type in calltype
            call_type = self.calltypes.pop(rhs)
            arg_typs = tuple(
                BooleanLiteral(True) if i == 3 else call_type.args[i]
                for i in range(len(call_type.args))
            )
            self.calltypes[rhs] = self.typemap[rhs.func.name].get_call_type(
                self.typingctx, arg_typs, {}
            )
            out = [ir.Assign(ir.Const(True, loc), true_var, loc), assign]

        if (
            fdef
            == (
                "rolling_variable",
                "bodo.hiframes.rolling",
            )
            and self._is_1D_or_1D_Var_arr(rhs.args[0].name)
        ):
            # set parallel flag to true
            true_var = ir.Var(scope, mk_unique_var("true_var"), loc)
            self.typemap[true_var.name] = BooleanLiteral(True)
            rhs.args[4] = true_var
            # fix parallel arg type in calltype
            call_type = self.calltypes.pop(rhs)
            arg_typs = tuple(
                BooleanLiteral(True) if i == 4 else call_type.args[i]
                for i in range(len(call_type.args))
            )
            self.calltypes[rhs] = self.typemap[rhs.func.name].get_call_type(
                self.typingctx, arg_typs, {}
            )
            out = [ir.Assign(ir.Const(True, loc), true_var, loc), assign]

        if (
            func_mod == "bodo.hiframes.rolling"
            and func_name in ("shift", "pct_change")
            and self._is_1D_or_1D_Var_arr(rhs.args[0].name)
        ):
            # set parallel flag to true
            true_var = ir.Var(scope, mk_unique_var("true_var"), loc)
            self.typemap[true_var.name] = types.boolean
            rhs.args[2] = true_var
            out = [ir.Assign(ir.Const(True, loc), true_var, loc), assign]

        if fdef == ("array_isin", "bodo.libs.array") and self._is_1D_or_1D_Var_arr(
            rhs.args[2].name
        ):
            # array_isin requires shuffling data only if values array is distributed
            f = lambda out_arr, in_arr, vals, p: bodo.libs.array.array_isin(
                out_arr, in_arr, vals, True
            )
            return compile_func_single_block(f, rhs.args, assign.target, self)

        if (
            fdef
            == (
                "quantile",
                "bodo.libs.array_kernels",
            )
            and self._is_1D_or_1D_Var_arr(rhs.args[0].name)
        ):
            arr = rhs.args[0]
            nodes = []
            size_var = self._get_dist_var_len(arr, nodes, equiv_set)
            rhs.args.append(size_var)

            f = lambda arr, q, size: bodo.libs.array_kernels.quantile_parallel(
                arr, q, size
            )
            return nodes + compile_func_single_block(f, rhs.args, assign.target, self)

        if fdef == ("nunique", "bodo.libs.array_kernels") and self._is_1D_or_1D_Var_arr(
            rhs.args[0].name
        ):
            f = lambda arr: bodo.libs.array_kernels.nunique_parallel(arr)
            return compile_func_single_block(f, rhs.args, assign.target, self)

        if fdef == ("unique", "bodo.libs.array_kernels") and self._is_1D_or_1D_Var_arr(
            rhs.args[0].name
        ):
            self._set_last_arg_to_true(assign.value)
            return [assign]

        if fdef == ("nonzero", "bodo.libs.array_kernels") and self._is_1D_or_1D_Var_arr(
            rhs.args[0].name
        ):
            self._set_last_arg_to_true(assign.value)
            return [assign]

        if (
            fdef
            == (
                "nlargest",
                "bodo.libs.array_kernels",
            )
            and self._is_1D_or_1D_Var_arr(rhs.args[0].name)
        ):
            f = lambda arr, I, k, i, f: bodo.libs.array_kernels.nlargest_parallel(
                arr, I, k, i, f
            )
            return compile_func_single_block(f, rhs.args, assign.target, self)

        if fdef == ("nancorr", "bodo.libs.array_kernels") and (
            self._is_1D_or_1D_Var_arr(rhs.args[0].name)
        ):
            self._set_last_arg_to_true(assign.value)
            return [assign]

        if fdef == ("series_monotonicity", "bodo.libs.array_kernels") and (
            self._is_1D_or_1D_Var_arr(rhs.args[0].name)
        ):
            self._set_last_arg_to_true(assign.value)
            return [assign]

        if fdef == ("autocorr", "bodo.libs.array_kernels") and (
            self._is_1D_or_1D_Var_arr(rhs.args[0].name)
        ):
            self._set_last_arg_to_true(assign.value)
            return [assign]

        if fdef == ("median", "bodo.libs.array_kernels") and (
            self._is_1D_or_1D_Var_arr(rhs.args[0].name)
        ):
            self._set_last_arg_to_true(assign.value)
            return [assign]

        if fdef == ("duplicated", "bodo.libs.array_kernels") and (
            self._is_1D_tup(rhs.args[0].name) or self._is_1D_Var_tup(rhs.args[0].name)
        ):
            self._set_last_arg_to_true(assign.value)
            return [assign]

        if fdef == ("drop_duplicates", "bodo.libs.array_kernels") and (
            self._is_1D_tup(rhs.args[0].name) or self._is_1D_Var_tup(rhs.args[0].name)
        ):
            self._set_last_arg_to_true(assign.value)
            return [assign]

        if func_name == "rebalance" and func_mod in {
            "bodo.libs.distributed_api",
            "bodo",
        }:
            if self._is_1D_or_1D_Var_arr(rhs.args[0].name):
                self._set_last_arg_to_true(assign.value)
                return [assign]
            else:
                warnings.warn("Invoking rebalance on a replicated array has no effect")

        if fdef == ("sample_table_operation", "bodo.libs.array_kernels") and (
            self._is_1D_tup(rhs.args[0].name) or self._is_1D_Var_tup(rhs.args[0].name)
        ):
            self._set_last_arg_to_true(assign.value)
            return [assign]

        if fdef == ("convert_rec_to_tup", "bodo.utils.typing"):
            # optimize Series back to back map pattern with tuples
            # TODO: create another optimization pass?
            arg_def = guard(get_definition, self.func_ir, rhs.args[0])
            if is_call(arg_def) and guard(find_callname, self.func_ir, arg_def) == (
                "convert_tup_to_rec",
                "bodo.utils.typing",
            ):
                assign.value = arg_def.args[0]
            return out

        if (
            fdef
            == (
                "init_range_index",
                "bodo.hiframes.pd_index_ext",
            )
            and self._is_1D_or_1D_Var_arr(lhs)
        ):
            return self._run_call_init_range_index(
                lhs, assign, rhs.args, avail_vars, equiv_set
            )

        # no need to gather if input data is replicated
        if (
            fdef == ("gatherv", "bodo") or fdef == ("allgatherv", "bodo")
        ) and self._is_REP(rhs.args[0].name):
            assign.value = rhs.args[0]
            return [assign]

        if fdef == ("dist_return", "bodo.libs.distributed_api"):
            assign.value = rhs.args[0]
            return [assign]

        if fdef == ("threaded_return", "bodo.libs.distributed_api"):
            assign.value = rhs.args[0]
            return [assign]

        if fdef == ("file_read", "bodo.io.np_io") and self._is_1D_or_1D_Var_arr(
            rhs.args[1].name
        ):
            fname = rhs.args[0]
            arr = rhs.args[1]
            # File offset in readfile is needed for the parallel seek
            file_offset = rhs.args[3]

            nodes, start_var, count_var = self._get_dist_var_start_count(
                arr, equiv_set, avail_vars
            )

            def impl(fname, data_ptr, start, count, offset):  # pragma: no cover
                return bodo.io.np_io.file_read_parallel(
                    fname, data_ptr, start, count, offset
                )

            return nodes + compile_func_single_block(
                # Increment start_var by the file offset
                impl,
                [fname, arr, start_var, count_var, file_offset],
                assign.target,
                self,
            )

        # replace get_type_max_value(arr.dtype) since parfors
        # arr.dtype transformation produces invalid code for dt64
        if fdef == ("get_type_max_value", "numba.cpython.builtins"):
            if self.typemap[rhs.args[0].name] == types.DType(types.NPDatetime("ns")):
                # XXX: not using replace since init block of parfor can't be
                # processed. test_series_idxmin
                # return replace_func(self,
                #     lambda: bodo.hiframes.pd_timestamp_ext.integer_to_dt64(
                #         numba.cpython.builtins.get_type_max_value(
                #             numba.core.types.int64)), [])
                f_block = compile_to_numba_ir(
                    lambda: bodo.hiframes.pd_timestamp_ext.integer_to_dt64(
                        numba.cpython.builtins.get_type_max_value(
                            numba.core.types.uint64
                        )
                    ),
                    {"bodo": bodo, "numba": numba},
                    self.typingctx,
                    (),
                    self.typemap,
                    self.calltypes,
                ).blocks.popitem()[1]
                out = f_block.body[:-2]
                out[-1].target = assign.target

        if fdef == ("get_type_min_value", "numba.cpython.builtins"):
            if self.typemap[rhs.args[0].name] == types.DType(types.NPDatetime("ns")):
                f_block = compile_to_numba_ir(
                    lambda: bodo.hiframes.pd_timestamp_ext.integer_to_dt64(
                        numba.cpython.builtins.get_type_min_value(
                            numba.core.types.uint64
                        )
                    ),
                    {"bodo": bodo, "numba": numba},
                    self.typingctx,
                    (),
                    self.typemap,
                    self.calltypes,
                ).blocks.popitem()[1]
                out = f_block.body[:-2]
                out[-1].target = assign.target

        return out

    def _run_call_init_range_index(self, lhs, assign, args, avail_vars, equiv_set):
        """transform init_range_index() calls"""
        assert len(args) == 4, "invalid init_range_index() call"
        # parallelize init_range_index() similar to parfors
        is_simple_range = (
            guard(
                get_const_value_inner,
                self.func_ir,
                args[0],
                typemap=self.typemap,
            )
            == 0
            and guard(
                get_const_value_inner,
                self.func_ir,
                args[2],
                typemap=self.typemap,
            )
            == 1
        )
        out = []

        # range size is equal to stop if simple range with start = 0 and step = 1
        if is_simple_range:
            size_var = args[1]
        else:
            f = lambda start, stop, step: max(0, -(-(stop - start) // step))
            out += compile_func_single_block(f, args[:-1], None, self)
            size_var = out[-1].target

        if self._is_1D_arr(lhs):
            start_var = self._get_1D_start(size_var, avail_vars, out)
            end_var = self._get_1D_end(size_var, out)
            self._update_avail_vars(avail_vars, out)

            if is_simple_range:
                args[0] = start_var
                args[1] = end_var

                def impl(start, stop, step, name):
                    res = bodo.hiframes.pd_index_ext.init_range_index(
                        start, stop, step, name
                    )
                    return res

            else:

                def impl(start, stop, step, name, chunk_start, chunk_end):
                    chunk_start = start + step * chunk_start
                    chunk_end = start + step * chunk_end
                    res = bodo.hiframes.pd_index_ext.init_range_index(
                        chunk_start, chunk_end, step, name
                    )
                    return res

                args = args + [start_var, end_var]

            return out + compile_func_single_block(impl, args, assign.target, self)
        else:
            # 1D_Var case
            assert self._is_1D_Var_arr(lhs)
            assert is_simple_range, "only simple 1D_Var RangeIndex is supported"
            new_size_var = self._get_1D_Var_size(size_var, equiv_set, avail_vars, out)

            def impl(stop, name):  # pragma: no cover
                prefix = bodo.libs.distributed_api.dist_exscan(stop, _op)
                return bodo.hiframes.pd_index_ext.init_range_index(
                    prefix, prefix + stop, 1, name
                )

            return out + compile_func_single_block(
                impl,
                [new_size_var, args[3]],
                assign.target,
                self,
                extra_globals={"_op": np.int32(Reduce_Type.Sum.value)},
            )

    def _run_call_np(self, lhs, func_name, assign, args, kws, equiv_set):
        """transform np.func() calls"""
        # allocs are handled separately
        assert not (
            self._is_1D_or_1D_Var_arr(lhs)
            and func_name in bodo.utils.utils.np_alloc_callnames
        ), (
            "allocation calls handled separately "
            "'empty', 'zeros', 'ones', 'full' etc."
        )
        out = [assign]
        scope = assign.target.scope
        loc = assign.loc

        if func_name == "reshape" and self._is_1D_or_1D_Var_arr(args[0].name):
            # shape argument can be int or tuple of ints
            shape_typ = self.typemap[args[1].name]
            if isinstance(types.unliteral(shape_typ), types.Integer):
                shape_vars = [args[1]]
            else:
                isinstance(shape_typ, types.BaseTuple)
                shape_vars = find_build_tuple(self.func_ir, args[1])
            return self._run_np_reshape(assign, args[0], shape_vars, equiv_set)

        if func_name == "ravel" and self._is_1D_arr(args[0].name):
            assert self.typemap[args[0].name].ndim == 1, "only 1D ravel supported"

        if func_name in list_cumulative and self._is_1D_or_1D_Var_arr(args[0].name):
            in_arr_var = args[0]
            lhs_var = assign.target
            # TODO: compute inplace if input array is dead
            def impl(A):  # pragma: no cover
                B = np.empty_like(A)
                _func(A, B)
                return B

            func = getattr(bodo.libs.distributed_api, "dist_" + func_name)
            return compile_func_single_block(
                impl, [in_arr_var], lhs_var, self, extra_globals={"_func": func}
            )

        # sum over the first axis is distributed, A.sum(0)
        if func_name == "sum" and self._is_1D_or_1D_Var_arr(args[0].name):
            axis = get_call_expr_arg("sum", args, kws, 1, "axis", "")
            if guard(find_const, self.func_ir, axis) == 0:
                reduce_op = Reduce_Type.Sum
                reduce_var = assign.target
                return out + self._gen_reduce(reduce_var, reduce_op, scope, loc)

        if func_name == "dot":
            return self._run_call_np_dot(lhs, assign, args)

        return out

    def _run_call_array(self, lhs, arr, func_name, assign, args, equiv_set, avail_vars):
        """transform distributed ndarray.func calls"""
        out = [assign]

        if func_name == "reshape" and self._is_1D_or_1D_Var_arr(arr.name):
            shape_vars = args
            arg_typ = self.typemap[args[0].name]
            if isinstance(arg_typ, types.BaseTuple):
                shape_vars = find_build_tuple(self.func_ir, args[0])
            return self._run_np_reshape(assign, arr, shape_vars, equiv_set)

        # TODO: refactor
        # TODO: add unittest
        if func_name == "tofile":
            if self._is_1D_arr(arr.name):
                _fname = args[0]
                nodes, start_var, count_var = self._get_dist_var_start_count(
                    arr, equiv_set, avail_vars
                )

                def f(fname, arr, start, count):  # pragma: no cover
                    return bodo.io.np_io.file_write_parallel(fname, arr, start, count)

                return nodes + compile_func_single_block(
                    f, [_fname, arr, start_var, count_var], assign.target, self
                )

            if self._is_1D_Var_arr(arr.name):
                _fname = args[0]

                def f(fname, arr):  # pragma: no cover
                    count = len(arr)
                    start = bodo.libs.distributed_api.dist_exscan(count, _op)
                    return bodo.io.np_io.file_write_parallel(fname, arr, start, count)

                return compile_func_single_block(
                    f,
                    [_fname, arr],
                    assign.target,
                    self,
                    extra_globals={"_op": np.int32(Reduce_Type.Sum.value)},
                )

        return out

    def _run_call_df(self, lhs, df, func_name, assign, args):
        if func_name == "to_parquet" and (self._is_1D_or_1D_Var_arr(df.name)):
            self._set_last_arg_to_true(assign.value)
            return [assign]
        elif func_name == "to_sql" and (self._is_1D_or_1D_Var_arr(df.name)):
            # Calling in parallel case
            self._set_last_arg_to_true(assign.value)
            return [assign]
        elif func_name == "to_csv" and self._is_1D_or_1D_Var_arr(df.name):
            # avoid header for non-zero ranks
            # write to string then parallel file write
            # df.to_csv(fname) ->
            # header = header and is_root  # only first line has header
            # str_out = df.to_csv(None, header=header)
            # bodo.io.csv_cpp(fname, str_out)

            df_typ = self.typemap[df.name]
            rhs = assign.value
            kws = dict(rhs.kws)
            fname = args[0]
            # convert StringLiteral to Unicode to make ._data available
            self.typemap.pop(fname.name)
            self.typemap[fname.name] = string_type
            nodes = []

            true_var = ir.Var(assign.target.scope, mk_unique_var("true"), rhs.loc)
            self.typemap[true_var.name] = types.bool_
            nodes.append(ir.Assign(ir.Const(True, df.loc), true_var, df.loc))
            header_var = get_call_expr_arg(
                "to_csv", rhs.args, kws, 5, "header", true_var
            )
            nodes += self._gen_csv_header_node(header_var, fname)
            header_var = nodes[-1].target
            if len(rhs.args) > 5:
                rhs.args[5] = header_var
            else:
                kws["header"] = header_var
                rhs.kws = kws

            # fix to_csv() type to have None as 1st arg
            call_type = self.calltypes.pop(rhs)
            arg_typs = list((types.none,) + call_type.args[1:])
            arg_typs[5] = types.bool_
            arg_typs = tuple(arg_typs)
            # self.calltypes[rhs] = self.typemap[rhs.func.name].get_call_type(
            #      self.typingctx, arg_typs, {})
            self.calltypes[rhs] = numba.core.typing.Signature(
                string_type, arg_typs, df_typ, call_type.pysig
            )

            # None as 1st arg
            none_var = ir.Var(assign.target.scope, mk_unique_var("none"), rhs.loc)
            self.typemap[none_var.name] = types.none
            none_assign = ir.Assign(ir.Const(None, rhs.loc), none_var, rhs.loc)
            nodes.append(none_assign)
            rhs.args[0] = none_var

            # str_out = df.to_csv(None)
            str_out = ir.Var(assign.target.scope, mk_unique_var("write_csv"), rhs.loc)
            self.typemap[str_out.name] = string_type
            new_assign = ir.Assign(rhs, str_out, rhs.loc)
            nodes.append(new_assign)

            # print_node = ir.Print([str_out], None, rhs.loc)
            # self.calltypes[print_node] = signature(types.none, string_type)
            # nodes.append(print_node)

            # TODO: fix lazy IO load
            def f(fname, str_out):  # pragma: no cover
                utf8_str, utf8_len = unicode_to_utf8_and_len(str_out)
                start = bodo.libs.distributed_api.dist_exscan(utf8_len, _op)
                # Assuming that path_or_buf is a string
                bucket_region = bodo.io.fs_io.get_s3_bucket_region_njit(fname)
                # TODO: unicode file name
                _csv_write(
                    unicode_to_utf8(fname),
                    utf8_str,
                    start,
                    utf8_len,
                    True,
                    unicode_to_utf8(bucket_region),
                )
                # Check if there was an error in the C++ code. If so, raise it.
                bodo.utils.utils.check_and_propagate_cpp_exception()

            return nodes + compile_func_single_block(
                f,
                [fname, str_out],
                assign.target,
                self,
                extra_globals={
                    "unicode_to_utf8_and_len": unicode_to_utf8_and_len,
                    "unicode_to_utf8": unicode_to_utf8,
                    "_op": np.int32(Reduce_Type.Sum.value),
                    "_csv_write": _csv_write,
                    "bodo": bodo,
                },
            )

        elif func_name == "to_json" and self._is_1D_or_1D_Var_arr(df.name):
            # write to string then parallel file write
            # df.to_json(fname) ->
            # str_out = df.to_json(None, header=header)
            # bodo.io.json_cpp(fname, str_out)

            df_typ = self.typemap[df.name]
            rhs = assign.value
            fname = args[0]
            # convert StringLiteral to Unicode to make ._data available
            self.typemap.pop(fname.name)
            self.typemap[fname.name] = string_type
            nodes = []

            kws = dict(rhs.kws)

            is_records = False
            if "orient" in kws:
                orient_var = get_call_expr_arg(
                    "to_json", rhs.args, kws, 1, "orient", None
                )
                orient_val = self.typemap[orient_var.name]
                is_records = True if orient_val.literal_value == "records" else False

            is_lines = False
            if "lines" in kws:
                lines_var = get_call_expr_arg(
                    "to_json", rhs.args, kws, 7, "lines", None
                )
                is_lines = self.typemap[lines_var.name].literal_value

            is_records_lines = ir.Var(
                assign.target.scope, mk_unique_var("is_records_lines"), rhs.loc
            )
            self.typemap[is_records_lines.name] = types.bool_
            nodes.append(
                ir.Assign(
                    ir.Const(is_records and is_lines, df.loc), is_records_lines, df.loc
                )
            )

            # fix to_json() type to have None as 1st arg
            call_type = self.calltypes.pop(rhs)
            arg_typs = list((types.none,) + call_type.args[1:])
            arg_typs = tuple(arg_typs)
            # self.calltypes[rhs] = self.typemap[rhs.func.name].get_call_type(
            #      self.typingctx, arg_typs, {})
            self.calltypes[rhs] = numba.core.typing.Signature(
                string_type, arg_typs, df_typ, call_type.pysig
            )

            # None as 1st arg
            none_var = ir.Var(assign.target.scope, mk_unique_var("none"), rhs.loc)
            self.typemap[none_var.name] = types.none
            none_assign = ir.Assign(ir.Const(None, rhs.loc), none_var, rhs.loc)
            nodes.append(none_assign)
            rhs.args[0] = none_var

            # str_out = df.to_json(None)
            str_out = ir.Var(assign.target.scope, mk_unique_var("write_json"), rhs.loc)
            self.typemap[str_out.name] = string_type
            new_assign = ir.Assign(rhs, str_out, rhs.loc)
            nodes.append(new_assign)

            # print_node = ir.Print([str_out], None, rhs.loc)
            # self.calltypes[print_node] = signature(types.none, string_type)
            # nodes.append(print_node)

            # TODO: fix lazy IO load

            def f(fname, str_out, is_records_lines):  # pragma: no cover
                utf8_str, utf8_len = unicode_to_utf8_and_len(str_out)
                start = bodo.libs.distributed_api.dist_exscan(utf8_len, _op)
                # Assuming that path_or_buf is a string
                bucket_region = bodo.io.fs_io.get_s3_bucket_region_njit(fname)
                # TODO: unicode file name
                _json_write(
                    unicode_to_utf8(fname),
                    utf8_str,
                    start,
                    utf8_len,
                    True,
                    is_records_lines,
                    unicode_to_utf8(bucket_region),
                )
                # Check if there was an error in the C++ code. If so, raise it.
                bodo.utils.utils.check_and_propagate_cpp_exception()

            return nodes + compile_func_single_block(
                f,
                [fname, str_out, is_records_lines],
                assign.target,
                self,
                extra_globals={
                    "unicode_to_utf8_and_len": unicode_to_utf8_and_len,
                    "unicode_to_utf8": unicode_to_utf8,
                    "_op": np.int32(Reduce_Type.Sum.value),
                    "_json_write": _json_write,
                    "bodo": bodo,
                },
            )
        return [assign]

    def _gen_csv_header_node(self, cond_var, fname_var):
        """
        cond_var is the original header node.
        If the original header node was true, there are two cases:
            a) output is a directory: every rank needs to write the header,
               so file in the directory has header, and thus all ranks have
               the new header node to be true
            b) output is a single file: only rank 0 writes the header, and thus
               only rank 0 have the new header node to be true, others are
               false
        If the original header node was false, the new header node is always false.
        """

        def f(cond, fname):  # pragma: no cover
            return cond & (
                (bodo.libs.distributed_api.get_rank() == 0)
                | _csv_output_is_dir(fname._data)
            )

        f_block = compile_to_numba_ir(
            f,
            {
                "bodo": bodo,
                "_csv_output_is_dir": _csv_output_is_dir,
            },
            self.typingctx,
            (self.typemap[cond_var.name], self.typemap[fname_var.name]),
            self.typemap,
            self.calltypes,
        ).blocks.popitem()[1]
        replace_arg_nodes(f_block, [cond_var, fname_var])
        nodes = f_block.body[:-2]
        return nodes

    def _run_permutation_int(self, assign, args):
        lhs = assign.target
        n = args[0]

        def f(lhs, n):  # pragma: no cover
            bodo.libs.distributed_api.dist_permutation_int(lhs, n)

        f_block = compile_to_numba_ir(
            f,
            {"bodo": bodo},
            self.typingctx,
            (self.typemap[lhs.name], types.intp),
            self.typemap,
            self.calltypes,
        ).blocks.popitem()[1]
        replace_arg_nodes(f_block, [lhs, n])
        f_block.body = [assign] + f_block.body
        return f_block.body[:-3]

    # Returns an IR node that defines a variable holding the size of |dtype|.
    def dtype_size_assign_ir(self, dtype, scope, loc):
        context = numba.core.cpu.CPUContext(self.typingctx)
        dtype_size = context.get_abi_sizeof(context.get_data_type(dtype))
        dtype_size_var = ir.Var(scope, mk_unique_var("dtype_size"), loc)
        self.typemap[dtype_size_var.name] = types.intp
        return ir.Assign(ir.Const(dtype_size, loc), dtype_size_var, loc)

    # def _run_permutation_array_index(self, lhs, rhs, idx):
    #     scope, loc = lhs.scope, lhs.loc
    #     dtype = self.typemap[lhs.name].dtype
    #     out = mk_alloc(self.typemap, self.calltypes, lhs,
    #                    (self._array_counts[lhs.name][0],
    #                     *self._array_sizes[lhs.name][1:]), dtype, scope, loc)

    #     def f(lhs, lhs_len, dtype_size, rhs, idx, idx_len):
    #         bodo.libs.distributed_api.dist_permutation_array_index(
    #             lhs, lhs_len, dtype_size, rhs, idx, idx_len)

    #     f_block = compile_to_numba_ir(f, {'bodo': bodo},
    #                                   self.typingctx,
    #                                   (self.typemap[lhs.name],
    #                                    types.intp,
    #                                    types.intp,
    #                                    self.typemap[rhs.name],
    #                                    self.typemap[idx.name],
    #                                    types.intp),
    #                                   self.typemap,
    #                                   self.calltypes).blocks.popitem()[1]
    #     dtype_ir = self.dtype_size_assign_ir(dtype, scope, loc)
    #     out.append(dtype_ir)
    #     replace_arg_nodes(f_block, [lhs, self._array_sizes[lhs.name][0],
    #                                 dtype_ir.target, rhs, idx,
    #                                 self._array_sizes[idx.name][0]])
    #     f_block.body = out + f_block.body
    #     return f_block.body[:-3]

    def _run_np_reshape(self, assign, in_arr, shape_vars, equiv_set):
        """distribute array reshape operation by finding new data offsets on every
        processor and exchanging data using alltoallv.
        Data exchange is necessary since data distribution is based on first dimension
        so the actual data may not be available for fully local reshape. Example:
        A = np.arange(6).reshape(3, 2) on 2 processors
        rank | data    =>    rank | data
        0    | 0             0    | 0  1
        0    | 1             0    | 2  3
        0    | 2             1    | 4  5
        1    | 3
        1    | 4
        1    | 5
        """
        lhs = assign.target
        scope = lhs.scope
        loc = lhs.loc
        nodes = []

        # optimization: just reshape locally if output has only 1 dimension
        if len(shape_vars) == 1:
            assert self._is_1D_Var_arr(lhs.name)
            return compile_func_single_block(
                lambda A: A.reshape(A.size), [in_arr], lhs, self
            )

        # get local size for 1st dimension and allocate output array
        # shape_vars[0] is global size
        count_var = self._get_1D_count(shape_vars[0], nodes)
        dtype = self.typemap[in_arr.name].dtype
        nodes += mk_alloc(
            self.typemap,
            self.calltypes,
            lhs,
            (count_var,) + tuple(shape_vars[1:]),
            dtype,
            scope,
            loc,
        )

        # shuffle the data to fill output arrays on different ranks properly
        return nodes + compile_func_single_block(
            lambda lhs, in_arr, new_dim0_global_len: bodo.libs.distributed_api.dist_oneD_reshape_shuffle(
                lhs, in_arr, new_dim0_global_len
            ),
            [lhs, in_arr, shape_vars[0]],
            None,
            self,
        )

    def _run_call_np_dot(self, lhs, assign, args):
        out = [assign]
        arg0 = args[0].name
        arg1 = args[1].name
        ndim0 = self.typemap[arg0].ndim
        ndim1 = self.typemap[arg1].ndim
        t0 = arg0 in self._T_arrs
        t1 = arg1 in self._T_arrs

        # reduction across dataset
        if self._is_1D_or_1D_Var_arr(arg0) and self._is_1D_or_1D_Var_arr(arg1):
            dprint("run dot dist reduce:", arg0, arg1)
            reduce_op = Reduce_Type.Sum
            reduce_var = assign.target
            out += self._gen_reduce(
                reduce_var, reduce_op, reduce_var.scope, reduce_var.loc
            )

        return out

    def _run_alloc(self, size_var, scope, loc):
        """divides array sizes and assign its sizes/starts/counts attributes
        returns generated nodes and the new size variable to enable update of
        the alloc call.
        """
        out = []
        new_size_var = None

        # size is single int var
        if isinstance(size_var, ir.Var) and isinstance(
            self.typemap[size_var.name], types.Integer
        ):
            # n_bytes = (n + 7) >> 3 is used in bitmap arrays like
            # IntegerArray's mask
            # use the total number of elements for 1D calculation
            # XXX: bitmasks can be only 1D arrays
            # TODO: is n_bytes calculation ever used in other parallel sizes
            # like parfors?
            size_def = guard(get_definition, self.func_ir, size_var)
            if (
                is_expr(size_def, "binop")
                and size_def.fn == operator.rshift
                and find_const(self.func_ir, size_def.rhs) == 3
            ):
                lhs_def = guard(get_definition, self.func_ir, size_def.lhs)
                if (
                    is_expr(lhs_def, "binop")
                    and lhs_def.fn == operator.add
                    and find_const(self.func_ir, lhs_def.rhs) == 7
                ):
                    num_elems = lhs_def.lhs
                    count_var = self._get_1D_count(num_elems, out)
                    out += compile_func_single_block(
                        lambda n: (n + 7) >> 3, (count_var,), None, self
                    )
                    new_size_var = out[-1].target
                    return out, new_size_var

            count_var = self._get_1D_count(size_var, out)
            new_size_var = count_var
            return out, new_size_var

        # tuple variable of ints
        if isinstance(size_var, ir.Var):
            # see if size_var is a 1D array's shape
            # it is already the local size, no need to transform
            var_def = guard(get_definition, self.func_ir, size_var)
            oned_varnames = set(
                v
                for v in self._dist_analysis.array_dists
                if self._dist_analysis.array_dists[v] == Distribution.OneD
            )
            if (
                isinstance(var_def, ir.Expr)
                and var_def.op == "getattr"
                and var_def.attr == "shape"
                and var_def.value.name in oned_varnames
            ):
                return out, size_var

            # size should be either int or tuple of ints
            # assert size_var.name in self._tuple_table
            # self._tuple_table[size_var.name]
            size_list = self._get_tuple_varlist(size_var, out)
            size_list = [
                ir_utils.convert_size_to_var(s, self.typemap, scope, loc, out)
                for s in size_list
            ]
        # tuple of int vars
        else:
            assert isinstance(size_var, (tuple, list))
            size_list = list(size_var)

        count_var = self._get_1D_count(size_list[0], out)
        ndims = len(size_list)
        new_size_list = copy.copy(size_list)
        new_size_list[0] = count_var
        tuple_var = ir.Var(scope, mk_unique_var("$tuple_var"), loc)
        self.typemap[tuple_var.name] = types.containers.UniTuple(types.intp, ndims)
        tuple_call = ir.Expr.build_tuple(new_size_list, loc)
        tuple_assign = ir.Assign(tuple_call, tuple_var, loc)
        out.append(tuple_assign)
        self.func_ir._definitions[tuple_var.name] = [tuple_call]
        new_size_var = tuple_var
        return out, new_size_var

    def _fix_1D_Var_alloc(self, size_var, scope, loc, equiv_set, avail_vars):
        """1D_Var allocs use global sizes of other 1D_var variables,
        so find the local size of one those variables for replacement.
        Assuming 1D_Var alloc is resulting from an operation with another
        1D_Var array and cannot be standalone.
        """
        out = []
        is_tuple = False

        # size is either integer or tuple
        if not isinstance(types.unliteral(self.typemap[size_var.name]), types.Integer):
            assert isinstance(self.typemap[size_var.name], types.BaseTuple)
            is_tuple = True

        # tuple variable of ints
        if is_tuple:
            # size should be either int or tuple of ints
            size_list = self._get_tuple_varlist(size_var, out)
            size_list = [
                ir_utils.convert_size_to_var(s, self.typemap, scope, loc, out)
                for s in size_list
            ]
            size_var = size_list[0]

        # find another 1D_Var array this alloc is associated with
        new_size_var = self._get_1D_Var_size(size_var, equiv_set, avail_vars, out)

        if not is_tuple:
            return out, new_size_var

        ndims = len(size_list)
        new_size_list = copy.copy(size_list)
        new_size_list[0] = new_size_var
        tuple_var = ir.Var(scope, mk_unique_var("$tuple_var"), loc)
        self.typemap[tuple_var.name] = types.containers.UniTuple(types.intp, ndims)
        tuple_call = ir.Expr.build_tuple(new_size_list, loc)
        tuple_assign = ir.Assign(tuple_call, tuple_var, loc)
        out.append(tuple_assign)
        self.func_ir._definitions[tuple_var.name] = [tuple_call]
        return out, tuple_var

    # new_body += self._run_1D_array_shape(
    #                                inst.target, rhs.value)
    # def _run_1D_array_shape(self, lhs, arr):
    #     """return shape tuple with global size of 1D/1D_Var arrays
    #     """
    #     nodes = []
    #     if self._is_1D_arr(arr.name):
    #         dim1_size = self._array_sizes[arr.name][0]
    #     else:
    #         assert self._is_1D_Var_arr(arr.name)
    #         nodes += self._gen_1D_Var_len(arr)
    #         dim1_size = nodes[-1].target
    #
    #     ndim = self._get_arr_ndim(arr.name)
    #
    #     func_text = "def f(arr, dim1):\n"
    #     func_text += "    s = (dim1, {})\n".format(
    #         ",".join(["arr.shape[{}]".format(i) for i in range(1, ndim)]))
    #     loc_vars = {}
    #     exec(func_text, {}, loc_vars)
    #     f = loc_vars['f']
    #
    #     f_ir = compile_to_numba_ir(f, {'np': np}, self.typingctx,
    #                                (self.typemap[arr.name], types.intp),
    #                                self.typemap, self.calltypes)
    #     f_block = f_ir.blocks.popitem()[1]
    #     replace_arg_nodes(f_block, [arr, dim1_size])
    #     nodes += f_block.body[:-3]
    #     nodes[-1].target = lhs
    #     return nodes

    def _get_1D_Var_size(self, size_var, equiv_set, avail_vars, out):
        """distributed transform needs to find local sizes for some operations on 1D_Var
        arrays and parfors since sizes are transformed to global sizes previously.
        For example, consider this program (after transformation of 'n', right before
        transformation of parfor):
            C = A[B]
            n = len(C)
            n = allreduce(n)  # transformed by distributed pass
            for i in prange(n):  # parfor needs local size of C, not 'n'
                ...
        The number of iterations in the prange on each processor should be the same as
        the local length of C which is not the same across processors. Therefore,
        _get_1D_Var_size finds that 'n' is the size of 'C', and replaces 'n' with
        'len(C)'.
        """
        size_def = guard(get_definition, self.func_ir, size_var)
        # find trivial calc_nitems(0, n, 1) call and use n instead
        if (
            guard(find_callname, self.func_ir, size_def)
            == ("calc_nitems", "bodo.libs.array_kernels")
            and guard(find_const, self.func_ir, size_def.args[0]) == 0
            and guard(find_const, self.func_ir, size_def.args[2]) == 1
        ):  # pragma: no cover
            # TODO: unittest for this case
            size_var = size_def.args[1]
            size_def = guard(get_definition, self.func_ir, size_var)

        # corner case: empty dataframe/series could be both input/output of concat()
        # see test_append_empty_df
        if isinstance(size_def, ir.Const) and size_def.value == 0:
            return size_var

        new_size_var = None
        for v in equiv_set.get_equiv_set(size_var):
            if "#" in v and self._is_1D_Var_arr(v.split("#")[0]):
                arr_name = v.split("#")[0]
                if arr_name not in avail_vars:
                    continue
                arr_var = ir.Var(size_var.scope, arr_name, size_var.loc)
                out += compile_func_single_block(
                    lambda A: len(A), (arr_var,), None, self
                )  # pragma: no cover
                new_size_var = out[-1].target
                break

        # branches can cause array analysis to remove size equivalences for some array
        # definitions since array analysis pass is not proper data flow yet.
        # This code tries pattern matching for definition of the size.
        # e.g. size = arr.shape[0]
        if new_size_var is None:
            arr_var = guard(_get_array_var_from_size, size_var, self.func_ir)
            if arr_var is not None:
                out += compile_func_single_block(
                    lambda A: len(A), (arr_var,), None, self
                )  # pragma: no cover
                new_size_var = out[-1].target

        if new_size_var is None:
            # Series.combine() uses max(s1, s2) to get output size
            calc_call = guard(find_callname, self.func_ir, size_def)
            if calc_call == ("max", "builtins"):
                s1 = self._get_1D_Var_size(size_def.args[0], equiv_set, avail_vars, out)
                s2 = self._get_1D_Var_size(size_def.args[1], equiv_set, avail_vars, out)
                out += compile_func_single_block(
                    lambda a1, a2: max(a1, a2), (s1, s2), None, self
                )
                new_size_var = out[-1].target

            # index_to_array() uses np.arange(I._start, I._stop, I._step)
            # on RangeIndex.
            # Get the local size of range
            if calc_call == ("calc_nitems", "bodo.libs.array_kernels"):
                start_def = guard(get_definition, self.func_ir, size_def.args[0])
                if (
                    is_expr(start_def, "getattr")
                    and start_def.attr == "_start"
                    and isinstance(
                        self.typemap[start_def.value.name],
                        bodo.hiframes.pd_index_ext.RangeIndexType,
                    )
                    and self._is_1D_Var_arr(start_def.value.name)
                ):
                    range_val = start_def.value
                    stop_def = guard(get_definition, self.func_ir, size_def.args[1])
                    step_def = guard(get_definition, self.func_ir, size_def.args[2])
                    if (
                        is_expr(stop_def, "getattr")
                        and stop_def.attr == "_stop"
                        and stop_def.value.name == range_val.name
                        and is_expr(step_def, "getattr")
                        and step_def.attr == "_step"
                        and step_def.value.name == range_val.name
                    ):
                        out += compile_func_single_block(
                            lambda I: bodo.libs.array_kernels.calc_nitems(
                                I._start, I._stop, I._step
                            ),
                            (range_val,),
                            None,
                            self,
                        )
                        new_size_var = out[-1].target

            # n_bytes = (n + 7) >> 3 pattern is used for calculating bitmap
            # size in int_arr_ext
            if (
                is_expr(size_def, "binop")
                and size_def.fn == operator.rshift
                and find_const(self.func_ir, size_def.rhs) == 3
            ):
                lhs_def = guard(get_definition, self.func_ir, size_def.lhs)
                if (
                    is_expr(lhs_def, "binop")
                    and lhs_def.fn == operator.add
                    and find_const(self.func_ir, lhs_def.rhs) == 7
                ):
                    size = self._get_1D_Var_size(
                        lhs_def.lhs, equiv_set, avail_vars, out
                    )
                    out += compile_func_single_block(
                        lambda n: (n + 7) >> 3, (size,), None, self
                    )
                    new_size_var = out[-1].target

        assert new_size_var, "1D Var size not found"
        return new_size_var

    def _run_array_shape(self, lhs, arr, equiv_set):
        ndims = self.typemap[arr.name].ndim

        if arr.name not in self._T_arrs:
            nodes = []
            size_var = self._get_dist_var_len(arr, nodes, equiv_set)
            # XXX: array.shape could be generated by array analysis to provide
            # size_var, so size_var may not be valid yet.
            # if size_var uses this shape variable, calculate global size
            size_def = guard(get_definition, self.func_ir, size_var)
            if (
                isinstance(size_def, ir.Expr)
                and size_def.op == "static_getitem"
                and size_def.value.name == lhs.name
            ):
                nodes += self._gen_1D_Var_len(arr)
                size_var = nodes[-1].target

            if ndims == 1:
                return nodes + compile_func_single_block(
                    lambda A, size_var: (size_var,), (arr, size_var), lhs, self
                )
            else:
                return nodes + compile_func_single_block(
                    lambda A, size_var: (size_var,) + A.shape[1:],
                    (arr, size_var),
                    lhs,
                    self,
                )

        # last dimension of transposed arrays is partitioned
        if arr.name in self._T_arrs:
            assert not self._is_1D_Var_arr(arr.name), "1D_Var arrays cannot transpose"
            nodes = []
            last_size_var = self._get_dist_var_dim_size(arr, (ndims - 1), nodes)
            return nodes + compile_func_single_block(
                lambda A, size_var: A.shape[:-1] + (size_var,),
                (arr, last_size_var),
                lhs,
                self,
            )

    def _run_array_size(self, lhs, arr, equiv_set):
        # get total size by multiplying all dimension sizes
        nodes = []
        if self._is_1D_arr(arr.name):
            dim1_size = self._get_dist_var_len(arr, nodes, equiv_set)
        else:
            assert self._is_1D_Var_arr(arr.name)
            nodes += self._gen_1D_Var_len(arr)
            dim1_size = nodes[-1].target

        def f(arr, dim1):  # pragma: no cover
            sizes = np.array(arr.shape)
            sizes[0] = dim1
            s = sizes.prod()

        f_ir = compile_to_numba_ir(
            f,
            {"np": np},
            self.typingctx,
            (self.typemap[arr.name], types.intp),
            self.typemap,
            self.calltypes,
        )
        f_block = f_ir.blocks.popitem()[1]
        replace_arg_nodes(f_block, [arr, dim1_size])
        nodes += f_block.body[:-3]
        nodes[-1].target = lhs
        return nodes

    def _run_getsetitem(self, arr, index_var, node, full_node, equiv_set, avail_vars):
        """Transform distributed getitem/setitem operations"""
        out = [full_node]

        # no need for transformation for getitem/setitem of distributed List/Dict
        if isinstance(self.typemap[arr.name], (types.List, types.DictType)):
            return out

        # adjust parallel access indices (in parfors)
        # 1D_Var arrays need adjustment if 1D_Var parfor has start adjusted
        if (
            self._is_1D_arr(arr.name)
            or (
                self._is_1D_Var_arr(arr.name)
                and arr.name in self._1D_Var_array_accesses
                and index_var.name in self._1D_Var_array_accesses[arr.name]
            )
        ) and (arr.name, index_var.name) in self._parallel_accesses:
            return self._run_parallel_access_getsetitem(
                arr, index_var, node, full_node, equiv_set, avail_vars
            )
        # parallel access in 1D_Var case, no need to transform
        elif (arr.name, index_var.name) in self._parallel_accesses:
            return out
        elif self._is_1D_or_1D_Var_arr(arr.name) and isinstance(
            node, (ir.StaticSetItem, ir.SetItem)
        ):
            return self._run_dist_setitem(
                node, arr, index_var, equiv_set, avail_vars, out
            )

        elif self._is_1D_or_1D_Var_arr(arr.name) and (
            is_expr(node, "getitem") or is_expr(node, "static_getitem")
        ):
            return self._run_dist_getitem(
                node, full_node, arr, index_var, equiv_set, avail_vars, out
            )

        return out

    def _run_parallel_access_getsetitem(
        self, arr, index_var, node, full_node, equiv_set, avail_vars
    ):
        """adjust index of getitem/setitem using parfor index on dist arrays"""
        start_var, nodes = self._get_parallel_access_start_var(
            arr, equiv_set, index_var, avail_vars
        )
        # multi-dimensional array could be indexed with 1D index
        if isinstance(self.typemap[index_var.name], types.Integer):
            # TODO: avoid repeated start/end generation
            sub_nodes = self._get_ind_sub(index_var, start_var)
            out = nodes + sub_nodes
            _set_getsetitem_index(node, sub_nodes[-1].target)
        else:
            index_list = guard(find_build_tuple, self.func_ir, index_var)
            assert index_list is not None
            # TODO: avoid repeated start/end generation
            sub_nodes = self._get_ind_sub(index_list[0], start_var)
            out = nodes + sub_nodes
            new_index_list = copy.copy(index_list)
            new_index_list[0] = sub_nodes[-1].target
            tuple_var = ir.Var(arr.scope, mk_unique_var("$tuple_var"), arr.loc)
            self.typemap[tuple_var.name] = self.typemap[index_var.name]
            tuple_call = ir.Expr.build_tuple(new_index_list, arr.loc)
            tuple_assign = ir.Assign(tuple_call, tuple_var, arr.loc)
            out.append(tuple_assign)
            _set_getsetitem_index(node, tuple_var)

        out.append(full_node)
        return out

    def _run_dist_getitem(
        self, node, full_node, arr, index_var, equiv_set, avail_vars, out
    ):
        """Transform distributed getitem"""
        full_index_var = index_var
        is_multi_dim = False
        lhs = full_node.target
        orig_index_var = index_var

        # we only consider 1st dimension for multi-dim arrays
        inds = guard(find_build_tuple, self.func_ir, index_var)
        if inds is not None:
            index_var = inds[0]
            is_multi_dim = True

        index_typ = self.typemap[index_var.name]

        # no need for transformation for whole slices
        # e.g. A = X[:,3]
        if guard(is_whole_slice, self.typemap, self.func_ir, index_var) or guard(
            is_slice_equiv_arr, arr, index_var, self.func_ir, equiv_set
        ):
            pass

        # strided whole slice
        # e.g. A = X[::2,5]
        elif guard(
            is_whole_slice,
            self.typemap,
            self.func_ir,
            index_var,
            accept_stride=True,
        ) or guard(
            is_slice_equiv_arr,
            arr,
            index_var,
            self.func_ir,
            equiv_set,
            accept_stride=True,
        ):
            # on each processor, the slice has to start from an offset:
            # |step-(start%step)|
            in_arr = full_node.value.value
            start_var, out = self._get_dist_start_var(in_arr, equiv_set, avail_vars)
            step = get_slice_step(self.typemap, self.func_ir, index_var)

            def f(A, start, step):  # pragma: no cover
                offset = abs(step - (start % step)) % step
                return A[offset::step]

            out += compile_func_single_block(f, [in_arr, start_var, step], None, self)
            out[-1].target = lhs

        # general slice access like A[3:7]
        elif isinstance(index_typ, types.SliceType):
            in_arr = full_node.value.value
            start_var, nodes = self._get_dist_start_var(in_arr, equiv_set, avail_vars)
            size_var = self._get_dist_var_len(in_arr, nodes, equiv_set)
            # for multi-dim case, perform selection in other dimensions then handle
            # the first dimension
            if is_multi_dim:
                # gen index with first dimension as full slice, other dimensions as
                # full getitem index
                nodes += compile_func_single_block(
                    lambda ind: (slice(None),) + ind[1:],
                    [full_index_var],
                    None,
                    self,
                )
                other_ind = nodes[-1].target
                return nodes + compile_func_single_block(
                    lambda arr, slice_index, start, tot_len, other_ind: bodo.libs.distributed_api.slice_getitem(
                        operator.getitem(arr, other_ind),
                        slice_index,
                        start,
                        tot_len,
                    ),
                    [in_arr, index_var, start_var, size_var, other_ind],
                    lhs,
                    self,
                    extra_globals={"operator": operator},
                )
            return nodes + compile_func_single_block(
                lambda arr, slice_index, start, tot_len: bodo.libs.distributed_api.slice_getitem(
                    arr,
                    slice_index,
                    start,
                    tot_len,
                ),
                [in_arr, index_var, start_var, size_var],
                lhs,
                self,
            )
        # int index like A[11]
        elif (
            isinstance(index_typ, types.Integer)
            and (arr.name, orig_index_var.name) not in self._parallel_accesses
        ):
            # TODO: handle multi-dim cases like A[0,:]
            in_arr = full_node.value.value
            start_var, nodes = self._get_dist_start_var(in_arr, equiv_set, avail_vars)
            size_var = self._get_dist_var_len(in_arr, nodes, equiv_set)
            is_1D = self._is_1D_arr(arr.name)
            return nodes + compile_func_single_block(
                lambda arr, ind, start, tot_len: bodo.libs.distributed_api.int_getitem(
                    arr, ind, start, tot_len, _is_1D
                ),
                [in_arr, orig_index_var, start_var, size_var],
                lhs,
                self,
                extra_globals={"_is_1D": is_1D},
            )

        return out

    def _run_dist_setitem(self, node, arr, index_var, equiv_set, avail_vars, out):
        """Transform distributed setitem"""
        is_multi_dim = False
        # we only consider 1st dimension for multi-dim arrays
        inds = guard(find_build_tuple, self.func_ir, index_var)
        if inds is not None:
            index_var = inds[0]
            is_multi_dim = True

        index_typ = types.unliteral(self.typemap[index_var.name])

        # no need for transformation for whole slices
        if guard(is_whole_slice, self.typemap, self.func_ir, index_var) or guard(
            is_slice_equiv_arr, arr, index_var, self.func_ir, equiv_set
        ):
            return out

        elif isinstance(index_typ, types.SliceType):

            start_var, nodes = self._get_dist_start_var(arr, equiv_set, avail_vars)
            arr_len = self._get_dist_var_len(arr, nodes, equiv_set)

            # create a tuple varialbe for lower dimension indices
            other_inds_var = ir.Var(arr.scope, mk_unique_var("$other_inds"), arr.loc)
            items = [] if not is_multi_dim else inds
            other_inds_tuple = ir.Expr.build_tuple(items, arr.loc)
            nodes.append(ir.Assign(other_inds_tuple, other_inds_var, arr.loc))
            self.typemap[other_inds_var.name] = types.BaseTuple.from_types(
                [self.typemap[v.name] for v in items]
            )

            # convert setitem with global range to setitem with local range
            # that overlaps with the local array chunk
            def f(A, val, idx, other_inds, chunk_start, arr_len):  # pragma: no cover
                new_slice = bodo.libs.distributed_api.get_local_slice(
                    idx, chunk_start, arr_len
                )
                new_ind = (new_slice,) + other_inds
                # avoid tuple index for cases like Series that don't support it
                new_ind = bodo.utils.indexing.untuple_if_one_tuple(new_ind)
                A[new_ind] = val

            return nodes + compile_func_single_block(
                f,
                [arr, node.value, index_var, other_inds_var, start_var, arr_len],
                None,
                self,
            )

        elif isinstance(index_typ, types.Integer):
            start_var, nodes = self._get_dist_start_var(arr, equiv_set, avail_vars)

            def f(A, val, index, chunk_start):  # pragma: no cover
                bodo.libs.distributed_api._set_if_in_range(A, val, index, chunk_start)

            return nodes + compile_func_single_block(
                f, [arr, node.value, index_var, start_var], None, self
            )

        # no need to transform for other cases like setitem of scalar value with bool
        # index
        return out

    def _run_parfor(self, parfor, equiv_set, avail_vars):

        # Thread and 1D parfors turn to gufunc in multithread mode
        if (
            bodo.multithread_mode
            and self._dist_analysis.parfor_dists[parfor.id] != Distribution.REP
        ):
            parfor.no_sequential_lowering = True

        if self._dist_analysis.parfor_dists[parfor.id] == Distribution.OneD_Var:
            return self._run_parfor_1D_Var(parfor, equiv_set, avail_vars)

        if self._dist_analysis.parfor_dists[parfor.id] != Distribution.OneD:
            if debug_prints():  # pragma: no cover
                print("parfor " + str(parfor.id) + " not parallelized.")
            return [parfor]

        range_size = parfor.loop_nests[0].stop
        out = []
        start_var = self._get_1D_start(range_size, avail_vars, out)
        end_var = self._get_1D_end(range_size, out)
        # update available vars to make start_var available for 1D accesses
        self._update_avail_vars(avail_vars, out)
        # print_node = ir.Print([start_var, end_var, range_size], None, loc)
        # self.calltypes[print_node] = signature(types.none, types.int64, types.int64, types.intp)
        # out.append(print_node)
        index_var = parfor.loop_nests[0].index_variable
        self._1D_parfor_starts[index_var.name] = start_var

        parfor.loop_nests[0].start = start_var
        parfor.loop_nests[0].stop = end_var
        out.append(parfor)

        init_reduce_nodes, reduce_nodes = self._gen_parfor_reductions(parfor)
        parfor.init_block.body += init_reduce_nodes
        out += reduce_nodes
        return out

    def _run_parfor_1D_Var(self, parfor, equiv_set, avail_vars):
        # the variable for range of the parfor represents the global range,
        # replace it with the local range, using len() on an associated array
        # XXX: assuming 1D_Var parfors are only possible when there is at least
        # one associated 1D_Var array
        prepend = []
        # assuming first dimension is parallelized
        # TODO: support transposed arrays
        index_name = parfor.loop_nests[0].index_variable.name
        stop_var = parfor.loop_nests[0].stop
        new_stop_var = None
        size_found = False
        array_accesses = _get_array_accesses(
            parfor.loop_body, self.func_ir, self.typemap
        )
        for (arr, index, is_bitwise) in array_accesses:
            # XXX avail_vars is used since accessed array could be defined in
            # init_block
            # arrays that are access bitwise don't have the same size
            # e.g. IntegerArray mask
            if (
                not is_bitwise
                and self._is_1D_Var_arr(arr)
                and self._index_has_par_index(index, index_name)
                and arr in avail_vars
            ):
                arr_var = ir.Var(stop_var.scope, arr, stop_var.loc)
                prepend += compile_func_single_block(
                    lambda A: len(A), (arr_var,), None, self
                )
                size_found = True
                new_stop_var = prepend[-1].target
                break

        # try equivalences
        if not size_found:
            new_stop_var = self._get_1D_Var_size(
                stop_var, equiv_set, avail_vars, prepend
            )

        # TODO: test multi-dim array sizes and complex indexing like slice
        parfor.loop_nests[0].stop = new_stop_var

        for (arr, index, _) in array_accesses:
            assert (
                arr not in self._T_arrs
            ), "1D_Var parfor for transposed parallel array not supported"

        # see if parfor index is used in compute other than array access
        # (e.g. argmin)
        l_nest = parfor.loop_nests[0]
        ind_varname = l_nest.index_variable.name
        ind_varnames = set((ind_varname,))
        ind_used = False

        for block in parfor.loop_body.values():
            for stmt in block.body:
                # assignment of parfor tuple index for multi-dim cases
                if is_assign(stmt) and stmt.target.name == parfor.index_var.name:
                    continue
                # parfor index is assigned to other variables here due to
                # copy propagation limitations, e.g. test_series_str_isna1
                if (
                    is_assign(stmt)
                    and isinstance(stmt.value, ir.Var)
                    and stmt.value.name in ind_varnames
                ):
                    ind_varnames.add(stmt.target.name)
                    continue
                if not self._is_array_access_stmt(stmt) and ind_varnames & set(
                    v.name for v in stmt.list_vars()
                ):
                    ind_used = True
                    dprint(
                        "index of 1D_Var pafor {} used in {}".format(parfor.id, stmt)
                    )
                    break

        # fix parfor start and stop bounds using ex_scan on ranges
        if ind_used:
            scope = l_nest.index_variable.scope
            loc = l_nest.index_variable.loc
            if isinstance(l_nest.start, int):
                start_var = ir.Var(scope, mk_unique_var("loop_start"), loc)
                self.typemap[start_var.name] = types.intp
                prepend.append(ir.Assign(ir.Const(l_nest.start, loc), start_var, loc))
                l_nest.start = start_var

            def _fix_ind_bounds(start, stop):  # pragma: no cover
                prefix = bodo.libs.distributed_api.dist_exscan(stop - start, _op)
                # rank = bodo.libs.distributed_api.get_rank()
                # print(rank, prefix, start, stop)
                return start + prefix, stop + prefix

            f_block = compile_to_numba_ir(
                _fix_ind_bounds,
                {"bodo": bodo, "_op": np.int32(Reduce_Type.Sum.value)},
                self.typingctx,
                (types.intp, types.intp),
                self.typemap,
                self.calltypes,
            ).blocks.popitem()[1]
            replace_arg_nodes(f_block, [l_nest.start, l_nest.stop])
            nodes = f_block.body[:-2]
            ret_var = nodes[-1].target
            gen_getitem(l_nest.start, ret_var, 0, self.calltypes, nodes)
            gen_getitem(l_nest.stop, ret_var, 1, self.calltypes, nodes)
            prepend += nodes
            self._1D_Var_parfor_starts[ind_varname] = l_nest.start

            for (arr, index, _) in array_accesses:
                if self._index_has_par_index(index, ind_varname):
                    self._1D_Var_array_accesses[arr].append(index)

        init_reduce_nodes, reduce_nodes = self._gen_parfor_reductions(parfor)
        parfor.init_block.body += init_reduce_nodes
        out = prepend + [parfor] + reduce_nodes
        return out

    def _index_has_par_index(self, index, par_index):
        """check if parfor index is used in 1st dimension of access index"""
        ind_def = self.func_ir._definitions[index]
        if len(ind_def) == 1 and isinstance(ind_def[0], ir.Var):
            index = ind_def[0].name
        if index == par_index:
            return True
        # multi-dim case
        tup_list = guard(find_build_tuple, self.func_ir, index)
        return (
            tup_list is not None and len(tup_list) > 0 and tup_list[0].name == par_index
        )

    def _gen_parfor_reductions(self, parfor):
        scope = parfor.init_block.scope
        loc = parfor.init_block.loc
        pre = []
        out = []

        for reduce_varname, (_init_val, reduce_nodes, _op) in parfor.reddict.items():
            reduce_op = guard(
                get_reduce_op, reduce_varname, reduce_nodes, self.func_ir, self.typemap
            )
            reduce_var = reduce_nodes[-1].target
            # TODO: initialize reduction vars (arrays)
            pre += self._gen_init_reduce(reduce_var, reduce_op)
            out += self._gen_reduce(reduce_var, reduce_op, scope, loc)

        return pre, out

    # def _get_var_const_val(self, var):
    #     if isinstance(var, int):
    #         return var
    #     node = guard(get_definition, self.func_ir, var)
    #     if isinstance(node, ir.Const):
    #         return node.value
    #     if isinstance(node, ir.Expr):
    #         if node.op == 'unary' and node.fn == '-':
    #             return -self._get_var_const_val(node.value)
    #         if node.op == 'binop':
    #             lhs = self._get_var_const_val(node.lhs)
    #             rhs = self._get_var_const_val(node.rhs)
    #             if node.fn == '+':
    #                 return lhs + rhs
    #             if node.fn == '-':
    #                 return lhs - rhs
    #             if node.fn == '//':
    #                 return lhs // rhs
    #     return None

    def _run_print(self, print_node):
        args = print_node.args
        arg_names = ", ".join("v{}".format(i) for i in range(len(print_node.args)))
        print_args = arg_names

        # handle vararg like print(*a)
        if print_node.vararg is not None:
            arg_names += "{}vararg".format(", " if args else "")
            print_args += "{}*vararg".format(", " if args else "")
            args.append(print_node.vararg)

        func_text = "def impl({}):\n".format(arg_names)
        func_text += "  bodo.libs.distributed_api.single_print({})\n".format(print_args)
        loc_vars = {}
        exec(func_text, {"bodo": bodo}, loc_vars)
        impl = loc_vars["impl"]

        return compile_func_single_block(impl, args, None, self)

    def _get_dist_var_start_count(self, arr, equiv_set, avail_vars):
        """get distributed chunk start/count of current rank for 1D_Block arrays"""
        nodes = []
        if arr.name in self._1D_Var_array_accesses:
            # using the start variable of the first parfor on this array
            # TODO(ehsan): use avail_vars to make sure parfor start variable is valid?
            index_name = self._get_dim1_index_name(
                self._1D_Var_array_accesses[arr.name][0].name
            )
            start_var = self._1D_Var_parfor_starts[index_name]
            f_block = compile_to_numba_ir(
                lambda A: len(A),
                {},
                self.typingctx,
                (self.typemap[arr.name],),
                self.typemap,
                self.calltypes,
            ).blocks.popitem()[1]
            replace_arg_nodes(f_block, [arr])
            nodes = f_block.body[:-3]  # remove none return
            count_var = nodes[-1].target
            return nodes, start_var, count_var

        size_var = self._get_dist_var_len(arr, nodes, equiv_set)
        start_var = self._get_1D_start(size_var, avail_vars, nodes)
        count_var = self._get_1D_count(size_var, nodes)
        return nodes, start_var, count_var

    def _get_dist_start_var(self, arr, equiv_set, avail_vars):
        if arr.name in self._1D_Var_array_accesses:
            # using the start variable of the first parfor on this array
            # TODO(ehsan): use avail_vars to make sure parfor start variable is valid?
            index_name = self._get_dim1_index_name(
                self._1D_Var_array_accesses[arr.name][0].name
            )
            return self._1D_Var_parfor_starts[index_name], []

        if self._is_1D_arr(arr.name):
            nodes = []
            size_var = self._get_dist_var_len(arr, nodes, equiv_set)
            start_var = self._get_1D_start(size_var, avail_vars, nodes)
        else:
            assert self._is_1D_Var_arr(arr.name)
            nodes = compile_func_single_block(
                lambda arr: bodo.libs.distributed_api.dist_exscan(len(arr), _op),
                [arr],
                None,
                self,
                extra_globals={"_op": np.int32(Reduce_Type.Sum.value)},
            )
            start_var = nodes[-1].target
        return start_var, nodes

    def _get_dist_var_dim_size(self, var, dim, nodes):
        # XXX just call _gen_1D_var_len() for now
        # TODO: get value from array analysis
        def f(A, dim, op):  # pragma: no cover
            c = A.shape[dim]
            res = bodo.libs.distributed_api.dist_reduce(c, op)

        f_block = compile_to_numba_ir(
            f,
            {"bodo": bodo},
            self.typingctx,
            (self.typemap[var.name], types.int64, types.int32),
            self.typemap,
            self.calltypes,
        ).blocks.popitem()[1]
        replace_arg_nodes(
            f_block,
            [var, ir.Const(dim, var.loc), ir.Const(Reduce_Type.Sum.value, var.loc)],
        )
        nodes += f_block.body[:-3]  # remove none return
        return nodes[-1].target

    def _get_dist_var_len(self, var, nodes, equiv_set):
        """
        Get global length of distributed data structure (Array, Series,
        DataFrame) if available. Otherwise, generate reduction code to get
        global length.
        """
        shape = equiv_set.get_shape(var)
        if isinstance(shape, (list, tuple)) and len(shape) > 0:
            return shape[0]
        # XXX just call _gen_1D_var_len() for now
        nodes += self._gen_1D_Var_len(var)
        return nodes[-1].target

    def _dist_arr_needs_adjust(self, varname, index_name):
        return self._is_1D_arr(varname) or (
            self._is_1D_Var_arr(varname)
            and varname in self._1D_Var_array_accesses
            and index_name in self._1D_Var_array_accesses[varname]
        )

    def _get_parallel_access_start_var(self, arr, equiv_set, index_var, avail_vars):
        """Same as _get_dist_start_var() but avoids generating reduction for
        getting global size since this is an error inside a parfor loop.
        """

        # XXX we return parfors start assuming parfor and parallel accessed
        # array are equivalent in size and have equivalent distribution
        # TODO: is this always the case?
        index_name = self._get_dim1_index_name(index_var.name)

        if (
            arr.name in self._1D_Var_array_accesses
            and index_name in self._1D_Var_parfor_starts
        ):
            return self._1D_Var_parfor_starts[index_name], []

        if index_name in self._1D_parfor_starts:
            return self._1D_parfor_starts[index_name], []

        # use shape if parfor start not found (TODO shouldn't reach here?)
        shape = equiv_set.get_shape(arr)
        if isinstance(shape, (list, tuple)) and len(shape) > 0:
            size_var = shape[0]
            nodes = []
            start_var = self._get_1D_start(size_var, avail_vars, nodes)
            return start_var, nodes

        raise BodoError("invalid parallel access")

    def _gen_1D_Var_len(self, arr):
        def f(A, op):  # pragma: no cover
            c = len(A)
            res = bodo.libs.distributed_api.dist_reduce(c, op)

        f_block = compile_to_numba_ir(
            f,
            {"bodo": bodo},
            self.typingctx,
            (self.typemap[arr.name], types.int32),
            self.typemap,
            self.calltypes,
        ).blocks.popitem()[1]
        replace_arg_nodes(f_block, [arr, ir.Const(Reduce_Type.Sum.value, arr.loc)])
        nodes = f_block.body[:-3]  # remove none return
        return nodes

    def _get_1D_start(self, size_var, avail_vars, nodes):
        """get start index of size_var in 1D_Block distribution"""
        # reuse start var if available
        if (
            size_var.name in self._start_vars
            and self._start_vars[size_var.name].name in avail_vars
        ):
            return self._start_vars[size_var.name]
        nodes += compile_func_single_block(
            lambda n, rank, n_pes: rank * (n // n_pes) + min(rank, n % n_pes),
            (size_var, self.rank_var, self.n_pes_var),
            None,
            self,
        )
        start_var = nodes[-1].target
        # rename for readability
        start_var.name = mk_unique_var("start_var")
        self.typemap[start_var.name] = types.int64
        self._start_vars[size_var.name] = start_var
        return start_var

    def _get_1D_count(self, size_var, nodes):
        """get chunk size for size_var in 1D_Block distribution"""

        def impl(n, rank, n_pes):  # pragma: no cover
            res = n % n_pes
            # The formula we would like is if (rank < res): blk_size +=1 but this does not compile
            blk_size = n // n_pes + min(rank + 1, res) - min(rank, res)
            return blk_size

        nodes += compile_func_single_block(
            impl, (size_var, self.rank_var, self.n_pes_var), None, self
        )
        count_var = nodes[-1].target
        # rename for readability
        count_var.name = mk_unique_var("count_var")
        self.typemap[count_var.name] = types.int64
        return count_var

    def _get_1D_end(self, size_var, nodes):
        """get end index of size_var in 1D_Block distribution"""
        nodes += compile_func_single_block(
            lambda n, rank, n_pes: (rank + 1) * (n // n_pes) + min(rank + 1, n % n_pes),
            (size_var, self.rank_var, self.n_pes_var),
            None,
            self,
        )
        end_var = nodes[-1].target
        # rename for readability
        end_var.name = mk_unique_var("end_var")
        self.typemap[end_var.name] = types.int64
        return end_var

    def _get_ind_sub(self, ind_var, start_var):
        if isinstance(ind_var, slice) or isinstance(
            self.typemap[ind_var.name], types.SliceType
        ):
            return self._get_ind_sub_slice(ind_var, start_var)
        # gen sub
        f_ir = compile_to_numba_ir(
            lambda ind, start: ind - start,
            {},
            self.typingctx,
            (types.intp, types.intp),
            self.typemap,
            self.calltypes,
        )
        block = f_ir.blocks.popitem()[1]
        replace_arg_nodes(block, [ind_var, start_var])
        return block.body[:-2]

    def _get_ind_sub_slice(self, slice_var, offset_var):
        if isinstance(slice_var, slice):
            f_text = """def f(offset):
                return slice({} - offset, {} - offset)
            """.format(
                slice_var.start, slice_var.stop
            )
            loc = {}
            exec(f_text, {}, loc)
            f = loc["f"]
            args = [offset_var]
            arg_typs = (types.intp,)
        else:

            def f(old_slice, offset):  # pragma: no cover
                return slice(old_slice.start - offset, old_slice.stop - offset)

            args = [slice_var, offset_var]
            slice_type = self.typemap[slice_var.name]
            arg_typs = (slice_type, types.intp)
        _globals = self.func_ir.func_id.func.__globals__
        f_ir = compile_to_numba_ir(
            f, _globals, self.typingctx, arg_typs, self.typemap, self.calltypes
        )
        _, block = f_ir.blocks.popitem()
        replace_arg_nodes(block, args)
        return block.body[:-2]  # ignore return nodes

    def _file_open_set_parallel(self, file_varname):
        """Finds file open call (h5py.File) for file_varname and sets the parallel flag."""
        # TODO: find and handle corner cases
        var = file_varname
        while True:
            var_def = get_definition(self.func_ir, var)
            require(isinstance(var_def, ir.Expr))
            if var_def.op == "call":
                fdef = find_callname(self.func_ir, var_def)
                if (
                    fdef[0] in ("create_dataset", "create_group")
                    and isinstance(fdef[1], ir.Var)
                    and self.typemap[fdef[1].name] in (h5file_type, h5group_type)
                ):
                    self._file_open_set_parallel(fdef[1].name)
                    return
                else:
                    assert fdef == ("File", "h5py")
                    call_type = self.calltypes.pop(var_def)
                    arg_typs = tuple(call_type.args[:-1] + (types.IntegerLiteral(1),))
                    self.calltypes[var_def] = self.typemap[
                        var_def.func.name
                    ].get_call_type(self.typingctx, arg_typs, {})
                    kws = dict(var_def.kws)
                    kws["_is_parallel"] = self._set1_var
                    var_def.kws = kws
                    return
            # TODO: handle control flow
            require(var_def.op in ("getitem", "static_getitem"))
            var = var_def.value.name

    def _gen_barrier(self):
        return compile_func_single_block(
            lambda: _barrier(),
            (),
            None,
            self,
            extra_globals={"_barrier": bodo.libs.distributed_api.barrier},
        )

    def _gen_reduce(self, reduce_var, reduce_op, scope, loc):
        """generate distributed reduction code for after parfor's local execution"""
        # concat reduction variables don't need aggregation since output is distributed
        # see test_concat_reduction
        if reduce_op == Reduce_Type.Concat:
            return []

        op_var = ir.Var(scope, mk_unique_var("$reduce_op"), loc)
        self.typemap[op_var.name] = types.int32
        op_assign = ir.Assign(ir.Const(reduce_op.value, loc), op_var, loc)

        def f(val, op):  # pragma: no cover
            bodo.libs.distributed_api.dist_reduce(val, op)

        f_ir = compile_to_numba_ir(
            f,
            {"bodo": bodo},
            self.typingctx,
            (self.typemap[reduce_var.name], types.int32),
            self.typemap,
            self.calltypes,
        )
        _, block = f_ir.blocks.popitem()

        replace_arg_nodes(block, [reduce_var, op_var])
        dist_reduce_nodes = [op_assign] + block.body[:-3]
        dist_reduce_nodes[-1].target = reduce_var
        return dist_reduce_nodes

    def _gen_init_reduce(self, reduce_var, reduce_op):
        """generate code to initialize reduction variables on non-root
        processors.
        """
        # TODO: support initialization for concat reductions
        if reduce_op == Reduce_Type.Concat:
            return []

        red_var_typ = self.typemap[reduce_var.name]
        el_typ = red_var_typ
        if is_np_array_typ(self.typemap[reduce_var.name]):
            el_typ = red_var_typ.dtype
        init_val = None
        pre_init_val = ""

        if reduce_op in [Reduce_Type.Sum, Reduce_Type.Or]:
            init_val = str(el_typ(0))
        if reduce_op == Reduce_Type.Prod:
            init_val = str(el_typ(1))
        if reduce_op == Reduce_Type.Min:
            if el_typ == types.bool_:
                init_val = "True"
            else:
                init_val = "numba.cpython.builtins.get_type_max_value(np.ones(1,dtype=np.{}).dtype)".format(
                    el_typ
                )
        if reduce_op == Reduce_Type.Max:
            if el_typ == types.bool_:
                init_val = "False"
            else:
                init_val = "numba.cpython.builtins.get_type_min_value(np.ones(1,dtype=np.{}).dtype)".format(
                    el_typ
                )
        if reduce_op in [Reduce_Type.Argmin, Reduce_Type.Argmax]:
            # don't generate initialization for argmin/argmax since they are not
            # initialized by user and correct initialization is already there
            return []

        assert init_val is not None

        if is_np_array_typ(self.typemap[reduce_var.name]):
            pre_init_val = "v = np.full_like(s, {}, s.dtype)".format(init_val)
            init_val = "v"

        f_text = "def f(s):\n  {}\n  s = bodo.libs.distributed_api._root_rank_select(s, {})".format(
            pre_init_val, init_val
        )
        loc_vars = {}
        exec(f_text, {}, loc_vars)
        f = loc_vars["f"]

        f_block = compile_to_numba_ir(
            f,
            {"bodo": bodo, "numba": numba, "np": np},
            self.typingctx,
            (red_var_typ,),
            self.typemap,
            self.calltypes,
        ).blocks.popitem()[1]
        replace_arg_nodes(f_block, [reduce_var])
        nodes = f_block.body[:-3]
        nodes[-1].target = reduce_var
        return nodes

    def _gen_init_code(self, blocks):
        """generate get_rank() and get_size() calls and store the variables
        to avoid repeated generation.
        """
        # get rank variable
        nodes = compile_func_single_block(
            lambda: _get_rank(),
            (),
            None,
            self,
            extra_globals={"_get_rank": bodo.libs.distributed_api.get_rank},
        )
        rank_var = nodes[-1].target
        # rename rank variable for readability
        rank_var.name = mk_unique_var("rank_var")
        self.typemap[rank_var.name] = types.int32
        self.rank_var = rank_var

        # get n_pes variable
        nodes += compile_func_single_block(
            lambda: _get_size(),
            (),
            None,
            self,
            extra_globals={"_get_size": bodo.libs.distributed_api.get_size},
        )
        n_pes_var = nodes[-1].target
        # rename n_pes variable for readability
        n_pes_var.name = mk_unique_var("n_pes_var")
        self.typemap[n_pes_var.name] = types.int32
        self.n_pes_var = n_pes_var

        # add nodes to first block
        topo_order = find_topo_order(blocks)
        first_block = blocks[topo_order[0]]
        first_block.body = nodes + first_block.body
        return

    def _set_last_arg_to_true(self, rhs):
        """set last argument of call expr 'rhs' to True, assuming that it is an Omitted
        arg with value of False.
        This is usually used for Bodo overloads that have an extra flag as last argument
        to enable parallelism.
        """
        call_type = self.calltypes.pop(rhs)
        assert call_type.args[-1] == types.Omitted(False)
        self.calltypes[rhs] = self.typemap[rhs.func.name].get_call_type(
            self.typingctx, call_type.args[:-1] + (types.Omitted(True),), {}
        )

    def _update_avail_vars(self, avail_vars, nodes):
        for stmt in nodes:
            if type(stmt) in numba.core.analysis.ir_extension_usedefs:
                def_func = numba.core.analysis.ir_extension_usedefs[type(stmt)]
                _uses, defs = def_func(stmt)
                avail_vars |= defs
            if is_assign(stmt):
                avail_vars.add(stmt.target.name)

    def _is_array_access_stmt(self, stmt):
        if is_get_setitem(stmt):
            return True

        if is_call_assign(stmt):
            rhs = stmt.value
            fdef = guard(find_callname, self.func_ir, rhs, self.typemap)
            if fdef in (
                ("isna", "bodo.libs.array_kernels"),
                ("setna", "bodo.libs.array_kernels"),
                ("str_arr_item_to_numeric", "bodo.libs.str_arr_ext"),
                ("setitem_str_arr_ptr", "bodo.libs.str_arr_ext"),
                ("get_str_arr_item_length", "bodo.libs.str_arr_ext"),
                ("inplace_eq", "bodo.libs.str_arr_ext"),
                ("str_arr_setitem_int_to_str", "bodo.libs.str_arr_ext"),
                ("str_arr_setitem_NA_str", "bodo.libs.str_arr_ext"),
                ("str_arr_set_not_na", "bodo.libs.str_arr_ext"),
                ("get_split_view_index", "bodo.hiframes.split_impl"),
                ("get_bit_bitmap_arr", "bodo.libs.int_arr_ext"),
                ("set_bit_to_arr", "bodo.libs.int_arr_ext"),
            ):
                return True

        return False

    def _fix_index_var(self, index_var):
        if index_var is None:  # TODO: fix None index in static_getitem/setitem
            return None

        # fix index if copy propagation didn't work
        ind_def = self.func_ir._definitions[index_var.name]
        if len(ind_def) == 1 and isinstance(ind_def[0], ir.Var):
            return ind_def[0]

        return index_var

    def _get_dim1_index_name(self, index_name):
        """given index variable name 'index_name', get index varibale name for the first
        dimension if it is a tuple. Also, get the first definition of the variable name
        if available. This helps matching the index name to first loop index name of
        parfors.
        """

        # multi-dim case
        tup_list = guard(find_build_tuple, self.func_ir, index_name)
        if tup_list is not None:
            assert len(tup_list) > 0
            index_name = tup_list[0].name

        # fix index if copy propagation didn't work
        ind_def = self.func_ir._definitions[index_name]
        if len(ind_def) == 1 and isinstance(ind_def[0], ir.Var):
            index_name = ind_def[0].name

        return index_name

    def _get_tuple_varlist(self, tup_var, out):
        """get the list of variables that hold values in the tuple variable.
        add node to out if code generation needed.
        """
        t_list = guard(find_build_tuple, self.func_ir, tup_var)
        if t_list is not None:
            return t_list
        assert isinstance(self.typemap[tup_var.name], types.UniTuple)
        ndims = self.typemap[tup_var.name].count
        f_text = "def f(tup_var):\n"
        for i in range(ndims):
            f_text += "  val{} = tup_var[{}]\n".format(i, i)
        loc_vars = {}
        exec(f_text, {}, loc_vars)
        f = loc_vars["f"]
        f_block = compile_to_numba_ir(
            f,
            {},
            self.typingctx,
            (self.typemap[tup_var.name],),
            self.typemap,
            self.calltypes,
        ).blocks.popitem()[1]
        replace_arg_nodes(f_block, [tup_var])
        nodes = f_block.body[:-3]
        vals_list = []
        for stmt in nodes:
            assert isinstance(stmt, ir.Assign)
            rhs = stmt.value
            assert isinstance(rhs, (ir.Var, ir.Const, ir.Expr))
            if isinstance(rhs, ir.Expr):
                assert rhs.op == "static_getitem"
                vals_list.append(stmt.target)
        out += nodes
        return vals_list

    def _get_arr_ndim(self, arrname):
        if self.typemap[arrname] == string_array_type:
            return 1
        return self.typemap[arrname].ndim

    def _is_1D_arr(self, arr_name):
        # some arrays like stencil buffers are added after analysis so
        # they are not in dists list
        return (
            arr_name in self._dist_analysis.array_dists
            and self._dist_analysis.array_dists[arr_name] == Distribution.OneD
        )

    def _is_1D_or_1D_Var_arr(self, arr_name):
        return (
            arr_name in self._dist_analysis.array_dists
            and self._dist_analysis.array_dists[arr_name]
            in (
                Distribution.OneD,
                Distribution.OneD_Var,
            )
        )

    def _is_1D_Var_arr(self, arr_name):
        # some arrays like stencil buffers are added after analysis so
        # they are not in dists list
        return (
            arr_name in self._dist_analysis.array_dists
            and self._dist_analysis.array_dists[arr_name] == Distribution.OneD_Var
        )

    def _is_1D_tup(self, var_name):
        return var_name in self._dist_analysis.array_dists and all(
            a == Distribution.OneD for a in self._dist_analysis.array_dists[var_name]
        )

    def _is_1D_Var_tup(self, var_name):
        return var_name in self._dist_analysis.array_dists and all(
            a == Distribution.OneD_Var
            for a in self._dist_analysis.array_dists[var_name]
        )

    def _is_REP(self, arr_name):
        return (
            arr_name not in self._dist_analysis.array_dists
            or self._dist_analysis.array_dists[arr_name] == Distribution.REP
        )


def _set_getsetitem_index(node, new_ind):
    if (isinstance(node, ir.Expr) and node.op == "static_getitem") or isinstance(
        node, ir.StaticSetItem
    ):
        node.index_var = new_ind
        node.index = None
        return

    assert (isinstance(node, ir.Expr) and node.op == "getitem") or isinstance(
        node, ir.SetItem
    )
    node.index = new_ind


def dprint(*s):  # pragma: no cover
    if debug_prints():
        print(*s)


def _get_array_var_from_size(size_var, func_ir):
    """
    Return 'arr' from pattern 'size_var = len(arr)' or 'size_var = arr.shape[0]' if
    exists. Otherwise, raise GuardException.
    """
    size_def = get_definition(func_ir, size_var)

    # len(arr) case
    if is_call(size_def) and guard(find_callname, func_ir, size_def) == (
        "len",
        "builtins",
    ):
        return size_def.args[0]

    require(is_expr(size_def, "static_getitem") and size_def.index == 0)
    shape_var = size_def.value
    get_attr = get_definition(func_ir, shape_var)
    require(is_expr(get_attr, "getattr") and get_attr.attr == "shape")
    arr_var = get_attr.value
    return arr_var


def find_available_vars(blocks, cfg, init_avail=None):
    """
    Finds available variables at entry point of each basic block by gathering all
    variables defined in the block's dominators in CFG.
    """
    # TODO: unittest
    in_avail_vars = defaultdict(set)
    var_def_map = numba.core.analysis.compute_use_defs(blocks).defmap

    if init_avail:
        assert 0 in blocks
        for label in var_def_map:
            in_avail_vars[label] = init_avail.copy()

    for label, doms in cfg.dominators().items():
        strict_doms = doms - {label}
        for d in strict_doms:
            in_avail_vars[label] |= var_def_map[d]

    return in_avail_vars


# copied from Numba and modified to avoid ir.Del generation, which is invalid in 0.49
# https://github.com/numba/numba/blob/1ea770564cb3c0c6cb9d8ab92e7faf23cd4c4c19/numba/parfors/parfor.py#L3050
def lower_parfor_sequential(typingctx, func_ir, typemap, calltypes):
    ir_utils._max_label = max(
        ir_utils._max_label, ir_utils.find_max_label(func_ir.blocks)
    )
    parfor_found = False
    new_blocks = {}
    for (block_label, block) in func_ir.blocks.items():
        block_label, parfor_found = _lower_parfor_sequential_block(
            block_label, block, new_blocks, typemap, calltypes, parfor_found
        )
        # old block stays either way
        new_blocks[block_label] = block
    func_ir.blocks = new_blocks
    # rename only if parfor found and replaced (avoid test_flow_control error)
    if parfor_found:
        func_ir.blocks = rename_labels(func_ir.blocks)
    dprint_func_ir(func_ir, "after parfor sequential lowering")
    simplify(func_ir, typemap, calltypes)
    dprint_func_ir(func_ir, "after parfor sequential simplify")
    # changed from Numba code: comment out id.Del generation that causes errors in 0.49
    # # add dels since simplify removes dels
    # post_proc = postproc.PostProcessor(func_ir)
    # post_proc.run(True)
