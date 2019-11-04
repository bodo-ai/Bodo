# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Parallelizes the IR for distributed execution and inserts MPI calls.
"""
import operator
import types as pytypes  # avoid confusion with numba.types
import copy
import warnings
from collections import defaultdict
import math
import numpy as np
import pandas as pd
import numba
from numba import ir, types, typing, config, numpy_support, ir_utils, postproc
from numba.ir_utils import (
    mk_unique_var,
    replace_vars_inner,
    find_topo_order,
    dprint_func_ir,
    remove_dead,
    mk_alloc,
    get_global_func_typ,
    get_name_var_table,
    get_call_table,
    get_tuple_table,
    remove_dels,
    compile_to_numba_ir,
    replace_arg_nodes,
    guard,
    get_definition,
    require,
    GuardException,
    find_callname,
    build_definitions,
    find_build_sequence,
    find_const,
    is_get_setitem,
    compute_cfg_from_blocks,
)
from numba.typing import signature
from numba.parfor import (
    get_parfor_reductions,
    get_parfor_params,
    wrap_parfor_blocks,
    unwrap_parfor_blocks,
)
from numba.parfor import Parfor, lower_parfor_sequential
import numpy as np

import bodo
from bodo.io.h5_api import h5file_type, h5group_type
from bodo.libs import distributed_api
from bodo.libs.str_ext import string_type, unicode_to_utf8_and_len
from bodo.libs.str_arr_ext import string_array_type
from bodo.transforms.distributed_analysis import (
    Distribution,
    DistributedAnalysis,
    _get_array_accesses,
)

import bodo.utils.utils
from bodo.utils.transform import compile_func_single_block
from bodo.utils.utils import (
    is_alloc_callname,
    is_whole_slice,
    get_slice_step,
    is_np_array_typ,
    find_build_tuple,
    debug_prints,
    ReplaceFunc,
    gen_getitem,
    is_call,
    is_const_slice,
    is_assign,
    is_expr,
    is_call_assign,
    get_getsetitem_index_var,
)
from bodo.libs.distributed_api import Reduce_Type
from bodo.hiframes.pd_dataframe_ext import DataFrameType

distributed_run_extensions = {}

# analysis data for debugging
dist_analysis = None
fir_text = None


class DistributedPass(object):
    """analyze program and transfrom to distributed"""

    def __init__(self, func_ir, typingctx, targetctx, typemap, calltypes, metadata):
        self.func_ir = func_ir
        self.typingctx = typingctx
        self.targetctx = targetctx
        self.typemap = typemap
        self.calltypes = calltypes
        # Loc object of current location being translated
        self.curr_loc = self.func_ir.loc
        self.metadata = metadata
        self.arr_analysis = numba.array_analysis.ArrayAnalysis(
            self.typingctx, self.func_ir, self.typemap, self.calltypes
        )

        self._dist_analysis = None
        self._T_arrs = None  # set of transposed arrays (taken from analysis)
        self._1D_parfor_starts = {}
        self._1D_Var_parfor_starts = {}
        # keep start vars for 1D dist to reuse in parfor loop array accesses
        self._start_vars = {}

    def run(self):
        remove_dels(self.func_ir.blocks)
        dprint_func_ir(self.func_ir, "starting distributed pass")
        self.func_ir._definitions = build_definitions(self.func_ir.blocks)
        self.arr_analysis.run(self.func_ir.blocks)
        dist_analysis_pass = DistributedAnalysis(
            self.func_ir, self.typemap, self.calltypes, self.typingctx, self.metadata
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
        post_proc = postproc.PostProcessor(self.func_ir)
        post_proc.run()

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
                        self,
                    )
                elif isinstance(inst, Parfor):
                    out_nodes = self._run_parfor(inst, equiv_set, avail_vars)
                    # run dist pass recursively
                    p_blocks = wrap_parfor_blocks(inst)
                    #
                    self._run_dist_pass(p_blocks, avail_vars)
                    unwrap_parfor_blocks(inst)
                elif isinstance(inst, ir.Assign):
                    rhs = inst.value
                    if isinstance(rhs, ir.Expr):
                        out_nodes = self._run_expr(inst, equiv_set, avail_vars)
                elif isinstance(inst, (ir.StaticSetItem, ir.SetItem)):
                    out_nodes = []
                    index_var = get_getsetitem_index_var(inst, self.typemap, out_nodes)
                    out_nodes += self._run_getsetitem(
                        inst.target, index_var, inst, inst, equiv_set, avail_vars
                    )
                elif isinstance(inst, ir.Return):
                    out_nodes = self._gen_barrier() + [inst]
                elif isinstance(inst, ir.Print):
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
            and (self._is_1D_arr(rhs.value.name) or self._is_1D_Var_arr(rhs.value.name))
        ):
            return self._run_array_shape(inst.target, rhs.value, equiv_set)

        # array.size
        if (
            rhs.op == "getattr"
            and rhs.attr == "size"
            and (self._is_1D_arr(rhs.value.name) or self._is_1D_Var_arr(rhs.value.name))
        ):
            return self._run_array_size(inst.target, rhs.value, equiv_set)

        # RangeIndex._stop, get global value
        if (
            rhs.op == "getattr"
            and rhs.attr == "_stop"
            and isinstance(
                self.typemap[rhs.value.name], bodo.hiframes.pd_index_ext.RangeIndexType
            )
            and (self._is_1D_arr(rhs.value.name) or self._is_1D_Var_arr(rhs.value.name))
        ):
            return [inst] + compile_func_single_block(
                lambda r: bodo.libs.distributed_api.dist_reduce(
                    r._stop - r._start, _op
                ),
                (rhs.value,),
                inst.target,
                self,
                extra_globals={"_op": np.int32(Reduce_Type.Sum.value)},
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
            and (self._is_1D_arr(rhs.value.name) or self._is_1D_Var_arr(rhs.value.name))
        ):
            return [inst] + compile_func_single_block(
                lambda r: 0,
                (rhs.value,),
                inst.target,
                self,
                extra_globals={"_op": np.int32(Reduce_Type.Sum.value)},
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

        # divide 1D alloc
        # XXX allocs should be matched before going to _run_call_np
        if self._is_1D_arr(lhs) and is_alloc_callname(func_name, func_mod):
            # XXX for pre_alloc_string_array(n, nc), we assume nc is local
            # value (updated only in parfor like _str_replace_regex_impl)
            size_var = rhs.args[0]
            out, new_size_var = self._run_alloc(size_var, lhs, scope, loc)
            # empty_inferred is tuple for some reason
            rhs.args = list(rhs.args)
            rhs.args[0] = new_size_var
            out.append(assign)
            return out

        # fix 1D_Var allocs in case global len of another 1DVar is used
        if self._is_1D_Var_arr(lhs) and is_alloc_callname(func_name, func_mod):
            size_var = rhs.args[0]
            out, new_size_var = self._fix_1D_Var_alloc(
                size_var, lhs, scope, loc, equiv_set, avail_vars
            )
            # empty_inferred is tuple for some reason
            rhs.args = list(rhs.args)
            rhs.args[0] = new_size_var
            out.append(assign)
            return out

        # numpy direct functions
        if isinstance(func_mod, str) and func_mod == "numpy":
            return self._run_call_np(lhs, func_name, assign, rhs.args, equiv_set)

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
            and (
                self._is_1D_arr(rhs.args[0].name)
                or self._is_1D_Var_arr(rhs.args[0].name)
            )
        ):
            arr = rhs.args[0]
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
            start_tuple_call = ir.Expr.build_tuple(
                [start_var] + prev_starts[1:], loc
            )
            starts_assign = ir.Assign(start_tuple_call, starts_var, loc)
            rhs.args[2] = starts_var

            # new counts
            counts_var = ir.Var(scope, mk_unique_var("$h5_counts"), loc)
            self.typemap[counts_var.name] = types.UniTuple(types.int64, ndims)
            prev_counts = self._get_tuple_varlist(rhs.args[3], nodes)
            count_tuple_call = ir.Expr.build_tuple(
                [count_var] + prev_counts[1:], loc
            )
            counts_assign = ir.Assign(count_tuple_call, counts_var, loc)

            nodes += [starts_assign, counts_assign, assign]
            rhs.args[3] = counts_var
            rhs.args[4] = one_var

            # set parallel arg in file open
            file_varname = rhs.args[0].name
            self._file_open_set_parallel(file_varname)
            return nodes

        # TODO: fix numba.extending
        if bodo.config._has_xenon and (
            fdef == ("read_xenon_col", "numba.extending")
            and self._is_1D_arr(rhs.args[3].name)
        ):
            arr = rhs.args[3]
            nodes = []
            size_var = self._get_dist_var_len(arr, nodes, equiv_set)
            start_var = self._get_1D_start(size_var, avail_vars, nodes)
            count_var = self._get_1D_count(size_var, nodes)
            assert self.typemap[arr.name].ndim == 1, "only 1D arrs in Xenon"
            rhs.args += [start_var, count_var]

            def f(
                connect_tp, dset_tp, col_id_tp, column_tp, schema_arr_tp, start, count
            ):  # pragma: no cover
                return bodo.io.xenon_ext.read_xenon_col_parallel(
                    connect_tp,
                    dset_tp,
                    col_id_tp,
                    column_tp,
                    schema_arr_tp,
                    start,
                    count,
                )

            return nodes + compile_func_single_block(f, rhs.args, assign.target, self)

        if bodo.config._has_xenon and (
            fdef == ("read_xenon_str", "numba.extending") and self._is_1D_arr(lhs)
        ):
            arr = lhs
            size_var = rhs.args[3]
            assert self.typemap[size_var.name] == types.intp
            out = []
            start_var = self._get_1D_start(size_var, avail_vars, out)
            count_var = self._get_1D_count(size_var, out)
            rhs.args.remove(size_var)
            rhs.args.append(start_var)
            rhs.args.append(count_var)

            def f(
                connect_tp, dset_tp, col_id_tp, schema_arr_tp, start_tp, count_tp
            ):  # pragma: no cover
                return bodo.io.xenon_ext.read_xenon_str_parallel(
                    connect_tp, dset_tp, col_id_tp, schema_arr_tp, start_tp, count_tp
                )

            out += compile_func_single_block(f, rhs.args, assign.target, self)
            return out

        if fdef == (
            "get_split_view_index",
            "bodo.hiframes.split_impl",
        ) and self._dist_arr_needs_adjust(rhs.args[0].name):
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

        if fdef == (
            "setitem_str_arr_ptr",
            "bodo.libs.str_arr_ext",
        ) and self._dist_arr_needs_adjust(rhs.args[0].name):
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

        if fdef == (
            "str_arr_item_to_numeric",
            "bodo.libs.str_arr_ext",
        ) and self._dist_arr_needs_adjust(rhs.args[0].name):
            # TODO: test parallel
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

        if fdef == ("setitem_arr_nan", "bodo.ir.join") and self._dist_arr_needs_adjust(
            rhs.args[0].name
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

        if fdef in (
            ("isna", "bodo.hiframes.api"),
            ("get_bit_bitmap_arr", "bodo.libs.int_arr_ext"),
            ("set_bit_to_arr", "bodo.libs.int_arr_ext"),
            ("get_str_arr_item_length", "bodo.libs.str_arr_ext"),
        ) and self._dist_arr_needs_adjust(rhs.args[0].name):
            # fix index in call to isna
            arr = rhs.args[0]
            ind = self._fix_index_var(rhs.args[1])
            start_var, out = self._get_parallel_access_start_var(
                arr, equiv_set, ind, avail_vars
            )
            out += self._get_ind_sub(ind, start_var)
            rhs.args[1] = out[-1].target
            out.append(assign)

        if fdef == ("rolling_fixed", "bodo.hiframes.rolling") and (
            self._is_1D_arr(rhs.args[0].name) or self._is_1D_Var_arr(rhs.args[0].name)
        ):
            # set parallel flag to true
            true_var = ir.Var(scope, mk_unique_var("true_var"), loc)
            self.typemap[true_var.name] = types.boolean
            rhs.args[3] = true_var
            out = [ir.Assign(ir.Const(True, loc), true_var, loc), assign]

        if fdef == ("rolling_variable", "bodo.hiframes.rolling") and (
            self._is_1D_arr(rhs.args[0].name) or self._is_1D_Var_arr(rhs.args[0].name)
        ):
            # set parallel flag to true
            true_var = ir.Var(scope, mk_unique_var("true_var"), loc)
            self.typemap[true_var.name] = types.boolean
            rhs.args[4] = true_var
            out = [ir.Assign(ir.Const(True, loc), true_var, loc), assign]

        if (
            func_mod == "bodo.hiframes.rolling"
            and func_name in ("shift", "pct_change")
            and (
                self._is_1D_arr(rhs.args[0].name)
                or self._is_1D_Var_arr(rhs.args[0].name)
            )
        ):
            # set parallel flag to true
            true_var = ir.Var(scope, mk_unique_var("true_var"), loc)
            self.typemap[true_var.name] = types.boolean
            rhs.args[2] = true_var
            out = [ir.Assign(ir.Const(True, loc), true_var, loc), assign]

        if fdef == ("quantile", "bodo.libs.array_kernels") and (
            self._is_1D_arr(rhs.args[0].name) or self._is_1D_Var_arr(rhs.args[0].name)
        ):
            arr = rhs.args[0]
            nodes = []
            size_var = self._get_dist_var_len(arr, nodes, equiv_set)
            rhs.args.append(size_var)

            f = lambda arr, q, size: bodo.libs.array_kernels.quantile_parallel(
                arr, q, size
            )
            return nodes + compile_func_single_block(f, rhs.args, assign.target, self)

        if fdef == ("nunique", "bodo.hiframes.api") and (
            self._is_1D_arr(rhs.args[0].name) or self._is_1D_Var_arr(rhs.args[0].name)
        ):
            f = lambda arr: bodo.hiframes.api.nunique_parallel(arr)
            return compile_func_single_block(f, rhs.args, assign.target, self)

        if fdef == ("unique", "bodo.hiframes.api") and (
            self._is_1D_arr(rhs.args[0].name) or self._is_1D_Var_arr(rhs.args[0].name)
        ):
            f = lambda arr: bodo.hiframes.api.unique_parallel(arr)
            return compile_func_single_block(f, rhs.args, assign.target, self)

        if fdef == ("nlargest", "bodo.libs.array_kernels") and (
            self._is_1D_arr(rhs.args[0].name) or self._is_1D_Var_arr(rhs.args[0].name)
        ):
            f = lambda arr, I, k, i, f: bodo.libs.array_kernels.nlargest_parallel(
                arr, I, k, i, f
            )
            return compile_func_single_block(f, rhs.args, assign.target, self)

        if fdef == ("nancorr", "bodo.libs.array_kernels") and (
            self._is_1D_arr(rhs.args[0].name) or self._is_1D_Var_arr(rhs.args[0].name)
        ):
            f = lambda mat, cov, minpv: bodo.libs.array_kernels.nancorr(
                mat, cov, minpv, True
            )
            return compile_func_single_block(f, rhs.args, assign.target, self)

        if fdef == ("median", "bodo.libs.array_kernels") and (
            self._is_1D_arr(rhs.args[0].name) or self._is_1D_Var_arr(rhs.args[0].name)
        ):
            f = lambda arr: bodo.libs.array_kernels.median(arr, True)
            return compile_func_single_block(f, rhs.args, assign.target, self)

        if fdef == ("duplicated", "bodo.libs.array_kernels") and (
            self._is_1D_tup(rhs.args[0].name) or self._is_1D_Var_tup(rhs.args[0].name)
        ):
            f = lambda arr, ind_arr: bodo.libs.array_kernels.duplicated(
                arr, ind_arr, True
            )
            return compile_func_single_block(f, rhs.args, assign.target, self)

        if fdef == ("drop_duplicates", "bodo.libs.array_kernels") and (
            self._is_1D_tup(rhs.args[0].name) or self._is_1D_Var_tup(rhs.args[0].name)
        ):
            f = lambda arr, ind_arr: bodo.libs.array_kernels.drop_duplicates(
                arr, ind_arr, True
            )
            return compile_func_single_block(f, rhs.args, assign.target, self)

        if fdef == ("convert_rec_to_tup", "bodo.hiframes.api"):
            # optimize Series back to back map pattern with tuples
            # TODO: create another optimization pass?
            arg_def = guard(get_definition, self.func_ir, rhs.args[0])
            if is_call(arg_def) and guard(find_callname, self.func_ir, arg_def) == (
                "convert_tup_to_rec",
                "bodo.hiframes.api",
            ):
                assign.value = arg_def.args[0]
            return out

        if fdef == ("dist_return", "bodo.libs.distributed_api"):
            # always rebalance returned distributed arrays
            # TODO: need different flag for 1D_Var return (distributed_var)?
            # TODO: rebalance strings?
            # return [assign]  # self._run_call_rebalance_array(lhs, assign, rhs.args)
            assign.value = rhs.args[0]
            return [assign]

        if fdef == ("threaded_return", "bodo.libs.distributed_api"):
            assign.value = rhs.args[0]
            return [assign]

        if fdef == ("rebalance_array", "bodo.libs.distributed_api"):
            return self._run_call_rebalance_array(lhs, assign, rhs.args)

        if fdef == ("file_read", "bodo.io.np_io") and (
            self._is_1D_arr(rhs.args[1].name) or self._is_1D_Var_arr(rhs.args[1].name)
        ):
            fname = rhs.args[0]
            arr = rhs.args[1]
            nodes, start_var, count_var = self._get_dist_var_start_count(
                arr, equiv_set, avail_vars
            )

            def impl(fname, data_ptr, start, count):  # pragma: no cover
                return bodo.io.np_io.file_read_parallel(fname, data_ptr, start, count)

            return nodes + compile_func_single_block(
                impl, [fname, arr, start_var, count_var], assign.target, self
            )

        # replace get_type_max_value(arr.dtype) since parfors
        # arr.dtype transformation produces invalid code for dt64
        if fdef == ("get_type_max_value", "numba.targets.builtins"):
            if self.typemap[rhs.args[0].name] == types.DType(types.NPDatetime("ns")):
                # XXX: not using replace since init block of parfor can't be
                # processed. test_series_idxmin
                # return self._replace_func(
                #     lambda: bodo.hiframes.pd_timestamp_ext.integer_to_dt64(
                #         numba.targets.builtins.get_type_max_value(
                #             numba.types.int64)), [])
                f_block = compile_to_numba_ir(
                    lambda: bodo.hiframes.pd_timestamp_ext.integer_to_dt64(
                        numba.targets.builtins.get_type_max_value(numba.types.uint64)
                    ),
                    {"bodo": bodo, "numba": numba},
                    self.typingctx,
                    (),
                    self.typemap,
                    self.calltypes,
                ).blocks.popitem()[1]
                out = f_block.body[:-2]
                out[-1].target = assign.target

        if fdef == ("get_type_min_value", "numba.targets.builtins"):
            if self.typemap[rhs.args[0].name] == types.DType(types.NPDatetime("ns")):
                f_block = compile_to_numba_ir(
                    lambda: bodo.hiframes.pd_timestamp_ext.integer_to_dt64(
                        numba.targets.builtins.get_type_min_value(numba.types.uint64)
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

    def _run_call_np(self, lhs, func_name, assign, args, equiv_set):
        """transform np.func() calls
        """
        # allocs are handled separately
        assert not (
            (self._is_1D_Var_arr(lhs) or self._is_1D_arr(lhs))
            and func_name in bodo.utils.utils.np_alloc_callnames
        ), (
            "allocation calls handled separately "
            "'empty', 'zeros', 'ones', 'full' etc."
        )
        out = [assign]
        scope = assign.target.scope
        loc = assign.loc

        if func_name == "reshape" and self._is_1D_arr(args[0].name):
            # TODO: handle and test reshape properly
            return self._run_reshape(assign, args[0], args[1:], equiv_set)

        if func_name == "ravel" and self._is_1D_arr(args[0].name):
            assert self.typemap[args[0].name].ndim == 1, "only 1D ravel supported"

        if func_name in ("cumsum", "cumprod") and (
            self._is_1D_arr(args[0].name) or self._is_1D_Var_arr(args[0].name)
        ):
            in_arr_var = args[0]
            lhs_var = assign.target
            # TODO: compute inplace if input array is dead
            def impl(A):
                B = np.empty_like(A)
                _func(A, B)
                return B

            func = getattr(bodo.libs.distributed_api, "dist_" + func_name)
            return compile_func_single_block(
                impl, [in_arr_var], lhs_var, self, extra_globals={"_func": func}
            )

        # sum over the first axis is distributed, A.sum(0)
        if func_name == "sum" and len(args) == 2:
            axis_def = guard(get_definition, self.func_ir, args[1])
            if isinstance(axis_def, ir.Const) and axis_def.value == 0:
                reduce_op = Reduce_Type.Sum
                reduce_var = assign.target
                return out + self._gen_reduce(reduce_var, reduce_op, scope, loc)

        if func_name == "dot":
            return self._run_call_np_dot(lhs, assign, args)

        return out

    def _run_call_array(self, lhs, arr, func_name, assign, args, equiv_set, avail_vars):
        #
        out = [assign]

        # HACK support A.reshape(n, 1) for 1D_Var
        if func_name == "reshape" and self._is_1D_Var_arr(arr.name):
            assert len(args) == 2 and guard(find_const, self.func_ir, args[1]) == 1
            size_var = args[0]
            out, new_size_var = self._fix_1D_Var_alloc(
                size_var, lhs, assign.target.scope, assign.loc, equiv_set, avail_vars
            )
            # empty_inferred is tuple for some reason
            assign.value.args = list(args)
            assign.value.args[0] = new_size_var
            out.append(assign)
            return out

        if func_name == "reshape" and self._is_1D_arr(arr.name):
            return self._run_reshape(assign, arr, args, equiv_set)

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
        if func_name == "to_csv" and (
            self._is_1D_arr(df.name) or self._is_1D_Var_arr(df.name)
        ):
            # set index to proper range if None
            # avoid header for non-zero ranks
            # write to string then parallel file write
            # df.to_csv(fname) ->
            # l = len(df)
            # index_start = dist_exscan(l)
            # df2 = df(index=range(index_start, index_start+l))
            # header = header and is_root  # only first line has header
            # str_out = df2.to_csv(None, header=header)
            # bodo.io.np_io._file_write_parallel(fname, str_out)

            df_typ = self.typemap[df.name]
            rhs = assign.value
            fname = args[0]
            # convert StringLiteral to Unicode to make ._data available
            self.typemap.pop(fname.name)
            self.typemap[fname.name] = string_type

            # update df index and get to_csv from new df
            nodes = self._fix_parallel_df_index(df)
            new_df = nodes[-1].target
            new_df_typ = self.typemap[new_df.name]
            new_to_csv = ir.Expr.getattr(new_df, "to_csv", new_df.loc)
            new_func = ir.Var(new_df.scope, mk_unique_var("func"), new_df.loc)
            self.typemap[new_func.name] = self.typingctx.resolve_getattr(
                new_df_typ, "to_csv"
            )
            nodes.append(ir.Assign(new_to_csv, new_func, new_df.loc))
            rhs.func = new_func

            # # header = header and is_root
            kws = dict(rhs.kws)
            true_var = ir.Var(assign.target.scope, mk_unique_var("true"), rhs.loc)
            self.typemap[true_var.name] = types.bool_
            nodes.append(ir.Assign(ir.Const(True, new_df.loc), true_var, new_df.loc))
            header_var = self._get_arg("to_csv", rhs.args, kws, 5, "header", true_var)
            nodes += self._gen_is_root_and_cond(header_var)
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
            self.calltypes[rhs] = numba.typing.Signature(
                string_type, arg_typs, new_df_typ, call_type.pysig
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
            from bodo.libs import hio
            import llvmlite.binding as ll

            ll.add_symbol("file_write_parallel", hio.file_write_parallel)

            def f(fname, str_out):  # pragma: no cover
                utf8_str, utf8_len = unicode_to_utf8_and_len(str_out)
                start = bodo.libs.distributed_api.dist_exscan(utf8_len, _op)
                # TODO: unicode file name
                bodo.io.np_io._file_write_parallel(
                    fname._data, utf8_str, start, utf8_len, 1
                )

            return nodes + compile_func_single_block(
                f,
                [fname, str_out],
                assign.target,
                self,
                extra_globals={
                    "unicode_to_utf8_and_len": unicode_to_utf8_and_len,
                    "_op": np.int32(Reduce_Type.Sum.value),
                },
            )

        return [assign]

    def _gen_is_root_and_cond(self, cond_var):
        def f(cond):
            return cond & (bodo.libs.distributed_api.get_rank() == 0)

        f_block = compile_to_numba_ir(
            f,
            {"bodo": bodo},
            self.typingctx,
            (self.typemap[cond_var.name],),
            self.typemap,
            self.calltypes,
        ).blocks.popitem()[1]
        replace_arg_nodes(f_block, [cond_var])
        nodes = f_block.body[:-2]
        return nodes

    def _fix_parallel_df_index(self, df):
        def f(df):  # pragma: no cover
            l = len(df)
            start = bodo.libs.distributed_api.dist_exscan(l, _op)
            ind = np.arange(start, start + l)
            df2 = bodo.hiframes.pd_dataframe_ext.set_df_index(df, ind)
            return df2

        f_block = compile_to_numba_ir(
            f,
            {"bodo": bodo, "np": np, "_op": np.int32(Reduce_Type.Sum.value)},
            self.typingctx,
            (self.typemap[df.name],),
            self.typemap,
            self.calltypes,
        ).blocks.popitem()[1]
        replace_arg_nodes(f_block, [df])
        nodes = f_block.body[:-2]
        return nodes

    def _run_permutation_int(self, assign, args):
        lhs = assign.target
        n = args[0]

        def f(lhs, n):
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
        context = numba.targets.cpu.CPUContext(self.typingctx)
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

    def _run_reshape(self, assign, in_arr, args, equiv_set):
        lhs = assign.target
        scope = assign.target.scope
        loc = assign.target.loc
        if len(args) == 1:
            new_shape = args[0]
        else:
            # reshape can take list of ints
            new_shape = args
        # TODO: avoid alloc and copy if no communication necessary
        # get new local shape in reshape and set start/count vars like new allocation
        out, new_local_shape_var = self._run_alloc(new_shape, lhs.name, scope, loc)
        # get actual tuple for mk_alloc
        if isinstance(self.typemap[new_local_shape_var.name], types.BaseTuple):
            sh_list = guard(find_build_tuple, self.func_ir, new_local_shape_var)
            assert sh_list is not None, "invalid shape in reshape"
            new_local_shape_var = tuple(sh_list)
        dtype = self.typemap[in_arr.name].dtype
        out += mk_alloc(
            self.typemap, self.calltypes, lhs, new_local_shape_var, dtype, scope, loc
        )

        def f(
            lhs, in_arr, new_0dim_global_len, old_0dim_global_len, dtype_size
        ):  # pragma: no cover
            bodo.libs.distributed_api.dist_oneD_reshape_shuffle(
                lhs, in_arr, new_0dim_global_len, old_0dim_global_len, dtype_size
            )

        f_block = compile_to_numba_ir(
            f,
            {"bodo": bodo},
            self.typingctx,
            (
                self.typemap[lhs.name],
                self.typemap[in_arr.name],
                types.intp,
                types.intp,
                types.intp,
            ),
            self.typemap,
            self.calltypes,
        ).blocks.popitem()[1]
        dtype_ir = self.dtype_size_assign_ir(dtype, scope, loc)
        out.append(dtype_ir)
        lhs_size = self._get_dist_var_len(lhs, out, equiv_set)
        in_arr_size = self._get_dist_var_len(in_arr, out, equiv_set)
        replace_arg_nodes(
            f_block, [lhs, in_arr, lhs_size, in_arr_size, dtype_ir.target]
        )
        out += f_block.body[:-3]
        return out

    def _run_call_rebalance_array(self, lhs, assign, args):
        out = [assign]
        if not self._is_1D_Var_arr(args[0].name):
            if not self._is_1D_arr(args[0].name):
                warnings.warn("array {} is not 1D_Block_Var".format(args[0].name))
            return out

        arr = args[0]
        out = self._gen_1D_Var_len(arr)
        total_length = out[-1].target
        count_var = self._get_1D_count(total_length, out)

        def f(arr, count):  # pragma: no cover
            b_arr = bodo.libs.distributed_api.rebalance_array_parallel(arr, count)

        f_block = compile_to_numba_ir(
            f,
            {"bodo": bodo},
            self.typingctx,
            (self.typemap[arr.name], types.intp),
            self.typemap,
            self.calltypes,
        ).blocks.popitem()[1]
        replace_arg_nodes(f_block, [arr, count_var])
        out += f_block.body[:-3]
        out[-1].target = assign.target
        return out

    def _run_call_np_dot(self, lhs, assign, args):
        out = [assign]
        arg0 = args[0].name
        arg1 = args[1].name
        ndim0 = self.typemap[arg0].ndim
        ndim1 = self.typemap[arg1].ndim
        t0 = arg0 in self._T_arrs
        t1 = arg1 in self._T_arrs

        # reduction across dataset
        if self._is_1D_arr(arg0) and self._is_1D_arr(arg1):
            dprint("run dot dist reduce:", arg0, arg1)
            reduce_op = Reduce_Type.Sum
            reduce_var = assign.target
            out += self._gen_reduce(
                reduce_var, reduce_op, reduce_var.scope, reduce_var.loc
            )

        return out

    def _run_alloc(self, size_var, lhs, scope, loc):
        """ divides array sizes and assign its sizes/starts/counts attributes
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

    def _fix_1D_Var_alloc(self, size_var, lhs, scope, loc, equiv_set, avail_vars):
        """ 1D_Var allocs use global sizes of other 1D_var variables,
        so find the local size of one those variables for replacement.
        Assuming 1D_Var alloc is resulting from an operation with another
        1D_Var array and cannot be standalone.
        """
        out = []
        is_tuple = False

        # size is either integer or tuple
        if not isinstance(self.typemap[size_var.name], types.Integer):
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
        new_size_var = None
        for v in equiv_set.get_equiv_set(size_var):
            if "#" in v and self._is_1D_Var_arr(v.split("#")[0]):
                arr_name = v.split("#")[0]
                if arr_name not in avail_vars:
                    continue
                arr_var = ir.Var(size_var.scope, arr_name, size_var.loc)
                out += compile_func_single_block(
                    lambda A: len(A), (arr_var,), None, self
                )
                new_size_var = out[-1].target
                break

        if new_size_var is None:
            # Series.combine() uses max(s1, s2) to get output size
            size_def = guard(get_definition, self.func_ir, size_var)
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
        # non-Arrays like Series, DataFrame etc. are all 1 dim
        ndims = (
            self.typemap[arr.name].ndim
            if isinstance(self.typemap[arr.name], types.Array)
            else 1
        )

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
        out = [full_node]
        index_var = self._fix_index_var(index_var)

        # adjust parallel access indices (in parfors)
        # 1D_Var arrays need adjustment if 1D_Var parfor has start adjusted
        if (
            self._is_1D_arr(arr.name)
            or (
                self._is_1D_Var_arr(arr.name) and arr.name in self._1D_Var_parfor_starts
            )
        ) and (arr.name, index_var.name) in self._parallel_accesses:
            return self._run_parallel_access_getsetitem(
                arr, index_var, node, full_node, equiv_set, avail_vars
            )
        elif self._is_1D_arr(arr.name) and isinstance(
            node, (ir.StaticSetItem, ir.SetItem)
        ):
            is_multi_dim = False
            # we only consider 1st dimension for multi-dim arrays
            inds = guard(find_build_tuple, self.func_ir, index_var)
            if inds is not None:
                index_var = inds[0]
                is_multi_dim = True

            # no need for transformation for whole slices
            if guard(is_whole_slice, self.typemap, self.func_ir, index_var):
                return out

            # TODO: support multi-dim slice setitem like X[a:b, c:d]
            assert not is_multi_dim
            nodes, start_var, count_var = self._get_dist_var_start_count(
                arr, equiv_set, avail_vars
            )

            if isinstance(self.typemap[index_var.name], types.Integer):

                def f(A, val, index, chunk_start, chunk_count):  # pragma: no cover
                    bodo.libs.distributed_api._set_if_in_range(
                        A, val, index, chunk_start, chunk_count
                    )

                return nodes + compile_func_single_block(
                    f, [arr, node.value, index_var, start_var, count_var], None, self
                )

            assert isinstance(
                self.typemap[index_var.name], types.misc.SliceType
            ), "slice index expected"

            # convert setitem with global range to setitem with local range
            # that overlaps with the local array chunk
            def f(A, val, start, stop, chunk_start, chunk_count):  # pragma: no cover
                loc_start, loc_stop = bodo.libs.distributed_api._get_local_range(
                    start, stop, chunk_start, chunk_count
                )
                A[loc_start:loc_stop] = val

            slice_call = get_definition(self.func_ir, index_var)
            slice_start = slice_call.args[0]
            slice_stop = slice_call.args[1]
            return nodes + compile_func_single_block(
                f,
                [arr, node.value, slice_start, slice_stop, start_var, count_var],
                None,
                self,
            )
            # print_node = ir.Print([start_var, end_var], None, loc)
            # self.calltypes[print_node] = signature(types.none, types.int64, types.int64)
            # out.append(print_node)
            #
            # setitem_attr_var = ir.Var(scope, mk_unique_var("$setitem_attr"), loc)
            # setitem_attr_call = ir.Expr.getattr(self._g_dist_var, "dist_setitem", loc)
            # self.typemap[setitem_attr_var.name] = get_global_func_typ(
            #                                 distributed_api.dist_setitem)
            # setitem_assign = ir.Assign(setitem_attr_call, setitem_attr_var, loc)
            # out = [setitem_assign]
            # setitem_call = ir.Expr.call(setitem_attr_var,
            #                     [arr, index_var, node.value, start, count], (), loc)
            # self.calltypes[setitem_call] = self.typemap[setitem_attr_var.name].get_call_type(
            #     self.typingctx, [self.typemap[arr.name],
            #     self.typemap[index_var.name], self.typemap[node.value.name],
            #     types.intp, types.intp], {})
            # err_var = ir.Var(scope, mk_unique_var("$setitem_err_var"), loc)
            # self.typemap[err_var.name] = types.int32
            # setitem_assign = ir.Assign(setitem_call, err_var, loc)
            # out.append(setitem_assign)

        elif (self._is_1D_arr(arr.name) or self._is_1D_Var_arr(arr.name)) and (
            is_expr(node, "getitem") or is_expr(node, "static_getitem")
        ):
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
            if guard(is_whole_slice, self.typemap, self.func_ir, index_var):
                # A = X[:,3]
                pass

            # strided whole slice
            # e.g. A = X[::2,5]
            elif guard(
                is_whole_slice,
                self.typemap,
                self.func_ir,
                index_var,
                accept_stride=True,
            ):
                # FIXME: we use rebalance array to handle the output array
                # TODO: convert to neighbor exchange
                # on each processor, the slice has to start from an offset:
                # |step-(start%step)|
                in_arr = full_node.value.value
                start_var, out = self._get_dist_start_var(in_arr, equiv_set, avail_vars)
                step = get_slice_step(self.typemap, self.func_ir, index_var)

                def f(A, start, step):
                    offset = abs(step - (start % step)) % step
                    return A[offset::step]

                out += compile_func_single_block(
                    f, [in_arr, start_var, step], None, self
                )
                imb_arr = out[-1].target

                # call rebalance
                self._dist_analysis.array_dists[imb_arr.name] = Distribution.OneD_Var
                out += self._run_call_rebalance_array(lhs.name, full_node, [imb_arr])
                out[-1].target = lhs

            # general slice access like A[3:7]
            elif self._is_REP(lhs.name) and isinstance(index_typ, types.SliceType):
                # TODO: handle multi-dim cases like A[:3, 4]
                # cases like S.head()
                # bcast if all in rank 0, otherwise gatherv
                in_arr = full_node.value.value
                start_var, nodes = self._get_dist_start_var(
                    in_arr, equiv_set, avail_vars
                )
                size_var = self._get_dist_var_len(in_arr, nodes, equiv_set)
                is_1D = self._is_1D_arr(arr.name)
                return nodes + compile_func_single_block(
                    lambda arr, slice_index, start, tot_len: bodo.libs.distributed_api.slice_getitem(
                        arr, slice_index, start, tot_len, _is_1D
                    ),
                    [in_arr, index_var, start_var, size_var],
                    lhs,
                    self,
                    extra_globals={"_is_1D": is_1D},
                )
            # int index like A[11]
            elif (
                isinstance(index_typ, types.Integer)
                and (arr.name, orig_index_var.name) not in self._parallel_accesses
            ):
                # TODO: handle multi-dim cases like A[0,:]
                in_arr = full_node.value.value
                start_var, nodes = self._get_dist_start_var(
                    in_arr, equiv_set, avail_vars
                )
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

    def _run_parallel_access_getsetitem(
        self, arr, index_var, node, full_node, equiv_set, avail_vars
    ):
        """adjust index of getitem/setitem using parfor index on dist arrays
        """
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

            def _fix_ind_bounds(start, stop):
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

            for (arr, index, _) in array_accesses:
                if self._index_has_par_index(index, ind_varname):
                    self._1D_Var_parfor_starts[arr] = l_nest.start

        init_reduce_nodes, reduce_nodes = self._gen_parfor_reductions(parfor)
        parfor.init_block.body += init_reduce_nodes
        out = prepend + [parfor] + reduce_nodes
        return out

    def _index_has_par_index(self, index, par_index):
        """check if parfor index is used in 1st dimension of access index
        """
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
        _, reductions = get_parfor_reductions(parfor, parfor.params, self.calltypes)

        for reduce_varname, (_init_val, reduce_nodes) in reductions.items():
            reduce_op = guard(self._get_reduce_op, reduce_nodes)
            reduce_var = reduce_nodes[-1].target
            assert reduce_var.name == reduce_varname
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
        nodes = []
        if arr.name in self._1D_Var_parfor_starts:
            start_var = self._1D_Var_parfor_starts[arr.name]
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
        if arr.name in self._1D_Var_parfor_starts:
            return self._1D_Var_parfor_starts[arr.name], []

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

    def _dist_arr_needs_adjust(self, varname):
        return self._is_1D_arr(varname) or (
            self._is_1D_Var_arr(varname) and varname in self._1D_Var_parfor_starts
        )

    def _get_parallel_access_start_var(self, arr, equiv_set, index_var, avail_vars):
        """Same as _get_dist_start_var() but avoids generating reduction for
        getting global size since this is an error inside a parfor loop.
        """
        if arr.name in self._1D_Var_parfor_starts:
            return self._1D_Var_parfor_starts[arr.name], []

        # XXX we return parfors start assuming parfor and parallel accessed
        # array are equivalent in size and have equivalent distribution
        # TODO: is this always the case?
        if index_var.name in self._1D_parfor_starts:
            return self._1D_parfor_starts[index_var.name], []

        # use shape if parfor start not found (TODO shouldn't reach here?)
        shape = equiv_set.get_shape(arr)
        if isinstance(shape, (list, tuple)) and len(shape) > 0:
            size_var = shape[0]
            nodes = []
            start_var = self._get_1D_start(size_var, avail_vars, nodes)
            return start_var, nodes

        raise ValueError("invalid parallel access")

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
        """get start index of size_var in 1D_Block distribution
        """
        # reuse start var if available
        if (
            size_var.name in self._start_vars
            and self._start_vars[size_var.name].name in avail_vars
        ):
            return self._start_vars[size_var.name]
        nodes += compile_func_single_block(
            lambda n, rank, n_pes: min(n, rank * math.ceil(n / n_pes)),
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
        """get chunk size for size_var in 1D_Block distribution
        """

        def impl(n, rank, n_pes):
            chunk = math.ceil(n / n_pes)
            return min(n, (rank + 1) * chunk) - min(n, rank * chunk)

        nodes += compile_func_single_block(
            impl, (size_var, self.rank_var, self.n_pes_var), None, self
        )
        count_var = nodes[-1].target
        # rename for readability
        count_var.name = mk_unique_var("count_var")
        self.typemap[count_var.name] = types.int64
        return count_var

    def _get_1D_end(self, size_var, nodes):
        """get end index of size_var in 1D_Block distribution
        """
        nodes += compile_func_single_block(
            lambda n, rank, n_pes: min(n, (rank + 1) * math.ceil(n / n_pes)),
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
            self.typemap[ind_var.name], types.misc.SliceType
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
        """Finds file open call (h5py.File) for file_varname and sets the parallel flag.
        """
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
                    self.calltypes[var_def] = self.typemap[var_def.func.name].get_call_type(self.typingctx, arg_typs, {})
                    kws = dict(var_def.kws)
                    kws['_is_parallel'] = self._set1_var
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

    def _get_reduce_op(self, reduce_nodes):
        require(len(reduce_nodes) == 2)
        require(isinstance(reduce_nodes[0], ir.Assign))
        require(isinstance(reduce_nodes[1], ir.Assign))
        require(isinstance(reduce_nodes[0].value, ir.Expr))
        require(isinstance(reduce_nodes[1].value, ir.Var))
        rhs = reduce_nodes[0].value

        if rhs.op == "inplace_binop":
            if rhs.fn in ("+=", operator.iadd):
                return Reduce_Type.Sum
            if rhs.fn in ("|=", operator.ior):
                return Reduce_Type.Or
            if rhs.fn in ("*=", operator.imul):
                return Reduce_Type.Prod

        if rhs.op == "call":
            func = find_callname(self.func_ir, rhs, self.typemap)
            if func == ("min", "builtins"):
                if isinstance(
                    self.typemap[rhs.args[0].name], numba.typing.builtins.IndexValueType
                ):
                    return Reduce_Type.Argmin
                return Reduce_Type.Min
            if func == ("max", "builtins"):
                if isinstance(
                    self.typemap[rhs.args[0].name], numba.typing.builtins.IndexValueType
                ):
                    return Reduce_Type.Argmax
                return Reduce_Type.Max

        raise GuardException  # pragma: no cover

    def _gen_init_reduce(self, reduce_var, reduce_op):
        """generate code to initialize reduction variables on non-root
        processors.
        """
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
                init_val = "numba.targets.builtins.get_type_max_value(np.ones(1,dtype=np.{}).dtype)".format(
                    el_typ
                )
        if reduce_op == Reduce_Type.Max:
            if el_typ == types.bool_:
                init_val = "False"
            else:
                init_val = "numba.targets.builtins.get_type_min_value(np.ones(1,dtype=np.{}).dtype)".format(
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

    def _update_avail_vars(self, avail_vars, nodes):
        for stmt in nodes:
            if type(stmt) in numba.analysis.ir_extension_usedefs:
                def_func = numba.analysis.ir_extension_usedefs[type(stmt)]
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
                ("isna", "bodo.hiframes.api"),
                ("setitem_arr_nan", "bodo.ir.join"),
                ("str_arr_item_to_numeric", "bodo.libs.str_arr_ext"),
                ("setitem_str_arr_ptr", "bodo.libs.str_arr_ext"),
                ("get_str_arr_item_length", "bodo.libs.str_arr_ext"),
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

    def _get_tuple_varlist(self, tup_var, out):
        """ get the list of variables that hold values in the tuple variable.
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


def find_available_vars(blocks, cfg, init_avail=None):
    """
    Find available variables to use at ENTRY point of basic blocks. Similar
    to available expressions algorithm but does not kill values on defs.
    In_l = intersect(Out_i for i in pred)
    Out_l = In_l | def(l)
    `init_avail` is used to initialize first block of parfors with available vars
    before the parfor. The label of first block is assumed to be 0.
    """
    # TODO: unittest
    in_avail_vars = defaultdict(set)
    usedefs = numba.analysis.compute_use_defs(blocks)
    var_def_map = usedefs.defmap
    out_avail_vars = var_def_map.copy()

    if init_avail:
        assert 0 in blocks
        for label in var_def_map:
            in_avail_vars[label] = init_avail
            out_avail_vars[label] |= init_avail

    old_point = None
    new_point = tuple(len(v) for v in in_avail_vars.values())

    while old_point != new_point:
        for label in var_def_map:
            if label == 0:
                continue
            # intersect out of predecessors
            preds = list(l for l, _ in cfg.predecessors(label))
            in_avail_vars[label] = out_avail_vars[preds[0]] if preds else {}
            for inc_blk in preds:
                in_avail_vars[label] &= out_avail_vars[inc_blk]
            # include defs
            out_avail_vars[label] = in_avail_vars[label] | var_def_map[label]

        old_point = new_point
        new_point = tuple(len(v) for v in in_avail_vars.values())

    return in_avail_vars
